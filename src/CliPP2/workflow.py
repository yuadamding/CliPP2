"""Three-stage orchestration with frozen native fits and explicit lineage."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from collections.abc import Mapping

import numpy as np

from .assignment import (AssignmentBundle, AssignmentReference, _freeze, _json_value,
                         assign_mutations, content_hash, saturated_log_mass)
from .config import WorkflowConfig
from .io.data import MultiplicityBounds, readonly_array
from .io.tumor_txt import load_original_catalog
from .kernel.backend import NativeBackend, NativeExecutionError
from .kernel.observations import prepare_prediction


def _biological_hash(catalog, mutation_ids):
    positions = {mutation: i for i, mutation in enumerate(catalog.mutation_ids)}
    rows = [positions[mutation] for mutation in mutation_ids]
    return content_hash(dict(mutation_ids=mutation_ids, region_ids=catalog.region_ids,
        **{key: getattr(catalog, key)[rows] for key in ("alt_counts", "total_counts", "count_observed",
           "eligible", "major_cn", "minor_cn", "normal_cn", "mean_total_cn", "purity",
           "cn_state_count", "segment_ids", "mapping_genuine")}))


@dataclass(frozen=True, slots=True)
class AcceptedFit:
    """Full-precision native estimates, actual support, and acceptance receipt.

    Labels are public source labels. Merging never changes this record.
    Import validation evaluates the existing estimates and never refits them.
    """

    source_id: str
    mutation_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    labels: np.ndarray
    centers: np.ndarray
    weights: np.ndarray
    multiplicities: np.ndarray
    support: MultiplicityBounds
    biological_hash: str
    model_identity: str
    score: float
    conditional_log_likelihood: float
    mixture_log_likelihood: float
    complexity_penalty: float
    acceptance_evidence: object
    normalization: str = "uniform_interval_width_v1"

    def __post_init__(self):
        object.__setattr__(self, "mutation_ids", tuple(self.mutation_ids))
        object.__setattr__(self, "region_ids", tuple(self.region_ids))
        n, r = len(self.mutation_ids), len(self.region_ids)
        labels, centers = np.asarray(self.labels), np.asarray(self.centers, float)
        if (not n or not r or len(set(self.mutation_ids)) != n or len(set(self.region_ids)) != r
                or labels.shape != (n,) or labels.dtype.kind not in "iu"
                or centers.ndim != 2 or centers.shape[1] != r or not len(centers)
                or not np.array_equal(np.unique(labels), np.arange(len(centers)))
                or not np.isfinite(centers).all() or np.any(centers < 0) or np.any(centers > 1)):
            raise ValueError("Accepted fit requires occupied source labels and finite native centers")
        for name in ("labels", "centers", "weights", "multiplicities"):
            object.__setattr__(self, name, readonly_array(getattr(self, name), dtype=np.int64 if name == "labels" else float))
        if self.multiplicities.shape != (n, r) or self.weights.shape != (len(centers),):
            raise ValueError("Accepted native parameters do not align with populations")
        if not np.isfinite(self.weights).all() or np.any(self.weights <= 0) or not np.isclose(self.weights.sum(), 1, atol=1e-12, rtol=0):
            raise ValueError("Accepted fit requires positive normalized occupied-component weights")
        if tuple(self.support.mutation_ids) != self.mutation_ids or tuple(self.support.region_ids) != self.region_ids:
            raise ValueError("Accepted support must be keyed to its exact fitted population")
        if self.normalization != "uniform_interval_width_v1":
            raise ValueError("Historical fit uses an incompatible support normalization")
        if not np.isfinite([self.score, self.conditional_log_likelihood, self.mixture_log_likelihood, self.complexity_penalty]).all():
            raise ValueError("Accepted numerical evidence must be finite")
        object.__setattr__(self, "acceptance_evidence", _freeze(self.acceptance_evidence))

    @property
    def phi(self):
        return self.centers[self.labels]

    @property
    def state_fingerprint(self):
        """Bind acceptance evidence to these exact public-ordered native values."""
        names = ("source_id", "mutation_ids", "region_ids", "labels", "centers", "weights",
                 "multiplicities", "biological_hash", "model_identity", "score",
                 "conditional_log_likelihood", "mixture_log_likelihood", "complexity_penalty", "normalization")
        return content_hash(dict(schema="clipp2.accepted-state.v1", support_hash=self.support.content_hash,
                                 **{name: getattr(self, name) for name in names}))

    def to_dict(self):
        fields = ("source_id", "mutation_ids", "region_ids", "labels", "centers", "weights",
            "multiplicities", "biological_hash", "model_identity", "score", "conditional_log_likelihood",
            "mixture_log_likelihood", "complexity_penalty", "acceptance_evidence", "normalization")
        return _json_value(dict(schema="clipp2.accepted-fit.v1", **{k: getattr(self, k) for k in fields},
                               support=self.support.to_dict()))

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        if values.pop("schema", None) != "clipp2.accepted-fit.v1":
            raise ValueError("A compatible full accepted-fit receipt is required; three TSVs are insufficient")
        values["support"] = MultiplicityBounds.from_dict(values["support"])
        return cls(**values)

    def validate(self, catalog, *, expected_ids=None, expected_support=None, device="cpu", cpu_threads=None):
        """Validate historical biological/numerical compatibility without optimization."""
        from .io.reporting import _verify_fitted_numerics
        from .kernel.model import FitResult, RegionalModel
        from .kernel.topology import Membership
        if self.biological_hash != _biological_hash(catalog, self.mutation_ids):
            raise ValueError("Historical denominators, masks, mapping, or observations differ")
        if tuple(self.region_ids) != tuple(catalog.region_ids):
            raise ValueError("Historical fit has different requested regions")
        if expected_ids is not None and set(self.mutation_ids) != set(expected_ids):
            raise ValueError("Historical fit population differs from the requested stage")
        if expected_support is not None:
            numerical = expected_support.for_data(catalog.fit_data(self.mutation_ids))
            if not np.array_equal(numerical[0], self.support.lower) or not np.array_equal(numerical[1], self.support.upper):
                raise ValueError("Historical fit has different actual multiplicity support")
        evidence = self.acceptance_evidence
        audit = evidence.get("membership_audit", {}) if isinstance(evidence, Mapping) else {}
        if (not isinstance(evidence, Mapping) or not isinstance(audit, Mapping)
                or evidence.get("independently_verified") is not True
                or audit.get("selected_count_certified") is not True or audit.get("complete") is not True):
            raise ValueError("Historical fit lacks accepted independent and terminal-audit evidence")
        if evidence.get("accepted_state_hash") != self.state_fingerprint:
            raise ValueError("Historical acceptance evidence is not bound to this exact fitted state")
        model = RegionalModel(catalog.fit_data(self.mutation_ids), device=device,
                              cpu_threads=cpu_threads, multiplicity_support=self.support)
        try:
            if model.mutation_ids != self.mutation_ids or model.identity != self.model_identity:
                raise ValueError("Historical model identity or canonical population order differs")
            fit = FitResult(Membership(explicit_labels=self.labels), self.centers, self.weights,
                self.conditional_log_likelihood, self.mixture_log_likelihood, self.complexity_penalty,
                self.score, True)
            _verify_fitted_numerics(model, fit)
            calls = model.multiplicity(self.phi)
            if not np.array_equal(calls, self.multiplicities, equal_nan=True):
                raise ValueError("Historical accepted multiplicities disagree with their native support/CCFs")
        finally:
            model.backend.close()
        return self


def _fit_accepted(catalog, ids, bounds, outdir, stage, config, progress_callback):
    from .io.reporting import publish
    from .kernel.model import RegionalModel
    from .tree.selection import fit_regions
    data = catalog.fit_data(ids)
    model = RegionalModel(data, device=config.device, cpu_threads=config.cpu_threads,
                          multiplicity_support=bounds)
    try:
        callback = None if progress_callback is None else lambda record: progress_callback(dict(record, stage=stage))
        result = fit_regions(model, config.max_clusters, audit_scope=config.audit_scope,
                             progress_callback=callback, retain_evidence=False)
        published = publish(model, result, outdir, catalog.tumor_id)
        fit = result.selected.fit
        order = np.argsort(-np.linalg.norm(fit.centers, axis=1), kind="stable")
        mapping = np.empty(len(order), int)
        mapping[order] = np.arange(len(order))
        support = MultiplicityBounds(model.mutation_ids, model.region_ids,
            model.multiplicity_min, model.multiplicity_max, bounds.policy_id)
        accepted = AcceptedFit(stage, model.mutation_ids, model.region_ids, mapping[fit.labels],
            fit.centers[order], fit.weights[order], model.multiplicity(fit.centers[fit.labels]), support,
            _biological_hash(catalog, model.mutation_ids), model.identity, fit.score,
            fit.conditional_log_likelihood, fit.mixture_log_likelihood, fit.complexity_penalty,
            published["verification"])
        return replace(accepted, acceptance_evidence=dict(accepted.acceptance_evidence,
                                                         accepted_state_hash=accepted.state_fingerprint))
    finally:
        model.backend.close()


def _diagnostics(catalog, ids, phi, bounds, config, calls=None):
    observation = prepare_prediction(catalog, ids, bounds, device=config.fit.device,
                                     cpu_threads=config.fit.cpu_threads)
    with NativeBackend(observation, cpu_threads=config.fit.cpu_threads) as backend:
        diagnostic = backend.component_diagnostics(phi, accepted_calls=calls)
    if np.any((diagnostic["status"] == 2) | (diagnostic["status"] == 3)):
        raise NativeExecutionError("Workflow component diagnostics failed")
    sat = saturated_log_mass(observation.alt, observation.depth)
    diagnostic["profile_deviance"] = np.where(observation.observed,
        np.maximum(0., 2 * (sat - diagnostic["max_component_log_mass"])), np.nan)
    return diagnostic, observation


def qualify_contributors(catalog, fit, config, *, mutation_ids=None):
    """Audit the same accepted call at the same CCF under both supports."""
    wanted = set(fit.mutation_ids if mutation_ids is None else mutation_ids)
    native_phi = fit.phi
    native, observation = _diagnostics(catalog, fit.mutation_ids, native_phi, fit.support,
                                       config, fit.multiplicities)
    original, _ = _diagnostics(catalog, fit.mutation_ids, native_phi,
                               catalog.default_bounds().subset(fit.mutation_ids), config, fit.multiplicities)
    positions = {mutation: i for i, mutation in enumerate(catalog.mutation_ids)}
    rows, records = [positions[mutation] for mutation in fit.mutation_ids], []
    for i, mutation in enumerate(fit.mutation_ids):
        if mutation not in wanted:
            continue
        for j, region in enumerate(catalog.region_ids):
            call, phi = fit.multiplicities[i, j], native_phi[i, j]
            native_lp, original_lp = native["accepted_log_probability"][i, j], original["accepted_log_probability"][i, j]
            valid = np.isfinite(call) and fit.support.lower[i, j] <= call <= fit.support.upper[i, j]
            if valid and np.isfinite(native_lp) and original_lp > native_lp + 1e-10:
                raise ArithmeticError("Original-support accepted-call confidence exceeds native confidence")
            if not catalog.mapping_genuine[rows[i], j]:
                reason = "nongenuine_mapping"
            elif not catalog.eligible[rows[i], j] or catalog.cn_state_count[rows[i], j] != 1:
                reason = "originally_ineligible_or_mixed"
            elif not observation.observed[i, j]:
                reason = "no_information"
            elif not phi > 0:
                reason = "zero_ccf"
            elif not valid:
                reason = "invalid_native_call"
            elif native["map_multiplicity"][i, j] != call or native["top_two_gap"][i, j] <= config.assignment.tie_atol:
                reason = "native_not_unique_map"
            elif np.exp(native_lp) < config.contributor_probability_min:
                reason = "native_low_confidence"
            elif not np.isfinite(native["profile_deviance"][i, j]) or native["profile_deviance"][i, j] > config.contributor_deviance_max:
                reason = "native_residual"
            elif original["map_multiplicity"][i, j] != call or original["top_two_gap"][i, j] <= config.assignment.tie_atol:
                reason = "original_not_unique_map"
            elif not np.isfinite(original_lp) or np.exp(original_lp) < config.contributor_probability_min:
                reason = "original_low_confidence"
            else:
                reason = "usable"
            records.append(dict(mutation_id=mutation, region_id=region,
                segment_id=str(catalog.segment_ids[rows[i], j]), source_stage=fit.source_id,
                source_label=int(fit.labels[i]), native_phi=float(phi), native_call=float(call),
                native_lower=int(fit.support.lower[i, j]), native_upper=int(fit.support.upper[i, j]),
                native_call_probability=float(np.exp(native_lp)), original_call_probability=float(np.exp(original_lp)),
                native_gap=float(native["top_two_gap"][i, j]), original_gap=float(original["top_two_gap"][i, j]),
                profile_deviance=float(native["profile_deviance"][i, j]), usable=reason == "usable", reason=reason))
    return records


def classify_groups(catalog, fit, config):
    native_phi = fit.phi
    native, observation = _diagnostics(catalog, fit.mutation_ids, native_phi, fit.support, config)
    clonal, _ = _diagnostics(catalog, fit.mutation_ids, np.ones_like(native_phi), fit.support, config)
    groups = []
    for label, center in enumerate(fit.centers):
        members = fit.labels == label
        coverage = observation.observed[members].sum(axis=0)
        depths = np.where(observation.observed[members], observation.depth[members], 0).sum(axis=0)
        residuals = []
        for diagnostic in (native, clonal):
            means = []
            for region in range(observation.r):
                values = diagnostic["profile_deviance"][members, region]
                means.append(float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan"))
            residuals.append(np.asarray(means))
        distance = float(np.linalg.norm(center - 1))
        coverage_ok = bool(np.all(coverage >= config.min_informative_per_region) and np.all(depths >= config.min_depth_per_region))
        native_ok = bool(np.isfinite(residuals[0]).all() and np.all(residuals[0] <= config.group_mean_deviance_max))
        all_one_ok = bool(np.isfinite(residuals[1]).all() and np.all(residuals[1] <= config.group_mean_deviance_max))
        if not coverage_ok or not native_ok:
            classification = "unresolved"
        elif distance >= config.clonal_distance_max:
            classification = "nonclonal"
        elif all_one_ok:
            classification = "clonal_merge_eligible"
        else:
            classification = "unresolved"
        groups.append(dict(source_stage=fit.source_id, source_label=label, size=int(members.sum()),
            distance_to_one=distance, informative_counts=coverage.tolist(), informative_depths=depths.tolist(),
            minimum_regional_depth=float(depths.min()), native_mean_deviance=residuals[0].tolist(),
            all_one_mean_deviance=residuals[1].tolist(), coverage_pass=coverage_ok,
            native_fit_pass=native_ok, all_one_fit_pass=all_one_ok, classification=classification))
    return groups


def _segment_bounds(catalog, audits, *, lower, policy):
    original = catalog.default_bounds()
    lo, hi = np.array(original.lower), np.array(original.upper)
    contributors = {}
    for record in audits:
        if record["usable"]:
            contributors.setdefault((record["region_id"], record["segment_id"]), []).append(record)
    rows = []
    for i, mutation in enumerate(catalog.mutation_ids):
        for j, region in enumerate(catalog.region_ids):
            key = region, str(catalog.segment_ids[i, j])
            candidates = contributors.get(key, []) if catalog.mapping_genuine[i, j] and catalog.eligible[i, j] else []
            endpoint = (max if lower else min)(int(v["native_call"]) for v in candidates) if candidates else (1 if lower else int(hi[i, j]))
            if lower:
                lo[i, j] = endpoint
            else:
                hi[i, j] = endpoint
            if lo[i, j] > hi[i, j]:
                raise ValueError("Segment-derived support exceeds the original biological support")
            rows.append(dict(mutation_id=mutation, region_id=region, segment_id=key[1],
                multiplicity_lower=int(lo[i, j]), multiplicity_upper=int(hi[i, j]),
                contributor_count=len(candidates), fallback=not candidates,
                contributor_ids=[v["mutation_id"] for v in candidates], policy=policy))
    return MultiplicityBounds(catalog.mutation_ids, catalog.region_ids, lo, hi, policy), rows


def run_workflow(input_path, outdir, workflow_config, *, progress_callback=None, accepted_stage_one=None):
    """Run all applicable stages; failed or blocked attempts never imply success."""
    from .io.assignment import (load_accepted_fit, publish_workflow_receipt, reserve_output,
                                save_accepted_fit, save_assignment_bundle, save_original_catalog,
                                write_supporting_table)
    from .io.region_files import input_fingerprint
    if type(workflow_config) is not WorkflowConfig:
        raise TypeError("workflow_config must be a WorkflowConfig")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    config, output = workflow_config, reserve_output(outdir)
    try:
        catalog = load_original_catalog(input_path, max_major_cn=config.fit.max_major_cn)
    except Exception as error:
        publish_workflow_receipt(output, dict(status="failed", failed_stage="input",
            error_type=type(error).__name__, error=str(error), config=asdict(config)))
        raise
    save_original_catalog(catalog, output / "original_catalog.json")
    if np.any((catalog.cn_state_count > 1) & catalog.mapping_genuine):
        publish_workflow_receipt(output, dict(status="failed", reason="mixed_cn_outside_workflow_scope",
                                              config=asdict(config), input_identity=catalog.input_identity))
        raise ValueError("The three-stage workflow currently requires genuine single-state CN")
    informative = catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)
    population = tuple(m for m, keep in zip(catalog.mutation_ids, informative.any(axis=1)) if keep)
    full = catalog.default_bounds()
    receipt = dict(schema="clipp2.workflow.v1", status="running", config=asdict(config),
        input_identity=catalog.input_identity, original_mutation_count=len(catalog.mutation_ids),
        stage_one="pending", stage_two="not_applicable", stage_three="not_applicable",
        counts=dict(N1=len(population), N0=0, N2=0, Q2=0, merged_groups=0, combined_mutations=0, K=0))
    current_stage = "stage_one"
    try:
        if accepted_stage_one is None:
            stage_one = _fit_accepted(catalog, population, full, output / "stage_one", "stage_one", config.fit, progress_callback)
        else:
            stage_one = load_accepted_fit(accepted_stage_one) if not isinstance(accepted_stage_one, AcceptedFit) else accepted_stage_one
            stage_one.validate(catalog, expected_ids=population, expected_support=full,
                               device=config.fit.device, cpu_threads=config.fit.cpu_threads)
        receipt["stage_one"] = "accepted"
        save_accepted_fit(stage_one, output / "stage_one_accepted.json")
        group_one = classify_groups(catalog, stage_one, config)
        write_supporting_table(output / "stage_one_groups.tsv", group_one)
        possible = [g for g in group_one if g["classification"] == "clonal_merge_eligible"]
        if not possible:
            receipt.update(status="no_eligible_clonal_reference", reference=None)
            _all_original_comparison(output, catalog, stage_one, None, {}, None)
            if input_fingerprint(input_path) != catalog.input_identity:
                raise RuntimeError("Original inputs changed during workflow execution")
            publish_workflow_receipt(output, receipt)
            return receipt
        selected = min(possible, key=lambda g: (g["distance_to_one"], -g["minimum_regional_depth"], g["source_label"]))
        g0 = tuple(m for m, label in zip(stage_one.mutation_ids, stage_one.labels) if label == selected["source_label"])
        receipt["reference"] = dict(source_id=stage_one.source_id, source_label=selected["source_label"], mutation_ids=g0)
        receipt["counts"]["N0"] = len(g0)
        reference_audit = qualify_contributors(catalog, stage_one, config, mutation_ids=g0)
        write_supporting_table(output / "reference_contributors.tsv", reference_audit)
        caps, cap_rows = _segment_bounds(catalog, reference_audit, lower=False, policy="stage_two_caps")
        write_supporting_table(output / "stage_two_bounds.tsv", cap_rows)
        reference_members = set(g0)
        remainder = tuple(m for m in population if m not in reference_members)
        if reference_members & set(remainder):
            raise AssertionError("Reference membership overlaps the stage-two remainder")
        current_stage = "stage_two"
        stage_two, group_two = None, []
        if not remainder:
            receipt["stage_two"] = "skipped_no_eligible_informative_remainder"
        else:
            positions = {mutation: i for i, mutation in enumerate(catalog.mutation_ids)}
            indices = [positions[m] for m in remainder]
            unsupported = [region for region, ok in zip(catalog.region_ids, informative[indices].any(axis=0)) if not ok]
            if unsupported:
                receipt.update(status="blocked", stage_two="unsupported_requested_regions", unsupported_regions=unsupported)
                receipt["counts"]["N2"] = len(remainder)
                _all_original_comparison(output, catalog, stage_one, None, {}, None)
                if input_fingerprint(input_path) != catalog.input_identity:
                    raise RuntimeError("Original inputs changed during workflow execution")
                publish_workflow_receipt(output, receipt)
                return receipt
            stage_two_config = replace(config.fit, max_clusters=config.stage_two_max_clusters)
            stage_two = _fit_accepted(catalog, remainder, caps.subset(remainder), output / "stage_two", "stage_two", stage_two_config, progress_callback)
            save_accepted_fit(stage_two, output / "stage_two_accepted.json")
            group_two = classify_groups(catalog, stage_two, config)
            receipt["stage_two"] = "accepted"
        write_supporting_table(output / "stage_two_groups.tsv", group_two)
        merged = {g["source_label"] for g in group_two if g["classification"] == "clonal_merge_eligible"}
        source_map = [dict(source_stage="stage_one", source_label=selected["source_label"], final_label=0,
                           classification="frozen_clonal_reference")]
        centers, next_label = [np.ones(len(catalog.region_ids))], 1
        for group in group_two:
            label = group["source_label"]
            final = 0 if label in merged else next_label
            if label not in merged:
                centers.append(stage_two.centers[label])
                next_label += 1
            source_map.append(dict(source_stage="stage_two", source_label=label, final_label=final,
                                   classification=group["classification"]))
        label_map = {(v["source_stage"], v["source_label"]): v["final_label"] for v in source_map}
        merged_ids = () if stage_two is None else tuple(m for m, label in zip(stage_two.mutation_ids, stage_two.labels) if label in merged)
        merged_members = set(merged_ids)
        targets = tuple(sorted(g0 + merged_ids))
        donors = () if stage_two is None else tuple(m for m in stage_two.mutation_ids if m not in merged_members)
        combined = reference_members | set(remainder)
        if set(targets) & set(donors) or set(targets) | set(donors) != combined:
            raise AssertionError("Frozen targets and donors do not reconcile the combined population")
        q2 = 0 if stage_two is None else len(stage_two.centers)
        if len(centers) != 1 + q2 - len(merged):
            raise AssertionError("Final reference count does not reconcile merged source groups")
        donor_audit = [] if stage_two is None else qualify_contributors(catalog, stage_two, config, mutation_ids=donors)
        floors, floor_rows = _segment_bounds(catalog, donor_audit, lower=True, policy="stage_three_restricted")
        if np.any(catalog.eligible & (floors.lower > caps.upper)):
            raise ArithmeticError("Donor floor exceeds the frozen reference-derived cap")
        write_supporting_table(output / "donor_contributors.tsv", donor_audit)
        write_supporting_table(output / "stage_three_bounds.tsv", floor_rows)
        write_supporting_table(output / "source_to_final.tsv", source_map)
        reference = AssignmentReference(tuple(range(len(centers))), catalog.region_ids,
                                        np.asarray(centers), tuple(source_map))
        write_supporting_table(output / "reference_vectors.tsv", [dict(final_label=label,
            prior=1 / len(centers), **{"phi_" + region: float(center[j]) for j, region in enumerate(catalog.region_ids)})
            for label, center in enumerate(centers)])
        baselines = {}
        for fit, members in ((stage_one, reference_members), (stage_two, merged_members)):
            if fit is not None:
                for i, mutation in enumerate(fit.mutation_ids):
                    if mutation in members:
                        baselines[mutation] = (fit, i)
        baseline_rows = [baselines[mutation] for mutation in targets]
        bundle = AssignmentBundle(catalog, reference, targets, donors, floors.subset(targets), config.assignment,
            np.asarray([fit.centers[fit.labels[i]] for fit, i in baseline_rows]),
            np.asarray([fit.multiplicities[i] for fit, i in baseline_rows]),
            np.asarray([fit.support.lower[i] for fit, i in baseline_rows]),
            np.asarray([fit.support.upper[i] for fit, i in baseline_rows]),
            tuple(fit.source_id for fit, _ in baseline_rows),
            tuple(int(fit.labels[i]) for fit, i in baseline_rows),
            config.fit.device, config.fit.cpu_threads,
            dict(caps_hash=caps.content_hash, floors_hash=floors.content_hash,
                 reference_contributors=reference_audit, donor_contributors=donor_audit,
                 native_stage_two_centers_preserved=True))
        save_assignment_bundle(bundle, output / "assignment_bundle.json")
        current_stage = "stage_three"
        assignment = assign_mutations(bundle, outdir=output / "assignment", progress_callback=progress_callback)
        _all_original_comparison(output, catalog, stage_one, stage_two, label_map, assignment)
        if input_fingerprint(input_path) != catalog.input_identity:
            raise RuntimeError("Original inputs changed during workflow execution")
        receipt.update(status="complete", stage_three="assigned", bundle_hash=bundle.content_hash,
            counts=dict(N1=len(stage_one.mutation_ids), N0=len(g0), N2=len(remainder), Q2=q2,
                        merged_groups=len(merged), combined_mutations=len(combined), K=len(centers),
                        targets=len(targets), donors=len(donors)),
            membership_merge_is_refit=False)
        publish_workflow_receipt(output, receipt)
        return receipt
    except Exception as error:
        receipt.update(status="failed", failed_stage=current_stage,
                       error_type=type(error).__name__, error=str(error))
        receipt[current_stage] = "failed"
        publish_workflow_receipt(output, receipt)
        raise


def _all_original_comparison(output, catalog, first, second, mapping, assignment):
    from .io.assignment import write_supporting_table
    stages, positions = [], []
    for fit in (first, second):
        stages.append({} if fit is None else {m: int(label) for m, label in zip(fit.mutation_ids, fit.labels)})
        positions.append({} if fit is None else {m: i for i, m in enumerate(fit.mutation_ids)})
    assignments = {} if assignment is None else {m: (a, b, c) for m, a, b, c in zip(
        assignment.target_ids, assignment.restricted, assignment.original, assignment.comparison)}
    rows = []
    for i, mutation in enumerate(catalog.mutation_ids):
        one, two = stages[0].get(mutation), stages[1].get(mutation)
        final = mapping.get(("stage_two", two)) if two is not None else mapping.get(("stage_one", one))
        row = dict(mutation_id=mutation, original_eligible=bool(catalog.eligible[i].any()),
            original_status="filtered" if not catalog.eligible[i].any() else "no_information"
                if not (catalog.eligible[i] & catalog.count_observed[i] & (catalog.total_counts[i] > 0)).any() else "eligible_informative",
            stage_one_label=one, stage_two_label=two, final_reference_label=final,
            role="target" if mutation in assignments else "donor" if two is not None else "not_assigned")
        for name, fit, lookup in (("stage_one", first, positions[0]), ("stage_two", second, positions[1])):
            index = lookup.get(mutation)
            row[name + "_native_phi"] = None if index is None else fit.centers[fit.labels[index]].tolist()
            row[name + "_native_calls"] = None if index is None else fit.multiplicities[index].tolist()
        if mutation in assignments:
            a, b, c = assignments[mutation]
            row.update(restricted_status=a["status"], original_status=b["status"],
                restricted_accepted_label=a["accepted_label"], original_accepted_label=b["accepted_label"],
                support_stable_label=c["support_stable_label"])
        rows.append(row)
    write_supporting_table(output / "all_original_mutations.tsv", rows)
