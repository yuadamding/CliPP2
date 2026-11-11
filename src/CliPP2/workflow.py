"""Protocol 2.3 orchestration with frozen native fits and biological lineage.

Biological/numerical identities and native build IDs are retained; file-byte
and Python-source checksums are intentionally omitted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from collections.abc import Mapping

import numpy as np

from .assignment import (AssignmentBundle, AssignmentReference, EVIDENCE_STATUSES,
                         assign_mutations, conditional_multiplicity)
from .config import (PROTOCOL_VERSION, COMPATIBLE_NATIVE_FIT_PROTOCOLS, WorkflowConfig,
                     STAGE_THREE_REFERENCE_POLICY)
from ._version import __version__
from .io.data import (ImmutableArrayRecord, MultiplicityBounds, content_hash, freeze_record as _freeze,
                      json_value as _json_value, readonly_array)
from .io.tumor_txt import load_original_catalog
from .kernel.backend import NativeBackend, NativeExecutionError
from .kernel.observations import prepare_prediction

CONTRIBUTOR_COLUMNS = ("mutation_id", "region_id", "segment_id", "source_stage", "source_label",
    "native_phi", "native_source_call", "cap_source_map_call", "cap_source_map_upper", "native_lower", "native_upper",
    "map_probability", "map_unique", "map_tied_values", "map_score_gap", "profile_deviance", "multiplicity_read_informative",
    "contributes_to_cap", "availability_reason", "diagnostic_status", "diagnostic_error")
GROUP_COLUMNS = ("source_stage", "source_label", "size", "native_center", "distance_to_one", "informative_counts",
    "informative_depths", "minimum_regional_depth", "diagnostic_status", "diagnostic_error") + tuple(
        prefix + "_" + field for prefix in ("native", "all_one", "original_native", "original_all_one")
        for field in ("informative_count", "mean_profile_deviance", "maximum_profile_deviance"))
SEGMENT_POLICY_COLUMNS = ("region_id", "segment_id", "multiplicity_endpoint", "contributor_count",
    "fallback", "fallback_policy", "applies_to", "policy")
ORIGINAL_COMPARISON_COLUMNS = ("mutation_id", "original_eligible", "original_input_status", "exclusion_reasons",
    "stage_one_label", "stage_two_label", "baseline_reference_label", "final_assignment_label", "role", "target_reason", "membership_origin",
    "stage_one_native_phi", "stage_one_native_calls", "stage_two_native_phi", "stage_two_native_calls",
    "assignment_status", "evidence_status", "map_cluster_label", "map_label", "map_set", "posterior_defined",
    "reference_labels", "probabilities", "reassessment_status", "baseline_changed",
    "historical_cap_contributor_reassigned")


def _biological_hash(catalog, mutation_ids):
    positions = {mutation: i for i, mutation in enumerate(catalog.mutation_ids)}
    rows = [positions[mutation] for mutation in mutation_ids]
    payload = dict(schema="clipp2.accepted-biological-input.v3", tumor_id=catalog.tumor_id, biological_identity=catalog.biological_identity,
        mutation_ids=mutation_ids, region_ids=catalog.region_ids,
        **{key: getattr(catalog, key)[rows] for key in ("alt_counts", "total_counts", "count_observed",
           "eligible", "major_cn", "minor_cn", "normal_cn", "total_cn", "purity",
           "segment_ids", "mapping_genuine", "normal_cn_supplied")})
    if catalog.input_excluded[rows].any():
        payload["input_excluded"] = catalog.input_excluded[rows]
    return content_hash(payload)


@dataclass(frozen=True, slots=True)
class AcceptedFit(ImmutableArrayRecord):
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
    producer_receipt: object = None
    compatibility_receipts: tuple = ()
    biological_identity: object = None
    protocol_version: str = PROTOCOL_VERSION
    tumor_id: str = field(kw_only=True)

    def __post_init__(self):
        from .io.tumor_txt import _safe_tumor_id
        object.__setattr__(self, "tumor_id", _safe_tumor_id(self.tumor_id))
        if self.protocol_version not in COMPATIBLE_NATIVE_FIT_PROTOCOLS:
            raise ValueError("Accepted fit requires explicitly supported native-fit protocol evidence")
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
        object.__setattr__(self, "producer_receipt", _freeze(self.producer_receipt))
        object.__setattr__(self, "compatibility_receipts", _freeze(self.compatibility_receipts))
        object.__setattr__(self, "biological_identity", _freeze(self.biological_identity))

    @property
    def phi(self):
        return self.centers[self.labels]

    @property
    def state_fingerprint(self):
        """Bind acceptance evidence to these exact public-ordered native values."""
        names = ("tumor_id", "source_id", "mutation_ids", "region_ids", "labels", "centers", "weights",
                 "multiplicities", "biological_hash", "model_identity", "score",
                 "conditional_log_likelihood", "mixture_log_likelihood", "complexity_penalty", "normalization",
                 "producer_receipt", "biological_identity", "protocol_version")
        return content_hash(dict(schema="clipp2.accepted-state.v3", support_hash=self.support.content_hash,
                                 **{name: getattr(self, name) for name in names}))

    def to_dict(self):
        fields = ("tumor_id", "source_id", "mutation_ids", "region_ids", "labels", "centers", "weights",
            "multiplicities", "biological_hash", "model_identity", "score", "conditional_log_likelihood",
            "mixture_log_likelihood", "complexity_penalty", "acceptance_evidence", "normalization",
            "producer_receipt", "compatibility_receipts", "biological_identity", "protocol_version")
        return _json_value(dict(schema="clipp2.accepted-fit.v3", **{k: getattr(self, k) for k in fields},
                               support=self.support.to_dict()))

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        if (values.pop("schema", None) != "clipp2.accepted-fit.v3"
                or values.get("protocol_version") not in COMPATIBLE_NATIVE_FIT_PROTOCOLS or not values.get("tumor_id")):
            raise ValueError("Replay requires accepted-fit.v3 from a compatible native-fit protocol with explicit originating tumor_id; rebuild missing-origin legacy evidence from its original input")
        values["support"] = MultiplicityBounds.from_dict(values["support"])
        return cls(**values)

    def validate_input_identity(self, catalog, original_config=None):
        """Reject changed input content before any new search or numerical reuse."""
        from .provenance import validate_fit_execution_receipt, validate_run_manifest
        producer = self.producer_receipt
        if original_config is None:
            original_config = validate_fit_execution_receipt(producer.get("fit_config"), producer.get("execution"))
        if self.tumor_id != catalog.tumor_id or content_hash(self.biological_identity) != content_hash(catalog.biological_identity):
            raise ValueError("Historical fit belongs to different biological input identities")
        manifest = producer.get("run_manifest")
        validate_run_manifest(manifest, tumor_id=catalog.tumor_id, biological_identity=self.biological_identity)
        if manifest["input_identity"].get("original_catalog_identity") not in catalog.compatible_source_identities(original_config.max_major_cn):
            raise ValueError("Historical run manifest belongs to different original input content")
        if self.biological_hash != _biological_hash(catalog, self.mutation_ids):
            raise ValueError("Historical denominators, masks, mapping, or observations differ")

    def validate(self, catalog, *, expected_ids=None, expected_support=None, device="cpu", cpu_threads=None, run_manifest_id=None):
        """Return a v2 replay receipt with a separate current compatibility check.

        Original execution evidence is preserved. Older summary/TSV formats are
        not imported, and insufficient evidence never triggers optimization.
        """
        from .io.reporting import _verify_fitted_numerics
        from .kernel.model import FitResult, RegionalModel
        from .kernel.topology import Membership
        if self.tumor_id != catalog.tumor_id:
            raise ValueError("Accepted fit belongs to a different originating tumor")
        from .provenance import (resolve_current_run_manifest_id, validate_fit_execution_receipt,
                                 validate_inherited_audit)
        from .kernel import ALGORITHM, KERNEL_VERSION
        from .kernel.model import MODEL_VERSION, INTERVAL_MODEL_VERSION
        run_manifest_id = resolve_current_run_manifest_id(run_manifest_id)
        if (not isinstance(self.biological_identity, Mapping)
                or content_hash(self.biological_identity) != content_hash(catalog.biological_identity)):
            raise ValueError("Historical tumor, regional mapping, mutation convention, or reference build differs")
        producer = self.producer_receipt
        if (not isinstance(producer, Mapping) or producer.get("schema") != "clipp2.fit-producer.v2"
                or producer.get("protocol_version") != self.protocol_version
                or producer.get("tumor_id") != self.tumor_id
                or any(not isinstance(producer.get(key), str) or not producer[key]
                       or producer[key] != producer[key].strip()
                       for key in ("implementation_version", "algorithm", "model_version", "kernel_version", "model_identity"))
                or producer.get("algorithm") != ALGORITHM or producer.get("kernel_version") != KERNEL_VERSION
                or producer.get("model_version") not in (MODEL_VERSION, INTERVAL_MODEL_VERSION)
                or content_hash(producer.get("biological_identity")) != content_hash(self.biological_identity)
                or not isinstance(producer.get("fit_config"), Mapping)
                or not isinstance(producer.get("execution"), Mapping)
                or producer.get("model_identity") != self.model_identity):
            raise ValueError("Historical fit lacks its original execution/configuration receipt")
        from .provenance import validate_run_manifest
        validate_run_manifest(producer.get("run_manifest"), tumor_id=catalog.tumor_id,
                              biological_identity=self.biological_identity)
        if producer.get("run_manifest_id") != producer["run_manifest"]["run_manifest_id"]:
            raise ValueError("Original producer run manifest link differs from its input binding")
        build = producer["run_manifest"].get("build")
        original_config = validate_fit_execution_receipt(producer["fit_config"], producer["execution"], build=build)
        if "normal_cn_policy" not in producer or producer["normal_cn_policy"] not in (None, "diploid"):
            raise ValueError("Original fit lacks its explicit normal-CN assumption")
        if len(self.centers) > original_config.max_clusters:
            raise ValueError("Accepted occupied count exceeds the original search capacity")
        validate_inherited_audit(self, original_config)
        if build is not None:
            artifact = build["artifact"]
            if (artifact["protocol_version"] != producer["protocol_version"]
                    or artifact["implementation_version"] != producer["implementation_version"]):
                raise ValueError("Original run/build manifest disagrees with producer execution identity")
        self.validate_input_identity(catalog, original_config)
        if tuple(self.region_ids) != tuple(catalog.region_ids):
            raise ValueError("Historical fit has different requested regions")
        if expected_ids is not None and set(self.mutation_ids) != set(expected_ids):
            raise ValueError("Historical fit population differs from the requested stage")
        if expected_support is not None:
            data = catalog.fit_data(self.mutation_ids, normal_cn_policy=producer["normal_cn_policy"])
            numerical, accepted_support = expected_support.for_data(data), self.support.for_data(data)
            if any(not np.array_equal(actual, expected) for actual, expected in zip(accepted_support, numerical)):
                raise ValueError("Historical fit has different actual multiplicity support")
        evidence = self.acceptance_evidence
        audit = evidence.get("membership_audit", {}) if isinstance(evidence, Mapping) else {}
        if (not isinstance(evidence, Mapping) or not isinstance(audit, Mapping)
                or evidence.get("independently_verified") is not True
                or audit.get("selected_count_certified") is not True or audit.get("complete") is not True):
            raise ValueError("Historical fit lacks accepted independent and terminal-audit evidence")
        publication = producer.get("publication", {})
        original_verification = publication.get("verification") if isinstance(publication, Mapping) else None
        if (not isinstance(original_verification, Mapping) or content_hash(original_verification) !=
                content_hash({key: value for key, value in evidence.items() if key != "accepted_state_hash"})):
            raise ValueError("Historical acceptance evidence disagrees with its original producer receipt")
        if evidence.get("accepted_state_hash") != self.state_fingerprint:
            raise ValueError("Historical acceptance evidence is not bound to this exact fitted state")
        model = RegionalModel(catalog.fit_data(self.mutation_ids, normal_cn_policy=producer["normal_cn_policy"]), device=device,
                              cpu_threads=cpu_threads, multiplicity_support=self.support)
        try:
            if (model.mutation_ids != self.mutation_ids or model.identity != self.model_identity
                    or model.model_version != producer["model_version"]):
                raise ValueError("Historical model identity or canonical population order differs")
            fit = FitResult(Membership(explicit_labels=self.labels), self.centers, self.weights,
                self.conditional_log_likelihood, self.mixture_log_likelihood, self.complexity_penalty,
                self.score, True)
            _verify_fitted_numerics(model, fit)
            calls = model.multiplicity(self.phi)
            if not np.array_equal(calls, self.multiplicities, equal_nan=True):
                raise ValueError("Historical accepted multiplicities disagree with their native support/CCFs")
            compatibility = dict(schema="clipp2.fit-compatibility.v2", protocol_version=PROTOCOL_VERSION,
                implementation_version=__version__, biological_identity=catalog.biological_identity,
                tumor_id=self.tumor_id, run_manifest_id=run_manifest_id,
                originating_run_manifest_id=producer.get("run_manifest_id"),
                originating_native_fit_protocol=self.protocol_version, current_workflow_protocol=PROTOCOL_VERSION,
                accepted_state_hash=self.state_fingerprint, model_identity=model.identity,
                execution=model.execution_diagnostics(), refitted=False,
                checks="biological_inputs_support_native_calls_independent_score",
                inherited_terminal_audit=True)
        finally:
            model.backend.close()
        return replace(self, compatibility_receipts=(*self.compatibility_receipts, compatibility))


def _fit_accepted(catalog, ids, bounds, outdir, stage, config, progress_callback, *, normal_cn_policy=None,
                  run_manifest=None, run_manifest_id=None):
    from .fit_execution import fit_prepared
    if run_manifest is None:
        from .provenance import prepare_run_manifest
        run_manifest = prepare_run_manifest(catalog, run_manifest_id=run_manifest_id)
    from .provenance import validate_run_manifest
    if run_manifest_id is not None and run_manifest_id != run_manifest["run_manifest_id"]:
        raise ValueError("Fit execution run manifest links disagree")
    run_manifest_id = run_manifest["run_manifest_id"]
    validate_run_manifest(run_manifest, tumor_id=catalog.tumor_id, biological_identity=catalog.biological_identity)
    if run_manifest["input_identity"].get("original_catalog_identity") != content_hash(catalog.to_dict()):
        raise ValueError("Run manifest does not identify this original input")
    callback = None if progress_callback is None else lambda record: progress_callback(dict(record, stage=stage))
    execution = fit_prepared(catalog.fit_data(ids, normal_cn_policy=normal_cn_policy), outdir, config, multiplicity_support=bounds,
        progress_callback=callback, run_manifest=run_manifest, run_manifest_id=run_manifest_id)
    fit, order = execution.fit, execution.public_order
    accepted = AcceptedFit(stage, execution.mutation_ids, execution.region_ids, execution.public_labels,
        fit.centers[order], fit.weights[order], execution.multiplicities, execution.support,
        _biological_hash(catalog, execution.mutation_ids), execution.summary["model_identity"], fit.score,
        fit.conditional_log_likelihood, fit.mixture_log_likelihood, fit.complexity_penalty,
        execution.summary["publication"]["verification"],
        producer_receipt=execution.producer_receipt,
        biological_identity=catalog.biological_identity, tumor_id=catalog.tumor_id)
    return replace(accepted, acceptance_evidence=dict(accepted.acceptance_evidence,
                                                     accepted_state_hash=accepted.state_fingerprint))


def _diagnostics_many(catalog, ids, coordinates, bounds, config, calls=None):
    """Reuse prepared observations and one backend for all coordinate screens."""
    observation = prepare_prediction(catalog, ids, bounds, device=config.fit.device,
                                     cpu_threads=config.fit.cpu_threads, normal_cn_policy=config.normal_cn_policy)
    from .screening import profile_deviance_from_diagnostics
    diagnostics = []
    with NativeBackend(observation, cpu_threads=config.fit.cpu_threads) as backend:
        for phi in coordinates:
            diagnostic = backend.component_diagnostics(phi, accepted_calls=calls)
            if np.any((diagnostic["status"] == 2) | (diagnostic["status"] == 3)):
                raise NativeExecutionError("Workflow component diagnostics failed")
            diagnostic["profile_deviance"] = profile_deviance_from_diagnostics(observation, phi, diagnostic)
            values = np.asarray(diagnostic["profile_deviance"])
            if (values.shape != observation.observed.shape or np.isnan(values[observation.observed]).any()
                    or np.any(values[observation.observed] < 0)):
                raise ArithmeticError("Group profile diagnostics returned invalid informative measurements")
            diagnostics.append(diagnostic)
    return diagnostics, observation


def _diagnostics(catalog, ids, phi, bounds, config, calls=None):
    diagnostics, observation = _diagnostics_many(catalog, ids, (phi,), bounds, config, calls)
    return diagnostics[0], observation


def collect_cap_contributors(catalog, fit, config, *, mutation_ids=None):
    """Collect eligible full-support MAP sets, preserving historical point calls."""
    wanted = set(fit.mutation_ids if mutation_ids is None else mutation_ids)
    if not wanted.issubset(fit.mutation_ids):
        raise ValueError("Requested contributors must belong to the source fit")
    selected = [i for i, mutation in enumerate(fit.mutation_ids) if mutation in wanted]
    if not selected:
        return []
    ids = tuple(fit.mutation_ids[i] for i in selected)
    support = fit.support.subset(ids, catalog.region_ids)
    original_support = catalog.default_bounds().subset(ids, catalog.region_ids)
    positions = {mutation: i for i, mutation in enumerate(catalog.mutation_ids)}
    eligible = catalog.eligible[[positions[mutation] for mutation in ids]]
    # Excluded cells use inert [1,1] numerical support, not their catalog CN.
    # They cannot supply caps; require original support only on eligible pairs.
    if not (np.array_equal(support.lower[eligible], original_support.lower[eligible])
            and np.array_equal(support.upper[eligible], original_support.upper[eligible])):
        raise ValueError("Cap source requires the original full multiplicity support")
    observation = prepare_prediction(catalog, ids, support, device=config.fit.device,
        cpu_threads=config.fit.cpu_threads, normal_cn_policy=config.normal_cn_policy)
    records = []
    for i, mutation in enumerate(ids):
        source, original = selected[i], positions[mutation]
        for j, region in enumerate(catalog.region_ids):
            phi = float(fit.centers[fit.labels[source], j])
            evidence = None
            if not catalog.eligible[original, j]:
                reason = "excluded"
            elif not catalog.mapping_genuine[original, j]:
                reason = "nongenuine_mapping"
            elif not observation.observed[i, j]:
                reason = "missing_observation"
            else:
                evidence = conditional_multiplicity(catalog, original, j, phi,
                    int(support.lower[i, j]), int(support.upper[i, j]))
                reason = ("flat_native_likelihood" if len(evidence["map_tied_values"]) == len(evidence["candidates"])
                          else "available") if evidence["defined"] else "model_incompatible"
                if evidence["defined"] and not (support.lower[i, j] <= evidence["map_call"] <= support.upper[i, j]
                        and evidence["map_call"] == int(evidence["map_call"])):
                    raise ArithmeticError("Conditional MAP call lies outside the native support")
            available = reason in ("available", "flat_native_likelihood")
            records.append(dict(mutation_id=mutation, region_id=region,
                segment_id=str(catalog.segment_ids[original, j]), source_stage=fit.source_id,
                source_label=int(fit.labels[source]), native_phi=phi,
                native_source_call=float(fit.multiplicities[source, j]),
                cap_source_map_call=evidence["map_call"] if available else None,
                cap_source_map_upper=max(evidence["map_tied_values"]) if available else None,
                native_lower=int(support.lower[i, j]), native_upper=int(support.upper[i, j]),
                map_probability=evidence["map_probability"] if available else None,
                map_unique=evidence["map_unique"] if available else None,
                map_tied_values=evidence["map_tied_values"] if available else (),
                map_score_gap=evidence["map_score_gap"] if available else None,
                profile_deviance=evidence["profile_deviance"] if evidence is not None else None,
                multiplicity_read_informative=available and phi > 0,
                contributes_to_cap=available, availability_reason=reason,
                diagnostic_status=evidence["diagnostic_status"] if evidence is not None else "not_applicable",
                diagnostic_error=evidence["diagnostic_error"] if evidence is not None else None))
    return records


def classify_groups(catalog, fit, config):
    """Measure native groups without acceptance or geometric routing decisions."""
    from .screening import summarize_profile_deviance
    observation = prepare_prediction(catalog, fit.mutation_ids, fit.support, device=config.fit.device,
        cpu_threads=config.fit.cpu_threads, normal_cn_policy=config.normal_cn_policy)
    coordinates = (fit.phi, np.ones_like(fit.phi))
    diagnostics, error, same = {}, None, False
    try:
        native, _ = _diagnostics_many(catalog, fit.mutation_ids, coordinates, fit.support, config)
        original_bounds = catalog.default_bounds().subset(fit.mutation_ids, catalog.region_ids)
        lo, hi = original_bounds.for_data(observation.data)
        same = np.array_equal(observation.multiplicity_min, lo) and np.array_equal(observation.multiplicity_max, hi)
        original = native if same else _diagnostics_many(catalog, fit.mutation_ids, coordinates, original_bounds, config)[0]
        diagnostics = dict(native=native[0], all_one=native[1], original_native=original[0], original_all_one=original[1])
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught))
    groups = []
    for label, center in enumerate(fit.centers):
        members = fit.labels == label
        mask = observation.observed[members]
        coverage = mask.sum(axis=0)
        depths = np.where(mask, observation.depth[members], 0).sum(axis=0)
        distance = float(np.linalg.norm(center - 1))
        record = dict(source_stage=fit.source_id, source_label=label, size=int(members.sum()),
            native_center=center.tolist(), distance_to_one=distance,
            informative_counts=coverage.tolist(), informative_depths=depths.tolist(),
            minimum_regional_depth=float(depths.min()),
            diagnostic_status="failed" if error else "complete", diagnostic_error=error)
        for name in (("native", "all_one") if same else ("native", "all_one", "original_native", "original_all_one")):
            summary = (summarize_profile_deviance(diagnostics[name]["profile_deviance"][members], mask, axis=0)
                if diagnostics else dict(informative_count=coverage,
                    mean_profile_deviance=np.full(len(coverage), np.nan), maximum_profile_deviance=np.full(len(coverage), np.nan)))
            record.update({name + "_" + key: values.tolist() for key, values in summary.items()})
        if same:
            for name in ("native", "all_one"):
                for field in ("informative_count", "mean_profile_deviance", "maximum_profile_deviance"):
                    record["original_"+name+"_"+field] = record[name+"_"+field]
        groups.append(record)
    return groups


def select_reference_source(groups):
    """Nearest occupied native center; depth breaks only an exact distance tie."""
    if not groups or any(group["size"] < 1 or not np.isfinite(group["distance_to_one"]) for group in groups):
        raise ValueError("Reference source selection requires occupied groups with finite native centers")
    selected = min(groups, key=lambda g: (g["distance_to_one"], -g["minimum_regional_depth"], g["source_label"]))
    return dict(selected, reference_source_role="provisional_reference_source", selection_rule="nearest_to_one")


def _build_stage_three_inputs(catalog, stage_one, stage_two, selected_group, group_two, config):
    """Use one fixed clonal hypothesis; retain Stage 1's estimate as provenance."""
    selected = selected_group
    label = selected["source_label"]
    if label not in range(len(stage_one.centers)) or not np.any(stage_one.labels == label):
        raise ValueError("The provisional source must be an occupied Stage 1 group")
    if not np.array_equal(selected["native_center"], stage_one.centers[label]):
        raise ValueError("Provisional source diagnostics describe a different native center")
    expected = set() if stage_two is None else set(range(len(stage_two.centers)))
    if len(group_two) != len(expected) or {group["source_label"] for group in group_two} != expected:
        raise ValueError("Stage 2 group diagnostics must cover exactly the accepted native groups")
    r = len(catalog.region_ids)
    centers, source_map = [np.ones(r)], []
    sources = [[dict(center_source="fixed_all_ones", interpretation="imposed_hypothesis",
        diagnostic_source=dict(source_stage=stage_one.source_id, source_label=label,
            interpretation="provisional_source_reads_at_fixed_all_ones"))]]
    measurements = [[]]
    inventory = {tuple(centers[0]): 0}
    for fit, groups in ((stage_one, [selected]), (stage_two, sorted(group_two, key=lambda g: g["source_label"]))):
        for group in groups:
            source_label = group["source_label"]
            center = fit.centers[source_label]
            if (not np.array_equal(group["native_center"], center)
                    or group.get("source_stage", fit.source_id) != fit.source_id):
                raise ValueError("Group diagnostics describe a different accepted native source")
            # The selected Stage 1 source and the fixed clonal hypothesis are
            # one reference, even when the fitted source is not exactly one.
            # Other native centers are not geometrically merged or altered.
            key = tuple(centers[0] if fit is stage_one else center)
            if key not in inventory:
                inventory[key] = len(centers)
                centers.append(center)
                sources.append([])
                measurements.append([])
            baseline = inventory[key]
            role = "provisional_reference_source" if fit is stage_one else "native_discovery_group"
            lineage = dict(source_stage=fit.source_id, source_label=source_label,
                baseline_reference_label=baseline, reference_source_role=role,
                native_center=center.tolist(), distance_to_one=float(np.linalg.norm(center - 1)))
            if fit is stage_one:
                lineage["selection_rule"] = "nearest_to_one"
                lineage["reference_center_policy"] = "fixed_all_ones"
            else:
                lineage["reference_center_policy"] = "native_stage_two"
            source_map.append(lineage)
            sources[baseline].append(dict(lineage,
                center_source="accepted_stage_one_clonal_source" if fit is stage_one else "accepted_stage_two_native",
                accepted_state_hash=fit.state_fingerprint, support_hash=fit.support.content_hash,
                source_measurements={name: group[name] for name in (
                    "informative_counts", "informative_depths", "native_mean_profile_deviance",
                    "native_maximum_profile_deviance", "diagnostic_status") + ((
                    "all_one_mean_profile_deviance", "all_one_maximum_profile_deviance")
                    if fit is stage_one else ())},
                reference_measurement="all_one" if fit is stage_one else "native",
                diagnostic_error=group.get("diagnostic_error")))
            measurements[baseline].append((group, "all_one" if fit is stage_one else "native"))
    if sum(np.all(center == 1) for center in centers) != 1 or source_map[0]["baseline_reference_label"] != 0:
        raise AssertionError("Stage 3 requires one all-one reference including the Stage 1 source")
    counts, depths, means, maxima, availability = [], [], [], [], []
    for records in measurements:
        count = np.asarray([g["informative_counts"] for g, _ in records], float)
        depth = np.asarray([g["informative_depths"] for g, _ in records], float)
        mean = np.asarray([g[p+"_mean_profile_deviance"] for g, p in records], float)
        maximum = np.asarray([g[p+"_maximum_profile_deviance"] for g, p in records], float)
        total, regional_mean, regional_maximum = count.sum(axis=0), np.full(r, np.nan), np.full(r, np.nan)
        state = []
        for j in range(r):
            active = count[:, j] > 0
            failed = any(g["diagnostic_status"] == "failed" for (g, _), use in zip(records, active) if use)
            missing = np.isnan(mean[active, j]).any() or np.isnan(maximum[active, j]).any()
            state.append("failed" if failed else "missing" if not active.any() or missing else "available")
            if state[-1] == "available":
                regional_mean[j] = np.sum(mean[active, j] * (count[active, j] / total[j]))
                regional_maximum[j] = maximum[active, j].max()
        counts.append(total)
        depths.append(depth.sum(axis=0))
        means.append(regional_mean)
        maxima.append(regional_maximum)
        availability.append(state)
    reference = AssignmentReference(tuple(range(len(centers))), catalog.region_ids, np.asarray(centers), tuple(source_map),
        source_informative_count=np.asarray(counts), source_informative_depth=np.asarray(depths),
        source_mean_profile_deviance=np.asarray(means), source_maximum_profile_deviance=np.asarray(maxima),
        diagnostic_availability=availability,
        center_sources=tuple(dict(center_source="single_clonal_and_exact_native_consolidation", sources=entries,
            diagnostic_aggregation="disjoint_native_source_members_or_declared_fixed_hypothesis_source") for entries in sources))
    g0 = tuple(mutation for mutation, label in zip(stage_one.mutation_ids, stage_one.labels)
               if label == selected["source_label"])
    second_ids = () if stage_two is None else stage_two.mutation_ids
    if not g0 or set(g0) & set(second_ids) or set(g0) | set(second_ids) != set(stage_one.mutation_ids):
        raise ValueError("Accepted Stage 2 and provisional source must partition Stage 1")
    return reference, tuple(sorted(g0 + second_ids)), {}, source_map


def _stage_two_caps(catalog, audits):
    """Minimum upper endpoint of available exact native MAP sets."""
    original = catalog.default_bounds()
    upper = np.array(original.upper)
    contributors = {}
    for record in audits:
        if record["contributes_to_cap"]:
            key, call = (record["region_id"], record["segment_id"]), int(record["cap_source_map_upper"])
            previous, count = contributors.get(key, (call, 0))
            contributors[key] = min(previous, call), count + 1
    policies = {}
    for i, _ in enumerate(catalog.mutation_ids):
        for j, region in enumerate(catalog.region_ids):
            key = region, str(catalog.segment_ids[i, j])
            endpoint, count = contributors.get(key, (None, 0))
            if key not in policies:
                policies[key] = dict(region_id=region, segment_id=key[1],
                    multiplicity_endpoint=endpoint, contributor_count=count,
                    fallback=count == 0, fallback_policy="original_upper",
                    applies_to="originally_eligible_genuine_pairs", policy="minimum_native_map_set_upper")
            if count and catalog.mapping_genuine[i, j] and catalog.eligible[i, j]:
                upper[i, j] = endpoint
            if original.lower[i, j] > upper[i, j] or upper[i, j] > original.upper[i, j]:
                raise ValueError("Segment-derived cap exceeds the original biological support")
    return MultiplicityBounds(catalog.mutation_ids, catalog.region_ids,
        original.lower, upper, "minimum_native_map_set_upper"), list(policies.values())


def run_workflow(input_path, outdir, workflow_config, *, progress_callback=None,
                 accepted_stage_one=None, accepted_stage_two=None, run_build_manifest=None, run_manifest_id=None):
    """Run applicable stages or replay complete accepted-fit.v3 receipts.

    Both native stages support explicit replay. Older API summaries/TSVs lack
    the required producer evidence and are not imported or implicitly refitted.
    """
    from .io.assignment import (load_accepted_fit, publish_workflow_receipt, reserve_output,
                                save_accepted_fit, save_assignment_bundle, save_original_catalog,
                                write_supporting_table, _publish_json, publish_assignment)
    if type(workflow_config) is not WorkflowConfig:
        raise TypeError("workflow_config must be a WorkflowConfig")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    config, output = workflow_config, reserve_output(outdir)
    common_receipt = dict(schema="clipp2.workflow.v5", protocol_version=PROTOCOL_VERSION,
        implementation_version=__version__, config=asdict(config),
        source_policy="nearest_to_one", cap_policy="minimum_native_map_set_upper",
        reference_policy=STAGE_THREE_REFERENCE_POLICY,
        reassessment_policy="all_combined_fitted_mutations",
        provenance_policy="biological_and_numerical_identity_no_file_checksums")
    try:
        catalog = load_original_catalog(input_path, max_major_cn=config.fit.max_major_cn, missing_counts=config.missing_counts)
    except Exception as error:
        from .io.tumor_txt import audit_input
        audit = audit_input(input_path, max_major_cn=config.fit.max_major_cn, normal_cn_policy=config.normal_cn_policy,
                            missing_counts=config.missing_counts)
        publish_workflow_receipt(output, dict(common_receipt, status="failed", failed_stage="input",
            evidence_status=audit["status"] if audit["status"] != "ready" else "computation_failed",
            input_audit=audit, error_type=type(error).__name__, error=str(error)))
        raise
    save_original_catalog(catalog, output / "original_catalog.json")
    common_receipt["read_counts"] = catalog.read_count_report
    from .io.data import validate_workflow_mapping
    try:
        validate_workflow_mapping(catalog, normal_cn_policy=config.normal_cn_policy)
    except ValueError as error:
        from .io.tumor_txt import audit_input
        audit = audit_input(input_path, max_major_cn=config.fit.max_major_cn, normal_cn_policy=config.normal_cn_policy,
                            missing_counts=config.missing_counts)
        publish_workflow_receipt(output, dict(common_receipt, status="failed", failed_stage="input",
            evidence_status=audit["status"] if audit["status"] != "ready" else "input_invalid", input_audit=audit,
            reason="unsupported_biological_mapping", error=str(error),
            biological_identity=catalog.biological_identity))
        raise
    from .provenance import prepare_run_manifest
    try:
        run_manifest = prepare_run_manifest(catalog, supplied=run_build_manifest, run_manifest_id=run_manifest_id)
    except (TypeError, ValueError, OSError) as error:
        publish_workflow_receipt(output, dict(common_receipt, status="failed", failed_stage="provenance",
            error_type=type(error).__name__, error=str(error)))
        raise
    _publish_json(output / "run_manifest.json", run_manifest)
    run_manifest_id = run_manifest["run_manifest_id"]
    common_receipt.update(run_manifest=run_manifest, run_manifest_id=run_manifest_id)
    def replay_policy(value, stage, settings):
        from .provenance import validate_replay_policy
        accepted = value if isinstance(value, AcceptedFit) else load_accepted_fit(value)
        if accepted.source_id != stage:
            raise ValueError("Accepted receipt source stage does not match requested replay stage")
        original = validate_replay_policy(accepted, settings, normal_cn_policy=config.normal_cn_policy)
        accepted.validate_input_identity(catalog, original)
        return accepted
    def replay(accepted, ids, bounds, settings):
        return accepted.validate(catalog, expected_ids=ids, expected_support=bounds,
                                 device=settings.device, cpu_threads=settings.cpu_threads, run_manifest_id=run_manifest_id)
    informative = catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)
    population = tuple(m for m, keep in zip(catalog.mutation_ids, informative.any(axis=1)) if keep)
    full = catalog.default_bounds()
    write_supporting_table(output / "mutation_segments.tsv", [dict(mutation_id=mutation,
        region_id=region, segment_id=str(catalog.segment_ids[i, j]),
        mapping_genuine=bool(catalog.mapping_genuine[i, j]), original_eligible=bool(catalog.eligible[i, j]),
        input_excluded=bool(catalog.input_excluded[i, j]),
        original_lower=int(full.lower[i, j]), original_upper=int(full.upper[i, j]))
        for i, mutation in enumerate(catalog.mutation_ids) for j, region in enumerate(catalog.region_ids)],
        columns=("mutation_id", "region_id", "segment_id", "mapping_genuine", "original_eligible", "input_excluded",
                 "original_lower", "original_upper"))
    receipt = dict(common_receipt, status="running",
        biological_identity=catalog.biological_identity,
        stage_one_candidate_count=len(population), stage_two_candidate_count=0,
        original_mutation_count=len(catalog.mutation_ids),
        stage_one="pending", stage_two="not_applicable", stage_three="not_applicable",
        counts=dict(N1=0, N0=0, N2=0, Q2=0, merged_groups=0, combined_mutations=0, K=0))
    current_stage = "stage_one"
    receipt["current_phase"] = "stage_one_fit"
    def unavailable_original(first):
        rows, accounting = _reconcile_final_memberships(catalog, first, None, [], {}, None)
        write_supporting_table(output / "all_original_mutations.tsv", rows, columns=ORIGINAL_COMPARISON_COLUMNS)
        receipt["final_accounting"] = accounting
    try:
        stage_two_config = replace(config.fit, max_clusters=config.stage_two_max_clusters)
        # A supplied replay must not silently start a different search in either
        # stage before its historical scientific policy has been checked.
        for stage, value, settings in (("stage_one", accepted_stage_one, config.fit),
                                        ("stage_two", accepted_stage_two, stage_two_config)):
            if value is not None:
                current_stage = stage
                receipt["current_phase"] = stage + "_replay_policy"
                accepted = replay_policy(value, stage, settings)
                if stage == "stage_one":
                    accepted_stage_one = accepted
                else:
                    accepted_stage_two = accepted
        current_stage = "stage_one"
        receipt["current_phase"] = "stage_one_fit"
        if accepted_stage_one is None:
            stage_one = _fit_accepted(catalog, population, full, output / "stage_one", "stage_one", config.fit, progress_callback,
                                      normal_cn_policy=config.normal_cn_policy, run_manifest=run_manifest, run_manifest_id=run_manifest_id)
        else:
            stage_one = replay(accepted_stage_one, population, full, config.fit)
        if set(stage_one.mutation_ids) != set(population):
            raise AssertionError("Accepted Stage 1 population differs from the numerical candidate population")
        receipt["counts"]["N1"] = len(stage_one.mutation_ids)
        receipt["stage_one"] = "accepted"
        receipt["current_phase"] = "stage_one_record_publication"
        save_accepted_fit(stage_one, output / "stage_one_accepted.json")
        receipt["current_phase"] = "stage_one_diagnostics"
        group_one = classify_groups(catalog, stage_one, config)
        receipt["current_phase"] = "stage_one_group_publication"
        write_supporting_table(output / "stage_one_groups.tsv", group_one, columns=GROUP_COLUMNS)
        selected = select_reference_source(group_one)
        g0 = tuple(m for m, label in zip(stage_one.mutation_ids, stage_one.labels) if label == selected["source_label"])
        receipt["reference"] = dict(source_id=stage_one.source_id, source_label=selected["source_label"], mutation_ids=g0,
            reference_source_role="provisional_reference_source", selection_rule="nearest_to_one",
            native_center=selected["native_center"], distance_to_one=selected["distance_to_one"])
        receipt["counts"]["N0"] = len(g0)
        reference_members = set(g0)
        remainder = tuple(m for m in population if m not in reference_members)
        if reference_members & set(remainder):
            raise AssertionError("Reference membership overlaps the stage-two remainder")
        receipt["stage_two_candidate_count"] = len(remainder)
        current_stage = "stage_two"
        receipt["stage_two"] = "preparing"
        receipt["current_phase"] = "stage_two_cap_construction"
        reference_audit = collect_cap_contributors(catalog, stage_one, config, mutation_ids=g0)
        write_supporting_table(output / "stage2_cap_contributors.tsv", reference_audit, columns=CONTRIBUTOR_COLUMNS)
        caps, cap_rows = _stage_two_caps(catalog, reference_audit)
        write_supporting_table(output / "stage2_segment_multiplicity_bounds.tsv", cap_rows, columns=SEGMENT_POLICY_COLUMNS)
        receipt["current_phase"] = "stage_two_feasibility"
        stage_two, group_two = None, []
        if not remainder:
            if accepted_stage_two is not None:
                raise ValueError("Stage-two replay is inapplicable to an empty remainder")
            receipt["stage_two"] = "skipped_no_eligible_informative_remainder"
        else:
            positions = {mutation: i for i, mutation in enumerate(catalog.mutation_ids)}
            indices = [positions[m] for m in remainder]
            unsupported = [region for region, ok in zip(catalog.region_ids, informative[indices].any(axis=0)) if not ok]
            if unsupported:
                receipt.update(status="blocked", stage_two="unsupported_requested_regions", unsupported_regions=unsupported)
                unavailable_original(stage_one)
                publish_workflow_receipt(output, receipt)
                return receipt
            receipt["current_phase"] = "stage_two_fit"
            if accepted_stage_two is None:
                stage_two = _fit_accepted(catalog, remainder, caps.subset(remainder), output / "stage_two",
                    "stage_two", stage_two_config, progress_callback, normal_cn_policy=config.normal_cn_policy,
                    run_manifest=run_manifest, run_manifest_id=run_manifest_id)
            else:
                stage_two = replay(accepted_stage_two, remainder, caps.subset(remainder), stage_two_config)
            if set(stage_two.mutation_ids) != set(remainder):
                raise AssertionError("Accepted Stage 2 population differs from its numerical candidate population")
            receipt["counts"].update(N2=len(stage_two.mutation_ids), Q2=len(stage_two.centers))
            receipt["stage_two"] = "accepted"
            receipt["current_phase"] = "stage_two_record_publication"
            save_accepted_fit(stage_two, output / "stage_two_accepted.json")
            receipt["current_phase"] = "stage_two_diagnostics"
            group_two = classify_groups(catalog, stage_two, config)
        receipt["current_phase"] = "stage_two_group_publication"
        write_supporting_table(output / "stage_two_groups.tsv", group_two, columns=GROUP_COLUMNS)
        current_stage = "stage_three"
        receipt["current_phase"] = "stage_three_construction"
        reference, targets, unchanged_map, source_map = _build_stage_three_inputs(
            catalog, stage_one, stage_two, selected, group_two, config)
        centers = reference.centers
        stage_two_members = set(targets) - reference_members
        non_targets = tuple(unchanged_map)
        if non_targets or set(targets) != set(stage_one.mutation_ids):
            raise AssertionError("Protocol 2.3 must reassess the entire combined fitted population")
        combined = set(targets) | set(non_targets)
        q2 = 0 if stage_two is None else len(stage_two.centers)
        receipt["counts"].update(merged_groups=0, clonal_source_consolidated=1,
            exact_duplicate_centers=1 + q2 - len(centers),
            combined_mutations=len(combined), K=len(centers),
            targets=len(targets), non_targets=len(non_targets), unchanged_non_target=len(non_targets))
        write_supporting_table(output / "source_to_reference.tsv", source_map)
        write_supporting_table(output / "reference_vectors.tsv", [dict(reference_label=label,
            prior=1 / len(centers), **{"phi_" + region: float(center[j]) for j, region in enumerate(catalog.region_ids)})
            for label, center in enumerate(centers)])
        baselines = {}
        for fit, members in ((stage_one, reference_members), (stage_two, stage_two_members)):
            if fit is not None:
                for i, mutation in enumerate(fit.mutation_ids):
                    if mutation in members:
                        baselines[mutation] = (fit, i)
        baseline_rows = [baselines[mutation] for mutation in targets]
        source_labels = {(row["source_stage"], row["source_label"]): row["baseline_reference_label"] for row in source_map}
        bundle = AssignmentBundle(catalog, reference, targets, non_targets,
            np.asarray([fit.centers[fit.labels[i]] for fit, i in baseline_rows]),
            np.asarray([fit.multiplicities[i] for fit, i in baseline_rows]),
            np.asarray([fit.support.lower[i] for fit, i in baseline_rows]),
            np.asarray([fit.support.upper[i] for fit, i in baseline_rows]),
            tuple(fit.source_id for fit, _ in baseline_rows),
            tuple(int(fit.labels[i]) for fit, i in baseline_rows),
            tuple(source_labels[(fit.source_id, int(fit.labels[i]))] for fit, i in baseline_rows),
            device=config.fit.device, cpu_threads=config.fit.cpu_threads,
            normal_cn_policy=config.normal_cn_policy, run_manifest_id=run_manifest_id,
            provenance=dict(stage_two_caps_hash=caps.content_hash,
                 reference_contributors=reference_audit, native_stage_two_centers_preserved=True,
                 reference_policy=STAGE_THREE_REFERENCE_POLICY,
                 run_manifest_identity=run_manifest["manifest_identity"]))
        positions = {mutation: i for i, mutation in enumerate(catalog.mutation_ids)}
        target_rows = [positions[mutation] for mutation in targets]
        eligible_targets = catalog.eligible[target_rows]
        if (np.any(bundle.bounds.lower[eligible_targets] != 1)
                or not np.array_equal(bundle.bounds.upper[eligible_targets], catalog.major_cn[target_rows][eligible_targets])):
            raise AssertionError("Stage 3 must use full original multiplicity support")
        save_assignment_bundle(bundle, output / "assignment_bundle.json")
        receipt["current_phase"] = "stage_three_assignment"
        assignment = assign_mutations(bundle, progress_callback=progress_callback, run_manifest_id=run_manifest_id)
        receipt["stage_three"] = "scored"
        if assignment.target_ids != targets:
            raise AssertionError("Stage 3 result population differs from the frozen targets")
        reassessment = {status: sum(row["reassessment_status"] == status for row in assignment.assignments)
                        for status in ("retained", "reassigned", "unresolved")}
        if sum(reassessment.values()) != len(targets):
            raise AssertionError("Stage 3 reassessments do not reconcile the frozen targets")
        receipt["current_phase"] = "final_reconciliation"
        original_rows, final_accounting = _reconcile_final_memberships(
            catalog, stage_one, stage_two, source_map, unchanged_map, assignment, contributors=reference_audit)
        receipt.update(final_accounting=final_accounting, bundle_hash=bundle.content_hash,
                       exact_reference_consolidation_is_refit=False)
        receipt["counts"].update(reassessment)
        receipt["current_phase"] = "whole_original_publication"
        write_supporting_table(output / "all_original_mutations.tsv", original_rows, columns=ORIGINAL_COMPARISON_COLUMNS)
        receipt["current_phase"] = "assignment_publication"
        publish_assignment(bundle, assignment, output / "assignment", workflow_accounting=final_accounting)
        receipt.update(status="complete", stage_three="assigned", current_phase="workflow_publication")
        publish_workflow_receipt(output, dict(receipt, current_phase="complete"))
        receipt["current_phase"] = "complete"
        return receipt
    except Exception as error:
        receipt.update(status="failed", evidence_status="computation_failed", failed_stage=current_stage, failed_phase=receipt["current_phase"],
                       error_type=type(error).__name__, error=str(error))
        if receipt[current_stage] not in ("accepted", "scored", "assigned"):
            receipt[current_stage] = "failed"
        if not (output / "all_original_mutations.tsv").exists():
            try:
                rows, accounting = _reconcile_final_memberships(catalog, locals().get("stage_one"),
                    locals().get("stage_two"), locals().get("source_map", []), {}, locals().get("assignment"))
                write_supporting_table(output / "all_original_mutations.tsv", rows, columns=ORIGINAL_COMPARISON_COLUMNS)
                receipt["final_accounting"] = accounting
            except Exception as publication_error:
                receipt["failure_accounting_error"] = dict(type=type(publication_error).__name__, message=str(publication_error))
        publish_workflow_receipt(output, receipt)
        raise


def _reconcile_final_memberships(catalog, stage_one, stage_two, source_map, unchanged_map, result,
                                 *, contributors=()):
    """Construct whole-original final labels and combined accounting exactly once."""
    if unchanged_map:
        raise ValueError("The current workflow reassesses every combined fitted mutation")
    first, second, assignment = stage_one, stage_two, result
    mapping = {(row["source_stage"], row["source_label"]): row["baseline_reference_label"] for row in source_map}
    positions = [{} if fit is None else {m: i for i, m in enumerate(fit.mutation_ids)} for fit in (first, second)]
    assignments = {} if assignment is None else dict(zip(assignment.target_ids, assignment.assignments, strict=True))
    historical = {row["mutation_id"] for row in contributors if row["contributes_to_cap"]}
    rows = []
    for i, mutation in enumerate(catalog.mutation_ids):
        one = None if mutation not in positions[0] else int(first.labels[positions[0][mutation]])
        two = None if mutation not in positions[1] else int(second.labels[positions[1][mutation]])
        baseline = mapping.get((second.source_id, two)) if two is not None else (mapping.get((first.source_id, one)) if one is not None else None)
        record = assignments.get(mutation)
        non_target = mutation in unchanged_map
        target = baseline is not None and not non_target
        reason = ("stage2_recheck" if two is not None else "stage1_reference_recheck") if target else None
        if non_target and (two is None or unchanged_map[mutation] != baseline):
            raise ValueError("Unchanged population does not match accepted Stage 2 source construction")
        row = dict(mutation_id=mutation, original_eligible=bool(catalog.eligible[i].any()),
            original_input_status="filtered" if not catalog.eligible[i].any() else "no_information"
                if not (catalog.eligible[i] & catalog.count_observed[i] & (catalog.total_counts[i] > 0)).any() else "eligible_informative",
            exclusion_reasons=tuple(reason for reason, applies in (
                ("explicit_input_exclusion", catalog.input_excluded[i].any()),
                ("declared_major_cn_policy", catalog.max_major_cn is not None
                    and (catalog.major_cn[i] > catalog.max_major_cn).any())) if applies),
            stage_one_label=one, stage_two_label=two, baseline_reference_label=baseline,
            final_assignment_label=unchanged_map.get(mutation), target_reason=reason,
            membership_origin="stage2_unchanged" if non_target else reason,
            historical_cap_contributor_reassigned=False,
            role="non_target" if non_target else "target" if target else "not_in_combined_baseline",
            assignment_status=None,
            evidence_status="excluded" if not catalog.eligible[i].any() else "computation_failed",
            map_cluster_label=None, map_label=None, map_set=(), posterior_defined=False,
            reference_labels=() if assignment is None else assignment.reference.labels, probabilities=None,
            reassessment_status="unchanged_non_target" if non_target else "unavailable" if target else None,
            baseline_changed=False if non_target else None)
        for name, fit, lookup in (("stage_one", first, positions[0]), ("stage_two", second, positions[1])):
            index = lookup.get(mutation)
            row[name + "_native_phi"] = None if index is None else fit.centers[fit.labels[index]].tolist()
            row[name + "_native_calls"] = None if index is None else fit.multiplicities[index].tolist()
        if record is not None:
            if not target or record["baseline_label"] != baseline or record["target_reason"] != reason:
                raise ValueError("Target reassessment does not match its native source and baseline")
            row.update(assignment_status=record["status"], evidence_status=record["evidence_status"], map_cluster_label=record["map_cluster_label"],
                map_label=record["map_label"], map_set=record["map_set"], posterior_defined=record["posterior_defined"],
                probabilities=record["probabilities"],
                final_assignment_label=record["final_assignment_label"], reassessment_status=record["reassessment_status"],
                baseline_changed=record["baseline_changed"],
                historical_cap_contributor_reassigned=bool(mutation in historical and record["final_assignment_label"] is not None
                    and record["final_assignment_label"] != baseline))
        elif row["original_input_status"] == "no_information":
            row["evidence_status"] = "prior_only"
            if assignment is not None:
                row.update(posterior_defined=True, probabilities=assignment.reference.priors,
                    map_label=min(assignment.reference.labels), map_cluster_label=min(assignment.reference.labels),
                    map_set=tuple(sorted(assignment.reference.labels)))
        rows.append(row)
    if assignment is None:
        return rows, dict(status="unavailable", population_count=0, unchanged_map=dict(unchanged_map))
    targets = {row["mutation_id"] for row in rows if row["role"] == "target"}
    if targets != set(assignment.target_ids) or targets & set(unchanged_map):
        raise ValueError("Result targets and unchanged members do not match the declared population partition")
    if set(unchanged_map) - set(catalog.mutation_ids):
        raise ValueError("Unchanged members must belong to the original catalog")
    counts = {str(label): sum(row["final_assignment_label"] == label for row in rows) for label in assignment.reference.labels}
    unchanged = {str(label): sum(value == label for value in unchanged_map.values()) for label in assignment.reference.labels}
    unresolved = sum(row["role"] == "target" and row["final_assignment_label"] is None for row in rows)
    population = len(targets) + len(unchanged_map)
    if sum(counts.values()) + unresolved != population:
        raise AssertionError("Final MAP occupancy and unassigned targets do not reconcile T/F")
    for label in assignment.reference.labels:
        if counts[str(label)] != unchanged[str(label)] + sum(row["final_assignment_label"] == label for row in assignment.assignments):
            raise AssertionError("Final occupancy differs from unchanged members plus observed MAP targets")
    origins = {reason: {outcome: sum(row["target_reason"] == reason and row["reassessment_status"] == outcome for row in rows)
                       for outcome in ("retained", "reassigned", "unresolved")}
               for reason in ("stage1_reference_recheck", "stage2_recheck")}
    return rows, dict(status="complete", reassessment_policy="all_combined_fitted_mutations",
        population_count=population, final_occupancy=counts,
        unchanged_map=dict(unchanged_map), unchanged_non_target_occupancy=unchanged, unresolved_targets=unresolved,
        outcomes_by_origin=origins,
        evidence_status_counts={status: sum(row["evidence_status"] == status for row in assignment.assignments)
            for status in EVIDENCE_STATUSES},
        original_evidence_status_counts={status: sum(row["evidence_status"] == status for row in rows)
            for status in EVIDENCE_STATUSES},
        historical_cap_contributors_reassigned=sum(row["historical_cap_contributor_reassigned"] for row in rows),
        historical_caps_recomputed=False)
