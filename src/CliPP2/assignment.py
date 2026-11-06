"""Immutable fixed-reference prediction, separate from occupied fitted states."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy, xlog1py

from .config import AssignmentThresholds
from .io.data import (ImmutableArrayRecord, MultiplicityBounds, OriginalCatalog, readonly_array,
                      json_value as _json_value, freeze_record as _freeze, content_hash)
from .kernel.backend import NativeBackend, NativeExecutionError
from .kernel.observations import prepare_prediction
from .screening import summarize_profile_deviance


@dataclass(frozen=True, slots=True)
class AssignmentReference(ImmutableArrayRecord):
    labels: tuple[int, ...]
    region_ids: tuple[str, ...]
    centers: np.ndarray
    lineage: tuple = ()
    clonal_label: int | None = 0

    def __post_init__(self):
        labels, regions = tuple(self.labels), tuple(self.region_ids)
        centers = np.asarray(self.centers, dtype=float)
        if (not labels or len(set(labels)) != len(labels) or any(type(k) is not int or k < 0 for k in labels)
                or not regions or len(set(regions)) != len(regions)
                or any(not isinstance(region, str) or not region for region in regions)
                or centers.shape != (len(labels), len(regions)) or not np.isfinite(centers).all()
                or np.any(centers < 0) or np.any(centers > 1)):
            raise ValueError("Reference needs unique labels and complete finite regional CCF vectors")
        if self.clonal_label is not None and (type(self.clonal_label) is not int or self.clonal_label not in labels or
                not np.all(centers[labels.index(self.clonal_label)] == 1)):
            raise ValueError("The designated clonal reference must be exactly all-one")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "region_ids", regions)
        object.__setattr__(self, "centers", readonly_array(centers))
        object.__setattr__(self, "lineage", _freeze(self.lineage))

    @property
    def priors(self):
        return readonly_array(np.full(len(self.labels), 1 / len(self.labels)))

    def to_dict(self):
        return _json_value(dict(labels=self.labels, region_ids=self.region_ids, centers=self.centers,
                               priors=self.priors, lineage=self.lineage, clonal_label=self.clonal_label))

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        priors = np.asarray(value.pop("priors"), float)
        result = cls(**value)
        if not np.array_equal(priors, result.priors):
            raise ValueError("Assignment reference must have fixed equal priors")
        return result


@dataclass(frozen=True, slots=True)
class AssignmentBundle:
    catalog: OriginalCatalog
    reference: AssignmentReference
    target_ids: tuple[str, ...]
    donor_ids: tuple[str, ...]
    bounds: MultiplicityBounds
    thresholds: AssignmentThresholds
    native_centers: np.ndarray
    native_calls: np.ndarray
    native_lower: np.ndarray
    native_upper: np.ndarray
    source_stages: tuple[str, ...]
    source_labels: tuple[int, ...]
    device: str = "cpu"
    cpu_threads: int | None = None
    provenance: object = field(default_factory=dict)
    original_bounds: MultiplicityBounds = field(init=False, repr=False)
    _content_hash: str = field(init=False, repr=False)

    def __post_init__(self):
        targets, donors = tuple(self.target_ids), tuple(self.donor_ids)
        from .io.data import validate_workflow_mapping
        validate_workflow_mapping(self.catalog, targets + donors)
        if (len(set(targets)) != len(targets) or len(set(donors)) != len(donors)
                or set(targets) & set(donors) or not set(targets + donors) <= set(self.catalog.mutation_ids)):
            raise ValueError("Targets and donors must be unique disjoint original mutation keys")
        if tuple(self.reference.region_ids) != tuple(self.catalog.region_ids):
            raise ValueError("Reference must cover the complete original regional vector in order")
        if type(self.thresholds) is not AssignmentThresholds or self.device not in ("cpu", "cuda"):
            raise ValueError("Assignment requires validated thresholds and an explicit device")
        if self.cpu_threads is not None and (type(self.cpu_threads) is not int or not 0 < self.cpu_threads < 2**31):
            raise ValueError("cpu_threads must be a positive int32")
        shape = len(targets), len(self.reference.region_ids)
        data = self.catalog.prediction_data(targets)
        selected_support = self.bounds.subset(targets, self.reference.region_ids)
        selected_support.for_data(data)
        original = self.catalog.default_bounds().subset(targets, self.reference.region_ids)
        object.__setattr__(self, "bounds", selected_support)
        object.__setattr__(self, "original_bounds", original)
        if not np.array_equal(selected_support.upper, original.upper):
            raise ValueError("Stage-three restricted support must retain the original upper endpoints")
        native_support = MultiplicityBounds(targets, self.reference.region_ids, self.native_lower,
                                            self.native_upper, "accepted_native_baseline")
        native_support.for_data(data)
        object.__setattr__(self, "native_lower", native_support.lower)
        object.__setattr__(self, "native_upper", native_support.upper)
        for name in ("native_centers", "native_calls"):
            raw = np.asarray(getattr(self, name), dtype=float)
            if not raw.size:
                raw = raw.reshape(shape)
            if raw.shape != shape:
                raise ValueError(f"{name} must align with every target-region key")
            if name == "native_centers" and (np.isinf(raw).any() or np.any(raw[np.isfinite(raw)] < 0) or np.any(raw[np.isfinite(raw)] > 1)):
                raise ValueError("Native baseline CCFs must be missing or in [0,1]")
            object.__setattr__(self, name, readonly_array(raw))
        finite = np.isfinite(self.native_calls)
        if (np.isinf(self.native_calls).any()
                or np.any(self.native_calls[finite] != np.rint(self.native_calls[finite]))
                or np.any(self.native_calls[finite] < self.native_lower[finite])
                or np.any(self.native_calls[finite] > self.native_upper[finite])):
            raise ValueError("Native baseline calls must lie in their accepted support")
        informative = np.asarray(data.count_observed, bool) & (data.total_counts > 0)
        if np.any(informative & (~np.isfinite(self.native_centers) | ~finite)):
            raise ValueError("Every informative target needs a finite accepted native baseline CCF and call")
        if np.any(finite & ~informative):
            raise ValueError("Missing/excluded native baseline multiplicities must remain missing")
        for name in ("source_stages", "source_labels"):
            values = tuple(getattr(self, name))
            if len(values) != len(targets):
                raise ValueError("Every target needs native source lineage")
            if name == "source_stages" and any(not isinstance(value, str) or not value
                    or value != value.strip() or any(char in value for char in "\t\r\n") for value in values):
                raise ValueError("Native source stages must be nonempty unpadded strings")
            if name == "source_labels" and any(type(value) is not int or value < 0 for value in values):
                raise ValueError("Native source labels must be nonnegative integers")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "target_ids", targets)
        object.__setattr__(self, "donor_ids", donors)
        object.__setattr__(self, "provenance", _freeze(self.provenance))
        object.__setattr__(self, "_content_hash", content_hash(self.to_dict()))

    def to_dict(self):
        values = {name: getattr(self, name) for name in ("target_ids", "donor_ids", "native_centers",
            "native_calls", "native_lower", "native_upper", "source_stages", "source_labels",
            "device", "cpu_threads", "provenance")}
        return _json_value(dict(schema="clipp2.assignment.bundle.v2", **values,
            catalog=self.catalog.to_dict(), reference=self.reference.to_dict(),
            bounds=self.bounds.to_dict(), thresholds=asdict(self.thresholds)))

    @property
    def content_hash(self):
        return self._content_hash

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        if values.pop("schema", None) != "clipp2.assignment.bundle.v2":
            raise ValueError("Unsupported assignment bundle schema")
        values["catalog"] = OriginalCatalog.from_dict(values["catalog"])
        values["reference"] = AssignmentReference.from_dict(values["reference"])
        values["bounds"] = MultiplicityBounds.from_dict(values["bounds"])
        values["thresholds"] = AssignmentThresholds.from_dict(values["thresholds"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class AssignmentResult:
    bundle_hash: str
    target_ids: tuple[str, ...]
    reference: AssignmentReference
    restricted: object
    original: object
    comparison: object
    multiplicities: tuple
    summary: object

    def __post_init__(self):
        targets = tuple(self.target_ids)
        if len(set(targets)) != len(targets):
            raise ValueError("Assignment result target identities must be unique")
        object.__setattr__(self, "target_ids", targets)
        for name in ("restricted", "original", "comparison", "multiplicities", "summary"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        for name in ("restricted", "original", "comparison"):
            if tuple(row.get("mutation_id") for row in getattr(self, name)) != targets:
                raise ValueError("Assignment result records must preserve exact target order")
        expected = {(m, r) for m in targets for r in self.reference.region_ids}
        keys = [(row.get("mutation_id"), row.get("region_id")) for row in self.multiplicities]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("Assignment multiplicity records must cover exactly every target-region key")


def saturated_log_mass(alt, depth):
    alt, depth = np.asarray(alt, float), np.asarray(depth, float)
    p = np.divide(alt, depth, out=np.zeros_like(alt), where=depth > 0)
    return gammaln(depth + 1) - gammaln(alt + 1) - gammaln(depth - alt + 1) + xlogy(alt, p) + xlog1py(depth - alt, -p)


POLICY_FIELDS = ("status", "best_label", "second_best_label", "accepted_label", "posterior",
    "second_best_probability", "entropy", "log_score_margin", "mean_profile_deviance",
    "outlier_count", "outlier_fraction", "informative_regions", "pair_deviance_max",
    "mean_deviance_max", "outlier_fraction_max", "mean_gate_pass", "outlier_gate_pass",
    "descriptive_fit_status", "calibrated_fit_status")
COMPARISON_FIELDS = ("comparison_status", "supports_identical", "support_sensitive_label",
    "support_sensitive_decision", "support_sensitive_probability", "probability_sensitivity_atol",
    "max_posterior_difference", "decision_change_cause", "support_stable_label")
REGIONAL_BASE_FIELDS = ("mutation_id", "region_id", "original_major_cn", "original_minor_cn",
    "original_normal_cn", "original_total_cn", "segment_id", "mapping_genuine", "count_observed",
    "positive_depth", "original_eligible", "excluded", "informative", "native_phi", "native_call",
    "native_lower", "native_upper", "restricted_lower", "restricted_upper", "original_lower",
    "original_upper", "native_call_excluded", "source_stage", "source_label")
REGIONAL_POLICY_FIELDS = ("best_label", "phi", "conditional_call", "call_probability",
    "call_log_probability", "top_two_gap", "has_second_component", "native_call_log_probability",
    "multiplicity_tie_atol", "membership_status", "multiplicity_status")


@dataclass(frozen=True, slots=True)
class _PolicyEvaluation:
    records: tuple
    scores: np.ndarray
    probabilities: np.ndarray
    mean_deviances: np.ndarray
    best_indices: np.ndarray
    best_diagnostics: dict
    observed: np.ndarray
    execution: dict


def _policy_scores(bundle, bounds, policy, progress_callback):
    observation = prepare_prediction(bundle.catalog, bundle.target_ids, bounds,
                                     device=bundle.device, cpu_threads=bundle.cpu_threads)
    n, r, k = observation.n, observation.r, len(bundle.reference.labels)
    observed = np.asarray(observation.observed, bool)
    coverage = observed.sum(axis=1)
    scores = np.full((n, k), -np.inf)
    means, fractions = np.full((n, k), np.nan), np.full((n, k), np.nan)
    outliers, passes = np.zeros((n, k), dtype=int), np.zeros((n, k), dtype=bool)
    if not n:
        return _PolicyEvaluation((), scores, scores.copy(), means, np.empty(0, dtype=int), {}, observed,
                                 dict(status="skipped_empty_targets", device=bundle.device))
    sat = saturated_log_mass(observation.alt, observation.depth)
    with NativeBackend(observation, cpu_threads=bundle.cpu_threads) as backend:
        for index, center in enumerate(bundle.reference.centers):
            diagnostic = backend.component_diagnostics(np.broadcast_to(center, (n, r)))
            if np.any(np.isin(diagnostic["status"], [2, 3])):
                raise NativeExecutionError("Assignment diagnostic execution failed")
            scores[:, index] = np.where(observed, diagnostic["log_emission"], 0.).sum(axis=1) - np.log(k)
            residual = np.maximum(0., 2 * (sat - diagnostic["max_component_log_mass"]))
            screen = summarize_profile_deviance(residual, observed, bundle.thresholds.screen, axis=1)
            means[:, index], fractions[:, index] = screen["mean_profile_deviance"], screen["outlier_fraction"]
            outliers[:, index], passes[:, index] = screen["outlier_count"], screen["descriptive_pass"]
            if progress_callback:
                progress_callback(dict(operation="assignment", policy=policy, completed_candidates=index + 1,
                                       candidate_count=k))
        compatible = np.isfinite(scores).any(axis=1) & (coverage > 0)
        probabilities = np.full_like(scores, np.nan)
        probabilities[compatible] = np.exp(scores[compatible] - logsumexp(scores[compatible], axis=1)[:, None])
        # Exact score order chooses the actual maximizer, then the lowest public
        # label. The near-tie tolerance affects acceptance, never this ordering.
        order = np.lexsort((np.broadcast_to(bundle.reference.labels, scores.shape), -scores), axis=1)
        best = order[:, 0]
        best_diagnostics = backend.component_diagnostics(bundle.reference.centers[best], bundle.native_calls)
        if np.any(np.isin(best_diagnostics["status"], [2, 3])):
            raise NativeExecutionError("Assignment multiplicity diagnostic execution failed")
        execution = dict(status="evaluated", **backend.diagnostics())
    records = []
    screen = bundle.thresholds.screen
    for i, mutation in enumerate(bundle.target_ids):
        finite, chosen = bool(compatible[i]), best[i]
        second = order[i, 1] if finite and k > 1 else None
        margin = float(scores[i, chosen] - scores[i, second]) if second is not None else float("nan")
        probability = float(probabilities[i, chosen]) if finite else float("nan")
        residual = float(means[i, chosen]) if finite else float("nan")
        fraction = float(fractions[i, chosen]) if finite else float("nan")
        fit_pass = bool(passes[i, chosen]) if finite else False
        if not observation.eligible[i].any():
            decision = "filtered"
        elif not coverage[i]:
            decision = "no_information"
        elif not finite:
            decision = "incompatible"
        elif k > 1 and margin <= bundle.thresholds.tie_atol:
            decision = "ambiguous_tie"
        elif probability < bundle.thresholds.posterior_min:
            decision = "low_confidence"
        elif not fit_pass:
            decision = "unresolved_fit"
        else:
            decision = "assigned_relative"
        records.append(dict(mutation_id=mutation, status=decision,
            best_label=bundle.reference.labels[chosen] if finite else None,
            second_best_label=bundle.reference.labels[second] if second is not None else None,
            accepted_label=bundle.reference.labels[chosen] if decision == "assigned_relative" else None,
            posterior=probability,
            second_best_probability=float(probabilities[i, second]) if second is not None else float("nan"),
            entropy=float(-np.sum(xlogy(probabilities[i], probabilities[i]))) if finite else float("nan"),
            log_score_margin=margin, mean_profile_deviance=residual,
            outlier_count=int(outliers[i, chosen]) if finite else None, outlier_fraction=fraction,
            informative_regions=int(coverage[i]), pair_deviance_max=screen.pair_deviance_max,
            mean_deviance_max=screen.mean_deviance_max, outlier_fraction_max=screen.outlier_fraction_max,
            mean_gate_pass=bool(residual <= screen.mean_deviance_max) if finite else None,
            outlier_gate_pass=bool(fraction <= screen.outlier_fraction_max) if finite else None,
            descriptive_fit_status=("pass" if fit_pass else "fail") if finite else "not_evaluated",
            calibrated_fit_status="not_calibrated", log_scores=scores[i], probabilities=probabilities[i],
            source_stage=bundle.source_stages[i], source_label=bundle.source_labels[i]))
    return _PolicyEvaluation(tuple(records), scores, probabilities, means, best, best_diagnostics, observed, execution)


def _comparison(bundle, i, left, right, identical):
    p, q = left["probabilities"], right["probabilities"]
    difference = float(np.max(np.abs(p-q))) if np.isfinite(p).all() and np.isfinite(q).all() else float("nan")
    label_changed = (left["best_label"] is not None and right["best_label"] is not None
                     and left["best_label"] != right["best_label"])
    decision_changed = (left["status"], left["accepted_label"]) != (right["status"], right["accepted_label"])
    sensitive_probability = bool(difference > bundle.thresholds.probability_sensitivity_atol) if np.isfinite(difference) else None
    accepted = left["accepted_label"]
    stable = accepted if accepted is not None and accepted == right["accepted_label"] else None
    cause = None
    if left["status"] == "incompatible" and right["status"] != "incompatible":
        status, cause = "restricted_support_incompatible", "restricted_support_incompatible"
    elif left["status"] in ("filtered", "no_information"):
        status = left["status"]
    elif left["status"] == right["status"] == "incompatible":
        status = "incompatible_both"
    elif label_changed:
        status = "support_sensitive_label"
    elif decision_changed:
        status = "support_sensitive_decision"
    elif sensitive_probability:
        status = "support_sensitive_probability"
    elif stable is not None:
        status = "support_stable_assignment"
    else:
        status = "unchanged_unresolved"
    if decision_changed and cause is None:
        changed_statuses = {left["status"], right["status"]}
        cause = ("best_label_changed" if label_changed else "cluster_score_tie_gate" if "ambiguous_tie" in changed_statuses
                 else "posterior_gate" if "low_confidence" in changed_statuses
                 else "descriptive_fit_gate" if "unresolved_fit" in changed_statuses else "assignment_status_changed")
    return dict(mutation_id=bundle.target_ids[i], comparison_status=status, supports_identical=bool(identical),
        support_sensitive_label=bool(label_changed), support_sensitive_decision=bool(decision_changed),
        support_sensitive_probability=sensitive_probability,
        probability_sensitivity_atol=bundle.thresholds.probability_sensitivity_atol,
        max_posterior_difference=difference, decision_change_cause=cause, support_stable_label=stable)


def assign_mutations(bundle, *, outdir=None, progress_callback=None):
    """Score a frozen reference under paired policies, without refitting it."""
    if type(bundle) is not AssignmentBundle:
        raise TypeError("assign_mutations requires a validated AssignmentBundle")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    support, original_bounds = bundle.bounds, bundle.original_bounds
    identical = (support.lower == original_bounds.lower).all(axis=1) & (support.upper == original_bounds.upper).all(axis=1)
    restricted = _policy_scores(bundle, support, "restricted", progress_callback)
    original = restricted if identical.all() else _policy_scores(bundle, original_bounds, "original", progress_callback)
    if np.any(np.isfinite(restricted.scores) & ~np.isfinite(original.scores)):
        raise ArithmeticError("Restricted compatibility must imply original compatibility")
    both = np.isfinite(restricted.mean_deviances) & np.isfinite(original.mean_deviances)
    if np.any(original.mean_deviances[both] > restricted.mean_deviances[both] + 1e-8):
        raise ArithmeticError("Original profile deviance exceeds restricted at the same candidate")
    comparison, multiplicities = [], []
    catalog = bundle.catalog
    original_rows = {mutation: i for i, mutation in enumerate(catalog.mutation_ids)}
    for i, mutation in enumerate(bundle.target_ids):
        comparison.append(_comparison(bundle, i, restricted.records[i], original.records[i], identical[i]))
        source = original_rows[mutation]
        for j, region in enumerate(bundle.reference.region_ids):
            informative = bool(restricted.observed[i, j])
            genuine_call = (informative and catalog.mapping_genuine[source, j]
                            and np.isfinite(bundle.native_calls[i, j]) and bundle.native_centers[i, j] > 0)
            row = dict(mutation_id=mutation, region_id=region,
                original_major_cn=float(catalog.major_cn[source, j]), original_minor_cn=float(catalog.minor_cn[source, j]),
                original_normal_cn=float(catalog.normal_cn[source, j]), original_total_cn=float(catalog.mean_total_cn[source, j]),
                segment_id=str(catalog.segment_ids[source, j]), mapping_genuine=bool(catalog.mapping_genuine[source, j]),
                count_observed=bool(catalog.count_observed[source, j]), positive_depth=bool(catalog.total_counts[source, j] > 0),
                original_eligible=bool(catalog.eligible[source, j]), excluded=not bool(catalog.eligible[source, j]),
                informative=informative, native_phi=float(bundle.native_centers[i, j]), native_call=float(bundle.native_calls[i, j]),
                native_lower=int(bundle.native_lower[i, j]), native_upper=int(bundle.native_upper[i, j]),
                restricted_lower=int(support.lower[i, j]), restricted_upper=int(support.upper[i, j]),
                original_lower=int(original_bounds.lower[i, j]), original_upper=int(original_bounds.upper[i, j]),
                native_call_excluded=bool(not support.lower[i, j] <= bundle.native_calls[i, j] <= support.upper[i, j]) if genuine_call else None,
                source_stage=bundle.source_stages[i], source_label=bundle.source_labels[i])
            for name, evaluation in (("restricted", restricted), ("original", original)):
                record, diagnostic = evaluation.records[i], evaluation.best_diagnostics
                present = informative and record["best_label"] is not None
                phi = bundle.reference.centers[evaluation.best_indices[i], j]
                usable = present and phi > 0 and diagnostic["status"][i, j] == 0
                tied = usable and diagnostic["has_second"][i, j] and diagnostic["top_two_gap"][i, j] <= bundle.thresholds.multiplicity_tie_atol
                row.update({name + "_best_label": record["best_label"], name + "_phi": float(phi) if present else float("nan"),
                    name + "_conditional_call": float(diagnostic["map_multiplicity"][i, j]) if usable else float("nan"),
                    name + "_call_probability": float(np.exp(diagnostic["map_log_probability"][i, j])) if usable else float("nan"),
                    name + "_call_log_probability": float(diagnostic["map_log_probability"][i, j]) if usable else float("nan"),
                    name + "_top_two_gap": float(diagnostic["top_two_gap"][i, j]) if usable else float("nan"),
                    name + "_has_second_component": bool(diagnostic["has_second"][i, j]) if usable else None,
                    name + "_native_call_log_probability": float(diagnostic["accepted_log_probability"][i, j]) if usable else float("nan"),
                    name + "_multiplicity_tie_atol": bundle.thresholds.multiplicity_tie_atol,
                    name + "_membership_status": record["status"], name + "_multiplicity_status":
                        "ambiguous_tie" if tied else "conditional" if usable else "zero_ccf" if present and phi == 0 else "missing_or_incompatible"})
            multiplicities.append(row)
    policies = (("restricted", restricted), ("original", original))
    summary = dict(schema="clipp2.assignment.result.v2", tumor_id=catalog.tumor_id,
        bundle_hash=bundle.content_hash, reference=bundle.reference.to_dict(), thresholds=asdict(bundle.thresholds),
        bounds_hash=support.content_hash,
        target_count=len(bundle.target_ids), region_count=len(bundle.reference.region_ids),
        candidate_count=len(bundle.reference.labels), supports_identical=bool(identical.all()),
        original_evaluation_reused=original is restricted,
        assigned_occupancy={policy: {str(label): sum(row["accepted_label"] == label for row in evaluation.records)
            for label in bundle.reference.labels} for policy, evaluation in policies},
        status_counts={policy: {status: sum(row["status"] == status for row in evaluation.records)
            for status in sorted({row["status"] for row in evaluation.records})} for policy, evaluation in policies},
        comparison_status_precedence=["restricted_support_incompatible", "filtered", "no_information",
            "incompatible_both", "support_sensitive_label", "support_sensitive_decision",
            "support_sensitive_probability", "support_stable_assignment", "unchanged_unresolved"],
        evaluation_receipt=dict(device=bundle.device, cpu_threads=bundle.cpu_threads,
            configuration=asdict(bundle.thresholds), restricted=restricted.execution,
            original=dict(status="reused_identical_support", reused_from="restricted") if original is restricted else original.execution),
        interpretation="relative_assignment_at_frozen_reference_not_a_refit",
        original_companion_scope="support_sensitivity_at_same_reference_not_uncapped_stage_two")
    result = AssignmentResult(bundle.content_hash, bundle.target_ids, bundle.reference,
                              restricted.records, original.records, comparison, tuple(multiplicities), summary)
    if outdir is not None:
        from .io.assignment import publish_assignment
        publish_assignment(bundle, result, outdir)
    return result
