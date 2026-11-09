"""Immutable fixed-reference prediction, separate from occupied fitted states."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp, xlogy, xlog1py

from .config import PROTOCOL_VERSION
from ._version import __version__
from .io.data import (ImmutableArrayRecord, MultiplicityBounds, OriginalCatalog, readonly_array, thaw_record,
                      json_value as _json_value, freeze_record as _freeze, content_hash)
from .kernel.backend import NativeBackend
from .kernel.observations import prepare_prediction
from .screening import binomial_profile_deviance, summarize_profile_deviance

SUPPORT_POLICY = "original_full_1_to_major_cn"
EVIDENCE_STATUSES = ("observed", "prior_only", "model_incompatible", "excluded")


def _integer_tuple(values, name):
    values = tuple(values)
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in values):
        raise ValueError(f"{name} requires exact integers")
    return values


@dataclass(frozen=True, slots=True)
class AssignmentReference(ImmutableArrayRecord):
    labels: tuple[int, ...]
    region_ids: tuple[str, ...]
    centers: np.ndarray
    lineage: tuple = ()
    clonal_label: int | None = 0
    source_informative_count: np.ndarray | None = field(default=None, kw_only=True)
    source_informative_depth: np.ndarray | None = field(default=None, kw_only=True)
    source_mean_profile_deviance: np.ndarray | None = field(default=None, kw_only=True)
    source_maximum_profile_deviance: np.ndarray | None = field(default=None, kw_only=True)
    diagnostic_availability: object = field(default=None, kw_only=True)
    center_sources: tuple = field(kw_only=True)

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
        for name in ("source_informative_count", "source_informative_depth",
                     "source_mean_profile_deviance", "source_maximum_profile_deviance"):
            raw = getattr(self, name)
            values = np.full(centers.shape, np.nan) if raw is None else np.asarray(raw, float)
            if values.shape != centers.shape or np.any(values < 0):
                raise ValueError("Reference source measurements must align and be nonnegative or missing")
            if name in ("source_informative_count", "source_informative_depth") and np.isinf(values).any():
                raise ValueError("Source observation counts/depths must be finite or missing")
            if name == "source_informative_count" and np.any(values[np.isfinite(values)] != np.rint(values[np.isfinite(values)])):
                raise ValueError("Source informative counts must be integer or missing")
            object.__setattr__(self, name, readonly_array(values))
        availability = (np.full(centers.shape, "missing") if self.diagnostic_availability is None
                        else np.asarray(self.diagnostic_availability))
        if (availability.shape != centers.shape
                or not np.isin(availability, ("available", "missing", "failed")).all()):
            raise ValueError("Reference diagnostic availability must be available, missing or failed per pair")
        available = availability == "available"
        count, depth = self.source_informative_count, self.source_informative_depth
        mean, maximum = self.source_mean_profile_deviance, self.source_maximum_profile_deviance
        if (np.any(available & (np.isnan(count) | np.isnan(depth)))
                or np.any(available & (count > 0) & (np.isnan(mean) | np.isnan(maximum)))
                or np.any(available & (count == 0) & (~np.isnan(mean) | ~np.isnan(maximum)))
                or np.any(np.isfinite(count) & np.isfinite(depth) & (depth < count))
                or np.any(~np.isnan(mean) & ~np.isnan(maximum) & (mean > maximum))):
            raise ValueError("Available source diagnostics must preserve counts and empty-domain measurements")
        sources = tuple(self.center_sources)
        if len(sources) != len(labels) or any(not isinstance(source, Mapping) or not source for source in sources):
            raise ValueError("Reference center-source metadata must align with every candidate")
        object.__setattr__(self, "diagnostic_availability", _freeze(availability.tolist()))
        object.__setattr__(self, "center_sources", _freeze(sources))

    @property
    def priors(self):
        return readonly_array(np.full(len(self.labels), 1 / len(self.labels)))

    def to_dict(self):
        return _json_value(dict(labels=self.labels, region_ids=self.region_ids, centers=self.centers,
            priors=self.priors, lineage=self.lineage, clonal_label=self.clonal_label,
            source_informative_count=self.source_informative_count,
            source_informative_depth=self.source_informative_depth,
            source_mean_profile_deviance=self.source_mean_profile_deviance,
            source_maximum_profile_deviance=self.source_maximum_profile_deviance,
            diagnostic_availability=self.diagnostic_availability,
            center_sources=self.center_sources))

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if not {"source_informative_count", "source_informative_depth", "source_mean_profile_deviance",
                "source_maximum_profile_deviance", "diagnostic_availability", "center_sources"} <= set(value):
            raise ValueError("Protocol 2.3 references require source measurements and provenance")
        priors = np.asarray(value.pop("priors"), float)
        result = cls(**value)
        if not np.array_equal(priors, result.priors):
            raise ValueError("Assignment reference must have fixed equal priors")
        return result


@dataclass(frozen=True, slots=True)
class AssignmentBundle(ImmutableArrayRecord):
    catalog: OriginalCatalog
    reference: AssignmentReference
    target_ids: tuple[str, ...]
    non_target_ids: tuple[str, ...]
    native_centers: np.ndarray
    native_calls: np.ndarray
    native_lower: np.ndarray
    native_upper: np.ndarray
    source_stages: tuple[str, ...]
    source_labels: tuple[int, ...]
    baseline_labels: tuple[int, ...]
    device: str = "cpu"
    cpu_threads: int | None = None
    normal_cn_policy: str | None = None
    provenance: object = field(default_factory=dict)
    run_manifest_id: str | None = field(default=None, kw_only=True)
    bounds: MultiplicityBounds = field(init=False, repr=False)
    _content_hash: str = field(init=False, repr=False)

    def __post_init__(self):
        from .provenance import validate_run_manifest_id
        validate_run_manifest_id(self.run_manifest_id)
        targets, others = tuple(self.target_ids), tuple(self.non_target_ids)
        if type(self.catalog) is not OriginalCatalog or type(self.reference) is not AssignmentReference:
            raise TypeError("Assignment requires a validated original catalog and frozen reference")
        from .io.data import validate_workflow_mapping
        validate_workflow_mapping(self.catalog, targets + others, normal_cn_policy=self.normal_cn_policy)
        if (len(set(targets)) != len(targets) or len(set(others)) != len(others)
                or set(targets) & set(others) or not set(targets + others) <= set(self.catalog.mutation_ids)):
            raise ValueError("Targets and unchanged non-targets must be unique disjoint original mutation keys")
        if tuple(self.reference.region_ids) != tuple(self.catalog.region_ids):
            raise ValueError("Reference must cover the complete original regional vector in order")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("Assignment requires an explicit supported device")
        if self.cpu_threads is not None and (type(self.cpu_threads) is not int or not 0 < self.cpu_threads < 2**31):
            raise ValueError("cpu_threads must be a positive int32")
        shape = len(targets), len(self.reference.region_ids)
        data = self.catalog.prediction_data(targets, normal_cn_policy=self.normal_cn_policy)
        support = self.catalog.default_bounds().subset(targets, self.reference.region_ids)
        support.for_data(data)
        positions = {mutation: i for i, mutation in enumerate(self.catalog.mutation_ids)}
        rows = [positions[mutation] for mutation in targets]
        eligible = self.catalog.eligible[rows]
        if np.any(eligible & ((support.lower != 1) | (support.upper != self.catalog.major_cn[rows]))):
            raise ValueError("Stage three requires original full single-state support on every eligible target pair")
        object.__setattr__(self, "bounds", support)
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
        for name in ("source_stages", "source_labels", "baseline_labels"):
            values = tuple(getattr(self, name))
            if len(values) != len(targets):
                raise ValueError("Every target needs native source lineage and a pre-assignment baseline label")
            if name == "source_stages" and any(not isinstance(value, str) or not value
                    or value != value.strip() or any(char in value for char in "\t\r\n") for value in values):
                raise ValueError("Native source stages must be nonempty unpadded strings")
            if name != "source_stages" and any(type(value) is not int or value < 0 for value in values):
                raise ValueError("Native source and baseline labels must be nonnegative integers")
            if name == "baseline_labels" and not set(values) <= set(self.reference.labels):
                raise ValueError("Pre-assignment baseline labels must belong to the frozen reference")
            object.__setattr__(self, name, values)
        for stage, baseline in zip(self.source_stages, self.baseline_labels):
            assignment_outcome(stage, baseline, None, self.reference.clonal_label)
        object.__setattr__(self, "target_ids", targets)
        object.__setattr__(self, "non_target_ids", others)
        object.__setattr__(self, "provenance", _freeze(self.provenance))
        object.__setattr__(self, "_content_hash", content_hash(self.to_dict()))

    def to_dict(self):
        values = {name: getattr(self, name) for name in ("target_ids", "non_target_ids", "native_centers",
            "native_calls", "native_lower", "native_upper", "source_stages", "source_labels", "baseline_labels",
            "device", "cpu_threads", "normal_cn_policy", "provenance", "run_manifest_id")}
        return _json_value(dict(schema="clipp2.assignment.bundle.v4", protocol_version=PROTOCOL_VERSION,
            support_policy=SUPPORT_POLICY, **values,
            catalog=self.catalog.to_dict(), reference=self.reference.to_dict(),
            bounds=self.bounds.to_dict()))

    @property
    def content_hash(self):
        return self._content_hash

    def validated_payload(self):
        """Validate live values, including nested payloads, at execution boundaries."""
        payload = self.to_dict()
        if content_hash(payload) != self._content_hash:
            raise ValueError("Assignment bundle live content differs from its frozen identity")
        restored = type(self).from_dict(payload)
        if restored.content_hash != self._content_hash:
            raise ValueError("Assignment bundle no longer has its canonical validated identity")
        return payload

    def validate_integrity(self):
        self.validated_payload()

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        if (values.pop("schema", None) != "clipp2.assignment.bundle.v4"
                or values.pop("protocol_version", None) != PROTOCOL_VERSION
                or values.pop("support_policy", None) != SUPPORT_POLICY):
            raise ValueError("Assignment requires a protocol 2.3 bundle with schema v4")
        bounds = MultiplicityBounds.from_dict(values.pop("bounds"))
        values["catalog"] = OriginalCatalog.from_dict(values["catalog"])
        values["reference"] = AssignmentReference.from_dict(values["reference"])
        result = cls(**values)
        if bounds.to_dict() != result.bounds.to_dict():
            raise ValueError("Stage-three support must be freshly derived original full support")
        return result


@dataclass(frozen=True, slots=True)
class AssignmentResult(ImmutableArrayRecord):
    bundle_hash: str
    target_ids: tuple[str, ...]
    reference: AssignmentReference
    assignments: tuple
    multiplicities: tuple
    summary: object
    run_manifest_id: str | None = field(default=None, kw_only=True)

    def __post_init__(self):
        from .provenance import validate_run_manifest_id
        validate_run_manifest_id(self.run_manifest_id)
        targets = tuple(self.target_ids)
        if type(self.reference) is not AssignmentReference or len(set(targets)) != len(targets):
            raise ValueError("Assignment result needs unique target identities and a validated reference")
        object.__setattr__(self, "target_ids", targets)
        records = []
        for row in self.assignments:
            row = dict(row)
            for name in ("best_label", "second_best_label", "map_cluster_label", "baseline_label",
                         "baseline_reference_label", "final_assignment_label", "source_label"):
                value = row.get(name)
                if value is not None:
                    _integer_tuple((value,), name)
            for key in ("probabilities", "log_scores"):
                values = np.asarray(row[key], dtype=float)
                if values.shape != (len(self.reference.labels),):
                    raise ValueError("Assignment probability/score vectors must cover the complete reference")
                row[key] = readonly_array(values)
            if _integer_tuple(row.get("reference_labels", ()), "reference_labels") != self.reference.labels:
                raise ValueError("Assignment probability labels must preserve exact reference order")
            probabilities, scores = row["probabilities"], row["log_scores"]
            if np.isnan(scores).any() or np.isposinf(scores).any():
                raise ValueError("Assignment log scores must be finite or negative infinity")
            if np.isfinite(scores).any():
                if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                        or not np.isclose(probabilities.sum(), 1., rtol=0, atol=1e-12)):
                    raise ValueError("Defined assignment probabilities must be finite, nonnegative and normalized")
            elif not np.isnan(probabilities).all():
                raise ValueError("Uninformative/incompatible assignment probabilities must be missing")
            ties = _integer_tuple(row.get("map_tied_labels", ()), "map_tied_labels")
            expected_ties = (tuple(sorted(label for label, score in zip(self.reference.labels, scores)
                                          if score == np.max(scores))) if np.isfinite(scores).any() else ())
            if ties != expected_ties or row.get("map_cluster_label") != (ties[0] if ties else None):
                raise ValueError("Exact maximizing ties and their smallest public representative must agree")
            unique = row.get("map_unique")
            if (ties and (not isinstance(unique, (bool, np.bool_)) or unique != (len(ties) == 1))
                    or not ties and unique is not None):
                raise ValueError("MAP uniqueness must describe the exact maximizing set")
            records.append(row)
        object.__setattr__(self, "assignments", _freeze(records))
        for row in self.multiplicities:
            candidates = _integer_tuple(row.get("multiplicity_candidates", ()), "multiplicity_candidates")
            ties = _integer_tuple(row.get("map_tied_multiplicities", ()), "map_tied_multiplicities")
            if (candidates != tuple(sorted(set(candidates))) or ties != tuple(sorted(set(ties)))
                    or not set(ties) <= set(candidates)):
                raise ValueError("Conditional candidates and exact tie sets must retain integer order")
        object.__setattr__(self, "multiplicities", _freeze(self.multiplicities))
        object.__setattr__(self, "summary", _freeze(self.summary))
        if tuple(row.get("mutation_id") for row in self.assignments) != targets:
            raise ValueError("Assignment records must preserve exact target order")
        expected = {(m, r) for m in targets for r in self.reference.region_ids}
        keys = [(row.get("mutation_id"), row.get("region_id")) for row in self.multiplicities]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("Assignment multiplicity records must cover exactly every target-region key")

    def to_dict(self):
        # Keep numeric NaN/inf in this plain copy payload; JSON publishers apply
        # the shared scientific-value codec at the actual serialization boundary.
        return thaw_record(dict(schema="clipp2.assignment.result.v5", protocol_version=PROTOCOL_VERSION,
            support_policy=SUPPORT_POLICY,
            bundle_hash=self.bundle_hash, target_ids=self.target_ids, reference=self.reference.to_dict(),
            assignments=self.assignments, multiplicities=self.multiplicities, summary=self.summary,
            run_manifest_id=self.run_manifest_id), arrays_as_lists=True)

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        if (values.pop("schema", None) != "clipp2.assignment.result.v5"
                or values.pop("protocol_version", None) != PROTOCOL_VERSION
                or values.pop("support_policy", None) != SUPPORT_POLICY):
            raise ValueError("Assignment requires a protocol 2.3 result with schema v5")
        values["reference"] = AssignmentReference.from_dict(values["reference"])
        return cls(**values)


POLICY_FIELDS = ("status", "evidence_status", "best_label", "second_best_label", "posterior",
    "map_cluster_label", "map_tied_labels", "map_unique", "best_log_probability", "second_best_probability",
    "entropy", "log_score_margin", "log_competitor_ratio", "comparison_scope", "has_finite_alternative",
    "mean_profile_deviance", "maximum_profile_deviance", "diagnostic_status", "diagnostic_error",
    "informative_regions", "baseline_label", "reassessment_status", "baseline_changed", "reference_labels",
    "baseline_reference_label", "final_assignment_label", "target_reason")
REGIONAL_BASE_FIELDS = ("mutation_id", "region_id", "alt_count", "ref_count", "total_depth", "purity",
    "original_major_cn", "original_minor_cn", "original_normal_cn", "original_total_cn", "normal_cn_supplied",
    "segment_id", "mapping_genuine", "count_observed", "positive_depth", "original_eligible", "original_input_excluded", "excluded",
    "informative", "native_phi", "native_call", "native_lower", "native_upper", "original_lower",
    "original_upper", "source_stage", "source_label")
REGIONAL_POLICY_FIELDS = ("best_label", "phi", "conditional_map", "conditional_call", "call_probability",
    "call_log_probability", "top_two_gap", "has_second_component", "native_call_log_probability",
    "conditional_distribution_defined", "multiplicity_tie", "map_tied_multiplicities", "map_unique",
    "structural_singleton", "multiplicity_read_informative", "membership_status", "multiplicity_status",
    "multiplicity_candidates", "conditional_probabilities", "profile_deviance", "assigned_multiplicity",
    "diagnostic_status", "diagnostic_error", "center_source")


def target_reason(source_stage):
    return {"stage_one": "stage1_reference_recheck", "stage1": "stage1_reference_recheck",
            "stage_two": "stage2_near_clonal_recheck", "stage2": "stage2_near_clonal_recheck"}.get(
                source_stage, "fixed_reference_recheck")


REASSESSMENT_OUTCOMES = ("retained_clonal", "reassigned", "unresolved", "not_applicable")


def assignment_outcome(source_stage, baseline_label, final_label, clonal_label):
    """Separate workflow clonal rechecks from generic baseline comparisons."""
    if target_reason(source_stage) == "fixed_reference_recheck":
        status = "not_applicable"
    else:
        if clonal_label is None or baseline_label != clonal_label:
            raise ValueError("Workflow reassessment requires a designated clonal reference and clonal baseline")
        status = ("unresolved" if final_label is None else "retained_clonal"
                  if final_label == clonal_label else "reassigned")
    return dict(reassessment_status=status,
                baseline_changed=None if final_label is None else final_label != baseline_label)


def conditional_multiplicity(catalog, source, region, phi, lower, upper):
    """Exact MAP of computed conditional log masses, ties resolved by integer order.

    The binomial coefficient and uniform multiplicity prior cancel in this
    conditional distribution. Retained native calls are separate provenance.
    """
    if (not np.isfinite(phi) or not 0 <= phi <= 1 or lower != int(lower) or upper != int(upper)
            or not 1 <= lower <= upper <= catalog.major_cn[source, region]):
        raise ValueError("Conditional multiplicity needs a valid CCF and original-support interval")
    candidates = np.arange(lower, upper + 1, dtype=np.int64)
    purity = catalog.purity[source, region]
    total_cn = catalog.major_cn[source, region] + catalog.minor_cn[source, region]
    denominator = (1-purity)*catalog.normal_cn[source, region] + purity*total_cn
    p = np.minimum((purity/denominator*candidates)*phi, 1.)
    alt, depth = catalog.alt_counts[source, region], catalog.total_counts[source, region]
    logs = xlogy(alt, p) + xlog1py(depth-alt, -p)
    if np.isnan(logs).any() or np.isposinf(logs).any():
        raise ArithmeticError("Conditional multiplicity log masses are invalid")
    if not np.isfinite(logs).any():
        return dict(defined=False, candidates=candidates, log_masses=logs,
                    probabilities=np.full(logs.shape, np.nan), map_call=np.nan,
                    map_tied_values=(), map_unique=None, map_probability=np.nan,
                    map_log_probability=np.nan, map_score_gap=np.nan, profile_deviance=np.inf,
                    diagnostic_status="available", diagnostic_error=None)
    best = int(np.argmax(logs))
    others = np.delete(logs, best)
    total = np.exp(others-logs[best]).sum()
    log_probability = float(-np.log1p(total))
    probabilities = np.exp(logs-logs[best])/(1.+total)
    ties = tuple(int(value) for value in candidates[logs == logs[best]])
    deviance, error = np.nan, None
    try:
        deviance = float(binomial_profile_deviance(alt, depth, p[best]))
        if np.isnan(deviance) or deviance < 0:
            raise ArithmeticError("Profile deviance must be nonnegative and nonmissing")
    except Exception as failure:
        deviance = np.nan
        error = f"{type(failure).__name__}: {failure}"
    return dict(defined=True, candidates=candidates, log_masses=logs, probabilities=probabilities,
        map_call=int(candidates[best]), map_tied_values=ties, map_unique=len(ties) == 1,
        map_probability=float(np.exp(log_probability)), map_log_probability=log_probability,
        map_score_gap=float(logs[best]-np.max(others)) if len(others) else np.nan,
        profile_deviance=deviance, diagnostic_status="failed" if error else "available", diagnostic_error=error)


def score_summary(scores, labels):
    """Normalize fixed-reference scores and select the exact MAP, then lowest label."""
    scores, labels = np.asarray(scores, dtype=float), tuple(labels)
    if (scores.shape != (len(labels),) or not labels or np.isnan(scores).any() or np.isposinf(scores).any()
            or len(set(labels)) != len(labels) or any(type(label) is not int or label < 0 for label in labels)):
        raise ValueError("Assignment scores need one finite or negative-infinite value per distinct reference label")
    order = np.lexsort((labels, -scores))
    best = int(order[0]) if np.isfinite(scores).any() else None
    second = int(order[1]) if best is not None and len(labels) > 1 else None
    probabilities = np.full(scores.shape, np.nan)
    finite_alternative = None
    probability = log_probability = second_probability = margin = entropy = competitor_ratio = float("nan")
    ties, unique = (), None
    if best is not None:
        alternatives = np.delete(scores, best)
        finite_alternative = bool(np.isfinite(alternatives).any())
        competitor_ratio = float(logsumexp(alternatives - scores[best])) if alternatives.size else -np.inf
        ratios = np.exp(scores - scores[best])
        denominator = 1. + np.sum(np.delete(ratios, best))
        log_probability = float(-np.log1p(np.sum(np.delete(ratios, best))))
        probabilities = ratios / denominator
        probability = float(probabilities[best])
        if second is not None:
            second_probability = float(probabilities[second])
            margin = float(scores[best] - scores[second])
        entropy = float(-np.sum(xlogy(probabilities, probabilities)))
        ties = tuple(sorted(labels[index] for index in np.flatnonzero(scores == scores[best])))
        unique = len(ties) == 1
    return dict(best_index=best, second_index=second, best_label=labels[best] if best is not None else None,
        second_best_label=labels[second] if second is not None else None, probabilities=probabilities,
        map_cluster_label=labels[best] if best is not None else None, map_tied_labels=ties, map_unique=unique,
        posterior=probability, best_log_probability=log_probability, second_best_probability=second_probability,
        entropy=entropy, log_score_margin=margin, has_finite_alternative=finite_alternative,
        log_competitor_ratio=competitor_ratio,
        comparison_scope="single_candidate" if len(labels) == 1 else "multiple_candidates")


def assignment_decision(summary, eligible, informative_regions):
    """Distinguish data-supported estimates, priors, and model incompatibility."""
    if not eligible:
        return "excluded"
    if not informative_regions:
        return "prior_only"
    return "model_incompatible" if summary["best_label"] is None else "observed"


@dataclass(frozen=True, slots=True)
class _ReferenceEvaluation:
    records: tuple
    best_indices: np.ndarray
    best_diagnostics: dict
    observed: np.ndarray
    execution: dict


def _score_reference(bundle, progress_callback):
    observation = prepare_prediction(bundle.catalog, bundle.target_ids, bundle.bounds,
                                     device=bundle.device, cpu_threads=bundle.cpu_threads, normal_cn_policy=bundle.normal_cn_policy)
    n, r, k = observation.n, observation.r, len(bundle.reference.labels)
    observed = np.asarray(observation.observed, bool)
    coverage = observed.sum(axis=1)
    scores = np.full((n, k), -np.inf)
    if not n:
        return _ReferenceEvaluation((), np.empty(0, dtype=int), {}, observed,
                           dict(status="skipped_empty_targets", device=bundle.device))
    with NativeBackend(observation, cpu_threads=bundle.cpu_threads) as backend:
        for index, center in enumerate(bundle.reference.centers):
            loss = backend.node_terms(np.broadcast_to(center, (n, r)), derivatives=False)
            scores[:, index] = np.where(observed, -loss, 0.).sum(axis=1) - np.log(k)
            scores[~observation.eligible.any(axis=1), index] = -np.inf
            if progress_callback:
                progress_callback(dict(operation="assignment", protocol_version=PROTOCOL_VERSION,
                                       completed_candidates=index + 1, candidate_count=k))
        summaries = [score_summary(row, bundle.reference.labels) for row in scores]
        best = np.array([row["best_index"] if row["best_index"] is not None else 0 for row in summaries], dtype=int)
        diagnostic_error = None
        try:
            diagnostics = backend.component_diagnostics(bundle.reference.centers[best], bundle.native_calls)
        except Exception as error:
            diagnostics = {"status": np.where(observed, 3, 4)}
            diagnostic_error = f"{type(error).__name__}: {error}"
        execution = dict(status="evaluated", **backend.diagnostics())
    diagnostics["error"] = diagnostic_error
    records = []
    for i, mutation in enumerate(bundle.target_ids):
        summary = summaries[i]
        status = assignment_decision(summary, bool(observation.eligible[i].any()), int(coverage[i]))
        final = summary["map_cluster_label"] if status == "observed" else None
        records.append(dict(mutation_id=mutation, status=status, evidence_status=status,
            **{key: value for key, value in summary.items() if key not in ("best_index", "second_index")},
            informative_regions=int(coverage[i]), log_scores=scores[i], reference_labels=bundle.reference.labels,
            source_stage=bundle.source_stages[i], source_label=bundle.source_labels[i],
            baseline_label=bundle.baseline_labels[i], baseline_reference_label=bundle.baseline_labels[i],
            final_assignment_label=final, target_reason=target_reason(bundle.source_stages[i]),
            **assignment_outcome(bundle.source_stages[i], bundle.baseline_labels[i], final,
                                 bundle.reference.clonal_label)))
    return _ReferenceEvaluation(tuple(records), best, diagnostics, observed, execution)


def assign_mutations(bundle, *, outdir=None, progress_callback=None, run_manifest_id=None):
    """Return MAP estimates and diagnostic measurements at a frozen reference."""
    if type(bundle) is not AssignmentBundle:
        raise TypeError("assign_mutations requires a validated AssignmentBundle")
    bundle.validated_payload()
    from .provenance import resolve_current_run_manifest_id
    execution_manifest_id = resolve_current_run_manifest_id(run_manifest_id)
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    evaluation = _score_reference(bundle, progress_callback)
    multiplicities = []
    catalog, support = bundle.catalog, bundle.bounds
    original_rows = {mutation: i for i, mutation in enumerate(catalog.mutation_ids)}
    for i, mutation in enumerate(bundle.target_ids):
        source = original_rows[mutation]
        record, diagnostic = evaluation.records[i], evaluation.best_diagnostics
        residuals, errors = [], []
        for j, region in enumerate(bundle.reference.region_ids):
            informative = bool(evaluation.observed[i, j])
            present = informative and record["best_label"] is not None
            phi = float(bundle.reference.centers[evaluation.best_indices[i], j]) if present else float("nan")
            error = None
            if present and diagnostic["status"][i, j] in (2, 3):
                error = diagnostic["error"] or f"Native component diagnostic status {diagnostic['status'][i, j]}"
            distribution = None
            if present and error is None:
                try:
                    distribution = conditional_multiplicity(catalog, source, j, phi,
                        int(support.lower[i, j]), int(support.upper[i, j]))
                    error = distribution["diagnostic_error"]
                except Exception as failure:
                    error = f"{type(failure).__name__}: {failure}"
            defined = bool(distribution is not None and distribution["defined"])
            singleton = bool(support.lower[i, j] == support.upper[i, j])
            call = distribution["map_call"] if defined else np.nan
            deviance = (distribution["profile_deviance"] if defined else
                        np.nan if error or not present else np.inf)
            diagnostic_status = "failed" if error else "available" if present else "not_applicable"
            if error:
                errors.append(f"{region}: {error}")
            residuals.append(deviance)
            if not catalog.eligible[source, j]:
                multiplicity_status = "excluded"
            elif not informative:
                multiplicity_status = "missing_observation"
            elif error and not defined:
                multiplicity_status = "diagnostic_failed"
            elif not defined:
                multiplicity_status = "model_incompatible"
            elif phi == 0:
                multiplicity_status = "uninformative_zero_ccf"
            elif singleton:
                multiplicity_status = "structural_singleton"
            else:
                multiplicity_status = "conditional"
            native_lp = np.nan
            if defined:
                index = int(bundle.native_calls[i, j])-int(support.lower[i, j])
                logs = distribution["log_masses"]
                native_lp = (float(logs[index]-np.max(logs)+distribution["map_log_probability"])
                             if 0 <= index < len(logs) else -np.inf)
            multiplicities.append(dict(mutation_id=mutation, region_id=region,
                alt_count=float(catalog.alt_counts[source, j]),
                ref_count=float(catalog.total_counts[source, j] - catalog.alt_counts[source, j]),
                total_depth=float(catalog.total_counts[source, j]), purity=float(catalog.purity[source, j]),
                original_major_cn=float(catalog.major_cn[source, j]), original_minor_cn=float(catalog.minor_cn[source, j]),
                original_normal_cn=float(catalog.normal_cn[source, j]),
                original_total_cn=float(catalog.major_cn[source, j] + catalog.minor_cn[source, j]),
                normal_cn_supplied=bool(catalog.normal_cn_supplied[source, j]),
                segment_id=str(catalog.segment_ids[source, j]), mapping_genuine=bool(catalog.mapping_genuine[source, j]),
                count_observed=bool(catalog.count_observed[source, j]), positive_depth=bool(catalog.total_counts[source, j] > 0),
                original_eligible=bool(catalog.eligible[source, j]), excluded=not bool(catalog.eligible[source, j]),
                original_input_excluded=bool(catalog.input_excluded[source, j]),
                informative=informative, native_phi=float(bundle.native_centers[i, j]), native_call=float(bundle.native_calls[i, j]),
                native_lower=int(bundle.native_lower[i, j]), native_upper=int(bundle.native_upper[i, j]),
                original_lower=int(support.lower[i, j]), original_upper=int(support.upper[i, j]),
                source_stage=bundle.source_stages[i], source_label=bundle.source_labels[i],
                best_label=record["best_label"], phi=phi, conditional_map=call,
                conditional_call=call if defined and phi > 0 else np.nan,
                assigned_multiplicity=call if defined and phi > 0 and record["final_assignment_label"] is not None else np.nan,
                center_source=(bundle.reference.center_sources[evaluation.best_indices[i]]
                               if record["best_label"] is not None else None),
                call_probability=distribution["map_probability"] if defined else np.nan,
                call_log_probability=distribution["map_log_probability"] if defined else np.nan,
                top_two_gap=distribution["map_score_gap"] if defined else np.nan,
                has_second_component=not singleton if defined else None,
                native_call_log_probability=native_lp, conditional_distribution_defined=defined,
                multiplicity_tie=not distribution["map_unique"] if defined else None,
                map_tied_multiplicities=distribution["map_tied_values"] if defined else (),
                map_unique=distribution["map_unique"] if defined else None,
                structural_singleton=singleton, multiplicity_read_informative=bool(defined and phi > 0),
                membership_status=record["status"], multiplicity_status=multiplicity_status,
                multiplicity_candidates=distribution["candidates"] if defined else (),
                conditional_probabilities=distribution["probabilities"] if defined else (),
                profile_deviance=deviance, diagnostic_status=diagnostic_status, diagnostic_error=error))
        diagnostic_mask = evaluation.observed[i] & (record["best_label"] is not None)
        measured = (summarize_profile_deviance(residuals, diagnostic_mask, axis=0) if not errors else
                    dict(mean_profile_deviance=np.nan, maximum_profile_deviance=np.nan))
        record.update(mean_profile_deviance=float(measured["mean_profile_deviance"]),
            maximum_profile_deviance=float(measured["maximum_profile_deviance"]),
            diagnostic_status="failed" if errors else "available" if diagnostic_mask.any() else "not_applicable",
            diagnostic_error="; ".join(errors) if errors else None)
    summary = dict(schema="clipp2.assignment.result.v5", protocol_version=PROTOCOL_VERSION,
        implementation_version=__version__, tumor_id=catalog.tumor_id,
        bundle_hash=bundle.content_hash, reference=bundle.reference.to_dict(),
        bounds_hash=support.content_hash, support_policy=SUPPORT_POLICY, normal_cn_policy=bundle.normal_cn_policy,
        run_manifest_id=execution_manifest_id, bundle_run_manifest_id=bundle.run_manifest_id,
        target_count=len(bundle.target_ids), unchanged_non_target_count=len(bundle.non_target_ids),
        region_count=len(bundle.reference.region_ids), candidate_count=len(bundle.reference.labels),
        assigned_occupancy={str(label): sum(row["final_assignment_label"] == label for row in evaluation.records)
                            for label in bundle.reference.labels},
        status_counts={status: sum(row["status"] == status for row in evaluation.records)
                       for status in EVIDENCE_STATUSES},
        reassessment_counts={status: sum(row["reassessment_status"] == status for row in evaluation.records)
                             for status in REASSESSMENT_OUTCOMES},
        evaluation_receipt=dict(device=bundle.device, cpu_threads=bundle.cpu_threads,
            run_manifest_id=execution_manifest_id, bundle_run_manifest_id=bundle.run_manifest_id,
            assignment_policy="observed_exact_map_lowest_public_label", execution=evaluation.execution),
        population_scope="targets_only",
        comparison_scope="single_candidate" if len(bundle.reference.labels) == 1 else "multiple_candidates",
        interpretation="relative_map_at_frozen_reference_not_a_refit")
    bundle.validated_payload()
    result = AssignmentResult(bundle.content_hash, bundle.target_ids, bundle.reference,
                              evaluation.records, tuple(multiplicities), summary, run_manifest_id=execution_manifest_id)
    if outdir is not None:
        from .io.assignment import publish_assignment
        publish_assignment(bundle, result, outdir)
    return result
