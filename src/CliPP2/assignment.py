"""Immutable fixed-reference prediction, separate from occupied fitted states."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy, xlog1py

from .config import AssignmentThresholds, PROTOCOL_VERSION
from ._version import __version__
from .io.data import (ImmutableArrayRecord, MultiplicityBounds, OriginalCatalog, readonly_array, thaw_record,
                      json_value as _json_value, freeze_record as _freeze, content_hash)
from .kernel.backend import NativeBackend, NativeExecutionError
from .kernel.observations import prepare_prediction
from .screening import summarize_profile_deviance

SUPPORT_POLICY = "original_full_1_to_major_cn"


@dataclass(frozen=True, slots=True)
class AssignmentReference(ImmutableArrayRecord):
    labels: tuple[int, ...]
    region_ids: tuple[str, ...]
    centers: np.ndarray
    lineage: tuple = ()
    clonal_label: int | None = 0
    regional_quality: np.ndarray = field(kw_only=True)
    regional_quality_reasons: tuple = field(kw_only=True)
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
        if self.regional_quality is None or self.regional_quality_reasons is None or self.center_sources is None:
            raise ValueError("Reference quality, reasons, and center-source metadata must be explicitly supplied")
        quality = np.asarray(self.regional_quality)
        if quality.shape != centers.shape or quality.dtype.kind != "b":
            raise ValueError("Reference quality must be a boolean flag for every candidate-region pair")
        reasons = self.regional_quality_reasons
        if len(reasons) != len(labels) or any(len(row) != len(regions) for row in reasons):
            raise ValueError("Reference quality reasons must align with every candidate-region pair")
        normalized = []
        for i, row in enumerate(reasons):
            values = []
            for j, reasons_for_pair in enumerate(row):
                if isinstance(reasons_for_pair, str):
                    raise ValueError("Reference quality reasons must be collections of reason strings")
                items = tuple(reasons_for_pair)
                if (any(not isinstance(reason, str) or not reason or reason != reason.strip()
                        or any(char in reason for char in "\t\r\n") for reason in items)
                        or len(set(items)) != len(items) or (not quality[i, j] and not items)):
                    raise ValueError("Every failed reference pair requires explicit unique quality reasons")
                values.append(items)
            normalized.append(tuple(values))
        sources = tuple(self.center_sources)
        if len(sources) != len(labels) or any(not isinstance(source, Mapping) or not source for source in sources):
            raise ValueError("Reference center-source metadata must align with every candidate")
        object.__setattr__(self, "regional_quality", readonly_array(quality))
        object.__setattr__(self, "regional_quality_reasons", tuple(normalized))
        object.__setattr__(self, "center_sources", _freeze(sources))

    @property
    def priors(self):
        return readonly_array(np.full(len(self.labels), 1 / len(self.labels)))

    def to_dict(self):
        return _json_value(dict(labels=self.labels, region_ids=self.region_ids, centers=self.centers,
            priors=self.priors, lineage=self.lineage, clonal_label=self.clonal_label,
            regional_quality=self.regional_quality, regional_quality_reasons=self.regional_quality_reasons,
            center_sources=self.center_sources))

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if not {"regional_quality", "regional_quality_reasons", "center_sources"} <= set(value):
            raise ValueError("Protocol 2.2 references require explicit frozen quality and center-source metadata")
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
    thresholds: AssignmentThresholds
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
        if type(self.thresholds) is not AssignmentThresholds or self.device not in ("cpu", "cuda"):
            raise ValueError("Assignment requires validated thresholds and an explicit device")
        if self.cpu_threads is not None and (type(self.cpu_threads) is not int or not 0 < self.cpu_threads < 2**31):
            raise ValueError("cpu_threads must be a positive int32")
        shape = len(targets), len(self.reference.region_ids)
        data = self.catalog.prediction_data(targets)
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
        return _json_value(dict(schema="clipp2.assignment.bundle.v3", protocol_version=PROTOCOL_VERSION,
            support_policy=SUPPORT_POLICY, **values,
            catalog=self.catalog.to_dict(), reference=self.reference.to_dict(),
            bounds=self.bounds.to_dict(), thresholds=asdict(self.thresholds)))

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
        if (values.pop("schema", None) != "clipp2.assignment.bundle.v3"
                or values.pop("protocol_version", None) != PROTOCOL_VERSION
                or values.pop("support_policy", None) != SUPPORT_POLICY):
            raise ValueError("Assignment requires a protocol 2.2 bundle with schema v3")
        bounds = MultiplicityBounds.from_dict(values.pop("bounds"))
        values["catalog"] = OriginalCatalog.from_dict(values["catalog"])
        values["reference"] = AssignmentReference.from_dict(values["reference"])
        values["thresholds"] = AssignmentThresholds.from_dict(values["thresholds"])
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
            for key in ("probabilities", "log_scores"):
                values = np.asarray(row[key], dtype=float)
                if values.shape != (len(self.reference.labels),):
                    raise ValueError("Assignment probability/score vectors must cover the complete reference")
                row[key] = readonly_array(values)
            if tuple(row.get("reference_labels", ())) != self.reference.labels:
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
            records.append(row)
        object.__setattr__(self, "assignments", _freeze(records))
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
        return thaw_record(dict(schema="clipp2.assignment.result.v4", protocol_version=PROTOCOL_VERSION,
            support_policy=SUPPORT_POLICY,
            bundle_hash=self.bundle_hash, target_ids=self.target_ids, reference=self.reference.to_dict(),
            assignments=self.assignments, multiplicities=self.multiplicities, summary=self.summary,
            run_manifest_id=self.run_manifest_id), arrays_as_lists=True)

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        if (values.pop("schema", None) != "clipp2.assignment.result.v4"
                or values.pop("protocol_version", None) != PROTOCOL_VERSION
                or values.pop("support_policy", None) != SUPPORT_POLICY):
            raise ValueError("Assignment requires a protocol 2.2 result with schema v3")
        values["reference"] = AssignmentReference.from_dict(values["reference"])
        return cls(**values)


def saturated_log_mass(alt, depth):
    alt, depth = np.asarray(alt, float), np.asarray(depth, float)
    p = np.divide(alt, depth, out=np.zeros_like(alt), where=depth > 0)
    return gammaln(depth + 1) - gammaln(alt + 1) - gammaln(depth - alt + 1) + xlogy(alt, p) + xlog1py(depth - alt, -p)


POLICY_FIELDS = ("status", "best_label", "second_best_label", "accepted_label", "posterior",
    "best_log_probability", "second_best_probability", "entropy", "log_score_margin", "cluster_tie",
    "log_competitor_ratio", "comparison_scope", "posterior_pass", "has_finite_alternative",
    "reference_quality_pass", "reference_quality_failed_regions", "mean_profile_deviance", "outlier_count", "outlier_fraction",
    "informative_regions", "pair_deviance_max", "mean_deviance_max", "outlier_fraction_max", "mean_gate_pass",
    "outlier_gate_pass", "fit_screen_status", "fit_status", "baseline_label", "reassessment_status",
    "baseline_changed", "reference_labels", "baseline_reference_label", "final_assignment_label", "target_reason")
REGIONAL_BASE_FIELDS = ("mutation_id", "region_id", "alt_count", "ref_count", "total_depth", "purity",
    "original_major_cn", "original_minor_cn", "original_normal_cn", "original_total_cn", "normal_cn_supplied",
    "segment_id", "mapping_genuine", "count_observed", "positive_depth", "original_eligible", "original_input_excluded", "excluded",
    "informative", "native_phi", "native_call", "native_lower", "native_upper", "original_lower",
    "original_upper", "source_stage", "source_label")
REGIONAL_POLICY_FIELDS = ("best_label", "phi", "conditional_map", "conditional_call", "call_probability",
    "call_log_probability", "top_two_gap", "has_second_component", "native_call_log_probability",
    "multiplicity_tie_atol", "multiplicity_probability_min", "conditional_distribution_defined",
    "multiplicity_tie", "conditional_concentrated", "structural_singleton", "multiplicity_read_informative",
    "membership_status", "multiplicity_status", "multiplicity_candidates", "conditional_probabilities",
    "profile_deviance", "assigned_multiplicity", "reference_quality", "reference_quality_reasons", "center_source")


def confidence_passes(log_probability, threshold, has_finite_alternative):
    """Compare confidence in log space without mistaking rounded one for certainty."""
    log_probability = np.asarray(log_probability, dtype=float)
    finite = np.isfinite(log_probability)
    if threshold == 1:
        answer = finite & ~np.asarray(has_finite_alternative, dtype=bool)
    else:
        answer = finite & (log_probability >= np.log(threshold))
    return bool(answer) if answer.ndim == 0 else answer


def posterior_passes(log_competitor_ratio, threshold, has_finite_alternative):
    """Exact log-competitor boundary, also used for decision-only rescoring."""
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Posterior threshold must lie in (0, 1]")
    ratio = np.asarray(log_competitor_ratio, float)
    defined = ~np.isnan(ratio) & ~np.isposinf(ratio)
    answer = defined & (~np.asarray(has_finite_alternative, bool) if threshold == 1 else
                        ratio <= np.log1p(-threshold)-np.log(threshold))
    return bool(answer) if answer.ndim == 0 else answer


def target_reason(source_stage):
    return {"stage_one": "stage1_clonal_recheck", "stage1": "stage1_clonal_recheck",
            "stage_two": "stage2_near_clonal_recheck", "stage2": "stage2_near_clonal_recheck"}.get(
                source_stage, "fixed_reference_recheck")


REASSESSMENT_OUTCOMES = ("retained_clonal", "reassigned", "unresolved", "not_applicable")


def assignment_outcome(source_stage, baseline_label, accepted_label, clonal_label):
    """Separate workflow clonal rechecks from generic baseline comparisons."""
    if target_reason(source_stage) == "fixed_reference_recheck":
        status = "not_applicable"
    else:
        if clonal_label is None or baseline_label != clonal_label:
            raise ValueError("Workflow reassessment requires a designated clonal reference and clonal baseline")
        status = ("unresolved" if accepted_label is None else "retained_clonal"
                  if accepted_label == clonal_label else "reassigned")
    return dict(reassessment_status=status,
                baseline_changed=None if accepted_label is None else accepted_label != baseline_label)


def reference_quality(reference, best_index, informative):
    """Gate the original winner only, on the target's common informative mask."""
    if best_index is None:
        return None, ()
    failed = np.asarray(informative, bool) & ~reference.regional_quality[best_index]
    return not bool(failed.any()), tuple(region for region, bad in zip(reference.region_ids, failed) if bad)


def _conditional_probabilities(catalog, source, region, phi, lower, upper, native_map):
    """Report the original-support conditional distribution, one pair at a time.

    The native kernel determines the MAP. This compact reporting calculation
    preserves its endpoint convention and computes that MAP's log probability
    without subtracting nearly equal log normalizers.
    """
    candidates = np.arange(lower, upper + 1, dtype=np.int64)
    purity = catalog.purity[source, region]
    denominator = ((1 - purity) * catalog.normal_cn[source, region]
                   + purity * catalog.mean_total_cn[source, region])
    p = np.minimum((purity / denominator * candidates) * phi, 1.)
    alt, depth = catalog.alt_counts[source, region], catalog.total_counts[source, region]
    logs = xlogy(alt, p) + xlog1py(depth - alt, -p)
    chosen = int(native_map) - lower
    ratios = np.exp(logs - logs[chosen])
    others = np.delete(ratios, chosen)
    log_probability = float(-np.log1p(np.sum(others)))
    probabilities = ratios / (1. + np.sum(others))
    finite_alternative = bool(np.isfinite(np.delete(logs, chosen)).any())
    return candidates, probabilities, log_probability, finite_alternative


def score_summary(scores, labels, posterior_min, tie_atol):
    """Pure fixed-reference ordering and relative-confidence decision inputs.

    Exact ties choose the lowest public label. Keeping the winning log posterior
    as -log1p(sum(other/best)) avoids cancellation when its displayed probability
    rounds to one. A threshold of one separately excludes every finite rival.
    """
    scores, labels = np.asarray(scores, dtype=float), tuple(labels)
    if (scores.shape != (len(labels),) or not labels or np.isnan(scores).any() or np.isposinf(scores).any()
            or len(set(labels)) != len(labels) or not np.isfinite(posterior_min) or not 0 < posterior_min <= 1
            or not np.isfinite(tie_atol) or tie_atol < 0):
        raise ValueError("Assignment scores need one finite or negative-infinite value per distinct reference label")
    order = np.lexsort((labels, -scores))
    best = int(order[0]) if np.isfinite(scores).any() else None
    second = int(order[1]) if best is not None and len(labels) > 1 else None
    probabilities = np.full(scores.shape, np.nan)
    finite_alternative = None
    probability = log_probability = second_probability = margin = entropy = competitor_ratio = float("nan")
    tied = confident = None
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
        tied = bool(second is not None and margin <= tie_atol)
        confident = posterior_passes(competitor_ratio, posterior_min, finite_alternative)
    return dict(best_index=best, second_index=second, best_label=labels[best] if best is not None else None,
        second_best_label=labels[second] if second is not None else None, probabilities=probabilities,
        posterior=probability, best_log_probability=log_probability, second_best_probability=second_probability,
        entropy=entropy, log_score_margin=margin, cluster_tie=tied, posterior_pass=confident,
        has_finite_alternative=finite_alternative, log_competitor_ratio=competitor_ratio,
        comparison_scope="single_candidate" if len(labels) == 1 else "multiple_candidates")


def assignment_decision(summary, eligible, informative_regions, descriptive_pass, reference_pass=True):
    """The single-policy acceptance precedence, shared with publication checks."""
    if not eligible:
        return "filtered"
    if not informative_regions:
        return "no_information"
    if summary["best_label"] is None:
        return "incompatible"
    if summary["cluster_tie"]:
        return "ambiguous_tie"
    if not summary["posterior_pass"]:
        return "low_confidence"
    if not reference_pass:
        return "unresolved_reference"
    return "assigned_relative" if descriptive_pass else "unresolved_fit"


@dataclass(frozen=True, slots=True)
class _ReferenceEvaluation:
    records: tuple
    best_indices: np.ndarray
    best_diagnostics: dict
    observed: np.ndarray
    execution: dict


def _score_reference(bundle, progress_callback):
    observation = prepare_prediction(bundle.catalog, bundle.target_ids, bundle.bounds,
                                     device=bundle.device, cpu_threads=bundle.cpu_threads)
    n, r, k = observation.n, observation.r, len(bundle.reference.labels)
    observed = np.asarray(observation.observed, bool)
    coverage = observed.sum(axis=1)
    scores = np.full((n, k), -np.inf)
    if not n:
        return _ReferenceEvaluation((), np.empty(0, dtype=int), {}, observed,
                           dict(status="skipped_empty_targets", device=bundle.device))
    sat = saturated_log_mass(observation.alt, observation.depth)
    with NativeBackend(observation, cpu_threads=bundle.cpu_threads) as backend:
        for index, center in enumerate(bundle.reference.centers):
            diagnostic = backend.component_diagnostics(np.broadcast_to(center, (n, r)))
            if np.any(np.isin(diagnostic["status"], [2, 3])):
                raise NativeExecutionError("Assignment diagnostic execution failed")
            scores[:, index] = np.where(observed, diagnostic["log_emission"], 0.).sum(axis=1) - np.log(k)
            scores[coverage == 0, index] = -np.inf
            if progress_callback:
                progress_callback(dict(operation="assignment", protocol_version=PROTOCOL_VERSION,
                                       completed_candidates=index + 1, candidate_count=k))
        summaries = [score_summary(row, bundle.reference.labels, bundle.thresholds.posterior_min,
                                  bundle.thresholds.tie_atol) for row in scores]
        best = np.array([row["best_index"] if row["best_index"] is not None else 0 for row in summaries], dtype=int)
        best_diagnostics = backend.component_diagnostics(bundle.reference.centers[best], bundle.native_calls)
        if np.any(np.isin(best_diagnostics["status"], [2, 3])):
            raise NativeExecutionError("Assignment multiplicity diagnostic execution failed")
        execution = dict(status="evaluated", **backend.diagnostics())
    residuals = np.maximum(0., 2 * (sat - best_diagnostics["max_component_log_mass"]))
    selected_screen = summarize_profile_deviance(residuals, observed, bundle.thresholds.screen, axis=1)
    records = []
    screen = bundle.thresholds.screen
    for i, mutation in enumerate(bundle.target_ids):
        summary = summaries[i]
        finite = summary["best_label"] is not None
        residual = float(selected_screen["mean_profile_deviance"][i]) if finite else float("nan")
        fraction = float(selected_screen["outlier_fraction"][i]) if finite else float("nan")
        fit_pass = bool(selected_screen["descriptive_pass"][i]) if finite else False
        reference_pass, failed_regions = reference_quality(bundle.reference, summary["best_index"], observed[i])
        decision = assignment_decision(summary, bool(observation.eligible[i].any()), int(coverage[i]), fit_pass,
                                       reference_pass=reference_pass)
        accepted = summary["best_label"] if decision == "assigned_relative" else None
        records.append(dict(mutation_id=mutation, status=decision, accepted_label=accepted,
            **{key: value for key, value in summary.items() if key not in ("best_index", "second_index")},
            reference_quality_pass=reference_pass, reference_quality_failed_regions=failed_regions,
            mean_profile_deviance=residual, outlier_count=int(selected_screen["outlier_count"][i]) if finite else None,
            outlier_fraction=fraction, informative_regions=int(coverage[i]), pair_deviance_max=screen.pair_deviance_max,
            mean_deviance_max=screen.mean_deviance_max, outlier_fraction_max=screen.outlier_fraction_max,
            mean_gate_pass=bool(residual <= screen.mean_deviance_max) if finite else None,
            outlier_gate_pass=bool(fraction <= screen.outlier_fraction_max) if finite else None,
            fit_screen_status=("pass" if fit_pass else "fail") if finite else "not_evaluated",
            fit_status="not_tested", log_scores=scores[i], reference_labels=bundle.reference.labels,
            source_stage=bundle.source_stages[i], source_label=bundle.source_labels[i],
            baseline_label=bundle.baseline_labels[i],
            baseline_reference_label=bundle.baseline_labels[i], final_assignment_label=accepted,
            target_reason=target_reason(bundle.source_stages[i]),
            **assignment_outcome(bundle.source_stages[i], bundle.baseline_labels[i], accepted,
                                 bundle.reference.clonal_label)))
    return _ReferenceEvaluation(tuple(records), best, best_diagnostics, observed, execution)


def assign_mutations(bundle, *, outdir=None, progress_callback=None, run_manifest_id=None):
    """Reassess targets once at a frozen reference using original full support."""
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
        for j, region in enumerate(bundle.reference.region_ids):
            informative = bool(evaluation.observed[i, j])
            present = informative and record["best_label"] is not None
            phi = float(bundle.reference.centers[evaluation.best_indices[i], j]) if present else float("nan")
            defined = bool(present and diagnostic["status"][i, j] == 0)
            singleton = bool(support.lower[i, j] == support.upper[i, j])
            read_informative = bool(defined and phi > 0 and not singleton)
            has_second = bool(diagnostic["has_second"][i, j]) if defined else None
            gap = float(diagnostic["top_two_gap"][i, j]) if defined and has_second else float("nan")
            candidates, conditional = (), ()
            log_probability, profile_deviance = float("nan"), float("nan")
            finite_alternative = False
            if defined:
                candidates, conditional, log_probability, finite_alternative = _conditional_probabilities(
                    catalog, source, j, phi, int(support.lower[i, j]), int(support.upper[i, j]),
                    diagnostic["map_multiplicity"][i, j])
                profile_deviance = max(0., float(2 * (saturated_log_mass(catalog.alt_counts[source, j],
                    catalog.total_counts[source, j]) - diagnostic["max_component_log_mass"][i, j])))
            tied = bool(has_second and gap <= bundle.thresholds.multiplicity_tie_atol) if defined else None
            concentrated = confidence_passes(log_probability, bundle.thresholds.multiplicity_probability_min,
                                              finite_alternative) if defined else None
            if not catalog.eligible[source, j]:
                multiplicity_status = "excluded"
            elif not informative:
                multiplicity_status = "no_information"
            elif not defined:
                multiplicity_status = "incompatible"
            elif phi == 0:
                multiplicity_status = "uninformative_zero_ccf"
            elif singleton:
                multiplicity_status = "structural_singleton"
            elif tied:
                multiplicity_status = "ambiguous_tie"
            elif not concentrated:
                multiplicity_status = "low_concentration"
            else:
                multiplicity_status = "conditional"
            multiplicities.append(dict(mutation_id=mutation, region_id=region,
                alt_count=float(catalog.alt_counts[source, j]),
                ref_count=float(catalog.total_counts[source, j] - catalog.alt_counts[source, j]),
                total_depth=float(catalog.total_counts[source, j]), purity=float(catalog.purity[source, j]),
                original_major_cn=float(catalog.major_cn[source, j]), original_minor_cn=float(catalog.minor_cn[source, j]),
                original_normal_cn=float(catalog.normal_cn[source, j]), original_total_cn=float(catalog.mean_total_cn[source, j]),
                normal_cn_supplied=bool(catalog.normal_cn_supplied[source, j]),
                segment_id=str(catalog.segment_ids[source, j]), mapping_genuine=bool(catalog.mapping_genuine[source, j]),
                count_observed=bool(catalog.count_observed[source, j]), positive_depth=bool(catalog.total_counts[source, j] > 0),
                original_eligible=bool(catalog.eligible[source, j]), excluded=not bool(catalog.eligible[source, j]),
                original_input_excluded=bool(catalog.input_excluded[source, j]),
                informative=informative, native_phi=float(bundle.native_centers[i, j]), native_call=float(bundle.native_calls[i, j]),
                native_lower=int(bundle.native_lower[i, j]), native_upper=int(bundle.native_upper[i, j]),
                original_lower=int(support.lower[i, j]), original_upper=int(support.upper[i, j]),
                source_stage=bundle.source_stages[i], source_label=bundle.source_labels[i],
                best_label=record["best_label"], phi=phi,
                conditional_map=float(diagnostic["map_multiplicity"][i, j]) if defined else float("nan"),
                conditional_call=float(diagnostic["map_multiplicity"][i, j]) if defined and phi > 0 else float("nan"),
                assigned_multiplicity=(float(diagnostic["map_multiplicity"][i, j])
                    if defined and phi > 0 and record["status"] == "assigned_relative" else float("nan")),
                reference_quality=(bool(bundle.reference.regional_quality[evaluation.best_indices[i], j])
                    if record["best_label"] is not None else None),
                reference_quality_reasons=(bundle.reference.regional_quality_reasons[evaluation.best_indices[i]][j]
                    if record["best_label"] is not None else ()),
                center_source=(bundle.reference.center_sources[evaluation.best_indices[i]]
                    if record["best_label"] is not None else None),
                call_probability=float(np.exp(log_probability)), call_log_probability=log_probability,
                top_two_gap=gap, has_second_component=has_second,
                native_call_log_probability=float(diagnostic["accepted_log_probability"][i, j]) if defined else float("nan"),
                multiplicity_tie_atol=bundle.thresholds.multiplicity_tie_atol,
                multiplicity_probability_min=bundle.thresholds.multiplicity_probability_min,
                conditional_distribution_defined=defined, multiplicity_tie=tied, conditional_concentrated=concentrated,
                structural_singleton=singleton, multiplicity_read_informative=read_informative,
                membership_status=record["status"], multiplicity_status=multiplicity_status,
                multiplicity_candidates=candidates, conditional_probabilities=conditional,
                profile_deviance=profile_deviance))
    summary = dict(schema="clipp2.assignment.result.v4", protocol_version=PROTOCOL_VERSION,
        implementation_version=__version__, tumor_id=catalog.tumor_id,
        bundle_hash=bundle.content_hash, reference=bundle.reference.to_dict(), thresholds=asdict(bundle.thresholds),
        bounds_hash=support.content_hash, support_policy=SUPPORT_POLICY, normal_cn_policy=bundle.normal_cn_policy,
        run_manifest_id=execution_manifest_id, bundle_run_manifest_id=bundle.run_manifest_id,
        target_count=len(bundle.target_ids), unchanged_non_target_count=len(bundle.non_target_ids),
        region_count=len(bundle.reference.region_ids), candidate_count=len(bundle.reference.labels),
        assigned_occupancy={str(label): sum(row["accepted_label"] == label for row in evaluation.records)
                            for label in bundle.reference.labels},
        status_counts={status: sum(row["status"] == status for row in evaluation.records)
                       for status in sorted({row["status"] for row in evaluation.records})},
        reassessment_counts={status: sum(row["reassessment_status"] == status for row in evaluation.records)
                             for status in REASSESSMENT_OUTCOMES},
        evaluation_receipt=dict(device=bundle.device, cpu_threads=bundle.cpu_threads,
            run_manifest_id=execution_manifest_id, bundle_run_manifest_id=bundle.run_manifest_id,
            configuration=asdict(bundle.thresholds), execution=evaluation.execution),
        fit_status="not_tested", population_scope="targets_only",
        comparison_scope="single_candidate" if len(bundle.reference.labels) == 1 else "multiple_candidates",
        interpretation="relative_assignment_at_frozen_reference_not_a_refit")
    bundle.validated_payload()
    result = AssignmentResult(bundle.content_hash, bundle.target_ids, bundle.reference,
                              evaluation.records, tuple(multiplicities), summary, run_manifest_id=execution_manifest_id)
    if outdir is not None:
        from .io.assignment import publish_assignment
        publish_assignment(bundle, result, outdir)
    return result
