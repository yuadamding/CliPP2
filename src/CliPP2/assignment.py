"""Immutable fixed-reference prediction, separate from occupied fitted states."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections.abc import Mapping
import hashlib
import json
from types import MappingProxyType

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy, xlog1py

from .config import AssignmentThresholds
from .io.data import ImmutableArrayRecord, MultiplicityBounds, OriginalCatalog, readonly_array
from .kernel.backend import NativeBackend, NativeExecutionError
from .kernel.observations import prepare_prediction


def _json_value(value):
    """Lossless JSON codec for scientific nonfinite values, never bare NaN."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "NaN" if np.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def content_hash(value):
    return hashlib.sha256(json.dumps(_json_value(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return readonly_array(value) if isinstance(value, np.ndarray) else value


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

    def __post_init__(self):
        targets, donors = tuple(self.target_ids), tuple(self.donor_ids)
        if np.any((self.catalog.cn_state_count > 1) & self.catalog.mapping_genuine):
            raise ValueError("Fixed-reference assignment currently requires genuine single-state CN; mixed CN is unsupported")
        if (not targets or len(set(targets)) != len(targets) or len(set(donors)) != len(donors)
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
        self.bounds.for_data(data)
        original = self.catalog.default_bounds().subset(targets)
        selected_support = self.bounds.subset(targets)
        if not np.array_equal(selected_support.upper, original.upper):
            raise ValueError("Stage-three restricted support must retain the original upper endpoints")
        native_support = MultiplicityBounds(targets, self.reference.region_ids, self.native_lower,
                                            self.native_upper, "accepted_native_baseline")
        native_support.for_data(data)
        object.__setattr__(self, "native_lower", native_support.lower)
        object.__setattr__(self, "native_upper", native_support.upper)
        for name in ("native_centers", "native_calls"):
            raw = np.asarray(getattr(self, name), dtype=float)
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

    def to_dict(self):
        values = {name: getattr(self, name) for name in ("target_ids", "donor_ids", "native_centers",
            "native_calls", "native_lower", "native_upper", "source_stages", "source_labels",
            "device", "cpu_threads", "provenance")}
        return _json_value(dict(schema="clipp2.assignment.bundle.v1", **values,
            catalog=self.catalog.to_dict(), reference=self.reference.to_dict(),
            bounds=self.bounds.to_dict(), thresholds=asdict(self.thresholds)))

    @property
    def content_hash(self):
        return content_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        values = dict(value)
        if values.pop("schema", None) != "clipp2.assignment.bundle.v1":
            raise ValueError("Unsupported assignment bundle schema")
        values["catalog"] = OriginalCatalog.from_dict(values["catalog"])
        values["reference"] = AssignmentReference.from_dict(values["reference"])
        values["bounds"] = MultiplicityBounds.from_dict(values["bounds"])
        values["thresholds"] = AssignmentThresholds(**values["thresholds"])
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


def _policy_scores(bundle, bounds, policy, progress_callback):
    observation = prepare_prediction(bundle.catalog, bundle.target_ids, bounds,
                                     device=bundle.device, cpu_threads=bundle.cpu_threads)
    n, r, k = observation.n, observation.r, len(bundle.reference.labels)
    observed = np.asarray(observation.observed, bool)
    eligible, coverage = observation.eligible, observed.sum(axis=1)
    sat = saturated_log_mass(observation.alt, observation.depth)
    scores, deviances = np.full((n, k), -np.inf), np.full((n, k), np.inf)
    with NativeBackend(observation, cpu_threads=bundle.cpu_threads) as backend:
        for index, center in enumerate(bundle.reference.centers):
            x = np.broadcast_to(center, (n, r))
            diagnostic = backend.component_diagnostics(x)
            if np.any((diagnostic["status"] == 2) | (diagnostic["status"] == 3)):
                raise NativeExecutionError("Assignment diagnostic execution failed")
            emissions = np.where(observed, diagnostic["log_emission"], 0.)
            scores[:, index] = np.sum(emissions, axis=1) - np.log(k)
            residual = np.where(observed, np.maximum(0., 2 * (sat - diagnostic["max_component_log_mass"])), 0.)
            deviances[:, index] = np.divide(residual.sum(axis=1), coverage,
                out=np.full(n, np.nan), where=coverage > 0)
            if progress_callback:
                progress_callback(dict(operation="assignment", policy=policy, completed_candidates=index + 1, candidate_count=k))
        compatible = np.isfinite(scores).any(axis=1) & (coverage > 0)
        posteriors = np.full_like(scores, np.nan)
        posteriors[compatible] = np.exp(scores[compatible] - logsumexp(scores[compatible], axis=1)[:, None])
        # Reference order is a display tie-break only. It never turns a tie into acceptance.
        best = np.argmax(scores, axis=1)
        best_x = bundle.reference.centers[best]
        best_diagnostics = backend.component_diagnostics(best_x, accepted_calls=bundle.native_calls)
    records = []
    for i, mutation in enumerate(bundle.target_ids):
        informative = int(coverage[i])
        has_information = informative > 0
        finite = bool(compatible[i]) and has_information
        ordered = np.sort(scores[i])[::-1]
        margin = float(ordered[0] - ordered[1]) if finite and k > 1 else float("inf") if finite else float("nan")
        probability = float(posteriors[i, best[i]]) if finite else float("nan")
        residual = float(deviances[i, best[i]]) if finite else float("nan")
        if not eligible[i].any():
            decision = "filtered"
        elif not has_information:
            decision = "no_information"
        elif not compatible[i]:
            decision = "incompatible"
        elif k > 1 and margin <= bundle.thresholds.tie_atol:
            decision = "ambiguous_tie"
        elif probability < bundle.thresholds.posterior_min:
            decision = "low_confidence"
        elif not np.isfinite(residual) or residual > bundle.thresholds.mean_deviance_max:
            decision = "unresolved_fit"
        else:
            decision = "assigned_relative"
        records.append(dict(mutation_id=mutation, status=decision,
            best_label=bundle.reference.labels[best[i]] if finite else None,
            accepted_label=bundle.reference.labels[best[i]] if decision == "assigned_relative" else None,
            posterior=probability, log_score_margin=margin, mean_profile_deviance=residual,
            informative_regions=informative, log_scores=scores[i], probabilities=posteriors[i],
            source_stage=bundle.source_stages[i], source_label=bundle.source_labels[i]))
    return records, scores, posteriors, deviances, best, best_diagnostics, observed


def assign_mutations(bundle, *, outdir=None, progress_callback=None):
    """Score the same frozen reference twice, without optimizing any parameter."""
    if type(bundle) is not AssignmentBundle:
        raise TypeError("assign_mutations requires a validated AssignmentBundle")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable")
    original_bounds = bundle.catalog.default_bounds().subset(bundle.target_ids)
    restricted = _policy_scores(bundle, bundle.bounds, "restricted", progress_callback)
    original = _policy_scores(bundle, original_bounds, "original", progress_callback)
    finite_r = np.isfinite(restricted[1])
    finite_o = np.isfinite(original[1])
    if np.any(finite_r & ~finite_o):
        raise ArithmeticError("Restricted compatibility must imply original compatibility")
    both = np.isfinite(restricted[3]) & np.isfinite(original[3])
    if np.any(original[3][both] > restricted[3][both] + 1e-8):
        raise ArithmeticError("Original profile deviance exceeds restricted at the same candidate")
    comparison, multiplicities = [], []
    support = bundle.bounds.subset(bundle.target_ids)
    for i, mutation in enumerate(bundle.target_ids):
        left, right = restricted[0][i], original[0][i]
        p, q = restricted[2][i], original[2][i]
        accepted = left["accepted_label"]
        comparison.append(dict(mutation_id=mutation,
            support_sensitive_label=left["best_label"] != right["best_label"],
            support_sensitive_decision=left["status"] != right["status"],
            max_posterior_difference=float(np.max(np.abs(p - q))) if np.isfinite(p).all() and np.isfinite(q).all() else float("nan"),
            support_stable_label=accepted if accepted is not None and accepted == right["accepted_label"] else None))
        for j, region in enumerate(bundle.reference.region_ids):
            row = dict(mutation_id=mutation, region_id=region,
                native_phi=float(bundle.native_centers[i, j]), native_call=float(bundle.native_calls[i, j]),
                native_lower=int(bundle.native_lower[i, j]), native_upper=int(bundle.native_upper[i, j]),
                restricted_lower=int(support.lower[i, j]), restricted_upper=int(support.upper[i, j]),
                native_call_excluded=bool(np.isfinite(bundle.native_calls[i, j]) and
                    not support.lower[i, j] <= bundle.native_calls[i, j] <= support.upper[i, j]),
                source_stage=bundle.source_stages[i], source_label=bundle.source_labels[i])
            for name, result in (("restricted", restricted), ("original", original)):
                record, diagnostic = result[0][i], result[5]
                informative = bool(result[6][i, j]) and record["best_label"] is not None
                phi = bundle.reference.centers[result[4][i], j]
                # A zero CCF has no multiplicity information even for singleton support.
                usable = informative and phi > 0 and diagnostic["status"][i, j] == 0
                row.update({name + "_best_label": record["best_label"],
                    name + "_phi": float(phi) if informative else float("nan"),
                    name + "_conditional_call": float(diagnostic["map_multiplicity"][i, j]) if usable else float("nan"),
                    name + "_call_probability": float(np.exp(diagnostic["map_log_probability"][i, j])) if usable else float("nan"),
                    name + "_call_log_probability": float(diagnostic["map_log_probability"][i, j]) if usable else float("nan"),
                    name + "_top_two_gap": float(diagnostic["top_two_gap"][i, j]) if usable else float("nan"),
                    name + "_has_second_component": bool(diagnostic["has_second"][i, j]) if usable else None,
                    name + "_native_call_log_probability": float(diagnostic["accepted_log_probability"][i, j]) if usable else float("nan"),
                    name + "_membership_status": record["status"],
                    name + "_multiplicity_status": "conditional" if usable else "zero_ccf" if informative and phi == 0 else "missing_or_incompatible"})
            multiplicities.append(row)
    occupancy = {policy: {str(label): sum(record["accepted_label"] == label for record in results[0])
                         for label in bundle.reference.labels}
                 for policy, results in (("restricted", restricted), ("original", original))}
    summary = dict(schema="clipp2.assignment.result.v1", tumor_id=bundle.catalog.tumor_id,
        bundle_hash=bundle.content_hash, reference=bundle.reference.to_dict(), thresholds=asdict(bundle.thresholds),
        input_identity=bundle.catalog.input_identity, bounds_hash=bundle.bounds.content_hash,
        target_count=len(bundle.target_ids), region_count=len(bundle.reference.region_ids),
        candidate_count=len(bundle.reference.labels), assigned_occupancy=occupancy,
        status_counts={name: {status: sum(v["status"] == status for v in result[0])
            for status in sorted({v["status"] for v in result[0]})}
            for name, result in (("restricted", restricted), ("original", original))},
        interpretation="relative_assignment_at_frozen_reference_not_a_refit",
        original_companion_scope="support_sensitivity_at_same_reference_not_uncapped_stage_two")
    result = AssignmentResult(bundle.content_hash, bundle.target_ids, bundle.reference,
                              restricted[0], original[0], comparison, tuple(multiplicities), summary)
    if outdir is not None:
        from .io.assignment import publish_assignment
        publish_assignment(bundle, result, outdir)
    return result
