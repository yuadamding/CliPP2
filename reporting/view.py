"""One immutable reporting view for every CliPP2 analysis tier."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
from pathlib import Path

import numpy as np

from .. import __version__ as SOFTWARE_VERSION
from ..config import FitConfig, PRODUCTION_SELECTION_POLICY
from ..core.fusion.types import RawFit
from ..core.objective import (
    ObservedModel,
    ObservedTerms,
    compile_observed_model,
    make_base_objective_key,
    observed_terms_numpy,
)
from ..core.posterior import PosteriorSummary, summarize_posterior_numpy
from ..core.scalar import PartitionFit
from ..io.data import ExclusionCode, TumorData
from ..model_selection.partitions import partition_signature
from ..model_selection.types import (
    BICSelectionResult,
    DiagnosticOnlyResult,
    DirectPartition,
    FusionPartition,
    RawFusionCandidate,
    SearchReport,
    SecondaryFallbackResult,
    SelectablePartitionCandidate,
    SelectionScore,
    TumorSelectionOutcome,
)

SUMMARY_SCHEMA_VERSION = 11
OUTPUT_SCHEMA_VERSION = 2
CCF_CLUSTER_ORDERING_METHOD = "identified_region_rms_distance_to_one_v1"


@dataclass(frozen=True, slots=True)
class TableView:
    """A dependency-free ordered tabular projection."""

    columns: tuple[str, ...]
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _PreparedAnalysis:
    available: np.ndarray
    supported: np.ndarray
    included: np.ndarray
    reasons: np.ndarray
    labels: np.ndarray | None
    centers: np.ndarray | None
    lower: np.ndarray | None
    upper: np.ndarray | None
    identified: np.ndarray | None
    ordered_labels: np.ndarray | None
    distances: np.ndarray | None
    identified_counts: np.ndarray | None
    model: ObservedModel
    terms: ObservedTerms | None
    posterior: PosteriorSummary | None
    eps: float


@dataclass(frozen=True, slots=True)
class _NormalizedOutcome:
    """The common reporting surface of the three closed outcome variants."""

    primary_estimator_available: bool
    selected_candidate: SelectablePartitionCandidate | None
    raw_reference: RawFusionCandidate | None
    partition_parent_raw: RawFusionCandidate | None
    diagnostic_raw_fit: RawFit | None
    failure_reason: str
    report: SearchReport
    selected_lambda_representative: float | None
    selected_kkt_residual: float | None


def _normalize_outcome(result: TumorSelectionOutcome) -> _NormalizedOutcome:
    if isinstance(result, BICSelectionResult):
        model = result.selected_model
        raw_reference = model.raw_reference
        return _NormalizedOutcome(
            primary_estimator_available=True,
            selected_candidate=model.partition_candidate,
            raw_reference=raw_reference,
            partition_parent_raw=model.partition_parent_raw,
            diagnostic_raw_fit=raw_reference.raw_fit,
            failure_reason="",
            report=result.report,
            selected_lambda_representative=result.selected_lambda_representative,
            selected_kkt_residual=result.selected_kkt_residual,
        )
    if isinstance(result, SecondaryFallbackResult):
        return _NormalizedOutcome(
            primary_estimator_available=False,
            selected_candidate=result.selected_partition,
            raw_reference=None,
            partition_parent_raw=None,
            diagnostic_raw_fit=result.best_raw_attempt,
            failure_reason=str(result.reason),
            report=result.report,
            selected_lambda_representative=None,
            selected_kkt_residual=None,
        )
    if isinstance(result, DiagnosticOnlyResult):
        return _NormalizedOutcome(
            primary_estimator_available=False,
            selected_candidate=None,
            raw_reference=None,
            partition_parent_raw=None,
            diagnostic_raw_fit=result.best_raw_attempt,
            failure_reason=str(result.reason),
            report=result.report,
            selected_lambda_representative=None,
            selected_kkt_residual=None,
        )
    raise TypeError(f"Unsupported selection outcome: {type(result).__name__}.")


def _reported_exclusion_reasons(codes: np.ndarray) -> np.ndarray:
    """Expand compact input provenance only at the reporting boundary."""

    values = np.asarray(codes, dtype=np.uint8)
    return np.asarray(
        [
            None
            if int(value) == int(ExclusionCode.INCLUDED)
            else ExclusionCode(int(value)).name
            for value in values.flat
        ],
        dtype=object,
    ).reshape(values.shape)


def _array_fingerprint(values: np.ndarray, *, dtype: np.dtype) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AnalysisView:
    """One normalized, output-ready view of any typed selection outcome."""

    data: TumorData
    input_file: Path
    fit_config: FitConfig
    selection_result: TumorSelectionOutcome
    outcome: _NormalizedOutcome = field(init=False, repr=False)
    prepared: _PreparedAnalysis = field(init=False, repr=False)
    clusters: TableView = field(init=False)
    mutations: TableView = field(init=False)
    attempts: TableView = field(init=False)
    regions: tuple[dict[str, object], ...] = field(init=False)

    def __post_init__(self) -> None:
        outcome = _normalize_outcome(self.selection_result)
        object.__setattr__(self, "outcome", outcome)
        prepared = _prepare_analysis(self)
        object.__setattr__(self, "prepared", prepared)
        object.__setattr__(self, "clusters", _cluster_table(self, prepared))
        object.__setattr__(self, "mutations", _mutation_table(self, prepared))
        object.__setattr__(self, "attempts", _attempt_table(self))
        object.__setattr__(self, "regions", _region_rows(self, prepared))

    @property
    def primary_estimator_available(self) -> bool:
        return self.outcome.primary_estimator_available

    @property
    def selected_candidate(self) -> SelectablePartitionCandidate | None:
        return self.outcome.selected_candidate

    @property
    def raw_reference(self) -> RawFusionCandidate | None:
        return self.outcome.raw_reference

    @property
    def partition_parent_raw(self) -> RawFusionCandidate | None:
        return self.outcome.partition_parent_raw

    @property
    def raw_fit(self) -> RawFit | None:
        reference = self.raw_reference
        return None if reference is None else reference.raw_fit

    @property
    def diagnostic_raw_fit(self) -> RawFit | None:
        return self.outcome.diagnostic_raw_fit

    @property
    def partition(self) -> FusionPartition | DirectPartition | None:
        candidate = self.selected_candidate
        return None if candidate is None else candidate.partition

    @property
    def refit(self) -> PartitionFit | None:
        candidate = self.selected_candidate
        return None if candidate is None else candidate.refit

    @property
    def score(self) -> SelectionScore | None:
        candidate = self.selected_candidate
        return None if candidate is None else candidate.score

    @property
    def failure_reason(self) -> str:
        return self.outcome.failure_reason


def ccf_cluster_order(
    centers: np.ndarray,
    *,
    statistically_identified: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Order clusters by RMS distance to CCF one over identified regions."""

    values = np.asarray(centers, dtype=np.float64)
    identified = np.asarray(statistically_identified, dtype=bool)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("cluster centers must be a finite two-dimensional array.")
    if identified.shape != values.shape:
        raise ValueError("cluster-center identification mask has the wrong shape.")
    counts = np.sum(identified, axis=1)
    squared = np.sum(np.where(identified, np.square(values - 1.0), 0.0), axis=1)
    distances = np.full(values.shape[0], np.nan, dtype=np.float64)
    np.sqrt(squared, out=distances, where=counts > 0)
    distances[counts > 0] /= np.sqrt(counts[counts > 0])
    canonical = np.arange(values.shape[0], dtype=np.int64)
    order = np.lexsort((canonical, np.where(np.isnan(distances), np.inf, distances)))
    ordered = np.empty(values.shape[0], dtype=np.int64)
    ordered[order] = canonical
    return ordered, distances, counts.astype(np.int64, copy=False)


def _validated_identity(
    data: TumorData,
    candidate: SelectablePartitionCandidate,
    *,
    model: ObservedModel,
    eps: float,
) -> np.ndarray:
    partition = candidate.partition
    refit = candidate.refit
    labels = np.asarray(partition.labels, dtype=np.int64)
    if labels.shape != (int(data.num_mutations),):
        raise ValueError("Selected partition does not cover every tumor mutation.")
    if partition.signature != partition_signature(
        labels,
        partition.mutation_ids if partition.mutation_ids else None,
    ):
        raise ValueError("Selected partition signature is not reconstructible.")
    if partition.signature != refit.partition_signature or not np.array_equal(
        labels,
        np.asarray(refit.labels, dtype=np.int64),
    ):
        raise ValueError("Selected partition and fixed-label refit disagree.")
    try:
        refit.validate_observed_model(model, eps=eps)
    except ValueError as exc:
        raise ValueError(
            "Selected fixed-partition refit does not match the reporting model."
        ) from exc
    return labels


def _prepare_analysis(analysis: AnalysisView) -> _PreparedAnalysis:
    data = analysis.data
    shape = (int(data.num_mutations), int(data.num_regions))
    available = np.asarray(data.count_available, dtype=bool)
    supported = np.asarray(data.likelihood_supported, dtype=bool)
    included = np.asarray(data.objective_inclusion_mask(), dtype=bool)
    reasons = _reported_exclusion_reasons(data.exclusion_code)
    for name, values in (
        ("count_available", available),
        ("likelihood_supported", supported),
        ("likelihood_included", included),
        ("likelihood_exclusion_reason", reasons),
    ):
        if values.shape != shape:
            raise ValueError(f"{name} must have shape {shape}.")
    diagnostic = analysis.diagnostic_raw_fit
    eps = float(
        analysis.fit_config.eps
        if diagnostic is None
        else diagnostic.provenance.likelihood_eps
    )
    model = compile_observed_model(
        data,
        major_prior=float(analysis.fit_config.major_prior),
        eps=eps,
    )
    if diagnostic is not None:
        provenance = diagnostic.provenance
        objective_key = provenance.objective_key
        if str(objective_key.base.likelihood_hash) != str(model.likelihood_fingerprint):
            raise ValueError(
                "Reporting ObservedModel does not match the fitted likelihood identity."
            )
        rebound = make_base_objective_key(
            model,
            graph_hash=str(objective_key.base.graph_hash),
            eps=eps,
        )
        if str(rebound.fingerprint) != str(provenance.objective_spec_hash):
            raise ValueError(
                "Reporting ObservedModel does not match the fitted objective identity."
            )
    candidate = analysis.selected_candidate
    if candidate is None:
        return _PreparedAnalysis(
            available,
            supported,
            included,
            reasons,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            model,
            None,
            None,
            eps,
        )
    labels = _validated_identity(data, candidate, model=model, eps=eps)
    raw_reference = analysis.raw_reference
    if raw_reference is not None and raw_reference is not candidate:
        _validated_identity(data, raw_reference, model=model, eps=eps)
    partition = candidate.partition
    refit = candidate.refit
    centers = np.asarray(refit.cluster_centers, dtype=np.float64)
    expected = (int(partition.n_clusters), shape[1])
    if centers.shape != expected or not np.all(np.isfinite(centers)):
        raise ValueError(f"Selected centers must have finite shape {expected}.")
    phi = np.asarray(refit.phi, dtype=np.float64)
    if phi.shape != shape or not np.array_equal(phi, centers[labels]):
        raise ValueError("Selected refit phi must equal its immutable labeled centers.")
    lower = np.asarray(refit.coordinate_argmin_lower, dtype=np.float64)
    upper = np.asarray(refit.coordinate_argmin_upper, dtype=np.float64)
    identified = np.asarray(refit.coordinate_statistically_identified, dtype=bool)
    if (
        lower.shape != expected
        or upper.shape != expected
        or identified.shape != expected
    ):
        raise ValueError("Selected coordinate status arrays do not match centers.")
    ordered, distances, counts = ccf_cluster_order(
        centers,
        statistically_identified=identified,
    )
    terms = observed_terms_numpy(model, phi, eps=eps)
    posterior = summarize_posterior_numpy(
        model,
        phi,
        eps=eps,
        terms=terms,
        reportable=included & identified[labels],
    )
    return _PreparedAnalysis(
        available,
        supported,
        included,
        reasons,
        labels,
        centers,
        lower,
        upper,
        identified,
        ordered,
        distances,
        counts,
        model,
        terms,
        posterior,
        eps,
    )


_CLUSTER_COLUMNS = (
    "tumor_id",
    "cluster_label",
    "ccf_ordered_cluster_label",
    "ccf_distance_to_one",
    "ccf_distance_identified_region_count",
    "region_id",
    "cluster_size",
    "phi",
    "phi_best_available",
    "phi_joint_raw",
    "phi_conditional_refit",
    "phi_single_region",
    "phi_interval_lower",
    "phi_interval_upper",
    "phi_interval_component_count",
    "phi_interval_disconnected",
    "statistically_identified",
    "estimate_tier",
    "estimate_source",
    "primary_estimator_available",
    "observed_mutation_count",
    "supported_mutation_count",
    "included_mutation_count",
    "profile_loglik",
    "profile_optimality_gap",
    "partition_signature",
    "observed_model_hash",
    "reporting_model_hash",
    "objective_spec_hash",
    "original_graph_hash",
    "observation_mask_hash",
)


def _mask_hash(mask: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(mask, dtype=bool))
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _analysis_tier(analysis: AnalysisView) -> str:
    if analysis.primary_estimator_available:
        return "joint_certified"
    if analysis.selected_candidate is not None:
        return "conditional_partition_refit"
    return "unsupported_or_unidentified"


def _cluster_table(
    analysis: AnalysisView,
    prepared: _PreparedAnalysis,
) -> TableView:
    if prepared.labels is None or prepared.centers is None:
        return TableView(_CLUSTER_COLUMNS, ())
    assert prepared.identified is not None
    assert prepared.lower is not None and prepared.upper is not None
    assert prepared.ordered_labels is not None
    assert prepared.distances is not None and prepared.identified_counts is not None
    assert prepared.terms is not None
    data = analysis.data
    labels = prepared.labels
    centers = prepared.centers
    candidate = analysis.selected_candidate
    assert candidate is not None
    sizes = np.bincount(labels, minlength=centers.shape[0])
    diagnostic = analysis.diagnostic_raw_fit
    raw_phi = (
        np.asarray(diagnostic.phi, dtype=np.float64)
        if diagnostic is not None and analysis.primary_estimator_available
        else None
    )
    observed_model_hash = str(prepared.model.fingerprint)
    objective_spec_hash = (
        "" if diagnostic is None else str(diagnostic.provenance.objective_spec_hash)
    )
    original_graph_hash = (
        "" if diagnostic is None else str(diagnostic.provenance.original_graph_hash)
    )
    rows: list[dict[str, object]] = []
    for cluster in range(centers.shape[0]):
        members = np.flatnonzero(labels == cluster)
        for region, region_id in enumerate(data.region_ids):
            identified = bool(prepared.identified[cluster, region])
            tier = (
                _analysis_tier(analysis)
                if identified
                else "structural_representative_only"
            )
            phi = float(centers[cluster, region])
            rows.append(
                {
                    "tumor_id": data.tumor_id,
                    "cluster_label": cluster + 1,
                    "ccf_ordered_cluster_label": int(prepared.ordered_labels[cluster]),
                    "ccf_distance_to_one": float(prepared.distances[cluster]),
                    "ccf_distance_identified_region_count": int(
                        prepared.identified_counts[cluster]
                    ),
                    "region_id": str(region_id),
                    "cluster_size": int(sizes[cluster]),
                    "phi": phi,
                    "phi_best_available": phi,
                    "phi_joint_raw": None
                    if raw_phi is None
                    else float(np.mean(raw_phi[members, region])),
                    "phi_conditional_refit": phi,
                    "phi_single_region": None,
                    "phi_interval_lower": float(prepared.lower[cluster, region]),
                    "phi_interval_upper": float(prepared.upper[cluster, region]),
                    "phi_interval_component_count": None,
                    "phi_interval_disconnected": None,
                    "statistically_identified": identified,
                    "estimate_tier": tier,
                    "estimate_source": str(candidate.partition.source),
                    "primary_estimator_available": analysis.primary_estimator_available,
                    "observed_mutation_count": int(
                        np.sum(prepared.available[members, region])
                    ),
                    "supported_mutation_count": int(
                        np.sum(prepared.supported[members, region])
                    ),
                    "included_mutation_count": int(
                        np.sum(prepared.included[members, region])
                    ),
                    "profile_loglik": float(
                        -np.sum(prepared.terms.loss[members, region])
                    ),
                    "profile_optimality_gap": None,
                    "partition_signature": str(candidate.partition.signature),
                    "observed_model_hash": observed_model_hash,
                    "reporting_model_hash": str(
                        prepared.model.reporting_fingerprint
                    ),
                    "objective_spec_hash": objective_spec_hash,
                    "original_graph_hash": original_graph_hash,
                    "observation_mask_hash": _mask_hash(prepared.included),
                }
            )
    return TableView(_CLUSTER_COLUMNS, tuple(rows))


_MUTATION_COLUMNS = (
    "tumor_id",
    "mutation_id",
    "region_id",
    "cluster_label",
    "ccf_ordered_cluster_label",
    "phi",
    "phi_best_available",
    "major_cn",
    "minor_cn",
    "alt_count",
    "ref_count",
    "count_available",
    "likelihood_supported",
    "likelihood_included",
    "likelihood_exclusion_reason",
    "phi_estimate_tier",
    "phi_estimate_source",
    "phi_statistically_identified",
    "phi_interval_lower",
    "phi_interval_upper",
    "partition_signature",
    "observed_model_hash",
    "reporting_model_hash",
    "objective_spec_hash",
    "original_graph_hash",
    "multiplicity_estimated",
    "gamma_major",
    "major_call",
    "multiplicity_call",
    "path_supported",
    "map_path",
    "pre_switch_path_probability",
    "post_switch_path_probability",
    "switch_boundary_ambiguity_probability",
    "posterior_mutant_copy_mass",
    "posterior_effective_multiplicity",
    "map_mutant_copy_mass",
    "map_effective_multiplicity",
    "amplified_mutant_copy_probability",
    "amplified_mutant_copy_call",
    "path_entropy",
)


def _finite_or_none(value: object) -> object:
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return None
    return value


def _mutation_table(
    analysis: AnalysisView,
    prepared: _PreparedAnalysis,
) -> TableView:
    data = analysis.data
    shape = prepared.included.shape
    labels = prepared.labels
    posterior = prepared.posterior
    alt = np.asarray(data.alt_counts, dtype=np.float64)
    ref = np.asarray(data.total_counts, dtype=np.float64) - alt
    if labels is None:
        phi = np.full(shape, np.nan)
        cluster = np.full(shape, -1, dtype=np.int64)
        ordered = np.full(shape, -1, dtype=np.int64)
        identified = np.zeros(shape, dtype=bool)
        lower = np.full(shape, prepared.eps)
        upper = np.asarray(prepared.model.upper, dtype=np.float64)
        source = "none"
        signature = ""
    else:
        assert prepared.centers is not None and prepared.identified is not None
        assert prepared.ordered_labels is not None
        assert prepared.lower is not None and prepared.upper is not None
        phi = prepared.centers[labels]
        cluster = np.broadcast_to(labels[:, None], shape)
        ordered = np.broadcast_to(prepared.ordered_labels[labels, None], shape)
        identified = prepared.identified[labels]
        lower = prepared.lower[labels]
        upper = prepared.upper[labels]
        source = str(analysis.selected_candidate.partition.source)
        signature = str(analysis.selected_candidate.partition.signature)
    diagnostic = analysis.diagnostic_raw_fit
    observed_model_hash = str(prepared.model.fingerprint)
    legacy_multiplicity_report = bool(
        prepared.model.uses_binary_linear_mixture_fast_path
        and prepared.model.major_indicator is not None
    )
    objective_spec_hash = (
        "" if diagnostic is None else str(diagnostic.provenance.objective_spec_hash)
    )
    original_graph_hash = (
        "" if diagnostic is None else str(diagnostic.provenance.original_graph_hash)
    )
    rows: list[dict[str, object]] = []
    for mutation, mutation_id in enumerate(data.mutation_ids):
        for region, region_id in enumerate(data.region_ids):
            available = bool(prepared.available[mutation, region])
            included = bool(prepared.included[mutation, region])
            reportable = bool(included and identified[mutation, region])
            tier = (
                _analysis_tier(analysis)
                if reportable
                else (
                    "structural_representative_only"
                    if labels is not None
                    else "unsupported_or_unidentified"
                )
            )
            row: dict[str, object] = {column: None for column in _MUTATION_COLUMNS}
            row.update(
                {
                    "tumor_id": data.tumor_id,
                    "mutation_id": str(mutation_id),
                    "region_id": str(region_id),
                    "cluster_label": None
                    if cluster[mutation, region] < 0
                    else int(cluster[mutation, region]) + 1,
                    "ccf_ordered_cluster_label": None
                    if ordered[mutation, region] < 0
                    else int(ordered[mutation, region]),
                    "phi": _finite_or_none(phi[mutation, region]),
                    "phi_best_available": _finite_or_none(phi[mutation, region]),
                    "major_cn": float(data.major_cn[mutation, region]),
                    "minor_cn": float(data.minor_cn[mutation, region]),
                    "alt_count": float(alt[mutation, region]) if available else None,
                    "ref_count": float(ref[mutation, region]) if available else None,
                    "count_available": int(available),
                    "likelihood_supported": int(prepared.supported[mutation, region]),
                    "likelihood_included": int(included),
                    "likelihood_exclusion_reason": None
                    if included
                    else prepared.reasons[mutation, region],
                    "phi_estimate_tier": tier,
                    "phi_estimate_source": source,
                    "phi_statistically_identified": int(identified[mutation, region]),
                    "phi_interval_lower": float(lower[mutation, region]),
                    "phi_interval_upper": float(upper[mutation, region]),
                    "partition_signature": signature,
                    "observed_model_hash": observed_model_hash,
                    "reporting_model_hash": str(
                        prepared.model.reporting_fingerprint
                    ),
                    "objective_spec_hash": objective_spec_hash,
                    "original_graph_hash": original_graph_hash,
                }
            )
            if (
                posterior is not None
                and legacy_multiplicity_report
            ):
                assert posterior.multiplicity_estimated is not None
                assert (
                    posterior.major_probability is not None
                    and posterior.major_call is not None
                )
                if reportable:
                    row.update(
                        {
                            "multiplicity_estimated": int(
                                posterior.multiplicity_estimated[mutation, region]
                            ),
                            "gamma_major": float(
                                posterior.major_probability[mutation, region]
                            ),
                            "major_call": int(posterior.major_call[mutation, region]),
                            "multiplicity_call": float(
                                posterior.map_multiplicity[mutation, region]
                            ),
                        }
                    )
            elif not legacy_multiplicity_report:
                row.update(
                    {
                        "path_supported": int(prepared.supported[mutation, region]),
                    }
                )
                if posterior is not None and reportable:
                    row.update(
                        {
                            "map_path": int(posterior.map_path[mutation, region]) + 1,
                            "pre_switch_path_probability": float(
                                posterior.pre_switch_probability[mutation, region]
                            ),
                            "post_switch_path_probability": float(
                                posterior.post_switch_probability[mutation, region]
                            ),
                            "switch_boundary_ambiguity_probability": float(
                                posterior.switch_boundary_probability[mutation, region]
                            ),
                            "posterior_mutant_copy_mass": float(
                                posterior.expected_mutant_copy_mass[mutation, region]
                            ),
                            "posterior_effective_multiplicity": _finite_or_none(
                                posterior.expected_multiplicity[mutation, region]
                            ),
                            "map_mutant_copy_mass": float(
                                posterior.map_mutant_copy_mass[mutation, region]
                            ),
                            "map_effective_multiplicity": _finite_or_none(
                                posterior.map_multiplicity[mutation, region]
                            ),
                            "amplified_mutant_copy_probability": float(
                                posterior.amplified_mutant_copy_probability[
                                    mutation, region
                                ]
                            ),
                            "amplified_mutant_copy_call": int(
                                posterior.amplified_mutant_copy_call[mutation, region]
                            ),
                            "path_entropy": float(posterior.entropy[mutation, region]),
                        }
                    )
            rows.append(row)
    return TableView(_MUTATION_COLUMNS, tuple(rows))


_ATTEMPT_COLUMNS = (
    "candidate_id",
    "candidate_selected",
    "search_round",
    "search_phase",
    "source",
    "start_value",
    "breakpoint_escape_changed_count",
    "mathematically_certified",
    "outer_max_iter",
    "inner_max_iter",
    "certificate_max_iter",
    "lambda_value",
    "objective",
    "stationarity",
    "edge_subgradient",
    "dual_ball",
    "box",
    "kkt_residual",
    "kkt_tolerance",
    "certificate_status",
    "certificate_admissible",
    "working_dtype",
    "audit_dtype",
    "precision_polished",
    "precision_polish_delta",
    "promotion_status",
    "mm_consistency_violations",
    "stage_outer_iterations",
    "stage_outer_max_iter",
    "stage_inner_iterations",
    "stage_inner_max_iter",
    "stage_inner_solve_calls",
    "stop_reason",
    "progress_residual_method",
    "solve_tolerance",
    "legacy_stop_kkt_residual",
    "componentwise_stop_kkt_residual",
    "accepted_full_steps",
    "accepted_damped_steps",
    "rejected_outer_steps",
    "work_inner_iterations",
    "work_inner_stationarity_checks",
    "work_inner_full_kkt_audits",
    "work_outer_kkt_audits",
    "work_certificate_iterations",
    "work_certificate_full_graph_passes",
    "work_partition_refit_coordinates",
    "work_partition_refit_objective_evaluations",
    "work_edge_pass_equivalents",
    "work_edge_region_visits",
    "device",
    "dtype",
    "objective_spec_hash",
    "original_graph_hash",
    "certificate_problem_hash",
    "fallback_reason",
)


def _attempt_table(analysis: AnalysisView) -> TableView:
    rows: list[dict[str, object]] = []
    report = analysis.outcome.report
    for item in report.records:
        for attempt in item.trace.raw_attempts:
            audit = attempt.fit
            certificate = audit.certificate
            components = certificate.components
            convergence = audit.convergence
            work = audit.work
            provenance = audit.provenance
            limits = attempt.limits
            row = {
                "candidate_id": int(item.candidate_id),
                "candidate_selected": bool(
                    report.selected_id is not None
                    and int(item.candidate_id) == int(report.selected_id)
                ),
                "search_round": int(item.trace.search_round),
                "search_phase": str(item.trace.search_phase),
                "source": attempt.source,
                "start_value": attempt.start_value,
                "breakpoint_escape_changed_count": (
                    attempt.breakpoint_escape_changed_count
                ),
                "mathematically_certified": attempt.mathematically_certified,
                "outer_max_iter": limits.outer_max_iter,
                "inner_max_iter": limits.inner_max_iter,
                "certificate_max_iter": limits.certificate_max_iter,
                "lambda_value": provenance.lambda_value,
                "objective": audit.objective.total,
                "stationarity": components.stationarity,
                "edge_subgradient": components.edge_subgradient,
                "dual_ball": components.dual_ball,
                "box": components.box,
                "kkt_residual": components.residual,
                "kkt_tolerance": certificate.tolerance,
                "certificate_status": certificate.status,
                "certificate_admissible": certificate.admissible,
                "working_dtype": certificate.working_dtype,
                "audit_dtype": certificate.audit_dtype,
                "precision_polished": certificate.precision_polished,
                "precision_polish_delta": certificate.precision_polish_delta,
                "promotion_status": attempt.promotion_status,
                "mm_consistency_violations": (convergence.mm_consistency_violations),
                **{
                    name: getattr(convergence, name)
                    for name in (
                        "stage_outer_iterations",
                        "stage_outer_max_iter",
                        "stage_inner_iterations",
                        "stage_inner_max_iter",
                        "stage_inner_solve_calls",
                        "stop_reason",
                        "progress_residual_method",
                        "solve_tolerance",
                        "legacy_stop_kkt_residual",
                        "componentwise_stop_kkt_residual",
                        "accepted_full_steps",
                        "accepted_damped_steps",
                        "rejected_outer_steps",
                    )
                },
                **{
                    f"work_{field.name}": getattr(work, field.name)
                    for field in fields(work)
                },
                "device": provenance.device,
                "dtype": provenance.dtype,
                "objective_spec_hash": provenance.objective_spec_hash,
                "original_graph_hash": provenance.original_graph_hash,
                "certificate_problem_hash": provenance.certificate_problem_hash,
                "fallback_reason": certificate.fallback_reason,
            }
            if set(row) != set(_ATTEMPT_COLUMNS):
                raise AssertionError("Attempt reporting projection is incomplete.")
            rows.append(row)
    return TableView(_ATTEMPT_COLUMNS, tuple(rows))


def _dominant_kkt_component(raw_fit: RawFit | None) -> str:
    if raw_fit is None:
        return "not_available"
    components = raw_fit.certificate.components
    values = {
        "stationarity": float(components.stationarity),
        "edge_subgradient": float(components.edge_subgradient),
        "dual_ball": float(components.dual_ball),
        "box": float(components.box),
    }
    finite = {name: value for name, value in values.items() if np.isfinite(value)}
    return max(finite, key=finite.get) if finite else "not_available"


def _region_rows(
    analysis: AnalysisView,
    prepared: _PreparedAnalysis,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for region, region_id in enumerate(analysis.data.region_ids):
        fraction = (
            0.0
            if prepared.identified is None or prepared.identified.shape[0] == 0
            else float(np.mean(prepared.identified[:, region]))
        )
        tier = _analysis_tier(analysis)
        if analysis.selected_candidate is not None and fraction == 0.0:
            tier = "structural_representative_only"
        diagnostic = analysis.diagnostic_raw_fit
        rows.append(
            {
                "tumor_id": analysis.data.tumor_id,
                "region_id": str(region_id),
                "region_order": region,
                "analysis_tier": tier,
                "estimate_source": "none"
                if analysis.partition is None
                else str(analysis.partition.source),
                "primary_estimator_available": analysis.primary_estimator_available,
                "joint_objective_certified": analysis.primary_estimator_available,
                "regional_conditional_certified": bool(
                    analysis.refit is not None
                    and analysis.refit.global_optimum_certified
                ),
                "count_available_count": int(np.sum(prepared.available[:, region])),
                "count_available_fraction": float(
                    np.mean(prepared.available[:, region])
                ),
                "likelihood_supported_count": int(
                    np.sum(prepared.supported[:, region])
                ),
                "likelihood_supported_fraction": float(
                    np.mean(prepared.supported[:, region])
                ),
                "likelihood_included_count": int(np.sum(prepared.included[:, region])),
                "likelihood_included_fraction": float(
                    np.mean(prepared.included[:, region])
                ),
                "identified_cluster_fraction": fraction,
                "stationarity_residual": None,
                "kkt_tolerance": None
                if diagnostic is None
                else float(diagnostic.certificate.tolerance),
                "kkt_ratio": None,
                "dominant_failure_component": "none"
                if analysis.primary_estimator_available
                else _dominant_kkt_component(diagnostic),
                "rescue_attempted": False,
                "rescue_stage": "conditional_partition_refit"
                if analysis.selected_candidate is not None
                and not analysis.primary_estimator_available
                else "none",
                "failure_reason": analysis.failure_reason,
            }
        )
    return tuple(rows)


def _raw_summary(raw_fit: RawFit | None) -> dict[str, object]:
    if raw_fit is None:
        return {
            "diagnostic_best_raw_lambda": None,
            "diagnostic_best_raw_objective": None,
            "diagnostic_best_raw_kkt_residual": None,
            "diagnostic_best_raw_kkt_tolerance": None,
            "diagnostic_best_raw_certificate_status": None,
        }
    certificate = raw_fit.certificate
    return {
        "diagnostic_best_raw_lambda": float(raw_fit.provenance.lambda_value),
        "diagnostic_best_raw_objective": float(raw_fit.objective.total),
        "diagnostic_best_raw_kkt_residual": float(certificate.components.residual),
        "diagnostic_best_raw_kkt_tolerance": float(certificate.tolerance),
        "diagnostic_best_raw_certificate_status": str(certificate.status),
    }


def _terminal_box_summary(
    raw_fit: RawFit | None,
    model: ObservedModel,
) -> dict[str, object]:
    """Report the persisted box residual and its source-space maximum."""

    empty = {
        "terminal_box_residual": None,
        "terminal_max_lower_violation": None,
        "terminal_max_upper_violation": None,
        "terminal_box_violation_flat_index": None,
    }
    if raw_fit is None:
        return empty
    phi = np.asarray(raw_fit.phi, dtype=np.float64)
    lower = np.asarray(model.lower, dtype=np.float64)
    upper = np.asarray(model.upper, dtype=np.float64)
    if phi.shape != lower.shape or phi.shape != upper.shape:
        raise ValueError("Terminal raw fit and reporting box have different shapes.")
    lower_violation = np.maximum(lower - phi, 0.0)
    upper_violation = np.maximum(phi - upper, 0.0)
    violation = np.maximum(lower_violation, upper_violation)
    maximum = float(np.max(violation)) if violation.size else 0.0
    return {
        "terminal_box_residual": float(raw_fit.certificate.components.box),
        "terminal_max_lower_violation": (
            float(np.max(lower_violation)) if lower_violation.size else 0.0
        ),
        "terminal_max_upper_violation": (
            float(np.max(upper_violation)) if upper_violation.size else 0.0
        ),
        "terminal_box_violation_flat_index": (
            None if maximum == 0.0 else int(np.argmax(violation))
        ),
    }


def _search_work_summary(report: SearchReport) -> dict[str, int]:
    """Project the search's authoritative work ledger."""

    authoritative = report.search_work
    return {
        f"search_work_{item.name}": int(getattr(authoritative, item.name))
        for item in fields(authoritative)
    }


def _candidate_count_summary(report: SearchReport) -> dict[str, int]:
    """Keep raw certification distinct from direct-refit eligibility."""

    raw_records = tuple(record for record in report.records if record.family == "raw_fusion")
    direct_records = tuple(
        record for record in report.records if record.family == "direct_partition"
    )
    return {
        "num_raw_candidates": len(raw_records),
        "num_raw_solver_attempts": sum(
            len(record.trace.raw_attempts) for record in raw_records
        ),
        "num_raw_objective_certified": sum(
            bool(record.candidate.raw_objective_certified) for record in raw_records
        ),
        "num_raw_partition_certified": sum(
            bool(record.candidate.partition.certified) for record in raw_records
        ),
        "num_direct_candidates": len(direct_records),
        "num_direct_refit_eligible": sum(
            bool(record.eligible_for_selection) for record in direct_records
        ),
        "num_selection_eligible_candidates": sum(
            bool(record.eligible_for_selection) for record in report.records
        ),
    }


def _scalar_refit_budget_summary(
    report: SearchReport,
    search_work: dict[str, int],
) -> dict[str, int | str]:
    """Separate objective-defining guide work from the post-guide cap ledger."""

    mandatory = report.mandatory_guide_work
    output: dict[str, int | str] = {
        "max_partition_refit_objective_evaluations_scope": ("post_mandatory_guide")
    }
    for field_name in (
        "partition_refit_coordinates",
        "partition_refit_objective_evaluations",
    ):
        total = int(search_work[f"search_work_{field_name}"])
        guide = int(getattr(mandatory, field_name))
        if guide > total:
            raise ValueError("Mandatory guide work exceeds total search work.")
        output[f"mandatory_guide_{field_name}"] = guide
        output[f"post_guide_{field_name}"] = int(total - guide)
    return output


def analysis_summary(
    analysis: AnalysisView,
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Serialize the analysis summary without changing estimator state."""

    data = analysis.data
    fit_config = analysis.fit_config
    outcome = analysis.outcome
    candidate = analysis.selected_candidate
    partition = analysis.partition
    refit = analysis.refit
    score = analysis.score
    raw_reference = analysis.raw_reference
    raw_fit = analysis.raw_fit
    attempted_raw_fit = analysis.diagnostic_raw_fit
    parent_raw = analysis.partition_parent_raw
    primary = bool(analysis.primary_estimator_available)
    report = outcome.report
    optimum_resolved = bool(report.selection_optimum_resolved)
    selected_lambda = outcome.selected_lambda_representative
    analysis_tier = (
        "joint_certified"
        if primary
        else (
            "conditional_partition_refit"
            if candidate is not None
            else "unsupported_or_unidentified"
        )
    )
    summary: dict[str, object] = {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "ccf_cluster_ordering_method": CCF_CLUSTER_ORDERING_METHOD,
        "tumor_id": data.tumor_id,
        "input_file": str(analysis.input_file),
        "computation_profile": str(fit_config.profile_name),
        "analysis_tier": analysis_tier,
        "primary_estimator_available": primary,
        "failure_reason": analysis.failure_reason,
        "selection_status": (
            ("resolved" if optimum_resolved else "provisional_unresolved")
            if primary
            else analysis_tier
        ),
        "selection_policy_id": PRODUCTION_SELECTION_POLICY.policy_id,
        "selection_optimum_resolved": optimum_resolved,
        "selection_boundary_unresolved": bool(report.selection_boundary_unresolved),
        "selection_pool_stop_reason": str(report.selection_pool_stop_reason),
        "selection_hits_lower_boundary": bool(report.selection_hits_lower_boundary),
        "selection_hits_upper_boundary": bool(report.selection_hits_upper_boundary),
        "selected_lambda": (
            None if selected_lambda is None else float(selected_lambda)
        ),
        "raw_reference_lambda": (
            None if raw_fit is None else float(raw_fit.provenance.lambda_value)
        ),
        "raw_reference_objective_certified": bool(
            raw_reference is not None and raw_reference.raw_objective_certified
        ),
        "selected_candidate_family": (
            None
            if candidate is None
            else (
                "raw_fusion"
                if isinstance(candidate, RawFusionCandidate)
                else "direct_partition"
            )
        ),
        "selected_partition_source": (
            None if partition is None else str(partition.source)
        ),
        "selected_partition_parent_lambda": (
            float(partition.parent_raw_lambda)
            if isinstance(partition, DirectPartition)
            and partition.parent_raw_lambda is not None
            else None
        ),
        "selected_partition_parent_phi_hash": (
            str(partition.parent_raw_phi_hash)
            if isinstance(partition, DirectPartition) and partition.parent_raw_phi_hash
            else ""
        ),
        "selected_partition_parent_signature": (
            str(parent_raw.partition.signature) if parent_raw is not None else ""
        ),
        "selected_n_clusters": (
            None if partition is None else int(partition.n_clusters)
        ),
        "selected_partition_signature": (
            "" if partition is None else str(partition.signature)
        ),
        "selected_partition_certified": bool(
            isinstance(partition, FusionPartition) and partition.certified
        ),
        "selected_labels_hash": (
            ""
            if partition is None
            else _array_fingerprint(partition.labels, dtype=np.dtype(np.int64))
        ),
        "raw_reference_phi_hash": (
            ""
            if raw_fit is None
            else _array_fingerprint(raw_fit.phi, dtype=np.dtype(np.float64))
        ),
        "selected_fixed_partition_refit_centers_hash": (
            ""
            if refit is None
            else _array_fingerprint(refit.cluster_centers, dtype=np.dtype(np.float64))
        ),
        "selection_score_name": None if score is None else str(score.name),
        "selection_score": None if score is None else float(score.value),
        "selection_score_numerical_uncertainty": (
            None if score is None else float(score.numerical_uncertainty)
        ),
        "selection_loglik": None if score is None else float(score.loglik),
        "selection_df": None if score is None else int(score.degrees_of_freedom),
        "selection_penalty": None if score is None else float(score.penalty),
        "selection_n_eff": None if score is None else int(score.n_eff),
        "selected_raw_penalized_objective": (
            None if raw_fit is None else float(raw_fit.objective.total)
        ),
        "selected_refit_numerically_resolved": bool(
            refit is not None and refit.refit_numerically_resolved
        ),
        "selected_refit_global_optimum_certified": bool(
            refit is not None and refit.global_optimum_certified
        ),
        "selected_refit_global_optimality_gap": (
            None if refit is None else float(refit.global_optimality_gap)
        ),
        "selected_refit_global_lower_bound": (
            None if refit is None else float(refit.global_lower_bound)
        ),
        "selected_refit_global_certificate_method": (
            None if refit is None else str(refit.global_certificate_method)
        ),
        "selected_raw_solver_primal_tol": float(fit_config.solver.tolerance),
        "stagnation_audit_patience": int(fit_config.solver.stagnation_audit_patience),
        "lambda_no_progress_patience": int(
            fit_config.selection.lambda_search.no_progress_patience
        ),
        "max_tumor_edge_pass_equivalents": (
            fit_config.solver.resources.max_tumor_edge_pass_equivalents
        ),
        "max_partition_refit_objective_evaluations": (
            fit_config.solver.resources.max_partition_refit_objective_evaluations
        ),
        "max_direct_partition_candidates": (
            fit_config.solver.resources.max_direct_partition_candidates
        ),
        "selected_full_kkt_tolerance": (
            None if raw_fit is None else float(raw_fit.certificate.tolerance)
        ),
        "selected_full_kkt_residual_method": (
            None if raw_fit is None else str(raw_fit.certificate.residual_method)
        ),
        "selected_working_precision_kkt_residual": (
            None if raw_fit is None else float(raw_fit.certificate.working_residual)
        ),
        "selected_working_dtype": (
            None if raw_fit is None else str(raw_fit.certificate.working_dtype)
        ),
        "selected_certificate_audit_dtype": (
            None if raw_fit is None else str(raw_fit.certificate.audit_dtype)
        ),
        "selected_precision_polish_applied": bool(
            raw_fit is not None and raw_fit.certificate.precision_polished
        ),
        "selected_precision_polish_max_abs_phi_delta": (
            None
            if raw_fit is None
            else float(raw_fit.certificate.precision_polish_delta)
        ),
        "observed_model_hash": str(analysis.prepared.model.fingerprint),
        "observed_likelihood_hash": str(analysis.prepared.model.likelihood_fingerprint),
        "reporting_model_hash": str(analysis.prepared.model.reporting_fingerprint),
        "selected_objective_spec_hash": (
            "" if raw_fit is None else str(raw_fit.provenance.objective_spec_hash)
        ),
        "selected_original_graph_hash": (
            "" if raw_fit is None else str(raw_fit.provenance.original_graph_hash)
        ),
        "selected_raw_reference_objective_spec_hash": (
            "" if raw_fit is None else str(raw_fit.provenance.objective_spec_hash)
        ),
        "selected_raw_reference_original_graph_hash": (
            "" if raw_fit is None else str(raw_fit.provenance.original_graph_hash)
        ),
        "attempted_objective_spec_hash": (
            ""
            if attempted_raw_fit is None
            else str(attempted_raw_fit.provenance.objective_spec_hash)
        ),
        "attempted_original_graph_hash": (
            ""
            if attempted_raw_fit is None
            else str(attempted_raw_fit.provenance.original_graph_hash)
        ),
        "attempted_observed_likelihood_hash": (
            ""
            if attempted_raw_fit is None
            else str(attempted_raw_fit.provenance.objective_key.base.likelihood_hash)
        ),
        "attempted_objective_box_hash": (
            ""
            if attempted_raw_fit is None
            else str(attempted_raw_fit.provenance.objective_key.base.box_hash)
        ),
        "selection_method": str(report.selection_method),
        "num_candidates": int(report.num_candidates),
        "num_candidates_certified": int(report.num_candidates_certified),
        "ward_candidate_pool_complete": bool(report.ward_candidate_pool_complete),
        "raw_lambda_path_complete": bool(report.raw_lambda_path_resolved),
        "global_hybrid_optimum_certified": bool(report.global_hybrid_optimum_certified),
        "selected_kkt_residual": outcome.selected_kkt_residual,
        "search_stop_reason": str(report.adaptive_search_stop_reason),
        "device": None if raw_fit is None else str(raw_fit.provenance.device),
        "dtype": None if raw_fit is None else str(raw_fit.provenance.dtype),
        "count_available_fraction": float(np.mean(data.count_available)),
        "likelihood_supported_fraction": float(np.mean(data.likelihood_supported)),
        "likelihood_included_fraction": float(np.mean(data.objective_inclusion_mask())),
        "elapsed_seconds": float(elapsed_seconds),
        "elapsed_seconds_scope": "current_process_segment",
        "cumulative_search_active_seconds": float(
            report.cumulative_search_active_seconds
        ),
        "resumed_from_checkpoint": bool(report.resumed_from_checkpoint),
        "software_version": SOFTWARE_VERSION,
    }
    summary.update(_raw_summary(analysis.diagnostic_raw_fit))
    summary.update(
        _terminal_box_summary(
            analysis.diagnostic_raw_fit,
            analysis.prepared.model,
        )
    )
    summary.update(_candidate_count_summary(report))
    search_work_summary = _search_work_summary(report)
    summary.update(search_work_summary)
    summary.update(_scalar_refit_budget_summary(report, search_work_summary))
    summary["regions"] = list(analysis.regions)
    return summary


__all__ = [
    "AnalysisView",
    "CCF_CLUSTER_ORDERING_METHOD",
    "OUTPUT_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "TableView",
    "analysis_summary",
    "ccf_cluster_order",
]
