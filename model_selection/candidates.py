from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core.bic import (
    cluster_sizes_from_labels,
    _dirichlet_exact_partition_log_mass_and_uncertainty,
    fixed_partition_dirichlet_score,
)
from ..config import DIRICHLET_ALPHA, DIRICHLET_CODE_WEIGHT, _FitOptions, SELECTION_SCORE
from ..core.objective import ObservedModel
from ..core.fusion.types import RawFit
from ..core.fusion.partition_starts import PartitionCandidate
from ..core.scalar import (
    PartitionRefitResult,
    canonical_partition_labels as _canonical_partition_labels,
    partition_constrained_observed_refit,
)
from ..io.data import TumorData, tumor_data_fingerprint
from .partitions import (
    _partition_signature,
    extract_certified_fusion_partition,
)
from .scoring import (
    _effective_bic_partition_tol,
)
from .types import (
    DirectPartition,
    DirectPartitionCandidate,
    FusionPartition,
    PartitionRefitSummary,
    RawFusionCandidate,
    SelectionScore,
    SelectablePartitionCandidate,
    is_zero_edge_singleton,
)


def validate_partition_identity(
    partition: FusionPartition | DirectPartition, refit: PartitionRefitSummary,
) -> None:
    """Check the immutable labels and their stored fixed-label refit once."""
    if not np.array_equal(partition.labels, refit.labels):
        raise AssertionError("Selected partition and fixed refit labels differ.")
    if partition.signature != _partition_signature(partition.labels, partition.mutation_ids or None):
        raise AssertionError("Partition signature does not match its labels.")
    if refit.partition_signature != partition.signature:
        raise AssertionError("Refit partition signature does not match partition.")
    labels = np.asarray(partition.labels)
    if not np.array_equal(np.unique(labels), np.arange(partition.n_clusters)):
        raise AssertionError("Partition labels must be canonical zero-based IDs.")
    centers = np.asarray(refit.cluster_centers)
    if centers.ndim != 2 or centers.shape[0] != partition.n_clusters:
        raise AssertionError("Refit centers do not match partition clusters.")
    expected_phi = centers[labels]
    if refit.phi.shape != expected_phi.shape or not np.allclose(refit.phi, expected_phi, rtol=0.0, atol=1e-12):
        raise AssertionError("Refit phi does not match centers indexed by labels.")


def validate_candidate_identity(candidate: SelectablePartitionCandidate) -> None:
    """Fail fast when partition, refit, score, or estimator identity diverges."""

    partition = candidate.partition
    refit = candidate.refit
    score = candidate.score
    validate_partition_identity(partition, refit)
    if score.partition_signature != partition.signature:
        raise AssertionError("Selection score does not match raw partition.")
    if score.name != SELECTION_SCORE:
        raise AssertionError("Only the fixed-partition Dirichlet score is supported.")
    if (
        float(score.assignment_dirichlet_alpha) != DIRICHLET_ALPHA
        or float(score.assignment_code_weight) != DIRICHLET_CODE_WEIGHT
    ):
        raise AssertionError("Dirichlet score parameters differ from the production policy.")
    score_tolerance = 1e-10 * (1.0 + abs(float(score.value)))
    if not np.isclose(
        float(score.loglik),
        float(refit.loglik),
        rtol=0.0,
        atol=score_tolerance,
    ):
        raise AssertionError("Score likelihood differs from stored refit likelihood.")
    expected_bic_penalty = float(score.degrees_of_freedom) * np.log(
        max(int(score.n_eff), 1)
    )
    expected_log_evidence, _ = _dirichlet_exact_partition_log_mass_and_uncertainty(
        cluster_sizes_from_labels(partition.labels),
        alpha=float(score.assignment_dirichlet_alpha),
    )
    expected_assignment_penalty = float(
        -2.0 * float(score.assignment_code_weight) * expected_log_evidence
    )
    if not np.isclose(
        float(score.assignment_log_evidence),
        expected_log_evidence,
        rtol=0.0,
        atol=score_tolerance,
    ):
        raise AssertionError(
            "Stored Dirichlet allocation mass is not reconstructible."
        )
    if not np.isclose(
        float(score.assignment_penalty),
        expected_assignment_penalty,
        rtol=0.0,
        atol=score_tolerance,
    ):
        raise AssertionError(
            "Stored Dirichlet allocation penalty is not reconstructible."
        )
    if (
        not np.isfinite(float(score.assignment_arithmetic_uncertainty))
        or float(score.assignment_arithmetic_uncertainty) < 0.0
        or float(score.numerical_uncertainty) + score_tolerance
        < float(score.assignment_arithmetic_uncertainty)
    ):
        raise AssertionError(
            "Score uncertainty does not cover Dirichlet arithmetic."
        )
    expected_penalty = expected_bic_penalty + expected_assignment_penalty
    if not np.isclose(
        float(score.penalty), expected_penalty, rtol=0.0, atol=score_tolerance
    ):
        raise AssertionError("Stored selection penalty is not reconstructible.")
    expected_score = -2.0 * float(refit.loglik) + expected_penalty
    if not np.isclose(
        float(score.value), expected_score, rtol=0.0, atol=score_tolerance
    ):
        raise AssertionError("Stored score is not reconstructible.")
    minimum_uncertainty = 2.0 * max(float(refit.global_optimality_gap), 0.0)
    if (
        refit.global_optimum_certified
        and float(score.numerical_uncertainty) + score_tolerance < minimum_uncertainty
    ):
        raise AssertionError(
            "Score uncertainty does not cover the refit certificate gap."
        )
    if isinstance(candidate, RawFusionCandidate):
        if candidate.raw_fit.provenance.multiplicity_policy != refit.multiplicity_policy:
            raise ValueError("Raw fit and refit multiplicity policies differ.")
        if candidate.raw_fit.provenance.source_data_hash != refit.source_data_hash:
            raise ValueError("Raw fit and fixed refit source data identities differ.")
        if candidate.raw_fit.provenance.likelihood_eps != refit.likelihood_eps:
            raise ValueError("Raw fit and fixed refit epsilon identities differ.")
        if candidate.eligible_for_selection and (
            not candidate.raw_objective_certified or not candidate.partition.certified
        ):
            raise AssertionError("Selectable raw candidates require both certificates.")
    elif isinstance(candidate, DirectPartitionCandidate):
        if candidate.partition.mutation_ids != tuple(
            str(value) for value in partition.mutation_ids
        ):
            raise AssertionError("Direct partition mutation identity changed.")
    else:  # pragma: no cover - union exhaustiveness guard
        raise TypeError(f"Unsupported candidate type: {type(candidate)!r}")


def _candidate_ineligibility_reason(
    *,
    partition: FusionPartition | DirectPartition,
    refit: PartitionRefitSummary,
    score: SelectionScore,
    raw_fit: RawFit | None = None,
) -> str:
    if isinstance(partition, FusionPartition):
        if raw_fit is None:
            raise ValueError("Raw-fusion eligibility requires its raw fit.")
        if float(raw_fit.provenance.lambda_value) <= 0.0 and not is_zero_edge_singleton(raw_fit):
            return "nonpositive_lambda"
        if not bool(raw_fit.certificate.certified) or not bool(
            raw_fit.certificate.admissible
        ):
            return "raw_objective_not_kkt_certified"
        if not partition.certified:
            return str(partition.certification_failure_reason)
    else:
        if raw_fit is not None:
            raise ValueError("Direct-partition eligibility cannot inherit a raw fit.")
        if partition.n_clusters < 1:
            return "empty_partition"
    if not refit.finite_candidate_found:
        return "fixed_partition_refit_nonfinite"
    if not np.isfinite(score.value):
        return "fixed_partition_score_nonfinite"
    return "none"


@dataclass(frozen=True)
class PartitionRefitCacheEntry:
    result: PartitionRefitResult
    numerically_resolved: bool


@dataclass(frozen=True, slots=True)
class PartitionEvaluation:
    """Source-neutral fixed-label refit and score for one partition."""

    refit: PartitionRefitSummary
    score: SelectionScore


def _selection_refit_cache_key(
    *,
    data: TumorData,
    partition_signature: str,
    selection_options: _FitOptions,
) -> tuple[object, ...]:
    refit = selection_options.selection.refit
    return (
        str(partition_signature),
        float(selection_options.eps),
        selection_options.multiplicity_policy,
        float(refit.tolerance),
        int(refit.max_iter),
        # A family name does not identify counts, candidate support or priors.
        # Eligibility-only provenance is deliberately absent from this hash.
        tumor_data_fingerprint(data),
        "grid_local",
        int(refit.grid_points),
        int(refit.local_steps),
        "unanchored_profiled_partition_refit_v4",
    )


def _fixed_labels_refit(
    *,
    data: TumorData,
    labels: np.ndarray,
    partition_signature: str,
    selection_options: _FitOptions,
    cache: dict[object, PartitionRefitCacheEntry] | None,
    source_model: ObservedModel | None = None,
) -> PartitionRefitCacheEntry:
    refit_config = selection_options.selection.refit
    refit_spec_key = _selection_refit_cache_key(
        data=data,
        partition_signature=partition_signature,
        selection_options=selection_options,
    )
    if cache is not None and refit_spec_key in cache:
        return cache[refit_spec_key]

    kwargs = dict(
        eps=float(selection_options.eps),
        tol=float(refit_config.tolerance),
        max_iter=int(refit_config.max_iter),
        scalar_grid_points=int(refit_config.grid_points),
        scalar_local_steps=int(refit_config.local_steps),
    )
    refined = partition_constrained_observed_refit(
        data,
        np.asarray(labels, dtype=np.int64),
        _model=source_model,
        **kwargs,
    )
    numerically_resolved = bool(
        refined.finite_candidate_found
        and int(refined.refit_finite_coordinate_count)
        == int(refined.refit_coordinate_count)
        and (
            (np.isfinite(float(refined.loglik)))
        )
    )
    cached = PartitionRefitCacheEntry(
        result=refined,
        numerically_resolved=numerically_resolved,
    )
    if cache is not None:
        cache[refit_spec_key] = cached
    return cached


def _build_refit_summary(
    refit: PartitionRefitResult,
    *,
    partition_signature: str,
    resolution: PartitionRefitCacheEntry,
    data: TumorData,
    eps: float,
) -> PartitionRefitSummary:
    return PartitionRefitSummary(
        labels=np.asarray(refit.labels, dtype=np.int64).copy(),
        partition_signature=str(partition_signature),
        phi=np.asarray(refit.phi, dtype=np.float64).copy(),
        cluster_centers=np.asarray(refit.cluster_centers, dtype=np.float64).copy(),
        loglik=float(refit.loglik),
        finite_candidate_found=bool(refit.finite_candidate_found),
        global_optimum_certified=bool(refit.global_optimum_certified),
        source_data_hash=tumor_data_fingerprint(data),
        likelihood_eps=float(eps),
        refit_numerically_resolved=bool(resolution.numerically_resolved),
        global_lower_bound=float(refit.global_lower_bound),
        global_optimality_gap=float(refit.global_optimality_gap),
        global_certificate_method=str(refit.global_certificate_method),
        refit_mode=str(refit.refit_mode),
        multiplicity_policy="independent_broad",
        locally_converged=None,
    )


def _score_fixed_labels(
    *,
    data: TumorData,
    labels: np.ndarray,
    partition_signature: str,
    refit_result: PartitionRefitResult,
    selection_options: _FitOptions,
) -> SelectionScore:
    return fixed_partition_dirichlet_score(
        loglik=float(refit_result.loglik),
        num_clusters=int(np.unique(labels).size),
        data=data,
        partition_signature=str(partition_signature),
        labels=np.asarray(labels, dtype=np.int64),
        loglik_uncertainty=(
            (0.0)
        ),
        alpha=float(selection_options.selection.dirichlet_alpha),
        code_weight=float(selection_options.selection.dirichlet_code_weight),
    )


def evaluate_partition(
    *,
    data: TumorData,
    partition: FusionPartition | DirectPartition,
    selection_options: _FitOptions,
    refit_cache: dict[object, PartitionRefitCacheEntry] | None,
    source_model: ObservedModel | None = None,
) -> PartitionEvaluation:
    """Evaluate raw and direct label sets through one refit/score path."""

    cached_refit = _fixed_labels_refit(
        data=data,
        labels=partition.labels,
        partition_signature=partition.signature,
        selection_options=selection_options,
        cache=refit_cache,
        source_model=source_model,
    )
    refit_result = cached_refit.result
    score = _score_fixed_labels(
        data=data,
        labels=partition.labels,
        partition_signature=partition.signature,
        refit_result=refit_result,
        selection_options=selection_options,
    )
    return PartitionEvaluation(
        refit=_build_refit_summary(
            refit_result,
            partition_signature=partition.signature,
            resolution=cached_refit,
            data=data,
            eps=selection_options.eps,
            ),
        score=score,
    )


def evaluate_raw_fusion_candidate(
    *,
    data: TumorData,
    fit_options: _FitOptions,
    lambda_value: float,
    precomputed_fit: RawFit,
    bic_refit_cache: dict[object, PartitionRefitCacheEntry] | None = None,
    source_model: ObservedModel | None = None,
) -> tuple[RawFit, RawFusionCandidate]:
    selection_options = fit_options

    fit = precomputed_fit
    if not np.isclose(
        float(fit.provenance.lambda_value), float(lambda_value), rtol=0.0, atol=1e-12
    ):
        raise ValueError("Precomputed raw fit has the wrong lambda value.")
    graph = selection_options.graph.graph
    if graph is None:
        raise RuntimeError(
            "Model-selection candidates require a resolved pairwise-fusion graph."
        )
    if fit.provenance.original_graph_hash != graph.fingerprint:
        raise ValueError("Precomputed raw fit belongs to a different fusion graph.")
    partition_tolerance = _effective_bic_partition_tol(selection_options)
    partition = extract_certified_fusion_partition(
        fit,
        graph=graph,
        tolerance=partition_tolerance,
        mutation_ids=tuple(str(value) for value in data.mutation_ids),
    )

    evaluation = evaluate_partition(
        data=data,
        partition=partition,
        selection_options=selection_options,
        refit_cache=bic_refit_cache,
        source_model=source_model,
    )
    refit = evaluation.refit
    score = evaluation.score
    reason = _candidate_ineligibility_reason(
        partition=partition,
        refit=refit,
        score=score,
        raw_fit=fit,
    )
    candidate = RawFusionCandidate(
        raw_fit=fit,
        partition=partition,
        refit=refit,
        score=score,
        eligible_for_selection=reason == "none",
        ineligibility_reason=reason,
    )
    validate_candidate_identity(candidate)

    return fit, candidate


def evaluate_direct_partition_candidate(
    *,
    data: TumorData,
    proposal: PartitionCandidate,
    selection_options: _FitOptions,
    source: str,
    parent_raw_candidate_id: int | None,
    parent_raw_lambda: float | None,
    refit_cache: dict[object, PartitionRefitCacheEntry] | None,
    parent_raw_phi_hash: str = "",
    source_model: ObservedModel | None = None,
) -> DirectPartitionCandidate:
    """Evaluate one deterministic non-fusion partition under the common score."""

    labels = _canonical_partition_labels(
        np.asarray(proposal.labels, dtype=np.int64).reshape(-1)
    )
    mutation_ids = tuple(str(value) for value in data.mutation_ids)
    signature = _partition_signature(labels, mutation_ids)
    partition = DirectPartition(
        labels=labels,
        signature=signature,
        source=source,
        mutation_ids=mutation_ids,
        parent_raw_candidate_id=parent_raw_candidate_id,
        parent_raw_lambda=parent_raw_lambda,
        parent_raw_phi_hash=str(parent_raw_phi_hash),
    )
    evaluation = evaluate_partition(
        data=data,
        partition=partition,
        selection_options=selection_options,
        refit_cache=refit_cache,
        source_model=source_model,
    )
    refit = evaluation.refit
    score = evaluation.score
    reason = _candidate_ineligibility_reason(
        partition=partition,
        refit=refit,
        score=score,
        raw_fit=None,
    )
    candidate = DirectPartitionCandidate(
        partition=partition,
        refit=refit,
        score=score,
        eligible_for_selection=reason == "none",
        ineligibility_reason=reason,
    )
    validate_candidate_identity(candidate)
    return candidate


__all__ = [
    "PartitionEvaluation",
    "PartitionRefitCacheEntry",
    "evaluate_direct_partition_candidate",
    "evaluate_partition",
    "evaluate_raw_fusion_candidate",
    "validate_candidate_identity",
]
