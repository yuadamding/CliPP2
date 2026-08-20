from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from ..core.bic import (
    cluster_sizes_from_labels,
    compute_dirichlet_exact_partition_log_mass,
    fixed_partition_bic,
    fixed_partition_dirichlet_score,
)
from ..config import FitConfig
from ..core.model import fit_fixed_objective
from ..core.fusion.types import RawFit
from ..core.fusion.partition_starts import PartitionCandidate
from ..core.scalar import (
    PartitionRefitResult,
    canonical_partition_labels as _canonical_partition_labels,
    partition_constrained_observed_refit,
)
from ..core.fusion.types import SolverState
from ..io.data import TumorData
from .partitions import (
    _partition_signature,
    extract_certified_fusion_partition,
    extract_connected_component_partition,
)
from .scoring import (
    _effective_bic_partition_tol,
    _normalize_selection_score_name,
)
from .types import (
    DirectPartition,
    DirectPartitionCandidate,
    FusionPartition,
    PartitionRefitSummary,
    RawFusionCandidate,
    SelectionScore,
    SelectablePartitionCandidate,
    StartArray,
)


def validate_candidate_identity(candidate: SelectablePartitionCandidate) -> None:
    """Fail fast when partition, refit, score, or estimator identity diverges."""

    partition = candidate.partition
    refit = candidate.refit
    score = candidate.score
    if not np.array_equal(partition.labels, refit.labels):
        raise AssertionError("Fixed-partition refit changed selected labels.")
    actual_signature = _partition_signature(
        partition.labels,
        partition.mutation_ids if partition.mutation_ids else None,
    )
    if partition.signature != actual_signature:
        raise AssertionError("Partition signature does not match its labels.")
    if int(partition.n_clusters) != int(np.unique(partition.labels).size):
        raise AssertionError("Partition cluster count is inconsistent.")
    if refit.partition_signature != partition.signature:
        raise AssertionError("Refit partition signature does not match partition.")
    if score.partition_signature != partition.signature:
        raise AssertionError("Selection score does not match raw partition.")
    if str(score.name).startswith("clonal_"):
        raise AssertionError("Clonal-anchor selection scores were removed.")
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
    dirichlet_score = score.name == "fixed_partition_dirichlet_score"
    if dirichlet_score:
        expected_log_evidence = compute_dirichlet_exact_partition_log_mass(
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
    else:
        expected_assignment_penalty = 0.0
        if not np.isclose(
            float(score.assignment_penalty), 0.0, rtol=0.0, atol=score_tolerance
        ):
            raise AssertionError("BIC score contains an ICL allocation penalty.")
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
    expected_phi = np.asarray(refit.cluster_centers)[np.asarray(refit.labels)]
    if not np.allclose(np.asarray(refit.phi), expected_phi, rtol=0.0, atol=1e-12):
        raise AssertionError("Refit phi does not match centers indexed by labels.")
    if isinstance(candidate, RawFusionCandidate):
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
    require_global_refit: bool = True,
) -> str:
    if isinstance(partition, FusionPartition):
        if raw_fit is None:
            raise ValueError("Raw-fusion eligibility requires its raw fit.")
        if float(raw_fit.provenance.lambda_value) <= 0.0:
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
    resolved = (
        refit.refit_numerically_resolved
        if isinstance(partition, FusionPartition)
        else refit.global_optimum_certified
    )
    if require_global_refit and not resolved:
        return "fixed_partition_refit_numerically_unresolved"
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


def _likelihood_model_id(data: TumorData) -> str:
    path = getattr(data, "path_likelihood", None)
    return "legacy_major_minor_mixture_v1" if path is None else str(path.model_id)


def _selection_refit_cache_key(
    *,
    data: TumorData,
    partition_signature: str,
    selection_options: FitConfig,
) -> tuple[object, ...]:
    refit = selection_options.selection.refit
    return (
        str(partition_signature),
        float(selection_options.major_prior),
        float(selection_options.eps),
        float(refit.tolerance),
        int(refit.max_iter),
        _likelihood_model_id(data),
        str(refit.mode),
        int(refit.grid_points),
        int(refit.local_steps),
        "unanchored_profiled_partition_refit_v4",
    )


def _fixed_labels_refit(
    *,
    data: TumorData,
    labels: np.ndarray,
    partition_signature: str,
    selection_options: FitConfig,
    cache: dict[object, PartitionRefitCacheEntry] | None,
) -> PartitionRefitCacheEntry:
    profile = selection_options.computation_profile
    refit_config = selection_options.selection.refit
    refit_spec_key = _selection_refit_cache_key(
        data=data,
        partition_signature=partition_signature,
        selection_options=selection_options,
    )
    if cache is not None and refit_spec_key in cache:
        return cache[refit_spec_key]

    kwargs = dict(
        major_prior=float(selection_options.major_prior),
        eps=float(selection_options.eps),
        tol=float(refit_config.tolerance),
        max_iter=int(refit_config.max_iter),
        scalar_mode=str(refit_config.mode),
        scalar_grid_points=int(refit_config.grid_points),
        scalar_local_steps=int(refit_config.local_steps),
    )
    refined = partition_constrained_observed_refit(
        data,
        np.asarray(labels, dtype=np.int64),
        **kwargs,
    )
    loglik_delta = float(refined.global_optimality_gap)
    loglik_tolerance = max(
        float(refit_config.tolerance)
        * (1.0 + abs(float(refined.loglik))),
        1e-10,
    )
    numerically_resolved = bool(
        refined.finite_candidate_found
        and int(refined.refit_finite_coordinate_count)
        == int(refined.refit_coordinate_count)
        and (
            (
                refined.global_optimum_certified
                and np.isfinite(loglik_delta)
                and loglik_delta <= loglik_tolerance
            )
            if profile.is_strict
            else np.isfinite(float(refined.loglik))
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
) -> PartitionRefitSummary:
    return PartitionRefitSummary(
        labels=np.asarray(refit.labels, dtype=np.int64).copy(),
        partition_signature=str(partition_signature),
        phi=np.asarray(refit.phi, dtype=np.float64).copy(),
        cluster_centers=np.asarray(refit.cluster_centers, dtype=np.float64).copy(),
        loglik=float(refit.loglik),
        finite_candidate_found=bool(refit.finite_candidate_found),
        global_optimum_certified=bool(refit.global_optimum_certified),
        refit_numerically_resolved=bool(resolution.numerically_resolved),
        global_lower_bound=float(refit.global_lower_bound),
        global_optimality_gap=float(refit.global_optimality_gap),
        global_certificate_method=str(refit.global_certificate_method),
        refit_mode=str(refit.refit_mode),
    )


def _score_fixed_labels(
    *,
    data: TumorData,
    labels: np.ndarray,
    partition_signature: str,
    refit_result: PartitionRefitResult,
    selection_options: FitConfig,
    selection_score: str,
) -> SelectionScore:
    canonical_score_name = _normalize_selection_score_name(selection_score)
    computation_profile = selection_options.computation_profile
    score_function = (
        fixed_partition_dirichlet_score
        if canonical_score_name == "fixed_partition_dirichlet_score"
        else fixed_partition_bic
    )
    score_kwargs: dict[str, object] = {
        "loglik": float(refit_result.loglik),
        "num_clusters": int(np.unique(labels).size),
        "data": data,
        "partition_signature": str(partition_signature),
        "labels": np.asarray(labels, dtype=np.int64),
        "loglik_uncertainty": (
            float(refit_result.global_optimality_gap)
            if computation_profile.is_strict
            else 0.0
        ),
    }
    if canonical_score_name == "fixed_partition_dirichlet_score":
        score_kwargs.update(
            alpha=float(selection_options.selection.dirichlet_alpha),
            code_weight=float(selection_options.selection.dirichlet_code_weight),
        )
    score = score_function(**score_kwargs)
    if str(score.name) != canonical_score_name:
        raise AssertionError(
            f"Requested score {canonical_score_name} produced {score.name}."
        )
    return score


def evaluate_partition(
    *,
    data: TumorData,
    partition: FusionPartition | DirectPartition,
    selection_options: FitConfig,
    refit_cache: dict[object, PartitionRefitCacheEntry] | None,
    selection_score: str | None = None,
) -> PartitionEvaluation:
    """Evaluate raw and direct label sets through one refit/score path."""

    cached_refit = _fixed_labels_refit(
        data=data,
        labels=partition.labels,
        partition_signature=partition.signature,
        selection_options=selection_options,
        cache=refit_cache,
    )
    refit_result = cached_refit.result
    score = _score_fixed_labels(
        data=data,
        labels=partition.labels,
        partition_signature=partition.signature,
        refit_result=refit_result,
        selection_options=selection_options,
        selection_score=(
            selection_options.selection.score
            if selection_score is None
            else selection_score
        ),
    )
    return PartitionEvaluation(
        refit=_build_refit_summary(
            refit_result,
            partition_signature=partition.signature,
            resolution=cached_refit,
        ),
        score=score,
    )


def evaluate_raw_fusion_candidate(
    *,
    data: TumorData,
    fit_options: FitConfig,
    candidate_fit_options: FitConfig | None,
    phi_start: StartArray | None,
    exact_pilot: StartArray | None,
    pooled_start: StartArray | None,
    scalar_well_starts: list[StartArray] | None,
    start_mode: str,
    runtime,
    torch_data,
    solver_context,
    solver_state: SolverState | None,
    lambda_value: float,
    selection_score: str,
    bic_refit_cache: dict[object, PartitionRefitCacheEntry] | None = None,
    precomputed_fit: RawFit | None = None,
) -> tuple[RawFit, RawFusionCandidate]:
    canonical_score_name = _normalize_selection_score_name(selection_score)
    raw_fit_options = (
        fit_options if candidate_fit_options is None else candidate_fit_options
    )
    selection_options = fit_options
    computation_profile = selection_options.computation_profile

    fit = precomputed_fit
    if fit is None:
        fit = fit_fixed_objective(
            data=data,
            config=replace(raw_fit_options, lambda_value=float(lambda_value)),
            phi_start=phi_start,
            exact_pilot=exact_pilot,
            pooled_start=pooled_start,
            scalar_well_starts=scalar_well_starts,
            start_mode=start_mode,
            runtime=runtime,
            torch_data=torch_data,
            solver_context=solver_context,
            solver_state=solver_state,
        )
    elif not np.isclose(
        float(fit.provenance.lambda_value), float(lambda_value), rtol=0.0, atol=1e-12
    ):
        raise ValueError("Precomputed raw fit has the wrong lambda value.")
    graph = selection_options.graph.graph
    if graph is None:
        raise RuntimeError(
            "Model-selection candidates require a resolved pairwise-fusion graph."
        )
    partition_tolerance = _effective_bic_partition_tol(selection_options)
    contract = selection_options.selection.contract
    partition_extractor = (
        extract_connected_component_partition
        if contract.raw_partition_rule == "legacy_connected_components"
        else extract_certified_fusion_partition
    )
    partition = partition_extractor(
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
        selection_score=canonical_score_name,
    )
    refit = evaluation.refit
    score = evaluation.score
    reason = _candidate_ineligibility_reason(
        partition=partition,
        refit=refit,
        score=score,
        raw_fit=fit,
        require_global_refit=bool(computation_profile.is_strict),
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
    selection_options: FitConfig,
    source: str,
    parent_raw_candidate_id: int | None,
    parent_raw_lambda: float | None,
    refit_cache: dict[object, PartitionRefitCacheEntry] | None,
    parent_raw_phi_hash: str = "",
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
    )
    refit = evaluation.refit
    score = evaluation.score
    profile = selection_options.computation_profile
    reason = _candidate_ineligibility_reason(
        partition=partition,
        refit=refit,
        score=score,
        raw_fit=None,
        require_global_refit=bool(profile.is_strict),
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
