from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from ..core.bic import (
    DIRICHLET_EXACT_PARTITION_MODEL_ID,
    cluster_sizes_from_labels,
    compute_bic_with_df,
    compute_dirichlet_exact_partition_log_mass,
    effective_bic_depth_count,
    fixed_partition_bic,
    fixed_partition_dirichlet_score,
)
from ..core.model import FitOptions, FitResult, fit_fixed_objective
from ..core.fusion.partition_starts import PartitionCandidate
from ..core.fusion.refit import (
    PartitionRefitResult,
    _canonical_labels as _canonical_partition_labels,
    partition_constrained_observed_refit,
)
from ..core.fusion.profiles import get_computation_profile
from ..core.fusion.types import SolverState
from ..io.data import TumorData
from .partitions import (
    _cluster_sizes_text,
    _partition_signature,
    extract_certified_fusion_partition,
    extract_connected_component_partition,
)
from .contracts import get_selection_contract
from .scoring import (
    _effective_bic_partition_tol,
    _normalize_selection_score_name,
    _profile_penalty_from_fit,
)
from .types import (
    CandidateStaticMetadata,
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
    if int(score.degrees_of_freedom) != int(refit.nominal_df):
        raise AssertionError("Score and refit degrees of freedom differ.")
    expected_bic_penalty = float(score.degrees_of_freedom) * np.log(
        max(int(score.n_eff), 1)
    )
    dirichlet_score = bool(
        str(score.assignment_model_id) == DIRICHLET_EXACT_PARTITION_MODEL_ID
        or score.name
        in {
            "fixed_partition_dirichlet_score",
        }
    )
    if dirichlet_score:
        contract = get_selection_contract(score.selection_contract_id)
        expected_weight = float(contract.partition_config.classification_code_weight)
        expected_alpha = float(contract.partition_config.classification_alpha)
        if not np.isclose(
            float(score.assignment_code_weight), expected_weight, rtol=0.0, atol=0.0
        ):
            raise AssertionError("Dirichlet code weight differs from its contract.")
        if not np.isclose(
            float(score.assignment_dirichlet_alpha), expected_alpha, rtol=0.0, atol=0.0
        ):
            raise AssertionError("Dirichlet alpha differs from its contract.")
        if str(score.assignment_model_id) != DIRICHLET_EXACT_PARTITION_MODEL_ID:
            raise AssertionError("Dirichlet assignment-model provenance is missing.")
        if str(score.assignment_symmetry_mode) != "all_blocks_exchangeable":
            raise AssertionError("Dirichlet partition symmetry is inconsistent.")
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
    strict = str(candidate.computation_profile) == "strict"
    minimum_uncertainty = 2.0 * max(float(refit.global_optimality_gap), 0.0)
    if (
        strict
        and float(score.numerical_uncertainty) + score_tolerance < minimum_uncertainty
    ):
        raise AssertionError(
            "Score uncertainty does not cover the refit certificate gap."
        )
    if (
        strict
        and candidate.eligible_for_selection
        and not refit.global_optimum_certified
    ):
        raise AssertionError("Selectable BIC requires a globally certified refit.")
    expected_phi = np.asarray(refit.cluster_centers)[np.asarray(refit.labels)]
    if not np.allclose(np.asarray(refit.phi), expected_phi, rtol=0.0, atol=1e-12):
        raise AssertionError("Refit phi does not match centers indexed by labels.")
    if isinstance(candidate, RawFusionCandidate):
        if candidate.eligible_for_selection and (
            not candidate.raw_objective_certified or not candidate.partition.certified
        ):
            raise AssertionError("Selectable raw candidates require both certificates.")
    elif isinstance(candidate, DirectPartitionCandidate):
        if not candidate.partition.deterministic_generation:
            raise AssertionError("Direct partition generation is not deterministic.")
        if candidate.partition.mutation_ids != tuple(
            str(value) for value in partition.mutation_ids
        ):
            raise AssertionError("Direct partition mutation identity changed.")
    else:  # pragma: no cover - union exhaustiveness guard
        raise TypeError(f"Unsupported candidate type: {type(candidate)!r}")


def _ineligibility_reason(
    *,
    fit: FitResult,
    partition: FusionPartition,
    refit_finite: bool,
    refit_numerically_resolved: bool,
    score_finite: bool,
    require_global_refit: bool = True,
) -> str:
    if float(fit.lambda_value) <= 0.0:
        return "nonpositive_lambda"
    if str(fit.estimator_role) != "raw_fused_lambda_path":
        return "not_raw_fused_lambda_path"
    if not bool(fit.objective_faithful):
        return "raw_objective_not_faithful"
    if not bool(fit.full_kkt_certified) or not bool(fit.selection_eligible):
        return "raw_objective_not_kkt_certified"
    if not partition.certified:
        return str(partition.certification_failure_reason)
    if not refit_finite:
        return "fixed_partition_refit_nonfinite"
    if require_global_refit and not refit_numerically_resolved:
        return "fixed_partition_refit_numerically_unresolved"
    if not score_finite:
        return "fixed_partition_score_nonfinite"
    return "none"


def _direct_partition_ineligibility_reason(
    *,
    partition: DirectPartition,
    refit: PartitionRefitSummary,
    score: SelectionScore,
    require_global_refit: bool,
) -> str:
    if partition.n_clusters < 1:
        return "empty_partition"
    if not partition.deterministic_generation:
        return "nondeterministic_partition_generation"
    if not refit.finite_candidate_found:
        return "fixed_partition_refit_nonfinite"
    if require_global_refit and not refit.global_optimum_certified:
        return "fixed_partition_refit_numerically_unresolved"
    if not np.isfinite(score.value):
        return "fixed_partition_score_nonfinite"
    return "none"


@dataclass(frozen=True)
class _CachedPartitionRefit:
    result: PartitionRefitResult
    loglik_refinement_delta: float
    max_center_refinement_delta: float
    numerically_resolved: bool


def _likelihood_model_id(data: TumorData) -> str:
    path = getattr(data, "path_likelihood", None)
    return "legacy_major_minor_mixture_v1" if path is None else str(path.model_id)


def _selection_refit_cache_key(
    *,
    data: TumorData,
    partition_signature: str,
    selection_options: FitOptions,
) -> tuple[object, ...]:
    profile = get_computation_profile(selection_options.computation_profile)
    return (
        str(partition_signature),
        float(selection_options.major_prior),
        float(selection_options.eps),
        float(selection_options.selection_refit_tol),
        int(selection_options.selection_refit_max_iter),
        _likelihood_model_id(data),
        str(profile.scalar_mode),
        int(profile.scalar_grid_points),
        int(profile.scalar_local_steps),
        "unanchored_profiled_partition_refit_v4",
    )


def _fixed_labels_refit(
    *,
    data: TumorData,
    labels: np.ndarray,
    partition_signature: str,
    selection_options: FitOptions,
    cache: dict[object, _CachedPartitionRefit] | None,
) -> tuple[_CachedPartitionRefit, bool]:
    profile = get_computation_profile(selection_options.computation_profile)
    refit_spec_key = _selection_refit_cache_key(
        data=data,
        partition_signature=partition_signature,
        selection_options=selection_options,
    )
    if cache is not None and refit_spec_key in cache:
        return cache[refit_spec_key], True

    kwargs = dict(
        major_prior=float(selection_options.major_prior),
        eps=float(selection_options.eps),
        tol=float(selection_options.selection_refit_tol),
        max_iter=int(selection_options.selection_refit_max_iter),
        scalar_mode=str(profile.scalar_mode),
        scalar_grid_points=int(profile.scalar_grid_points),
        scalar_local_steps=int(profile.scalar_local_steps),
    )
    refined = partition_constrained_observed_refit(
        data,
        np.asarray(labels, dtype=np.int64),
        **kwargs,
    )
    loglik_delta = float(refined.global_optimality_gap)
    center_delta = 0.0
    loglik_tolerance = max(
        float(selection_options.selection_refit_tol)
        * (1.0 + abs(float(refined.loglik))),
        1e-10,
    )
    numerically_resolved = bool(
        refined.finite_candidate_found
        and int(refined.refit_finite_coordinate_count)
        == int(refined.refit_coordinate_count)
        and np.isfinite(center_delta)
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
    cached = _CachedPartitionRefit(
        result=refined,
        loglik_refinement_delta=float(loglik_delta),
        max_center_refinement_delta=float(center_delta),
        numerically_resolved=numerically_resolved,
    )
    if cache is not None:
        cache[refit_spec_key] = cached
    return cached, False


def _fixed_partition_refit(
    *,
    data: TumorData,
    partition: FusionPartition,
    selection_options: FitOptions,
    cache: dict[object, _CachedPartitionRefit] | None,
) -> tuple[_CachedPartitionRefit, bool]:
    """Compatibility wrapper around the source-neutral fixed-label refit."""

    return _fixed_labels_refit(
        data=data,
        labels=partition.labels,
        partition_signature=partition.signature,
        selection_options=selection_options,
        cache=cache,
    )


def _build_refit_summary(
    refit: PartitionRefitResult,
    *,
    partition_signature: str,
    nominal_df: int,
    resolution: _CachedPartitionRefit,
) -> PartitionRefitSummary:
    return PartitionRefitSummary(
        labels=np.asarray(refit.labels, dtype=np.int64).copy(),
        partition_signature=str(partition_signature),
        phi=np.asarray(refit.phi, dtype=np.float64).copy(),
        cluster_centers=np.asarray(refit.cluster_centers, dtype=np.float64).copy(),
        loglik=float(refit.loglik),
        fit_loss=float(refit.fit_loss),
        nominal_df=int(nominal_df),
        active_df=int(refit.active_degrees_of_freedom),
        finite_candidate_found=bool(refit.finite_candidate_found),
        global_optimum_certified=bool(refit.global_optimum_certified),
        loglik_source=str(refit.loglik_source),
        refit_numerically_resolved=bool(resolution.numerically_resolved),
        refit_loglik_refinement_delta=float(resolution.loglik_refinement_delta),
        refit_max_center_refinement_delta=float(resolution.max_center_refinement_delta),
        refit_coordinate_count=int(refit.refit_coordinate_count),
        refit_finite_coordinate_count=int(refit.refit_finite_coordinate_count),
        refit_total_grid_points=int(refit.refit_total_grid_points),
        refit_max_grid_spacing=float(refit.refit_max_grid_spacing),
        refit_total_candidate_basins=int(refit.refit_total_candidate_basins),
        refit_total_refined_candidates=int(refit.refit_total_refined_candidates),
        refit_min_best_second_loss_gap=float(refit.refit_min_best_second_loss_gap),
        global_lower_bound=float(refit.global_lower_bound),
        global_optimality_gap=float(refit.global_optimality_gap),
        global_certificate_method=str(refit.global_certificate_method),
        global_certificate_intervals=int(refit.global_certificate_intervals),
        refit_mode=str(refit.refit_mode),
    )


def _selection_score_diagnostics(
    *,
    data: TumorData,
    refit: PartitionRefitSummary,
    score: SelectionScore,
) -> dict[str, float]:
    diagnostic_df = int(score.degrees_of_freedom)
    return {
        "classic_bic": compute_bic_with_df(
            refit.loglik,
            diagnostic_df,
            score.n_eff,
        ),
        "classic_bic_active_df": compute_bic_with_df(
            refit.loglik,
            refit.active_df,
            score.n_eff,
        ),
        "classic_bic_depth_n": compute_bic_with_df(
            refit.loglik,
            diagnostic_df,
            effective_bic_depth_count(data),
        ),
    }


def _selection_refit_row(
    *,
    data: TumorData,
    refit: PartitionRefitSummary,
    score: SelectionScore,
    cache_hit: bool,
    raw_details: bool,
) -> dict[str, object]:
    diagnostics = _selection_score_diagnostics(data=data, refit=refit, score=score)
    row: dict[str, object] = {
        "selection_score_name": str(score.name),
        "selection_score": float(score.value),
        "selection_score_numerical_uncertainty": float(score.numerical_uncertainty),
        "selection_score_lower_bound": float(score.lower_bound),
        "selection_score_upper_bound": float(score.upper_bound),
        "selection_loglik": float(score.loglik),
        "selection_df": int(score.degrees_of_freedom),
        "selection_penalty": float(score.penalty),
        "selection_n_eff": int(score.n_eff),
        "selection_assignment_log_evidence": float(score.assignment_log_evidence),
        "selection_assignment_code_weight": float(score.assignment_code_weight),
        "selection_assignment_penalty": float(score.assignment_penalty),
        "selection_assignment_dirichlet_alpha": float(score.assignment_dirichlet_alpha),
        "selection_assignment_model_id": str(score.assignment_model_id),
        "selection_assignment_symmetry_mode": str(score.assignment_symmetry_mode),
        "selection_assignment_arithmetic_uncertainty": float(
            score.assignment_arithmetic_uncertainty
        ),
        "classic_bic": float(diagnostics["classic_bic"]),
        "bic_loglik_source": str(refit.loglik_source),
        "bic_refit_finite_candidate_found": bool(refit.finite_candidate_found),
        "bic_refit_cache_hit": bool(cache_hit),
        "refit_global_optimum_certified": bool(refit.global_optimum_certified),
        "refit_global_lower_bound": float(refit.global_lower_bound),
        "refit_global_optimality_gap": float(refit.global_optimality_gap),
        "refit_global_certificate_method": str(refit.global_certificate_method),
        "refit_global_certificate_intervals": int(refit.global_certificate_intervals),
        "refit_numerically_resolved": bool(refit.refit_numerically_resolved),
        "refit_loglik": float(refit.loglik),
        "refit_fit_loss": float(refit.fit_loss),
        "refit_active_df": int(refit.active_df),
        "refit_mode": str(refit.refit_mode),
        "refit_coordinate_count": int(refit.refit_coordinate_count),
        "refit_finite_coordinate_count": int(refit.refit_finite_coordinate_count),
        "refit_total_grid_points": int(refit.refit_total_grid_points),
        "refit_max_grid_spacing": float(refit.refit_max_grid_spacing),
        "refit_total_candidate_basins": int(refit.refit_total_candidate_basins),
        "refit_total_refined_candidates": int(refit.refit_total_refined_candidates),
        "refit_min_best_second_loss_gap": float(refit.refit_min_best_second_loss_gap),
    }
    if raw_details:
        row.update(
            classic_bic_depth_n=float(diagnostics["classic_bic_depth_n"]),
            classic_bic_active_df=float(diagnostics["classic_bic_active_df"]),
            refit_loglik_refinement_delta=float(refit.refit_loglik_refinement_delta),
            refit_max_center_refinement_delta=float(
                refit.refit_max_center_refinement_delta
            ),
        )
    return row


def _score_fixed_labels(
    *,
    data: TumorData,
    labels: np.ndarray,
    partition_signature: str,
    refit_result: PartitionRefitResult,
    selection_options: FitOptions,
    selection_score: str,
) -> SelectionScore:
    canonical_score_name = _normalize_selection_score_name(selection_score)
    computation_profile = get_computation_profile(selection_options.computation_profile)
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
        "selection_contract_id": str(selection_options.selection_contract),
    }
    if canonical_score_name == "fixed_partition_dirichlet_score":
        score_kwargs.update(
            alpha=float(selection_options.selection_dirichlet_alpha),
            code_weight=float(selection_options.selection_dirichlet_code_weight),
        )
    score = score_function(**score_kwargs)
    if str(score.name) != canonical_score_name:
        raise AssertionError(
            f"Requested score {canonical_score_name} produced {score.name}."
        )
    return score


def _evaluate_candidate(
    *,
    data: TumorData,
    fit_options: FitOptions,
    candidate_fit_options: FitOptions | None,
    phi_start: StartArray | None,
    exact_pilot: StartArray | None,
    pooled_start: StartArray | None,
    scalar_well_starts: list[StartArray] | None,
    start_mode: str,
    runtime,
    torch_data,
    solver_context,
    solver_state: SolverState | None,
    compute_summary: bool,
    selection_method: str,
    profile_name: str,
    selection_step: int,
    lambda_value: float,
    selection_score: str,
    static_metadata: CandidateStaticMetadata,
    bic_refit_cache: dict[object, _CachedPartitionRefit] | None = None,
    precomputed_fit: FitResult | None = None,
) -> tuple[
    FitResult,
    dict[str, float | int | str | bool],
    RawFusionCandidate,
]:
    candidate_start_time = perf_counter()
    canonical_score_name = _normalize_selection_score_name(selection_score)
    raw_fit_options = (
        fit_options if candidate_fit_options is None else candidate_fit_options
    )
    selection_options = fit_options
    computation_profile = get_computation_profile(selection_options.computation_profile)

    raw_fit_start_time = perf_counter()
    fit = precomputed_fit
    if fit is None:
        fit = fit_fixed_objective(
            data=data,
            options=replace(raw_fit_options, lambda_value=float(lambda_value)),
            phi_start=phi_start,
            exact_pilot=exact_pilot,
            pooled_start=pooled_start,
            scalar_well_starts=scalar_well_starts,
            start_mode=start_mode,
            runtime=runtime,
            torch_data=torch_data,
            solver_context=solver_context,
            solver_state=solver_state,
            compute_summary=compute_summary,
        )
    elif not np.isclose(
        float(fit.lambda_value), float(lambda_value), rtol=0.0, atol=1e-12
    ):
        raise ValueError("Precomputed raw fit has the wrong lambda value.")
    raw_fit_elapsed_seconds = float(perf_counter() - raw_fit_start_time)
    graph = selection_options.graph
    if graph is None:
        raise RuntimeError(
            "Model-selection candidates require a resolved pairwise-fusion graph."
        )
    partition_tolerance = _effective_bic_partition_tol(selection_options)
    contract = get_selection_contract(selection_options.selection_contract)
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

    refit_start_time = perf_counter()
    cached_refit, cache_hit = _fixed_partition_refit(
        data=data,
        partition=partition,
        selection_options=selection_options,
        cache=bic_refit_cache,
    )
    refit_result = cached_refit.result
    refit_elapsed_seconds = (
        0.0 if cache_hit else float(perf_counter() - refit_start_time)
    )

    score = _score_fixed_labels(
        data=data,
        labels=partition.labels,
        partition_signature=partition.signature,
        refit_result=refit_result,
        selection_options=selection_options,
        selection_score=canonical_score_name,
    )
    refit = _build_refit_summary(
        refit_result,
        partition_signature=partition.signature,
        nominal_df=int(score.degrees_of_freedom),
        resolution=cached_refit,
    )
    raw_objective_certified = bool(
        float(fit.lambda_value) > 0.0
        and str(fit.estimator_role) == "raw_fused_lambda_path"
        and bool(fit.objective_faithful)
        and bool(fit.full_kkt_certified)
        and bool(fit.selection_eligible)
    )
    reason = _ineligibility_reason(
        fit=fit,
        partition=partition,
        refit_finite=bool(refit.finite_candidate_found),
        refit_numerically_resolved=bool(refit.refit_numerically_resolved),
        score_finite=bool(np.isfinite(score.value)),
        require_global_refit=bool(computation_profile.is_strict),
    )
    candidate = RawFusionCandidate(
        raw_fit=fit,
        partition=partition,
        refit=refit,
        score=score,
        raw_objective_certified=raw_objective_certified,
        eligible_for_selection=reason == "none",
        ineligibility_reason=reason,
        computation_profile=str(computation_profile.name),
    )
    validate_candidate_identity(candidate)

    penalty_value, profile_penalty_value = _profile_penalty_from_fit(fit)
    raw_objective_uncertainty = max(
        1e-10 * (1.0 + abs(float(fit.penalized_objective))),
        32.0 * np.finfo(np.float64).eps * (1.0 + abs(float(fit.penalized_objective))),
    )
    row: dict[str, float | int | str | bool] = {
        "tumor_id": data.tumor_id,
        "selection_method": selection_method,
        "selection_contract_id": str(contract.contract_id),
        "selection_contract_json": str(contract.to_json()),
        "selection_profile": profile_name,
        "computation_profile": str(computation_profile.name),
        "target_estimator": "complete_graph_pairwise_fusion",
        "solution_mode": (
            "strict_certified"
            if computation_profile.is_strict
            else "approximate_single_tumor_search"
        ),
        "objective_equivalent_to_strict_graph": bool(
            computation_profile.objective_equivalent_to_strict
        ),
        "refit_mode": str(refit.refit_mode),
        "refit_global_certificate_required": bool(computation_profile.is_strict),
        "selection_step": int(selection_step),
        "lambda": float(fit.lambda_value),
        "raw_objective_numerical_uncertainty": float(raw_objective_uncertainty),
        "raw_objective_lower_bound": float(
            fit.penalized_objective - raw_objective_uncertainty
        ),
        "raw_objective_upper_bound": float(
            fit.penalized_objective + raw_objective_uncertainty
        ),
        "raw_objective_uncertainty_certified": False,
        "lambda_applicable": True,
        "candidate_pool_source": "raw_fused_lambda_path",
        "candidate_family": "raw_fusion",
        "estimator_role": str(fit.estimator_role),
        **_selection_refit_row(
            data=data,
            refit=refit,
            score=score,
            cache_hit=cache_hit,
            raw_details=True,
        ),
        "partition_signature": str(partition.signature),
        "partition_source": str(partition.source),
        "partition_tol": float(partition.tolerance),
        "partition_certified": bool(partition.certified),
        "partition_certification_applicable": True,
        "partition_maximal": bool(partition.maximal),
        "partition_cross_close_edge_found": bool(partition.cross_close_edge_found),
        "partition_certificate_graph_hash_matches": bool(
            partition.certificate_graph_hash_matches
        ),
        "partition_certification_failure_reason": str(
            partition.certification_failure_reason
        ),
        "partition_max_diameter": float(partition.max_diameter),
        "partition_diameter_exact": bool(partition.diameter_exact),
        "n_clusters": int(partition.n_clusters),
        "cluster_sizes": _cluster_sizes_text(partition.labels),
        "partition_labels_0based": ",".join(
            str(int(value)) for value in np.asarray(partition.labels, dtype=np.int64)
        ),
        "eligible_for_selection": bool(candidate.eligible_for_selection),
        "ineligibility_reason": str(candidate.ineligibility_reason),
        "raw_kkt_eligible": bool(fit.selection_eligible),
        "raw_objective_certified": bool(raw_objective_certified),
        "converged": bool(fit.converged),
        "raw_fit_status": str(fit.failure_reason),
        "loglik": float(fit.loglik),
        "fit_loss": float(-fit.loglik),
        "penalized_objective": float(fit.penalized_objective),
        "penalty": float(penalty_value),
        "profile_penalty": float(profile_penalty_value),
        "fixed_objective_kkt_residual": float(fit.fixed_objective_kkt_residual),
        "working_precision_kkt_residual": float(
            fit.working_precision_kkt_residual
        ),
        "outer_backward_error_stationarity_residual": float(
            fit.outer_backward_error_stationarity_residual
        ),
        "outer_backward_error_edge_subgradient_residual": float(
            fit.outer_backward_error_edge_subgradient_residual
        ),
        "outer_backward_error_dual_ball_residual": float(
            fit.outer_backward_error_dual_ball_residual
        ),
        "certificate_residual_method": str(
            fit.exactness_provenance.residual_method
            if fit.exactness_provenance is not None
            else "unknown"
        ),
        "working_dtype": str(fit.working_dtype),
        "certificate_audit_dtype": str(fit.certificate_audit_dtype),
        "precision_polish_applied": bool(fit.precision_polish_applied),
        "precision_polish_max_abs_phi_delta": float(
            fit.precision_polish_max_abs_phi_delta
        ),
        "directional_kink_admissible": bool(
            fit.exactness_provenance.directional_kink_admissible
            if fit.exactness_provenance is not None
            else False
        ),
        "outer_stationarity_residual": float(fit.outer_stationarity_residual),
        "outer_projected_stationarity_norm": float(
            fit.outer_projected_stationarity_norm
        ),
        "outer_stationarity_normalizer": float(fit.outer_stationarity_normalizer),
        "outer_smooth_gradient_norm": float(fit.outer_smooth_gradient_norm),
        "outer_fusion_adjustment_norm": float(fit.outer_fusion_adjustment_norm),
        "outer_edge_subgradient_residual": float(fit.outer_edge_subgradient_residual),
        "outer_dual_ball_residual": float(fit.outer_dual_ball_residual),
        "outer_box_residual": float(fit.outer_box_residual),
        "outer_box_primal_violation": float(fit.outer_box_primal_violation),
        "outer_stationarity_residual_before_dual_refine": float(
            fit.outer_stationarity_residual_before_dual_refine
        ),
        "outer_stationarity_residual_after_dual_refine": float(
            fit.outer_stationarity_residual_after_dual_refine
        ),
        "outer_kkt_fused_edges": int(fit.outer_kkt_fused_edges),
        "outer_kkt_nonzero_edges": int(fit.outer_kkt_nonzero_edges),
        "stationarity_certified": bool(fit.stationarity_certified),
        "global_optimality_certified": bool(fit.global_optimality_certified),
        "global_optimality_basis": str(fit.global_optimality_basis),
        "number_of_starts": int(fit.number_of_starts),
        "number_of_finite_starts": int(fit.number_of_finite_starts),
        "best_start_objective": float(fit.best_start_objective),
        "second_best_start_objective": float(fit.second_best_start_objective),
        "objective_spread_across_starts": float(fit.objective_spread_across_starts),
        "selected_start_objective_rank": int(fit.selected_start_objective_rank),
        "iterations": int(fit.iterations),
        "inner_iterations": int(fit.inner_iterations),
        "admm_iterations": int(fit.admm_iterations),
        "inner_solver": str(fit.inner_solver),
        "inner_backend": str(fit.inner_backend),
        "backend_iterations": int(fit.backend_iterations),
        "workset_iterations": int(fit.workset_iterations),
        "workset_expansions": int(fit.workset_expansions),
        "streamed_edge_passes": int(fit.streamed_edge_passes),
        "dense_iterations": int(fit.dense_iterations),
        "certificate_iterations": int(fit.certificate_iterations),
        "accepted_outer_steps": int(fit.accepted_outer_steps),
        "attempted_outer_steps": int(fit.attempted_outer_steps),
        "accepted_full_steps": int(fit.accepted_full_steps),
        "accepted_damped_steps": int(fit.accepted_damped_steps),
        "failed_majorization_checks": int(fit.failed_majorization_checks),
        "failed_inner_model_checks": int(fit.failed_inner_model_checks),
        "failed_em_envelope_checks": int(fit.failed_em_envelope_checks),
        "failed_descent_checks": int(fit.failed_descent_checks),
        "failed_nonfinite_checks": int(fit.failed_nonfinite_checks),
        "final_relative_objective_change": float(fit.final_relative_objective_change),
        "final_step_residual": float(fit.final_step_residual),
        "converged_inner": bool(fit.converged_inner),
        "converged_outer": bool(fit.converged_outer),
        "full_certificate_audit_passes": int(fit.full_certificate_audit_passes),
        "fallback_reason": str(fit.fallback_reason),
        "exactness_provenance_version": int(fit.exactness_provenance_version),
        "objective_faithful": bool(fit.objective_faithful),
        "objective_spec_hash": str(fit.objective_spec_hash),
        "base_fusion_objective_hash": str(fit.base_fusion_objective_hash),
        "original_graph_hash": str(fit.original_graph_hash),
        "certificate_problem_hash": str(fit.certificate_problem_hash),
        "certificate_scope": str(fit.certificate_scope),
        "certificate_gradient_scope": str(fit.certificate_gradient_scope),
        "full_kkt_certified": bool(fit.full_kkt_certified),
        "full_kkt_certificate_status": str(fit.full_kkt_certificate_status),
        "full_kkt_tolerance": float(fit.full_kkt_tolerance),
        "raw_solver_primal_tol": float(raw_fit_options.tol),
        "outer_kkt_certificate_status": str(fit.outer_kkt_certificate_status),
        "outer_num_frozen_coordinates": int(fit.outer_num_frozen_coordinates),
        "mm_consistency_violations": int(fit.mm_consistency_violations),
        "failure_reason": str(fit.failure_reason),
        "candidate_elapsed_seconds": float(perf_counter() - candidate_start_time),
        "raw_fit_elapsed_seconds": float(raw_fit_elapsed_seconds),
        "bic_refit_elapsed_seconds": float(refit_elapsed_seconds),
        "primary_phi_source": "raw_pairwise_fusion",
        "refit_phi_source": "fixed_partition_refit",
        "device": str(fit.device),
        "dtype": str(fit.dtype),
        "tol": float(raw_fit_options.tol),
        "outer_max_iter": int(raw_fit_options.outer_max_iter),
        "inner_max_iter": int(raw_fit_options.inner_max_iter),
        "eps": float(raw_fit_options.eps),
        "major_prior": float(raw_fit_options.major_prior),
        "selection_refit_tol": float(selection_options.selection_refit_tol),
        "selection_refit_max_iter": int(selection_options.selection_refit_max_iter),
        "graph_name": str(fit.graph_name),
        "num_edges": int(static_metadata.edge_count),
        "edge_weight_min": float(static_metadata.edge_weight_min),
        "edge_weight_max": float(static_metadata.edge_weight_max),
        "edge_weight_mean": float(static_metadata.edge_weight_mean),
        "edge_list_hash": str(static_metadata.edge_list_hash),
        "pilot_matrix_hash": str(static_metadata.pilot_matrix_hash),
        "input_data_hash": str(static_metadata.input_data_hash),
        "fit_compute_summary": bool(compute_summary),
        "fit_start_mode": str(start_mode),
        "solver_state_warm_start": bool(solver_state is not None),
    }
    return fit, row, candidate


def evaluate_direct_partition_candidate(
    *,
    data: TumorData,
    proposal: PartitionCandidate,
    selection_options: FitOptions,
    candidate_id: int,
    source: str,
    parent_raw_candidate_id: int | None,
    parent_raw_lambda: float | None,
    generation_contract_id: str,
    refit_cache: dict[object, _CachedPartitionRefit] | None,
    parent_raw_phi_hash: str = "",
) -> tuple[dict[str, object], DirectPartitionCandidate]:
    """Evaluate one deterministic non-fusion partition under the common score."""

    started = perf_counter()
    labels = _canonical_partition_labels(
        np.asarray(proposal.labels, dtype=np.int64).reshape(-1)
    )
    mutation_ids = tuple(str(value) for value in data.mutation_ids)
    signature = _partition_signature(labels, mutation_ids)
    n_clusters = int(np.unique(labels).size)
    requested_k_value = proposal.diagnostics.get("requested_K", proposal.K)
    requested_k = int(round(float(requested_k_value)))
    pre_signature = str(proposal.diagnostics.get("pre_refinement_signature", signature))
    partition = DirectPartition(
        labels=labels,
        signature=signature,
        n_clusters=n_clusters,
        source=source,
        requested_k=requested_k,
        mutation_ids=mutation_ids,
        generation_contract_id=str(generation_contract_id),
        parent_raw_candidate_id=parent_raw_candidate_id,
        parent_raw_lambda=parent_raw_lambda,
        parent_raw_phi_hash=str(parent_raw_phi_hash),
        cem_iterations=int(proposal.diagnostics.get("cem_iterations", 0.0)),
        component_death_count=int(
            proposal.diagnostics.get("component_death_count", 0.0)
        ),
        refinement_score_before=float(
            proposal.diagnostics.get("refinement_score_before", np.nan)
        ),
        refinement_score_after=float(
            proposal.diagnostics.get("refinement_score_after", np.nan)
        ),
        deterministic_generation=bool(
            proposal.diagnostics.get("deterministic_generation", 1.0)
        ),
    )
    cached_refit, cache_hit = _fixed_labels_refit(
        data=data,
        labels=labels,
        partition_signature=signature,
        selection_options=selection_options,
        cache=refit_cache,
    )
    refit_result = cached_refit.result
    score = _score_fixed_labels(
        data=data,
        labels=labels,
        partition_signature=signature,
        refit_result=refit_result,
        selection_options=selection_options,
        selection_score=selection_options.selection_score,
    )
    refit = _build_refit_summary(
        refit_result,
        partition_signature=signature,
        nominal_df=int(score.degrees_of_freedom),
        resolution=cached_refit,
    )
    profile = get_computation_profile(selection_options.computation_profile)
    reason = _direct_partition_ineligibility_reason(
        partition=partition,
        refit=refit,
        score=score,
        require_global_refit=bool(profile.is_strict),
    )
    candidate = DirectPartitionCandidate(
        partition=partition,
        refit=refit,
        score=score,
        eligible_for_selection=reason == "none",
        ineligibility_reason=reason,
        computation_profile=str(profile.name),
    )
    validate_candidate_identity(candidate)
    row: dict[str, object] = {
        "_candidate_id": int(candidate_id),
        "tumor_id": str(data.tumor_id),
        "candidate_id": int(candidate_id),
        "candidate_family": "direct_partition",
        "candidate_pool_source": str(source),
        "partition_source": str(source),
        "partition_signature": str(signature),
        "requested_K": int(requested_k),
        "n_clusters": int(n_clusters),
        "cluster_sizes": _cluster_sizes_text(labels),
        "partition_labels_0based": ",".join(str(int(value)) for value in labels),
        "lambda": float("nan"),
        "lambda_applicable": False,
        "parent_raw_candidate_id": (
            float("nan")
            if parent_raw_candidate_id is None
            else int(parent_raw_candidate_id)
        ),
        "parent_raw_lambda": (
            float("nan") if parent_raw_lambda is None else float(parent_raw_lambda)
        ),
        "parent_raw_phi_hash": str(parent_raw_phi_hash),
        "selection_contract_id": str(generation_contract_id),
        "selection_contract_json": get_selection_contract(
            generation_contract_id
        ).to_json(),
        "generation_contract_id": str(generation_contract_id),
        "pre_refinement_signature": str(pre_signature),
        "cem_iterations": int(partition.cem_iterations),
        "component_death_count": int(partition.component_death_count),
        "refinement_score_before": float(partition.refinement_score_before),
        "refinement_score_after": float(partition.refinement_score_after),
        "deterministic_partition_generation": bool(partition.deterministic_generation),
        "direct_partition_identity_certified": True,
        "partition_certified": False,
        "partition_certification_applicable": False,
        # These are fusion-summary predicates, not direct-partition
        # certification claims.  The direct candidate is certified through
        # its deterministic identity, fixed-label refit, and reconstructible
        # score instead.
        "partition_maximal": False,
        "partition_diameter_exact": False,
        "partition_max_diameter": float("nan"),
        "partition_certification_failure_reason": "not_applicable_direct_partition",
        **_selection_refit_row(
            data=data,
            refit=refit,
            score=score,
            cache_hit=cache_hit,
            raw_details=False,
        ),
        "eligible_for_selection": bool(candidate.eligible_for_selection),
        "ineligibility_reason": str(candidate.ineligibility_reason),
        "converged": bool(refit.finite_candidate_found),
        "estimator_role": "direct_partition_candidate",
        "raw_objective_certified": False,
        "raw_kkt_eligible": False,
        "objective_faithful": False,
        "raw_certificate_status": "not_applicable_direct_partition",
        "full_kkt_certified": False,
        "full_kkt_certificate_status": "not_applicable_direct_partition",
        "fixed_objective_kkt_residual": float("nan"),
        "penalized_objective": float("nan"),
        "mm_consistency_violations": 0,
        "selection_step": int(candidate_id),
        "candidate_elapsed_seconds": float(perf_counter() - started),
        "computation_profile": str(profile.name),
        "primary_phi_source": "raw_pairwise_fusion_reference",
        "refit_phi_source": "selected_direct_partition_refit",
    }
    return row, candidate


__all__ = [
    "_evaluate_candidate",
    "_fixed_labels_refit",
    "_fixed_partition_refit",
    "_selection_refit_cache_key",
    "_selection_score_diagnostics",
    "evaluate_direct_partition_candidate",
    "validate_candidate_identity",
]
