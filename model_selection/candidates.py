from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np

from ..core.bic import (
    compute_bic_with_df,
    effective_bic_depth_count,
    fixed_partition_bic,
)
from ..core.model import FitOptions, FitResult, fit_fixed_objective
from ..core.fusion.refit import (
    PartitionRefitResult,
    partition_constrained_observed_refit,
)
from ..core.fusion.types import SolverState
from ..io.data import TumorData
from .partitions import (
    _cluster_sizes_text,
    _partition_signature,
    extract_certified_fusion_partition,
)
from .scoring import (
    _effective_bic_partition_tol,
    _normalize_selection_score_name,
    _profile_penalty_from_fit,
)
from .types import (
    CandidateStaticMetadata,
    PartitionRefitSummary,
    RawFusionCandidate,
    StartArray,
)


def validate_candidate_identity(candidate: RawFusionCandidate) -> None:
    """Fail fast when raw partition, refit, and score identities diverge."""

    partition = candidate.partition
    refit = candidate.refit
    score = candidate.score
    if not np.array_equal(partition.labels, refit.labels):
        raise AssertionError("Fixed-partition refit changed raw fusion labels.")
    actual_signature = _partition_signature(partition.labels)
    if partition.signature != actual_signature:
        raise AssertionError(
            "Raw fusion partition signature does not match its labels."
        )
    if int(partition.n_clusters) != int(np.unique(partition.labels).size):
        raise AssertionError("Raw fusion partition cluster count is inconsistent.")
    if refit.partition_signature != partition.signature:
        raise AssertionError("Refit partition signature does not match raw partition.")
    if score.partition_signature != partition.signature:
        raise AssertionError("Selection score does not match raw partition.")


def _ineligibility_reason(
    *,
    fit: FitResult,
    partition_certified: bool,
    refit_finite: bool,
    score_finite: bool,
) -> str:
    if float(fit.lambda_value) <= 0.0:
        return "nonpositive_lambda"
    if str(fit.estimator_role) != "raw_fused_lambda_path":
        return "not_raw_fused_lambda_path"
    if not bool(fit.objective_faithful):
        return "raw_objective_not_faithful"
    if not bool(fit.full_kkt_certified) or not bool(fit.selection_eligible):
        return "raw_objective_not_kkt_certified"
    if not partition_certified:
        return "raw_partition_chaining_or_solver_tolerance"
    if not refit_finite:
        return "fixed_partition_refit_nonfinite"
    if not score_finite:
        return "fixed_partition_score_nonfinite"
    return "none"


def _build_refit_summary(
    refit: PartitionRefitResult,
    *,
    partition_signature: str,
    nominal_df: int,
    path_likelihood_present: bool,
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
        anchor_mode=str(refit.anchor_mode),
        clonal_cluster=(
            None if refit.clonal_cluster is None else int(refit.clonal_cluster)
        ),
        anchor_deviance_increase=float(refit.anchor_deviance_increase),
        second_best_anchor_deviance_increase=float(
            refit.second_best_anchor_deviance_increase
        ),
        finite_candidate_found=bool(refit.finite_candidate_found),
        global_optimum_certified=bool(
            refit.finite_candidate_found and not path_likelihood_present
        ),
        loglik_source=str(refit.loglik_source),
    )


def _evaluate_candidate(
    *,
    data: TumorData,
    fit_options: FitOptions,
    candidate_fit_options: FitOptions | None,
    bic_df_scale: float,
    bic_cluster_penalty: float,
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
    bic_refit_cache: dict[object, PartitionRefitResult] | None = None,
) -> tuple[
    FitResult,
    dict[str, float | int | str | bool],
    RawFusionCandidate,
]:
    candidate_start_time = perf_counter()
    canonical_score_name = _normalize_selection_score_name(selection_score)
    effective_fit_options = (
        fit_options if candidate_fit_options is None else candidate_fit_options
    )

    raw_fit_start_time = perf_counter()
    fit = fit_fixed_objective(
        data=data,
        options=replace(effective_fit_options, lambda_value=float(lambda_value)),
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
    raw_fit_elapsed_seconds = float(perf_counter() - raw_fit_start_time)
    graph = effective_fit_options.graph
    if graph is None:
        raise RuntimeError(
            "Model-selection candidates require a resolved pairwise-fusion graph."
        )
    partition_tolerance = _effective_bic_partition_tol(effective_fit_options)
    partition = extract_certified_fusion_partition(
        fit,
        graph=graph,
        tolerance=partition_tolerance,
    )

    refit_spec_key = (
        partition.signature,
        str(effective_fit_options.selection_anchor),
        float(effective_fit_options.major_prior),
        float(effective_fit_options.eps),
        float(effective_fit_options.tol),
        max(int(effective_fit_options.inner_max_iter), 32),
    )
    cache_allowed = getattr(data, "path_likelihood", None) is None
    cache_hit = bool(
        cache_allowed
        and bic_refit_cache is not None
        and refit_spec_key in bic_refit_cache
    )
    refit_start_time = perf_counter()
    if cache_hit:
        refit_result = bic_refit_cache[refit_spec_key]
    else:
        refit_result = partition_constrained_observed_refit(
            data,
            partition.labels,
            major_prior=float(effective_fit_options.major_prior),
            eps=float(effective_fit_options.eps),
            tol=float(effective_fit_options.tol),
            max_iter=max(int(effective_fit_options.inner_max_iter), 32),
            anchor_mode=str(effective_fit_options.selection_anchor),
        )
        if cache_allowed and bic_refit_cache is not None:
            bic_refit_cache[refit_spec_key] = refit_result
    refit_elapsed_seconds = (
        0.0 if cache_hit else float(perf_counter() - refit_start_time)
    )

    score = fixed_partition_bic(
        loglik=float(refit_result.loglik),
        num_clusters=int(partition.n_clusters),
        data=data,
        anchor_mode=str(effective_fit_options.selection_anchor),
        partition_signature=partition.signature,
    )
    if str(score.name) != canonical_score_name:
        raise AssertionError(
            f"Requested score {canonical_score_name} produced {score.name}."
        )
    refit = _build_refit_summary(
        refit_result,
        partition_signature=partition.signature,
        nominal_df=int(score.degrees_of_freedom),
        path_likelihood_present=getattr(data, "path_likelihood", None) is not None,
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
        partition_certified=bool(partition.certified),
        refit_finite=bool(refit.finite_candidate_found),
        score_finite=bool(np.isfinite(score.value)),
    )
    candidate = RawFusionCandidate(
        raw_fit=fit,
        partition=partition,
        refit=refit,
        score=score,
        raw_objective_certified=raw_objective_certified,
        eligible_for_selection=reason == "none",
        ineligibility_reason=reason,
    )
    validate_candidate_identity(candidate)

    penalty_value, profile_penalty_value = _profile_penalty_from_fit(fit)
    classic_df = max(int(partition.n_clusters) - 1, 0) * int(data.num_regions)
    classic_bic = compute_bic_with_df(
        refit.loglik,
        classic_df,
        score.n_eff,
    )
    active_bic = compute_bic_with_df(
        refit.loglik,
        refit.active_df,
        score.n_eff,
    )
    depth_bic = compute_bic_with_df(
        refit.loglik,
        classic_df,
        effective_bic_depth_count(data),
    )
    row: dict[str, float | int | str | bool] = {
        "tumor_id": data.tumor_id,
        "selection_method": selection_method,
        "selection_profile": profile_name,
        "selection_step": int(selection_step),
        "lambda": float(fit.lambda_value),
        "lambda_applicable": True,
        "candidate_pool_source": "raw_fused_lambda_path",
        "estimator_role": str(fit.estimator_role),
        "selection_score_name": str(score.name),
        "selection_score": float(score.value),
        "bic": float(score.value),
        "bic_value": float(score.value),
        "classic_bic": float(classic_bic),
        "clonal_fixed_partition_bic": (
            float(score.value)
            if score.name == "clonal_fixed_partition_bic"
            else float("nan")
        ),
        "fixed_partition_bic": (
            float(score.value) if score.name == "fixed_partition_bic" else float("nan")
        ),
        "bic_loglik": float(score.loglik),
        "bic_loglik_source": str(refit.loglik_source),
        "bic_df": int(score.degrees_of_freedom),
        "bic_active_df": int(refit.active_df),
        "bic_penalty": float(score.penalty),
        "bic_active_penalty": float(refit.active_df * np.log(max(score.n_eff, 1))),
        "bic_n_eff": int(score.n_eff),
        "classic_bic_depth_n": float(depth_bic),
        "classic_bic_active_df": float(active_bic),
        "bic_refit_finite_candidate_found": bool(refit.finite_candidate_found),
        "bic_refit_converged": bool(refit.finite_candidate_found),
        "bic_refit_cache_hit": bool(cache_hit),
        "refit_global_optimum_certified": bool(refit.global_optimum_certified),
        "refit_loglik": float(refit.loglik),
        "refit_fit_loss": float(refit.fit_loss),
        "refit_active_df": int(refit.active_df),
        "refit_anchor_mode": str(refit.anchor_mode),
        "refit_clonal_cluster": (
            -1 if refit.clonal_cluster is None else int(refit.clonal_cluster)
        ),
        "anchor_deviance_increase": float(refit.anchor_deviance_increase),
        "second_best_anchor_deviance_increase": float(
            refit.second_best_anchor_deviance_increase
        ),
        "partition_signature": str(partition.signature),
        "partition_hash": str(partition.signature),
        "partition_source": str(partition.source),
        "partition_tol": float(partition.tolerance),
        "reporting_partition_tol": float(effective_fit_options.reporting_partition_tol),
        "partition_certified": bool(partition.certified),
        "partition_max_diameter": float(partition.max_diameter),
        "partition_diameter_exact": bool(partition.diameter_exact),
        "n_clusters": int(partition.n_clusters),
        "bic_n_clusters": int(partition.n_clusters),
        "cluster_sizes": _cluster_sizes_text(partition.labels),
        "eligible_for_selection": bool(candidate.eligible_for_selection),
        "bic_selection_eligible": bool(candidate.eligible_for_selection),
        "selection_eligible": bool(candidate.eligible_for_selection),
        "ineligibility_reason": str(candidate.ineligibility_reason),
        "raw_kkt_eligible": bool(fit.selection_eligible),
        "raw_objective_certified": bool(raw_objective_certified),
        "converged": bool(fit.converged),
        "raw_fit_status": str(fit.failure_reason),
        "loglik": float(fit.loglik),
        "raw_loglik": float(fit.loglik),
        "fit_loss": float(-fit.loglik),
        "penalized_objective": float(fit.penalized_objective),
        "raw_objective": float(fit.penalized_objective),
        "penalty": float(penalty_value),
        "raw_penalty": float(penalty_value),
        "profile_penalty": float(profile_penalty_value),
        "fixed_objective_kkt_residual": float(fit.fixed_objective_kkt_residual),
        "raw_kkt_residual": float(fit.fixed_objective_kkt_residual),
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
        "quotient_iterations": int(fit.quotient_iterations),
        "workset_iterations": int(fit.workset_iterations),
        "workset_expansions": int(fit.workset_expansions),
        "streamed_edge_passes": int(fit.streamed_edge_passes),
        "dense_iterations": int(fit.dense_iterations),
        "certificate_iterations": int(fit.certificate_iterations),
        "full_certificate_audit_passes": int(fit.full_certificate_audit_passes),
        "fallback_reason": str(fit.fallback_reason),
        "exactness_provenance_version": int(fit.exactness_provenance_version),
        "objective_faithful": bool(fit.objective_faithful),
        "objective_spec_hash": str(fit.objective_spec_hash),
        "original_graph_hash": str(fit.original_graph_hash),
        "certificate_problem_hash": str(fit.certificate_problem_hash),
        "certificate_scope": str(fit.certificate_scope),
        "certificate_gradient_scope": str(fit.certificate_gradient_scope),
        "full_kkt_certified": bool(fit.full_kkt_certified),
        "full_kkt_certificate_status": str(fit.full_kkt_certificate_status),
        "full_kkt_tolerance": float(fit.full_kkt_tolerance),
        "outer_kkt_certificate_status": str(fit.outer_kkt_certificate_status),
        "mm_consistency_violations": int(fit.mm_consistency_violations),
        "failure_reason": str(fit.failure_reason),
        "candidate_elapsed_seconds": float(perf_counter() - candidate_start_time),
        "raw_fit_elapsed_seconds": float(raw_fit_elapsed_seconds),
        "bic_refit_elapsed_seconds": float(refit_elapsed_seconds),
        "primary_phi_source": "raw_pairwise_fusion",
        "refit_phi_source": "fixed_partition_refit",
        "device": str(fit.device),
        "dtype": str(fit.dtype),
        "tol": float(effective_fit_options.tol),
        "outer_max_iter": int(effective_fit_options.outer_max_iter),
        "inner_max_iter": int(effective_fit_options.inner_max_iter),
        "eps": float(effective_fit_options.eps),
        "major_prior": float(effective_fit_options.major_prior),
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
        "bic_df_scale": float(bic_df_scale),
        "bic_cluster_penalty": float(bic_cluster_penalty),
    }
    return fit, row, candidate


__all__ = ["_evaluate_candidate", "validate_candidate_identity"]
