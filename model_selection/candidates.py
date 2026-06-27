from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from ..core.model import FitOptions, FitResult, fit_fixed_objective
from ..core.fusion.partition_starts import PartitionCandidate
from ..core.fusion.refit import PartitionRefitResult, partition_constrained_observed_refit
from ..core.fusion.solver import cluster_diameters_from_edges, cluster_labels_from_edges
from ..core.fusion.types import SolverState
from ..io.data import TumorData
from ..metrics.evaluation import (
    SimulationEvaluation,
    SimulationTruth,
    evaluate_ari_against_simulation,
    evaluate_fit_against_simulation,
)
from .config import LIKELIHOOD_PARTITION_SENTINEL_LAMBDA
from .partitions import (
    _canonical_partition_labels,
    _centers_from_partition_labels,
    _cluster_sizes_text,
    _max_cluster_diameter,
    _multiplicity_summary_for_phi,
    _partition_signature,
)
from .scoring import (
    _effective_bic_partition_tol,
    _is_bic_selection_eligible,
    _normalize_selection_score_name,
    _profile_penalty_from_fit,
    _selection_score_value,
)
from .types import CandidateStaticMetadata, SelectionArtifact, StartArray
from ..runners.selection import (
    bic_degrees_of_freedom,
    compute_bic_with_df,
    compute_classic_bic_depth_n,
    effective_bic_cell_count,
    effective_bic_depth_count,
)

def _evaluate_candidate(
    *,
    data: TumorData,
    fit_options: FitOptions,
    candidate_fit_options: FitOptions | None,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    simulation_root: Path | None,
    simulation_truth: SimulationTruth | None,
    evaluate_candidate: bool,
    ari_only_evaluation: bool,
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
    bic_refit_cache: dict[str, PartitionRefitResult] | None = None,
    static_metadata: CandidateStaticMetadata | None = None,
) -> tuple[FitResult, SimulationEvaluation | None, dict[str, float | int | str | bool], SelectionArtifact]:
    candidate_start_time = perf_counter()
    canonical_score_name = _normalize_selection_score_name(selection_score)
    effective_fit_options = fit_options if candidate_fit_options is None else candidate_fit_options
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
    bic_partition_tol = _effective_bic_partition_tol(effective_fit_options)
    graph = effective_fit_options.graph
    if graph is None:
        raise RuntimeError("Model-selection candidates require a resolved pairwise-fusion graph.")
    candidate_static = (
        static_metadata
        if static_metadata is not None
        else _candidate_static_metadata(data, graph)
    )
    bic_labels = cluster_labels_from_edges(
        fit.phi,
        edge_u=graph.edge_u,
        edge_v=graph.edge_v,
        tol=bic_partition_tol,
    )
    bic_partition_diameters, bic_partition_diameter_exact = cluster_diameters_from_edges(
        fit.phi,
        bic_labels,
        edge_u=graph.edge_u,
        edge_v=graph.edge_v,
    )
    bic_partition_max_diameter = _max_cluster_diameter(bic_partition_diameters)
    partition_hash = _partition_signature(bic_labels)
    bic_refit_cache_hit = False
    if bic_refit_cache is not None and partition_hash in bic_refit_cache:
        bic_refit = bic_refit_cache[partition_hash]
        bic_refit_cache_hit = True
        bic_refit_elapsed_seconds = 0.0
    else:
        bic_refit_start_time = perf_counter()
        bic_refit = partition_constrained_observed_refit(
            data,
            bic_labels,
            major_prior=float(effective_fit_options.major_prior),
            eps=float(effective_fit_options.eps),
            tol=float(effective_fit_options.tol),
            max_iter=max(int(effective_fit_options.inner_max_iter), 32),
            hint_phi=fit.phi,
        )
        bic_refit_elapsed_seconds = float(perf_counter() - bic_refit_start_time)
        if bic_refit_cache is not None:
            bic_refit_cache[partition_hash] = bic_refit
    bic_n_clusters = int(bic_refit.n_clusters)
    bic_loglik = float(bic_refit.loglik)
    bic, classic_bic, extended_bic = _selection_score_value(
        loglik=bic_loglik,
        num_clusters=bic_n_clusters,
        data=data,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
        selection_score=selection_score,
    )
    classic_bic_depth_n = compute_classic_bic_depth_n(bic_loglik, bic_n_clusters, data)
    bic_df_value = float(bic_degrees_of_freedom(bic_n_clusters, data))
    bic_active_df_value = float(bic_refit.active_degrees_of_freedom)
    bic_n_eff_value = float(effective_bic_cell_count(data))
    bic_depth_n_eff_value = float(effective_bic_depth_count(data))
    classic_bic_active_df = compute_bic_with_df(bic_loglik, bic_active_df_value, bic_n_eff_value)
    classic_bic_active_df_depth_n = compute_bic_with_df(
        bic_loglik,
        bic_active_df_value,
        bic_depth_n_eff_value,
    )
    bic_refit_finite_candidate_found = bool(bic_refit.finite_candidate_found)
    bic_refit_global_optimum_certified = False
    artifact = SelectionArtifact(
        bic=float(bic),
        classic_bic=float(classic_bic),
        extended_bic=float(extended_bic),
        classic_bic_depth_n=float(classic_bic_depth_n),
        classic_bic_active_df=float(classic_bic_active_df),
        classic_bic_active_df_depth_n=float(classic_bic_active_df_depth_n),
        bic_loglik=float(bic_loglik),
        bic_loglik_source=str(bic_refit.loglik_source),
        bic_df=float(bic_df_value),
        bic_active_df=float(bic_active_df_value),
        bic_n_eff=float(bic_n_eff_value),
        bic_depth_n_eff=float(bic_depth_n_eff_value),
        bic_partition_tol=float(bic_partition_tol),
        bic_refit_boundary_count=int(bic_refit.boundary_count),
        bic_refit_finite_candidate_found=bool(bic_refit_finite_candidate_found),
        bic_refit_global_optimum_certified=bool(bic_refit_global_optimum_certified),
        bic_refit_coordinate_count=int(bic_refit.refit_coordinate_count),
        bic_refit_finite_coordinate_count=int(bic_refit.refit_finite_coordinate_count),
        bic_refit_total_grid_points=int(bic_refit.refit_total_grid_points),
        bic_refit_max_grid_spacing=float(bic_refit.refit_max_grid_spacing),
        bic_refit_total_candidate_basins=int(bic_refit.refit_total_candidate_basins),
        bic_refit_total_refined_candidates=int(bic_refit.refit_total_refined_candidates),
        bic_refit_min_best_second_loss_gap=float(bic_refit.refit_min_best_second_loss_gap),
        bic_refit_converged=bool(bic_refit_finite_candidate_found),
        bic_refit_phi=bic_refit.phi.astype(fit.phi.dtype, copy=False),
        bic_refit_cluster_centers=bic_refit.cluster_centers.astype(fit.phi.dtype, copy=False),
        bic_partition_labels=bic_labels.astype(np.int64, copy=False),
        selection_score_name=str(canonical_score_name),
    )
    penalty_value, profile_penalty_value = _profile_penalty_from_fit(fit)
    bic_penalty_value = float(bic_df_value * np.log(max(bic_n_eff_value, 1.0)))
    bic_active_penalty_value = float(bic_active_df_value * np.log(max(bic_n_eff_value, 1.0)))
    raw_kkt_eligible = bool(fit.selection_eligible)
    bic_selection_eligible = _is_bic_selection_eligible(
        raw_kkt_eligible=raw_kkt_eligible,
        classic_bic=float(classic_bic),
        bic_refit_finite_candidate_found=bic_refit_finite_candidate_found,
    )

    evaluation = None
    ari_value = np.nan
    cp_rmse_value = np.nan
    raw_cp_rmse_value = np.nan
    summary_cp_rmse_value = np.nan
    bic_refit_cp_rmse_value = np.nan
    multiplicity_f1_value = np.nan
    estimated_clonal_fraction_value = np.nan
    true_clonal_fraction_value = np.nan
    clonal_fraction_error_value = np.nan
    estimated_clusters_value = np.nan
    true_clusters_value = np.nan
    n_eval_mutations_value = np.nan
    n_filtered_mutations_value = np.nan
    evaluation_elapsed_seconds = 0.0
    if evaluate_candidate and simulation_truth is not None:
        evaluation_start_time = perf_counter()
        if ari_only_evaluation:
            ari_value = evaluate_ari_against_simulation(
                fit=fit,
                data=data,
                simulation_truth=simulation_truth,
            )
        else:
            evaluation = evaluate_fit_against_simulation(
                fit=fit,
                data=data,
                simulation_truth=simulation_truth,
                bic_refit_phi=artifact.bic_refit_phi,
                bic_partition_labels=artifact.bic_partition_labels,
            )
            ari_value = float(evaluation.ari)
            cp_rmse_value = float(evaluation.cp_rmse)
            raw_cp_rmse_value = float(
                evaluation.raw_cp_rmse if evaluation.raw_cp_rmse is not None else np.nan
            )
            summary_cp_rmse_value = float(
                evaluation.summary_cp_rmse if evaluation.summary_cp_rmse is not None else evaluation.cp_rmse
            )
            bic_refit_cp_rmse_value = float(
                evaluation.bic_refit_cp_rmse if evaluation.bic_refit_cp_rmse is not None else np.nan
            )
            multiplicity_f1_value = float(evaluation.multiplicity_f1)
            estimated_clonal_fraction_value = float(evaluation.estimated_clonal_fraction)
            true_clonal_fraction_value = float(evaluation.true_clonal_fraction)
            clonal_fraction_error_value = float(evaluation.clonal_fraction_error)
            estimated_clusters_value = int(evaluation.estimated_clusters)
            true_clusters_value = int(evaluation.true_clusters)
            n_eval_mutations_value = int(evaluation.n_eval_mutations)
            n_filtered_mutations_value = int(evaluation.n_filtered_mutations)
        evaluation_elapsed_seconds = float(perf_counter() - evaluation_start_time)

    candidate_elapsed_seconds = float(perf_counter() - candidate_start_time)

    row: dict[str, float | int | str | bool] = {
        "tumor_id": data.tumor_id,
        "selection_method": selection_method,
        "selection_profile": profile_name,
        "selection_step": int(selection_step),
        "lambda": float(fit.lambda_value),
        "lambda_applicable": True,
        "candidate_pool_source": "raw_fused_lambda_path",
        "bic_df_scale": float(bic_df_scale),
        "bic_cluster_penalty": float(bic_cluster_penalty),
        "bic": float(bic),
        "bic_value": float(bic),
        "selection_score_name": str(canonical_score_name),
        "classic_bic": float(classic_bic),
        "extended_bic": float(extended_bic),
        "classic_bic_cell_n": float(classic_bic),
        "classic_bic_depth_n": float(classic_bic_depth_n),
        "classic_bic_active_df": float(classic_bic_active_df),
        "classic_bic_active_df_depth_n": float(classic_bic_active_df_depth_n),
        "bic_loglik": float(bic_loglik),
        "bic_loglik_source": str(bic_refit.loglik_source),
        "bic_df": float(bic_df_value),
        "bic_active_df": float(bic_active_df_value),
        "bic_penalty": float(bic_penalty_value),
        "bic_active_penalty": float(bic_active_penalty_value),
        "bic_n_eff": float(bic_n_eff_value),
        "bic_depth_n_eff": float(bic_depth_n_eff_value),
        "delta_loglik_vs_one_cluster": np.nan,
        "delta_bic_vs_one_cluster": np.nan,
        "bic_partition_tol": float(bic_partition_tol),
        "bic_n_clusters": int(bic_n_clusters),
        "bic_partition_max_diameter": float(bic_partition_max_diameter),
        "bic_partition_diameter_exact": bool(bic_partition_diameter_exact),
        "loglik": float(fit.loglik),
        "raw_loglik": float(fit.loglik),
        "fit_loss": float(-fit.loglik),
        "summary_loglik": float(fit.summary_loglik),
        "refit_loglik": float(bic_loglik),
        "refit_fit_loss": float(bic_refit.fit_loss),
        "refit_finite_candidate_found": bool(bic_refit_finite_candidate_found),
        "bic_refit_finite_candidate_found": bool(bic_refit_finite_candidate_found),
        "refit_global_optimum_certified": bool(bic_refit_global_optimum_certified),
        "bic_refit_global_optimum_certified": bool(bic_refit_global_optimum_certified),
        "bic_refit_coordinate_count": int(bic_refit.refit_coordinate_count),
        "bic_refit_finite_coordinate_count": int(bic_refit.refit_finite_coordinate_count),
        "bic_refit_total_grid_points": int(bic_refit.refit_total_grid_points),
        "bic_refit_max_grid_spacing": float(bic_refit.refit_max_grid_spacing),
        "bic_refit_total_candidate_basins": int(bic_refit.refit_total_candidate_basins),
        "bic_refit_total_refined_candidates": int(bic_refit.refit_total_refined_candidates),
        "bic_refit_min_best_second_loss_gap": float(bic_refit.refit_min_best_second_loss_gap),
        "refit_converged": bool(bic_refit_finite_candidate_found),
        "bic_refit_converged": bool(bic_refit_finite_candidate_found),
        "bic_refit_cache_hit": bool(bic_refit_cache_hit),
        "refit_boundary_count": int(bic_refit.boundary_count),
        "refit_active_df": int(bic_refit.active_degrees_of_freedom),
        "penalized_objective": float(fit.penalized_objective),
        "raw_objective": float(fit.penalized_objective),
        "penalty": float(penalty_value),
        "raw_penalty": float(penalty_value),
        "profile_penalty": float(profile_penalty_value),
        "summary_n_clusters": int(fit.n_clusters),
        "summary_cluster_max_diameter": float(fit.max_cluster_diameter),
        "summary_cluster_diameter_exact": bool(fit.cluster_diameter_exact),
        "n_clusters": int(bic_n_clusters),
        "partition_signature": partition_hash,
        "partition_hash": partition_hash,
        "cluster_sizes": _cluster_sizes_text(bic_labels),
        "converged": bool(fit.converged),
        "raw_fit_status": str(fit.failure_reason),
        "stationarity_certified": bool(fit.stationarity_certified),
        "global_optimality_certified": bool(fit.global_optimality_certified),
        "global_optimality_basis": str(fit.global_optimality_basis),
        "number_of_starts": int(fit.number_of_starts),
        "number_of_finite_starts": int(fit.number_of_finite_starts),
        "best_start_objective": float(fit.best_start_objective),
        "second_best_start_objective": float(fit.second_best_start_objective),
        "objective_spread_across_starts": float(fit.objective_spread_across_starts),
        "selected_start_objective_rank": int(fit.selected_start_objective_rank),
        "converged_inner": bool(fit.converged_inner),
        "converged_outer": bool(fit.converged_outer),
        "iterations": int(fit.iterations),
        "inner_kkt_residual": float(fit.inner_kkt_residual),
        "accepted_inner_kkt_residual": float(fit.accepted_inner_kkt_residual),
        "last_attempted_inner_kkt_residual": float(fit.last_attempted_inner_kkt_residual),
        "best_attempted_inner_kkt_residual": float(fit.best_attempted_inner_kkt_residual),
        "last_attempted_objective_gap": float(fit.last_attempted_objective_gap),
        "best_attempted_objective_gap": float(fit.best_attempted_objective_gap),
        "last_attempted_surrogate_gap": float(fit.last_attempted_surrogate_gap),
        "best_attempted_surrogate_gap": float(fit.best_attempted_surrogate_gap),
        "last_attempted_inner_model_gap": float(fit.last_attempted_inner_model_gap),
        "best_attempted_inner_model_gap": float(fit.best_attempted_inner_model_gap),
        "last_attempted_em_envelope_gap": float(fit.last_attempted_em_envelope_gap),
        "best_attempted_em_envelope_gap": float(fit.best_attempted_em_envelope_gap),
        "outer_stationarity_residual": float(fit.outer_stationarity_residual),
        "outer_projected_stationarity_residual": float(fit.outer_projected_stationarity_residual),
        "outer_projected_stationarity_norm": float(fit.outer_projected_stationarity_norm),
        "outer_stationarity_normalizer": float(fit.outer_stationarity_normalizer),
        "outer_smooth_gradient_norm": float(fit.outer_smooth_gradient_norm),
        "outer_fusion_adjustment_norm": float(fit.outer_fusion_adjustment_norm),
        "outer_edge_subgradient_residual": float(fit.outer_edge_subgradient_residual),
        "outer_dual_ball_residual": float(fit.outer_dual_ball_residual),
        "outer_box_primal_violation": float(fit.outer_box_primal_violation),
        "outer_num_interior_coordinates": int(fit.outer_num_interior_coordinates),
        "outer_num_lower_active_coordinates": int(fit.outer_num_lower_active_coordinates),
        "outer_num_upper_active_coordinates": int(fit.outer_num_upper_active_coordinates),
        "outer_num_frozen_coordinates": int(fit.outer_num_frozen_coordinates),
        "outer_box_residual": float(fit.outer_box_residual),
        "fixed_objective_kkt_residual": float(fit.fixed_objective_kkt_residual),
        "raw_kkt_residual": float(fit.fixed_objective_kkt_residual),
        "outer_kkt_certificate_status": str(fit.outer_kkt_certificate_status),
        "outer_kkt_dual_refined": bool(fit.outer_kkt_dual_refined),
        "outer_kkt_fused_edges": int(fit.outer_kkt_fused_edges),
        "outer_kkt_nonzero_edges": int(fit.outer_kkt_nonzero_edges),
        "outer_stationarity_residual_before_dual_refine": float(fit.outer_stationarity_residual_before_dual_refine),
        "outer_stationarity_residual_after_dual_refine": float(fit.outer_stationarity_residual_after_dual_refine),
        "final_relative_objective_change": float(fit.final_relative_objective_change),
        "final_step_residual": float(fit.final_step_residual),
        "accepted_outer_steps": int(fit.accepted_outer_steps),
        "accepted_full_steps": int(fit.accepted_full_steps),
        "accepted_damped_steps": int(fit.accepted_damped_steps),
        "attempted_outer_steps": int(fit.attempted_outer_steps),
        "failed_majorization_checks": int(fit.failed_majorization_checks),
        "failed_inner_model_checks": int(fit.failed_inner_model_checks),
        "failed_em_envelope_checks": int(fit.failed_em_envelope_checks),
        "failed_descent_checks": int(fit.failed_descent_checks),
        "failed_nonfinite_checks": int(fit.failed_nonfinite_checks),
        "mm_consistency_violations": int(fit.mm_consistency_violations),
        "accepted_step_type": str(fit.accepted_step_type),
        "last_reject_reason": str(fit.last_reject_reason),
        "failure_reason": str(fit.failure_reason),
        "selection_eligible": bic_selection_eligible,
        "raw_kkt_eligible": raw_kkt_eligible,
        "bic_selection_eligible": bic_selection_eligible,
        "candidate_elapsed_seconds": float(candidate_elapsed_seconds),
        "raw_fit_elapsed_seconds": float(raw_fit_elapsed_seconds),
        "bic_refit_elapsed_seconds": float(bic_refit_elapsed_seconds),
        "candidate_evaluation_elapsed_seconds": float(evaluation_elapsed_seconds),
        "partition_tol": float(bic_partition_tol),
        "primary_phi_source": "raw_penalized_fit",
        "bic_refit_phi_source": "secondary_partition_refit",
        "device": str(fit.device),
        "dtype": str(fit.dtype),
        "tol": float(effective_fit_options.tol),
        "outer_max_iter": int(effective_fit_options.outer_max_iter),
        "inner_max_iter": int(effective_fit_options.inner_max_iter),
        "summary_tol": float(effective_fit_options.summary_tol)
        if effective_fit_options.summary_tol is not None
        else np.nan,
        "eps": float(effective_fit_options.eps),
        "major_prior": float(effective_fit_options.major_prior),
        "graph_name": str(fit.graph_name),
        "num_edges": int(candidate_static.edge_count),
        "edge_weight_min": float(candidate_static.edge_weight_min),
        "edge_weight_max": float(candidate_static.edge_weight_max),
        "edge_weight_mean": float(candidate_static.edge_weight_mean),
        "adaptive_weight_gamma": float(effective_fit_options.adaptive_weight_gamma),
        "adaptive_weight_floor": float(effective_fit_options.adaptive_weight_floor),
        "adaptive_weight_baseline": float(effective_fit_options.adaptive_weight_baseline),
        "edge_list_hash": str(candidate_static.edge_list_hash),
        "pilot_matrix_hash": str(candidate_static.pilot_matrix_hash),
        "input_data_hash": str(candidate_static.input_data_hash),
        "evaluation_mode": "ari_only" if ari_only_evaluation else "full",
        "fit_compute_summary": bool(compute_summary),
        "fit_start_mode": str(start_mode),
        "solver_state_warm_start": bool(solver_state is not None),
        "ARI": ari_value,
        "cp_rmse": cp_rmse_value,
        "raw_cp_rmse": raw_cp_rmse_value,
        "summary_cp_rmse": summary_cp_rmse_value,
        "bic_refit_cp_rmse": bic_refit_cp_rmse_value,
        "multiplicity_f1": multiplicity_f1_value,
        "estimated_clonal_fraction": estimated_clonal_fraction_value,
        "true_clonal_fraction": true_clonal_fraction_value,
        "clonal_fraction_error": clonal_fraction_error_value,
        "estimated_clusters": estimated_clusters_value,
        "true_clusters": true_clusters_value,
        "n_eval_mutations": n_eval_mutations_value,
        "n_filtered_mutations": n_filtered_mutations_value,
    }
    return fit, evaluation, row, artifact



def _evaluate_partition_candidate(
    *,
    data: TumorData,
    fit_options: FitOptions,
    candidate: PartitionCandidate,
    candidate_rank: int,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    simulation_truth: SimulationTruth | None,
    evaluate_candidate: bool,
    ari_only_evaluation: bool,
    selection_method: str,
    profile_name: str,
    selection_step: int,
    selection_score: str,
    static_metadata: CandidateStaticMetadata,
) -> tuple[FitResult, SimulationEvaluation | None, dict[str, float | int | str | bool], SelectionArtifact]:
    candidate_start_time = perf_counter()
    canonical_score_name = _normalize_selection_score_name(selection_score)
    graph = fit_options.graph
    if graph is None:
        raise RuntimeError("Partition candidates require a resolved pairwise-fusion graph.")

    labels = _canonical_partition_labels(np.asarray(candidate.labels, dtype=np.int64))
    num_clusters = int(np.unique(labels).size)
    phi = np.asarray(candidate.phi_start, dtype=np.float64)
    centers = np.asarray(candidate.theta, dtype=np.float64)
    if centers.shape != (num_clusters, phi.shape[1]):
        centers = _centers_from_partition_labels(phi, labels, num_clusters)
    phi_clustered = centers[labels].astype(np.float64, copy=False)
    fit_loss = float(candidate.fit_loss)
    loglik = float(-fit_loss)

    bic, classic_bic, extended_bic = _selection_score_value(
        loglik=loglik,
        num_clusters=num_clusters,
        data=data,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
        selection_score=selection_score,
    )
    classic_bic_depth_n = compute_classic_bic_depth_n(loglik, num_clusters, data)
    bic_df_value = float(bic_degrees_of_freedom(num_clusters, data))
    active_df_value = float(candidate.active_df) if candidate.active_df is not None else bic_df_value
    bic_n_eff_value = float(effective_bic_cell_count(data))
    bic_depth_n_eff_value = float(effective_bic_depth_count(data))
    classic_bic_active_df = compute_bic_with_df(loglik, active_df_value, bic_n_eff_value)
    classic_bic_active_df_depth_n = compute_bic_with_df(
        loglik,
        active_df_value,
        bic_depth_n_eff_value,
    )
    bic_penalty_value = float(bic_df_value * np.log(max(bic_n_eff_value, 1.0)))
    bic_active_penalty_value = float(active_df_value * np.log(max(bic_n_eff_value, 1.0)))

    diameters, diameter_exact = cluster_diameters_from_edges(
        phi_clustered,
        labels,
        edge_u=graph.edge_u,
        edge_v=graph.edge_v,
    )
    max_diameter = _max_cluster_diameter(diameters)
    major_probability, major_call, multiplicity_call, multiplicity_mask = _multiplicity_summary_for_phi(
        data,
        phi_clustered,
    )
    summary_tol = (
        max(10.0 * float(fit_options.tol), 1e-4)
        if fit_options.summary_tol is None
        else max(float(fit_options.summary_tol), 1e-12)
    )
    bic_partition_tol = _effective_bic_partition_tol(fit_options)
    boundary_count = int(candidate.diagnostics.get("refit_boundary_count", -1.0))
    refit_coordinate_count = int(
        candidate.diagnostics.get("refit_coordinate_count", num_clusters * int(data.num_samples))
    )
    refit_finite_coordinate_count = int(
        candidate.diagnostics.get(
            "refit_finite_coordinate_count",
            refit_coordinate_count if candidate.finite_candidate_found else 0,
        )
    )
    refit_total_grid_points = int(candidate.diagnostics.get("refit_total_grid_points", -1.0))
    refit_max_grid_spacing = float(candidate.diagnostics.get("refit_max_grid_spacing", np.nan))
    refit_total_candidate_basins = int(candidate.diagnostics.get("refit_total_candidate_basins", -1.0))
    refit_total_refined_candidates = int(candidate.diagnostics.get("refit_total_refined_candidates", -1.0))
    refit_min_best_second_loss_gap = float(candidate.diagnostics.get("refit_min_best_second_loss_gap", np.nan))
    partition_generation_cuda = bool(candidate.diagnostics.get("partition_generation_cuda", 0.0))
    finite_candidate_found = bool(candidate.finite_candidate_found and np.isfinite(float(classic_bic)))
    bic_selection_eligible = bool(finite_candidate_found and np.isfinite(float(classic_bic)))
    partition_hash = _partition_signature(labels)

    fit = FitResult(
        phi=phi_clustered,
        phi_clustered=phi_clustered,
        cluster_labels=labels.astype(np.int64, copy=False),
        cluster_centers=centers.astype(np.float64, copy=False),
        cluster_diameters=diameters.astype(np.float64, copy=False),
        max_cluster_diameter=float(max_diameter),
        cluster_diameter_exact=bool(diameter_exact),
        gamma_major=major_probability,
        major_probability=major_probability,
        major_call=major_call,
        multiplicity_call=multiplicity_call,
        multiplicity_estimated_mask=multiplicity_mask,
        loglik=loglik,
        summary_loglik=loglik,
        penalized_objective=float("nan"),
        lambda_value=float("nan"),
        n_clusters=num_clusters,
        iterations=0,
        converged=finite_candidate_found,
        device=str(fit_options.device),
        dtype=str(fit_options.dtype),
        graph_name=str(graph.name),
        summary_tol=float(summary_tol),
        summary_available=True,
        inner_kkt_residual=float("nan"),
        accepted_inner_kkt_residual=float("nan"),
        last_attempted_inner_kkt_residual=float("nan"),
        best_attempted_inner_kkt_residual=float("nan"),
        last_attempted_objective_gap=float("nan"),
        best_attempted_objective_gap=float("nan"),
        last_attempted_surrogate_gap=float("nan"),
        best_attempted_surrogate_gap=float("nan"),
        last_attempted_inner_model_gap=float("nan"),
        best_attempted_inner_model_gap=float("nan"),
        last_attempted_em_envelope_gap=float("nan"),
        best_attempted_em_envelope_gap=float("nan"),
        outer_stationarity_residual=float("nan"),
        outer_projected_stationarity_residual=float("nan"),
        outer_projected_stationarity_norm=float("nan"),
        outer_stationarity_normalizer=float("nan"),
        outer_smooth_gradient_norm=float("nan"),
        outer_fusion_adjustment_norm=float("nan"),
        outer_edge_subgradient_residual=float("nan"),
        outer_dual_ball_residual=float("nan"),
        outer_box_primal_violation=0.0,
        outer_num_interior_coordinates=0,
        outer_num_lower_active_coordinates=0,
        outer_num_upper_active_coordinates=0,
        outer_num_frozen_coordinates=0,
        outer_box_residual=0.0,
        fixed_objective_kkt_residual=float("nan"),
        outer_kkt_certificate_status="not_applicable_partition_candidate",
        outer_kkt_dual_refined=False,
        outer_kkt_fused_edges=0,
        outer_kkt_nonzero_edges=0,
        outer_stationarity_residual_before_dual_refine=float("nan"),
        outer_stationarity_residual_after_dual_refine=float("nan"),
        converged_inner=finite_candidate_found,
        converged_outer=finite_candidate_found,
        final_relative_objective_change=0.0,
        final_step_residual=0.0,
        accepted_outer_steps=0,
        accepted_full_steps=0,
        accepted_damped_steps=0,
        attempted_outer_steps=0,
        failed_majorization_checks=0,
        failed_inner_model_checks=0,
        failed_em_envelope_checks=0,
        failed_descent_checks=0,
        failed_nonfinite_checks=0,
        mm_consistency_violations=0,
        accepted_step_type="partition_refit",
        last_reject_reason="none",
        failure_reason="likelihood_partition_candidate_refit",
        selection_eligible=bool(bic_selection_eligible),
        stationarity_certified=False,
        global_optimality_certified=False,
        global_optimality_basis="partition_refit_unimodal_coordinate_search",
        number_of_starts=1,
        number_of_finite_starts=1 if finite_candidate_found else 0,
        best_start_objective=fit_loss,
        second_best_start_objective=float("nan"),
        objective_spread_across_starts=0.0,
        selected_start_objective_rank=1,
        history=[],
        solver_state=None,
    )
    artifact = SelectionArtifact(
        bic=float(bic),
        classic_bic=float(classic_bic),
        extended_bic=float(extended_bic),
        classic_bic_depth_n=float(classic_bic_depth_n),
        classic_bic_active_df=float(classic_bic_active_df),
        classic_bic_active_df_depth_n=float(classic_bic_active_df_depth_n),
        bic_loglik=float(loglik),
        bic_loglik_source="likelihood_partition_candidate_refit",
        bic_df=float(bic_df_value),
        bic_active_df=float(active_df_value),
        bic_n_eff=float(bic_n_eff_value),
        bic_depth_n_eff=float(bic_depth_n_eff_value),
        bic_partition_tol=float(bic_partition_tol),
        bic_refit_boundary_count=int(boundary_count),
        bic_refit_finite_candidate_found=bool(finite_candidate_found),
        bic_refit_global_optimum_certified=False,
        bic_refit_coordinate_count=int(refit_coordinate_count),
        bic_refit_finite_coordinate_count=int(refit_finite_coordinate_count),
        bic_refit_total_grid_points=int(refit_total_grid_points),
        bic_refit_max_grid_spacing=float(refit_max_grid_spacing),
        bic_refit_total_candidate_basins=int(refit_total_candidate_basins),
        bic_refit_total_refined_candidates=int(refit_total_refined_candidates),
        bic_refit_min_best_second_loss_gap=float(refit_min_best_second_loss_gap),
        bic_refit_converged=bool(finite_candidate_found),
        bic_refit_phi=phi_clustered,
        bic_refit_cluster_centers=centers,
        bic_partition_labels=labels.astype(np.int64, copy=False),
        selection_score_name=str(canonical_score_name),
    )

    evaluation = None
    ari_value = np.nan
    cp_rmse_value = np.nan
    raw_cp_rmse_value = np.nan
    summary_cp_rmse_value = np.nan
    bic_refit_cp_rmse_value = np.nan
    multiplicity_f1_value = np.nan
    estimated_clonal_fraction_value = np.nan
    true_clonal_fraction_value = np.nan
    clonal_fraction_error_value = np.nan
    estimated_clusters_value = np.nan
    true_clusters_value = np.nan
    n_eval_mutations_value = np.nan
    n_filtered_mutations_value = np.nan
    evaluation_elapsed_seconds = 0.0
    if evaluate_candidate and simulation_truth is not None:
        evaluation_start_time = perf_counter()
        if ari_only_evaluation:
            ari_value = evaluate_ari_against_simulation(
                fit=fit,
                data=data,
                simulation_truth=simulation_truth,
            )
        else:
            evaluation = evaluate_fit_against_simulation(
                fit=fit,
                data=data,
                simulation_truth=simulation_truth,
                bic_refit_phi=artifact.bic_refit_phi,
                bic_partition_labels=artifact.bic_partition_labels,
            )
            ari_value = float(evaluation.ari)
            cp_rmse_value = float(evaluation.cp_rmse)
            raw_cp_rmse_value = float(
                evaluation.raw_cp_rmse if evaluation.raw_cp_rmse is not None else np.nan
            )
            summary_cp_rmse_value = float(
                evaluation.summary_cp_rmse if evaluation.summary_cp_rmse is not None else evaluation.cp_rmse
            )
            bic_refit_cp_rmse_value = float(
                evaluation.bic_refit_cp_rmse if evaluation.bic_refit_cp_rmse is not None else np.nan
            )
            multiplicity_f1_value = float(evaluation.multiplicity_f1)
            estimated_clonal_fraction_value = float(evaluation.estimated_clonal_fraction)
            true_clonal_fraction_value = float(evaluation.true_clonal_fraction)
            clonal_fraction_error_value = float(evaluation.clonal_fraction_error)
            estimated_clusters_value = int(evaluation.estimated_clusters)
            true_clusters_value = int(evaluation.true_clusters)
            n_eval_mutations_value = int(evaluation.n_eval_mutations)
            n_filtered_mutations_value = int(evaluation.n_filtered_mutations)
        evaluation_elapsed_seconds = float(perf_counter() - evaluation_start_time)

    candidate_elapsed_seconds = float(perf_counter() - candidate_start_time)

    requested_k = int(candidate.diagnostics.get("requested_K", candidate.K))
    row: dict[str, float | int | str | bool] = {
        "tumor_id": data.tumor_id,
        "selection_method": selection_method,
        "selection_profile": profile_name,
        "selection_step": int(selection_step),
        "lambda": float(LIKELIHOOD_PARTITION_SENTINEL_LAMBDA),
        "lambda_applicable": False,
        "candidate_pool_source": "likelihood_partition",
        "partition_candidate_source": str(candidate.source),
        "partition_candidate_rank": int(candidate_rank),
        "partition_candidate_requested_K": int(requested_k),
        "partition_candidate_K": int(num_clusters),
        "partition_candidate_fit_loss": float(fit_loss),
        "partition_candidate_bic": float(classic_bic),
        "partition_generation_cuda": bool(partition_generation_cuda),
        "bic_df_scale": float(bic_df_scale),
        "bic_cluster_penalty": float(bic_cluster_penalty),
        "bic": float(bic),
        "bic_value": float(bic),
        "selection_score_name": str(canonical_score_name),
        "classic_bic": float(classic_bic),
        "extended_bic": float(extended_bic),
        "classic_bic_cell_n": float(classic_bic),
        "classic_bic_depth_n": float(classic_bic_depth_n),
        "classic_bic_active_df": float(classic_bic_active_df),
        "classic_bic_active_df_depth_n": float(classic_bic_active_df_depth_n),
        "bic_loglik": float(loglik),
        "bic_loglik_source": "likelihood_partition_candidate_refit",
        "bic_df": float(bic_df_value),
        "bic_active_df": float(active_df_value),
        "bic_penalty": float(bic_penalty_value),
        "bic_active_penalty": float(bic_active_penalty_value),
        "bic_n_eff": float(bic_n_eff_value),
        "bic_depth_n_eff": float(bic_depth_n_eff_value),
        "delta_loglik_vs_one_cluster": np.nan,
        "delta_bic_vs_one_cluster": np.nan,
        "bic_partition_tol": float(bic_partition_tol),
        "bic_n_clusters": int(num_clusters),
        "bic_partition_max_diameter": float(max_diameter),
        "bic_partition_diameter_exact": bool(diameter_exact),
        "loglik": float(loglik),
        "raw_loglik": float(loglik),
        "fit_loss": float(fit_loss),
        "summary_loglik": float(loglik),
        "refit_loglik": float(loglik),
        "refit_fit_loss": float(fit_loss),
        "refit_finite_candidate_found": bool(finite_candidate_found),
        "bic_refit_finite_candidate_found": bool(finite_candidate_found),
        "refit_global_optimum_certified": False,
        "bic_refit_global_optimum_certified": False,
        "bic_refit_coordinate_count": int(refit_coordinate_count),
        "bic_refit_finite_coordinate_count": int(refit_finite_coordinate_count),
        "bic_refit_total_grid_points": int(refit_total_grid_points),
        "bic_refit_max_grid_spacing": float(refit_max_grid_spacing),
        "bic_refit_total_candidate_basins": int(refit_total_candidate_basins),
        "bic_refit_total_refined_candidates": int(refit_total_refined_candidates),
        "bic_refit_min_best_second_loss_gap": float(refit_min_best_second_loss_gap),
        "refit_converged": bool(finite_candidate_found),
        "bic_refit_converged": bool(finite_candidate_found),
        "bic_refit_cache_hit": False,
        "refit_boundary_count": int(boundary_count),
        "refit_active_df": int(active_df_value),
        "penalized_objective": np.nan,
        "raw_objective": np.nan,
        "penalty": np.nan,
        "raw_penalty": np.nan,
        "profile_penalty": np.nan,
        "summary_n_clusters": int(num_clusters),
        "summary_cluster_max_diameter": float(max_diameter),
        "summary_cluster_diameter_exact": bool(diameter_exact),
        "n_clusters": int(num_clusters),
        "partition_signature": partition_hash,
        "partition_hash": partition_hash,
        "cluster_sizes": _cluster_sizes_text(labels),
        "converged": bool(finite_candidate_found),
        "raw_fit_status": "not_raw_fused_fit",
        "stationarity_certified": False,
        "global_optimality_certified": False,
        "global_optimality_basis": "partition_refit_unimodal_coordinate_search",
        "number_of_starts": 1,
        "number_of_finite_starts": 1 if finite_candidate_found else 0,
        "best_start_objective": float(fit_loss),
        "second_best_start_objective": np.nan,
        "objective_spread_across_starts": 0.0,
        "selected_start_objective_rank": 1,
        "converged_inner": bool(finite_candidate_found),
        "converged_outer": bool(finite_candidate_found),
        "iterations": 0,
        "inner_kkt_residual": np.nan,
        "accepted_inner_kkt_residual": np.nan,
        "last_attempted_inner_kkt_residual": np.nan,
        "best_attempted_inner_kkt_residual": np.nan,
        "last_attempted_objective_gap": np.nan,
        "best_attempted_objective_gap": np.nan,
        "last_attempted_surrogate_gap": np.nan,
        "best_attempted_surrogate_gap": np.nan,
        "last_attempted_inner_model_gap": np.nan,
        "best_attempted_inner_model_gap": np.nan,
        "last_attempted_em_envelope_gap": np.nan,
        "best_attempted_em_envelope_gap": np.nan,
        "outer_stationarity_residual": np.nan,
        "outer_projected_stationarity_residual": np.nan,
        "outer_projected_stationarity_norm": np.nan,
        "outer_stationarity_normalizer": np.nan,
        "outer_smooth_gradient_norm": np.nan,
        "outer_fusion_adjustment_norm": np.nan,
        "outer_edge_subgradient_residual": np.nan,
        "outer_dual_ball_residual": np.nan,
        "outer_box_primal_violation": 0.0,
        "outer_num_interior_coordinates": 0,
        "outer_num_lower_active_coordinates": 0,
        "outer_num_upper_active_coordinates": 0,
        "outer_num_frozen_coordinates": 0,
        "outer_box_residual": 0.0,
        "fixed_objective_kkt_residual": np.nan,
        "raw_kkt_residual": np.nan,
        "outer_kkt_certificate_status": "not_applicable_partition_candidate",
        "outer_kkt_dual_refined": False,
        "outer_kkt_fused_edges": 0,
        "outer_kkt_nonzero_edges": 0,
        "outer_stationarity_residual_before_dual_refine": np.nan,
        "outer_stationarity_residual_after_dual_refine": np.nan,
        "final_relative_objective_change": 0.0,
        "final_step_residual": 0.0,
        "accepted_outer_steps": 0,
        "accepted_full_steps": 0,
        "accepted_damped_steps": 0,
        "attempted_outer_steps": 0,
        "failed_majorization_checks": 0,
        "failed_inner_model_checks": 0,
        "failed_em_envelope_checks": 0,
        "failed_descent_checks": 0,
        "failed_nonfinite_checks": 0,
        "mm_consistency_violations": 0,
        "accepted_step_type": "partition_refit",
        "last_reject_reason": "none",
        "failure_reason": "likelihood_partition_candidate_refit",
        "selection_eligible": bool(bic_selection_eligible),
        "raw_kkt_eligible": False,
        "bic_selection_eligible": bool(bic_selection_eligible),
        "candidate_elapsed_seconds": float(candidate_elapsed_seconds),
        "raw_fit_elapsed_seconds": 0.0,
        "bic_refit_elapsed_seconds": 0.0,
        "candidate_evaluation_elapsed_seconds": float(evaluation_elapsed_seconds),
        "partition_tol": float(bic_partition_tol),
        "primary_phi_source": "likelihood_partition_refit",
        "bic_refit_phi_source": "likelihood_partition_refit",
        "device": str(fit.device),
        "dtype": str(fit.dtype),
        "tol": float(fit_options.tol),
        "outer_max_iter": int(fit_options.outer_max_iter),
        "inner_max_iter": int(fit_options.inner_max_iter),
        "summary_tol": float(summary_tol),
        "eps": float(fit_options.eps),
        "major_prior": float(fit_options.major_prior),
        "graph_name": str(fit.graph_name),
        "num_edges": int(static_metadata.edge_count),
        "edge_weight_min": float(static_metadata.edge_weight_min),
        "edge_weight_max": float(static_metadata.edge_weight_max),
        "edge_weight_mean": float(static_metadata.edge_weight_mean),
        "adaptive_weight_gamma": float(fit_options.adaptive_weight_gamma),
        "adaptive_weight_floor": float(fit_options.adaptive_weight_floor),
        "adaptive_weight_baseline": float(fit_options.adaptive_weight_baseline),
        "edge_list_hash": str(static_metadata.edge_list_hash),
        "pilot_matrix_hash": str(static_metadata.pilot_matrix_hash),
        "input_data_hash": str(static_metadata.input_data_hash),
        "evaluation_mode": "ari_only" if ari_only_evaluation else "full",
        "fit_compute_summary": True,
        "fit_start_mode": "likelihood_partition",
        "solver_state_warm_start": False,
        "ARI": ari_value,
        "cp_rmse": cp_rmse_value,
        "raw_cp_rmse": raw_cp_rmse_value,
        "summary_cp_rmse": summary_cp_rmse_value,
        "bic_refit_cp_rmse": bic_refit_cp_rmse_value,
        "multiplicity_f1": multiplicity_f1_value,
        "estimated_clonal_fraction": estimated_clonal_fraction_value,
        "true_clonal_fraction": true_clonal_fraction_value,
        "clonal_fraction_error": clonal_fraction_error_value,
        "estimated_clusters": estimated_clusters_value,
        "true_clusters": true_clusters_value,
        "n_eval_mutations": n_eval_mutations_value,
        "n_filtered_mutations": n_filtered_mutations_value,
    }
    return fit, evaluation, row, artifact


