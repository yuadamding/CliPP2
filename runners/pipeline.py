from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .._version import __version__ as _SOFTWARE_VERSION
from ..core.fusion.graph import load_pairwise_fusion_graph_tsv
from ..core.model import FitOptions
from ..io.data import load_tumor_tsv
from ..metrics.evaluation import evaluate_fit_against_simulation
from .model_selection import select_model
from .outputs import write_fit_outputs
from .settings import summarize_tumor_regime


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def process_one_file_bundle(
    file_path: str | Path,
    outdir: str | Path,
    simulation_root: str | Path | None = None,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "adaptive_bic",
    fit_options: FitOptions | None = None,
    bic_df_scale: float = 1.0,
    bic_cluster_penalty: float = 0.0,
    selection_score: str = "bic",
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    graph_file: str | Path | None = None,
    finalize_selected_fit: bool | None = None,
    missing_cna_policy: str = "error",
    evaluate_all_candidates: bool | None = None,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame]:
    start_time = perf_counter()
    file_path = Path(file_path)
    outdir = Path(outdir)
    data = load_tumor_tsv(file_path, missing_cna_policy=missing_cna_policy)

    if fit_options is None:
        fit_options = FitOptions(lambda_value=0.0)
    if graph_file is not None:
        fit_options = replace(
            fit_options,
            graph=load_pairwise_fusion_graph_tsv(
                graph_file,
                num_mutations=data.num_mutations,
                mutation_ids=data.mutation_ids,
            ),
        )

    tumor_regime = summarize_tumor_regime(data)
    simulation_available = simulation_root is not None and (Path(simulation_root) / data.tumor_id).exists()
    if finalize_selected_fit is None:
        finalize_selected_fit = True
    evaluate_all_candidates_flag = (
        simulation_available and write_outputs
        if evaluate_all_candidates is None
        else simulation_available and bool(evaluate_all_candidates)
    )
    selection_result = select_model(
        data=data,
        simulation_root=Path(simulation_root) if simulation_root is not None else None,
        lambda_grid=lambda_grid,
        lambda_grid_mode=lambda_grid_mode,
        fit_options=fit_options,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
        selection_score=selection_score,
        use_warm_starts=use_warm_starts,
        evaluate_all_candidates=evaluate_all_candidates_flag,
        finalize_selected_fit=bool(finalize_selected_fit),
    )
    best_fit = selection_result.best_fit
    selection_artifact = getattr(selection_result, "selected_artifact", None)
    simulation_diagnostics = getattr(selection_result, "simulation", selection_result)
    best_evaluation = getattr(
        simulation_diagnostics,
        "selected_evaluation",
        getattr(selection_result, "best_evaluation", None),
    )
    search_df = selection_result.search_df
    selected_search_row: dict[str, object] = {}
    if not search_df.empty and "is_selected_best_row" in search_df.columns:
        selected_mask = search_df["is_selected_best_row"].astype(bool).to_numpy(dtype=bool)
        if np.any(selected_mask):
            selected_search_row = search_df.loc[selected_mask].iloc[0].to_dict()

    def _selected_value(name: str, default: object = np.nan) -> object:
        return selected_search_row.get(name, default)

    def _artifact_value(name: str, default: object = None) -> object:
        if selection_artifact is not None and hasattr(selection_artifact, name):
            return getattr(selection_artifact, name)
        return getattr(best_fit, name, default)

    def _artifact_float(name: str, default: float = np.nan) -> float:
        value = _artifact_value(name, None)
        return float(default) if value is None else float(value)

    def _artifact_int(name: str, default: int = -1) -> int:
        value = _artifact_value(name, None)
        return int(default) if value is None else int(value)

    def _artifact_bool(name: str, default: bool = False) -> bool:
        value = _artifact_value(name, None)
        return bool(default) if value is None else bool(value)

    def _selected_bool(name: str, default: bool = False) -> bool:
        value = _selected_value(name, default)
        if value is None:
            return bool(default)
        try:
            if pd.isna(value):
                return bool(default)
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"0", "false", "f", "no", "n", ""}:
                return False
            if normalized in {"1", "true", "t", "yes", "y"}:
                return True
        return bool(value)

    selected_lambda_applicable = _selected_bool("lambda_applicable", True)
    selected_lambda_value = float(best_fit.lambda_value) if selected_lambda_applicable else np.nan
    selected_lambda_representative = (
        np.nan
        if not selected_lambda_applicable or selection_result.selected_lambda_representative is None
        else float(selection_result.selected_lambda_representative)
    )
    if not search_df.empty:
        if "lambda_applicable" in search_df.columns:
            lambda_applicable_mask = search_df["lambda_applicable"].fillna(True).astype(bool).to_numpy(dtype=bool)
        else:
            lambda_applicable_mask = np.ones(search_df.shape[0], dtype=bool)
        tested_lambda_values = search_df.loc[lambda_applicable_mask, "lambda"].to_numpy(dtype=float)
        tested_lambda_values = tested_lambda_values[
            np.isfinite(tested_lambda_values) & (tested_lambda_values >= 0.0)
        ]
    else:
        tested_lambda_values = np.asarray([], dtype=float)
    tested_lambda_min = float(np.min(tested_lambda_values)) if tested_lambda_values.size else np.nan
    tested_lambda_max = float(np.max(tested_lambda_values)) if tested_lambda_values.size else np.nan
    tested_lambda_count = int(np.unique(np.round(tested_lambda_values, 12)).size)

    def _selected_str(name: str, default: str = "") -> str:
        value = _selected_value(name, default)
        if pd.isna(value):
            return str(default)
        return str(value)

    def _selected_int(name: str, default: int = -1) -> int:
        value = _selected_value(name, default)
        if pd.isna(value):
            return int(default)
        return int(value)

    if best_evaluation is None and simulation_available and bool(finalize_selected_fit):
        best_evaluation = evaluate_fit_against_simulation(
            fit=best_fit,
            data=data,
            simulation_root=simulation_root,
            bic_refit_phi=_artifact_value("bic_refit_phi"),
            bic_partition_labels=_artifact_value("bic_partition_labels"),
        )

    elapsed_seconds = float(perf_counter() - start_time)
    def _timing_sum(column: str, mask: pd.Series | None = None) -> float:
        if search_df.empty or column not in search_df.columns:
            return float(np.nan)
        values = search_df[column] if mask is None else search_df.loc[mask, column]
        numeric = pd.to_numeric(values, errors="coerce")
        return float(numeric.sum(skipna=True))

    def _timing_max(column: str) -> float:
        if search_df.empty or column not in search_df.columns:
            return float(np.nan)
        numeric = pd.to_numeric(search_df[column], errors="coerce")
        return float(numeric.max(skipna=True)) if numeric.notna().any() else float(np.nan)

    raw_candidate_mask = (
        search_df["candidate_pool_source"].astype(str).eq("raw_fused_lambda_path")
        if "candidate_pool_source" in search_df.columns
        else pd.Series(False, index=search_df.index)
    )
    partition_candidate_mask = (
        search_df["candidate_pool_source"].astype(str).eq("likelihood_partition")
        if "candidate_pool_source" in search_df.columns
        else pd.Series(False, index=search_df.index)
    )
    reported_selected_ari = (
        float(simulation_diagnostics.selected_ari)
        if simulation_diagnostics.selected_ari is not None
        else float(best_evaluation.ari)
        if best_evaluation is not None
        else None
    )
    selected_model_token = "selected_lambda" if selected_lambda_applicable else "selected_model"
    selected_ari_source = (
        f"candidate_{selected_model_token}"
        if simulation_diagnostics.selected_ari is not None
        else f"final_evaluation_{selected_model_token}"
        if best_evaluation is not None
        else "not_available"
    )
    best_classic_bic = _artifact_value("classic_bic")
    best_bic_refit_finite = (
        bool(_artifact_value("bic_refit_finite_candidate_found"))
        if _artifact_value("bic_refit_finite_candidate_found") is not None
        else bool(_artifact_value("bic_refit_converged"))
        if _artifact_value("bic_refit_converged") is not None
        else False
    )
    best_bic_refit_global = bool(_artifact_value("bic_refit_global_optimum_certified", False))

    summary = {
        "tumor_id": data.tumor_id,
        "estimator": "objective_faithful_pairwise_fusion",
        "selected_lambda": selected_lambda_value,
        "selected_lambda_applicable": bool(selected_lambda_applicable),
        "selected_candidate_pool_source": _selected_str("candidate_pool_source", "raw_fused_lambda_path"),
        "selected_partition_candidate_source": _selected_str("partition_candidate_source", ""),
        "selected_partition_candidate_requested_K": _selected_int("partition_candidate_requested_K", -1),
        "loglik": float(best_fit.loglik),
        "summary_loglik": float(best_fit.summary_loglik),
        "bic": _artifact_float("bic"),
        "classic_bic": float(best_classic_bic if best_classic_bic is not None else np.nan),
        "extended_bic": _artifact_float("extended_bic"),
        "n_clusters": int(best_fit.n_clusters),
        "selection_profile": selection_result.profile_name,
        "selection_method": selection_result.selection_method,
        "selection_score_name": str(_artifact_value("selection_score_name", None) or selection_score),
        "lambda_search_mode": str(selection_result.lambda_search_mode),
        "selected_lambda_representative": selected_lambda_representative,
        "selected_lambda_left": np.nan
        if selection_result.selected_lambda_left is None
        else float(selection_result.selected_lambda_left),
        "selected_lambda_right": np.nan
        if selection_result.selected_lambda_right is None
        else float(selection_result.selected_lambda_right),
        "selected_lambda_interval_log10_width": np.nan
        if selection_result.selected_lambda_interval_log10_width is None
        else float(selection_result.selected_lambda_interval_log10_width),
        "lambda_bracket_min": np.nan
        if selection_result.lambda_bracket_min is None
        else float(selection_result.lambda_bracket_min),
        "lambda_bracket_eq": np.nan
        if selection_result.lambda_bracket_eq is None
        else float(selection_result.lambda_bracket_eq),
        "lambda_bracket_full": np.nan
        if selection_result.lambda_bracket_full is None
        else float(selection_result.lambda_bracket_full),
        "adaptive_refinement_rounds_completed": int(selection_result.adaptive_refinement_rounds_completed),
        "adaptive_candidate_budget": _selected_int("adaptive_candidate_budget", -1),
        "adaptive_initial_anchor_count": _selected_int("adaptive_initial_anchor_count", -1),
        "adaptive_transition_probe_max_candidates": _selected_int(
            "adaptive_transition_probe_max_candidates",
            -1,
        ),
        "likelihood_partition_pool_enabled": _selected_bool("likelihood_partition_pool_enabled", False),
        "selection_elapsed_seconds": _timing_max("selection_elapsed_seconds"),
        "selection_prepare_elapsed_seconds": _timing_max("selection_prepare_elapsed_seconds"),
        "raw_candidate_elapsed_seconds": _timing_sum("candidate_elapsed_seconds", raw_candidate_mask),
        "raw_fit_elapsed_seconds": _timing_sum("raw_fit_elapsed_seconds", raw_candidate_mask),
        "raw_bic_refit_elapsed_seconds": _timing_sum("bic_refit_elapsed_seconds", raw_candidate_mask),
        "raw_candidate_evaluation_elapsed_seconds": _timing_sum(
            "candidate_evaluation_elapsed_seconds",
            raw_candidate_mask,
        ),
        "partition_candidate_elapsed_seconds": _timing_sum(
            "candidate_elapsed_seconds",
            partition_candidate_mask,
        ),
        "partition_generation_elapsed_seconds": _timing_max("partition_generation_elapsed_seconds"),
        "partition_curvature_elapsed_seconds": _timing_max("partition_curvature_elapsed_seconds"),
        "partition_ward_elapsed_seconds": _timing_max("partition_ward_elapsed_seconds"),
        "partition_initial_generation_elapsed_seconds": _timing_max(
            "partition_initial_generation_elapsed_seconds"
        ),
        "partition_refine_generation_elapsed_seconds": _timing_max(
            "partition_refine_generation_elapsed_seconds"
        ),
        "selection_loglik_kind": "partition_constrained_observed_mle",
        "finalize_selected_fit": bool(finalize_selected_fit),
        "evaluate_all_candidates": bool(evaluate_all_candidates_flag),
        "selection_used_convergence_fallback": bool(selection_result.selection_used_convergence_fallback),
        "num_candidates": int(selection_result.num_candidates),
        "num_converged_candidates": int(selection_result.num_converged_candidates),
        "num_candidates_all": int(getattr(selection_result, "num_candidates_all", selection_result.num_candidates)),
        "num_candidates_certified": int(
            getattr(selection_result, "num_candidates_certified", selection_result.num_converged_candidates)
        ),
        "selected_kkt_residual": np.nan
        if getattr(selection_result, "selected_kkt_residual", None) is None
        else float(selection_result.selected_kkt_residual),
        "best_score_all_evaluated_lambda": np.nan
        if getattr(selection_result, "best_score_all_evaluated_lambda", None) is None
        else float(selection_result.best_score_all_evaluated_lambda),
        "best_score_all_evaluated_kkt_residual": np.nan
        if getattr(selection_result, "best_score_all_evaluated_kkt_residual", None) is None
        else float(selection_result.best_score_all_evaluated_kkt_residual),
        "best_score_all_evaluated_selection_eligible": bool(
            getattr(selection_result, "best_score_all_evaluated_selection_eligible", False)
        ),
        "best_score_certified_lambda": np.nan
        if getattr(selection_result, "best_score_certified_lambda", None) is None
        else float(selection_result.best_score_certified_lambda),
        "best_score_certified_kkt_residual": np.nan
        if getattr(selection_result, "best_score_certified_kkt_residual", None) is None
        else float(selection_result.best_score_certified_kkt_residual),
        "selection_metric_value": np.nan
        if selection_result.selection_metric_value is None
        else float(selection_result.selection_metric_value),
        "selection_lambda_min": np.nan
        if selection_result.selection_lambda_min is None
        else float(selection_result.selection_lambda_min),
        "selection_lambda_max": np.nan
        if selection_result.selection_lambda_max is None
        else float(selection_result.selection_lambda_max),
        "selection_lambda_count": int(selection_result.selection_lambda_count),
        "selection_hits_lower_boundary": bool(selection_result.selection_hits_lower_boundary),
        "selection_hits_upper_boundary": bool(selection_result.selection_hits_upper_boundary),
        "selection_boundary_unresolved": bool(selection_result.selection_boundary_unresolved),
        "selection_optimum_resolved": bool(selection_result.selection_optimum_resolved),
        "selected_ari": np.nan if reported_selected_ari is None else float(reported_selected_ari),
        "selected_ari_source": selected_ari_source,
        "selected_ari_lambda": np.nan if reported_selected_ari is None else selected_lambda_value,
        "selected_ari_matches_selected_lambda": bool(
            reported_selected_ari is not None and selected_lambda_applicable
        ),
        "best_ari": np.nan if simulation_diagnostics.best_ari is None else float(simulation_diagnostics.best_ari),
        "best_ari_all_evaluated": np.nan
        if getattr(simulation_diagnostics, "best_ari_all_evaluated", None) is None
        else float(simulation_diagnostics.best_ari_all_evaluated),
        "best_ari_certified": np.nan
        if getattr(simulation_diagnostics, "best_ari_certified", None) is None
        else float(simulation_diagnostics.best_ari_certified),
        "best_ari_bic_eligible": np.nan
        if getattr(simulation_diagnostics, "best_ari_bic_eligible", None) is None
        else float(simulation_diagnostics.best_ari_bic_eligible),
        "selection_optimizer_limited": bool(getattr(selection_result, "selection_optimizer_limited", False)),
        "selection_optimizer_limited_reason": str(
            getattr(selection_result, "selection_optimizer_limited_reason", "none")
        ),
        "best_converged_ari": np.nan
        if simulation_diagnostics.best_converged_ari is None
        else float(simulation_diagnostics.best_converged_ari),
        "best_converged_lambda_min": np.nan
        if simulation_diagnostics.best_converged_lambda_min is None
        else float(simulation_diagnostics.best_converged_lambda_min),
        "best_converged_lambda_max": np.nan
        if simulation_diagnostics.best_converged_lambda_max is None
        else float(simulation_diagnostics.best_converged_lambda_max),
        "best_converged_lambda_count": int(simulation_diagnostics.best_converged_lambda_count),
        "ari_optimal_lambda_min": np.nan
        if simulation_diagnostics.ari_optimal_lambda_min is None
        else float(simulation_diagnostics.ari_optimal_lambda_min),
        "ari_optimal_lambda_max": np.nan
        if simulation_diagnostics.ari_optimal_lambda_max is None
        else float(simulation_diagnostics.ari_optimal_lambda_max),
        "ari_optimal_lambda_count": int(simulation_diagnostics.ari_optimal_lambda_count),
        "ari_hits_lower_boundary": bool(simulation_diagnostics.ari_hits_lower_boundary),
        "ari_hits_upper_boundary": bool(simulation_diagnostics.ari_hits_upper_boundary),
        "ari_boundary_unresolved": bool(simulation_diagnostics.ari_boundary_unresolved),
        "ari_optimum_resolved": bool(simulation_diagnostics.ari_optimum_resolved),
        "adaptive_search_rounds_completed": int(selection_result.adaptive_search_rounds_completed),
        "adaptive_search_stop_reason": str(selection_result.adaptive_search_stop_reason),
        "tested_lambda_min": tested_lambda_min,
        "tested_lambda_max": tested_lambda_max,
        "tested_lambda_count": tested_lambda_count,
        "num_regions": int(tumor_regime.num_regions),
        "num_mutations": int(tumor_regime.num_mutations),
        "depth_scale": float(tumor_regime.depth_scale),
        "mean_purity": float(tumor_regime.mean_purity),
        "non_diploid_rate": float(tumor_regime.non_diploid_rate),
        "lambda_grid_mode": str(lambda_grid_mode if lambda_grid is None else "explicit"),
        "input_data_hash": str(_selected_value("input_data_hash", "")),
        "edge_list_hash": str(_selected_value("edge_list_hash", "")),
        "pilot_matrix_hash": str(_selected_value("pilot_matrix_hash", "")),
        "eps": float(_selected_value("eps", fit_options.eps)),
        "major_prior": float(_selected_value("major_prior", fit_options.major_prior)),
        "adaptive_weight_gamma": float(
            _selected_value("adaptive_weight_gamma", fit_options.adaptive_weight_gamma)
        ),
        "adaptive_weight_floor": float(
            _selected_value("adaptive_weight_floor", fit_options.adaptive_weight_floor)
        ),
        "adaptive_weight_baseline": float(
            _selected_value("adaptive_weight_baseline", fit_options.adaptive_weight_baseline)
        ),
        "tol": float(_selected_value("tol", fit_options.tol)),
        "outer_max_iter": int(_selected_value("outer_max_iter", fit_options.outer_max_iter)),
        "inner_max_iter": int(_selected_value("inner_max_iter", fit_options.inner_max_iter)),
        "bic_df_scale": float(selection_result.bic_df_scale),
        "bic_cluster_penalty": float(selection_result.bic_cluster_penalty),
        "bic_loglik": _artifact_float("bic_loglik"),
        "bic_loglik_source": str(_artifact_value("bic_loglik_source", "") or ""),
        "bic_df": _artifact_float("bic_df"),
        "bic_active_df": _artifact_float("bic_active_df"),
        "bic_n_eff": _artifact_float("bic_n_eff"),
        "bic_depth_n_eff": _artifact_float("bic_depth_n_eff"),
        "classic_bic_depth_n": _artifact_float("classic_bic_depth_n"),
        "classic_bic_active_df": _artifact_float("classic_bic_active_df"),
        "classic_bic_active_df_depth_n": _artifact_float("classic_bic_active_df_depth_n"),
        "bic_partition_tol": _artifact_float("bic_partition_tol"),
        "summary_cluster_max_diameter": float(best_fit.max_cluster_diameter),
        "summary_cluster_diameter_exact": bool(best_fit.cluster_diameter_exact),
        "bic_partition_max_diameter": float(_selected_value("bic_partition_max_diameter", np.nan)),
        "bic_partition_diameter_exact": bool(_selected_value("bic_partition_diameter_exact", False)),
        "bic_refit_boundary_count": _artifact_int("bic_refit_boundary_count"),
        "bic_refit_finite_candidate_found": bool(best_bic_refit_finite),
        "bic_refit_global_optimum_certified": bool(best_bic_refit_global),
        "bic_refit_coordinate_count": _artifact_int("bic_refit_coordinate_count"),
        "bic_refit_finite_coordinate_count": _artifact_int("bic_refit_finite_coordinate_count"),
        "bic_refit_total_grid_points": _artifact_int("bic_refit_total_grid_points"),
        "bic_refit_max_grid_spacing": _artifact_float("bic_refit_max_grid_spacing"),
        "bic_refit_total_candidate_basins": _artifact_int("bic_refit_total_candidate_basins"),
        "bic_refit_total_refined_candidates": _artifact_int("bic_refit_total_refined_candidates"),
        "bic_refit_min_best_second_loss_gap": _artifact_float("bic_refit_min_best_second_loss_gap"),
        "bic_refit_converged": _artifact_bool("bic_refit_converged"),
        "primary_phi_source": _selected_str("primary_phi_source", "raw_penalized_fit"),
        "bic_refit_phi_source": _selected_str("bic_refit_phi_source", "secondary_partition_refit"),
        "converged": bool(best_fit.converged),
        "stationarity_certified": bool(best_fit.stationarity_certified),
        "global_optimality_certified": bool(best_fit.global_optimality_certified),
        "global_optimality_basis": str(best_fit.global_optimality_basis),
        "number_of_starts": int(best_fit.number_of_starts),
        "number_of_finite_starts": int(best_fit.number_of_finite_starts),
        "best_start_objective": float(best_fit.best_start_objective),
        "second_best_start_objective": float(best_fit.second_best_start_objective),
        "objective_spread_across_starts": float(best_fit.objective_spread_across_starts),
        "selected_start_objective_rank": int(best_fit.selected_start_objective_rank),
        "converged_inner": bool(best_fit.converged_inner),
        "converged_outer": bool(best_fit.converged_outer),
        "inner_kkt_residual": float(best_fit.inner_kkt_residual),
        "accepted_inner_kkt_residual": float(best_fit.accepted_inner_kkt_residual),
        "last_attempted_inner_kkt_residual": float(best_fit.last_attempted_inner_kkt_residual),
        "best_attempted_inner_kkt_residual": float(best_fit.best_attempted_inner_kkt_residual),
        "last_attempted_objective_gap": float(best_fit.last_attempted_objective_gap),
        "best_attempted_objective_gap": float(best_fit.best_attempted_objective_gap),
        "last_attempted_surrogate_gap": float(best_fit.last_attempted_surrogate_gap),
        "best_attempted_surrogate_gap": float(best_fit.best_attempted_surrogate_gap),
        "last_attempted_inner_model_gap": float(best_fit.last_attempted_inner_model_gap),
        "best_attempted_inner_model_gap": float(best_fit.best_attempted_inner_model_gap),
        "last_attempted_em_envelope_gap": float(best_fit.last_attempted_em_envelope_gap),
        "best_attempted_em_envelope_gap": float(best_fit.best_attempted_em_envelope_gap),
        "outer_stationarity_residual": float(best_fit.outer_stationarity_residual),
        "outer_projected_stationarity_residual": float(best_fit.outer_projected_stationarity_residual),
        "outer_projected_stationarity_norm": float(best_fit.outer_projected_stationarity_norm),
        "outer_stationarity_normalizer": float(best_fit.outer_stationarity_normalizer),
        "outer_smooth_gradient_norm": float(best_fit.outer_smooth_gradient_norm),
        "outer_fusion_adjustment_norm": float(best_fit.outer_fusion_adjustment_norm),
        "outer_edge_subgradient_residual": float(best_fit.outer_edge_subgradient_residual),
        "outer_dual_ball_residual": float(best_fit.outer_dual_ball_residual),
        "outer_box_primal_violation": float(best_fit.outer_box_primal_violation),
        "outer_num_interior_coordinates": int(best_fit.outer_num_interior_coordinates),
        "outer_num_lower_active_coordinates": int(best_fit.outer_num_lower_active_coordinates),
        "outer_num_upper_active_coordinates": int(best_fit.outer_num_upper_active_coordinates),
        "outer_num_frozen_coordinates": int(best_fit.outer_num_frozen_coordinates),
        "outer_box_residual": float(best_fit.outer_box_residual),
        "fixed_objective_kkt_residual": float(best_fit.fixed_objective_kkt_residual),
        "outer_kkt_certificate_status": str(best_fit.outer_kkt_certificate_status),
        "outer_kkt_dual_refined": bool(best_fit.outer_kkt_dual_refined),
        "outer_kkt_fused_edges": int(best_fit.outer_kkt_fused_edges),
        "outer_kkt_nonzero_edges": int(best_fit.outer_kkt_nonzero_edges),
        "outer_stationarity_residual_before_dual_refine": float(
            best_fit.outer_stationarity_residual_before_dual_refine
        ),
        "outer_stationarity_residual_after_dual_refine": float(
            best_fit.outer_stationarity_residual_after_dual_refine
        ),
        "final_relative_objective_change": float(best_fit.final_relative_objective_change),
        "final_step_residual": float(best_fit.final_step_residual),
        "accepted_outer_steps": int(best_fit.accepted_outer_steps),
        "accepted_full_steps": int(best_fit.accepted_full_steps),
        "accepted_damped_steps": int(best_fit.accepted_damped_steps),
        "attempted_outer_steps": int(best_fit.attempted_outer_steps),
        "failed_majorization_checks": int(best_fit.failed_majorization_checks),
        "failed_inner_model_checks": int(best_fit.failed_inner_model_checks),
        "failed_em_envelope_checks": int(best_fit.failed_em_envelope_checks),
        "failed_descent_checks": int(best_fit.failed_descent_checks),
        "failed_nonfinite_checks": int(best_fit.failed_nonfinite_checks),
        "mm_consistency_violations": int(best_fit.mm_consistency_violations),
        "accepted_step_type": str(best_fit.accepted_step_type),
        "last_reject_reason": str(best_fit.last_reject_reason),
        "failure_reason": str(best_fit.failure_reason),
        "raw_kkt_eligible": bool(_selected_value("raw_kkt_eligible", best_fit.selection_eligible)),
        "bic_selection_eligible": bool(
            _selected_value(
                "bic_selection_eligible",
                bool(
                    best_fit.selection_eligible
                    and bool(best_bic_refit_finite)
                    and best_classic_bic is not None
                    and np.isfinite(float(best_classic_bic))
                ),
            )
        ),
        "selection_eligible": bool(_selected_value("selection_eligible", best_fit.selection_eligible)),
        "device": best_fit.device,
        "dtype": str(best_fit.dtype),
        "graph_name": str(best_fit.graph_name),
        "summary_tol": float(best_fit.summary_tol),
        "ARI": np.nan if reported_selected_ari is None else float(reported_selected_ari),
        "cp_rmse": np.nan if best_evaluation is None else float(best_evaluation.cp_rmse),
        "raw_cp_rmse": np.nan
        if best_evaluation is None
        else float(getattr(best_evaluation, "raw_cp_rmse", np.nan)),
        "summary_cp_rmse": np.nan
        if best_evaluation is None
        else float(getattr(best_evaluation, "summary_cp_rmse", best_evaluation.cp_rmse)),
        "multiplicity_f1": np.nan if best_evaluation is None else float(best_evaluation.multiplicity_f1),
        "estimated_clonal_fraction": np.nan
        if best_evaluation is None
        else float(best_evaluation.estimated_clonal_fraction),
        "true_clonal_fraction": np.nan
        if best_evaluation is None
        else float(best_evaluation.true_clonal_fraction),
        "clonal_fraction_error": np.nan
        if best_evaluation is None
        else float(best_evaluation.clonal_fraction_error),
        "n_eval_mutations": np.nan if best_evaluation is None else int(best_evaluation.n_eval_mutations),
        "n_filtered_mutations": np.nan if best_evaluation is None else int(best_evaluation.n_filtered_mutations),
        "bic_refit_ari": np.nan
        if (best_evaluation is None or getattr(best_evaluation, "bic_refit_ari", None) is None)
        else float(best_evaluation.bic_refit_ari),
        "bic_refit_cp_rmse": np.nan
        if (best_evaluation is None or getattr(best_evaluation, "bic_refit_cp_rmse", None) is None)
        else float(best_evaluation.bic_refit_cp_rmse),
        "elapsed_seconds": elapsed_seconds,
        "software_version": _SOFTWARE_VERSION,
    }

    if write_outputs:
        write_fit_outputs(
            outdir=outdir,
            data=data,
            fit=best_fit,
            search_df=search_df,
            evaluation=best_evaluation,
            run_summary=summary,
            bic_refit_phi=_artifact_value("bic_refit_phi"),
            bic_refit_cluster_centers=_artifact_value("bic_refit_cluster_centers"),
        )
    return summary, search_df


def process_one_file(
    file_path: str | Path,
    outdir: str | Path,
    simulation_root: str | Path | None = None,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "adaptive_bic",
    fit_options: FitOptions | None = None,
    bic_df_scale: float = 1.0,
    bic_cluster_penalty: float = 0.0,
    selection_score: str = "bic",
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    graph_file: str | Path | None = None,
    finalize_selected_fit: bool | None = None,
    missing_cna_policy: str = "error",
    evaluate_all_candidates: bool | None = None,
) -> dict[str, float | int | str | bool]:
    summary, _ = process_one_file_bundle(
        file_path=file_path,
        outdir=outdir,
        simulation_root=simulation_root,
        lambda_grid=lambda_grid,
        lambda_grid_mode=lambda_grid_mode,
        fit_options=fit_options,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
        selection_score=selection_score,
        use_warm_starts=use_warm_starts,
        write_outputs=write_outputs,
        graph_file=graph_file,
        finalize_selected_fit=finalize_selected_fit,
        evaluate_all_candidates=evaluate_all_candidates,
        missing_cna_policy=missing_cna_policy,
    )
    return summary


def run_directory(
    input_dir: str | Path,
    outdir: str | Path,
    simulation_root: str | Path | None = None,
    lambda_grid: list[float] | None = None,
    lambda_grid_mode: str = "adaptive_bic",
    fit_options: FitOptions | None = None,
    max_files: int | None = None,
    bic_df_scale: float = 1.0,
    bic_cluster_penalty: float = 0.0,
    selection_score: str = "bic",
    use_warm_starts: bool = True,
    write_outputs: bool = True,
    graph_file: str | Path | None = None,
    finalize_selected_fit: bool | None = None,
    missing_cna_policy: str = "error",
    workers: int = 1,
    evaluate_all_candidates: bool | None = None,
) -> pd.DataFrame:
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.tsv"))
    if max_files is not None:
        files = files[: max(0, int(max_files))]

    if not files:
        raise RuntimeError(f"No TSV files found in {input_dir}")

    summaries = []
    worker_count = max(int(workers), 1)
    if worker_count <= 1:
        for file_path in files:
            summaries.append(
                process_one_file(
                    file_path=file_path,
                    outdir=outdir,
                    simulation_root=simulation_root,
                    lambda_grid=lambda_grid,
                    lambda_grid_mode=lambda_grid_mode,
                    fit_options=fit_options,
                    bic_df_scale=bic_df_scale,
                    bic_cluster_penalty=bic_cluster_penalty,
                    selection_score=selection_score,
                    use_warm_starts=use_warm_starts,
                    write_outputs=write_outputs,
                    graph_file=graph_file,
                    finalize_selected_fit=finalize_selected_fit,
                    evaluate_all_candidates=evaluate_all_candidates,
                    missing_cna_policy=missing_cna_policy,
                )
            )
    else:
        # Guard against multi-process CUDA: each spawned process would independently
        # initialize the same device, risking OOM or allocator contention.
        effective_device = getattr(fit_options, "device", "auto") if fit_options is not None else "auto"
        if str(effective_device).strip().lower() not in {"cpu", "auto"} or (
            str(effective_device).strip().lower() == "auto"
            and _cuda_available()
        ):
            import warnings
            warnings.warn(
                f"workers={worker_count} with device={effective_device!r}: multiple processes may "
                "each initialize the same CUDA device, causing OOM or contention. "
                "Use workers=1 for CUDA mode, or set device='cpu'.",
                RuntimeWarning,
                stacklevel=2,
            )
        with cf.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp.get_context("spawn"),
        ) as executor:
            future_map = {
                executor.submit(
                    process_one_file,
                    file_path=file_path,
                    outdir=outdir,
                    simulation_root=simulation_root,
                    lambda_grid=lambda_grid,
                    lambda_grid_mode=lambda_grid_mode,
                    fit_options=fit_options,
                    bic_df_scale=bic_df_scale,
                    bic_cluster_penalty=bic_cluster_penalty,
                    selection_score=selection_score,
                    use_warm_starts=use_warm_starts,
                    write_outputs=write_outputs,
                    graph_file=graph_file,
                    finalize_selected_fit=finalize_selected_fit,
                    evaluate_all_candidates=evaluate_all_candidates,
                    missing_cna_policy=missing_cna_policy,
                ): file_path
                for file_path in files
            }
            ordered: dict[str, dict[str, float | int | str | bool]] = {}
            for future in cf.as_completed(future_map):
                file_path = future_map[future]
                try:
                    ordered[file_path.stem] = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Worker failed on {file_path}: {exc}"
                    ) from exc
            summaries = [ordered[file_path.stem] for file_path in files]

    summary_df = pd.DataFrame(summaries)
    sort_column = "tumor_id"
    summary_df = summary_df.sort_values(sort_column).reset_index(drop=True)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(Path(outdir) / "single_stage_summary.tsv", sep="\t", index=False)
    return summary_df
