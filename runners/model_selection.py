from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from ..core.model import FitOptions, FitResult
from ..core.fusion.partition_starts import (
    generate_likelihood_partition_starts,
    hessian_weighted_ward_label_sets_torch,
    observed_curvature_at_pilot_torch,
)
from ..core.fusion.refit import PartitionRefitResult
from ..core.fusion.solver import (
    prepare_torch_problem,
    torch_data_from_context,
)
from ..core.fusion.types import SolverState
from ..io.data import TumorData
from ..metrics.evaluation import (
    SimulationEvaluation,
    SimulationTruth,
    load_simulation_truth,
)
from ..core.bic import (
    LAMBDA_GRID_MODES,
    LambdaBracket,
    is_adaptive_lambda_grid_mode,
)

from ..model_selection.adaptive import (
    _adaptive_first_pass_options,
    _adaptive_interval_proposal_records,
    _adaptive_interval_proposals,
    _adaptive_score_column,
    _adaptive_transition_probe_records,
    _best_candidate_rows_by_lambda,
    _full_fusion_box_residual_with_dual_balls,
    _initial_adaptive_lambda_bracket,
    _lambda_boundary_flags,
    _lambda_boundary_unresolved,
    _score_maximized,
    _score_strictly_better,
    _selected_lambda_signature_interval,
)
from ..model_selection.candidates import (
    _evaluate_candidate,
    _evaluate_partition_candidate,
)
from ..model_selection.config import (
    ADAPTIVE_FIRST_PASS_INNER_MAX_ITER,
    ADAPTIVE_FIRST_PASS_OUTER_MAX_ITER,
    ADAPTIVE_PATH_MAX_CANDIDATES,
    ADAPTIVE_PATH_MAX_ROUNDS,
    ADAPTIVE_PATH_PARTITION_POOL_MAX_CANDIDATES,
    ADAPTIVE_PATH_PARTITION_POOL_MAX_ROUNDS,
    ADAPTIVE_PATH_PARTITION_POOL_REFINE_PER_ROUND,
    ADAPTIVE_PATH_PARTITION_POOL_TRANSITION_PROBE_MAX_CANDIDATES,
    ADAPTIVE_PATH_REFINE_PER_ROUND,
    ADAPTIVE_PATH_TRANSITION_PROBE_MAX_CANDIDATES,
    ENABLE_LIKELIHOOD_PARTITION_CANDIDATES,
    LIKELIHOOD_PARTITION_CEM_MAX_ITER,
    LIKELIHOOD_PARTITION_MAX_CANDIDATES_PER_K,
    LIKELIHOOD_PARTITION_REFIT_MAX_ITER,
    LIKELIHOOD_PARTITION_SENTINEL_LAMBDA,
)
from ..model_selection.partitions import (
    _deduplicate_partition_candidates,
    _likelihood_partition_k_grid,
    _likelihood_partition_refinement_k_grid,
    _partition_candidate_requested_k,
    _partition_is_coarsening,
    _partition_signature,
)
from ..model_selection.scoring import (
    _add_bic_selection_eligible,
    _annotate_bic_diagnostics,
    _ari_candidate_frame,
    _bic_selection_eligible_mask,
    _canonical_lambda,
    _effective_bic_partition_tol,
    _is_bic_selection_eligible,
    _lambda_applicable_mask,
    _lambda_range_for_optimal_rows,
    _lambda_warm_start_distance,
    _normalize_selection_score_name,
    _optimal_lambda_range,
    _prefer_fit_candidate,
    _representative_optimal_row,
    _row_bic_selection_eligible,
    _row_lambda_applicable,
    _row_lambda_if_applicable,
    _sorted_unique_lambdas,
)
from ..model_selection.types import (
    BICSelectionResult,
    CandidateStaticMetadata,
    ModelSelectionResult,
    SelectionArtifact,
    SimulationDiagnostics,
    StartArray,
)


def _hash_array(hasher: "hashlib._Hash", array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous.tobytes())


def _input_data_hash(data: TumorData) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    for value in data.mutation_ids:
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"\0")
    hasher.update(b"\1")
    for value in data.region_ids:
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"\0")
    for array in (
        data.alt_counts,
        data.total_counts,
        data.purity,
        data.major_cn,
        data.minor_cn,
        data.normal_cn,
        data.has_cna.astype(np.int8, copy=False),
        data.scaling,
        data.phi_upper,
    ):
        _hash_array(hasher, np.asarray(array))
    return hasher.hexdigest()


def _edge_list_hash(edge_u: np.ndarray, edge_v: np.ndarray, edge_w: np.ndarray) -> str:
    hasher = hashlib.blake2b(digest_size=16)
    _hash_array(hasher, np.asarray(edge_u, dtype=np.int64))
    _hash_array(hasher, np.asarray(edge_v, dtype=np.int64))
    _hash_array(hasher, np.asarray(edge_w, dtype=np.float64))
    return hasher.hexdigest()


def _pilot_matrix_hash(pilot_phi: StartArray | None) -> str:
    if pilot_phi is None:
        return ""
    if torch.is_tensor(pilot_phi):
        array = pilot_phi.detach().cpu().numpy()
    else:
        array = np.asarray(pilot_phi)
    hasher = hashlib.blake2b(digest_size=16)
    _hash_array(hasher, np.asarray(array, dtype=np.float64))
    return hasher.hexdigest()


def _candidate_static_metadata(data: TumorData, graph, pilot_phi: StartArray | None = None) -> CandidateStaticMetadata:
    edge_count = int(graph.edge_u.size)
    if edge_count:
        edge_weight_min = float(np.min(graph.edge_w))
        edge_weight_max = float(np.max(graph.edge_w))
        edge_weight_mean = float(np.mean(graph.edge_w))
    else:
        edge_weight_min = float("nan")
        edge_weight_max = float("nan")
        edge_weight_mean = float("nan")
    return CandidateStaticMetadata(
        edge_count=edge_count,
        edge_weight_min=edge_weight_min,
        edge_weight_max=edge_weight_max,
        edge_weight_mean=edge_weight_mean,
        edge_list_hash=_edge_list_hash(graph.edge_u, graph.edge_v, graph.edge_w),
        pilot_matrix_hash=_pilot_matrix_hash(pilot_phi),
        input_data_hash=_input_data_hash(data),
    )


def _clone_start(start: StartArray) -> StartArray:
    if torch.is_tensor(start):
        return start.detach().clone()
    return np.asarray(start).copy()


def _fit_phi_start(fit: FitResult) -> StartArray:
    if fit.solver_state is not None and fit.solver_state.phi is not None:
        return fit.solver_state.phi
    return fit.phi


def _resolve_adaptive_path_config(use_partition_pool: bool) -> tuple[int, int, int, int]:
    """Adaptive lambda-path budgets as
    (max_candidates, max_rounds, refine_per_round, transition_probe_max_candidates),
    switching between the partition-pool and default budgets on one condition."""
    if use_partition_pool:
        return (
            ADAPTIVE_PATH_PARTITION_POOL_MAX_CANDIDATES,
            ADAPTIVE_PATH_PARTITION_POOL_MAX_ROUNDS,
            ADAPTIVE_PATH_PARTITION_POOL_REFINE_PER_ROUND,
            ADAPTIVE_PATH_PARTITION_POOL_TRANSITION_PROBE_MAX_CANDIDATES,
        )
    return (
        ADAPTIVE_PATH_MAX_CANDIDATES,
        ADAPTIVE_PATH_MAX_ROUNDS,
        ADAPTIVE_PATH_REFINE_PER_ROUND,
        ADAPTIVE_PATH_TRANSITION_PROBE_MAX_CANDIDATES,
    )


def _assemble_selection_result(
    *,
    search_df,
    data,
    normalized_score,
    result_entries,
    bic_df_scale,
    bic_cluster_penalty,
    selection_method,
    profile_name,
    lambda_search_mode,
    lambda_bracket,
    adaptive_search_stop_reason,
    adaptive_search_rounds_completed,
    adaptive_refinement_rounds_completed,
    selection_start_time,
) -> BICSelectionResult:
    search_df = _annotate_bic_diagnostics(search_df)
    num_candidates = int(search_df.shape[0])
    converged_mask = search_df["converged"].astype(bool).to_numpy(dtype=bool)
    candidate_selection_eligible_mask = _bic_selection_eligible_mask(search_df)
    num_converged_candidates = int(np.sum(converged_mask))
    num_selection_eligible_candidates = int(np.sum(candidate_selection_eligible_mask))
    if num_selection_eligible_candidates == 0:
        raise RuntimeError(f"No candidates were eligible for BIC model selection for tumor {data.tumor_id}.")
    selection_df = search_df.loc[candidate_selection_eligible_mask].copy()
    converged_ari_df = _ari_candidate_frame(selection_df.copy())
    selection_used_convergence_fallback = False

    if selection_df.empty:
        raise RuntimeError(f"No candidate fits were evaluated for tumor {data.tumor_id}.")

    selection_lambda_values = selection_df["lambda"].to_numpy(dtype=float)

    if normalized_score == "bic":
        _, _, _, selection_metric_value, selection_mask = _optimal_lambda_range(
            selection_df["classic_bic"].to_numpy(dtype=float),
            selection_lambda_values,
            maximize=False,
        )
        tied_df = selection_df.loc[selection_mask].copy()
        tied_df["_lambda_applicable_sort"] = _lambda_applicable_mask(tied_df)
        tied_df = tied_df.sort_values(
            ["classic_bic", "_lambda_applicable_sort", "lambda", "selection_step"],
            ascending=[True, False, True, True],
        )
        best_row = tied_df.iloc[0]
    else:
        _, _, _, selection_metric_value, selection_mask = _optimal_lambda_range(
            selection_df["bic"].to_numpy(dtype=float),
            selection_lambda_values,
            maximize=False,
        )
        tied_df = selection_df.loc[selection_mask].copy()
        tied_df["_lambda_applicable_sort"] = _lambda_applicable_mask(tied_df)
        tied_df = tied_df.sort_values(
            ["bic", "_lambda_applicable_sort", "lambda", "selection_step"],
            ascending=[True, False, True, True],
        )
        best_row = tied_df.iloc[0]
    selected_lambda_applicable = _row_lambda_applicable(best_row)
    selection_lambda_min, selection_lambda_max, selection_lambda_count = _lambda_range_for_optimal_rows(
        selection_df,
        selection_mask,
    )

    score_column = _adaptive_score_column(normalized_score)
    all_scores = (
        search_df[score_column].to_numpy(dtype=float)
        if score_column in search_df.columns
        else np.full(search_df.shape[0], np.nan, dtype=float)
    )
    all_objectives = (
        search_df["penalized_objective"].to_numpy(dtype=float)
        if "penalized_objective" in search_df.columns
        else np.full(search_df.shape[0], np.nan, dtype=float)
    )
    all_mm_violations = (
        search_df["mm_consistency_violations"].to_numpy(dtype=float)
        if "mm_consistency_violations" in search_df.columns
        else np.zeros(search_df.shape[0], dtype=float)
    )
    partition_candidate_mask = (
        search_df["candidate_pool_source"].astype(str).eq("likelihood_partition").to_numpy(dtype=bool)
        if "candidate_pool_source" in search_df.columns
        else np.zeros(search_df.shape[0], dtype=bool)
    )
    provisional_mask = np.isfinite(all_scores) & (
        (np.isfinite(all_objectives) & (all_mm_violations <= 0.0))
        | partition_candidate_mask
    )
    provisional_df = search_df.loc[provisional_mask].copy()
    if provisional_df.empty:
        best_score_all_row = None
    else:
        best_score_all_row = provisional_df.sort_values(
            [score_column, "lambda", "selection_step"],
            ascending=[not _score_maximized(normalized_score), True, True],
            na_position="last",
        ).iloc[0]
    certified_score_df = search_df.loc[candidate_selection_eligible_mask & provisional_mask].copy()
    if certified_score_df.empty:
        best_score_certified_row = None
    else:
        best_score_certified_row = certified_score_df.sort_values(
            [score_column, "lambda", "selection_step"],
            ascending=[not _score_maximized(normalized_score), True, True],
            na_position="last",
        ).iloc[0]

    num_candidates_all = int(search_df.shape[0])
    num_candidates_certified = int(np.sum(candidate_selection_eligible_mask))

    selected_kkt_residual = (
        float(best_row["fixed_objective_kkt_residual"])
        if "fixed_objective_kkt_residual" in best_row and np.isfinite(float(best_row["fixed_objective_kkt_residual"]))
        else None
    )
    selected_provisional_score = float(best_row.get(score_column, np.nan))
    best_score_all_evaluated_lambda = None
    best_score_all_evaluated_kkt_residual = None
    best_score_all_evaluated_selection_eligible = False
    if best_score_all_row is not None:
        best_score_all_evaluated_lambda = _row_lambda_if_applicable(best_score_all_row)
        best_score_all_evaluated_kkt_residual = (
            float(best_score_all_row["fixed_objective_kkt_residual"])
            if np.isfinite(float(best_score_all_row.get("fixed_objective_kkt_residual", np.nan)))
            else None
        )
        best_score_all_evaluated_selection_eligible = bool(
            _row_bic_selection_eligible(best_score_all_row)
        )
    best_score_certified_lambda = None
    best_score_certified_kkt_residual = None
    if best_score_certified_row is not None:
        best_score_certified_lambda = _row_lambda_if_applicable(best_score_certified_row)
        best_score_certified_kkt_residual = (
            float(best_score_certified_row["fixed_objective_kkt_residual"])
            if np.isfinite(float(best_score_certified_row.get("fixed_objective_kkt_residual", np.nan)))
            else None
        )

    selection_optimizer_limited = False
    selection_optimizer_limited_reason = "none"
    optimizer_limited_ids: set[int] = set()
    if best_score_all_row is not None:
        best_score_all_score = float(best_score_all_row.get(score_column, np.nan))
        best_score_all_eligible = bool(_row_bic_selection_eligible(best_score_all_row))
        if (
            _score_strictly_better(best_score_all_score, selected_provisional_score, normalized_score=normalized_score)
            and not best_score_all_eligible
        ):
            selection_optimizer_limited = True
            selection_optimizer_limited_reason = "best_provisional_score_failed_kkt"

    if "_candidate_id" in search_df.columns and np.isfinite(selected_provisional_score):
        for _, candidate_row in search_df.loc[provisional_mask & ~candidate_selection_eligible_mask].iterrows():
            candidate_score = float(candidate_row.get(score_column, np.nan))
            if _score_strictly_better(candidate_score, selected_provisional_score, normalized_score=normalized_score):
                optimizer_limited_ids.add(int(candidate_row["_candidate_id"]))

    best_ari_min, best_ari_max, best_ari_count, best_ari_value, ari_mask = _optimal_lambda_range(
        selection_df["ARI"].to_numpy(dtype=float),
        selection_lambda_values,
        maximize=True,
    )
    best_ari_min, best_ari_max, best_ari_count = _lambda_range_for_optimal_rows(selection_df, ari_mask)
    (
        best_converged_ari_min,
        best_converged_ari_max,
        best_converged_ari_count,
        best_converged_ari_value,
        best_converged_ari_mask,
    ) = _optimal_lambda_range(
        converged_ari_df["ARI"].to_numpy(dtype=float) if not converged_ari_df.empty else np.asarray([], dtype=float),
        converged_ari_df["lambda"].to_numpy(dtype=float) if not converged_ari_df.empty else np.asarray([], dtype=float),
        maximize=True,
    )
    best_converged_ari_min, best_converged_ari_max, best_converged_ari_count = _lambda_range_for_optimal_rows(
        converged_ari_df,
        best_converged_ari_mask,
    )
    _, _, _, best_ari_all_evaluated, _ = _optimal_lambda_range(
        search_df["ARI"].to_numpy(dtype=float) if "ARI" in search_df.columns else np.asarray([], dtype=float),
        search_df["lambda"].to_numpy(dtype=float) if "lambda" in search_df.columns else np.asarray([], dtype=float),
        maximize=True,
    )
    _, _, _, best_ari_certified, _ = _optimal_lambda_range(
        search_df.loc[candidate_selection_eligible_mask, "ARI"].to_numpy(dtype=float)
        if "ARI" in search_df.columns and np.any(candidate_selection_eligible_mask)
        else np.asarray([], dtype=float),
        search_df.loc[candidate_selection_eligible_mask, "lambda"].to_numpy(dtype=float)
        if "lambda" in search_df.columns and np.any(candidate_selection_eligible_mask)
        else np.asarray([], dtype=float),
        maximize=True,
    )
    lambda_applicable_mask = _lambda_applicable_mask(selection_df)
    selection_boundary_lambda_values = selection_lambda_values[lambda_applicable_mask]
    selection_lower_hit, selection_upper_hit = _lambda_boundary_flags(
        selection_boundary_lambda_values,
        best_lambda_min=selection_lambda_min,
        best_lambda_max=selection_lambda_max,
    )
    selection_boundary_unresolved = _lambda_boundary_unresolved(
        evaluated_lambdas=selection_boundary_lambda_values,
        lower_hit=selection_lower_hit,
        upper_hit=selection_upper_hit,
    )
    ari_lower_hit, ari_upper_hit = _lambda_boundary_flags(
        selection_boundary_lambda_values,
        best_lambda_min=best_ari_min,
        best_lambda_max=best_ari_max,
    )
    ari_boundary_unresolved = _lambda_boundary_unresolved(
        evaluated_lambdas=selection_boundary_lambda_values,
        lower_hit=ari_lower_hit,
        upper_hit=ari_upper_hit,
    )
    selection_optimal_ids = set(selection_df.loc[selection_mask, "_candidate_id"].astype(int).tolist())
    ari_optimal_ids = set(selection_df.loc[ari_mask, "_candidate_id"].astype(int).tolist())
    final_adaptive_search_stop_reason = adaptive_search_stop_reason
    eligible_mask = candidate_selection_eligible_mask
    search_df["eligible_for_selection"] = eligible_mask
    lambda_values_evaluated = ",".join(
        f"{float(value):.12g}" for value in _sorted_unique_lambdas(search_df["lambda"].to_numpy(dtype=float))
    )
    search_df["lambda_values_evaluated"] = lambda_values_evaluated
    if "optimizer_limited_candidate" not in search_df.columns:
        search_df["optimizer_limited_candidate"] = False
    if optimizer_limited_ids and "_candidate_id" in search_df.columns:
        search_df["optimizer_limited_candidate"] = search_df["_candidate_id"].astype(int).isin(optimizer_limited_ids)
    search_df["is_selection_optimal"] = search_df["_candidate_id"].astype(int).isin(selection_optimal_ids)
    search_df["is_ari_optimal"] = search_df["_candidate_id"].astype(int).isin(ari_optimal_ids)
    selected_candidate_id = int(best_row["_candidate_id"])
    search_df["is_selected_best_row"] = search_df["_candidate_id"].astype(int) == selected_candidate_id
    search_df["adaptive_search_stop_reason"] = str(final_adaptive_search_stop_reason)
    selected_lambda_left, selected_lambda_right, selected_lambda_log10_width = _selected_lambda_signature_interval(
        search_df,
        selected_candidate_id=selected_candidate_id,
    )
    selected_lambda_representative_value = (
        float(best_row["lambda"]) if selected_lambda_applicable else np.nan
    )
    search_df["selected_lambda_representative"] = selected_lambda_representative_value
    search_df["selected_lambda_left"] = np.nan if selected_lambda_left is None else float(selected_lambda_left)
    search_df["selected_lambda_right"] = np.nan if selected_lambda_right is None else float(selected_lambda_right)
    search_df["selected_lambda_interval_log10_width"] = (
        np.nan if selected_lambda_log10_width is None else float(selected_lambda_log10_width)
    )
    selection_elapsed_seconds = float(perf_counter() - selection_start_time)
    search_df["selection_elapsed_seconds"] = float(selection_elapsed_seconds)

    best_fit, best_evaluation, _, selected_artifact = result_entries[int(best_row["_candidate_id"])]
    selected_candidate_ari = float(best_row["ARI"]) if np.isfinite(float(best_row["ARI"])) else None
    selected_ari = float(best_evaluation.ari) if best_evaluation is not None else selected_candidate_ari
    search_df = search_df.drop(columns=["_candidate_id"])
    simulation_diagnostics = SimulationDiagnostics(
        selected_evaluation=best_evaluation,
        selected_ari=selected_ari,
        best_ari=best_ari_value,
        ari_optimal_lambda_min=best_ari_min,
        ari_optimal_lambda_max=best_ari_max,
        ari_optimal_lambda_count=best_ari_count,
        best_converged_ari=best_converged_ari_value,
        best_converged_lambda_min=best_converged_ari_min,
        best_converged_lambda_max=best_converged_ari_max,
        best_converged_lambda_count=best_converged_ari_count,
        ari_hits_lower_boundary=ari_lower_hit,
        ari_hits_upper_boundary=ari_upper_hit,
        ari_boundary_unresolved=ari_boundary_unresolved,
        ari_optimum_resolved=not ari_boundary_unresolved,
        best_ari_all_evaluated=best_ari_all_evaluated,
        best_ari_certified=best_ari_certified,
    )
    return BICSelectionResult(
        best_fit=best_fit,
        selected_artifact=selected_artifact,
        search_df=search_df,
        bic_df_scale=float(bic_df_scale),
        bic_cluster_penalty=float(bic_cluster_penalty),
        selection_method=selection_method,
        profile_name=profile_name,
        selection_metric_value=selection_metric_value,
        selection_lambda_min=selection_lambda_min,
        selection_lambda_max=selection_lambda_max,
        selection_lambda_count=selection_lambda_count,
        selection_hits_lower_boundary=selection_lower_hit,
        selection_hits_upper_boundary=selection_upper_hit,
        selection_boundary_unresolved=selection_boundary_unresolved,
        selection_optimum_resolved=not selection_boundary_unresolved,
        adaptive_search_rounds_completed=adaptive_search_rounds_completed,
        adaptive_search_stop_reason=str(final_adaptive_search_stop_reason),
        num_candidates=num_candidates,
        num_converged_candidates=num_converged_candidates,
        selection_used_convergence_fallback=selection_used_convergence_fallback,
        lambda_search_mode=str(lambda_search_mode),
        selected_lambda_representative=None
        if not selected_lambda_applicable
        else float(best_row["lambda"]),
        selected_lambda_left=selected_lambda_left,
        selected_lambda_right=selected_lambda_right,
        selected_lambda_interval_log10_width=selected_lambda_log10_width,
        lambda_bracket_min=None if lambda_bracket is None else float(lambda_bracket.lambda_min),
        lambda_bracket_eq=None if lambda_bracket is None else float(lambda_bracket.lambda_eq),
        lambda_bracket_full=None if lambda_bracket is None else float(lambda_bracket.lambda_full),
        adaptive_refinement_rounds_completed=int(adaptive_refinement_rounds_completed),
        num_candidates_all=num_candidates_all,
        num_candidates_certified=num_candidates_certified,
        selected_kkt_residual=selected_kkt_residual,
        best_score_all_evaluated_lambda=best_score_all_evaluated_lambda,
        best_score_all_evaluated_kkt_residual=best_score_all_evaluated_kkt_residual,
        best_score_all_evaluated_selection_eligible=best_score_all_evaluated_selection_eligible,
        best_score_certified_lambda=best_score_certified_lambda,
        best_score_certified_kkt_residual=best_score_certified_kkt_residual,
        selection_optimizer_limited=selection_optimizer_limited,
        selection_optimizer_limited_reason=selection_optimizer_limited_reason,
        simulation=simulation_diagnostics,
    )


def _grid_search_selection(
    *,
    data: TumorData,
    simulation_root: Path | None,
    lambda_grid: list[float] | None,
    lambda_grid_mode: str,
    fit_options: FitOptions,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    use_warm_starts: bool,
    evaluate_all_candidates: bool,
    profile_name: str,
    selection_method: str,
    selection_score: str,
    finalize_selected_fit: bool,
) -> BICSelectionResult:
    selection_start_time = perf_counter()
    explicit_lambda_grid = lambda_grid is not None
    normalized_lambda_grid_mode = str(lambda_grid_mode).strip().lower()
    if normalized_lambda_grid_mode not in LAMBDA_GRID_MODES:
        raise ValueError(f"Unknown lambda_grid_mode: {lambda_grid_mode}")
    adaptive_lambda_mode = bool(lambda_grid is None and is_adaptive_lambda_grid_mode(normalized_lambda_grid_mode))
    normalized_score = _normalize_selection_score_name(selection_score)
    lambda_search_mode = "explicit_grid" if explicit_lambda_grid else normalized_lambda_grid_mode
    lambda_bracket: LambdaBracket | None = None
    if lambda_grid is None and not adaptive_lambda_mode:
        raise ValueError("Default model selection uses lambda_grid_mode='adaptive_bic'.")
    lambda_grid = [] if lambda_grid is None else _sorted_unique_lambdas(lambda_grid)
    likelihood_partition_pool_enabled = bool(ENABLE_LIKELIHOOD_PARTITION_CANDIDATES)

    prepare_start_time = perf_counter()
    solver_context = prepare_torch_problem(
        data,
        major_prior=float(fit_options.major_prior),
        eps=float(fit_options.eps),
        tol=float(fit_options.tol),
        graph=fit_options.graph,
        inner_max_iter=max(int(fit_options.inner_max_iter), 16),
        adaptive_weight_gamma=float(fit_options.adaptive_weight_gamma),
        adaptive_weight_floor=float(fit_options.adaptive_weight_floor),
        adaptive_weight_baseline=float(fit_options.adaptive_weight_baseline),
        device=fit_options.device,
        dtype=fit_options.dtype,
    )
    prepare_elapsed_seconds = float(perf_counter() - prepare_start_time)
    runtime = solver_context.runtime
    torch_data = torch_data_from_context(solver_context)
    pilot_phi: StartArray = solver_context.exact_pilot
    pooled_start: StartArray = solver_context.pooled_start
    scalar_well_starts: list[StartArray] = list(solver_context.scalar_well_starts)
    effective_graph = solver_context.graph_spec
    effective_tensor_graph = solver_context.graph
    effective_fit_options = replace(fit_options, graph=effective_graph)
    static_metadata = _candidate_static_metadata(data, effective_graph, pilot_phi=pilot_phi)
    search_fit_options = (
        _adaptive_first_pass_options(effective_fit_options)
        if adaptive_lambda_mode
        else effective_fit_options
    )
    if adaptive_lambda_mode:
        lambda_bracket = _initial_adaptive_lambda_bracket(
            torch_data=torch_data,
            runtime=runtime,
            exact_pilot=pilot_phi,
            pooled_start=pooled_start,
            edge_u=effective_tensor_graph.edge_u,
            edge_v=effective_tensor_graph.edge_v,
            edge_w=effective_tensor_graph.weight,
            major_prior=float(fit_options.major_prior),
            eps=float(fit_options.eps),
            tol=float(fit_options.tol),
            degree_bound=int(effective_graph.degree_bound),
            sparse_anchors=bool(likelihood_partition_pool_enabled),
        )
        lambda_grid = list(lambda_bracket.anchors)
    simulation_truth: SimulationTruth | None = None
    if evaluate_all_candidates and simulation_root is not None and (simulation_root / data.tumor_id).exists():
        simulation_truth = load_simulation_truth(data, simulation_root)
    result_entries: list[
        tuple[FitResult, SimulationEvaluation | None, dict[str, float | int | str | bool], SelectionArtifact]
    ] = []
    fit_by_lambda: dict[float, FitResult] = {}
    solver_state_by_lambda: dict[float, SolverState] = {}
    partition_labels_by_candidate_id: dict[int, np.ndarray] = {}
    bic_refit_cache: dict[str, PartitionRefitResult] = {}
    next_step = 0
    (
        adaptive_max_candidates,
        adaptive_max_rounds,
        adaptive_refine_per_round,
        adaptive_transition_probe_max_candidates,
    ) = _resolve_adaptive_path_config(adaptive_lambda_mode and likelihood_partition_pool_enabled)
    adaptive_initial_anchor_count = int(len(lambda_grid))

    def _nearest_phi_start(target_lambda: float) -> StartArray:
        if not fit_by_lambda:
            return _clone_start(pilot_phi)
        nearest_lambda = min(
            fit_by_lambda,
            key=lambda value: _lambda_warm_start_distance(
                source_lambda=float(value),
                target_lambda=float(target_lambda),
            ),
        )
        return _clone_start(_fit_phi_start(fit_by_lambda[nearest_lambda]))

    def _nearest_solver_state(target_lambda: float) -> SolverState | None:
        if not solver_state_by_lambda:
            return None
        nearest_lambda = min(
            solver_state_by_lambda,
            key=lambda value: _lambda_warm_start_distance(
                source_lambda=float(value),
                target_lambda=float(target_lambda),
            ),
        )
        return solver_state_by_lambda.get(nearest_lambda)

    def _evaluate_lambda_sequence(
        lambda_values_to_run: list[float],
        *,
        search_round: int,
        search_phase: str,
        allow_revisit: bool = False,
        candidate_fit_options: FitOptions | None = None,
        start_mode: str = "full",
        compute_summary: bool = True,
        phi_start_by_lambda: dict[float, StartArray] | None = None,
        solver_state_start_by_lambda: dict[float, SolverState] | None = None,
        scalar_well_starts_by_lambda: dict[float, list[StartArray]] | None = None,
        lambda_metadata_by_lambda: dict[float, dict[str, float | int | str | bool]] | None = None,
    ) -> None:
        nonlocal next_step
        ordered_lambdas = [
            value
            for value in _sorted_unique_lambdas(lambda_values_to_run)
            if allow_revisit or _canonical_lambda(value) not in fit_by_lambda
        ]
        if not ordered_lambdas:
            return

        previous_phi: StartArray | None = None
        previous_solver_state: SolverState | None = None
        for lambda_value in ordered_lambdas:
            lambda_key = _canonical_lambda(lambda_value)
            solver_state_start: SolverState | None = None
            if phi_start_by_lambda is not None and lambda_key in phi_start_by_lambda:
                phi_start = _clone_start(phi_start_by_lambda[lambda_key])
                if solver_state_start_by_lambda is not None:
                    solver_state_start = solver_state_start_by_lambda.get(lambda_key)
                if solver_state_start is None:
                    solver_state_start = solver_state_by_lambda.get(lambda_key)
            elif use_warm_starts:
                solver_state_start = previous_solver_state if previous_solver_state is not None else _nearest_solver_state(lambda_value)
                if solver_state_start is not None:
                    phi_start = _clone_start(solver_state_start.phi)
                else:
                    phi_start = _clone_start(previous_phi) if previous_phi is not None else _nearest_phi_start(lambda_value)
            else:
                phi_start = _clone_start(pilot_phi)
            candidate_scalar_well_starts = scalar_well_starts
            if scalar_well_starts_by_lambda is not None and lambda_key in scalar_well_starts_by_lambda:
                candidate_scalar_well_starts = scalar_well_starts_by_lambda[lambda_key]
            fit, evaluation, row, artifact = _evaluate_candidate(
                data=data,
                fit_options=search_fit_options,
                candidate_fit_options=candidate_fit_options,
                bic_df_scale=bic_df_scale,
                bic_cluster_penalty=bic_cluster_penalty,
                simulation_root=simulation_root,
                simulation_truth=simulation_truth,
                evaluate_candidate=evaluate_all_candidates,
                phi_start=phi_start,
                exact_pilot=pilot_phi,
                pooled_start=pooled_start,
                scalar_well_starts=candidate_scalar_well_starts,
                start_mode=start_mode,
                runtime=runtime,
                torch_data=torch_data,
                solver_context=solver_context,
                solver_state=solver_state_start if use_warm_starts else None,
                compute_summary=compute_summary,
                selection_method=selection_method,
                profile_name=profile_name,
                selection_step=next_step,
                lambda_value=lambda_value,
                selection_score=selection_score,
                bic_refit_cache=bic_refit_cache,
                static_metadata=static_metadata,
            )
            row["search_round"] = int(search_round)
            row["search_phase"] = str(search_phase)
            row["lambda_source"] = str(search_phase)
            row["lambda_search_mode"] = str(lambda_search_mode)
            row["selection_prepare_elapsed_seconds"] = float(prepare_elapsed_seconds)
            row["adaptive_candidate_budget"] = int(adaptive_max_candidates)
            row["adaptive_max_rounds"] = int(adaptive_max_rounds)
            row["adaptive_refine_per_round"] = int(adaptive_refine_per_round)
            row["adaptive_transition_probe_max_candidates"] = int(adaptive_transition_probe_max_candidates)
            row["adaptive_initial_anchor_count"] = int(adaptive_initial_anchor_count)
            row["likelihood_partition_pool_enabled"] = bool(likelihood_partition_pool_enabled)
            row["lambda_bracket_min"] = np.nan if lambda_bracket is None else float(lambda_bracket.lambda_min)
            row["lambda_bracket_eq"] = np.nan if lambda_bracket is None else float(lambda_bracket.lambda_eq)
            row["lambda_bracket_full"] = np.nan if lambda_bracket is None else float(lambda_bracket.lambda_full)
            if lambda_bracket is None:
                row["lambda_full_residual"] = np.nan
            else:
                for diagnostic_name, diagnostic_value in lambda_bracket.diagnostics.items():
                    if np.isscalar(diagnostic_value):
                        row[str(diagnostic_name)] = float(diagnostic_value)
            if lambda_metadata_by_lambda is not None and lambda_key in lambda_metadata_by_lambda:
                row.update(lambda_metadata_by_lambda[lambda_key])
            candidate_id = int(len(result_entries))
            row["_candidate_id"] = candidate_id
            result_entries.append((fit, evaluation, row, artifact))
            if artifact.bic_partition_labels is not None:
                partition_labels_by_candidate_id[candidate_id] = np.asarray(
                    artifact.bic_partition_labels,
                    dtype=np.int64,
                ).copy()
            incumbent = fit_by_lambda.get(lambda_key)
            if _prefer_fit_candidate(fit, incumbent):
                fit_by_lambda[lambda_key] = fit
                if fit.solver_state is not None:
                    solver_state_by_lambda[lambda_key] = fit.solver_state
            next_step += 1
            if use_warm_starts:
                previous_phi = _clone_start(_fit_phi_start(fit))
                previous_solver_state = fit.solver_state

    _evaluate_lambda_sequence(
        lambda_grid,
        search_round=0,
        search_phase="base",
        start_mode="warm_only" if adaptive_lambda_mode and use_warm_starts else "full",
        compute_summary=True,
    )

    if adaptive_lambda_mode and lambda_bracket is not None:
        remaining_transition_budget = max(
            int(adaptive_max_candidates) - len(fit_by_lambda),
            0,
        )
        transition_probe_records = _adaptive_transition_probe_records(
            lambda_bracket,
            list(fit_by_lambda.keys()),
            max_new=min(
                int(adaptive_transition_probe_max_candidates),
                remaining_transition_budget,
            ),
        )
        if transition_probe_records:
            transition_phi_by_lambda: dict[float, StartArray] = {}
            transition_state_by_lambda: dict[float, SolverState] = {}
            transition_starts_by_lambda: dict[float, list[StartArray]] = {}
            transition_metadata_by_lambda: dict[float, dict[str, float | int | str | bool]] = {}
            for proposal in transition_probe_records:
                lambda_key = _canonical_lambda(proposal.lambda_value)
                left_fit = fit_by_lambda.get(_canonical_lambda(proposal.left_lambda))
                right_fit = fit_by_lambda.get(_canonical_lambda(proposal.right_lambda))
                transition_starts = [_clone_start(pilot_phi)]
                if right_fit is not None:
                    transition_starts.insert(0, _clone_start(_fit_phi_start(right_fit)))
                if left_fit is not None:
                    transition_phi_by_lambda[lambda_key] = _clone_start(_fit_phi_start(left_fit))
                    if left_fit.solver_state is not None:
                        transition_state_by_lambda[lambda_key] = left_fit.solver_state
                    transition_starts.insert(0, _clone_start(_fit_phi_start(left_fit)))
                elif right_fit is not None:
                    transition_phi_by_lambda[lambda_key] = _clone_start(_fit_phi_start(right_fit))
                    if right_fit.solver_state is not None:
                        transition_state_by_lambda[lambda_key] = right_fit.solver_state
                transition_starts_by_lambda[lambda_key] = transition_starts
                transition_metadata_by_lambda[lambda_key] = {
                    "adaptive_interval_left_lambda": float(proposal.left_lambda),
                    "adaptive_interval_right_lambda": float(proposal.right_lambda),
                    "adaptive_interval_log10_width": float(proposal.log_width),
                    "adaptive_interval_priority_class": int(proposal.priority_key[0]),
                    "adaptive_interval_reason": str(proposal.reason),
                    "adaptive_interval_partition_changed": bool(proposal.partition_changed),
                    "adaptive_interval_nonagglomerative_or_numerically_inconsistent": bool(
                        proposal.nonagglomerative_or_numerically_inconsistent
                    ),
                    "adaptive_transition_probe": True,
                }
            _evaluate_lambda_sequence(
                [float(proposal.lambda_value) for proposal in transition_probe_records],
                search_round=0,
                search_phase="adaptive_transition_probe",
                candidate_fit_options=_adaptive_first_pass_options(effective_fit_options),
                start_mode="full",
                compute_summary=True,
                phi_start_by_lambda=transition_phi_by_lambda,
                solver_state_start_by_lambda=transition_state_by_lambda,
                scalar_well_starts_by_lambda=transition_starts_by_lambda,
                lambda_metadata_by_lambda=transition_metadata_by_lambda,
            )

    adaptive_search_rounds_completed = 0
    adaptive_search_stop_reason = "not_applicable"
    adaptive_refinement_rounds_completed = 0
    if adaptive_lambda_mode:
        for adaptive_round in range(1, adaptive_max_rounds + 1):
            if len(fit_by_lambda) >= adaptive_max_candidates:
                adaptive_search_stop_reason = "adaptive_candidate_budget_reached"
                break
            interim_df = pd.DataFrame([row for _, _, row, _ in result_entries])
            proposal_records = _adaptive_interval_proposal_records(
                interim_df,
                normalized_score=normalized_score,
                tol=float(effective_fit_options.tol),
                max_new=min(
                    adaptive_refine_per_round,
                    max(adaptive_max_candidates - len(fit_by_lambda), 0),
                ),
                partition_labels_by_candidate_id=partition_labels_by_candidate_id,
            )
            proposal_records = [
                proposal
                for proposal in proposal_records
                if _canonical_lambda(proposal.lambda_value) not in fit_by_lambda
            ]
            proposals = [float(proposal.lambda_value) for proposal in proposal_records]
            if not proposals:
                adaptive_search_stop_reason = "adaptive_path_resolved"
                break
            fit_by_candidate_id = {
                int(row["_candidate_id"]): fit
                for fit, _, row, _ in result_entries
                if "_candidate_id" in row
            }
            proposal_phi_by_lambda: dict[float, StartArray] = {}
            proposal_state_by_lambda: dict[float, SolverState] = {}
            proposal_scalar_starts_by_lambda: dict[float, list[StartArray]] = {}
            proposal_metadata_by_lambda: dict[float, dict[str, float | int | str | bool]] = {}
            for proposal in proposal_records:
                lambda_key = _canonical_lambda(proposal.lambda_value)
                left_fit = (
                    fit_by_candidate_id.get(int(proposal.left_candidate_id))
                    if proposal.left_candidate_id is not None
                    else None
                )
                right_fit = (
                    fit_by_candidate_id.get(int(proposal.right_candidate_id))
                    if proposal.right_candidate_id is not None
                    else None
                )
                if left_fit is not None:
                    proposal_phi_by_lambda[lambda_key] = _clone_start(_fit_phi_start(left_fit))
                    if left_fit.solver_state is not None:
                        proposal_state_by_lambda[lambda_key] = left_fit.solver_state
                elif right_fit is not None:
                    proposal_phi_by_lambda[lambda_key] = _clone_start(_fit_phi_start(right_fit))
                    if right_fit.solver_state is not None:
                        proposal_state_by_lambda[lambda_key] = right_fit.solver_state
                proposal_starts = [_clone_start(start) for start in scalar_well_starts]
                if right_fit is not None:
                    proposal_starts.insert(0, _clone_start(_fit_phi_start(right_fit)))
                proposal_scalar_starts_by_lambda[lambda_key] = proposal_starts
                proposal_metadata_by_lambda[lambda_key] = {
                    "adaptive_interval_left_lambda": float(proposal.left_lambda),
                    "adaptive_interval_right_lambda": float(proposal.right_lambda),
                    "adaptive_interval_log10_width": float(proposal.log_width),
                    "adaptive_interval_priority_class": int(proposal.priority_key[0]),
                    "adaptive_interval_reason": str(proposal.reason),
                    "adaptive_interval_partition_changed": bool(proposal.partition_changed),
                    "adaptive_interval_nonagglomerative_or_numerically_inconsistent": bool(
                        proposal.nonagglomerative_or_numerically_inconsistent
                    ),
                }
            before = len(fit_by_lambda)
            _evaluate_lambda_sequence(
                proposals,
                search_round=adaptive_round,
                search_phase=f"adaptive_refine_{adaptive_round}",
                start_mode="full",
                compute_summary=True,
                phi_start_by_lambda=proposal_phi_by_lambda,
                solver_state_start_by_lambda=proposal_state_by_lambda,
                scalar_well_starts_by_lambda=proposal_scalar_starts_by_lambda,
                lambda_metadata_by_lambda=proposal_metadata_by_lambda,
            )
            if len(fit_by_lambda) == before:
                adaptive_search_stop_reason = "adaptive_no_new_lambdas"
                break
            adaptive_refinement_rounds_completed = adaptive_round
        else:
            adaptive_search_stop_reason = "adaptive_max_rounds_reached"
    adaptive_search_rounds_completed = int(adaptive_refinement_rounds_completed)
    search_df = (
        pd.DataFrame([row for _, _, row, _ in result_entries])
        .sort_values(["lambda", "selection_step"])
        .reset_index(drop=True)
    )
    partition_generation_elapsed_seconds = 0.0
    partition_curvature_elapsed_seconds = 0.0
    partition_ward_elapsed_seconds = 0.0
    partition_refine_ward_elapsed_seconds = 0.0
    partition_initial_generation_elapsed_seconds = 0.0
    partition_refine_generation_elapsed_seconds = 0.0
    partition_candidate_count = 0
    if likelihood_partition_pool_enabled:
        partition_generation_start_time = perf_counter()
        partition_k_grid = _likelihood_partition_k_grid(int(data.num_mutations))
        partition_curvature_start_time = perf_counter()
        partition_curvature = observed_curvature_at_pilot_torch(
            data,
            pilot_phi,
            major_prior=float(effective_fit_options.major_prior),
            eps=float(effective_fit_options.eps),
            torch_data=torch_data,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        partition_curvature_elapsed_seconds = float(perf_counter() - partition_curvature_start_time)
        partition_ward_start_time = perf_counter()
        partition_label_sets = hessian_weighted_ward_label_sets_torch(
            pilot_phi,
            partition_curvature,
            K_grid=partition_k_grid,
            device=runtime.device,
            dtype=runtime.dtype,
        )
        partition_ward_elapsed_seconds = float(perf_counter() - partition_ward_start_time)
        partition_initial_start_time = perf_counter()
        partition_candidates = generate_likelihood_partition_starts(
            data,
            exact_pilot=pilot_phi,
            major_prior=float(effective_fit_options.major_prior),
            eps=float(effective_fit_options.eps),
            K_grid=partition_k_grid,
            max_candidates_per_K=int(LIKELIHOOD_PARTITION_MAX_CANDIDATES_PER_K),
            cem_max_iter=int(LIKELIHOOD_PARTITION_CEM_MAX_ITER),
            refit_max_iter=int(LIKELIHOOD_PARTITION_REFIT_MAX_ITER),
            tol=float(effective_fit_options.tol),
            curvature=partition_curvature,
            label_sets=partition_label_sets,
            torch_data=torch_data,
            device=runtime.device,
            dtype=runtime.dtype,
            use_torch=True,
        )
        partition_initial_generation_elapsed_seconds = float(perf_counter() - partition_initial_start_time)
        partition_refine_k_grid, partition_refinement_reason = _likelihood_partition_refinement_k_grid(
            partition_candidates,
            partition_k_grid,
            num_mutations=int(data.num_mutations),
        )
        if partition_refine_k_grid:
            partition_refine_ward_start_time = perf_counter()
            partition_refine_label_sets = hessian_weighted_ward_label_sets_torch(
                pilot_phi,
                partition_curvature,
                K_grid=partition_refine_k_grid,
                device=runtime.device,
                dtype=runtime.dtype,
            )
            partition_refine_ward_elapsed_seconds = float(perf_counter() - partition_refine_ward_start_time)
            partition_refine_start_time = perf_counter()
            partition_refine_candidates = generate_likelihood_partition_starts(
                data,
                exact_pilot=pilot_phi,
                major_prior=float(effective_fit_options.major_prior),
                eps=float(effective_fit_options.eps),
                K_grid=partition_refine_k_grid,
                max_candidates_per_K=int(LIKELIHOOD_PARTITION_MAX_CANDIDATES_PER_K),
                cem_max_iter=int(LIKELIHOOD_PARTITION_CEM_MAX_ITER),
                refit_max_iter=int(LIKELIHOOD_PARTITION_REFIT_MAX_ITER),
                tol=float(effective_fit_options.tol),
                curvature=partition_curvature,
                label_sets=partition_refine_label_sets,
                torch_data=torch_data,
                device=runtime.device,
                dtype=runtime.dtype,
                use_torch=True,
            )
            partition_refine_generation_elapsed_seconds = float(perf_counter() - partition_refine_start_time)
            partition_candidates = _deduplicate_partition_candidates(
                partition_candidates + partition_refine_candidates
            )
        partition_combined_k_grid = sorted(set(partition_k_grid) | set(partition_refine_k_grid))
        partition_refine_k_set = set(partition_refine_k_grid)
        partition_generation_elapsed_seconds = float(perf_counter() - partition_generation_start_time)
        partition_candidate_count = int(len(partition_candidates))
        for partition_rank, partition_candidate in enumerate(partition_candidates, start=1):
            fit, evaluation, row, artifact = _evaluate_partition_candidate(
                data=data,
                fit_options=effective_fit_options,
                candidate=partition_candidate,
                candidate_rank=partition_rank,
                bic_df_scale=bic_df_scale,
                bic_cluster_penalty=bic_cluster_penalty,
                simulation_truth=simulation_truth,
                evaluate_candidate=evaluate_all_candidates,
                selection_method=selection_method,
                profile_name=profile_name,
                selection_step=next_step,
                selection_score=selection_score,
                static_metadata=static_metadata,
            )
            row["search_round"] = -1
            row["search_phase"] = "likelihood_partition"
            row["lambda_source"] = "likelihood_partition"
            row["lambda_search_mode"] = str(lambda_search_mode)
            row["selection_prepare_elapsed_seconds"] = float(prepare_elapsed_seconds)
            row["adaptive_candidate_budget"] = int(adaptive_max_candidates)
            row["adaptive_max_rounds"] = int(adaptive_max_rounds)
            row["adaptive_refine_per_round"] = int(adaptive_refine_per_round)
            row["adaptive_transition_probe_max_candidates"] = int(adaptive_transition_probe_max_candidates)
            row["adaptive_initial_anchor_count"] = int(adaptive_initial_anchor_count)
            row["likelihood_partition_pool_enabled"] = bool(likelihood_partition_pool_enabled)
            row["partition_generation_elapsed_seconds"] = float(partition_generation_elapsed_seconds)
            row["partition_curvature_elapsed_seconds"] = float(partition_curvature_elapsed_seconds)
            row["partition_ward_elapsed_seconds"] = float(partition_ward_elapsed_seconds)
            row["partition_refine_ward_elapsed_seconds"] = float(partition_refine_ward_elapsed_seconds)
            row["partition_initial_generation_elapsed_seconds"] = float(
                partition_initial_generation_elapsed_seconds
            )
            row["partition_refine_generation_elapsed_seconds"] = float(
                partition_refine_generation_elapsed_seconds
            )
            row["partition_candidate_count"] = int(partition_candidate_count)
            requested_k = _partition_candidate_requested_k(partition_candidate)
            row["partition_candidate_generation_pass"] = (
                "local_refine" if int(requested_k) in partition_refine_k_set else "sparse_anchor"
            )
            row["partition_candidate_refinement_reason"] = str(partition_refinement_reason)
            row["partition_candidate_k_grid_size"] = int(len(partition_combined_k_grid))
            row["partition_candidate_sparse_k_grid"] = ",".join(str(int(k)) for k in partition_k_grid)
            row["partition_candidate_refine_k_grid"] = ",".join(str(int(k)) for k in partition_refine_k_grid)
            row["partition_candidate_k_grid"] = ",".join(str(int(k)) for k in partition_combined_k_grid)
            row["lambda_bracket_min"] = np.nan if lambda_bracket is None else float(lambda_bracket.lambda_min)
            row["lambda_bracket_eq"] = np.nan if lambda_bracket is None else float(lambda_bracket.lambda_eq)
            row["lambda_bracket_full"] = np.nan if lambda_bracket is None else float(lambda_bracket.lambda_full)
            if lambda_bracket is None:
                row["lambda_full_residual"] = np.nan
            else:
                for diagnostic_name, diagnostic_value in lambda_bracket.diagnostics.items():
                    if np.isscalar(diagnostic_value):
                        row[str(diagnostic_name)] = float(diagnostic_value)
            candidate_id = int(len(result_entries))
            row["_candidate_id"] = candidate_id
            result_entries.append((fit, evaluation, row, artifact))
            if artifact.bic_partition_labels is not None:
                partition_labels_by_candidate_id[candidate_id] = np.asarray(
                    artifact.bic_partition_labels,
                    dtype=np.int64,
                ).copy()
            next_step += 1
        search_df = (
            pd.DataFrame([row for _, _, row, _ in result_entries])
            .sort_values(["lambda", "selection_step"])
            .reset_index(drop=True)
        )
    return _assemble_selection_result(
        search_df=search_df,
        data=data,
        normalized_score=normalized_score,
        result_entries=result_entries,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
        selection_method=selection_method,
        profile_name=profile_name,
        lambda_search_mode=lambda_search_mode,
        lambda_bracket=lambda_bracket,
        adaptive_search_stop_reason=adaptive_search_stop_reason,
        adaptive_search_rounds_completed=adaptive_search_rounds_completed,
        adaptive_refinement_rounds_completed=adaptive_refinement_rounds_completed,
        selection_start_time=selection_start_time,
    )


def select_model(
    *,
    data: TumorData,
    simulation_root: Path | None,
    lambda_grid: list[float] | None,
    lambda_grid_mode: str,
    fit_options: FitOptions,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    selection_score: str,
    use_warm_starts: bool,
    evaluate_all_candidates: bool,
    finalize_selected_fit: bool = True,
) -> BICSelectionResult:
    _normalize_selection_score_name(selection_score)

    effective_lambda_grid_mode = str(lambda_grid_mode)
    effective_bic_df_scale = float(bic_df_scale)
    effective_bic_cluster_penalty = float(bic_cluster_penalty)
    profile_name = "adaptive_bic_default"

    effective_lambda_grid_mode_normalized = str(effective_lambda_grid_mode).strip().lower()
    if effective_lambda_grid_mode_normalized not in LAMBDA_GRID_MODES:
        raise ValueError(f"Unknown lambda_grid_mode: {effective_lambda_grid_mode}")

    selection_method = "lambda_path_grid" if lambda_grid is not None else "adaptive_bic_path"

    return _grid_search_selection(
        data=data,
        simulation_root=simulation_root,
        lambda_grid=lambda_grid,
        lambda_grid_mode=effective_lambda_grid_mode,
        fit_options=fit_options,
        bic_df_scale=effective_bic_df_scale,
        bic_cluster_penalty=effective_bic_cluster_penalty,
        use_warm_starts=use_warm_starts,
        evaluate_all_candidates=evaluate_all_candidates,
        profile_name=profile_name,
        selection_method=selection_method,
        selection_score=selection_score,
        finalize_selected_fit=finalize_selected_fit,
    )


__all__ = [
    "BICSelectionResult",
    "ModelSelectionResult",
    "SelectionArtifact",
    "SimulationDiagnostics",
    "select_model",
]
