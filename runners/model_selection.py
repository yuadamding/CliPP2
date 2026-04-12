from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.model import FitOptions, FitResult, fit_single_stage_em
from ..core.fusion_solver import (
    compute_exact_observed_data_pilot,
    compute_pooled_observed_data_start,
    compute_scalar_cell_wells,
    compute_scalar_well_start_bank,
    resolve_pairwise_fusion_graph,
)
from ..core.fusion.torch_backend import resolve_runtime, to_torch_tumor_data
from ..io.data import TumorData
from ..metrics.evaluation import (
    SimulationEvaluation,
    SimulationTruth,
    evaluate_ari_against_simulation,
    evaluate_fit_against_simulation,
    load_simulation_truth,
)
from .selection import compute_classic_bic, compute_extended_bic, default_lambda_grid
from .settings import recommend_settings_from_data

ORACLE_REFINE_ROUNDS = 3
ORACLE_MAX_SEARCH_ROUNDS = 8
ORACLE_REFINE_POINTS = 17
ORACLE_ULTRA_DENSE_POINTS = 41
ORACLE_EXPANSION_FACTOR = 8.0
ORACLE_MIN_LAMBDA = 1e-6
ORACLE_MAX_LAMBDA = 1e6
ORACLE_POLISH_TOP_K = 5
ORACLE_LIGHT_BOUNDARY_POINTS = 9
ORACLE_LIGHT_LOCAL_ROUNDS = 2
ORACLE_LIGHT_LOCAL_POINTS = 11
ORACLE_LIGHT_LOG10_SPAN_TOL = 0.05
ORACLE_LIGHT_POLISH_TOP_K = 3


@dataclass(frozen=True)
class OracleSearchBehavior:
    ari_only_evaluation: bool
    exploratory_start_mode: str
    exploratory_compute_summary: bool
    use_heavy_refinement: bool
    finalize_selected_fit: bool


@dataclass
class ModelSelectionResult:
    best_fit: FitResult
    best_evaluation: SimulationEvaluation | None
    search_df: pd.DataFrame
    bic_df_scale: float
    bic_cluster_penalty: float
    selection_method: str
    profile_name: str
    selection_metric_value: float | None
    selection_lambda_min: float | None
    selection_lambda_max: float | None
    selection_lambda_count: int
    best_ari: float | None
    ari_optimal_lambda_min: float | None
    ari_optimal_lambda_max: float | None
    ari_optimal_lambda_count: int
    best_converged_ari: float | None
    best_converged_lambda_min: float | None
    best_converged_lambda_max: float | None
    best_converged_lambda_count: int
    selection_hits_lower_boundary: bool
    selection_hits_upper_boundary: bool
    selection_boundary_unresolved: bool
    selection_optimum_resolved: bool
    ari_hits_lower_boundary: bool
    ari_hits_upper_boundary: bool
    ari_boundary_unresolved: bool
    ari_optimum_resolved: bool
    oracle_search_rounds_completed: int
    oracle_search_stop_reason: str
    num_candidates: int
    num_converged_candidates: int
    selection_used_convergence_fallback: bool


def _normalize_selection_score_name(selection_score: str) -> str:
    normalized = str(selection_score).strip().lower()
    if normalized in {"ebic", "refit_ebic"}:
        return "summary_ebic"
    if normalized in {"classic_bic", "classic_refit_bic"}:
        return "summary_classic_bic"
    if normalized == "oracle_ari":
        return "oracle_ari"
    raise ValueError(f"Unknown selection_score: {selection_score}")


def _selection_score_value(
    *,
    loglik: float,
    num_clusters: int,
    data: TumorData,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    selection_score: str,
) -> tuple[float, float, float]:
    classic_bic = compute_classic_bic(loglik, num_clusters, data)
    extended_bic = compute_extended_bic(
        loglik,
        num_clusters,
        data,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
    )
    normalized = _normalize_selection_score_name(selection_score)
    if normalized == "summary_ebic":
        return float(extended_bic), float(classic_bic), float(extended_bic)
    if normalized == "summary_classic_bic":
        return float(classic_bic), float(classic_bic), float(extended_bic)
    if normalized == "oracle_ari":
        return float(extended_bic), float(classic_bic), float(extended_bic)
    raise ValueError(f"Unknown normalized selection_score: {selection_score}")


def _optimal_lambda_range(
    values: np.ndarray,
    lambdas: np.ndarray,
    *,
    maximize: bool,
) -> tuple[float | None, float | None, int, float | None, np.ndarray]:
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        empty_mask = np.zeros_like(values, dtype=bool)
        return None, None, 0, None, empty_mask

    finite_values = values[finite_mask]
    best_value = float(np.max(finite_values) if maximize else np.min(finite_values))
    optimal_mask = finite_mask & np.isclose(values, best_value, rtol=0.0, atol=1e-12)
    lambda_values = np.unique(np.round(lambdas[optimal_mask].astype(float, copy=False), 12))
    return (
        float(np.min(lambda_values)),
        float(np.max(lambda_values)),
        int(lambda_values.size),
        best_value,
        optimal_mask,
    )


def _canonical_lambda(value: float) -> float:
    return float(np.round(float(value), 12))


def _sorted_unique_lambdas(values: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array) & (array > 0.0)]
    if array.size == 0:
        return []
    return [float(value) for value in np.unique(np.round(np.sort(array), 12))]


def _densify_lambda_grid(base_grid: list[float]) -> list[float]:
    sorted_grid = _sorted_unique_lambdas(base_grid)
    if len(sorted_grid) <= 1:
        return sorted_grid
    augmented = list(sorted_grid)
    for left, right in zip(sorted_grid[:-1], sorted_grid[1:]):
        augmented.append(float(np.sqrt(left * right)))
    return _sorted_unique_lambdas(augmented)


def _adaptive_refine_points(lower: float, upper: float) -> int:
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower or lower <= 0.0:
        return ORACLE_REFINE_POINTS
    log_span = float(np.log10(upper) - np.log10(lower))
    adaptive = ORACLE_REFINE_POINTS + int(np.ceil(max(log_span, 0.0) * 4.0))
    return int(min(max(adaptive, ORACLE_REFINE_POINTS), 33))


def _lambda_boundary_flags(
    evaluated_lambdas: list[float],
    *,
    best_lambda_min: float | None,
    best_lambda_max: float | None,
) -> tuple[bool, bool]:
    sorted_lambdas = _sorted_unique_lambdas(evaluated_lambdas)
    if not sorted_lambdas or best_lambda_min is None or best_lambda_max is None:
        return False, False
    lower_hit = np.isclose(best_lambda_min, sorted_lambdas[0], rtol=0.0, atol=1e-12)
    upper_hit = np.isclose(best_lambda_max, sorted_lambdas[-1], rtol=0.0, atol=1e-12)
    return bool(lower_hit), bool(upper_hit)


def _lambda_boundary_unresolved(
    *,
    evaluated_lambdas: list[float],
    lower_hit: bool,
    upper_hit: bool,
) -> bool:
    sorted_lambdas = _sorted_unique_lambdas(evaluated_lambdas)
    if not sorted_lambdas:
        return False
    return bool(lower_hit or upper_hit)


def _oracle_refinement_grid(
    evaluated_lambdas: list[float],
    *,
    best_lambda_min: float | None,
    best_lambda_max: float | None,
    lower_hit: bool,
    upper_hit: bool,
) -> list[float]:
    if not evaluated_lambdas or best_lambda_min is None or best_lambda_max is None:
        return []
    sorted_lambdas = _sorted_unique_lambdas(evaluated_lambdas)
    lower_neighbors = [value for value in sorted_lambdas if value < best_lambda_min]
    upper_neighbors = [value for value in sorted_lambdas if value > best_lambda_max]

    if lower_hit:
        lower = max(best_lambda_min / ORACLE_EXPANSION_FACTOR, ORACLE_MIN_LAMBDA)
    else:
        lower = lower_neighbors[-1] if lower_neighbors else max(best_lambda_min / ORACLE_EXPANSION_FACTOR, ORACLE_MIN_LAMBDA)

    if upper_hit:
        upper = min(best_lambda_max * ORACLE_EXPANSION_FACTOR, ORACLE_MAX_LAMBDA)
    else:
        upper = upper_neighbors[0] if upper_neighbors else min(best_lambda_max * ORACLE_EXPANSION_FACTOR, ORACLE_MAX_LAMBDA)

    lower = max(float(lower), ORACLE_MIN_LAMBDA)
    upper = min(float(upper), ORACLE_MAX_LAMBDA)
    if upper <= lower:
        return []

    points = _adaptive_refine_points(lower, upper)
    candidates: list[float] = []

    def _append_geomspace(left: float, right: float, num: int) -> None:
        if not np.isfinite(left) or not np.isfinite(right):
            return
        left = float(max(left, ORACLE_MIN_LAMBDA))
        right = float(min(right, ORACLE_MAX_LAMBDA))
        if right <= left:
            return
        candidates.extend(float(value) for value in np.geomspace(left, right, num=max(int(num), 3), dtype=float))

    # Always refine across the current shoulder interval.
    _append_geomspace(lower, upper, points)

    # Densify around the current best interval itself.
    if best_lambda_max > best_lambda_min:
        _append_geomspace(best_lambda_min, best_lambda_max, max(points, ORACLE_REFINE_POINTS + 4))
    else:
        center = float(best_lambda_min)
        candidates.extend(
            [
                center,
                max(center / np.sqrt(ORACLE_EXPANSION_FACTOR), ORACLE_MIN_LAMBDA),
                min(center * np.sqrt(ORACLE_EXPANSION_FACTOR), ORACLE_MAX_LAMBDA),
            ]
        )

    # Refine between the best interval and its immediate interior neighbors.
    if lower_neighbors:
        _append_geomspace(lower_neighbors[-1], best_lambda_min, max(ORACLE_REFINE_POINTS // 2 + 3, 7))
    if upper_neighbors:
        _append_geomspace(best_lambda_max, upper_neighbors[0], max(ORACLE_REFINE_POINTS // 2 + 3, 7))

    # If the optimum touches a boundary, push farther outward on that side.
    if lower_hit and lower < best_lambda_min:
        _append_geomspace(lower, best_lambda_min, max(points, ORACLE_REFINE_POINTS + 4))
    if upper_hit and upper > best_lambda_max:
        _append_geomspace(best_lambda_max, upper, max(points, ORACLE_REFINE_POINTS + 4))

    return _sorted_unique_lambdas(candidates)


def _oracle_ultra_dense_grid(
    evaluated_lambdas: list[float],
    *,
    best_lambda_min: float | None,
    best_lambda_max: float | None,
) -> list[float]:
    if not evaluated_lambdas or best_lambda_min is None or best_lambda_max is None:
        return []
    sorted_lambdas = _sorted_unique_lambdas(evaluated_lambdas)
    if not sorted_lambdas:
        return []

    lower_neighbors = [value for value in sorted_lambdas if value < best_lambda_min]
    upper_neighbors = [value for value in sorted_lambdas if value > best_lambda_max]

    lower = lower_neighbors[-1] if lower_neighbors else max(best_lambda_min / ORACLE_EXPANSION_FACTOR, ORACLE_MIN_LAMBDA)
    upper = upper_neighbors[0] if upper_neighbors else min(best_lambda_max * ORACLE_EXPANSION_FACTOR, ORACLE_MAX_LAMBDA)
    lower = max(float(lower), ORACLE_MIN_LAMBDA)
    upper = min(float(upper), ORACLE_MAX_LAMBDA)
    if upper <= lower:
        return []

    candidates: list[float] = []

    def _append_geomspace(left: float, right: float, num: int) -> None:
        if not np.isfinite(left) or not np.isfinite(right):
            return
        left = float(max(left, ORACLE_MIN_LAMBDA))
        right = float(min(right, ORACLE_MAX_LAMBDA))
        if right <= left:
            return
        candidates.extend(float(value) for value in np.geomspace(left, right, num=max(int(num), 3), dtype=float))

    # Dense sweep over the local bracket around the selected lambda interval.
    _append_geomspace(lower, upper, ORACLE_ULTRA_DENSE_POINTS)

    if best_lambda_max > best_lambda_min:
        # If the optimum is already an interval, search very densely inside it.
        _append_geomspace(best_lambda_min, best_lambda_max, ORACLE_ULTRA_DENSE_POINTS)
    else:
        center = float(best_lambda_min)
        tight_lower = max(np.sqrt(lower * center), ORACLE_MIN_LAMBDA)
        tight_upper = min(np.sqrt(center * upper), ORACLE_MAX_LAMBDA)
        _append_geomspace(tight_lower, tight_upper, ORACLE_ULTRA_DENSE_POINTS)
        candidates.extend(
            [
                center,
                max(center / np.sqrt(ORACLE_EXPANSION_FACTOR), ORACLE_MIN_LAMBDA),
                min(center * np.sqrt(ORACLE_EXPANSION_FACTOR), ORACLE_MAX_LAMBDA),
            ]
        )

    return _sorted_unique_lambdas(candidates)


def _oracle_light_boundary_grid(
    evaluated_lambdas: list[float],
    *,
    best_lambda_min: float | None,
    best_lambda_max: float | None,
    lower_hit: bool,
    upper_hit: bool,
) -> list[float]:
    candidates = _oracle_refinement_grid(
        evaluated_lambdas,
        best_lambda_min=best_lambda_min,
        best_lambda_max=best_lambda_max,
        lower_hit=lower_hit,
        upper_hit=upper_hit,
    )
    if len(candidates) <= ORACLE_LIGHT_BOUNDARY_POINTS:
        return candidates
    pick_idx = np.linspace(0, len(candidates) - 1, num=ORACLE_LIGHT_BOUNDARY_POINTS, dtype=int)
    return [candidates[idx] for idx in sorted(set(pick_idx.tolist()))]


def _oracle_local_zoom_grid(
    evaluated_lambdas: list[float],
    *,
    best_lambda_min: float | None,
    best_lambda_max: float | None,
    points: int,
) -> list[float]:
    if not evaluated_lambdas or best_lambda_min is None or best_lambda_max is None:
        return []
    sorted_lambdas = _sorted_unique_lambdas(evaluated_lambdas)
    if not sorted_lambdas:
        return []

    lower_neighbors = [value for value in sorted_lambdas if value < best_lambda_min]
    upper_neighbors = [value for value in sorted_lambdas if value > best_lambda_max]
    lower = lower_neighbors[-1] if lower_neighbors else max(best_lambda_min / np.sqrt(ORACLE_EXPANSION_FACTOR), ORACLE_MIN_LAMBDA)
    upper = upper_neighbors[0] if upper_neighbors else min(best_lambda_max * np.sqrt(ORACLE_EXPANSION_FACTOR), ORACLE_MAX_LAMBDA)
    lower = max(float(lower), ORACLE_MIN_LAMBDA)
    upper = min(float(upper), ORACLE_MAX_LAMBDA)
    if upper <= lower:
        return []

    candidates: list[float] = []

    def _append_geomspace(left: float, right: float, num: int) -> None:
        if not np.isfinite(left) or not np.isfinite(right):
            return
        left = float(max(left, ORACLE_MIN_LAMBDA))
        right = float(min(right, ORACLE_MAX_LAMBDA))
        if right <= left:
            return
        candidates.extend(float(value) for value in np.geomspace(left, right, num=max(int(num), 3), dtype=float))

    _append_geomspace(lower, upper, points)
    if best_lambda_max > best_lambda_min:
        _append_geomspace(best_lambda_min, best_lambda_max, max(points, 7))
    else:
        center = float(best_lambda_min)
        tight_lower = max(np.sqrt(lower * center), ORACLE_MIN_LAMBDA)
        tight_upper = min(np.sqrt(center * upper), ORACLE_MAX_LAMBDA)
        _append_geomspace(tight_lower, tight_upper, max(points, 7))
        candidates.extend(
            [
                center,
                max(center / np.sqrt(np.sqrt(ORACLE_EXPANSION_FACTOR)), ORACLE_MIN_LAMBDA),
                min(center * np.sqrt(np.sqrt(ORACLE_EXPANSION_FACTOR)), ORACLE_MAX_LAMBDA),
            ]
        )

    return _sorted_unique_lambdas(candidates)


def _lambda_log10_span(
    *,
    lower: float | None,
    upper: float | None,
) -> float:
    if lower is None or upper is None or not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0.0 or upper <= 0.0:
        return float("inf")
    if upper < lower:
        return float("inf")
    return float(np.log10(upper) - np.log10(lower))


def _oracle_local_bracket(
    evaluated_lambdas: list[float],
    *,
    best_lambda_min: float | None,
    best_lambda_max: float | None,
) -> tuple[float | None, float | None]:
    if not evaluated_lambdas or best_lambda_min is None or best_lambda_max is None:
        return None, None
    sorted_lambdas = _sorted_unique_lambdas(evaluated_lambdas)
    if not sorted_lambdas:
        return None, None
    lower_neighbors = [value for value in sorted_lambdas if value < best_lambda_min]
    upper_neighbors = [value for value in sorted_lambdas if value > best_lambda_max]
    lower = lower_neighbors[-1] if lower_neighbors else max(best_lambda_min / np.sqrt(ORACLE_EXPANSION_FACTOR), ORACLE_MIN_LAMBDA)
    upper = upper_neighbors[0] if upper_neighbors else min(best_lambda_max * np.sqrt(ORACLE_EXPANSION_FACTOR), ORACLE_MAX_LAMBDA)
    return float(lower), float(upper)


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
    phi_start: np.ndarray | None,
    exact_pilot: np.ndarray | None,
    pooled_start: np.ndarray | None,
    scalar_well_starts: list[np.ndarray] | None,
    start_mode: str,
    runtime,
    torch_data,
    compute_summary: bool,
    selection_method: str,
    profile_name: str,
    selection_step: int,
    lambda_value: float,
    selection_score: str,
) -> tuple[FitResult, SimulationEvaluation | None, dict[str, float | int | str | bool]]:
    canonical_score_name = _normalize_selection_score_name(selection_score)
    effective_fit_options = fit_options if candidate_fit_options is None else candidate_fit_options
    fit = fit_single_stage_em(
        data=data,
        options=replace(effective_fit_options, lambda_value=float(lambda_value)),
        phi_start=phi_start,
        exact_pilot=exact_pilot,
        pooled_start=pooled_start,
        scalar_well_starts=scalar_well_starts,
        start_mode=start_mode,
        runtime=runtime,
        torch_data=torch_data,
        compute_summary=compute_summary,
    )
    if canonical_score_name == "oracle_ari":
        bic = float("nan")
        classic_bic = float("nan")
        extended_bic = float("nan")
    else:
        bic, classic_bic, extended_bic = _selection_score_value(
            loglik=fit.summary_loglik,
            num_clusters=fit.n_clusters,
            data=data,
            bic_df_scale=bic_df_scale,
            bic_cluster_penalty=bic_cluster_penalty,
            selection_score=selection_score,
        )
    fit.bic = bic
    fit.classic_bic = classic_bic
    fit.extended_bic = extended_bic
    fit.selection_score_name = canonical_score_name

    evaluation = None
    ari_value = np.nan
    cp_rmse_value = np.nan
    multiplicity_f1_value = np.nan
    estimated_clonal_fraction_value = np.nan
    true_clonal_fraction_value = np.nan
    clonal_fraction_error_value = np.nan
    estimated_clusters_value = np.nan
    true_clusters_value = np.nan
    n_eval_mutations_value = np.nan
    n_filtered_mutations_value = np.nan
    if evaluate_candidate and simulation_truth is not None:
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
            )
            ari_value = float(evaluation.ari)
            cp_rmse_value = float(evaluation.cp_rmse)
            multiplicity_f1_value = float(evaluation.multiplicity_f1)
            estimated_clonal_fraction_value = float(evaluation.estimated_clonal_fraction)
            true_clonal_fraction_value = float(evaluation.true_clonal_fraction)
            clonal_fraction_error_value = float(evaluation.clonal_fraction_error)
            estimated_clusters_value = int(evaluation.estimated_clusters)
            true_clusters_value = int(evaluation.true_clusters)
            n_eval_mutations_value = int(evaluation.n_eval_mutations)
            n_filtered_mutations_value = int(evaluation.n_filtered_mutations)

    row: dict[str, float | int | str | bool] = {
        "tumor_id": data.tumor_id,
        "selection_method": selection_method,
        "selection_profile": profile_name,
        "selection_step": int(selection_step),
        "lambda": float(fit.lambda_value),
        "bic_df_scale": float(bic_df_scale),
        "bic_cluster_penalty": float(bic_cluster_penalty),
        "bic": float(bic),
        "selection_score_name": str(canonical_score_name),
        "classic_bic": float(classic_bic),
        "extended_bic": float(extended_bic),
        "loglik": float(fit.loglik),
        "summary_loglik": float(fit.summary_loglik),
        "penalized_objective": float(fit.penalized_objective),
        "n_clusters": int(fit.n_clusters),
        "converged": bool(fit.converged),
        "iterations": int(fit.iterations),
        "device": str(fit.device),
        "graph_name": str(fit.graph_name),
        "evaluation_mode": "ari_only" if ari_only_evaluation else "full",
        "fit_compute_summary": bool(compute_summary),
        "fit_start_mode": str(start_mode),
        "ARI": ari_value,
        "cp_rmse": cp_rmse_value,
        "multiplicity_f1": multiplicity_f1_value,
        "estimated_clonal_fraction": estimated_clonal_fraction_value,
        "true_clonal_fraction": true_clonal_fraction_value,
        "clonal_fraction_error": clonal_fraction_error_value,
        "estimated_clusters": estimated_clusters_value,
        "true_clusters": true_clusters_value,
        "n_eval_mutations": n_eval_mutations_value,
        "n_filtered_mutations": n_filtered_mutations_value,
    }
    return fit, evaluation, row


def _oracle_candidate_frame(search_df: pd.DataFrame) -> pd.DataFrame:
    if search_df.empty or "ARI" not in search_df.columns:
        return search_df.iloc[0:0].copy()
    oracle_df = search_df.loc[np.isfinite(search_df["ARI"].to_numpy(dtype=float))].copy()
    return oracle_df.sort_values(["lambda", "selection_step"]).reset_index(drop=True)


def _representative_optimal_row(
    tied_df: pd.DataFrame,
    *,
    lambda_min: float | None,
    lambda_max: float | None,
) -> pd.Series:
    if tied_df.empty:
        raise ValueError("tied_df must contain at least one optimal candidate row.")
    if lambda_min is None or lambda_max is None:
        return tied_df.sort_values(
            ["converged", "iterations", "lambda", "selection_step"],
            ascending=[False, False, True, True],
        ).iloc[0]

    if np.isclose(lambda_min, lambda_max, rtol=0.0, atol=1e-12):
        target_lambda = float(lambda_min)
    else:
        target_lambda = float(np.sqrt(float(lambda_min) * float(lambda_max)))

    ranked_df = tied_df.copy()
    ranked_df["_repr_log_distance"] = np.abs(
        np.log(ranked_df["lambda"].to_numpy(dtype=float)) - np.log(target_lambda)
    )
    return ranked_df.sort_values(
        ["_repr_log_distance", "converged", "iterations", "lambda", "selection_step"],
        ascending=[True, False, False, True, True],
    ).iloc[0]


def _top_unique_oracle_lambdas(
    oracle_df: pd.DataFrame,
    *,
    max_lambdas: int,
) -> list[float]:
    if oracle_df.empty or max_lambdas <= 0:
        return []
    ranked = oracle_df.sort_values(
        ["ARI", "converged", "iterations", "lambda", "selection_step"],
        ascending=[False, False, False, True, True],
        na_position="last",
    )
    unique: list[float] = []
    seen: set[float] = set()
    for value in ranked["lambda"].to_numpy(dtype=float):
        key = _canonical_lambda(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(float(value))
        if len(unique) >= int(max_lambdas):
            break
    return unique


def _prefer_fit_candidate(candidate: FitResult, incumbent: FitResult | None) -> bool:
    if incumbent is None:
        return True
    if candidate.converged and not incumbent.converged:
        return True
    if candidate.converged != incumbent.converged:
        return False
    return bool(candidate.penalized_objective < incumbent.penalized_objective - 1e-8)


def _oracle_search_behavior(
    *,
    normalized_score: str,
    explicit_lambda_grid: bool,
    use_warm_starts: bool,
    finalize_selected_fit: bool,
) -> OracleSearchBehavior:
    oracle_ari = normalized_score == "oracle_ari"
    return OracleSearchBehavior(
        ari_only_evaluation=oracle_ari,
        exploratory_start_mode="warm_only" if oracle_ari and use_warm_starts else "full",
        exploratory_compute_summary=not oracle_ari,
        # Heavy oracle refinement is too expensive for benchmark-style ARI search
        # when we are not even finalizing a selected fit. In that case we still
        # use a dynamic lambda search, but via the lighter boundary-expand/local-zoom path.
        use_heavy_refinement=bool(oracle_ari and not explicit_lambda_grid and finalize_selected_fit),
        finalize_selected_fit=bool(oracle_ari and finalize_selected_fit),
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
) -> ModelSelectionResult:
    explicit_lambda_grid = lambda_grid is not None
    if lambda_grid is None:
        lambda_grid = default_lambda_grid(data, mode=lambda_grid_mode)
    normalized_score = _normalize_selection_score_name(selection_score)
    oracle_behavior = _oracle_search_behavior(
        normalized_score=normalized_score,
        explicit_lambda_grid=explicit_lambda_grid,
        use_warm_starts=use_warm_starts,
        finalize_selected_fit=finalize_selected_fit,
    )
    lambda_grid = _sorted_unique_lambdas(lambda_grid)
    if oracle_behavior.use_heavy_refinement:
        lambda_grid = _densify_lambda_grid(lambda_grid)

    runtime = resolve_runtime(fit_options.device, dtype=fit_options.dtype)
    torch_data = to_torch_tumor_data(data, runtime)
    pilot_phi, secondary_wells, valid_secondary = compute_scalar_cell_wells(
        data,
        major_prior=float(fit_options.major_prior),
        eps=float(fit_options.eps),
        tol=max(float(fit_options.tol), 1e-6),
        max_iter=max(int(fit_options.inner_max_iter), 16),
    )
    effective_graph = resolve_pairwise_fusion_graph(
        data.num_mutations,
        graph=fit_options.graph,
        pilot_phi=pilot_phi,
        gamma=float(fit_options.adaptive_weight_gamma),
        tau=max(float(fit_options.adaptive_weight_floor), float(fit_options.eps)),
        baseline=float(fit_options.adaptive_weight_baseline),
    )
    effective_fit_options = replace(fit_options, graph=effective_graph)
    pooled_start = compute_pooled_observed_data_start(
        data,
        runtime=runtime,
        major_prior=float(fit_options.major_prior),
        eps=float(fit_options.eps),
        tol=max(float(fit_options.tol), 1e-6),
        max_iter=max(int(fit_options.inner_max_iter), 16),
        beta_hints=pilot_phi,
    )
    scalar_well_starts = compute_scalar_well_start_bank(
        data,
        major_prior=float(fit_options.major_prior),
        eps=float(fit_options.eps),
        tol=max(float(fit_options.tol), 1e-6),
        max_iter=max(int(fit_options.inner_max_iter), 16),
        exact_pilot=pilot_phi,
        secondary_wells=secondary_wells,
        valid_secondary=valid_secondary,
    )
    simulation_truth: SimulationTruth | None = None
    if evaluate_all_candidates and simulation_root is not None and (simulation_root / data.tumor_id).exists():
        simulation_truth = load_simulation_truth(data, simulation_root)
    result_entries: list[tuple[FitResult, SimulationEvaluation | None, dict[str, float | int | str | bool]]] = []
    fit_by_lambda: dict[float, FitResult] = {}
    next_step = 0

    def _nearest_phi_start(target_lambda: float) -> np.ndarray:
        if not fit_by_lambda:
            return pilot_phi.copy()
        nearest_lambda = min(
            fit_by_lambda,
            key=lambda value: abs(np.log(value) - np.log(target_lambda)),
        )
        return fit_by_lambda[nearest_lambda].phi.copy()

    def _evaluate_lambda_sequence(
        lambda_values_to_run: list[float],
        *,
        search_round: int,
        search_phase: str,
        allow_revisit: bool = False,
        candidate_fit_options: FitOptions | None = None,
        ari_only_evaluation: bool = False,
        start_mode: str = "full",
        compute_summary: bool = True,
    ) -> None:
        nonlocal next_step
        ordered_lambdas = [
            value
            for value in _sorted_unique_lambdas(lambda_values_to_run)
            if allow_revisit or _canonical_lambda(value) not in fit_by_lambda
        ]
        if not ordered_lambdas:
            return

        previous_phi: np.ndarray | None = None
        for lambda_value in ordered_lambdas:
            if use_warm_starts:
                phi_start = previous_phi.copy() if previous_phi is not None else _nearest_phi_start(lambda_value)
            else:
                phi_start = pilot_phi.copy()
            fit, evaluation, row = _evaluate_candidate(
                data=data,
                fit_options=effective_fit_options,
                candidate_fit_options=candidate_fit_options,
                bic_df_scale=bic_df_scale,
                bic_cluster_penalty=bic_cluster_penalty,
                simulation_root=simulation_root,
                simulation_truth=simulation_truth,
                evaluate_candidate=evaluate_all_candidates,
                ari_only_evaluation=ari_only_evaluation,
                phi_start=phi_start,
                exact_pilot=pilot_phi,
                pooled_start=pooled_start,
                scalar_well_starts=scalar_well_starts,
                start_mode=start_mode,
                runtime=runtime,
                torch_data=torch_data,
                compute_summary=compute_summary,
                selection_method=selection_method,
                profile_name=profile_name,
                selection_step=next_step,
                lambda_value=lambda_value,
                selection_score=selection_score,
            )
            row["search_round"] = int(search_round)
            row["search_phase"] = str(search_phase)
            row["_candidate_id"] = int(len(result_entries))
            result_entries.append((fit, evaluation, row))
            lambda_key = _canonical_lambda(lambda_value)
            incumbent = fit_by_lambda.get(lambda_key)
            if _prefer_fit_candidate(fit, incumbent):
                fit_by_lambda[lambda_key] = fit
            next_step += 1
            if use_warm_starts:
                previous_phi = fit.phi.copy()

    _evaluate_lambda_sequence(
        lambda_grid,
        search_round=0,
        search_phase="base",
        ari_only_evaluation=oracle_behavior.ari_only_evaluation,
        start_mode=oracle_behavior.exploratory_start_mode,
        compute_summary=oracle_behavior.exploratory_compute_summary,
    )

    oracle_search_rounds_completed = 0
    oracle_search_stop_reason = "not_applicable"
    oracle_ultra_dense_ran = False
    if oracle_behavior.use_heavy_refinement:
        while oracle_search_rounds_completed < ORACLE_MAX_SEARCH_ROUNDS:
            interim_df = pd.DataFrame([row for _, _, row in result_entries])
            interim_selection_df = _oracle_candidate_frame(interim_df)
            if interim_selection_df.empty:
                oracle_search_stop_reason = "no_finite_ari_candidates"
                break
            best_lambda_min, best_lambda_max, _, _, _ = _optimal_lambda_range(
                interim_selection_df["ARI"].to_numpy(dtype=float),
                interim_selection_df["lambda"].to_numpy(dtype=float),
                maximize=True,
            )
            lower_hit, upper_hit = _lambda_boundary_flags(
                interim_selection_df["lambda"].to_numpy(dtype=float),
                best_lambda_min=best_lambda_min,
                best_lambda_max=best_lambda_max,
            )
            boundary_unresolved = _lambda_boundary_unresolved(
                evaluated_lambdas=interim_selection_df["lambda"].to_numpy(dtype=float),
                lower_hit=lower_hit,
                upper_hit=upper_hit,
            )
            if oracle_search_rounds_completed >= ORACLE_REFINE_ROUNDS and not boundary_unresolved:
                oracle_search_stop_reason = "interior_optimum"
                break
            refinement_grid = _oracle_refinement_grid(
                [float(value) for value in fit_by_lambda.keys()],
                best_lambda_min=best_lambda_min,
                best_lambda_max=best_lambda_max,
                lower_hit=lower_hit,
                upper_hit=upper_hit,
            )
            if not refinement_grid:
                oracle_search_stop_reason = "boundary_unresolved_at_search_limit" if boundary_unresolved else "no_refinement_needed"
                break
            before = len(fit_by_lambda)
            oracle_search_rounds_completed += 1
            _evaluate_lambda_sequence(
                refinement_grid,
                search_round=oracle_search_rounds_completed,
                search_phase=f"oracle_refine_{oracle_search_rounds_completed}",
                ari_only_evaluation=True,
                start_mode="warm_plus_pilot" if use_warm_starts else "full",
                compute_summary=False,
            )
            if len(fit_by_lambda) == before:
                oracle_search_stop_reason = "no_new_lambdas_generated"
                break
        else:
            oracle_search_stop_reason = "max_search_rounds_reached"

        final_interim_df = pd.DataFrame([row for _, _, row in result_entries])
        final_interim_selection_df = _oracle_candidate_frame(final_interim_df)
        ultra_best_lambda_min, ultra_best_lambda_max, _, _, _ = _optimal_lambda_range(
            final_interim_selection_df["ARI"].to_numpy(dtype=float),
            final_interim_selection_df["lambda"].to_numpy(dtype=float),
            maximize=True,
        )
        ultra_dense_grid = _oracle_ultra_dense_grid(
            [float(value) for value in fit_by_lambda.keys()],
            best_lambda_min=ultra_best_lambda_min,
            best_lambda_max=ultra_best_lambda_max,
        )
        before_ultra_dense = len(fit_by_lambda)
        _evaluate_lambda_sequence(
            ultra_dense_grid,
            search_round=oracle_search_rounds_completed + 1,
            search_phase="oracle_ultra_dense",
            ari_only_evaluation=True,
            start_mode="warm_plus_pilot" if use_warm_starts else "full",
            compute_summary=False,
        )
        if len(fit_by_lambda) > before_ultra_dense:
            oracle_ultra_dense_ran = True
            oracle_search_rounds_completed += 1

        final_oracle_df = _oracle_candidate_frame(pd.DataFrame([row for _, _, row in result_entries]))
        polish_lambdas = _top_unique_oracle_lambdas(
            final_oracle_df,
            max_lambdas=ORACLE_POLISH_TOP_K,
        )
        if polish_lambdas:
            polish_fit_options = replace(
                effective_fit_options,
                outer_max_iter=max(int(effective_fit_options.outer_max_iter) * 2, int(effective_fit_options.outer_max_iter) + 4),
                inner_max_iter=max(int(effective_fit_options.inner_max_iter) * 3, int(effective_fit_options.inner_max_iter) + 30),
                tol=max(float(effective_fit_options.tol) * 0.5, 1e-6),
            )
            _evaluate_lambda_sequence(
                polish_lambdas,
                search_round=oracle_search_rounds_completed + 1,
                search_phase="oracle_polish",
                allow_revisit=True,
                candidate_fit_options=polish_fit_options,
                ari_only_evaluation=True,
                start_mode="full",
                compute_summary=False,
            )
            oracle_search_rounds_completed += 1
    elif normalized_score == "oracle_ari":
        interim_df = pd.DataFrame([row for _, _, row in result_entries])
        interim_selection_df = _oracle_candidate_frame(interim_df)
        if interim_selection_df.empty:
            oracle_search_stop_reason = "no_finite_ari_candidates"
        else:
            explicit_round = 0
            zoom_performed = False
            while explicit_round < ORACLE_LIGHT_LOCAL_ROUNDS + 1:
                interim_df = pd.DataFrame([row for _, _, row in result_entries])
                interim_selection_df = _oracle_candidate_frame(interim_df)
                best_lambda_min, best_lambda_max, _, _, _ = _optimal_lambda_range(
                    interim_selection_df["ARI"].to_numpy(dtype=float),
                    interim_selection_df["lambda"].to_numpy(dtype=float),
                    maximize=True,
                )
                lower_hit, upper_hit = _lambda_boundary_flags(
                    interim_selection_df["lambda"].to_numpy(dtype=float),
                    best_lambda_min=best_lambda_min,
                    best_lambda_max=best_lambda_max,
                )
                boundary_unresolved = _lambda_boundary_unresolved(
                    evaluated_lambdas=interim_selection_df["lambda"].to_numpy(dtype=float),
                    lower_hit=lower_hit,
                    upper_hit=upper_hit,
                )
                if boundary_unresolved:
                    boundary_grid = _oracle_light_boundary_grid(
                        [float(value) for value in fit_by_lambda.keys()],
                        best_lambda_min=best_lambda_min,
                        best_lambda_max=best_lambda_max,
                        lower_hit=lower_hit,
                        upper_hit=upper_hit,
                    )
                    before = len(fit_by_lambda)
                    explicit_round += 1
                    _evaluate_lambda_sequence(
                        boundary_grid,
                        search_round=explicit_round,
                        search_phase="oracle_boundary_expand",
                        ari_only_evaluation=True,
                        start_mode=oracle_behavior.exploratory_start_mode,
                        compute_summary=False,
                    )
                    if len(fit_by_lambda) > before:
                        zoom_performed = True
                        oracle_search_rounds_completed = explicit_round
                        continue
                    oracle_search_stop_reason = "explicit_grid_boundary_unresolved"
                    break

                bracket_lower, bracket_upper = _oracle_local_bracket(
                    [float(value) for value in fit_by_lambda.keys()],
                    best_lambda_min=best_lambda_min,
                    best_lambda_max=best_lambda_max,
                )
                if _lambda_log10_span(lower=bracket_lower, upper=bracket_upper) <= ORACLE_LIGHT_LOG10_SPAN_TOL:
                    oracle_search_stop_reason = (
                        "explicit_grid_local_zoom_optimum" if zoom_performed else "explicit_grid_interior_optimum"
                    )
                    break

                zoom_grid = _oracle_local_zoom_grid(
                    [float(value) for value in fit_by_lambda.keys()],
                    best_lambda_min=best_lambda_min,
                    best_lambda_max=best_lambda_max,
                    points=ORACLE_LIGHT_LOCAL_POINTS,
                )
                before = len(fit_by_lambda)
                explicit_round += 1
                _evaluate_lambda_sequence(
                    zoom_grid,
                    search_round=explicit_round,
                    search_phase=f"oracle_local_zoom_{explicit_round}",
                    ari_only_evaluation=True,
                    start_mode=oracle_behavior.exploratory_start_mode,
                    compute_summary=False,
                )
                if len(fit_by_lambda) > before:
                    zoom_performed = True
                    oracle_search_rounds_completed = explicit_round
                    continue
                oracle_search_stop_reason = (
                    "explicit_grid_local_zoom_optimum" if zoom_performed else "explicit_grid_interior_optimum"
                )
                break
            else:
                oracle_search_stop_reason = "explicit_grid_local_zoom_limit"

        final_oracle_df = _oracle_candidate_frame(pd.DataFrame([row for _, _, row in result_entries]))
        polish_lambdas = _top_unique_oracle_lambdas(
            final_oracle_df,
            max_lambdas=ORACLE_LIGHT_POLISH_TOP_K,
        )
        if polish_lambdas:
            polish_fit_options = replace(
                effective_fit_options,
                outer_max_iter=max(int(effective_fit_options.outer_max_iter) * 2, int(effective_fit_options.outer_max_iter) + 2),
                inner_max_iter=max(int(effective_fit_options.inner_max_iter) * 2, int(effective_fit_options.inner_max_iter) + 16),
                tol=max(float(effective_fit_options.tol) * 0.5, 1e-6),
            )
            before_polish = len(result_entries)
            _evaluate_lambda_sequence(
                polish_lambdas,
                search_round=oracle_search_rounds_completed + 1,
                search_phase="oracle_light_polish",
                allow_revisit=True,
                candidate_fit_options=polish_fit_options,
                ari_only_evaluation=True,
                start_mode="full",
                compute_summary=False,
            )
            if len(result_entries) > before_polish:
                oracle_search_rounds_completed += 1
                if oracle_search_stop_reason in {
                    "explicit_grid_interior_optimum",
                    "explicit_grid_local_zoom_optimum",
                    "explicit_grid_local_zoom_limit",
                }:
                    oracle_search_stop_reason = "explicit_grid_local_polish_optimum"

    search_df = pd.DataFrame([row for _, _, row in result_entries]).sort_values(["lambda", "selection_step"]).reset_index(drop=True)
    num_candidates = int(search_df.shape[0])
    converged_mask = search_df["converged"].astype(bool).to_numpy(dtype=bool)
    num_converged_candidates = int(np.sum(converged_mask))
    converged_oracle_df = pd.DataFrame(columns=search_df.columns)
    if normalized_score == "oracle_ari":
        oracle_df = _oracle_candidate_frame(search_df)
        converged_oracle_df = _oracle_candidate_frame(search_df.loc[converged_mask].copy())
        if not converged_oracle_df.empty:
            selection_df = converged_oracle_df
            selection_used_convergence_fallback = False
        else:
            selection_df = oracle_df
            selection_used_convergence_fallback = bool(not oracle_df.empty)
    else:
        selection_df = search_df.loc[converged_mask].copy() if num_converged_candidates > 0 else search_df.copy()
        selection_used_convergence_fallback = bool(num_converged_candidates == 0 and num_candidates > 0)

    if selection_df.empty:
        raise RuntimeError(f"No candidate fits were evaluated for tumor {data.tumor_id}.")

    selection_lambda_values = selection_df["lambda"].to_numpy(dtype=float)

    if normalized_score == "oracle_ari":
        if simulation_root is None:
            raise ValueError("selection_score='oracle_ari' requires simulation_root.")
        if selection_df["ARI"].notna().sum() == 0:
            raise ValueError("selection_score='oracle_ari' requires candidate evaluations with ARI.")
        selection_min, selection_max, selection_count, selection_metric_value, selection_mask = _optimal_lambda_range(
            selection_df["ARI"].to_numpy(dtype=float),
            selection_lambda_values,
            maximize=True,
        )
        tied_df = selection_df.loc[selection_mask].copy()
        converged_tied_df = tied_df.loc[tied_df["converged"].astype(bool)].copy()
        if not converged_tied_df.empty:
            tied_df = converged_tied_df
        best_row = _representative_optimal_row(
            tied_df,
            lambda_min=selection_min,
            lambda_max=selection_max,
        )
    elif normalized_score == "summary_classic_bic":
        selection_min, selection_max, selection_count, selection_metric_value, selection_mask = _optimal_lambda_range(
            selection_df["classic_bic"].to_numpy(dtype=float),
            selection_lambda_values,
            maximize=False,
        )
        tied_df = selection_df.loc[selection_mask].sort_values(["classic_bic", "lambda", "selection_step"])
        best_row = tied_df.iloc[0]
    else:
        selection_min, selection_max, selection_count, selection_metric_value, selection_mask = _optimal_lambda_range(
            selection_df["bic"].to_numpy(dtype=float),
            selection_lambda_values,
            maximize=False,
        )
        tied_df = selection_df.loc[selection_mask].sort_values(["bic", "lambda", "selection_step"])
        best_row = tied_df.iloc[0]

    best_ari_min, best_ari_max, best_ari_count, best_ari_value, ari_mask = _optimal_lambda_range(
        selection_df["ARI"].to_numpy(dtype=float),
        selection_lambda_values,
        maximize=True,
    )
    best_converged_ari_min, best_converged_ari_max, best_converged_ari_count, best_converged_ari_value, _ = _optimal_lambda_range(
        converged_oracle_df["ARI"].to_numpy(dtype=float) if not converged_oracle_df.empty else np.asarray([], dtype=float),
        converged_oracle_df["lambda"].to_numpy(dtype=float) if not converged_oracle_df.empty else np.asarray([], dtype=float),
        maximize=True,
    )
    selection_lower_hit, selection_upper_hit = _lambda_boundary_flags(
        selection_lambda_values,
        best_lambda_min=selection_min,
        best_lambda_max=selection_max,
    )
    selection_boundary_unresolved = _lambda_boundary_unresolved(
        evaluated_lambdas=selection_lambda_values,
        lower_hit=selection_lower_hit,
        upper_hit=selection_upper_hit,
    )
    ari_lower_hit, ari_upper_hit = _lambda_boundary_flags(
        selection_lambda_values,
        best_lambda_min=best_ari_min,
        best_lambda_max=best_ari_max,
    )
    ari_boundary_unresolved = _lambda_boundary_unresolved(
        evaluated_lambdas=selection_lambda_values,
        lower_hit=ari_lower_hit,
        upper_hit=ari_upper_hit,
    )
    selection_optimal_ids = set(selection_df.loc[selection_mask, "_candidate_id"].astype(int).tolist())
    ari_optimal_ids = set(selection_df.loc[ari_mask, "_candidate_id"].astype(int).tolist())
    if normalized_score == "oracle_ari":
        if explicit_lambda_grid:
            if selection_boundary_unresolved or ari_boundary_unresolved:
                final_oracle_search_stop_reason = "explicit_grid_boundary_unresolved"
            elif oracle_search_stop_reason not in {"not_applicable", ""}:
                final_oracle_search_stop_reason = oracle_search_stop_reason
            else:
                final_oracle_search_stop_reason = "explicit_grid_interior_optimum"
        else:
            final_oracle_search_stop_reason = (
                "boundary_unresolved_after_oracle_search"
                if selection_boundary_unresolved or ari_boundary_unresolved
                else "ultra_dense_local_optimum"
                if oracle_ultra_dense_ran and not (selection_boundary_unresolved or ari_boundary_unresolved)
                else oracle_search_stop_reason
            )
    else:
        final_oracle_search_stop_reason = oracle_search_stop_reason
    if normalized_score == "oracle_ari":
        eligible_mask = np.isfinite(search_df["ARI"].to_numpy(dtype=float))
    else:
        eligible_mask = converged_mask if num_converged_candidates > 0 else np.ones(num_candidates, dtype=bool)
    search_df["eligible_for_selection"] = eligible_mask
    search_df["is_selection_optimal"] = search_df["_candidate_id"].astype(int).isin(selection_optimal_ids)
    search_df["is_ari_optimal"] = search_df["_candidate_id"].astype(int).isin(ari_optimal_ids)
    selected_candidate_id = int(best_row["_candidate_id"])
    search_df["is_selected_best_row"] = search_df["_candidate_id"].astype(int) == selected_candidate_id
    search_df["oracle_search_stop_reason"] = str(final_oracle_search_stop_reason)

    best_fit, best_evaluation, _ = result_entries[int(best_row["_candidate_id"])]
    if oracle_behavior.finalize_selected_fit:
        selected_fit = fit_single_stage_em(
            data=data,
            options=replace(effective_fit_options, lambda_value=float(best_row["lambda"])),
            phi_start=best_fit.phi.copy(),
            exact_pilot=pilot_phi,
            pooled_start=pooled_start,
            scalar_well_starts=scalar_well_starts,
            start_mode="full",
            runtime=runtime,
            torch_data=torch_data,
            compute_summary=True,
        )
        selected_fit.bic = None
        selected_fit.classic_bic = None
        selected_fit.extended_bic = None
        selected_fit.selection_score_name = normalized_score
        best_fit = selected_fit
        if simulation_truth is not None:
            best_evaluation = evaluate_fit_against_simulation(
                fit=best_fit,
                data=data,
                simulation_truth=simulation_truth,
            )
    search_df = search_df.drop(columns=["_candidate_id"])
    effective_graph.clear_torch_cache()
    return ModelSelectionResult(
        best_fit=best_fit,
        best_evaluation=best_evaluation,
        search_df=search_df,
        bic_df_scale=float(bic_df_scale),
        bic_cluster_penalty=float(bic_cluster_penalty),
        selection_method=selection_method,
        profile_name=profile_name,
        selection_metric_value=selection_metric_value,
        selection_lambda_min=selection_min,
        selection_lambda_max=selection_max,
        selection_lambda_count=selection_count,
        best_ari=best_ari_value,
        ari_optimal_lambda_min=best_ari_min,
        ari_optimal_lambda_max=best_ari_max,
        ari_optimal_lambda_count=best_ari_count,
        best_converged_ari=best_converged_ari_value,
        best_converged_lambda_min=best_converged_ari_min,
        best_converged_lambda_max=best_converged_ari_max,
        best_converged_lambda_count=best_converged_ari_count,
        selection_hits_lower_boundary=selection_lower_hit,
        selection_hits_upper_boundary=selection_upper_hit,
        selection_boundary_unresolved=selection_boundary_unresolved,
        selection_optimum_resolved=not selection_boundary_unresolved,
        ari_hits_lower_boundary=ari_lower_hit,
        ari_hits_upper_boundary=ari_upper_hit,
        ari_boundary_unresolved=ari_boundary_unresolved,
        ari_optimum_resolved=not ari_boundary_unresolved,
        oracle_search_rounds_completed=oracle_search_rounds_completed,
        oracle_search_stop_reason=str(final_oracle_search_stop_reason),
        num_candidates=num_candidates,
        num_converged_candidates=num_converged_candidates,
        selection_used_convergence_fallback=selection_used_convergence_fallback,
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
    settings_profile: str,
    selection_score: str,
    use_warm_starts: bool,
    evaluate_all_candidates: bool,
    finalize_selected_fit: bool = True,
) -> ModelSelectionResult:
    normalized_profile = settings_profile.strip().lower()
    if normalized_profile not in {"manual", "auto"}:
        raise ValueError(f"Unknown settings_profile: {settings_profile}")
    normalized_score = _normalize_selection_score_name(selection_score)

    effective_lambda_grid_mode = str(lambda_grid_mode)
    effective_bic_df_scale = float(bic_df_scale)
    effective_bic_cluster_penalty = float(bic_cluster_penalty)
    profile_name = "manual"

    if normalized_profile == "auto":
        recommended = recommend_settings_from_data(data, selection_score=selection_score)
        effective_lambda_grid_mode = str(recommended.lambda_grid_mode)
        effective_bic_df_scale = float(recommended.bic_df_scale)
        effective_bic_cluster_penalty = float(recommended.bic_cluster_penalty)
        profile_name = str(recommended.profile_name)

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
        selection_method="lambda_path_oracle_ari" if normalized_score == "oracle_ari" else "lambda_path_grid",
        selection_score=selection_score,
        finalize_selected_fit=finalize_selected_fit,
    )


__all__ = [
    "ModelSelectionResult",
    "ORACLE_EXPANSION_FACTOR",
    "ORACLE_MAX_LAMBDA",
    "ORACLE_MAX_SEARCH_ROUNDS",
    "ORACLE_MIN_LAMBDA",
    "ORACLE_REFINE_POINTS",
    "ORACLE_REFINE_ROUNDS",
    "select_model",
]
