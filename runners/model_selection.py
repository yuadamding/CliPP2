from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import adjusted_rand_score

from ..core.graph import GraphData, build_knn_graph
from ..core.model import FitOptions, FitResult, RawFitResult, finalize_raw_fit, fit_single_stage_em_raw, make_fit_context
from ..io.data import PatientData
from ..metrics.evaluation import SimulationEvaluation, evaluate_fit_against_simulation
from .selection import compute_classic_bic, compute_extended_bic, default_lambda_grid
from .settings import recommend_settings_from_data


@dataclass
class ModelSelectionResult:
    best_fit: FitResult
    best_evaluation: SimulationEvaluation | None
    search_df: pd.DataFrame
    graph_k: int
    graph_edges: int
    bic_df_scale: float
    bic_cluster_penalty: float
    center_merge_tol: float
    selection_method: str
    profile_name: str


def _selection_score_value(
    *,
    loglik: float,
    num_clusters: int,
    data: PatientData,
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
    if selection_score in {"ebic", "refit_ebic"}:
        return float(extended_bic), float(classic_bic), float(extended_bic)
    if selection_score in {"classic_bic", "classic_refit_bic"}:
        return float(classic_bic), float(classic_bic), float(extended_bic)
    raise ValueError(f"Unknown selection_score: {selection_score}")


def _evaluate_candidate(
    *,
    data: PatientData,
    graph: GraphData,
    graph_k: int,
    fit_options: FitOptions,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    simulation_root: Path | None,
    evaluate_candidate: bool,
    phi_start: np.ndarray | None,
    selection_method: str,
    profile_name: str,
    selection_step: int,
    lambda_value: float,
    center_merge_tol: float,
    center_candidates: list[float] | None = None,
    raw_fit_cache: dict[tuple[float, int], RawFitResult] | None = None,
    context=None,
    acquisition_value: float | None = None,
    selection_score: str = "ebic",
) -> tuple[FitResult, SimulationEvaluation | None, dict[str, float | int | str | bool]]:
    centers = [float(center_merge_tol)] if center_candidates is None else [float(value) for value in center_candidates]
    centers = sorted(set(round(value, 5) for value in centers))

    raw_fit = None
    raw_key = _raw_candidate_key(lambda_value=lambda_value, graph_k=graph_k)
    if raw_fit_cache is not None:
        raw_fit = raw_fit_cache.get(raw_key)
    if raw_fit is None:
        raw_options = replace(
            fit_options,
            lambda_value=float(lambda_value),
            center_merge_tol=float(centers[0]),
        )
        raw_fit = fit_single_stage_em_raw(data=data, graph=graph, options=raw_options, phi_start=phi_start)
        if raw_fit_cache is not None:
            raw_fit_cache[raw_key] = raw_fit

    best_fit: FitResult | None = None
    best_selection_score = np.inf
    best_classic_bic = np.inf
    best_extended_bic = np.inf
    best_center_merge_tol = float(centers[0])
    use_refit_centers = bool(
        fit_options.refit_cluster_centers or selection_score in {"classic_refit_bic", "refit_ebic"}
    )
    for candidate_center in centers:
        options = replace(
            fit_options,
            lambda_value=float(lambda_value),
            center_merge_tol=float(candidate_center),
            refit_cluster_centers=use_refit_centers,
        )
        fit = finalize_raw_fit(data=data, graph=graph, raw_fit=raw_fit, options=options, context=context)
        selection_value, classic_bic, extended_bic = _selection_score_value(
            loglik=fit.loglik,
            num_clusters=fit.n_clusters,
            data=data,
            bic_df_scale=bic_df_scale,
            bic_cluster_penalty=bic_cluster_penalty,
            selection_score=selection_score,
        )
        fit.bic = float(selection_value)
        fit.classic_bic = float(classic_bic)
        fit.extended_bic = float(extended_bic)
        fit.selection_score_name = selection_score
        if best_fit is None or float(selection_value) < float(best_selection_score):
            best_fit = fit
            best_selection_score = float(selection_value)
            best_classic_bic = float(classic_bic)
            best_extended_bic = float(extended_bic)
            best_center_merge_tol = float(candidate_center)

    assert best_fit is not None

    evaluation = None
    if evaluate_candidate and simulation_root is not None and (simulation_root / data.patient_id).exists():
        evaluation = evaluate_fit_against_simulation(fit=best_fit, data=data, simulation_root=simulation_root)

    largest_cluster_fraction, small_cluster_count, small_cluster_mass, raw_spread = _fit_cluster_metrics(best_fit, data)
    within_fraction, cluster_sep_ratio, cluster_min_sep_ratio, cluster_quality_score = _candidate_cluster_structure_metrics(
        best_fit,
        data,
    )

    row: dict[str, float | int | str | bool] = {
        "patient_id": data.patient_id,
        "selection_method": selection_method,
        "selection_profile": profile_name,
        "selection_step": int(selection_step),
        "lambda": float(best_fit.lambda_value),
        "graph_k": int(graph_k),
        "center_merge_tol": float(best_center_merge_tol),
        "bic_df_scale": float(bic_df_scale),
        "bic_cluster_penalty": float(bic_cluster_penalty),
        "bic": float(best_fit.bic),
        "selection_score_name": selection_score,
        "classic_bic": float(best_classic_bic),
        "extended_bic": float(best_extended_bic),
        "loglik": float(best_fit.loglik),
        "penalized_objective": float(best_fit.penalized_objective),
        "n_clusters": int(best_fit.n_clusters),
        "converged": bool(best_fit.converged),
        "iterations": int(best_fit.iterations),
        "device": str(best_fit.device),
        "refit_cluster_centers": bool(use_refit_centers),
        "largest_cluster_fraction": largest_cluster_fraction,
        "small_cluster_count": int(small_cluster_count),
        "small_cluster_mass": float(small_cluster_mass),
        "raw_phi_spread": float(raw_spread),
        "cluster_within_fraction": float(within_fraction),
        "cluster_sep_ratio": float(cluster_sep_ratio),
        "cluster_min_sep_ratio": float(cluster_min_sep_ratio),
        "cluster_quality_score": float(cluster_quality_score),
        "acquisition_value": np.nan if acquisition_value is None else float(acquisition_value),
        "ARI": np.nan if evaluation is None else float(evaluation.ari),
        "cp_rmse": np.nan if evaluation is None else float(evaluation.cp_rmse),
        "multiplicity_f1": np.nan if evaluation is None else float(evaluation.multiplicity_f1),
        "n_eval_mutations": np.nan if evaluation is None else int(evaluation.n_eval_mutations),
        "n_filtered_mutations": np.nan if evaluation is None else int(evaluation.n_filtered_mutations),
    }
    return best_fit, evaluation, row


def _graph_cache_get(
    graph_cache: dict[int, GraphData],
    *,
    data: PatientData,
    graph_k: int,
    device: str,
) -> GraphData:
    graph_k = int(graph_k)
    graph = graph_cache.get(graph_k)
    if graph is None:
        graph = build_knn_graph(data.phi_init, k=graph_k, device=device)
        graph_cache[graph_k] = graph
    return graph


def _context_cache_get(
    context_cache: dict[int, object],
    *,
    data: PatientData,
    graph_k: int,
    graph: GraphData,
    device: str,
):
    context = context_cache.get(int(graph_k))
    if context is None:
        context = make_fit_context(data=data, graph=graph, device=device)
        context_cache[int(graph_k)] = context
    return context


def _grid_search_selection(
    *,
    data: PatientData,
    simulation_root: Path | None,
    lambda_grid: list[float] | None,
    lambda_grid_mode: str,
    graph_k: int,
    fit_options: FitOptions,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    use_warm_starts: bool,
    evaluate_all_candidates: bool,
    profile_name: str,
    selection_method: str,
    selection_score: str,
) -> ModelSelectionResult:
    if lambda_grid is None:
        lambda_grid = default_lambda_grid(data, mode=lambda_grid_mode)
    lambda_grid = [float(value) for value in np.unique(np.sort(np.asarray(lambda_grid, dtype=float)))]

    graph_cache: dict[int, GraphData] = {}
    context_cache: dict[int, object] = {}
    graph = _graph_cache_get(graph_cache, data=data, graph_k=graph_k, device=fit_options.device)
    context = _context_cache_get(context_cache, data=data, graph_k=graph_k, graph=graph, device=fit_options.device)
    phi_start = data.phi_init.copy()
    candidate_fits: list[tuple[FitResult, SimulationEvaluation | None]] = []
    search_rows: list[dict[str, float | int | str | bool]] = []

    for step, lambda_value in enumerate(lambda_grid):
        fit, evaluation, row = _evaluate_candidate(
            data=data,
            graph=graph,
            graph_k=graph_k,
            fit_options=fit_options,
            bic_df_scale=bic_df_scale,
            bic_cluster_penalty=bic_cluster_penalty,
            simulation_root=simulation_root,
            evaluate_candidate=evaluate_all_candidates,
            phi_start=phi_start,
            selection_method=selection_method,
            profile_name=profile_name,
            selection_step=step,
            lambda_value=lambda_value,
            center_merge_tol=fit_options.center_merge_tol,
            context=context,
            selection_score=selection_score,
        )
        candidate_fits.append((fit, evaluation))
        search_rows.append(row)
        if use_warm_starts:
            phi_start = fit.phi.copy()

    best_index = int(np.argmin([fit.bic for fit, _ in candidate_fits]))
    best_fit, best_evaluation = candidate_fits[best_index]
    search_df = pd.DataFrame(search_rows).sort_values(["lambda", "selection_step"]).reset_index(drop=True)
    return ModelSelectionResult(
        best_fit=best_fit,
        best_evaluation=best_evaluation,
        search_df=search_df,
        graph_k=int(graph_k),
        graph_edges=int(graph.num_edges),
        bic_df_scale=float(bic_df_scale),
        bic_cluster_penalty=float(bic_cluster_penalty),
        center_merge_tol=float(fit_options.center_merge_tol),
        selection_method=selection_method,
        profile_name=profile_name,
    )


def _bo_bounds(
    data: PatientData,
    *,
    graph_k: int,
    lambda_grid: list[float] | None,
    lambda_grid_mode: str,
    center_merge_tol: float,
) -> tuple[list[int], tuple[float, float], tuple[float, float], list[float]]:
    max_graph_k = max(2, min(12, data.num_mutations - 1))
    min_graph_k = min(max_graph_k, 4)
    graph_values = list(range(min_graph_k, max_graph_k + 1))
    if graph_k not in graph_values:
        graph_values.append(int(np.clip(graph_k, min_graph_k, max_graph_k)))
        graph_values = sorted(set(graph_values))

    if lambda_grid is None:
        lambda_seed = default_lambda_grid(data, mode=lambda_grid_mode)
    else:
        lambda_seed = [float(value) for value in lambda_grid]
    lambda_seed = [value for value in lambda_seed if value > 0]
    if not lambda_seed:
        lambda_seed = default_lambda_grid(data, mode="dense_no_zero")

    lambda_low = max(min(lambda_seed) * 0.2, 1e-6)
    lambda_high = max(lambda_seed) * 1.5
    if lambda_high <= lambda_low:
        lambda_high = lambda_low * 2.0

    center_low = max(0.02, center_merge_tol * 0.5)
    center_high = min(0.20, max(center_low + 1e-3, center_merge_tol * 1.5))
    return graph_values, (float(lambda_low), float(lambda_high)), (float(center_low), float(center_high)), list(lambda_seed)


def _encode_candidate(
    lambda_value: float,
    graph_k: int,
    center_merge_tol: float,
    *,
    graph_values: list[int],
    lambda_bounds: tuple[float, float],
    center_bounds: tuple[float, float],
) -> np.ndarray:
    log_low = np.log(lambda_bounds[0])
    log_high = np.log(lambda_bounds[1])
    lambda_coord = 0.0 if log_high <= log_low else (np.log(lambda_value) - log_low) / (log_high - log_low)
    graph_coord = 0.0 if len(graph_values) <= 1 else (graph_k - graph_values[0]) / max(graph_values[-1] - graph_values[0], 1)
    center_coord = 0.0 if center_bounds[1] <= center_bounds[0] else (center_merge_tol - center_bounds[0]) / (center_bounds[1] - center_bounds[0])
    encoded = np.array([lambda_coord, graph_coord, center_coord], dtype=float)
    return np.clip(encoded, 0.0, 1.0)


def _decode_candidate(
    x: np.ndarray,
    *,
    graph_values: list[int],
    lambda_bounds: tuple[float, float],
    center_bounds: tuple[float, float],
) -> tuple[float, int, float]:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    log_low = np.log(lambda_bounds[0])
    log_high = np.log(lambda_bounds[1])
    lambda_value = float(np.exp(log_low + x[0] * (log_high - log_low)))
    if len(graph_values) <= 1:
        graph_k = int(graph_values[0])
    else:
        graph_float = graph_values[0] + x[1] * (graph_values[-1] - graph_values[0])
        graph_k = int(np.clip(np.round(graph_float), graph_values[0], graph_values[-1]))
    center_merge_tol = float(center_bounds[0] + x[2] * (center_bounds[1] - center_bounds[0]))
    return lambda_value, graph_k, center_merge_tol


def _candidate_key(lambda_value: float, graph_k: int, center_merge_tol: float) -> tuple[float, int, float]:
    return (round(float(lambda_value), 6), int(graph_k), round(float(center_merge_tol), 5))


def _raw_candidate_key(lambda_value: float, graph_k: int) -> tuple[float, int]:
    return (round(float(lambda_value), 6), int(graph_k))


def _small_cluster_threshold(data: PatientData) -> int:
    adaptive = int(np.round(0.01 * float(data.num_mutations) / max(np.sqrt(float(max(data.num_samples, 1))), 1.0)))
    return int(np.clip(max(adaptive, 2), 2, 25))


def _fit_cluster_metrics(
    fit: FitResult,
    data: PatientData,
) -> tuple[float, int, float, float]:
    cluster_sizes = np.bincount(fit.cluster_labels, minlength=max(fit.n_clusters, 1)).astype(np.int64)
    small_threshold = _small_cluster_threshold(data)
    small_cluster_count = int(np.sum(cluster_sizes <= small_threshold))
    small_cluster_mass = float(cluster_sizes[cluster_sizes <= small_threshold].sum() / max(data.num_mutations, 1))
    largest_cluster_fraction = float(cluster_sizes.max() / max(data.num_mutations, 1))
    raw_center = fit.phi.mean(axis=0, keepdims=True)
    raw_spread = float(np.sqrt(np.mean((fit.phi - raw_center) ** 2)))
    return largest_cluster_fraction, small_cluster_count, small_cluster_mass, raw_spread


def _candidate_cluster_structure_metrics(
    fit: FitResult,
    data: PatientData,
) -> tuple[float, float, float, float]:
    phi = np.asarray(fit.phi, dtype=np.float32)
    labels = np.asarray(fit.cluster_labels, dtype=np.int64)
    centers = np.asarray(fit.cluster_centers, dtype=np.float32)
    if phi.size == 0 or fit.n_clusters <= 0:
        return 1.0, 0.0, 0.0, 0.0

    overall_center = phi.mean(axis=0, keepdims=True)
    residual = phi - centers[labels]
    within_ss = float(np.sum(residual * residual))
    total_residual = phi - overall_center
    total_ss = float(np.sum(total_residual * total_residual))
    within_fraction = float(within_ss / max(total_ss, 1e-8)) if total_ss > 0.0 else 0.0

    cluster_sizes = np.bincount(labels, minlength=max(fit.n_clusters, 1)).astype(np.float64)
    between_ss = float(np.sum(cluster_sizes[:, None] * (centers - overall_center) ** 2))
    if fit.n_clusters > 1 and phi.shape[0] > fit.n_clusters:
        numerator = between_ss / max(fit.n_clusters - 1, 1)
        denominator = within_ss / max(phi.shape[0] - fit.n_clusters, 1)
        calinski_harabasz = float(numerator / max(denominator, 1e-8))
    else:
        calinski_harabasz = 0.0

    if fit.n_clusters <= 1:
        sep_ratio = 0.0
        min_sep_ratio = 0.0
    else:
        center_diff = centers[:, None, :] - centers[None, :, :]
        center_dist = np.sqrt(np.sum(center_diff * center_diff, axis=2))
        center_dist += np.eye(fit.n_clusters, dtype=np.float32) * 1e9
        nearest_center = center_dist.min(axis=1)

        cluster_scatter = np.zeros(fit.n_clusters, dtype=np.float64)
        for label in range(fit.n_clusters):
            members = labels == label
            if np.any(members):
                cluster_scatter[label] = float(np.sqrt(np.mean((phi[members] - centers[label]) ** 2)))
        safe_scatter = np.maximum(cluster_scatter, 1e-4)
        sep_ratio_values = np.clip(nearest_center / safe_scatter, 0.0, 50.0)
        sep_ratio = float(np.average(sep_ratio_values, weights=np.maximum(cluster_sizes, 1.0)))
        min_sep_ratio = float(np.min(sep_ratio_values))

    quality_score = float(
        np.log1p(max(calinski_harabasz, 0.0))
        + 0.35 * np.log1p(max(sep_ratio, 0.0))
        + 0.15 * np.log1p(max(min_sep_ratio, 0.0))
        - within_fraction
    )
    return within_fraction, sep_ratio, min_sep_ratio, quality_score


def _fit_has_tiny_clusters(fit: FitResult, data: PatientData) -> bool:
    if fit.n_clusters <= 1:
        return False
    _, small_cluster_count, _, _ = _fit_cluster_metrics(fit, data)
    return bool(small_cluster_count > 0)


def _fit_is_collapsed(fit: FitResult, data: PatientData) -> bool:
    largest_cluster_fraction, _, _, raw_spread = _fit_cluster_metrics(fit, data)
    return bool(
        largest_cluster_fraction >= 0.95
        and raw_spread >= 0.08
        and data.num_samples >= 3
    )


def _needs_local_refinement(fit: FitResult, data: PatientData) -> bool:
    return bool(
        data.num_samples <= 2
        or fit.n_clusters <= 1
        or _fit_has_tiny_clusters(fit, data)
        or _fit_is_collapsed(fit, data)
    )


def _candidate_pathology_penalty(
    fit: FitResult,
    data: PatientData,
) -> float:
    largest_cluster_fraction, small_cluster_count, small_cluster_mass, raw_spread = _fit_cluster_metrics(fit, data)
    penalty = 0.0
    if fit.n_clusters <= 1 and raw_spread >= 0.05:
        penalty += 1.50
    if _fit_is_collapsed(fit, data):
        penalty += 1.25
    if largest_cluster_fraction >= 0.90 and fit.n_clusters <= 2 and raw_spread >= 0.04:
        penalty += 0.75
    if small_cluster_count > 0:
        penalty += min(1.0, 0.20 * float(small_cluster_count) + 1.50 * float(small_cluster_mass))
    return float(penalty)


def _parameter_distance(
    row_a: dict[str, float | int | str | bool],
    row_b: dict[str, float | int | str | bool],
    *,
    lambda_bounds: tuple[float, float],
    center_bounds: tuple[float, float],
    graph_values: list[int],
) -> float:
    graph_span = max(int(graph_values[-1]) - int(graph_values[0]), 1)
    center_span = max(float(center_bounds[1]) - float(center_bounds[0]), 1e-6)
    lambda_span = max(float(np.log(lambda_bounds[1]) - np.log(lambda_bounds[0])), 1e-6)

    lambda_delta = abs(np.log(float(row_a["lambda"])) - np.log(float(row_b["lambda"]))) / lambda_span
    graph_delta = abs(int(row_a["graph_k"]) - int(row_b["graph_k"])) / graph_span
    center_delta = abs(float(row_a["center_merge_tol"]) - float(row_b["center_merge_tol"])) / center_span
    return float(np.sqrt(lambda_delta**2 + graph_delta**2 + center_delta**2))


def _neighborhood_indices(
    base_idx: int,
    search_rows: list[dict[str, float | int | str | bool]],
    *,
    lambda_bounds: tuple[float, float],
    center_bounds: tuple[float, float],
    graph_values: list[int],
) -> list[int]:
    base_row = search_rows[base_idx]
    indices: list[int] = []
    for idx, row in enumerate(search_rows):
        if abs(np.log(float(row["lambda"])) - np.log(float(base_row["lambda"]))) > np.log(1.8):
            continue
        if abs(int(row["graph_k"]) - int(base_row["graph_k"])) > 2:
            continue
        if abs(float(row["center_merge_tol"]) - float(base_row["center_merge_tol"])) > 0.03:
            continue
        _ = _parameter_distance(
            base_row,
            row,
            lambda_bounds=lambda_bounds,
            center_bounds=center_bounds,
            graph_values=graph_values,
        )
        indices.append(idx)
    return indices


def _stability_graph_grid(
    base_graph_k: int,
    *,
    graph_values: list[int],
    low_sample: bool,
    unstable: bool,
) -> list[int]:
    values = {
        int(np.clip(base_graph_k + delta, graph_values[0], graph_values[-1]))
        for delta in ((-2, -1, 0, 1, 2) if low_sample else (-1, 0, 1))
    }
    if low_sample:
        values.update(value for value in (4, 6, 8) if graph_values[0] <= value <= graph_values[-1])
    elif unstable:
        values.update(value for value in (6, 8, 10) if graph_values[0] <= value <= graph_values[-1])
    return sorted(values)


def _stability_shortlist_indices(
    candidate_fits: list[tuple[FitResult, SimulationEvaluation | None]],
    *,
    max_candidates: int,
) -> list[int]:
    if not candidate_fits:
        return []
    order = np.argsort([fit.bic for fit, _ in candidate_fits])
    return [int(idx) for idx in order[: min(len(order), max(int(max_candidates), 1))]]


def _selector_shortlist_indices(
    candidate_fits: list[tuple[FitResult, SimulationEvaluation | None]],
    search_rows: list[dict[str, float | int | str | bool]],
    *,
    max_bic_candidates: int,
    max_quality_candidates: int,
) -> list[int]:
    if not candidate_fits:
        return []

    indices = set(_stability_shortlist_indices(candidate_fits, max_candidates=max_bic_candidates))
    if search_rows and max_quality_candidates > 0:
        quality = np.asarray(
            [float(row.get("cluster_quality_score", np.nan)) for row in search_rows],
            dtype=float,
        )
        finite_idx = np.where(np.isfinite(quality))[0]
        if finite_idx.size > 0:
            quality_order = finite_idx[np.argsort(-quality[finite_idx])]
            indices.update(int(idx) for idx in quality_order[: min(len(quality_order), max_quality_candidates)])
    return sorted(indices, key=lambda idx: float(candidate_fits[int(idx)][0].bic))


def _candidate_stability_metrics(
    base_idx: int,
    neighborhood: list[int],
    candidate_fits: list[tuple[FitResult, SimulationEvaluation | None]],
    search_rows: list[dict[str, float | int | str | bool]],
    *,
    data: PatientData,
    lambda_bounds: tuple[float, float],
    center_bounds: tuple[float, float],
    graph_values: list[int],
) -> dict[str, float]:
    base_fit, _ = candidate_fits[base_idx]
    if len(neighborhood) <= 1:
        return {
            "stability_ari": 0.0,
            "cluster_variation": 1.0,
            "local_bic_median": float(base_fit.bic),
            "pathology_penalty": _candidate_pathology_penalty(base_fit, data),
            "neighbor_count": 0.0,
        }

    weights: list[float] = []
    aris: list[float] = []
    cluster_gaps: list[float] = []
    local_bics: list[float] = []
    for idx in neighborhood:
        fit, _ = candidate_fits[idx]
        local_bics.append(float(fit.bic))
        if idx == base_idx:
            continue
        distance = _parameter_distance(
            search_rows[base_idx],
            search_rows[idx],
            lambda_bounds=lambda_bounds,
            center_bounds=center_bounds,
            graph_values=graph_values,
        )
        weight = float(np.exp(-2.0 * distance))
        weights.append(weight)
        aris.append(float(adjusted_rand_score(base_fit.cluster_labels, fit.cluster_labels)))
        cluster_gaps.append(abs(int(fit.n_clusters) - int(base_fit.n_clusters)) / max(int(base_fit.n_clusters), 1))

    stability_ari = float(np.average(aris, weights=weights)) if weights else 0.0
    cluster_variation = float(np.average(cluster_gaps, weights=weights)) if weights else 1.0
    return {
        "stability_ari": stability_ari,
        "cluster_variation": cluster_variation,
        "local_bic_median": float(np.median(local_bics)),
        "pathology_penalty": _candidate_pathology_penalty(base_fit, data),
        "neighbor_count": float(len(neighborhood) - 1),
    }


def _lambda_refine_grid(
    lambda_value: float,
    *,
    lambda_bounds: tuple[float, float],
    mode: str,
) -> list[float]:
    if mode == "low_samples":
        ratios = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0]
    elif mode == "collapsed":
        ratios = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
    elif mode == "broad":
        ratios = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    else:
        ratios = [0.5, 0.75, 1.0, 1.25, 1.5]
    candidates = []
    for ratio in ratios:
        value = float(np.clip(lambda_value * ratio, lambda_bounds[0], lambda_bounds[1]))
        candidates.append(round(value, 6))
    return sorted(set(candidates))


def _center_search_grid(
    center_merge_tol: float,
    *,
    center_bounds: tuple[float, float],
    include_stable_anchor: bool = False,
) -> list[float]:
    deltas = (-0.02, -0.01, 0.0, 0.01, 0.02)
    candidates = []
    for delta in deltas:
        value = float(np.clip(center_merge_tol + delta, center_bounds[0], center_bounds[1]))
        candidates.append(round(value, 5))
    if include_stable_anchor:
        for anchor in (0.06, 0.08, 0.10, 0.12):
            value = float(np.clip(anchor, center_bounds[0], center_bounds[1]))
            candidates.append(round(value, 5))
    return sorted(set(candidates))


def _sample_random_candidate(
    rng: np.random.Generator,
    *,
    graph_values: list[int],
    lambda_bounds: tuple[float, float],
    center_bounds: tuple[float, float],
    existing_keys: set[tuple[float, int, float]],
) -> tuple[float, int, float]:
    for _ in range(256):
        x = rng.random(3)
        candidate = _decode_candidate(
            x,
            graph_values=graph_values,
            lambda_bounds=lambda_bounds,
            center_bounds=center_bounds,
        )
        if _candidate_key(*candidate) not in existing_keys:
            return candidate
    return _decode_candidate(
        rng.random(3),
        graph_values=graph_values,
        lambda_bounds=lambda_bounds,
        center_bounds=center_bounds,
    )


def _expected_improvement_min(
    mu: np.ndarray,
    sigma: np.ndarray,
    incumbent: float,
) -> np.ndarray:
    safe_sigma = np.maximum(sigma, 1e-9)
    improvement = incumbent - mu
    z = improvement / safe_sigma
    ei = improvement * norm.cdf(z) + safe_sigma * norm.pdf(z)
    ei[sigma <= 1e-9] = 0.0
    return ei


def _bayesian_selection(
    *,
    data: PatientData,
    simulation_root: Path | None,
    lambda_grid: list[float] | None,
    lambda_grid_mode: str,
    graph_k: int,
    fit_options: FitOptions,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    evaluate_all_candidates: bool,
    bo_max_evals: int,
    bo_init_points: int,
    bo_random_seed: int,
    selection_score: str,
) -> ModelSelectionResult:
    selection_method = "bayes"
    profile_name = "bayes_opt"
    graph_values, lambda_bounds, center_bounds, lambda_seed = _bo_bounds(
        data,
        graph_k=graph_k,
        lambda_grid=lambda_grid,
        lambda_grid_mode=lambda_grid_mode,
        center_merge_tol=fit_options.center_merge_tol,
    )
    rng = np.random.default_rng(int(bo_random_seed))
    graph_cache: dict[int, GraphData] = {}
    context_cache: dict[int, object] = {}
    raw_fit_cache: dict[tuple[float, int], RawFitResult] = {}
    search_rows: list[dict[str, float | int | str | bool]] = []
    candidate_fits: list[tuple[FitResult, SimulationEvaluation | None]] = []
    candidate_index_by_key: dict[tuple[float, int, float], int] = {}
    observed_x: list[np.ndarray] = []
    observed_y: list[float] = []
    existing_keys: set[tuple[float, int, float]] = set()

    center_mid = float(np.clip(fit_options.center_merge_tol, center_bounds[0], center_bounds[1]))
    lambda_mid = float(np.sqrt(lambda_bounds[0] * lambda_bounds[1]))
    initial_candidates: list[tuple[float, int, float]] = [
        (lambda_mid, int(np.clip(graph_k, graph_values[0], graph_values[-1])), center_mid),
        (float(lambda_bounds[0]), int(np.clip(graph_k, graph_values[0], graph_values[-1])), center_mid),
        (float(lambda_seed[0]), int(np.clip(graph_k, graph_values[0], graph_values[-1])), center_mid),
        (float(lambda_seed[len(lambda_seed) // 2]), int(np.clip(graph_k, graph_values[0], graph_values[-1])), center_mid),
        (float(lambda_seed[-1]), int(np.clip(graph_k, graph_values[0], graph_values[-1])), center_mid),
        (float(lambda_bounds[0]), 8 if 8 in graph_values else int(np.clip(graph_k, graph_values[0], graph_values[-1])), 0.08 if center_bounds[0] <= 0.08 <= center_bounds[1] else center_mid),
        (lambda_mid, int(graph_values[0]), center_bounds[0]),
        (lambda_mid, int(graph_values[-1]), center_bounds[1]),
    ]

    phi_start_cache: dict[int, np.ndarray] = {}
    total_evals = max(int(bo_max_evals), 1)
    init_evals = min(total_evals, max(int(bo_init_points), 1))

    for step in range(total_evals):
        if step < min(init_evals, len(initial_candidates)):
            lambda_value, candidate_graph_k, center_merge_tol = initial_candidates[step]
            if _candidate_key(lambda_value, candidate_graph_k, center_merge_tol) in existing_keys:
                lambda_value, candidate_graph_k, center_merge_tol = _sample_random_candidate(
                    rng,
                    graph_values=graph_values,
                    lambda_bounds=lambda_bounds,
                    center_bounds=center_bounds,
                    existing_keys=existing_keys,
                )
        elif step < init_evals:
            lambda_value, candidate_graph_k, center_merge_tol = _sample_random_candidate(
                rng,
                graph_values=graph_values,
                lambda_bounds=lambda_bounds,
                center_bounds=center_bounds,
                existing_keys=existing_keys,
            )
        else:
            x_obs = np.vstack(observed_x)
            y_obs = np.asarray(observed_y, dtype=float)
            y_mean = float(np.mean(y_obs))
            y_std = float(np.std(y_obs))
            y_train = (y_obs - y_mean) / max(y_std, 1e-6)
            kernel = ConstantKernel(1.0, (0.1, 10.0)) * Matern(length_scale=np.ones(3), nu=2.5) + WhiteKernel(noise_level=1e-5)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=False,
                random_state=int(bo_random_seed),
                n_restarts_optimizer=1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(x_obs, y_train)

            sampler = qmc.LatinHypercube(d=3, seed=int(rng.integers(0, 2**31 - 1)))
            candidate_x = sampler.random(n=256)
            mu_train, sigma_train = gp.predict(candidate_x, return_std=True)
            mu = mu_train * max(y_std, 1e-6) + y_mean
            sigma = sigma_train * max(y_std, 1e-6)
            acquisition = _expected_improvement_min(mu, sigma, incumbent=float(np.min(y_obs)))

            chosen = None
            acquisition_value = None
            for idx in np.argsort(-acquisition):
                candidate = _decode_candidate(
                    candidate_x[int(idx)],
                    graph_values=graph_values,
                    lambda_bounds=lambda_bounds,
                    center_bounds=center_bounds,
                )
                if _candidate_key(*candidate) not in existing_keys:
                    chosen = candidate
                    acquisition_value = float(acquisition[int(idx)])
                    break
            if chosen is None:
                chosen = _sample_random_candidate(
                    rng,
                    graph_values=graph_values,
                    lambda_bounds=lambda_bounds,
                    center_bounds=center_bounds,
                    existing_keys=existing_keys,
                )
                acquisition_value = np.nan
            lambda_value, candidate_graph_k, center_merge_tol = chosen

        acquisition_value = None if step < init_evals else acquisition_value
        existing_keys.add(_candidate_key(lambda_value, candidate_graph_k, center_merge_tol))
        graph = _graph_cache_get(graph_cache, data=data, graph_k=candidate_graph_k, device=fit_options.device)
        context = _context_cache_get(
            context_cache,
            data=data,
            graph_k=candidate_graph_k,
            graph=graph,
            device=fit_options.device,
        )
        phi_start = phi_start_cache.get(candidate_graph_k)
        fit, evaluation, row = _evaluate_candidate(
            data=data,
            graph=graph,
            graph_k=candidate_graph_k,
            fit_options=fit_options,
            bic_df_scale=bic_df_scale,
            bic_cluster_penalty=bic_cluster_penalty,
            simulation_root=simulation_root,
            evaluate_candidate=evaluate_all_candidates,
            phi_start=phi_start,
            selection_method=selection_method,
            profile_name=profile_name,
            selection_step=step,
            lambda_value=lambda_value,
            center_merge_tol=center_merge_tol,
            raw_fit_cache=raw_fit_cache,
            context=context,
            acquisition_value=acquisition_value,
            selection_score=selection_score,
        )
        phi_start_cache[candidate_graph_k] = fit.phi.copy()
        observed_x.append(
            _encode_candidate(
                lambda_value,
                candidate_graph_k,
                center_merge_tol,
                graph_values=graph_values,
                lambda_bounds=lambda_bounds,
                center_bounds=center_bounds,
            )
        )
        observed_y.append(float(fit.bic))
        row_key = _candidate_key(float(row["lambda"]), int(row["graph_k"]), float(row["center_merge_tol"]))
        if row_key not in candidate_index_by_key:
            candidate_index_by_key[row_key] = len(candidate_fits)
            candidate_fits.append((fit, evaluation))
            search_rows.append(row)

    best_index = int(np.argmin([fit.bic for fit, _ in candidate_fits]))
    best_fit, best_evaluation = candidate_fits[best_index]
    best_row = search_rows[best_index]
    best_graph_k = int(best_row["graph_k"])
    best_graph = _graph_cache_get(
        graph_cache,
        data=data,
        graph_k=best_graph_k,
        device=fit_options.device,
    )
    best_context = _context_cache_get(
        context_cache,
        data=data,
        graph_k=best_graph_k,
        graph=best_graph,
        device=fit_options.device,
    )
    refined_fit, refined_evaluation, refined_row = _evaluate_candidate(
        data=data,
        graph=best_graph,
        graph_k=best_graph_k,
        fit_options=fit_options,
        bic_df_scale=bic_df_scale,
        bic_cluster_penalty=bic_cluster_penalty,
        simulation_root=simulation_root,
        evaluate_candidate=evaluate_all_candidates,
        phi_start=phi_start_cache.get(best_graph_k),
        selection_method=selection_method,
        profile_name=profile_name,
        selection_step=total_evals,
        lambda_value=float(best_row["lambda"]),
        center_merge_tol=float(best_row["center_merge_tol"]),
        center_candidates=_center_search_grid(
            float(best_row["center_merge_tol"]),
            center_bounds=center_bounds,
        ),
        raw_fit_cache=raw_fit_cache,
        context=best_context,
        acquisition_value=np.nan,
        selection_score=selection_score,
    )
    refined_key = _candidate_key(
        float(refined_row["lambda"]),
        int(refined_row["graph_k"]),
        float(refined_row["center_merge_tol"]),
    )
    if refined_key not in candidate_index_by_key:
        candidate_index_by_key[refined_key] = len(candidate_fits)
        candidate_fits.append((refined_fit, refined_evaluation))
        search_rows.append(refined_row)

    incumbent_index = int(np.argmin([fit.bic for fit, _ in candidate_fits]))
    incumbent_fit, _ = candidate_fits[incumbent_index]
    incumbent_row = search_rows[incumbent_index]
    if _needs_local_refinement(incumbent_fit, data):
        collapsed_fit = _fit_is_collapsed(incumbent_fit, data)
        low_sample_fit = bool(data.num_samples <= 2)
        if collapsed_fit:
            lambda_mode = "collapsed"
        elif low_sample_fit:
            lambda_mode = "low_samples"
        else:
            lambda_mode = "broad"

        lambda_candidates = _lambda_refine_grid(
            float(incumbent_row["lambda"]),
            lambda_bounds=lambda_bounds,
            mode=lambda_mode,
        )
        if low_sample_fit:
            graph_candidates = [
                value
                for value in sorted(
                    set(
                        [
                            int(incumbent_row["graph_k"]),
                            4,
                            6,
                            8,
                            10,
                        ]
                        + [
                            int(np.clip(int(incumbent_row["graph_k"]) + delta, graph_values[0], graph_values[-1]))
                            for delta in (-2, -1, 1, 2)
                        ]
                    )
                )
                if graph_values[0] <= value <= graph_values[-1]
            ]
        elif collapsed_fit:
            graph_candidates = [
                value
                for value in sorted(
                    set(
                        [
                            int(incumbent_row["graph_k"]),
                            4,
                            6,
                            8,
                            10,
                            12,
                        ]
                    )
                )
                if graph_values[0] <= value <= graph_values[-1]
            ]
        else:
            graph_candidates = [int(incumbent_row["graph_k"])]

        center_candidates = _center_search_grid(
            float(incumbent_row["center_merge_tol"]),
            center_bounds=center_bounds,
            include_stable_anchor=bool(low_sample_fit or collapsed_fit),
        )

        for lambda_candidate in lambda_candidates:
            for graph_candidate in graph_candidates:
                if _candidate_key(lambda_candidate, graph_candidate, float(incumbent_row["center_merge_tol"])) in existing_keys:
                    continue
                existing_keys.add(_candidate_key(lambda_candidate, graph_candidate, float(incumbent_row["center_merge_tol"])))
                graph = _graph_cache_get(
                    graph_cache,
                    data=data,
                    graph_k=graph_candidate,
                    device=fit_options.device,
                )
                context = _context_cache_get(
                    context_cache,
                    data=data,
                    graph_k=graph_candidate,
                    graph=graph,
                    device=fit_options.device,
                )
                fit, evaluation, row = _evaluate_candidate(
                    data=data,
                    graph=graph,
                    graph_k=graph_candidate,
                    fit_options=fit_options,
                    bic_df_scale=bic_df_scale,
                    bic_cluster_penalty=bic_cluster_penalty,
                    simulation_root=simulation_root,
                    evaluate_candidate=evaluate_all_candidates,
                    phi_start=phi_start_cache.get(graph_candidate),
                    selection_method=selection_method,
                    profile_name=profile_name,
                    selection_step=len(search_rows),
                    lambda_value=lambda_candidate,
                    center_merge_tol=float(incumbent_row["center_merge_tol"]),
                    center_candidates=center_candidates,
                    raw_fit_cache=raw_fit_cache,
                    context=context,
                    acquisition_value=np.nan,
                    selection_score=selection_score,
                )
                phi_start_cache[graph_candidate] = fit.phi.copy()
                row_key = _candidate_key(float(row["lambda"]), int(row["graph_k"]), float(row["center_merge_tol"]))
                if row_key not in candidate_index_by_key:
                    candidate_index_by_key[row_key] = len(candidate_fits)
                    candidate_fits.append((fit, evaluation))
                    search_rows.append(row)

    shortlist_size = 4 if data.num_samples <= 2 else 3
    quality_shortlist_size = 4 if data.num_samples <= 2 else 2
    stability_bases = _selector_shortlist_indices(
        candidate_fits,
        search_rows,
        max_bic_candidates=shortlist_size,
        max_quality_candidates=quality_shortlist_size,
    )
    for base_idx in stability_bases:
        base_fit, _ = candidate_fits[base_idx]
        base_row = search_rows[base_idx]
        low_sample_fit = bool(data.num_samples <= 2)
        unstable_fit = bool(_needs_local_refinement(base_fit, data))
        collapsed_fit = bool(_fit_is_collapsed(base_fit, data))
        graph_candidates = _stability_graph_grid(
            int(base_row["graph_k"]),
            graph_values=graph_values,
            low_sample=low_sample_fit,
            unstable=unstable_fit or collapsed_fit,
        )
        if collapsed_fit:
            lambda_mode = "collapsed"
        elif low_sample_fit:
            lambda_mode = "low_samples"
        else:
            lambda_mode = "broad"
        lambda_candidates = _lambda_refine_grid(
            float(base_row["lambda"]),
            lambda_bounds=lambda_bounds,
            mode=lambda_mode,
        )
        center_candidates = _center_search_grid(
            float(base_row["center_merge_tol"]),
            center_bounds=center_bounds,
            include_stable_anchor=True,
        )
        for graph_candidate in graph_candidates:
            for lambda_candidate in lambda_candidates:
                graph = _graph_cache_get(
                    graph_cache,
                    data=data,
                    graph_k=graph_candidate,
                    device=fit_options.device,
                )
                context = _context_cache_get(
                    context_cache,
                    data=data,
                    graph_k=graph_candidate,
                    graph=graph,
                    device=fit_options.device,
                )
                fit, evaluation, row = _evaluate_candidate(
                    data=data,
                    graph=graph,
                    graph_k=graph_candidate,
                    fit_options=fit_options,
                    bic_df_scale=bic_df_scale,
                    bic_cluster_penalty=bic_cluster_penalty,
                    simulation_root=simulation_root,
                    evaluate_candidate=evaluate_all_candidates,
                    phi_start=phi_start_cache.get(graph_candidate),
                    selection_method=selection_method,
                    profile_name=profile_name,
                    selection_step=len(search_rows),
                    lambda_value=lambda_candidate,
                    center_merge_tol=float(base_row["center_merge_tol"]),
                    center_candidates=center_candidates,
                    raw_fit_cache=raw_fit_cache,
                    context=context,
                    acquisition_value=np.nan,
                    selection_score=selection_score,
                )
                phi_start_cache[graph_candidate] = fit.phi.copy()
                row_key = _candidate_key(float(row["lambda"]), int(row["graph_k"]), float(row["center_merge_tol"]))
                if row_key not in candidate_index_by_key:
                    candidate_index_by_key[row_key] = len(candidate_fits)
                    candidate_fits.append((fit, evaluation))
                    search_rows.append(row)

    if selection_score in {"ebic", "refit_ebic"}:
        if len(candidate_fits) <= 160:
            stability_indices = list(range(len(candidate_fits)))
        else:
            stability_indices = _selector_shortlist_indices(
                candidate_fits,
                search_rows,
                max_bic_candidates=max(shortlist_size + 2, 5),
                max_quality_candidates=5 if data.num_samples <= 2 else 3,
            )
        stability_rows: dict[int, dict[str, float]] = {}
        for idx in stability_indices:
            neighborhood = _neighborhood_indices(
                idx,
                search_rows,
                lambda_bounds=lambda_bounds,
                center_bounds=center_bounds,
                graph_values=graph_values,
            )
            stability_rows[idx] = _candidate_stability_metrics(
                idx,
                neighborhood,
                candidate_fits,
                search_rows,
                data=data,
                lambda_bounds=lambda_bounds,
                center_bounds=center_bounds,
                graph_values=graph_values,
            )

        if stability_rows:
            candidate_bics = np.asarray([candidate_fits[idx][0].bic for idx in stability_rows], dtype=float)
            min_bic = float(candidate_bics.min())
            bic_scale = max(float(candidate_bics.std()), 1.0)
            stability_weight = 3.0 if data.num_samples <= 2 else 2.0
            quality_values = np.asarray(
                [float(search_rows[idx].get("cluster_quality_score", np.nan)) for idx in stability_rows],
                dtype=float,
            )
            finite_quality = quality_values[np.isfinite(quality_values)]
            if finite_quality.size > 0:
                max_quality = float(finite_quality.max())
                quality_scale = max(float(finite_quality.std()), 0.25)
            else:
                max_quality = 0.0
                quality_scale = 1.0
            quality_weight = 1.50 if data.num_samples <= 2 else 0.75
            best_stable_index = None
            best_stable_score = np.inf
            for idx, metrics in stability_rows.items():
                fit, _ = candidate_fits[idx]
                bic_term = (float(fit.bic) - min_bic) / bic_scale
                basin_term = (float(metrics["local_bic_median"]) - min_bic) / bic_scale
                candidate_quality = float(search_rows[idx].get("cluster_quality_score", np.nan))
                if np.isfinite(candidate_quality):
                    quality_term = (max_quality - candidate_quality) / quality_scale
                else:
                    quality_term = 0.0
                stable_score = (
                    0.55 * bic_term
                    + 0.45 * basin_term
                    + stability_weight * (1.0 - float(metrics["stability_ari"]))
                    + 0.75 * float(metrics["cluster_variation"])
                    + quality_weight * float(quality_term)
                    + float(metrics["pathology_penalty"])
                )
                search_rows[idx]["stability_ari"] = float(metrics["stability_ari"])
                search_rows[idx]["local_cluster_variation"] = float(metrics["cluster_variation"])
                search_rows[idx]["local_bic_median"] = float(metrics["local_bic_median"])
                search_rows[idx]["pathology_penalty"] = float(metrics["pathology_penalty"])
                search_rows[idx]["stability_neighbor_count"] = int(metrics["neighbor_count"])
                search_rows[idx]["stable_selector_score"] = float(stable_score)
                if stable_score < best_stable_score:
                    best_stable_score = float(stable_score)
                    best_stable_index = int(idx)
            if best_stable_index is not None:
                best_index = int(best_stable_index)
            else:
                best_index = int(np.argmin([fit.bic for fit, _ in candidate_fits]))
        else:
            best_index = int(np.argmin([fit.bic for fit, _ in candidate_fits]))
    else:
        best_index = int(np.argmin([fit.bic for fit, _ in candidate_fits]))

    best_fit, best_evaluation = candidate_fits[best_index]
    best_row = search_rows[best_index]
    best_graph = _graph_cache_get(
        graph_cache,
        data=data,
        graph_k=int(best_row["graph_k"]),
        device=fit_options.device,
    )
    search_df = pd.DataFrame(search_rows).sort_values("selection_step").reset_index(drop=True)
    return ModelSelectionResult(
        best_fit=best_fit,
        best_evaluation=best_evaluation,
        search_df=search_df,
        graph_k=int(best_row["graph_k"]),
        graph_edges=int(best_graph.num_edges),
        bic_df_scale=float(bic_df_scale),
        bic_cluster_penalty=float(bic_cluster_penalty),
        center_merge_tol=float(best_row["center_merge_tol"]),
        selection_method=selection_method,
        profile_name=profile_name,
    )


def select_model(
    *,
    data: PatientData,
    simulation_root: Path | None,
    lambda_grid: list[float] | None,
    lambda_grid_mode: str,
    graph_k: int,
    fit_options: FitOptions,
    bic_df_scale: float,
    bic_cluster_penalty: float,
    settings_profile: str,
    selection_score: str,
    use_warm_starts: bool,
    evaluate_all_candidates: bool,
    bo_max_evals: int,
    bo_init_points: int,
    bo_random_seed: int,
) -> ModelSelectionResult:
    normalized_profile = settings_profile.strip().lower()
    if normalized_profile == "auto":
        normalized_profile = "bayes"

    if normalized_profile == "bayes":
        return _bayesian_selection(
            data=data,
            simulation_root=simulation_root,
            lambda_grid=lambda_grid,
            lambda_grid_mode=lambda_grid_mode,
            graph_k=graph_k,
            fit_options=fit_options,
            bic_df_scale=bic_df_scale,
            bic_cluster_penalty=bic_cluster_penalty,
            evaluate_all_candidates=evaluate_all_candidates,
            bo_max_evals=bo_max_evals,
            bo_init_points=bo_init_points,
            bo_random_seed=bo_random_seed,
            selection_score=selection_score,
        )

    profile_name = "manual"
    effective_graph_k = int(graph_k)
    effective_lambda_grid_mode = lambda_grid_mode
    effective_bic_df_scale = float(bic_df_scale)
    effective_bic_cluster_penalty = float(bic_cluster_penalty)
    effective_fit_options = fit_options

    if normalized_profile == "legacy_auto":
        recommended = recommend_settings_from_data(data, selection_score=selection_score)
        profile_name = recommended.profile_name
        effective_graph_k = recommended.graph_k
        effective_lambda_grid_mode = recommended.lambda_grid_mode
        effective_bic_df_scale = recommended.bic_df_scale
        effective_bic_cluster_penalty = recommended.bic_cluster_penalty
        effective_fit_options = replace(
            fit_options,
            center_merge_tol=recommended.center_merge_tol,
        )
    elif normalized_profile != "manual":
        raise ValueError(f"Unknown settings_profile: {settings_profile}")

    return _grid_search_selection(
        data=data,
        simulation_root=simulation_root,
        lambda_grid=lambda_grid,
        lambda_grid_mode=effective_lambda_grid_mode,
        graph_k=effective_graph_k,
        fit_options=effective_fit_options,
        bic_df_scale=effective_bic_df_scale,
        bic_cluster_penalty=effective_bic_cluster_penalty,
        use_warm_starts=use_warm_starts,
        evaluate_all_candidates=evaluate_all_candidates,
        profile_name=profile_name,
        selection_method=normalized_profile,
        selection_score=selection_score,
    )


__all__ = [
    "ModelSelectionResult",
    "select_model",
]
