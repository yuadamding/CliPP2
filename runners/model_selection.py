from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core.model import FitOptions, FitResult, fit_single_stage_em
from ..core.fusion_solver import (
    compute_exact_observed_data_pilot,
    compute_pooled_observed_data_start,
    compute_scalar_cell_wells,
    compute_scalar_well_start_bank,
    resolve_pairwise_fusion_graph,
)
from ..core.fusion.torch_backend import (
    cell_terms_torch,
    objective_value_torch,
    resolve_runtime,
    stationarity_residual_torch,
    to_torch_tumor_data,
)
from ..io.data import TumorData, compute_phi_init_from_counts
from ..metrics.evaluation import (
    SimulationEvaluation,
    SimulationTruth,
    evaluate_ari_against_simulation,
    evaluate_fit_against_simulation,
    load_simulation_truth,
)
from .selection import (
    LambdaBracket,
    compute_classic_bic,
    compute_extended_bic,
    default_lambda_grid,
    is_adaptive_lambda_grid_mode,
    is_cv_stability_lambda_grid_mode,
)
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
ADAPTIVE_PATH_MAX_CANDIDATES = 36
ADAPTIVE_PATH_MAX_ROUNDS = 4
ADAPTIVE_PATH_REFINE_PER_ROUND = 3
ADAPTIVE_PATH_LOG10_WIDTH_TOL = 0.05
ADAPTIVE_PATH_VALUE_CURVE_TOL = 1e-4
ADAPTIVE_PATH_FULL_FUSION_MAX_ITER = 80
CV_STABILITY_REPLICATES = 3
CV_STABILITY_TRAIN_FRACTION = 0.8
CV_STABILITY_THRESHOLD = 0.10
CV_STABILITY_RANDOM_SEED = 1729
CV_STABILITY_PAIR_SAMPLE_MAX = 50000
CV_STABILITY_MAX_PATH_CANDIDATES = 10
CV_STABILITY_MAX_PATH_ROUNDS = 1
CV_STABILITY_REFINE_PER_ROUND = 2
CV_STABILITY_MAX_EVALUATED_LAMBDAS = 8


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
    selected_ari: float | None
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
    lambda_search_mode: str
    selected_lambda_representative: float | None
    selected_lambda_left: float | None
    selected_lambda_right: float | None
    selected_lambda_interval_log10_width: float | None
    lambda_bracket_min: float | None
    lambda_bracket_eq: float | None
    lambda_bracket_full: float | None
    adaptive_refinement_rounds_completed: int
    selected_validation_loglik_mean: float | None
    selected_validation_loglik_se: float | None
    selected_instability: float | None
    cv_stability_replicates: int
    cv_stability_threshold: float


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


def _partition_signature(labels: np.ndarray) -> str:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return "empty"
    root_to_label: dict[int, int] = {}
    canonical = np.empty_like(labels)
    next_label = 0
    for idx, value in enumerate(labels):
        key = int(value)
        label = root_to_label.get(key)
        if label is None:
            label = next_label
            root_to_label[key] = label
            next_label += 1
        canonical[idx] = label
    digest = hashlib.blake2b(canonical.tobytes(), digest_size=12).hexdigest()
    return f"{int(next_label)}:{digest}"


def _profile_penalty_from_fit(fit: FitResult) -> tuple[float, float]:
    penalty = max(float(fit.penalized_objective + fit.loglik), 0.0)
    if float(fit.lambda_value) > 0.0:
        return penalty, float(penalty / float(fit.lambda_value))
    return penalty, float("nan")


def _tumor_data_with_counts(
    data: TumorData,
    *,
    alt_counts: np.ndarray,
    total_counts: np.ndarray,
    eps: float,
) -> TumorData:
    alt = np.asarray(alt_counts, dtype=np.float32)
    total = np.asarray(total_counts, dtype=np.float32)
    phi_init, init_major_mask = compute_phi_init_from_counts(
        alt_counts=alt,
        total_counts=total,
        scaling=data.scaling,
        major_cn=data.major_cn,
        minor_cn=data.minor_cn,
        phi_upper=data.phi_upper,
        eps=eps,
    )
    return TumorData(
        tumor_id=data.tumor_id,
        mutation_ids=list(data.mutation_ids),
        region_ids=list(data.region_ids),
        alt_counts=alt,
        total_counts=total,
        purity=data.purity.astype(np.float32, copy=True),
        major_cn=data.major_cn.astype(np.float32, copy=True),
        minor_cn=data.minor_cn.astype(np.float32, copy=True),
        normal_cn=data.normal_cn.astype(np.float32, copy=True),
        has_cna=data.has_cna.astype(bool, copy=True),
        scaling=data.scaling.astype(np.float32, copy=True),
        phi_upper=data.phi_upper.astype(np.float32, copy=True),
        phi_init=phi_init,
        init_major_mask=init_major_mask,
    )


def _thin_train_validation_data(
    data: TumorData,
    *,
    rng: np.random.Generator,
    train_fraction: float,
    eps: float,
) -> tuple[TumorData, TumorData]:
    total = np.rint(data.total_counts).astype(np.int64)
    alt = np.rint(data.alt_counts).astype(np.int64)
    total = np.maximum(total, 0)
    alt = np.clip(alt, 0, total)

    train_total = rng.binomial(total, float(train_fraction)).astype(np.int64)
    train_alt = rng.hypergeometric(
        ngood=alt,
        nbad=np.maximum(total - alt, 0),
        nsample=train_total,
    ).astype(np.int64)
    val_total = total - train_total
    val_alt = alt - train_alt

    train_data = _tumor_data_with_counts(
        data,
        alt_counts=train_alt,
        total_counts=train_total,
        eps=eps,
    )
    validation_data = _tumor_data_with_counts(
        data,
        alt_counts=val_alt,
        total_counts=val_total,
        eps=eps,
    )
    return train_data, validation_data


def _validation_loglik_for_fit(
    fit: FitResult,
    validation_data: TumorData,
    *,
    runtime,
    major_prior: float,
    eps: float,
) -> float:
    validation_torch_data = to_torch_tumor_data(validation_data, runtime)
    phi = torch.as_tensor(np.asarray(fit.phi), dtype=runtime.dtype, device=runtime.device)
    empty_edge = torch.empty((0,), dtype=torch.long, device=runtime.device)
    empty_weight = torch.empty((0,), dtype=runtime.dtype, device=runtime.device)
    fit_loss, _, _, _ = objective_value_torch(
        validation_torch_data,
        phi,
        edge_u=empty_edge,
        edge_v=empty_edge,
        edge_w=empty_weight,
        lambda_value=0.0,
        major_prior=major_prior,
        eps=eps,
    )
    return float(-fit_loss)


def _sample_pair_indices(
    *,
    num_mutations: int,
    rng: np.random.Generator,
    max_pairs: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if num_mutations < 2:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty, True
    total_pairs = num_mutations * (num_mutations - 1) // 2
    if total_pairs <= int(max_pairs):
        left, right = np.triu_indices(num_mutations, k=1)
        return left.astype(np.int64, copy=False), right.astype(np.int64, copy=False), True

    left = rng.integers(0, num_mutations, size=int(max_pairs), endpoint=False, dtype=np.int64)
    right = rng.integers(0, num_mutations - 1, size=int(max_pairs), endpoint=False, dtype=np.int64)
    right = np.where(right >= left, right + 1, right)
    pair_left = np.minimum(left, right)
    pair_right = np.maximum(left, right)
    return pair_left.astype(np.int64, copy=False), pair_right.astype(np.int64, copy=False), False


def _coclustering_instability(
    cluster_label_replicates: list[np.ndarray],
    *,
    rng: np.random.Generator,
    max_pairs: int,
) -> tuple[float, int, bool]:
    if not cluster_label_replicates:
        return float("nan"), 0, True
    num_mutations = int(np.asarray(cluster_label_replicates[0]).size)
    if num_mutations < 2:
        return 0.0, 0, True
    left, right, exact_pairs = _sample_pair_indices(
        num_mutations=num_mutations,
        rng=rng,
        max_pairs=max_pairs,
    )
    if left.size == 0:
        return 0.0, 0, exact_pairs
    same_counts = np.zeros(left.shape[0], dtype=np.float64)
    for labels in cluster_label_replicates:
        labels = np.asarray(labels, dtype=np.int64)
        same_counts += labels[left] == labels[right]
    probabilities = same_counts / float(len(cluster_label_replicates))
    instability = 4.0 * float(np.mean(probabilities * (1.0 - probabilities)))
    return instability, int(left.size), bool(exact_pairs)


def _evaluate_cv_stability_for_path(
    *,
    data: TumorData,
    search_df: pd.DataFrame,
    result_entries: list[tuple[FitResult, SimulationEvaluation | None, dict[str, float | int | str | bool]]],
    effective_fit_options: FitOptions,
    runtime,
    use_warm_starts: bool,
    replicates: int,
    train_fraction: float,
    stability_threshold: float,
    random_seed: int,
) -> pd.DataFrame:
    if search_df.empty:
        return search_df

    enriched = search_df.copy()
    metric_columns = {
        "validation_loglik_mean": np.nan,
        "validation_loglik_se": np.nan,
        "validation_loglik_std": np.nan,
        "validation_replicates": int(replicates),
        "instability": np.nan,
        "stability_pair_count": 0,
        "stability_pair_exact": False,
        "stability_threshold": float(stability_threshold),
        "cv_stability_score": np.nan,
        "cv_one_se_eligible": False,
        "cv_stability_eligible": False,
    }
    for column, value in metric_columns.items():
        enriched[column] = value

    fit_by_candidate_id = {
        int(row["_candidate_id"]): fit
        for fit, _, row in result_entries
        if "_candidate_id" in row
    }
    eligible_df = enriched.loc[
        enriched.apply(lambda row: bool(row.get("selection_eligible", row.get("converged", False))), axis=1)
    ].copy()
    if eligible_df.shape[0] > CV_STABILITY_MAX_EVALUATED_LAMBDAS:
        score_column = "classic_bic" if "classic_bic" in eligible_df.columns else "bic"
        top_k = max(CV_STABILITY_MAX_EVALUATED_LAMBDAS // 2, 1)
        top_ids = set(
            eligible_df.sort_values([score_column, "lambda"], ascending=[True, True], na_position="last")
            .head(top_k)["_candidate_id"]
            .astype(int)
            .tolist()
        )
        path_df = eligible_df.sort_values("lambda").reset_index(drop=True)
        spread_k = max(CV_STABILITY_MAX_EVALUATED_LAMBDAS - len(top_ids), 1)
        spread_idx = np.linspace(0, path_df.shape[0] - 1, num=min(spread_k, path_df.shape[0]), dtype=int)
        spread_ids = set(path_df.iloc[sorted(set(spread_idx.tolist()))]["_candidate_id"].astype(int).tolist())
        keep_ids = top_ids | spread_ids
        eligible_df = eligible_df.loc[eligible_df["_candidate_id"].astype(int).isin(keep_ids)].copy()
    eligible_candidate_ids = eligible_df["_candidate_id"].astype(int).tolist()
    if not eligible_candidate_ids:
        return enriched

    rng = np.random.default_rng(int(random_seed) + int(hashlib.blake2b(str(data.tumor_id).encode(), digest_size=4).hexdigest(), 16))
    replicate_data: list[tuple[TumorData, TumorData]] = [
        _thin_train_validation_data(
            data,
            rng=rng,
            train_fraction=float(train_fraction),
            eps=float(effective_fit_options.eps),
        )
        for _ in range(max(int(replicates), 1))
    ]
    stability_rng = np.random.default_rng(int(random_seed) + 991)
    cv_results: dict[int, dict[str, float | int | bool]] = {}
    for candidate_id in eligible_candidate_ids:
        full_fit = fit_by_candidate_id.get(candidate_id)
        if full_fit is None:
            continue
        lambda_value = float(full_fit.lambda_value)
        validation_logliks: list[float] = []
        replicate_labels: list[np.ndarray] = []
        previous_phi: np.ndarray | None = None
        for train_data, validation_data in replicate_data:
            phi_start = previous_phi.copy() if use_warm_starts and previous_phi is not None else full_fit.phi.copy()
            train_fit = fit_single_stage_em(
                data=train_data,
                options=replace(effective_fit_options, lambda_value=lambda_value),
                phi_start=phi_start,
                start_mode="warm_plus_pilot",
                runtime=None,
                torch_data=None,
                compute_summary=True,
            )
            validation_logliks.append(
                _validation_loglik_for_fit(
                    train_fit,
                    validation_data,
                    runtime=runtime,
                    major_prior=float(effective_fit_options.major_prior),
                    eps=float(effective_fit_options.eps),
                )
            )
            replicate_labels.append(train_fit.cluster_labels.astype(np.int64, copy=False))
            previous_phi = train_fit.phi.copy()

        values = np.asarray(validation_logliks, dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue
        mean = float(np.mean(finite_values))
        std = float(np.std(finite_values, ddof=1)) if finite_values.size > 1 else 0.0
        se = float(std / np.sqrt(float(finite_values.size))) if finite_values.size > 0 else float("nan")
        instability, pair_count, exact_pairs = _coclustering_instability(
            replicate_labels,
            rng=stability_rng,
            max_pairs=CV_STABILITY_PAIR_SAMPLE_MAX,
        )
        stable = bool(np.isfinite(instability) and instability <= float(stability_threshold))
        cv_results[candidate_id] = {
            "validation_loglik_mean": mean,
            "validation_loglik_se": se,
            "validation_loglik_std": std,
            "validation_replicates": int(finite_values.size),
            "instability": float(instability),
            "stability_pair_count": int(pair_count),
            "stability_pair_exact": bool(exact_pairs),
            "stability_threshold": float(stability_threshold),
            "cv_stability_score": float(mean - instability),
            "cv_stability_eligible": stable,
        }

    for candidate_id, metrics in cv_results.items():
        row_mask = enriched["_candidate_id"].astype(int) == int(candidate_id)
        for column, value in metrics.items():
            enriched.loc[row_mask, column] = value

    finite_cv = enriched["validation_loglik_mean"].to_numpy(dtype=float)
    finite_mask = np.isfinite(finite_cv)
    if np.any(finite_mask):
        best_idx = int(np.nanargmax(finite_cv))
        best_mean = float(enriched.iloc[best_idx]["validation_loglik_mean"])
        best_se = float(enriched.iloc[best_idx]["validation_loglik_se"])
        if not np.isfinite(best_se):
            best_se = 0.0
        one_se_mask = finite_mask & (finite_cv >= best_mean - best_se - 1e-12)
        enriched.loc[one_se_mask, "cv_one_se_eligible"] = True
    return enriched


def _full_fusion_box_residual_with_dual_balls(
    *,
    grad_smooth: torch.Tensor,
    phi: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
    atol: float,
    max_iter: int,
) -> float:
    if edge_u.numel() == 0 or lambda_value <= 0.0:
        stat = stationarity_residual_torch(
            total_grad=grad_smooth,
            phi=phi,
            lower=lower,
            upper=upper,
            atol=atol,
        )
        return float((torch.linalg.norm(stat) / (1.0 + torch.linalg.norm(grad_smooth))).item())

    dual = torch.zeros((int(edge_u.numel()), int(phi.shape[1])), dtype=phi.dtype, device=phi.device)
    radius = float(lambda_value) * edge_w
    degree = torch.bincount(
        torch.cat([edge_u, edge_v]),
        minlength=int(phi.shape[0]),
    ).max()
    step = 0.25 / max(float(degree.item()), 1.0)
    last_residual = float("inf")
    for _ in range(max(int(max_iter), 8)):
        adj = torch.zeros_like(phi)
        adj.index_add_(0, edge_u, dual)
        adj.index_add_(0, edge_v, -dual)
        total_grad = grad_smooth + adj
        stat = stationarity_residual_torch(
            total_grad=total_grad,
            phi=phi,
            lower=lower,
            upper=upper,
            atol=atol,
        )
        denom = 1.0 + torch.linalg.norm(grad_smooth) + torch.linalg.norm(adj)
        last_residual = float((torch.linalg.norm(stat) / denom).item())
        dual = dual - float(step) * (stat.index_select(0, edge_u) - stat.index_select(0, edge_v))
        dual_norm = torch.linalg.norm(dual, dim=1)
        scale = torch.maximum(torch.ones_like(dual_norm), dual_norm / radius.clamp_min(1e-12))
        dual = dual / scale[:, None]
    return float(last_residual)


def _estimate_lambda_full_light(
    *,
    torch_data,
    pooled_start: np.ndarray,
    runtime,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_eq: float,
    major_prior: float,
    eps: float,
    tol: float,
) -> tuple[float, float]:
    phi = torch.as_tensor(np.asarray(pooled_start), dtype=runtime.dtype, device=runtime.device)
    lower = torch.full_like(torch_data.phi_upper, float(eps))
    upper = torch.minimum(torch_data.phi_upper, torch.ones_like(torch_data.phi_upper))
    phi = torch.minimum(torch.maximum(phi, lower), upper)
    terms = cell_terms_torch(torch_data, phi, major_prior=major_prior, eps=eps)
    stat_tol = max(5.0 * float(tol), 1e-5)
    low = 0.0
    high = max(float(lambda_eq), ORACLE_MIN_LAMBDA)
    residual = _full_fusion_box_residual_with_dual_balls(
        grad_smooth=terms.grad,
        phi=phi,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=high,
        atol=max(float(tol), 1e-8),
        max_iter=ADAPTIVE_PATH_FULL_FUSION_MAX_ITER,
    )
    for _ in range(16):
        if residual <= stat_tol or high >= ORACLE_MAX_LAMBDA:
            break
        low = high
        high = min(high * 2.0, ORACLE_MAX_LAMBDA)
        residual = _full_fusion_box_residual_with_dual_balls(
            grad_smooth=terms.grad,
            phi=phi,
            lower=lower,
            upper=upper,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            lambda_value=high,
            atol=max(float(tol), 1e-8),
            max_iter=ADAPTIVE_PATH_FULL_FUSION_MAX_ITER,
        )
    if residual <= stat_tol:
        best_high = high
        for _ in range(12):
            mid = float(np.sqrt(max(low, ORACLE_MIN_LAMBDA) * best_high)) if low > 0.0 else 0.5 * best_high
            mid_residual = _full_fusion_box_residual_with_dual_balls(
                grad_smooth=terms.grad,
                phi=phi,
                lower=lower,
                upper=upper,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                lambda_value=mid,
                atol=max(float(tol), 1e-8),
                max_iter=ADAPTIVE_PATH_FULL_FUSION_MAX_ITER,
            )
            if mid_residual <= stat_tol:
                best_high = mid
                residual = mid_residual
            else:
                low = mid
        return float(best_high), float(residual)
    return float(high), float(residual)


def _initial_adaptive_lambda_bracket(
    *,
    torch_data,
    runtime,
    exact_pilot: np.ndarray,
    pooled_start: np.ndarray,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    major_prior: float,
    eps: float,
    tol: float,
) -> LambdaBracket:
    pilot = torch.as_tensor(np.asarray(exact_pilot), dtype=runtime.dtype, device=runtime.device)
    pooled = torch.as_tensor(np.asarray(pooled_start), dtype=runtime.dtype, device=runtime.device)
    pilot_loss, pilot_penalty_unit, _, _ = objective_value_torch(
        torch_data,
        pilot,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=1.0,
        major_prior=major_prior,
        eps=eps,
    )
    pooled_loss, pooled_penalty_unit, _, _ = objective_value_torch(
        torch_data,
        pooled,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=1.0,
        major_prior=major_prior,
        eps=eps,
    )
    denom = max(float(pilot_penalty_unit - pooled_penalty_unit), 1e-12)
    numerator = float(pooled_loss - pilot_loss)
    lambda_eq = numerator / denom if numerator > 0.0 and denom > 1e-12 else 1.0
    if not np.isfinite(lambda_eq) or lambda_eq <= 0.0:
        lambda_eq = 1.0
    lambda_full, lambda_full_residual = _estimate_lambda_full_light(
        torch_data=torch_data,
        pooled_start=pooled_start,
        runtime=runtime,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_eq=lambda_eq,
        major_prior=major_prior,
        eps=eps,
        tol=tol,
    )
    lambda_full = max(float(lambda_full), float(lambda_eq), ORACLE_MIN_LAMBDA)
    lambda_min = max(min(float(lambda_eq) / 16.0, float(lambda_full) / 256.0), ORACLE_MIN_LAMBDA)
    raw_anchors = [
        lambda_min,
        lambda_eq / 16.0,
        lambda_eq / 4.0,
        lambda_eq,
        min(lambda_eq * 4.0, lambda_full),
        min(lambda_eq * 16.0, lambda_full),
        lambda_full,
    ]
    anchors = _sorted_unique_lambdas(
        [min(max(float(value), lambda_min), lambda_full) for value in raw_anchors]
    )
    diagnostics = {
        "pilot_loss": float(pilot_loss),
        "pooled_loss": float(pooled_loss),
        "pilot_penalty_unit": float(pilot_penalty_unit),
        "pooled_penalty_unit": float(pooled_penalty_unit),
        "lambda_full_residual": float(lambda_full_residual),
    }
    return LambdaBracket(
        lambda_min=float(lambda_min),
        lambda_eq=float(lambda_eq),
        lambda_full=float(lambda_full),
        anchors=anchors,
        diagnostics=diagnostics,
    )


def _adaptive_score_column(normalized_score: str) -> str:
    if normalized_score == "summary_classic_bic":
        return "classic_bic"
    if normalized_score == "oracle_ari":
        return "ARI"
    return "bic"


def _best_candidate_rows_by_lambda(search_df: pd.DataFrame) -> pd.DataFrame:
    if search_df.empty:
        return search_df.copy()
    ranked = search_df.copy()
    ranked["_lambda_key"] = np.round(ranked["lambda"].to_numpy(dtype=float), 12)
    ranked = ranked.sort_values(
        ["_lambda_key", "converged", "penalized_objective", "selection_step"],
        ascending=[True, False, True, True],
    )
    return ranked.drop_duplicates("_lambda_key", keep="first").drop(columns=["_lambda_key"]).reset_index(drop=True)


def _adaptive_interval_proposals(
    search_df: pd.DataFrame,
    *,
    normalized_score: str,
    max_new: int,
) -> list[float]:
    path_df = _best_candidate_rows_by_lambda(search_df).sort_values("lambda").reset_index(drop=True)
    if path_df.shape[0] < 2:
        return []
    score_column = _adaptive_score_column(normalized_score)
    finite_scores = path_df[score_column].to_numpy(dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    best_score = float(np.max(finite_scores) if normalized_score == "oracle_ari" and finite_scores.size else np.min(finite_scores) if finite_scores.size else np.nan)
    intervals: list[tuple[float, float]] = []
    for idx in range(path_df.shape[0] - 1):
        left = path_df.iloc[idx]
        right = path_df.iloc[idx + 1]
        left_lambda = float(left["lambda"])
        right_lambda = float(right["lambda"])
        if left_lambda <= 0.0 or right_lambda <= left_lambda:
            continue
        log_width = float(np.log10(right_lambda) - np.log10(left_lambda))
        if log_width <= ADAPTIVE_PATH_LOG10_WIDTH_TOL:
            continue
        partition_changed = str(left.get("partition_signature", "")) != str(right.get("partition_signature", ""))
        p_left = float(left.get("profile_penalty", np.nan))
        p_right = float(right.get("profile_penalty", np.nan))
        monotonicity_violation = bool(
            np.isfinite(p_left)
            and np.isfinite(p_right)
            and p_right > p_left + 1e-6 * (1.0 + abs(p_left))
        )
        value_curve_score = 0.0
        if np.isfinite(p_left) and np.isfinite(p_right):
            v_left = float(left["penalized_objective"])
            v_right = float(right["penalized_objective"])
            trapezoid = 0.5 * (p_left + p_right) * (right_lambda - left_lambda)
            value_error = abs((v_right - v_left) - trapezoid) / (1.0 + abs(v_left) + abs(v_right))
            value_curve_score = min(value_error / ADAPTIVE_PATH_VALUE_CURVE_TOL, 4.0)
        score_focus = 0.0
        if np.isfinite(best_score):
            left_score = float(left.get(score_column, np.nan))
            right_score = float(right.get(score_column, np.nan))
            if normalized_score == "oracle_ari":
                near_best = max(left_score, right_score) >= best_score - 0.01
            else:
                near_best = min(left_score, right_score) <= best_score + max(1.0, abs(best_score) * 1e-5)
            score_focus = 1.0 if near_best else 0.0
        kkt_risk = 1.0 if (not bool(left["converged"]) or not bool(right["converged"])) else 0.0
        priority = (
            log_width
            + (3.0 if partition_changed else 0.0)
            + (3.0 if monotonicity_violation else 0.0)
            + value_curve_score
            + score_focus
            + kkt_risk
        )
        if priority <= ADAPTIVE_PATH_LOG10_WIDTH_TOL:
            continue
        intervals.append((float(priority), float(np.sqrt(left_lambda * right_lambda))))
    intervals.sort(key=lambda item: (-item[0], item[1]))
    proposals: list[float] = []
    seen: set[float] = set()
    for _, value in intervals:
        key = _canonical_lambda(value)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(float(value))
        if len(proposals) >= max_new:
            break
    return proposals


def _selected_lambda_signature_interval(
    search_df: pd.DataFrame,
    *,
    selected_candidate_id: int,
) -> tuple[float | None, float | None, float | None]:
    if search_df.empty or "_candidate_id" not in search_df.columns:
        return None, None, None
    selected = search_df.loc[search_df["_candidate_id"].astype(int) == int(selected_candidate_id)]
    if selected.empty:
        return None, None, None
    selected_row = selected.iloc[0]
    selected_lambda = float(selected_row["lambda"])
    signature = str(selected_row.get("partition_signature", ""))
    eligible = search_df.loc[search_df["converged"].astype(bool)].copy()
    if eligible.empty:
        eligible = search_df.copy()
    eligible = _best_candidate_rows_by_lambda(eligible).sort_values("lambda").reset_index(drop=True)
    lambdas = eligible["lambda"].to_numpy(dtype=float)
    signatures = eligible["partition_signature"].astype(str).to_numpy() if "partition_signature" in eligible.columns else np.full(eligible.shape[0], signature)
    if lambdas.size == 0:
        return selected_lambda, selected_lambda, 0.0
    idx = int(np.argmin(np.abs(np.log(lambdas) - np.log(selected_lambda))))
    left_idx = idx
    while left_idx > 0 and signatures[left_idx - 1] == signature:
        left_idx -= 1
    right_idx = idx
    while right_idx + 1 < lambdas.size and signatures[right_idx + 1] == signature:
        right_idx += 1
    left_lambda = float(lambdas[left_idx])
    right_lambda = float(lambdas[right_idx])
    log_width = float(np.log10(right_lambda) - np.log10(left_lambda)) if right_lambda > 0.0 and left_lambda > 0.0 else 0.0
    return left_lambda, right_lambda, log_width


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
    penalty_value, profile_penalty_value = _profile_penalty_from_fit(fit)

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
        "fit_loss": float(-fit.loglik),
        "summary_loglik": float(fit.summary_loglik),
        "penalized_objective": float(fit.penalized_objective),
        "penalty": float(penalty_value),
        "profile_penalty": float(profile_penalty_value),
        "n_clusters": int(fit.n_clusters),
        "partition_signature": _partition_signature(fit.cluster_labels),
        "converged": bool(fit.converged),
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
        "outer_edge_subgradient_residual": float(fit.outer_edge_subgradient_residual),
        "outer_dual_ball_residual": float(fit.outer_dual_ball_residual),
        "outer_box_residual": float(fit.outer_box_residual),
        "fixed_objective_kkt_residual": float(fit.fixed_objective_kkt_residual),
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
        "selection_eligible": bool(fit.selection_eligible),
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
    normalized_lambda_grid_mode = str(lambda_grid_mode).strip().lower()
    adaptive_lambda_mode = bool(lambda_grid is None and is_adaptive_lambda_grid_mode(normalized_lambda_grid_mode))
    cv_lambda_mode = bool(lambda_grid is None and is_cv_stability_lambda_grid_mode(normalized_lambda_grid_mode))
    cv_requires_stability = bool(cv_lambda_mode and normalized_lambda_grid_mode != "adaptive_cv")
    normalized_score = _normalize_selection_score_name(selection_score)
    oracle_behavior = _oracle_search_behavior(
        normalized_score=normalized_score,
        explicit_lambda_grid=explicit_lambda_grid,
        use_warm_starts=use_warm_starts,
        finalize_selected_fit=finalize_selected_fit,
    )
    lambda_search_mode = "explicit_grid" if explicit_lambda_grid else normalized_lambda_grid_mode if adaptive_lambda_mode else "fixed_grid"
    lambda_bracket: LambdaBracket | None = None
    if lambda_grid is None and not adaptive_lambda_mode:
        lambda_grid = default_lambda_grid(data, mode=lambda_grid_mode)
    lambda_grid = [] if lambda_grid is None else _sorted_unique_lambdas(lambda_grid)
    if oracle_behavior.use_heavy_refinement and lambda_grid:
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
    if adaptive_lambda_mode:
        edge_u_for_bracket, edge_v_for_bracket, edge_w_for_bracket = effective_graph.torch_cache.get(
            (str(runtime.device_name), str(runtime.dtype)),
            (None, None, None),
        )
        if edge_u_for_bracket is None or edge_v_for_bracket is None or edge_w_for_bracket is None:
            from ..core.fusion.solver import _graph_tensors

            edge_u_for_bracket, edge_v_for_bracket, edge_w_for_bracket = _graph_tensors(effective_graph, runtime)
        lambda_bracket = _initial_adaptive_lambda_bracket(
            torch_data=torch_data,
            runtime=runtime,
            exact_pilot=pilot_phi,
            pooled_start=pooled_start,
            edge_u=edge_u_for_bracket,
            edge_v=edge_v_for_bracket,
            edge_w=edge_w_for_bracket,
            major_prior=float(fit_options.major_prior),
            eps=float(fit_options.eps),
            tol=max(float(fit_options.tol), 1e-6),
        )
        lambda_grid = list(lambda_bracket.anchors)
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
            row["lambda_source"] = str(search_phase)
            row["lambda_search_mode"] = str(lambda_search_mode)
            row["lambda_bracket_min"] = np.nan if lambda_bracket is None else float(lambda_bracket.lambda_min)
            row["lambda_bracket_eq"] = np.nan if lambda_bracket is None else float(lambda_bracket.lambda_eq)
            row["lambda_bracket_full"] = np.nan if lambda_bracket is None else float(lambda_bracket.lambda_full)
            if lambda_bracket is None:
                row["lambda_full_residual"] = np.nan
            else:
                row["lambda_full_residual"] = float(lambda_bracket.diagnostics.get("lambda_full_residual", np.nan))
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
    adaptive_refinement_rounds_completed = 0
    if adaptive_lambda_mode and normalized_score != "oracle_ari":
        adaptive_max_rounds = CV_STABILITY_MAX_PATH_ROUNDS if cv_lambda_mode else ADAPTIVE_PATH_MAX_ROUNDS
        adaptive_max_candidates = CV_STABILITY_MAX_PATH_CANDIDATES if cv_lambda_mode else ADAPTIVE_PATH_MAX_CANDIDATES
        adaptive_refine_per_round = CV_STABILITY_REFINE_PER_ROUND if cv_lambda_mode else ADAPTIVE_PATH_REFINE_PER_ROUND
        for adaptive_round in range(1, adaptive_max_rounds + 1):
            if len(fit_by_lambda) >= adaptive_max_candidates:
                oracle_search_stop_reason = "adaptive_candidate_budget_reached"
                break
            interim_df = pd.DataFrame([row for _, _, row in result_entries])
            proposals = _adaptive_interval_proposals(
                interim_df,
                normalized_score=normalized_score,
                max_new=min(
                    adaptive_refine_per_round,
                    max(adaptive_max_candidates - len(fit_by_lambda), 0),
                ),
            )
            proposals = [
                value
                for value in proposals
                if _canonical_lambda(value) not in fit_by_lambda
            ]
            if not proposals:
                oracle_search_stop_reason = "adaptive_path_resolved"
                break
            before = len(fit_by_lambda)
            _evaluate_lambda_sequence(
                proposals,
                search_round=adaptive_round,
                search_phase=f"adaptive_refine_{adaptive_round}",
                start_mode="warm_plus_pilot" if use_warm_starts else "full",
                compute_summary=True,
            )
            if len(fit_by_lambda) == before:
                oracle_search_stop_reason = "adaptive_no_new_lambdas"
                break
            adaptive_refinement_rounds_completed = adaptive_round
        else:
            oracle_search_stop_reason = "adaptive_max_rounds_reached"
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
    if cv_lambda_mode and normalized_score != "oracle_ari":
        search_df = _evaluate_cv_stability_for_path(
            data=data,
            search_df=search_df,
            result_entries=result_entries,
            effective_fit_options=effective_fit_options,
            runtime=runtime,
            use_warm_starts=use_warm_starts,
            replicates=CV_STABILITY_REPLICATES,
            train_fraction=CV_STABILITY_TRAIN_FRACTION,
            stability_threshold=CV_STABILITY_THRESHOLD,
            random_seed=CV_STABILITY_RANDOM_SEED,
        )
    num_candidates = int(search_df.shape[0])
    converged_mask = search_df["converged"].astype(bool).to_numpy(dtype=bool)
    candidate_selection_eligible_mask = (
        search_df["selection_eligible"].astype(bool).to_numpy(dtype=bool)
        if "selection_eligible" in search_df.columns
        else converged_mask
    )
    num_converged_candidates = int(np.sum(converged_mask))
    converged_oracle_df = pd.DataFrame(columns=search_df.columns)
    if normalized_score == "oracle_ari":
        oracle_df = _oracle_candidate_frame(search_df)
        converged_oracle_df = _oracle_candidate_frame(search_df.loc[candidate_selection_eligible_mask].copy())
        if not converged_oracle_df.empty:
            selection_df = converged_oracle_df
            selection_used_convergence_fallback = False
        else:
            selection_df = oracle_df
            selection_used_convergence_fallback = bool(not oracle_df.empty)
    else:
        num_selection_eligible_candidates = int(np.sum(candidate_selection_eligible_mask))
        selection_df = search_df.loc[candidate_selection_eligible_mask].copy() if num_selection_eligible_candidates > 0 else search_df.copy()
        selection_used_convergence_fallback = bool(num_selection_eligible_candidates == 0 and num_candidates > 0)

    if selection_df.empty:
        raise RuntimeError(f"No candidate fits were evaluated for tumor {data.tumor_id}.")

    selection_lambda_values = selection_df["lambda"].to_numpy(dtype=float)

    cv_selection_used = False
    if cv_lambda_mode and normalized_score != "oracle_ari":
        cv_df = selection_df.loc[np.isfinite(selection_df["validation_loglik_mean"].to_numpy(dtype=float))].copy()
        if not cv_df.empty:
            cv_values = cv_df["validation_loglik_mean"].to_numpy(dtype=float)
            best_cv_pos = int(np.nanargmax(cv_values))
            best_cv_mean = float(cv_df.iloc[best_cv_pos]["validation_loglik_mean"])
            best_cv_se = float(cv_df.iloc[best_cv_pos]["validation_loglik_se"])
            if not np.isfinite(best_cv_se):
                best_cv_se = 0.0
            one_se_df = cv_df.loc[cv_df["validation_loglik_mean"].to_numpy(dtype=float) >= best_cv_mean - best_cv_se - 1e-12].copy()
            if one_se_df.empty:
                one_se_df = cv_df.iloc[[best_cv_pos]].copy()
            stable_df = one_se_df.loc[one_se_df["cv_stability_eligible"].astype(bool)].copy()
            candidate_df = stable_df if cv_requires_stability and not stable_df.empty else one_se_df
            best_row = candidate_df.sort_values(
                ["lambda", "validation_loglik_mean", "instability", "selection_step"],
                ascending=[False, False, True, True],
                na_position="last",
            ).iloc[0]
            selection_optimal_ids = set(candidate_df["_candidate_id"].astype(int).tolist())
            selection_mask = selection_df["_candidate_id"].astype(int).isin(selection_optimal_ids).to_numpy(dtype=bool)
            selection_lambda_values_for_range = candidate_df["lambda"].to_numpy(dtype=float)
            selection_min = float(np.min(selection_lambda_values_for_range))
            selection_max = float(np.max(selection_lambda_values_for_range))
            selection_count = int(np.unique(np.round(selection_lambda_values_for_range, 12)).size)
            selection_metric_value = float(best_row["validation_loglik_mean"])
            cv_selection_used = True

    if not cv_selection_used and normalized_score == "oracle_ari":
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
    elif not cv_selection_used and normalized_score == "summary_classic_bic":
        selection_min, selection_max, selection_count, selection_metric_value, selection_mask = _optimal_lambda_range(
            selection_df["classic_bic"].to_numpy(dtype=float),
            selection_lambda_values,
            maximize=False,
        )
        tied_df = selection_df.loc[selection_mask].sort_values(["classic_bic", "lambda", "selection_step"])
        best_row = tied_df.iloc[0]
    elif not cv_selection_used:
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
        eligible_mask = candidate_selection_eligible_mask
    search_df["eligible_for_selection"] = eligible_mask
    search_df["is_selection_optimal"] = search_df["_candidate_id"].astype(int).isin(selection_optimal_ids)
    search_df["is_ari_optimal"] = search_df["_candidate_id"].astype(int).isin(ari_optimal_ids)
    selected_candidate_id = int(best_row["_candidate_id"])
    search_df["is_selected_best_row"] = search_df["_candidate_id"].astype(int) == selected_candidate_id
    search_df["oracle_search_stop_reason"] = str(final_oracle_search_stop_reason)
    selected_lambda_left, selected_lambda_right, selected_lambda_log10_width = _selected_lambda_signature_interval(
        search_df,
        selected_candidate_id=selected_candidate_id,
    )
    search_df["selected_lambda_representative"] = float(best_row["lambda"])
    search_df["selected_lambda_left"] = np.nan if selected_lambda_left is None else float(selected_lambda_left)
    search_df["selected_lambda_right"] = np.nan if selected_lambda_right is None else float(selected_lambda_right)
    search_df["selected_lambda_interval_log10_width"] = (
        np.nan if selected_lambda_log10_width is None else float(selected_lambda_log10_width)
    )

    best_fit, best_evaluation, _ = result_entries[int(best_row["_candidate_id"])]
    selected_candidate_ari = float(best_row["ARI"]) if np.isfinite(float(best_row["ARI"])) else None
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
    selected_ari = float(best_evaluation.ari) if best_evaluation is not None else selected_candidate_ari
    selected_validation_loglik_mean = (
        float(best_row["validation_loglik_mean"])
        if "validation_loglik_mean" in best_row and np.isfinite(float(best_row["validation_loglik_mean"]))
        else None
    )
    selected_validation_loglik_se = (
        float(best_row["validation_loglik_se"])
        if "validation_loglik_se" in best_row and np.isfinite(float(best_row["validation_loglik_se"]))
        else None
    )
    selected_instability = (
        float(best_row["instability"])
        if "instability" in best_row and np.isfinite(float(best_row["instability"]))
        else None
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
        selected_ari=selected_ari,
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
        lambda_search_mode=str(lambda_search_mode),
        selected_lambda_representative=float(best_row["lambda"]),
        selected_lambda_left=selected_lambda_left,
        selected_lambda_right=selected_lambda_right,
        selected_lambda_interval_log10_width=selected_lambda_log10_width,
        lambda_bracket_min=None if lambda_bracket is None else float(lambda_bracket.lambda_min),
        lambda_bracket_eq=None if lambda_bracket is None else float(lambda_bracket.lambda_eq),
        lambda_bracket_full=None if lambda_bracket is None else float(lambda_bracket.lambda_full),
        adaptive_refinement_rounds_completed=int(adaptive_refinement_rounds_completed),
        selected_validation_loglik_mean=selected_validation_loglik_mean,
        selected_validation_loglik_se=selected_validation_loglik_se,
        selected_instability=selected_instability,
        cv_stability_replicates=int(CV_STABILITY_REPLICATES if cv_lambda_mode else 0),
        cv_stability_threshold=float(CV_STABILITY_THRESHOLD),
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

    if normalized_score == "oracle_ari":
        selection_method = "lambda_path_oracle_ari"
    elif is_cv_stability_lambda_grid_mode(effective_lambda_grid_mode):
        selection_method = "adaptive_cv_stability_one_se"
    else:
        selection_method = "lambda_path_grid"

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
    "ModelSelectionResult",
    "ORACLE_EXPANSION_FACTOR",
    "ORACLE_MAX_LAMBDA",
    "ORACLE_MAX_SEARCH_ROUNDS",
    "ORACLE_MIN_LAMBDA",
    "ORACLE_REFINE_POINTS",
    "ORACLE_REFINE_ROUNDS",
    "select_model",
]
