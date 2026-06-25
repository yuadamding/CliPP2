from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core.model import FitOptions, FitResult, fit_single_stage_em
from ..core.fusion_solver import (
    cluster_labels_from_edges,
    compute_exact_observed_data_pilot,
    compute_pooled_observed_data_start,
    compute_scalar_cell_wells,
    compute_scalar_well_start_bank,
    partition_constrained_observed_refit,
    resolve_pairwise_fusion_graph,
)
from ..core.fusion.torch_backend import (
    cell_terms_torch,
    objective_value_torch,
    resolve_runtime,
    stationarity_residual_torch,
    to_torch_tumor_data,
)
from ..io.data import TumorData
from ..metrics.evaluation import (
    SimulationEvaluation,
    SimulationTruth,
    evaluate_ari_against_simulation,
    evaluate_fit_against_simulation,
    load_simulation_truth,
)
from .selection import (
    LAMBDA_GRID_MODES,
    LambdaBracket,
    bic_degrees_of_freedom,
    compute_classic_bic,
    compute_classic_bic_depth_n,
    compute_extended_bic,
    default_lambda_grid,
    effective_bic_cell_count,
    effective_bic_depth_count,
    is_adaptive_lambda_grid_mode,
)
from .settings import recommend_settings_from_data

LAMBDA_SEARCH_MIN = 1e-6
LAMBDA_SEARCH_MAX = 1e6
ADAPTIVE_PATH_MAX_CANDIDATES = 8
ADAPTIVE_PATH_MAX_ROUNDS = 1
ADAPTIVE_PATH_REFINE_PER_ROUND = 1
ADAPTIVE_PATH_LOG10_WIDTH_TOL = 0.05
ADAPTIVE_PATH_VALUE_CURVE_TOL = 1e-4
ADAPTIVE_PATH_FULL_FUSION_MAX_ITER = 80
ADAPTIVE_FIRST_PASS_OUTER_MAX_ITER = 20
ADAPTIVE_FIRST_PASS_INNER_MAX_ITER = 30
KKT_POLISH_REPAIR_MULTIPLIER = 20.0
KKT_POLISH_REPAIR_FLOOR = 2e-3
KKT_POLISH_SCORE_WINDOW_ABS = 10.0
KKT_POLISH_SCORE_WINDOW_REL = 0.01
KKT_POLISH_MAX_CANDIDATES = 8
KKT_POLISH_OUTER_MAX_ITER = 100
KKT_POLISH_INNER_MAX_ITER = 60
KKT_POLISH_REPAIR_CEILING = 1e-1
KKT_POLISH_RESCUE_MAX_CANDIDATES = 3


@dataclass
class SimulationDiagnostics:
    selected_evaluation: SimulationEvaluation | None = None
    selected_ari: float | None = None
    best_ari: float | None = None
    ari_optimal_lambda_min: float | None = None
    ari_optimal_lambda_max: float | None = None
    ari_optimal_lambda_count: int = 0
    best_converged_ari: float | None = None
    best_converged_lambda_min: float | None = None
    best_converged_lambda_max: float | None = None
    best_converged_lambda_count: int = 0
    ari_hits_lower_boundary: bool = False
    ari_hits_upper_boundary: bool = False
    ari_boundary_unresolved: bool = False
    ari_optimum_resolved: bool = True
    best_ari_all_evaluated: float | None = None
    best_ari_certified: float | None = None
    best_ari_near_kkt: float | None = None
    best_ari_after_polish: float | None = None


@dataclass
class BICSelectionResult:
    best_fit: FitResult
    search_df: pd.DataFrame
    bic_df_scale: float
    bic_cluster_penalty: float
    selection_method: str
    profile_name: str
    selection_metric_value: float | None
    selection_lambda_min: float | None
    selection_lambda_max: float | None
    selection_lambda_count: int
    selection_hits_lower_boundary: bool
    selection_hits_upper_boundary: bool
    selection_boundary_unresolved: bool
    selection_optimum_resolved: bool
    adaptive_search_rounds_completed: int
    adaptive_search_stop_reason: str
    num_candidates: int
    num_converged_candidates: int
    num_candidates_all: int
    num_candidates_certified: int
    num_candidates_near_kkt: int
    num_candidates_polished: int
    num_polish_success: int
    num_polish_failed: int
    selected_kkt_residual: float | None
    best_score_all_evaluated_lambda: float | None
    best_score_all_evaluated_kkt_residual: float | None
    best_score_all_evaluated_selection_eligible: bool
    best_score_certified_lambda: float | None
    best_score_certified_kkt_residual: float | None
    selection_optimizer_limited: bool
    selection_optimizer_limited_reason: str
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
    simulation: SimulationDiagnostics = field(default_factory=SimulationDiagnostics)

    @property
    def best_evaluation(self) -> SimulationEvaluation | None:
        return self.simulation.selected_evaluation

    @property
    def selected_ari(self) -> float | None:
        return self.simulation.selected_ari

    @property
    def best_ari(self) -> float | None:
        return self.simulation.best_ari

    @property
    def ari_optimal_lambda_min(self) -> float | None:
        return self.simulation.ari_optimal_lambda_min

    @property
    def ari_optimal_lambda_max(self) -> float | None:
        return self.simulation.ari_optimal_lambda_max

    @property
    def ari_optimal_lambda_count(self) -> int:
        return self.simulation.ari_optimal_lambda_count

    @property
    def best_converged_ari(self) -> float | None:
        return self.simulation.best_converged_ari

    @property
    def best_converged_lambda_min(self) -> float | None:
        return self.simulation.best_converged_lambda_min

    @property
    def best_converged_lambda_max(self) -> float | None:
        return self.simulation.best_converged_lambda_max

    @property
    def best_converged_lambda_count(self) -> int:
        return self.simulation.best_converged_lambda_count

    @property
    def ari_hits_lower_boundary(self) -> bool:
        return self.simulation.ari_hits_lower_boundary

    @property
    def ari_hits_upper_boundary(self) -> bool:
        return self.simulation.ari_hits_upper_boundary

    @property
    def ari_boundary_unresolved(self) -> bool:
        return self.simulation.ari_boundary_unresolved

    @property
    def ari_optimum_resolved(self) -> bool:
        return self.simulation.ari_optimum_resolved

    @property
    def best_ari_all_evaluated(self) -> float | None:
        return self.simulation.best_ari_all_evaluated

    @property
    def best_ari_certified(self) -> float | None:
        return self.simulation.best_ari_certified

    @property
    def best_ari_near_kkt(self) -> float | None:
        return self.simulation.best_ari_near_kkt

    @property
    def best_ari_after_polish(self) -> float | None:
        return self.simulation.best_ari_after_polish


ModelSelectionResult = BICSelectionResult


@dataclass(frozen=True)
class FullFusionKKTResult:
    residual: float
    iterations: int
    converged: bool
    lambda_value: float


def _normalize_selection_score_name(selection_score: str) -> str:
    normalized = str(selection_score).strip().lower()
    if normalized == "bic":
        return "bic"
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
    if normalized == "bic":
        return float(classic_bic), float(classic_bic), float(extended_bic)
    raise ValueError(f"Unknown normalized selection_score: {selection_score}")


def _is_bic_selection_eligible(
    *,
    raw_kkt_eligible: bool,
    bic_refit_converged: bool,
    classic_bic: float,
) -> bool:
    return bool(
        bool(raw_kkt_eligible)
        and bool(bic_refit_converged)
        and np.isfinite(float(classic_bic))
    )


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


def _cluster_sizes_text(labels: np.ndarray) -> str:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return ""
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    return ",".join(str(int(value)) for value in counts.tolist())


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


def _effective_bic_partition_tol(options: FitOptions) -> float:
    value = options.bic_partition_tol
    if value is None:
        value = 1e-4
    return float(max(float(value), 1e-12))


def _profile_penalty_from_fit(fit: FitResult) -> tuple[float, float]:
    penalty = max(float(fit.penalized_objective + fit.loglik), 0.0)
    if float(fit.lambda_value) > 0.0:
        return penalty, float(penalty / float(fit.lambda_value))
    return penalty, float("nan")


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
    return_info: bool = False,
) -> float | tuple[float, FullFusionKKTResult]:
    lambda_float = float(lambda_value)
    iterations = 0
    converged = False
    if edge_u.numel() == 0 or lambda_value <= 0.0:
        stat = stationarity_residual_torch(
            total_grad=grad_smooth,
            phi=phi,
            lower=lower,
            upper=upper,
            atol=atol,
        )
        residual = float((torch.linalg.norm(stat) / (1.0 + torch.linalg.norm(grad_smooth))).item())
        info = FullFusionKKTResult(
            residual=residual,
            iterations=0,
            converged=True,
            lambda_value=lambda_float,
        )
        return (residual, info) if return_info else residual

    dual = torch.zeros((int(edge_u.numel()), int(phi.shape[1])), dtype=phi.dtype, device=phi.device)
    radius = lambda_float * edge_w
    degree = torch.bincount(
        torch.cat([edge_u, edge_v]),
        minlength=int(phi.shape[0]),
    ).max()
    step = 0.25 / max(float(degree.item()), 1.0)
    previous_residual = float("inf")
    last_residual = float("inf")
    max_iterations = max(int(max_iter), 1)
    for iteration in range(max_iterations):
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
        iterations = int(iteration + 1)
        if np.isfinite(previous_residual) and abs(previous_residual - last_residual) <= float(atol) * (
            1.0 + abs(previous_residual)
        ):
            converged = True
            break
        previous_residual = last_residual
        dual = dual - float(step) * (stat.index_select(0, edge_u) - stat.index_select(0, edge_v))
        dual_norm = torch.linalg.norm(dual, dim=1)
        scale = torch.maximum(torch.ones_like(dual_norm), dual_norm / radius.clamp_min(1e-12))
        dual = dual / scale[:, None]
    info = FullFusionKKTResult(
        residual=float(last_residual),
        iterations=iterations,
        converged=converged,
        lambda_value=lambda_float,
    )
    return (float(last_residual), info) if return_info else float(last_residual)


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
) -> tuple[float, float, FullFusionKKTResult]:
    phi = torch.as_tensor(np.asarray(pooled_start), dtype=runtime.dtype, device=runtime.device)
    lower = torch.full_like(torch_data.phi_upper, float(eps))
    upper = torch.minimum(torch_data.phi_upper, torch.ones_like(torch_data.phi_upper))
    phi = torch.minimum(torch.maximum(phi, lower), upper)
    terms = cell_terms_torch(torch_data, phi, major_prior=major_prior, eps=eps)
    stat_tol = max(5.0 * float(tol), 1e-5)
    low = 0.0
    high = max(float(lambda_eq), LAMBDA_SEARCH_MIN)
    residual, residual_info = _full_fusion_box_residual_with_dual_balls(
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
        return_info=True,
    )
    for _ in range(16):
        if residual <= stat_tol or high >= LAMBDA_SEARCH_MAX:
            break
        low = high
        high = min(high * 2.0, LAMBDA_SEARCH_MAX)
        residual, residual_info = _full_fusion_box_residual_with_dual_balls(
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
            return_info=True,
        )
    if residual <= stat_tol:
        best_high = high
        best_info = residual_info
        for _ in range(12):
            mid = float(np.sqrt(max(low, LAMBDA_SEARCH_MIN) * best_high)) if low > 0.0 else 0.5 * best_high
            mid_residual, mid_info = _full_fusion_box_residual_with_dual_balls(
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
                return_info=True,
            )
            if mid_residual <= stat_tol:
                best_high = mid
                residual = mid_residual
                best_info = mid_info
            else:
                low = mid
        return float(best_high), float(residual), best_info
    return float(high), float(residual), residual_info


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
    lambda_full, lambda_full_residual, lambda_full_info = _estimate_lambda_full_light(
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
    lambda_full = max(float(lambda_full), float(lambda_eq), LAMBDA_SEARCH_MIN)
    lambda_min = max(float(lambda_eq) / 4.0, LAMBDA_SEARCH_MIN)
    lambda_anchor_max = min(max(float(lambda_full) * 4.0, float(lambda_eq)), LAMBDA_SEARCH_MAX)
    raw_anchors = [
        lambda_min,
        lambda_eq / 4.0,
        lambda_eq,
        lambda_full,
        min(lambda_full * 2.0, LAMBDA_SEARCH_MAX),
        min(lambda_full * 4.0, LAMBDA_SEARCH_MAX),
    ]
    anchors = _sorted_unique_lambdas(
        [min(max(float(value), lambda_min), lambda_anchor_max) for value in raw_anchors]
    )
    diagnostics = {
        "pilot_loss": float(pilot_loss),
        "pooled_loss": float(pooled_loss),
        "pilot_penalty_unit": float(pilot_penalty_unit),
        "pooled_penalty_unit": float(pooled_penalty_unit),
        "lambda_full_residual": float(lambda_full_residual),
        "lambda_full_iterations": float(lambda_full_info.iterations),
        "lambda_full_converged": float(lambda_full_info.converged),
    }
    return LambdaBracket(
        lambda_min=float(lambda_min),
        lambda_eq=float(lambda_eq),
        lambda_full=float(lambda_full),
        anchors=anchors,
        diagnostics=diagnostics,
    )


def _adaptive_score_column(normalized_score: str) -> str:
    if normalized_score == "bic":
        return "classic_bic"
    return "bic"


def _score_maximized(normalized_score: str) -> bool:
    del normalized_score
    return False


def _kkt_polish_repair_tol(tol: float) -> float:
    return float(max(KKT_POLISH_REPAIR_MULTIPLIER * float(tol), KKT_POLISH_REPAIR_FLOOR))


def _kkt_polish_options(base_options: FitOptions) -> FitOptions:
    return replace(
        base_options,
        outer_max_iter=max(int(base_options.outer_max_iter) * 2, KKT_POLISH_OUTER_MAX_ITER),
        inner_max_iter=max(int(base_options.inner_max_iter) * 2, KKT_POLISH_INNER_MAX_ITER),
        tol=max(float(base_options.tol) * 0.5, 1e-6),
    )


def _adaptive_first_pass_options(base_options: FitOptions) -> FitOptions:
    return replace(
        base_options,
        outer_max_iter=max(int(base_options.outer_max_iter), ADAPTIVE_FIRST_PASS_OUTER_MAX_ITER),
        inner_max_iter=max(int(base_options.inner_max_iter), ADAPTIVE_FIRST_PASS_INNER_MAX_ITER),
    )


def _score_window(best_score: float) -> float:
    if not np.isfinite(best_score):
        return float(KKT_POLISH_SCORE_WINDOW_ABS)
    return float(max(KKT_POLISH_SCORE_WINDOW_ABS, KKT_POLISH_SCORE_WINDOW_REL * abs(float(best_score))))


def _score_competitive_mask(
    scores: np.ndarray,
    *,
    best_score: float,
    normalized_score: str,
) -> np.ndarray:
    window = _score_window(best_score)
    if _score_maximized(normalized_score):
        return scores >= best_score - window
    return scores <= best_score + window


def _score_strictly_better(score: float, reference: float, *, normalized_score: str) -> bool:
    if not np.isfinite(score) or not np.isfinite(reference):
        return False
    margin = 1e-8 * (1.0 + abs(float(reference)))
    if _score_maximized(normalized_score):
        return bool(float(score) > float(reference) + margin)
    return bool(float(score) < float(reference) - margin)


def _best_candidate_rows_by_lambda(search_df: pd.DataFrame) -> pd.DataFrame:
    if search_df.empty:
        return search_df.copy()
    ranked = search_df.copy()
    ranked["_lambda_key"] = np.round(ranked["lambda"].to_numpy(dtype=float), 12)
    sort_columns = ["_lambda_key"]
    ascending = [True]
    if "selection_eligible" in ranked.columns:
        sort_columns.append("selection_eligible")
        ascending.append(False)
    sort_columns.extend(["converged", "penalized_objective", "selection_step"])
    ascending.extend([False, True, True])
    ranked = ranked.sort_values(sort_columns, ascending=ascending)
    return ranked.drop_duplicates("_lambda_key", keep="first").drop(columns=["_lambda_key"]).reset_index(drop=True)


def _annotate_polish_candidates(
    search_df: pd.DataFrame,
    *,
    normalized_score: str,
    tol: float,
    max_candidates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if search_df.empty:
        return search_df.copy(), search_df.iloc[0:0].copy()

    enriched = search_df.copy()
    n_rows = int(enriched.shape[0])
    score_column = _adaptive_score_column(normalized_score)
    if score_column in enriched.columns:
        scores = enriched[score_column].to_numpy(dtype=float)
    else:
        scores = np.full(n_rows, np.nan, dtype=float)
    objectives = (
        enriched["penalized_objective"].to_numpy(dtype=float)
        if "penalized_objective" in enriched.columns
        else np.full(n_rows, np.nan, dtype=float)
    )
    mm_violations = (
        enriched["mm_consistency_violations"].to_numpy(dtype=float)
        if "mm_consistency_violations" in enriched.columns
        else np.zeros(n_rows, dtype=float)
    )
    kkt_residuals = (
        enriched["fixed_objective_kkt_residual"].to_numpy(dtype=float)
        if "fixed_objective_kkt_residual" in enriched.columns
        else np.full(n_rows, np.nan, dtype=float)
    )
    selection_eligible = (
        enriched["selection_eligible"].astype(bool).to_numpy(dtype=bool)
        if "selection_eligible" in enriched.columns
        else enriched["converged"].astype(bool).to_numpy(dtype=bool)
        if "converged" in enriched.columns
        else np.zeros(n_rows, dtype=bool)
    )
    provisional = np.isfinite(objectives) & np.isfinite(scores) & (mm_violations <= 0.0)
    failed = provisional & ~selection_eligible
    certified = provisional & selection_eligible
    repair_tol = _kkt_polish_repair_tol(tol)
    near_kkt = failed & np.isfinite(kkt_residuals) & (kkt_residuals <= repair_tol)
    repairable_kkt = failed & np.isfinite(kkt_residuals) & (kkt_residuals <= KKT_POLISH_REPAIR_CEILING)

    score_competitive = np.zeros(n_rows, dtype=bool)
    if np.any(provisional):
        certified_scores = scores[certified & np.isfinite(scores)]
        if certified_scores.size:
            best_score = float(np.max(certified_scores) if _score_maximized(normalized_score) else np.min(certified_scores))
        else:
            provisional_scores = scores[provisional & np.isfinite(scores)]
            best_score = (
                float(np.max(provisional_scores))
                if _score_maximized(normalized_score)
                else float(np.min(provisional_scores))
            )
        score_competitive = repairable_kkt & _score_competitive_mask(
            scores,
            best_score=best_score,
            normalized_score=normalized_score,
        )

    partition_diff = np.zeros(n_rows, dtype=bool)
    if np.any(certified) and "partition_signature" in enriched.columns:
        certified_df = enriched.loc[certified].copy()
        certified_df["_score_for_sort"] = certified_df[score_column].astype(float) if score_column in certified_df.columns else np.nan
        certified_df = certified_df.sort_values(
            ["_score_for_sort", "lambda", "selection_step"],
            ascending=[not _score_maximized(normalized_score), True, True],
            na_position="last",
        )
        best_signature = str(certified_df.iloc[0].get("partition_signature", ""))
        partition_values = enriched["partition_signature"].astype(str).to_numpy()
        partition_diff = failed & (partition_values != best_signature)

    boundary_neighbor = np.zeros(n_rows, dtype=bool)
    if "_candidate_id" in enriched.columns and np.any(certified):
        id_to_index = {
            int(candidate_id): int(pos)
            for pos, candidate_id in enumerate(enriched["_candidate_id"].astype(int).tolist())
        }
        path_df = _best_candidate_rows_by_lambda(enriched).sort_values("lambda").reset_index(drop=True)
        for idx in range(path_df.shape[0]):
            current = path_df.iloc[idx]
            if not bool(current.get("selection_eligible", current.get("converged", False))):
                continue
            current_score = float(current.get(score_column, np.nan))
            for neighbor_idx in (idx - 1, idx + 1):
                if neighbor_idx < 0 or neighbor_idx >= path_df.shape[0]:
                    continue
                neighbor = path_df.iloc[neighbor_idx]
                neighbor_id = int(neighbor.get("_candidate_id", -1))
                source_pos = id_to_index.get(neighbor_id)
                if source_pos is None or not failed[source_pos]:
                    continue
                neighbor_score = float(neighbor.get(score_column, np.nan))
                better_or_close = False
                if np.isfinite(current_score) and np.isfinite(neighbor_score):
                    better_or_close = bool(
                        _score_competitive_mask(
                            np.asarray([neighbor_score], dtype=float),
                            best_score=current_score,
                            normalized_score=normalized_score,
                        )[0]
                    )
                changed_partition = str(neighbor.get("partition_signature", "")) != str(current.get("partition_signature", ""))
                if repairable_kkt[source_pos] and (better_or_close or changed_partition):
                    boundary_neighbor[source_pos] = True

    priority = (
        4.0 * near_kkt.astype(float)
        + 3.0 * score_competitive.astype(float)
        + 2.0 * partition_diff.astype(float)
        + 1.0 * boundary_neighbor.astype(float)
    )
    core_polish_trigger = near_kkt | score_competitive | boundary_neighbor
    promising = failed & core_polish_trigger

    enriched["provisional_score"] = np.where(provisional, scores, np.nan)
    ranks = pd.Series(np.nan, index=enriched.index, dtype=float)
    if np.any(provisional):
        rank_values = enriched.loc[provisional, "provisional_score"].rank(
            method="min",
            ascending=not _score_maximized(normalized_score),
        )
        ranks.loc[rank_values.index] = rank_values
    enriched["provisional_rank"] = ranks
    enriched["near_kkt_for_polish"] = near_kkt
    enriched["score_competitive_for_polish"] = score_competitive
    enriched["path_boundary_for_polish"] = boundary_neighbor
    enriched["partition_diff_for_polish"] = partition_diff
    if "promising_for_polish" in enriched.columns:
        previous_promising = enriched["promising_for_polish"].astype(bool).to_numpy(dtype=bool)
    else:
        previous_promising = np.zeros(n_rows, dtype=bool)
    enriched["promising_for_polish"] = promising | previous_promising
    enriched["polish_priority"] = priority

    default_columns = {
        "polish_attempted": False,
        "polish_success": False,
        "polish_outer_max_iter": 0,
        "polish_inner_max_iter": 0,
        "pre_polish_bic": np.nan,
        "post_polish_bic": np.nan,
        "pre_polish_kkt_residual": np.nan,
        "post_polish_kkt_residual": np.nan,
        "post_polish_selection_eligible": False,
        "optimizer_limited_candidate": False,
    }
    for column, default in default_columns.items():
        if column not in enriched.columns:
            enriched[column] = default
    if "search_phase" in enriched.columns:
        polish_phase = enriched["search_phase"].astype(str).str.startswith("kkt_polish").to_numpy(dtype=bool)
        enriched.loc[polish_phase, "polish_attempted"] = True
        enriched.loc[polish_phase, "polish_success"] = selection_eligible[polish_phase]
        enriched.loc[polish_phase, "post_polish_bic"] = scores[polish_phase]
        enriched.loc[polish_phase, "post_polish_kkt_residual"] = kkt_residuals[polish_phase]
        enriched.loc[polish_phase, "post_polish_selection_eligible"] = selection_eligible[polish_phase]

    candidate_df = enriched.loc[promising].copy()
    if candidate_df.empty or int(max_candidates) <= 0:
        return enriched, candidate_df.iloc[0:0].copy()
    candidate_df["_lambda_key"] = np.round(candidate_df["lambda"].to_numpy(dtype=float), 12)
    candidate_df = candidate_df.sort_values(
        ["polish_priority", "provisional_score", "fixed_objective_kkt_residual", "lambda", "selection_step"],
        ascending=[False, not _score_maximized(normalized_score), True, True, True],
        na_position="last",
    )
    candidate_df = candidate_df.drop_duplicates("_lambda_key", keep="first").head(int(max_candidates)).drop(columns=["_lambda_key"])
    return enriched, candidate_df.reset_index(drop=True)


def _adaptive_interval_proposals(
    search_df: pd.DataFrame,
    *,
    normalized_score: str,
    tol: float,
    max_new: int,
) -> list[float]:
    path_df = _best_candidate_rows_by_lambda(search_df).sort_values("lambda").reset_index(drop=True)
    if path_df.shape[0] < 2:
        return []
    score_column = _adaptive_score_column(normalized_score)
    finite_scores = path_df[score_column].to_numpy(dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    best_score = float(np.min(finite_scores) if finite_scores.size else np.nan)
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
            near_best = min(left_score, right_score) <= best_score + max(1.0, abs(best_score) * 1e-5)
            score_focus = 1.0 if near_best else 0.0
        repair_tol = _kkt_polish_repair_tol(tol)
        left_ok = bool(left.get("selection_eligible", False))
        right_ok = bool(right.get("selection_eligible", False))
        left_kkt = float(left.get("fixed_objective_kkt_residual", np.inf))
        right_kkt = float(right.get("fixed_objective_kkt_residual", np.inf))
        kkt_risk = (
            1.0
            if (
                not left_ok
                or not right_ok
                or min(left_kkt, right_kkt) <= repair_tol
            )
            else 0.0
        )
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
    if "selection_eligible" in search_df.columns:
        eligible = search_df.loc[search_df["selection_eligible"].astype(bool)].copy()
    else:
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
    bic_partition_tol = _effective_bic_partition_tol(effective_fit_options)
    graph = effective_fit_options.graph
    if graph is None:
        raise RuntimeError("Model-selection candidates require a resolved pairwise-fusion graph.")
    bic_labels = cluster_labels_from_edges(
        fit.phi,
        edge_u=graph.edge_u,
        edge_v=graph.edge_v,
        tol=bic_partition_tol,
    )
    bic_refit = partition_constrained_observed_refit(
        data,
        bic_labels,
        major_prior=float(effective_fit_options.major_prior),
        eps=float(effective_fit_options.eps),
        tol=max(float(effective_fit_options.tol), 1e-8),
        max_iter=max(int(effective_fit_options.inner_max_iter), 32),
        hint_phi=fit.phi,
    )
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
    fit.bic = bic
    fit.classic_bic = classic_bic
    fit.extended_bic = extended_bic
    fit.classic_bic_depth_n = classic_bic_depth_n
    fit.bic_loglik = bic_loglik
    fit.bic_loglik_source = bic_refit.loglik_source
    fit.bic_df = float(bic_degrees_of_freedom(bic_n_clusters, data))
    fit.bic_active_df = float(bic_refit.active_degrees_of_freedom)
    fit.bic_n_eff = float(effective_bic_cell_count(data))
    fit.bic_depth_n_eff = float(effective_bic_depth_count(data))
    fit.bic_partition_tol = float(bic_partition_tol)
    fit.bic_refit_boundary_count = int(bic_refit.boundary_count)
    fit.bic_refit_converged = bool(bic_refit.converged)
    fit.bic_refit_phi = bic_refit.phi.astype(fit.phi.dtype, copy=False)
    fit.bic_refit_cluster_centers = bic_refit.cluster_centers.astype(fit.phi.dtype, copy=False)
    fit.bic_partition_labels = bic_labels.astype(np.int64, copy=False)
    fit.selection_score_name = canonical_score_name
    penalty_value, profile_penalty_value = _profile_penalty_from_fit(fit)
    bic_df_value = float(bic_degrees_of_freedom(bic_n_clusters, data))
    bic_n_eff_value = float(effective_bic_cell_count(data))
    bic_depth_n_eff_value = float(effective_bic_depth_count(data))
    edge_count = int(graph.edge_u.size)
    if edge_count:
        edge_weight_min = float(np.min(graph.edge_w))
        edge_weight_max = float(np.max(graph.edge_w))
        edge_weight_mean = float(np.mean(graph.edge_w))
    else:
        edge_weight_min = float("nan")
        edge_weight_max = float("nan")
        edge_weight_mean = float("nan")
    partition_hash = _partition_signature(bic_labels)
    raw_kkt_eligible = bool(fit.selection_eligible)
    bic_selection_eligible = _is_bic_selection_eligible(
        raw_kkt_eligible=raw_kkt_eligible,
        bic_refit_converged=bool(bic_refit.converged),
        classic_bic=float(classic_bic),
    )

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
        "bic_value": float(bic),
        "selection_score_name": str(canonical_score_name),
        "classic_bic": float(classic_bic),
        "extended_bic": float(extended_bic),
        "classic_bic_cell_n": float(classic_bic),
        "classic_bic_depth_n": float(classic_bic_depth_n),
        "bic_loglik": float(bic_loglik),
        "bic_loglik_source": str(bic_refit.loglik_source),
        "bic_df": float(bic_df_value),
        "bic_active_df": float(bic_refit.active_degrees_of_freedom),
        "bic_n_eff": float(bic_n_eff_value),
        "bic_depth_n_eff": float(bic_depth_n_eff_value),
        "bic_partition_tol": float(bic_partition_tol),
        "bic_n_clusters": int(bic_n_clusters),
        "loglik": float(fit.loglik),
        "raw_loglik": float(fit.loglik),
        "fit_loss": float(-fit.loglik),
        "summary_loglik": float(fit.summary_loglik),
        "refit_loglik": float(bic_loglik),
        "refit_fit_loss": float(bic_refit.fit_loss),
        "refit_converged": bool(bic_refit.converged),
        "bic_refit_converged": bool(bic_refit.converged),
        "refit_boundary_count": int(bic_refit.boundary_count),
        "refit_active_df": int(bic_refit.active_degrees_of_freedom),
        "penalized_objective": float(fit.penalized_objective),
        "raw_objective": float(fit.penalized_objective),
        "penalty": float(penalty_value),
        "raw_penalty": float(penalty_value),
        "profile_penalty": float(profile_penalty_value),
        "summary_n_clusters": int(fit.n_clusters),
        "n_clusters": int(bic_n_clusters),
        "partition_signature": partition_hash,
        "partition_hash": partition_hash,
        "cluster_sizes": _cluster_sizes_text(bic_labels),
        "converged": bool(fit.converged),
        "raw_fit_status": str(fit.failure_reason),
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
        "num_edges": edge_count,
        "edge_weight_min": edge_weight_min,
        "edge_weight_max": edge_weight_max,
        "edge_weight_mean": edge_weight_mean,
        "adaptive_weight_gamma": float(effective_fit_options.adaptive_weight_gamma),
        "adaptive_weight_floor": float(effective_fit_options.adaptive_weight_floor),
        "adaptive_weight_baseline": float(effective_fit_options.adaptive_weight_baseline),
        "edge_list_hash": _edge_list_hash(graph.edge_u, graph.edge_v, graph.edge_w),
        "input_data_hash": _input_data_hash(data),
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


def _ari_candidate_frame(search_df: pd.DataFrame) -> pd.DataFrame:
    if search_df.empty or "ARI" not in search_df.columns:
        return search_df.iloc[0:0].copy()
    ari_df = search_df.loc[np.isfinite(search_df["ARI"].to_numpy(dtype=float))].copy()
    return ari_df.sort_values(["lambda", "selection_step"]).reset_index(drop=True)


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


def _prefer_fit_candidate(candidate: FitResult, incumbent: FitResult | None) -> bool:
    if incumbent is None:
        return True
    if candidate.selection_eligible and not incumbent.selection_eligible:
        return True
    if candidate.selection_eligible != incumbent.selection_eligible:
        return False
    if candidate.selection_eligible and incumbent.selection_eligible:
        return bool(candidate.penalized_objective < incumbent.penalized_objective - 1e-8)
    candidate_kkt = float(candidate.fixed_objective_kkt_residual)
    incumbent_kkt = float(incumbent.fixed_objective_kkt_residual)
    if np.isfinite(candidate_kkt) and np.isfinite(incumbent_kkt) and abs(candidate_kkt - incumbent_kkt) > 1e-8:
        return bool(candidate_kkt < incumbent_kkt)
    if np.isfinite(candidate_kkt) and not np.isfinite(incumbent_kkt):
        return True
    if not np.isfinite(candidate_kkt) and np.isfinite(incumbent_kkt):
        return False
    return bool(candidate.penalized_objective < incumbent.penalized_objective - 1e-8)


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
    explicit_lambda_grid = lambda_grid is not None
    normalized_lambda_grid_mode = str(lambda_grid_mode).strip().lower()
    if normalized_lambda_grid_mode not in LAMBDA_GRID_MODES:
        raise ValueError(f"Unknown lambda_grid_mode: {lambda_grid_mode}")
    adaptive_lambda_mode = bool(lambda_grid is None and is_adaptive_lambda_grid_mode(normalized_lambda_grid_mode))
    normalized_score = _normalize_selection_score_name(selection_score)
    lambda_search_mode = "explicit_grid" if explicit_lambda_grid else normalized_lambda_grid_mode if adaptive_lambda_mode else "fixed_grid"
    lambda_bracket: LambdaBracket | None = None
    if lambda_grid is None and not adaptive_lambda_mode:
        lambda_grid = default_lambda_grid(data, mode=lambda_grid_mode)
    lambda_grid = [] if lambda_grid is None else _sorted_unique_lambdas(lambda_grid)

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
    search_fit_options = (
        _adaptive_first_pass_options(effective_fit_options)
        if adaptive_lambda_mode
        else effective_fit_options
    )
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
        phi_start_by_lambda: dict[float, np.ndarray] | None = None,
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
            lambda_key = _canonical_lambda(lambda_value)
            if phi_start_by_lambda is not None and lambda_key in phi_start_by_lambda:
                phi_start = phi_start_by_lambda[lambda_key].copy()
            elif use_warm_starts:
                phi_start = previous_phi.copy() if previous_phi is not None else _nearest_phi_start(lambda_value)
            else:
                phi_start = pilot_phi.copy()
            fit, evaluation, row = _evaluate_candidate(
                data=data,
                fit_options=search_fit_options,
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
                for diagnostic_name, diagnostic_value in lambda_bracket.diagnostics.items():
                    if np.isscalar(diagnostic_value):
                        row[str(diagnostic_name)] = float(diagnostic_value)
            row["_candidate_id"] = int(len(result_entries))
            result_entries.append((fit, evaluation, row))
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
        ari_only_evaluation=False,
        start_mode="warm_only" if adaptive_lambda_mode and use_warm_starts else "full",
        compute_summary=True,
    )

    adaptive_search_rounds_completed = 0
    adaptive_search_stop_reason = "not_applicable"
    adaptive_refinement_rounds_completed = 0
    if adaptive_lambda_mode:
        adaptive_max_rounds = ADAPTIVE_PATH_MAX_ROUNDS
        adaptive_max_candidates = ADAPTIVE_PATH_MAX_CANDIDATES
        adaptive_refine_per_round = ADAPTIVE_PATH_REFINE_PER_ROUND
        for adaptive_round in range(1, adaptive_max_rounds + 1):
            if len(fit_by_lambda) >= adaptive_max_candidates:
                adaptive_search_stop_reason = "adaptive_candidate_budget_reached"
                break
            interim_df = pd.DataFrame([row for _, _, row in result_entries])
            proposals = _adaptive_interval_proposals(
                interim_df,
                normalized_score=normalized_score,
                tol=float(effective_fit_options.tol),
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
                adaptive_search_stop_reason = "adaptive_path_resolved"
                break
            before = len(fit_by_lambda)
            _evaluate_lambda_sequence(
                proposals,
                search_round=adaptive_round,
                search_phase=f"adaptive_refine_{adaptive_round}",
                start_mode="warm_only" if use_warm_starts else "full",
                compute_summary=True,
            )
            if len(fit_by_lambda) == before:
                adaptive_search_stop_reason = "adaptive_no_new_lambdas"
                break
            adaptive_refinement_rounds_completed = adaptive_round
        else:
            adaptive_search_stop_reason = "adaptive_max_rounds_reached"
    adaptive_search_rounds_completed = int(adaptive_refinement_rounds_completed)
    search_df = pd.DataFrame([row for _, _, row in result_entries]).sort_values(["lambda", "selection_step"]).reset_index(drop=True)
    score_column = _adaptive_score_column(normalized_score)
    if normalized_score == "bic":
        search_df, polish_candidate_df = _annotate_polish_candidates(
            search_df,
            normalized_score=normalized_score,
            tol=float(effective_fit_options.tol),
            max_candidates=KKT_POLISH_MAX_CANDIDATES,
        )
        if not polish_candidate_df.empty:
            row_by_candidate_id = {
                int(row["_candidate_id"]): row
                for _, _, row in result_entries
                if "_candidate_id" in row
            }
            fit_by_candidate_id = {
                int(row["_candidate_id"]): fit
                for fit, _, row in result_entries
                if "_candidate_id" in row
            }
            polish_fit_options = _kkt_polish_options(effective_fit_options)
            polish_source_by_lambda: dict[float, dict[str, float | int | str | bool]] = {}
            polish_phi_by_lambda: dict[float, np.ndarray] = {}
            for _, candidate_row in polish_candidate_df.iterrows():
                candidate_id = int(candidate_row["_candidate_id"])
                lambda_key = _canonical_lambda(float(candidate_row["lambda"]))
                source_row = row_by_candidate_id.get(candidate_id)
                source_fit = fit_by_candidate_id.get(candidate_id)
                if source_row is None:
                    continue
                source_row["promising_for_polish"] = True
                source_row["polish_attempted"] = True
                source_row["polish_success"] = False
                source_row["polish_outer_max_iter"] = int(polish_fit_options.outer_max_iter)
                source_row["polish_inner_max_iter"] = int(polish_fit_options.inner_max_iter)
                source_row["pre_polish_bic"] = float(candidate_row.get(score_column, np.nan))
                source_row["post_polish_bic"] = np.nan
                source_row["pre_polish_kkt_residual"] = float(candidate_row["fixed_objective_kkt_residual"])
                source_row["post_polish_kkt_residual"] = np.nan
                source_row["post_polish_selection_eligible"] = False
                polish_source_by_lambda[lambda_key] = source_row
                if source_fit is not None:
                    polish_phi_by_lambda[lambda_key] = source_fit.phi.copy()

            polish_lambdas = [float(value) for value in polish_candidate_df["lambda"].to_numpy(dtype=float)]
            before_polish = len(result_entries)
            _evaluate_lambda_sequence(
                polish_lambdas,
                search_round=adaptive_refinement_rounds_completed + 1,
                search_phase="kkt_polish",
                allow_revisit=True,
                candidate_fit_options=polish_fit_options,
                ari_only_evaluation=False,
                start_mode="warm_only" if use_warm_starts else "full",
                compute_summary=True,
                phi_start_by_lambda=polish_phi_by_lambda,
            )
            for _, _, polished_row in result_entries[before_polish:]:
                lambda_key = _canonical_lambda(float(polished_row["lambda"]))
                source_row = polish_source_by_lambda.get(lambda_key)
                pre_kkt = (
                    float(source_row.get("pre_polish_kkt_residual", np.nan))
                    if source_row is not None
                    else np.nan
                )
                post_kkt = float(polished_row["fixed_objective_kkt_residual"])
                post_bic = float(polished_row.get(score_column, np.nan))
                polish_success = bool(polished_row["selection_eligible"])
                polished_row["promising_for_polish"] = True
                polished_row["polish_attempted"] = True
                polished_row["polish_success"] = polish_success
                polished_row["polish_outer_max_iter"] = int(polish_fit_options.outer_max_iter)
                polished_row["polish_inner_max_iter"] = int(polish_fit_options.inner_max_iter)
                polished_row["pre_polish_bic"] = float(source_row.get("pre_polish_bic", np.nan)) if source_row is not None else np.nan
                polished_row["post_polish_bic"] = post_bic
                polished_row["pre_polish_kkt_residual"] = pre_kkt
                polished_row["post_polish_kkt_residual"] = post_kkt
                polished_row["post_polish_selection_eligible"] = polish_success
                if source_row is not None:
                    source_row["polish_success"] = polish_success
                    source_row["post_polish_bic"] = post_bic
                    source_row["post_polish_kkt_residual"] = post_kkt
                    source_row["post_polish_selection_eligible"] = polish_success

            search_df = (
                pd.DataFrame([row for _, _, row in result_entries])
                .sort_values(["lambda", "selection_step"])
                .reset_index(drop=True)
            )
        search_df, _ = _annotate_polish_candidates(
            search_df,
            normalized_score=normalized_score,
            tol=float(effective_fit_options.tol),
            max_candidates=0,
        )
        has_certified_candidate = bool(
            "selection_eligible" in search_df.columns
            and search_df["selection_eligible"].astype(bool).to_numpy(dtype=bool).any()
        )
        if not has_certified_candidate:
            score_column = _adaptive_score_column(normalized_score)
            if score_column in search_df.columns and "_candidate_id" in search_df.columns:
                objectives = search_df["penalized_objective"].to_numpy(dtype=float)
                scores = search_df[score_column].to_numpy(dtype=float)
                mm_violations = (
                    search_df["mm_consistency_violations"].to_numpy(dtype=float)
                    if "mm_consistency_violations" in search_df.columns
                    else np.zeros(search_df.shape[0], dtype=float)
                )
                rescue_pool = search_df.loc[np.isfinite(objectives) & np.isfinite(scores) & (mm_violations <= 0.0)].copy()
            else:
                rescue_pool = search_df.iloc[0:0].copy()
            if not rescue_pool.empty:
                score_rescue = rescue_pool.sort_values(
                    [score_column, "lambda", "selection_step"],
                    ascending=[not _score_maximized(normalized_score), True, True],
                    na_position="last",
                ).head(max(KKT_POLISH_RESCUE_MAX_CANDIDATES - 1, 1))
                lambda_rescue = rescue_pool.sort_values(
                    ["lambda", "fixed_objective_kkt_residual", "selection_step"],
                    ascending=[False, True, True],
                    na_position="last",
                ).head(1)
                rescue_candidate_df = (
                    pd.concat([score_rescue, lambda_rescue], ignore_index=True)
                    .drop_duplicates("_candidate_id", keep="first")
                    .head(KKT_POLISH_RESCUE_MAX_CANDIDATES)
                )
                row_by_candidate_id = {
                    int(row["_candidate_id"]): row
                    for _, _, row in result_entries
                    if "_candidate_id" in row
                }
                fit_by_candidate_id = {
                    int(row["_candidate_id"]): fit
                    for fit, _, row in result_entries
                    if "_candidate_id" in row
                }
                rescue_fit_options = _kkt_polish_options(effective_fit_options)
                rescue_source_by_lambda: dict[float, dict[str, float | int | str | bool]] = {}
                rescue_phi_by_lambda: dict[float, np.ndarray] = {}
                for _, candidate_row in rescue_candidate_df.iterrows():
                    candidate_id = int(candidate_row["_candidate_id"])
                    lambda_key = _canonical_lambda(float(candidate_row["lambda"]))
                    source_row = row_by_candidate_id.get(candidate_id)
                    source_fit = fit_by_candidate_id.get(candidate_id)
                    if source_row is None:
                        continue
                    source_row["promising_for_polish"] = True
                    source_row["polish_attempted"] = True
                    source_row["polish_success"] = False
                    source_row["polish_outer_max_iter"] = int(rescue_fit_options.outer_max_iter)
                    source_row["polish_inner_max_iter"] = int(rescue_fit_options.inner_max_iter)
                    source_row["pre_polish_bic"] = float(candidate_row.get(score_column, np.nan))
                    source_row["post_polish_bic"] = np.nan
                    source_row["pre_polish_kkt_residual"] = float(candidate_row["fixed_objective_kkt_residual"])
                    source_row["post_polish_kkt_residual"] = np.nan
                    source_row["post_polish_selection_eligible"] = False
                    rescue_source_by_lambda[lambda_key] = source_row
                    if source_fit is not None:
                        rescue_phi_by_lambda[lambda_key] = source_fit.phi.copy()

                rescue_lambdas = [float(value) for value in rescue_candidate_df["lambda"].to_numpy(dtype=float)]
                before_rescue = len(result_entries)
                _evaluate_lambda_sequence(
                    rescue_lambdas,
                    search_round=adaptive_refinement_rounds_completed + 2,
                    search_phase="kkt_polish_rescue",
                    allow_revisit=True,
                    candidate_fit_options=rescue_fit_options,
                    ari_only_evaluation=False,
                    start_mode="warm_only" if use_warm_starts else "full",
                    compute_summary=True,
                    phi_start_by_lambda=rescue_phi_by_lambda,
                )
                for _, _, polished_row in result_entries[before_rescue:]:
                    lambda_key = _canonical_lambda(float(polished_row["lambda"]))
                    source_row = rescue_source_by_lambda.get(lambda_key)
                    pre_kkt = (
                        float(source_row.get("pre_polish_kkt_residual", np.nan))
                        if source_row is not None
                        else np.nan
                    )
                    post_kkt = float(polished_row["fixed_objective_kkt_residual"])
                    post_bic = float(polished_row.get(score_column, np.nan))
                    polish_success = bool(polished_row["selection_eligible"])
                    polished_row["promising_for_polish"] = True
                    polished_row["polish_attempted"] = True
                    polished_row["polish_success"] = polish_success
                    polished_row["polish_outer_max_iter"] = int(rescue_fit_options.outer_max_iter)
                    polished_row["polish_inner_max_iter"] = int(rescue_fit_options.inner_max_iter)
                    polished_row["pre_polish_bic"] = float(source_row.get("pre_polish_bic", np.nan)) if source_row is not None else np.nan
                    polished_row["post_polish_bic"] = post_bic
                    polished_row["pre_polish_kkt_residual"] = pre_kkt
                    polished_row["post_polish_kkt_residual"] = post_kkt
                    polished_row["post_polish_selection_eligible"] = polish_success
                    if source_row is not None:
                        source_row["polish_success"] = polish_success
                        source_row["post_polish_bic"] = post_bic
                        source_row["post_polish_kkt_residual"] = post_kkt
                        source_row["post_polish_selection_eligible"] = polish_success
                search_df = (
                    pd.DataFrame([row for _, _, row in result_entries])
                    .sort_values(["lambda", "selection_step"])
                    .reset_index(drop=True)
                )
                search_df, _ = _annotate_polish_candidates(
                    search_df,
                    normalized_score=normalized_score,
                    tol=float(effective_fit_options.tol),
                    max_candidates=0,
                )
        has_certified_candidate = bool(
            "selection_eligible" in search_df.columns
            and search_df["selection_eligible"].astype(bool).to_numpy(dtype=bool).any()
        )
        if not has_certified_candidate:
            score_column = _adaptive_score_column(normalized_score)
            if score_column in search_df.columns and "_candidate_id" in search_df.columns:
                objectives = search_df["penalized_objective"].to_numpy(dtype=float)
                scores = search_df[score_column].to_numpy(dtype=float)
                mm_violations = (
                    search_df["mm_consistency_violations"].to_numpy(dtype=float)
                    if "mm_consistency_violations" in search_df.columns
                    else np.zeros(search_df.shape[0], dtype=float)
                )
                full_start_pool = search_df.loc[
                    np.isfinite(objectives) & np.isfinite(scores) & (mm_violations <= 0.0)
                ].copy()
            else:
                full_start_pool = search_df.iloc[0:0].copy()
            if not full_start_pool.empty:
                score_rescue = full_start_pool.sort_values(
                    [score_column, "lambda", "selection_step"],
                    ascending=[not _score_maximized(normalized_score), True, True],
                    na_position="last",
                ).head(max(KKT_POLISH_RESCUE_MAX_CANDIDATES - 1, 1))
                lambda_rescue = full_start_pool.sort_values(
                    ["lambda", "fixed_objective_kkt_residual", "selection_step"],
                    ascending=[False, True, True],
                    na_position="last",
                ).head(1)
                full_start_rescue_df = (
                    pd.concat([score_rescue, lambda_rescue], ignore_index=True)
                    .drop_duplicates("_candidate_id", keep="first")
                    .head(KKT_POLISH_RESCUE_MAX_CANDIDATES)
                )
                fit_by_candidate_id = {
                    int(row["_candidate_id"]): fit
                    for fit, _, row in result_entries
                    if "_candidate_id" in row
                }
                full_start_phi_by_lambda: dict[float, np.ndarray] = {}
                for _, candidate_row in full_start_rescue_df.iterrows():
                    candidate_id = int(candidate_row["_candidate_id"])
                    source_fit = fit_by_candidate_id.get(candidate_id)
                    if source_fit is not None:
                        full_start_phi_by_lambda[_canonical_lambda(float(candidate_row["lambda"]))] = source_fit.phi.copy()

                full_start_lambdas = [float(value) for value in full_start_rescue_df["lambda"].to_numpy(dtype=float)]
                _evaluate_lambda_sequence(
                    full_start_lambdas,
                    search_round=adaptive_refinement_rounds_completed + 3,
                    search_phase="kkt_polish_full_start_rescue",
                    allow_revisit=True,
                    candidate_fit_options=_kkt_polish_options(effective_fit_options),
                    ari_only_evaluation=False,
                    start_mode="full",
                    compute_summary=True,
                    phi_start_by_lambda=full_start_phi_by_lambda,
                )
                search_df = (
                    pd.DataFrame([row for _, _, row in result_entries])
                    .sort_values(["lambda", "selection_step"])
                    .reset_index(drop=True)
                )
                search_df, _ = _annotate_polish_candidates(
                    search_df,
                    normalized_score=normalized_score,
                    tol=float(effective_fit_options.tol),
                    max_candidates=0,
                )
    num_candidates = int(search_df.shape[0])
    converged_mask = search_df["converged"].astype(bool).to_numpy(dtype=bool)
    candidate_selection_eligible_mask = (
        search_df["selection_eligible"].astype(bool).to_numpy(dtype=bool)
        if "selection_eligible" in search_df.columns
        else converged_mask
    )
    num_converged_candidates = int(np.sum(converged_mask))
    num_selection_eligible_candidates = int(np.sum(candidate_selection_eligible_mask))
    if num_selection_eligible_candidates == 0:
        raise RuntimeError(f"No KKT-certified candidates were eligible for model selection for tumor {data.tumor_id}.")
    selection_df = search_df.loc[candidate_selection_eligible_mask].copy()
    converged_ari_df = _ari_candidate_frame(selection_df.copy())
    selection_used_convergence_fallback = False

    if selection_df.empty:
        raise RuntimeError(f"No candidate fits were evaluated for tumor {data.tumor_id}.")

    selection_lambda_values = selection_df["lambda"].to_numpy(dtype=float)

    if normalized_score == "bic":
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
    provisional_mask = np.isfinite(all_scores) & np.isfinite(all_objectives) & (all_mm_violations <= 0.0)
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
    near_kkt_mask = (
        search_df["near_kkt_for_polish"].astype(bool).to_numpy(dtype=bool)
        if "near_kkt_for_polish" in search_df.columns
        else np.zeros(search_df.shape[0], dtype=bool)
    )
    polish_phase_mask = (
        search_df["search_phase"].astype(str).str.startswith("kkt_polish").to_numpy(dtype=bool)
        if "search_phase" in search_df.columns
        else np.zeros(search_df.shape[0], dtype=bool)
    )
    polish_success_mask = polish_phase_mask & candidate_selection_eligible_mask
    num_candidates_near_kkt = int(np.sum(near_kkt_mask))
    num_candidates_polished = int(np.sum(polish_phase_mask))
    num_polish_success = int(np.sum(polish_success_mask))
    num_polish_failed = int(num_candidates_polished - num_polish_success)

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
        best_score_all_evaluated_lambda = float(best_score_all_row["lambda"])
        best_score_all_evaluated_kkt_residual = (
            float(best_score_all_row["fixed_objective_kkt_residual"])
            if np.isfinite(float(best_score_all_row.get("fixed_objective_kkt_residual", np.nan)))
            else None
        )
        best_score_all_evaluated_selection_eligible = bool(
            best_score_all_row.get("selection_eligible", best_score_all_row.get("converged", False))
        )
    best_score_certified_lambda = None
    best_score_certified_kkt_residual = None
    if best_score_certified_row is not None:
        best_score_certified_lambda = float(best_score_certified_row["lambda"])
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
        best_score_all_eligible = bool(best_score_all_row.get("selection_eligible", best_score_all_row.get("converged", False)))
        same_lambda_polish_success = False
        if "_candidate_id" in search_df.columns:
            best_lambda_key = _canonical_lambda(float(best_score_all_row["lambda"]))
            same_lambda_polish_success = bool(
                np.any(
                    polish_phase_mask
                    & candidate_selection_eligible_mask
                    & (np.round(search_df["lambda"].to_numpy(dtype=float), 12) == best_lambda_key)
                )
            )
        if (
            _score_strictly_better(best_score_all_score, selected_provisional_score, normalized_score=normalized_score)
            and not best_score_all_eligible
        ):
            if same_lambda_polish_success:
                selection_optimizer_limited_reason = "best_provisional_score_repaired_by_polish"
            else:
                selection_optimizer_limited = True
                selection_optimizer_limited_reason = "best_provisional_score_failed_kkt_after_polish"

    if "_candidate_id" in search_df.columns and np.isfinite(selected_provisional_score):
        same_lambda_success_keys = {
            _canonical_lambda(value)
            for value in search_df.loc[polish_success_mask, "lambda"].to_numpy(dtype=float)
        }
        for _, candidate_row in search_df.loc[provisional_mask & ~candidate_selection_eligible_mask].iterrows():
            candidate_score = float(candidate_row.get(score_column, np.nan))
            candidate_lambda_key = _canonical_lambda(float(candidate_row["lambda"]))
            if (
                _score_strictly_better(candidate_score, selected_provisional_score, normalized_score=normalized_score)
                and candidate_lambda_key not in same_lambda_success_keys
            ):
                optimizer_limited_ids.add(int(candidate_row["_candidate_id"]))

    best_ari_min, best_ari_max, best_ari_count, best_ari_value, ari_mask = _optimal_lambda_range(
        selection_df["ARI"].to_numpy(dtype=float),
        selection_lambda_values,
        maximize=True,
    )
    best_converged_ari_min, best_converged_ari_max, best_converged_ari_count, best_converged_ari_value, _ = _optimal_lambda_range(
        converged_ari_df["ARI"].to_numpy(dtype=float) if not converged_ari_df.empty else np.asarray([], dtype=float),
        converged_ari_df["lambda"].to_numpy(dtype=float) if not converged_ari_df.empty else np.asarray([], dtype=float),
        maximize=True,
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
    _, _, _, best_ari_near_kkt, _ = _optimal_lambda_range(
        search_df.loc[near_kkt_mask, "ARI"].to_numpy(dtype=float)
        if "ARI" in search_df.columns and np.any(near_kkt_mask)
        else np.asarray([], dtype=float),
        search_df.loc[near_kkt_mask, "lambda"].to_numpy(dtype=float)
        if "lambda" in search_df.columns and np.any(near_kkt_mask)
        else np.asarray([], dtype=float),
        maximize=True,
    )
    _, _, _, best_ari_after_polish, _ = _optimal_lambda_range(
        search_df.loc[polish_phase_mask, "ARI"].to_numpy(dtype=float)
        if "ARI" in search_df.columns and np.any(polish_phase_mask)
        else np.asarray([], dtype=float),
        search_df.loc[polish_phase_mask, "lambda"].to_numpy(dtype=float)
        if "lambda" in search_df.columns and np.any(polish_phase_mask)
        else np.asarray([], dtype=float),
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
    search_df["selected_lambda_representative"] = float(best_row["lambda"])
    search_df["selected_lambda_left"] = np.nan if selected_lambda_left is None else float(selected_lambda_left)
    search_df["selected_lambda_right"] = np.nan if selected_lambda_right is None else float(selected_lambda_right)
    search_df["selected_lambda_interval_log10_width"] = (
        np.nan if selected_lambda_log10_width is None else float(selected_lambda_log10_width)
    )

    best_fit, best_evaluation, _ = result_entries[int(best_row["_candidate_id"])]
    selected_candidate_ari = float(best_row["ARI"]) if np.isfinite(float(best_row["ARI"])) else None
    selected_ari = float(best_evaluation.ari) if best_evaluation is not None else selected_candidate_ari
    search_df = search_df.drop(columns=["_candidate_id"])
    effective_graph.clear_torch_cache()
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
        best_ari_near_kkt=best_ari_near_kkt,
        best_ari_after_polish=best_ari_after_polish,
    )
    return BICSelectionResult(
        best_fit=best_fit,
        search_df=search_df,
        bic_df_scale=float(bic_df_scale),
        bic_cluster_penalty=float(bic_cluster_penalty),
        selection_method=selection_method,
        profile_name=profile_name,
        selection_metric_value=selection_metric_value,
        selection_lambda_min=selection_min,
        selection_lambda_max=selection_max,
        selection_lambda_count=selection_count,
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
        selected_lambda_representative=float(best_row["lambda"]),
        selected_lambda_left=selected_lambda_left,
        selected_lambda_right=selected_lambda_right,
        selected_lambda_interval_log10_width=selected_lambda_log10_width,
        lambda_bracket_min=None if lambda_bracket is None else float(lambda_bracket.lambda_min),
        lambda_bracket_eq=None if lambda_bracket is None else float(lambda_bracket.lambda_eq),
        lambda_bracket_full=None if lambda_bracket is None else float(lambda_bracket.lambda_full),
        adaptive_refinement_rounds_completed=int(adaptive_refinement_rounds_completed),
        num_candidates_all=num_candidates_all,
        num_candidates_certified=num_candidates_certified,
        num_candidates_near_kkt=num_candidates_near_kkt,
        num_candidates_polished=num_candidates_polished,
        num_polish_success=num_polish_success,
        num_polish_failed=num_polish_failed,
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
) -> BICSelectionResult:
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

    effective_lambda_grid_mode_normalized = str(effective_lambda_grid_mode).strip().lower()
    if effective_lambda_grid_mode_normalized not in LAMBDA_GRID_MODES:
        raise ValueError(f"Unknown lambda_grid_mode: {effective_lambda_grid_mode}")

    if is_adaptive_lambda_grid_mode(effective_lambda_grid_mode_normalized):
        selection_method = "adaptive_bic_path"
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
    "BICSelectionResult",
    "ModelSelectionResult",
    "SimulationDiagnostics",
    "select_model",
]
