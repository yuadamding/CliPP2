from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..core.model import FitOptions, FitResult, fit_single_stage_em
from ..core.fusion_solver import (
    PartitionRefitResult,
    SolverState,
    cluster_labels_from_edges,
    partition_constrained_observed_refit,
    prepare_torch_problem,
    torch_data_from_context,
)
from ..core.fusion.torch_backend import (
    cell_terms_torch,
    objective_value_torch,
    stationarity_residual_torch,
)
from ..core.fusion.graph_ops import graph_adjoint_edges, graph_forward_edges, project_dual_ball
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
ADAPTIVE_PATH_MAX_CANDIDATES = 40
ADAPTIVE_PATH_MAX_ROUNDS = 4
ADAPTIVE_PATH_REFINE_PER_ROUND = 5
ADAPTIVE_PATH_TRANSITION_PROBE_MAX_CANDIDATES = 3
ADAPTIVE_PATH_LOG10_WIDTH_TOL = 0.05
ADAPTIVE_PATH_VALUE_CURVE_TOL = 1e-4
ADAPTIVE_PATH_FULL_FUSION_MAX_ITER = 80
ADAPTIVE_FIRST_PASS_OUTER_MAX_ITER = 40
ADAPTIVE_FIRST_PASS_INNER_MAX_ITER = 60
KKT_POLISH_REPAIR_MULTIPLIER = 20.0
KKT_POLISH_REPAIR_FLOOR = 2e-3
KKT_POLISH_SCORE_WINDOW_ABS = 10.0
KKT_POLISH_SCORE_WINDOW_REL = 0.01
KKT_POLISH_MAX_CANDIDATES = 8
KKT_POLISH_OUTER_MAX_ITER = 140
KKT_POLISH_INNER_MAX_ITER = 96
KKT_POLISH_REPAIR_CEILING = 1e-1
KKT_POLISH_RESCUE_MAX_CANDIDATES = 3

StartArray = np.ndarray | torch.Tensor


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
class _AdaptiveIntervalProposal:
    lambda_value: float
    left_lambda: float
    right_lambda: float
    left_candidate_id: int | None
    right_candidate_id: int | None
    priority_key: tuple[int, float, float, float, float]
    reason: str
    log_width: float
    partition_changed: bool
    nonagglomerative_or_numerically_inconsistent: bool


@dataclass(frozen=True)
class FullFusionKKTResult:
    residual: float
    iterations: int
    converged: bool
    lambda_value: float


@dataclass(frozen=True)
class CandidateStaticMetadata:
    edge_count: int
    edge_weight_min: float
    edge_weight_max: float
    edge_weight_mean: float
    edge_list_hash: str
    input_data_hash: str


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


def _bic_selection_eligible_mask(search_df: pd.DataFrame) -> np.ndarray:
    n_rows = int(search_df.shape[0])
    if "bic_selection_eligible" in search_df.columns:
        return search_df["bic_selection_eligible"].astype(bool).to_numpy(dtype=bool)
    if "selection_eligible" in search_df.columns:
        return search_df["selection_eligible"].astype(bool).to_numpy(dtype=bool)
    if "converged" in search_df.columns:
        return search_df["converged"].astype(bool).to_numpy(dtype=bool)
    return np.zeros(n_rows, dtype=bool)


def _row_bic_selection_eligible(row: pd.Series) -> bool:
    return bool(row.get("bic_selection_eligible", row.get("selection_eligible", row.get("converged", False))))


def _add_bic_selection_eligible(search_df: pd.DataFrame) -> pd.DataFrame:
    if search_df.empty:
        return search_df.copy()
    enriched = search_df.copy()
    n_rows = int(enriched.shape[0])
    if "raw_kkt_eligible" in enriched.columns:
        raw_kkt = enriched["raw_kkt_eligible"].astype(bool).to_numpy(dtype=bool)
    elif "selection_eligible" in enriched.columns:
        raw_kkt = enriched["selection_eligible"].astype(bool).to_numpy(dtype=bool)
    elif "converged" in enriched.columns:
        raw_kkt = enriched["converged"].astype(bool).to_numpy(dtype=bool)
    else:
        raw_kkt = np.zeros(n_rows, dtype=bool)
    if "bic_refit_converged" in enriched.columns:
        bic_refit = enriched["bic_refit_converged"].astype(bool).to_numpy(dtype=bool)
    else:
        bic_refit = np.ones(n_rows, dtype=bool)
    if "classic_bic" in enriched.columns:
        classic_bic = enriched["classic_bic"].to_numpy(dtype=float)
    elif "bic" in enriched.columns:
        classic_bic = enriched["bic"].to_numpy(dtype=float)
    else:
        classic_bic = np.full(n_rows, np.nan, dtype=float)
    enriched["bic_selection_eligible"] = raw_kkt & bic_refit & np.isfinite(classic_bic)
    return enriched


def _annotate_bic_diagnostics(search_df: pd.DataFrame) -> pd.DataFrame:
    if search_df.empty:
        return search_df.copy()
    enriched = _add_bic_selection_eligible(search_df)
    if {"bic_df", "bic_n_eff"}.issubset(enriched.columns):
        bic_df = enriched["bic_df"].to_numpy(dtype=float)
        bic_n_eff = np.maximum(enriched["bic_n_eff"].to_numpy(dtype=float), 1.0)
        enriched["bic_penalty"] = bic_df * np.log(bic_n_eff)
    elif "bic_penalty" not in enriched.columns:
        enriched["bic_penalty"] = np.nan

    for column in ("delta_loglik_vs_one_cluster", "delta_bic_vs_one_cluster"):
        if column not in enriched.columns:
            enriched[column] = np.nan
    if not {"n_clusters", "classic_bic", "bic_loglik"}.issubset(enriched.columns):
        return enriched
    n_clusters = enriched["n_clusters"].to_numpy(dtype=float)
    one_cluster = enriched.loc[(n_clusters == 1.0) & np.isfinite(enriched["classic_bic"].to_numpy(dtype=float))].copy()
    if one_cluster.empty:
        return enriched
    one_cluster["_bic_eligible_for_baseline"] = _bic_selection_eligible_mask(one_cluster)
    baseline = one_cluster.sort_values(
        ["_bic_eligible_for_baseline", "classic_bic", "lambda", "selection_step"],
        ascending=[False, True, True, True],
    ).iloc[0]
    baseline_loglik = float(baseline.get("bic_loglik", np.nan))
    baseline_bic = float(baseline.get("classic_bic", np.nan))
    if np.isfinite(baseline_loglik):
        enriched["delta_loglik_vs_one_cluster"] = enriched["bic_loglik"].to_numpy(dtype=float) - baseline_loglik
    if np.isfinite(baseline_bic):
        enriched["delta_bic_vs_one_cluster"] = enriched["classic_bic"].to_numpy(dtype=float) - baseline_bic
    return enriched


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


def _lambda_warm_start_distance(*, source_lambda: float, target_lambda: float) -> float:
    source = float(source_lambda)
    target = float(target_lambda)
    if source > 0.0 and target > 0.0:
        return float(abs(np.log(source) - np.log(target)))
    if source <= 0.0 and target <= 0.0:
        return 0.0
    if target <= 0.0:
        return float(abs(source - target))
    return float("inf")


def _sorted_unique_lambdas(values: list[float] | np.ndarray) -> list[float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array) & (array >= 0.0)]
    if array.size == 0:
        return []
    return [float(value) for value in np.unique(np.round(np.sort(array), 12))]


def _canonical_partition_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return labels.copy()
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
    return canonical


def _partition_blocks(labels: np.ndarray) -> tuple[tuple[int, ...], ...]:
    canonical = _canonical_partition_labels(labels)
    if canonical.size == 0:
        return ()
    blocks = [
        tuple(int(idx) for idx in np.flatnonzero(canonical == int(label)).tolist())
        for label in np.unique(canonical)
    ]
    return tuple(sorted(blocks))


def _partition_signature(labels: np.ndarray) -> str:
    blocks = _partition_blocks(labels)
    if not blocks:
        return "empty"
    hasher = hashlib.blake2b(digest_size=12)
    for block in blocks:
        hasher.update(np.asarray([len(block)], dtype=np.int64).tobytes())
        if block:
            hasher.update(np.asarray(block, dtype=np.int64).tobytes())
    return f"{len(blocks)}:{hasher.hexdigest()}"


def _partition_is_coarsening(fine_labels: np.ndarray, coarse_labels: np.ndarray) -> bool:
    fine = _canonical_partition_labels(fine_labels)
    coarse = _canonical_partition_labels(coarse_labels)
    if fine.shape != coarse.shape:
        return False
    for label in np.unique(fine):
        coarse_values = np.unique(coarse[fine == int(label)])
        if coarse_values.size > 1:
            return False
    return True


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


def _candidate_static_metadata(data: TumorData, graph) -> CandidateStaticMetadata:
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
        input_data_hash=_input_data_hash(data),
    )


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


def _clone_start(start: StartArray) -> StartArray:
    if torch.is_tensor(start):
        return start.detach().clone()
    return np.asarray(start).copy()


def _fit_phi_start(fit: FitResult) -> StartArray:
    if fit.solver_state is not None and fit.solver_state.phi is not None:
        return fit.solver_state.phi
    return fit.phi


def _runtime_start_tensor(start: StartArray, runtime) -> torch.Tensor:
    if torch.is_tensor(start):
        return start.to(dtype=runtime.dtype, device=runtime.device)
    return torch.as_tensor(np.asarray(start), dtype=runtime.dtype, device=runtime.device)


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
    degree_bound: int | None = None,
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
    if degree_bound is None:
        degree = torch.bincount(
            torch.cat([edge_u, edge_v]),
            minlength=int(phi.shape[0]),
        ).max()
        effective_degree_bound = float(degree.item())
    else:
        effective_degree_bound = float(degree_bound)
    step = 0.25 / max(effective_degree_bound, 1.0)
    previous_residual = float("inf")
    last_residual = float("inf")
    max_iterations = max(int(max_iter), 1)
    for iteration in range(max_iterations):
        adj = graph_adjoint_edges(dual, edge_u=edge_u, edge_v=edge_v, num_nodes=int(phi.shape[0]))
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
        dual = dual - float(step) * graph_forward_edges(stat, edge_u=edge_u, edge_v=edge_v)
        dual = project_dual_ball(dual, radius)
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
    pooled_start: StartArray,
    runtime,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_eq: float,
    major_prior: float,
    eps: float,
    tol: float,
    degree_bound: int | None = None,
) -> tuple[float, float, FullFusionKKTResult]:
    phi = _runtime_start_tensor(pooled_start, runtime)
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
        atol=float(tol),
        max_iter=ADAPTIVE_PATH_FULL_FUSION_MAX_ITER,
        degree_bound=degree_bound,
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
            atol=float(tol),
            max_iter=ADAPTIVE_PATH_FULL_FUSION_MAX_ITER,
            degree_bound=degree_bound,
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
                atol=float(tol),
                max_iter=ADAPTIVE_PATH_FULL_FUSION_MAX_ITER,
                degree_bound=degree_bound,
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
    exact_pilot: StartArray,
    pooled_start: StartArray,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    major_prior: float,
    eps: float,
    tol: float,
    degree_bound: int | None = None,
) -> LambdaBracket:
    pilot = _runtime_start_tensor(exact_pilot, runtime)
    pooled = _runtime_start_tensor(pooled_start, runtime)
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
        degree_bound=degree_bound,
    )
    lambda_full = max(float(lambda_full), float(lambda_eq), LAMBDA_SEARCH_MIN)
    lambda_min = max(float(lambda_eq) / 128.0, LAMBDA_SEARCH_MIN)
    lambda_anchor_max = min(max(float(lambda_full), float(lambda_eq) * 4.0), LAMBDA_SEARCH_MAX)
    raw_anchors = [
        0.0,
        lambda_eq / 128.0,
        lambda_eq / 64.0,
        lambda_eq / 32.0,
        lambda_eq / 16.0,
        lambda_eq / 8.0,
        lambda_eq / 4.0,
        lambda_eq / 2.0,
        lambda_eq,
        lambda_eq * 2.0,
        lambda_eq * 4.0,
        lambda_full,
    ]
    anchors = _sorted_unique_lambdas(
        [
            0.0 if float(value) <= 0.0 else min(max(float(value), lambda_min), lambda_anchor_max)
            for value in raw_anchors
        ]
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
        tol=float(base_options.tol) * 0.5,
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
    ranked = _add_bic_selection_eligible(search_df)
    ranked["_lambda_key"] = np.round(ranked["lambda"].to_numpy(dtype=float), 12)
    defaults = {
        "selection_eligible": False,
        "bic_refit_converged": False,
        "converged": False,
        "classic_bic": np.inf,
        "penalized_objective": np.inf,
        "selection_step": np.arange(ranked.shape[0], dtype=int),
    }
    for column, default in defaults.items():
        if column not in ranked.columns:
            ranked[column] = default
    sort_columns = [
        "_lambda_key",
        "bic_selection_eligible",
        "selection_eligible",
        "bic_refit_converged",
        "classic_bic",
        "penalized_objective",
        "selection_step",
    ]
    ascending = [True, False, False, False, True, True, True]
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

    enriched = _add_bic_selection_eligible(search_df)
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
    accepted_steps = (
        enriched["accepted_outer_steps"].to_numpy(dtype=float)
        if "accepted_outer_steps" in enriched.columns
        else np.zeros(n_rows, dtype=float)
    )
    relative_changes = (
        enriched["final_relative_objective_change"].to_numpy(dtype=float)
        if "final_relative_objective_change" in enriched.columns
        else np.zeros(n_rows, dtype=float)
    )
    step_residuals = (
        enriched["final_step_residual"].to_numpy(dtype=float)
        if "final_step_residual" in enriched.columns
        else np.zeros(n_rows, dtype=float)
    )
    selection_eligible = _bic_selection_eligible_mask(enriched)
    provisional = np.isfinite(objectives) & np.isfinite(scores) & np.isfinite(kkt_residuals) & (mm_violations <= 0.0)
    failed = provisional & ~selection_eligible
    certified = provisional & selection_eligible
    repair_tol = _kkt_polish_repair_tol(tol)
    near_kkt = failed & np.isfinite(kkt_residuals) & (kkt_residuals <= repair_tol)
    repairable_kkt = failed & (kkt_residuals <= KKT_POLISH_REPAIR_CEILING)
    moving = (
        (accepted_steps > 0.0)
        & (
            (relative_changes > float(tol))
            | (step_residuals > float(np.sqrt(float(tol))))
        )
    )
    continuable = failed & moving

    enriched["provisional_score"] = np.where(provisional, scores, np.nan)
    ranks = pd.Series(np.nan, index=enriched.index, dtype=float)
    if np.any(provisional):
        rank_values = enriched.loc[provisional, "provisional_score"].rank(
            method="min",
            ascending=not _score_maximized(normalized_score),
        )
        ranks.loc[rank_values.index] = rank_values
    rank_array = ranks.to_numpy(dtype=float)
    rank_rescue = (
        continuable
        & np.isfinite(rank_array)
        & (rank_array <= float(KKT_POLISH_RESCUE_MAX_CANDIDATES))
    )
    stalled_nonstationary = failed & ~moving & (kkt_residuals > repair_tol)

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
        score_competitive = continuable & _score_competitive_mask(
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
            if not _row_bic_selection_eligible(current):
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
                if continuable[source_pos] and (better_or_close or changed_partition):
                    boundary_neighbor[source_pos] = True

    priority = (
        4.0 * near_kkt.astype(float)
        + 3.0 * score_competitive.astype(float)
        + 2.5 * rank_rescue.astype(float)
        + 2.0 * partition_diff.astype(float)
        + 1.0 * boundary_neighbor.astype(float)
    )
    core_polish_trigger = near_kkt | score_competitive | rank_rescue | boundary_neighbor
    promising = failed & core_polish_trigger

    enriched["provisional_rank"] = ranks
    enriched["repairable_kkt_for_polish"] = repairable_kkt
    enriched["moving_for_polish"] = moving
    enriched["continuable_for_polish"] = continuable
    enriched["near_kkt_for_polish"] = near_kkt
    enriched["score_competitive_for_polish"] = score_competitive
    enriched["rank_rescue_for_polish"] = rank_rescue
    enriched["stalled_nonstationary_for_polish"] = stalled_nonstationary
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


def _row_candidate_id(row: pd.Series) -> int | None:
    if "_candidate_id" not in row:
        return None
    value = row.get("_candidate_id", np.nan)
    if pd.isna(value):
        return None
    return int(value)


def _row_cluster_count(row: pd.Series) -> int | None:
    for column in ("n_clusters", "bic_n_clusters", "summary_n_clusters"):
        if column not in row:
            continue
        value = row.get(column, np.nan)
        if pd.isna(value):
            continue
        return int(value)
    return None


def _lambda_interval_log10_width(left_lambda: float, right_lambda: float) -> float:
    left_lambda = float(left_lambda)
    right_lambda = float(right_lambda)
    if not np.isfinite(left_lambda) or not np.isfinite(right_lambda) or right_lambda <= left_lambda:
        return 0.0
    if left_lambda > 0.0:
        return float(max(np.log10(right_lambda) - np.log10(left_lambda), 0.0))
    if right_lambda <= LAMBDA_SEARCH_MIN * (1.0 + 1e-12):
        return 0.0
    return float(max(np.log10(right_lambda) - np.log10(LAMBDA_SEARCH_MIN), 0.0))


def _lambda_interval_midpoint(left_lambda: float, right_lambda: float) -> float | None:
    left_lambda = float(left_lambda)
    right_lambda = float(right_lambda)
    if not np.isfinite(left_lambda) or not np.isfinite(right_lambda) or right_lambda <= left_lambda:
        return None
    if left_lambda <= 0.0:
        if right_lambda <= LAMBDA_SEARCH_MIN * (1.0 + 1e-12):
            return None
        midpoint = max(0.5 * right_lambda, LAMBDA_SEARCH_MIN)
    else:
        midpoint = float(np.sqrt(left_lambda * right_lambda))
    if not (left_lambda < midpoint < right_lambda):
        return None
    return float(midpoint)


def _adaptive_transition_probe_records(
    lambda_bracket: LambdaBracket | None,
    evaluated_lambdas: list[float] | np.ndarray,
    *,
    max_new: int,
) -> list[_AdaptiveIntervalProposal]:
    if lambda_bracket is None or int(max_new) <= 0:
        return []
    lambda_eq = float(lambda_bracket.lambda_eq)
    if not np.isfinite(lambda_eq) or lambda_eq <= 0.0:
        return []

    lambda_min = max(float(lambda_bracket.lambda_min), LAMBDA_SEARCH_MIN)
    if not np.isfinite(lambda_min) or lambda_min <= 0.0:
        lambda_min = LAMBDA_SEARCH_MIN
    lambda_full = float(lambda_bracket.lambda_full)
    if not np.isfinite(lambda_full) or lambda_full <= 0.0:
        lambda_full = lambda_eq
    transition_upper = min(max(lambda_full, 4.0 * lambda_eq), LAMBDA_SEARCH_MAX)
    evaluated_keys = {_canonical_lambda(value) for value in _sorted_unique_lambdas(evaluated_lambdas)}

    records: list[_AdaptiveIntervalProposal] = []
    seen = set(evaluated_keys)

    def _add_probe(left_lambda: float, right_lambda: float, reason: str, priority_order: int) -> None:
        if len(records) >= int(max_new):
            return
        left = float(max(left_lambda, 0.0))
        right = float(min(max(right_lambda, 0.0), LAMBDA_SEARCH_MAX))
        midpoint = _lambda_interval_midpoint(left, right)
        if midpoint is None:
            return
        key = _canonical_lambda(midpoint)
        if key in seen:
            return
        seen.add(key)
        log_width = _lambda_interval_log10_width(left, right)
        records.append(
            _AdaptiveIntervalProposal(
                lambda_value=float(midpoint),
                left_lambda=float(left),
                right_lambda=float(right),
                left_candidate_id=None,
                right_candidate_id=None,
                priority_key=(-1, float(priority_order), 0.0, -float(log_width), 0.0),
                reason=str(reason),
                log_width=float(log_width),
                partition_changed=False,
                nonagglomerative_or_numerically_inconsistent=False,
            )
        )

    upper_right = min(2.0 * lambda_eq, transition_upper)
    _add_probe(
        lambda_eq,
        upper_right,
        "lambda_eq_upper_transition_probe",
        0,
    )
    _add_probe(
        max(0.5 * lambda_eq, lambda_min),
        lambda_eq,
        "lambda_eq_lower_transition_probe",
        1,
    )
    _add_probe(
        upper_right,
        transition_upper,
        "lambda_eq_high_transition_probe",
        2,
    )
    return records


def _adaptive_interval_proposal_records(
    search_df: pd.DataFrame,
    *,
    normalized_score: str,
    tol: float,
    max_new: int,
    partition_labels_by_candidate_id: dict[int, np.ndarray] | None = None,
) -> list[_AdaptiveIntervalProposal]:
    path_df = _best_candidate_rows_by_lambda(search_df).sort_values("lambda").reset_index(drop=True)
    if path_df.shape[0] < 2 or int(max_new) <= 0:
        return []
    score_column = _adaptive_score_column(normalized_score)
    finite_scores = path_df[score_column].to_numpy(dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if _score_maximized(normalized_score):
        best_score = float(np.max(finite_scores) if finite_scores.size else np.nan)
    else:
        best_score = float(np.min(finite_scores) if finite_scores.size else np.nan)
    intervals: list[_AdaptiveIntervalProposal] = []
    for idx in range(path_df.shape[0] - 1):
        left = path_df.iloc[idx]
        right = path_df.iloc[idx + 1]
        left_lambda = float(left["lambda"])
        right_lambda = float(right["lambda"])
        if right_lambda <= left_lambda:
            continue
        log_width = _lambda_interval_log10_width(left_lambda, right_lambda)
        midpoint = _lambda_interval_midpoint(left_lambda, right_lambda)
        if midpoint is None:
            continue
        left_candidate_id = _row_candidate_id(left)
        right_candidate_id = _row_candidate_id(right)
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
            score_window = max(1.0, abs(best_score) * 1e-5)
            if _score_maximized(normalized_score):
                near_best = max(left_score, right_score) >= best_score - score_window
            else:
                near_best = min(left_score, right_score) <= best_score + score_window
            score_focus = 1.0 if near_best else 0.0
        repair_tol = _kkt_polish_repair_tol(tol)
        left_ok = _row_bic_selection_eligible(left)
        right_ok = _row_bic_selection_eligible(right)
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
        left_k = _row_cluster_count(left)
        right_k = _row_cluster_count(right)
        cluster_jump = 0 if left_k is None or right_k is None else abs(int(right_k) - int(left_k))
        k_increases = bool(left_k is not None and right_k is not None and int(right_k) > int(left_k))
        nonnested_partition_change = False
        if (
            partition_labels_by_candidate_id is not None
            and partition_changed
            and left_ok
            and right_ok
            and left_candidate_id is not None
            and right_candidate_id is not None
            and left_candidate_id in partition_labels_by_candidate_id
            and right_candidate_id in partition_labels_by_candidate_id
        ):
            nonnested_partition_change = not _partition_is_coarsening(
                partition_labels_by_candidate_id[left_candidate_id],
                partition_labels_by_candidate_id[right_candidate_id],
            )
        nonagglomerative_or_numerically_inconsistent = bool(
            nonnested_partition_change or k_increases or monotonicity_violation
        )
        failed_endpoint_near_best = bool(score_focus and (not left_ok or not right_ok))

        if nonagglomerative_or_numerically_inconsistent or failed_endpoint_near_best:
            priority_class = 0
            if nonnested_partition_change:
                reason = "nonnested_partition_change"
            elif k_increases:
                reason = "cluster_count_increases_with_lambda"
            elif monotonicity_violation:
                reason = "penalty_monotonicity_violation"
            else:
                reason = "failed_kkt_near_best_bic"
        elif partition_changed:
            priority_class = 1
            reason = "partition_signature_change"
        elif cluster_jump > 1:
            priority_class = 2
            reason = "cluster_count_jump"
        elif score_focus and partition_changed:
            priority_class = 3
            reason = "selected_partition_boundary"
        elif log_width > ADAPTIVE_PATH_LOG10_WIDTH_TOL:
            priority_class = 4
            reason = "wide_same_partition_interval"
        elif value_curve_score > 0.0 or kkt_risk > 0.0:
            priority_class = 5
            reason = "soft_path_diagnostic"
        else:
            continue

        priority_key = (
            int(priority_class),
            -float(score_focus),
            -float(cluster_jump),
            -float(log_width),
            -float(value_curve_score + kkt_risk),
        )
        intervals.append(
            _AdaptiveIntervalProposal(
                lambda_value=float(midpoint),
                left_lambda=float(left_lambda),
                right_lambda=float(right_lambda),
                left_candidate_id=left_candidate_id,
                right_candidate_id=right_candidate_id,
                priority_key=priority_key,
                reason=str(reason),
                log_width=float(log_width),
                partition_changed=bool(partition_changed),
                nonagglomerative_or_numerically_inconsistent=bool(
                    nonagglomerative_or_numerically_inconsistent
                ),
            )
        )
    intervals.sort(key=lambda item: (*item.priority_key, item.lambda_value))
    proposals: list[_AdaptiveIntervalProposal] = []
    seen: set[float] = set()
    for proposal in intervals:
        key = _canonical_lambda(proposal.lambda_value)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(proposal)
        if len(proposals) >= max_new:
            break
    return proposals


def _adaptive_interval_proposals(
    search_df: pd.DataFrame,
    *,
    normalized_score: str,
    tol: float,
    max_new: int,
    partition_labels_by_candidate_id: dict[int, np.ndarray] | None = None,
) -> list[float]:
    return [
        float(proposal.lambda_value)
        for proposal in _adaptive_interval_proposal_records(
            search_df,
            normalized_score=normalized_score,
            tol=tol,
            max_new=max_new,
            partition_labels_by_candidate_id=partition_labels_by_candidate_id,
        )
    ]


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
    eligible = search_df.loc[_bic_selection_eligible_mask(search_df)].copy()
    if eligible.empty:
        eligible = search_df.copy()
    eligible = _best_candidate_rows_by_lambda(eligible).sort_values("lambda").reset_index(drop=True)
    lambdas = eligible["lambda"].to_numpy(dtype=float)
    signatures = eligible["partition_signature"].astype(str).to_numpy() if "partition_signature" in eligible.columns else np.full(eligible.shape[0], signature)
    if lambdas.size == 0:
        return selected_lambda, selected_lambda, 0.0
    if selected_lambda > 0.0:
        distances = np.full(lambdas.shape, np.inf, dtype=float)
        positive_lambda_mask = lambdas > 0.0
        distances[positive_lambda_mask] = np.abs(
            np.log(lambdas[positive_lambda_mask]) - np.log(selected_lambda)
        )
        if not np.any(np.isfinite(distances)):
            distances = np.abs(lambdas - selected_lambda)
    else:
        distances = np.abs(lambdas - selected_lambda)
    idx = int(np.argmin(distances))
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
        solver_context=solver_context,
        solver_state=solver_state,
        compute_summary=compute_summary,
    )
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
    partition_hash = _partition_signature(bic_labels)
    bic_refit_cache_hit = False
    if bic_refit_cache is not None and partition_hash in bic_refit_cache:
        bic_refit = bic_refit_cache[partition_hash]
        bic_refit_cache_hit = True
    else:
        bic_refit = partition_constrained_observed_refit(
            data,
            bic_labels,
            major_prior=float(effective_fit_options.major_prior),
            eps=float(effective_fit_options.eps),
            tol=float(effective_fit_options.tol),
            max_iter=max(int(effective_fit_options.inner_max_iter), 32),
            hint_phi=fit.phi,
        )
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
    bic_penalty_value = float(bic_df_value * np.log(max(bic_n_eff_value, 1.0)))
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
        "bic_penalty": float(bic_penalty_value),
        "bic_n_eff": float(bic_n_eff_value),
        "bic_depth_n_eff": float(bic_depth_n_eff_value),
        "delta_loglik_vs_one_cluster": np.nan,
        "delta_bic_vs_one_cluster": np.nan,
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
        "bic_refit_cache_hit": bool(bic_refit_cache_hit),
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
        "input_data_hash": str(candidate_static.input_data_hash),
        "evaluation_mode": "ari_only" if ari_only_evaluation else "full",
        "fit_compute_summary": bool(compute_summary),
        "fit_start_mode": str(start_mode),
        "solver_state_warm_start": bool(solver_state is not None),
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
    elif lambda_min <= 0.0 or lambda_max <= 0.0:
        target_lambda = 0.0
    else:
        target_lambda = float(np.sqrt(float(lambda_min) * float(lambda_max)))

    ranked_df = tied_df.copy()
    lambda_values = ranked_df["lambda"].to_numpy(dtype=float)
    if target_lambda > 0.0:
        distances = np.full(lambda_values.shape, np.inf, dtype=float)
        positive_lambda_mask = lambda_values > 0.0
        distances[positive_lambda_mask] = np.abs(
            np.log(lambda_values[positive_lambda_mask]) - np.log(target_lambda)
        )
        ranked_df["_repr_log_distance"] = distances
    else:
        ranked_df["_repr_log_distance"] = np.abs(lambda_values - target_lambda)
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
    runtime = solver_context.runtime
    torch_data = torch_data_from_context(solver_context)
    pilot_phi: StartArray = solver_context.exact_pilot
    pooled_start: StartArray = solver_context.pooled_start
    scalar_well_starts: list[StartArray] = list(solver_context.scalar_well_starts)
    effective_graph = solver_context.graph_spec
    effective_tensor_graph = solver_context.graph
    effective_fit_options = replace(fit_options, graph=effective_graph)
    static_metadata = _candidate_static_metadata(data, effective_graph)
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
        )
        lambda_grid = list(lambda_bracket.anchors)
    simulation_truth: SimulationTruth | None = None
    if evaluate_all_candidates and simulation_root is not None and (simulation_root / data.tumor_id).exists():
        simulation_truth = load_simulation_truth(data, simulation_root)
    result_entries: list[tuple[FitResult, SimulationEvaluation | None, dict[str, float | int | str | bool]]] = []
    fit_by_lambda: dict[float, FitResult] = {}
    solver_state_by_lambda: dict[float, SolverState] = {}
    partition_labels_by_candidate_id: dict[int, np.ndarray] = {}
    bic_refit_cache: dict[str, PartitionRefitResult] = {}
    next_step = 0

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
        ari_only_evaluation: bool = False,
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
            result_entries.append((fit, evaluation, row))
            if fit.bic_partition_labels is not None:
                partition_labels_by_candidate_id[candidate_id] = np.asarray(
                    fit.bic_partition_labels,
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
        ari_only_evaluation=False,
        start_mode="warm_only" if adaptive_lambda_mode and use_warm_starts else "full",
        compute_summary=True,
    )

    if adaptive_lambda_mode and lambda_bracket is not None:
        remaining_transition_budget = max(
            ADAPTIVE_PATH_MAX_CANDIDATES - len(fit_by_lambda),
            0,
        )
        transition_probe_records = _adaptive_transition_probe_records(
            lambda_bracket,
            list(fit_by_lambda.keys()),
            max_new=min(
                ADAPTIVE_PATH_TRANSITION_PROBE_MAX_CANDIDATES,
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
                ari_only_evaluation=False,
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
        adaptive_max_rounds = ADAPTIVE_PATH_MAX_ROUNDS
        adaptive_max_candidates = ADAPTIVE_PATH_MAX_CANDIDATES
        adaptive_refine_per_round = ADAPTIVE_PATH_REFINE_PER_ROUND
        for adaptive_round in range(1, adaptive_max_rounds + 1):
            if len(fit_by_lambda) >= adaptive_max_candidates:
                adaptive_search_stop_reason = "adaptive_candidate_budget_reached"
                break
            interim_df = pd.DataFrame([row for _, _, row in result_entries])
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
                for fit, _, row in result_entries
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
            polish_phi_by_lambda: dict[float, StartArray] = {}
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
                    polish_phi_by_lambda[lambda_key] = _clone_start(_fit_phi_start(source_fit))

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
                polish_success = bool(polished_row.get("bic_selection_eligible", polished_row["selection_eligible"]))
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
            np.any(_bic_selection_eligible_mask(search_df))
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
                rescue_phi_by_lambda: dict[float, StartArray] = {}
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
                        rescue_phi_by_lambda[lambda_key] = _clone_start(_fit_phi_start(source_fit))

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
                    polish_success = bool(polished_row.get("bic_selection_eligible", polished_row["selection_eligible"]))
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
            np.any(_bic_selection_eligible_mask(search_df))
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
                full_start_phi_by_lambda: dict[float, StartArray] = {}
                for _, candidate_row in full_start_rescue_df.iterrows():
                    candidate_id = int(candidate_row["_candidate_id"])
                    source_fit = fit_by_candidate_id.get(candidate_id)
                    if source_fit is not None:
                        full_start_phi_by_lambda[_canonical_lambda(float(candidate_row["lambda"]))] = _clone_start(
                            _fit_phi_start(source_fit)
                        )

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
    search_df = _annotate_bic_diagnostics(search_df)
    num_candidates = int(search_df.shape[0])
    converged_mask = search_df["converged"].astype(bool).to_numpy(dtype=bool)
    candidate_selection_eligible_mask = _bic_selection_eligible_mask(search_df)
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
            _row_bic_selection_eligible(best_score_all_row)
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
        best_score_all_eligible = bool(_row_bic_selection_eligible(best_score_all_row))
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
