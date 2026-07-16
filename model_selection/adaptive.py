from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from ..core.model import FitOptions
from ..core.fusion.graph_ops import graph_adjoint_edges, graph_forward_edges, project_dual_ball
from ..core.fusion.torch_backend import (
    as_runtime_tensor,
    mutation_region_terms_torch,
    objective_value_torch,
    stationarity_residual_torch,
)
from .config import (
    ADAPTIVE_FIRST_PASS_INNER_MAX_ITER,
    ADAPTIVE_FIRST_PASS_OUTER_MAX_ITER,
    ADAPTIVE_PATH_FULL_FUSION_MAX_ITER,
    ADAPTIVE_PATH_LOG10_WIDTH_TOL,
    ADAPTIVE_PATH_VALUE_CURVE_TOL,
    LAMBDA_SEARCH_MAX,
    LAMBDA_SEARCH_MIN,
)
from .partitions import _partition_is_coarsening
from .scoring import (
    _add_bic_selection_eligible,
    _bic_selection_eligible_mask,
    _canonical_lambda,
    _row_bic_selection_eligible,
    _sorted_unique_lambdas,
)
from .types import FullFusionKKTResult, StartArray, _AdaptiveIntervalProposal
from ..core.bic import LambdaBracket

def _runtime_start_tensor(start: StartArray, runtime) -> torch.Tensor:
    return as_runtime_tensor(start, runtime)


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
    terms = mutation_region_terms_torch(torch_data, phi, major_prior=major_prior, eps=eps)
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
    sparse_anchors: bool = False,
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
    lambda_min_scale = 64.0 if sparse_anchors else 128.0
    lambda_min = max(float(lambda_eq) / lambda_min_scale, LAMBDA_SEARCH_MIN)
    lambda_anchor_max = min(max(float(lambda_full), float(lambda_eq) * 4.0), LAMBDA_SEARCH_MAX)
    if sparse_anchors:
        raw_anchors = [
            0.0,
            lambda_eq / 64.0,
            lambda_eq / 16.0,
            lambda_eq / 4.0,
            lambda_eq,
            lambda_eq * 4.0,
            lambda_full,
        ]
    else:
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
        "lambda_sparse_initial_anchors": float(sparse_anchors),
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
    if normalized_score == "extended_bic":
        return "extended_bic"
    if normalized_score == "partition_icl":
        return "partition_icl"
    raise ValueError(f"Unknown normalized selection score: {normalized_score}")


def _score_maximized(normalized_score: str) -> bool:
    del normalized_score
    return False


def _adaptive_first_pass_options(base_options: FitOptions) -> FitOptions:
    return replace(
        base_options,
        outer_max_iter=max(int(base_options.outer_max_iter), ADAPTIVE_FIRST_PASS_OUTER_MAX_ITER),
        inner_max_iter=max(int(base_options.inner_max_iter), ADAPTIVE_FIRST_PASS_INNER_MAX_ITER),
    )


def _score_strictly_better(score: float, reference: float, *, normalized_score: str) -> bool:
    if not np.isfinite(score) or not np.isfinite(reference):
        return False
    margin = 1e-8 * (1.0 + abs(float(reference)))
    if _score_maximized(normalized_score):
        return bool(float(score) > float(reference) + margin)
    return bool(float(score) < float(reference) - margin)


def _best_candidate_rows_by_lambda(
    search_df: pd.DataFrame,
    *,
    normalized_score: str = "bic",
) -> pd.DataFrame:
    if search_df.empty:
        return search_df.copy()
    ranked = _add_bic_selection_eligible(search_df)
    if "bic_refit_finite_candidate_found" not in ranked.columns and "bic_refit_converged" in ranked.columns:
        ranked["bic_refit_finite_candidate_found"] = ranked["bic_refit_converged"].astype(bool)
    if "bic_refit_converged" not in ranked.columns and "bic_refit_finite_candidate_found" in ranked.columns:
        ranked["bic_refit_converged"] = ranked["bic_refit_finite_candidate_found"].astype(bool)
    ranked["_lambda_key"] = np.round(ranked["lambda"].to_numpy(dtype=float), 12)
    score_column = _adaptive_score_column(normalized_score)
    defaults = {
        "selection_eligible": False,
        "bic_refit_finite_candidate_found": False,
        "bic_refit_converged": False,
        "converged": False,
        score_column: np.inf,
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
        "bic_refit_finite_candidate_found",
        "bic_refit_converged",
        score_column,
        "penalized_objective",
        "selection_step",
    ]
    ascending = [True, False, False, False, False, True, True, True]
    ranked = ranked.sort_values(sort_columns, ascending=ascending)
    return ranked.drop_duplicates("_lambda_key", keep="first").drop(columns=["_lambda_key"]).reset_index(drop=True)


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
    path_df = _best_candidate_rows_by_lambda(
        search_df,
        normalized_score=normalized_score,
    ).sort_values("lambda").reset_index(drop=True)
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
        near_kkt_tol = float(max(20.0 * float(tol), 2e-3))
        left_ok = _row_bic_selection_eligible(left)
        right_ok = _row_bic_selection_eligible(right)
        left_kkt = float(left.get("fixed_objective_kkt_residual", np.inf))
        right_kkt = float(right.get("fixed_objective_kkt_residual", np.inf))
        kkt_risk = (
            1.0
            if (
                not left_ok
                or not right_ok
                or min(left_kkt, right_kkt) <= near_kkt_tol
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
                reason = "failed_kkt_near_best_score"
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
    normalized_score: str = "bic",
) -> tuple[float | None, float | None, float | None]:
    if search_df.empty or "_candidate_id" not in search_df.columns:
        return None, None, None
    selected = search_df.loc[search_df["_candidate_id"].astype(int) == int(selected_candidate_id)]
    if selected.empty:
        return None, None, None
    selected_row = selected.iloc[0]
    if "lambda_applicable" in selected_row and not bool(selected_row.get("lambda_applicable", True)):
        return None, None, None
    selected_lambda = float(selected_row["lambda"])
    signature = str(selected_row.get("partition_signature", ""))
    eligible = search_df.loc[_bic_selection_eligible_mask(search_df)].copy()
    if eligible.empty:
        eligible = search_df.copy()
    eligible = _best_candidate_rows_by_lambda(
        eligible,
        normalized_score=normalized_score,
    ).sort_values("lambda").reset_index(drop=True)
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
