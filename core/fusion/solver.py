from __future__ import annotations

import numpy as np
import torch

from ...io.data import TumorData
from .graph import resolve_pairwise_fusion_graph
from .starts import (
    compute_exact_observed_data_pilot,
    compute_pooled_observed_data_start,
    compute_scalar_cell_wells,
    compute_scalar_well_start_bank,
)
from .torch_backend import (
    cell_terms_torch,
    em_surrogate_terms_torch,
    graph_fusion_kkt_residual_from_grad_torch,
    objective_value_torch,
    pairwise_penalty_torch,
    refine_graph_fusion_dual_certificate_torch,
    resolve_runtime,
    solve_majorized_subproblem_alm_torch,
    solve_majorized_subproblem_pdhg_torch,
    to_torch_tumor_data,
)
from .types import FusionFitArtifacts, PairwiseFusionGraph


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int64)

    def find(self, value: int) -> int:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            self.parent[parent] = self.parent[self.parent[parent]]
            parent = self.parent[parent]
        self.parent[value] = parent
        return int(parent)

    def union(self, left: int, right: int) -> None:
        root_left = self.find(int(left))
        root_right = self.find(int(right))
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            self.parent[root_left] = root_right
        elif self.rank[root_left] > self.rank[root_right]:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1


def _graph_tensors(
    graph: PairwiseFusionGraph,
    runtime,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache_key = (str(runtime.device_name), str(runtime.dtype))
    cached = graph.torch_cache.get(cache_key)
    if cached is not None:
        return cached

    tensors = (
        torch.as_tensor(graph.edge_u, dtype=torch.long, device=runtime.device),
        torch.as_tensor(graph.edge_v, dtype=torch.long, device=runtime.device),
        torch.as_tensor(graph.edge_w, dtype=runtime.dtype, device=runtime.device),
    )
    graph.torch_cache[cache_key] = tensors
    return tensors


def _cluster_labels(
    phi: np.ndarray,
    *,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    tol: float,
) -> np.ndarray:
    num_mutations = int(phi.shape[0])
    if num_mutations == 0:
        return np.zeros((0,), dtype=np.int64)

    if edge_u.size == 0:
        return np.arange(num_mutations, dtype=np.int64)

    fused = np.linalg.norm(phi[edge_u] - phi[edge_v], axis=1) <= float(tol)
    if not np.any(fused):
        return np.arange(num_mutations, dtype=np.int64)

    uf = _UnionFind(num_mutations)
    for left, right in zip(edge_u[fused], edge_v[fused]):
        uf.union(int(left), int(right))

    labels = np.empty(num_mutations, dtype=np.int64)
    root_to_label: dict[int, int] = {}
    next_label = 0
    for idx in range(num_mutations):
        root = uf.find(idx)
        label = root_to_label.get(root)
        if label is None:
            label = next_label
            root_to_label[root] = label
            next_label += 1
        labels[idx] = int(label)
    return labels


def _cluster_summary_from_labels(
    phi: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    centers = np.zeros((n_clusters, phi.shape[1]), dtype=phi.dtype)
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    np.add.at(centers, labels, phi)
    centers /= np.clip(counts[:, None], 1.0, None)
    phi_clustered = centers[labels]
    return centers.astype(phi.dtype, copy=False), phi_clustered.astype(phi.dtype, copy=False)


def _deduplicate_starts(starts: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for start in starts:
        start_arr = np.asarray(start)
        signature = np.round(start_arr, decimals=8).astype(np.float32, copy=False).tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(start_arr)
    return unique


def _inner_model_value_torch(
    phi: torch.Tensor,
    *,
    U: torch.Tensor,
    h: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:
    quad = 0.5 * torch.sum(h * torch.square(phi - U))
    penalty = pairwise_penalty_torch(
        phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
    )
    return quad + penalty


def _update_minimum(current: float, candidate: float) -> float:
    if not np.isfinite(candidate):
        return float(current)
    if not np.isfinite(current):
        return float(candidate)
    return float(min(current, candidate))


def _fit_from_start(
    data: TumorData,
    *,
    torch_data,
    runtime,
    graph: PairwiseFusionGraph,
    lambda_value: float,
    major_prior: float,
    eps: float,
    outer_max_iter: int,
    inner_max_iter: int,
    tol: float,
    phi_start: np.ndarray,
    summary_tol: float | None,
    compute_summary: bool,
    verbose: bool,
) -> FusionFitArtifacts:
    edge_u_np = graph.edge_u
    edge_v_np = graph.edge_v
    complete_edge_count = data.num_mutations * max(data.num_mutations - 1, 0) // 2
    use_alm = (
        edge_u_np.size == complete_edge_count
        and int(graph.degree_bound) == max(int(data.num_mutations) - 1, 1)
    )
    edge_u, edge_v, edge_w = _graph_tensors(graph, runtime)

    lower = torch.full_like(torch_data.phi_upper, float(eps))
    upper = torch.minimum(torch_data.phi_upper, torch.ones_like(torch_data.phi_upper))
    phi = torch.as_tensor(np.asarray(phi_start), dtype=runtime.dtype, device=runtime.device)
    phi = torch.minimum(torch.maximum(phi, lower), upper)

    dual = None
    dual_kkt = None
    history: list[float] = []
    converged = False
    converged_inner = False
    converged_outer = False
    iterations = 0
    current_inner_converged = False
    current_inner_kkt_residual = np.nan
    final_relative_objective_change = np.inf
    final_step_residual = np.inf
    final_inner_kkt_residual = np.nan
    final_outer_diag = {
        "stationarity_residual": np.inf,
        "edge_subgradient_residual": np.inf,
        "dual_ball_residual": np.inf,
        "box_residual": np.inf,
        "kkt_residual": np.inf,
    }
    outer_kkt_certificate_status = "not_audited"
    outer_kkt_dual_refined = False
    outer_kkt_fused_edges = 0
    outer_kkt_nonzero_edges = 0
    outer_stationarity_residual_before_dual_refine = np.inf
    outer_stationarity_residual_after_dual_refine = np.inf
    accepted_outer_steps = 0
    accepted_full_steps = 0
    accepted_damped_steps = 0
    attempted_outer_steps = 0
    failed_majorization_checks = 0
    failed_inner_model_checks = 0
    failed_em_envelope_checks = 0
    failed_descent_checks = 0
    failed_nonfinite_checks = 0
    mm_consistency_violations = 0
    last_attempted_inner_kkt_residual = np.nan
    best_attempted_inner_kkt_residual = np.nan
    last_attempted_objective_gap = np.nan
    best_attempted_objective_gap = np.nan
    last_attempted_surrogate_gap = np.nan
    best_attempted_surrogate_gap = np.nan
    last_attempted_inner_model_gap = np.nan
    best_attempted_inner_model_gap = np.nan
    last_attempted_em_envelope_gap = np.nan
    best_attempted_em_envelope_gap = np.nan
    accepted_step_type = "none"
    last_reject_reason = "not_attempted"
    failure_reason = "not_converged"

    fit_loss, penalty, objective, gamma_major = objective_value_torch(
        torch_data,
        phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        major_prior=major_prior,
        eps=eps,
    )
    history.append(float(objective))

    for outer_iter in range(max(int(outer_max_iter), 1)):
        iterations = outer_iter + 1
        previous_phi = phi.clone()
        previous_objective = objective
        surrogate_terms = em_surrogate_terms_torch(
            torch_data,
            phi,
            omega_major=gamma_major,
            major_prior=major_prior,
            eps=eps,
        )
        surrogate_fit_loss = float(torch.sum(surrogate_terms.loss).item())
        h_base = torch.clamp(surrogate_terms.hess_upper, min=1e-6)
        scale = 1.0
        accepted = False
        candidate_phi = phi
        candidate_dual = dual
        candidate_dual_kkt = dual_kkt
        candidate_objective = objective
        candidate_fit_loss = fit_loss
        candidate_gamma = gamma_major
        candidate_inner_residual = np.nan
        candidate_step_type = "none"
        inner_converged = False

        for _ in range(10):
            h = h_base * scale
            U = phi - surrogate_terms.grad / h
            q_current = _inner_model_value_torch(
                phi,
                U=U,
                h=h,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                lambda_value=lambda_value,
            )
            if use_alm:
                phi_trial, dual_trial, dual_kkt_trial, _, inner_ok, inner_residual = solve_majorized_subproblem_alm_torch(
                    runtime=runtime,
                    num_mutations=data.num_mutations,
                    U=U,
                    h=h,
                    lower=lower,
                    upper=upper,
                    lambda_value=lambda_value,
                    edge_u=edge_u,
                    edge_v=edge_v,
                    edge_w=edge_w,
                    tol=max(tol, 1e-6),
                    max_iter=max(inner_max_iter, 10),
                    phi_start=phi,
                    dual_start=dual,
                )
            else:
                phi_trial, dual_trial, dual_kkt_trial, _, inner_ok, inner_residual = solve_majorized_subproblem_pdhg_torch(
                    runtime=runtime,
                    num_mutations=data.num_mutations,
                    U=U,
                    h=h,
                    lower=lower,
                    upper=upper,
                    lambda_value=lambda_value,
                    edge_u=edge_u,
                    edge_v=edge_v,
                    edge_w=edge_w,
                    degree_bound=int(graph.degree_bound),
                    tol=max(tol, 1e-6),
                    max_iter=max(inner_max_iter, 10),
                    phi_start=phi,
                    dual_start=dual,
                )
            attempted_outer_steps += 1
            last_attempted_inner_kkt_residual = float(inner_residual)
            best_attempted_inner_kkt_residual = _update_minimum(
                float(best_attempted_inner_kkt_residual),
                float(inner_residual),
            )
            delta = phi_trial - phi
            quadratic_gap = float(torch.sum(surrogate_terms.grad * delta + 0.5 * h * torch.square(delta)).item())
            majorizer_rhs = surrogate_fit_loss + quadratic_gap
            q_trial = _inner_model_value_torch(
                phi_trial,
                U=U,
                h=h,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                lambda_value=lambda_value,
            )
            inner_model_gap = float((q_trial - q_current).item())
            trial_surrogate_terms = em_surrogate_terms_torch(
                torch_data,
                phi_trial,
                omega_major=gamma_major,
                major_prior=major_prior,
                eps=eps,
            )
            trial_surrogate_loss = float(torch.sum(trial_surrogate_terms.loss).item())
            surrogate_gap = float(trial_surrogate_loss - majorizer_rhs)
            trial_fit_loss, _, trial_objective, trial_gamma = objective_value_torch(
                torch_data,
                phi_trial,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                lambda_value=lambda_value,
                major_prior=major_prior,
                eps=eps,
            )
            objective_gap = float(trial_objective - previous_objective)
            em_envelope_gap = float((trial_fit_loss - fit_loss) - (trial_surrogate_loss - surrogate_fit_loss))
            last_attempted_objective_gap = objective_gap
            best_attempted_objective_gap = _update_minimum(float(best_attempted_objective_gap), objective_gap)
            last_attempted_surrogate_gap = surrogate_gap
            best_attempted_surrogate_gap = _update_minimum(float(best_attempted_surrogate_gap), surrogate_gap)
            last_attempted_inner_model_gap = inner_model_gap
            best_attempted_inner_model_gap = _update_minimum(float(best_attempted_inner_model_gap), inner_model_gap)
            last_attempted_em_envelope_gap = em_envelope_gap
            best_attempted_em_envelope_gap = _update_minimum(float(best_attempted_em_envelope_gap), em_envelope_gap)
            finite_attempt = all(
                np.isfinite(value)
                for value in [
                    inner_model_gap,
                    surrogate_gap,
                    em_envelope_gap,
                    objective_gap,
                    trial_fit_loss,
                    trial_objective,
                ]
            )
            inner_model_tol = 1e-8 * (1.0 + abs(float(q_current.item())))
            majorization_tol = 1e-8 * (1.0 + abs(surrogate_fit_loss))
            envelope_tol = 1e-8 * (1.0 + abs(fit_loss))
            objective_tol = 1e-8 * (1.0 + abs(previous_objective))
            if not finite_attempt:
                failed_nonfinite_checks += 1
                last_reject_reason = "rejected_nonfinite_objective"
                scale *= 2.0
                continue
            if inner_model_gap > inner_model_tol:
                failed_inner_model_checks += 1
                last_reject_reason = "rejected_inner_model_not_decreased"
                scale *= 2.0
                continue
            if surrogate_gap > majorization_tol:
                failed_majorization_checks += 1
                last_reject_reason = "rejected_majorization_failed"
                scale *= 2.0
                continue
            if em_envelope_gap > envelope_tol:
                failed_em_envelope_checks += 1
                last_reject_reason = "rejected_em_envelope_failed"
                scale *= 2.0
                continue
            if objective_gap <= objective_tol:
                accepted = True
                accepted_outer_steps += 1
                accepted_full_steps += 1
                candidate_phi = phi_trial
                candidate_dual = dual_trial
                candidate_dual_kkt = dual_kkt_trial
                candidate_objective = trial_objective
                candidate_fit_loss = trial_fit_loss
                candidate_gamma = trial_gamma
                candidate_inner_residual = float(inner_residual)
                candidate_step_type = "full_inner_step"
                accepted_step_type = candidate_step_type
                inner_converged = inner_ok and inner_residual <= 5.0 * tol
                break
            if inner_ok and inner_model_gap <= inner_model_tol and surrogate_gap <= majorization_tol and em_envelope_gap <= envelope_tol and objective_gap > max(1e-5, objective_tol):
                mm_consistency_violations += 1
            theta = 0.5
            damped_accepted = False
            for _line_search_iter in range(12):
                phi_theta = phi + theta * delta
                theta_fit_loss, _, theta_objective, theta_gamma = objective_value_torch(
                    torch_data,
                    phi_theta,
                    edge_u=edge_u,
                    edge_v=edge_v,
                    edge_w=edge_w,
                    lambda_value=lambda_value,
                    major_prior=major_prior,
                    eps=eps,
                )
                if np.isfinite(theta_objective) and theta_objective <= previous_objective + objective_tol:
                    accepted = True
                    damped_accepted = True
                    accepted_outer_steps += 1
                    accepted_damped_steps += 1
                    candidate_phi = phi_theta
                    candidate_dual = dual_trial
                    candidate_dual_kkt = None
                    candidate_objective = theta_objective
                    candidate_fit_loss = theta_fit_loss
                    candidate_gamma = theta_gamma
                    candidate_inner_residual = np.nan
                    candidate_step_type = "damped_mm_direction"
                    accepted_step_type = candidate_step_type
                    inner_converged = False
                    break
                theta *= 0.5
            if damped_accepted:
                break
            failed_descent_checks += 1
            last_reject_reason = "rejected_exact_descent_failed"
            scale *= 2.0

        if not accepted:
            candidate_phi = phi
            candidate_dual = dual
            candidate_dual_kkt = dual_kkt
            candidate_objective = objective
            candidate_fit_loss = fit_loss
            candidate_gamma = gamma_major
            candidate_step_type = "none"

        phi = candidate_phi
        dual = candidate_dual
        dual_kkt = candidate_dual_kkt
        objective = candidate_objective
        fit_loss = candidate_fit_loss
        gamma_major = candidate_gamma
        penalty = objective - fit_loss
        history.append(float(objective))

        if verbose:
            print(
                f"[pairwise-fusion:{runtime.device_name}] iter={iterations:02d} objective={objective:.6f} "
                f"fit={fit_loss:.6f} penalty={penalty:.6f}"
            )

        rel_change = abs(previous_objective - objective) / (1.0 + abs(previous_objective))
        step_residual = float((torch.linalg.norm(phi - previous_phi) / (1.0 + torch.linalg.norm(previous_phi))).item())
        outer_terms = cell_terms_torch(torch_data, phi, major_prior=major_prior, eps=eps)
        if dual_kkt is None:
            dual_for_audit = torch.zeros((int(edge_u.numel()), int(phi.shape[1])), dtype=runtime.dtype, device=runtime.device)
        else:
            dual_for_audit = dual_kkt
        outer_diag = graph_fusion_kkt_residual_from_grad_torch(
            phi=phi,
            grad_smooth=outer_terms.grad,
            dual_kkt=dual_for_audit,
            lower=lower,
            upper=upper,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            lambda_value=lambda_value,
            atol=max(tol, 1e-8),
        )
        outer_converged = bool(outer_diag["kkt_residual"] <= 5.0 * tol)
        final_relative_objective_change = float(rel_change)
        final_step_residual = float(step_residual)
        if accepted:
            current_inner_converged = bool(inner_converged)
            current_inner_kkt_residual = float(candidate_inner_residual)
        final_inner_kkt_residual = float(current_inner_kkt_residual)
        final_outer_diag = outer_diag
        converged_inner = bool(current_inner_converged)
        converged_outer = bool(outer_converged)
        if (
            rel_change <= tol
            and step_residual <= np.sqrt(max(tol, 1e-10))
            and current_inner_converged
            and outer_converged
        ):
            converged = True
            break

    final_terms = cell_terms_torch(torch_data, phi, major_prior=major_prior, eps=eps)
    final_dual_audit = refine_graph_fusion_dual_certificate_torch(
        phi=phi,
        grad_smooth=final_terms.grad,
        dual_kkt=dual_kkt,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        atol=max(tol, 1e-8),
        max_iter=64,
    )
    final_outer_diag = final_dual_audit["diag"]
    outer_kkt_certificate_status = str(final_dual_audit["status"])
    outer_kkt_dual_refined = bool(final_dual_audit["dual_refined"])
    outer_kkt_fused_edges = int(final_dual_audit["fused_edges"])
    outer_kkt_nonzero_edges = int(final_dual_audit["nonzero_edges"])
    outer_stationarity_residual_before_dual_refine = float(final_dual_audit["stationarity_before"])
    outer_stationarity_residual_after_dual_refine = float(final_dual_audit["stationarity_after"])
    converged_outer = bool(float(final_outer_diag["kkt_residual"]) <= 5.0 * tol)
    valid_dual_certificate = outer_kkt_certificate_status in {
        "zero_penalty_no_dual_needed",
        "analytic_nonfused_dual",
        "refined_fused_edge_dual",
    }
    selection_eligible = bool(
        np.isfinite(float(objective))
        and converged_outer
        and valid_dual_certificate
        and mm_consistency_violations == 0
    )

    if converged:
        failure_reason = "converged"
    elif selection_eligible and accepted_outer_steps == 0:
        failure_reason = "start_already_stationary"
    elif selection_eligible and accepted_damped_steps > 0 and accepted_full_steps == 0:
        failure_reason = "fixed_objective_kkt_certified_after_damped_steps"
    elif selection_eligible:
        failure_reason = "fixed_objective_kkt_certified"
    elif mm_consistency_violations > 0:
        failure_reason = "mm_consistency_violation"
    elif accepted_outer_steps == 0:
        final_penalty = float(objective - fit_loss)
        if final_penalty <= 1e-8 and final_outer_diag["kkt_residual"] > 5.0 * tol:
            failure_reason = "pooled_start_not_stationary_no_descent_step_found"
        elif failed_nonfinite_checks > 0 and failed_nonfinite_checks >= attempted_outer_steps:
            failure_reason = "all_trials_nonfinite"
        elif failed_inner_model_checks > 0 and failed_inner_model_checks >= max(attempted_outer_steps - failed_nonfinite_checks, 1):
            failure_reason = "all_trials_failed_inner_model_decrease"
        elif failed_majorization_checks > 0 and failed_descent_checks == 0 and failed_em_envelope_checks == 0:
            failure_reason = "all_trials_failed_majorization"
        elif failed_em_envelope_checks > 0 and failed_descent_checks == 0:
            failure_reason = "all_trials_failed_em_envelope"
        elif failed_descent_checks > 0:
            failure_reason = "all_trials_failed_exact_descent"
        else:
            failure_reason = "no_accepted_outer_step"
    elif accepted_damped_steps > 0 and accepted_full_steps == 0:
        failure_reason = "only_damped_steps_accepted"
    elif not converged_outer:
        failure_reason = "outer_stationarity_residual_above_tolerance"
    elif not converged_inner:
        failure_reason = "inner_kkt_residual_above_tolerance"
    else:
        failure_reason = "outer_stopping_criteria_not_met"

    phi_np = phi.detach().cpu().numpy()
    gamma_np = gamma_major.detach().cpu().numpy()
    effective_summary_tol = (
        max(10.0 * float(tol), 1e-4)
        if summary_tol is None
        else max(float(summary_tol), 1e-12)
    )
    cluster_labels = _cluster_labels(
        phi_np,
        edge_u=edge_u_np,
        edge_v=edge_v_np,
        tol=effective_summary_tol,
    )
    n_clusters = int(cluster_labels.max()) + 1 if cluster_labels.size else 0
    if compute_summary:
        cluster_centers, phi_clustered = _cluster_summary_from_labels(phi_np, cluster_labels)
        phi_clustered_torch = torch.as_tensor(phi_clustered, dtype=runtime.dtype, device=runtime.device)
        summary_fit_loss, _, _, _ = objective_value_torch(
            torch_data,
            phi_clustered_torch,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            lambda_value=0.0,
            major_prior=major_prior,
            eps=eps,
        )
        summary_loglik = float(-summary_fit_loss)
    else:
        cluster_centers = np.zeros((n_clusters, phi_np.shape[1]), dtype=phi_np.dtype)
        phi_clustered = phi_np.astype(phi_np.dtype, copy=False)
        summary_loglik = float("nan")

    major_probability = np.where(
        data.multiplicity_estimation_mask,
        gamma_np,
        1.0,
    ).astype(phi_np.dtype, copy=False)
    major_call = major_probability >= 0.5
    multiplicity_call = np.where(
        data.multiplicity_estimation_mask,
        np.where(major_call, data.major_cn, data.minor_cn),
        data.fixed_multiplicity,
    ).astype(phi_np.dtype, copy=False)

    return FusionFitArtifacts(
        phi=phi_np.astype(phi_np.dtype, copy=False),
        phi_clustered=phi_clustered.astype(phi_np.dtype, copy=False),
        cluster_labels=cluster_labels.astype(np.int64, copy=False),
        cluster_centers=cluster_centers.astype(phi_np.dtype, copy=False),
        gamma_major=major_probability.astype(phi_np.dtype, copy=False),
        major_probability=major_probability.astype(phi_np.dtype, copy=False),
        major_call=major_call.astype(bool, copy=False),
        multiplicity_call=multiplicity_call.astype(phi_np.dtype, copy=False),
        multiplicity_estimated_mask=data.multiplicity_estimation_mask.astype(bool, copy=False),
        loglik=float(-fit_loss),
        summary_loglik=summary_loglik,
        penalized_objective=float(objective),
        lambda_value=float(lambda_value),
        n_clusters=n_clusters,
        iterations=int(iterations),
        converged=bool(converged),
        device=runtime.device_name,
        dtype=str(runtime.dtype).replace("torch.", ""),
        graph_name=str(graph.name),
        summary_tol=float(effective_summary_tol),
        history=[float(value) for value in history],
        inner_kkt_residual=float(final_inner_kkt_residual),
        accepted_inner_kkt_residual=float(final_inner_kkt_residual),
        last_attempted_inner_kkt_residual=float(last_attempted_inner_kkt_residual),
        best_attempted_inner_kkt_residual=float(best_attempted_inner_kkt_residual),
        last_attempted_objective_gap=float(last_attempted_objective_gap),
        best_attempted_objective_gap=float(best_attempted_objective_gap),
        last_attempted_surrogate_gap=float(last_attempted_surrogate_gap),
        best_attempted_surrogate_gap=float(best_attempted_surrogate_gap),
        last_attempted_inner_model_gap=float(last_attempted_inner_model_gap),
        best_attempted_inner_model_gap=float(best_attempted_inner_model_gap),
        last_attempted_em_envelope_gap=float(last_attempted_em_envelope_gap),
        best_attempted_em_envelope_gap=float(best_attempted_em_envelope_gap),
        outer_stationarity_residual=float(final_outer_diag["stationarity_residual"]),
        outer_edge_subgradient_residual=float(final_outer_diag["edge_subgradient_residual"]),
        outer_dual_ball_residual=float(final_outer_diag["dual_ball_residual"]),
        outer_box_residual=float(final_outer_diag["box_residual"]),
        fixed_objective_kkt_residual=float(final_outer_diag["kkt_residual"]),
        outer_kkt_certificate_status=str(outer_kkt_certificate_status),
        outer_kkt_dual_refined=bool(outer_kkt_dual_refined),
        outer_kkt_fused_edges=int(outer_kkt_fused_edges),
        outer_kkt_nonzero_edges=int(outer_kkt_nonzero_edges),
        outer_stationarity_residual_before_dual_refine=float(outer_stationarity_residual_before_dual_refine),
        outer_stationarity_residual_after_dual_refine=float(outer_stationarity_residual_after_dual_refine),
        converged_inner=bool(converged_inner),
        converged_outer=bool(converged_outer),
        final_relative_objective_change=float(final_relative_objective_change),
        final_step_residual=float(final_step_residual),
        accepted_outer_steps=int(accepted_outer_steps),
        accepted_full_steps=int(accepted_full_steps),
        accepted_damped_steps=int(accepted_damped_steps),
        attempted_outer_steps=int(attempted_outer_steps),
        failed_majorization_checks=int(failed_majorization_checks),
        failed_inner_model_checks=int(failed_inner_model_checks),
        failed_em_envelope_checks=int(failed_em_envelope_checks),
        failed_descent_checks=int(failed_descent_checks),
        failed_nonfinite_checks=int(failed_nonfinite_checks),
        mm_consistency_violations=int(mm_consistency_violations),
        accepted_step_type=str(accepted_step_type),
        last_reject_reason=str(last_reject_reason),
        failure_reason=str(failure_reason),
        selection_eligible=bool(selection_eligible),
    )


def fit_observed_data_pairwise_fusion(
    data: TumorData,
    *,
    lambda_value: float,
    major_prior: float,
    eps: float,
    outer_max_iter: int,
    inner_max_iter: int,
    tol: float,
    phi_start: np.ndarray | None = None,
    graph: PairwiseFusionGraph | None = None,
    adaptive_weight_gamma: float = 1.0,
    adaptive_weight_floor: float = 1e-6,
    adaptive_weight_baseline: float = 1.0,
    exact_pilot: np.ndarray | None = None,
    pooled_start: np.ndarray | None = None,
    scalar_well_starts: list[np.ndarray] | None = None,
    start_mode: str = "full",
    device: str | None = "auto",
    dtype: str | None = "auto",
    summary_tol: float | None = None,
    runtime=None,
    torch_data=None,
    compute_summary: bool = True,
    verbose: bool = False,
) -> FusionFitArtifacts:
    effective_runtime = resolve_runtime(device, dtype=dtype) if runtime is None else runtime
    effective_torch_data = to_torch_tumor_data(data, effective_runtime) if torch_data is None else torch_data

    if exact_pilot is None:
        exact_pilot, secondary_wells, valid_secondary = compute_scalar_cell_wells(
            data,
            major_prior=major_prior,
            eps=eps,
            tol=max(tol, 1e-6),
            max_iter=max(inner_max_iter, 16),
        )
    else:
        secondary_wells = None
        valid_secondary = None
    effective_graph = resolve_pairwise_fusion_graph(
        data.num_mutations,
        graph=graph,
        pilot_phi=exact_pilot,
        gamma=float(adaptive_weight_gamma),
        tau=max(float(adaptive_weight_floor), float(eps)),
        baseline=float(adaptive_weight_baseline),
    )
    if pooled_start is None:
        pooled_start = compute_pooled_observed_data_start(
            data,
            runtime=effective_runtime,
            major_prior=major_prior,
            eps=eps,
            tol=max(tol, 1e-6),
            max_iter=max(inner_max_iter, 16),
            beta_hints=exact_pilot,
        )
    if scalar_well_starts is None:
        scalar_well_starts = compute_scalar_well_start_bank(
            data,
            major_prior=major_prior,
            eps=eps,
            tol=max(tol, 1e-6),
            max_iter=max(inner_max_iter, 16),
            exact_pilot=exact_pilot,
            secondary_wells=secondary_wells,
            valid_secondary=valid_secondary,
        )
    else:
        scalar_well_starts = [np.asarray(start) for start in scalar_well_starts]

    normalized_start_mode = str(start_mode).strip().lower()
    if normalized_start_mode not in {"full", "warm_plus_pilot", "warm_only"}:
        raise ValueError(f"Unknown start_mode: {start_mode}")

    start_bank: list[np.ndarray] = []
    if phi_start is not None:
        start_bank.append(np.asarray(phi_start))
    if normalized_start_mode == "full":
        start_bank.extend(scalar_well_starts)
        start_bank.append(pooled_start)
    elif normalized_start_mode == "warm_plus_pilot":
        if phi_start is None:
            start_bank.extend(scalar_well_starts)
            start_bank.append(pooled_start)
        else:
            start_bank.extend(scalar_well_starts)
    elif phi_start is None:
        start_bank.append(exact_pilot)
    start_bank = _deduplicate_starts(start_bank)

    best_artifacts: FusionFitArtifacts | None = None
    for start in start_bank:
        artifacts = _fit_from_start(
            data,
            torch_data=effective_torch_data,
            runtime=effective_runtime,
            graph=effective_graph,
            lambda_value=lambda_value,
            major_prior=major_prior,
            eps=eps,
            outer_max_iter=outer_max_iter,
            inner_max_iter=inner_max_iter,
            tol=tol,
            phi_start=start,
            summary_tol=summary_tol,
            compute_summary=compute_summary,
            verbose=verbose,
        )
        if best_artifacts is None:
            best_artifacts = artifacts
            continue
        if artifacts.converged and not best_artifacts.converged:
            best_artifacts = artifacts
            continue
        if (
            artifacts.converged == best_artifacts.converged
            and artifacts.penalized_objective < best_artifacts.penalized_objective - 1e-8
        ):
            best_artifacts = artifacts

    if best_artifacts is None:
        raise RuntimeError("No valid start produced a fusion fit.")
    return best_artifacts
