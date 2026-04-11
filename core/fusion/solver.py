from __future__ import annotations

import numpy as np
import torch

from ...io.data import TumorData
from .graph import coerce_graph
from .starts import compute_exact_observed_data_pilot, compute_pooled_observed_data_start
from .torch_backend import (
    cell_terms_torch,
    objective_value_torch,
    resolve_runtime,
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


def _cluster_summary(
    phi: np.ndarray,
    *,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_mutations = int(phi.shape[0])
    if num_mutations == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, phi.shape[1]), dtype=np.float32),
            np.zeros_like(phi, dtype=np.float32),
        )

    union_find = _UnionFind(num_mutations)
    if edge_u.size > 0:
        fused = np.linalg.norm(phi[edge_u] - phi[edge_v], axis=1) <= float(tol)
        for left, right in zip(edge_u[fused], edge_v[fused]):
            union_find.union(int(left), int(right))

    roots = np.asarray([union_find.find(i) for i in range(num_mutations)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    labels = labels.astype(np.int64, copy=False)
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    centers = np.zeros((n_clusters, phi.shape[1]), dtype=np.float64)
    counts = np.bincount(labels, minlength=n_clusters).astype(np.float64)
    np.add.at(centers, labels, phi)
    centers /= np.clip(counts[:, None], 1.0, None)
    phi_clustered = centers[labels]
    return labels, centers.astype(np.float32, copy=False), phi_clustered.astype(np.float32, copy=False)


def _deduplicate_starts(starts: list[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for start in starts:
        start_arr = np.asarray(start, dtype=np.float64)
        if any(np.allclose(start_arr, existing, rtol=1e-7, atol=1e-8) for existing in unique):
            continue
        unique.append(start_arr)
    return unique


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
    verbose: bool,
) -> FusionFitArtifacts:
    edge_u_np = graph.edge_u
    edge_v_np = graph.edge_v
    edge_w_np = graph.edge_w
    edge_u = torch.as_tensor(edge_u_np, dtype=torch.long, device=runtime.device)
    edge_v = torch.as_tensor(edge_v_np, dtype=torch.long, device=runtime.device)
    edge_w = torch.as_tensor(edge_w_np, dtype=runtime.dtype, device=runtime.device)

    lower = torch.full_like(torch_data.phi_upper, float(eps))
    upper = torch_data.phi_upper
    phi = torch.as_tensor(np.asarray(phi_start, dtype=np.float64), dtype=runtime.dtype, device=runtime.device)
    phi = torch.minimum(torch.maximum(phi, lower), upper)

    dual = None
    history: list[float] = []
    converged = False
    iterations = 0

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
        terms = cell_terms_torch(torch_data, phi, major_prior=major_prior, eps=eps)
        h_base = torch.clamp(terms.hess_upper, min=1e-6)
        scale = 1.0
        accepted = False
        candidate_phi = phi
        candidate_dual = dual
        candidate_objective = objective
        candidate_fit_loss = fit_loss
        candidate_gamma = gamma_major
        inner_converged = False

        for _ in range(10):
            h = h_base * scale
            U = phi - terms.grad / h
            phi_trial, dual_trial, _, inner_ok, inner_residual = solve_majorized_subproblem_pdhg_torch(
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
            delta = phi_trial - phi
            majorizer_rhs = fit_loss + float(torch.sum(terms.grad * delta + 0.5 * h * torch.square(delta)).item())
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
            if trial_fit_loss <= majorizer_rhs + 1e-8:
                accepted = True
                candidate_phi = phi_trial
                candidate_dual = dual_trial
                candidate_objective = trial_objective
                candidate_fit_loss = trial_fit_loss
                candidate_gamma = trial_gamma
                inner_converged = inner_ok and inner_residual <= 5.0 * tol
                break
            scale *= 2.0

        if not accepted:
            candidate_phi = phi
            candidate_dual = dual
            candidate_objective = objective
            candidate_fit_loss = fit_loss
            candidate_gamma = gamma_major

        phi = candidate_phi
        dual = candidate_dual
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
        if rel_change <= tol and step_residual <= np.sqrt(max(tol, 1e-10)) and accepted and inner_converged:
            converged = True
            break

    phi_np = phi.detach().cpu().numpy().astype(np.float64, copy=False)
    gamma_np = gamma_major.detach().cpu().numpy().astype(np.float64, copy=False)
    summary_tol = max(10.0 * float(tol), 1e-4)
    cluster_labels, cluster_centers, phi_clustered = _cluster_summary(
        phi_np,
        edge_u=edge_u_np,
        edge_v=edge_v_np,
        tol=summary_tol,
    )
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

    major_probability = np.where(
        data.multiplicity_estimation_mask,
        gamma_np,
        1.0,
    ).astype(np.float32, copy=False)
    major_call = major_probability >= 0.5
    multiplicity_call = np.where(
        data.multiplicity_estimation_mask,
        np.where(major_call, data.major_cn, data.minor_cn),
        data.fixed_multiplicity,
    ).astype(np.float32, copy=False)

    return FusionFitArtifacts(
        phi=phi_np.astype(np.float32, copy=False),
        phi_clustered=phi_clustered.astype(np.float32, copy=False),
        cluster_labels=cluster_labels.astype(np.int64, copy=False),
        cluster_centers=cluster_centers.astype(np.float32, copy=False),
        gamma_major=major_probability.astype(np.float32, copy=False),
        major_probability=major_probability.astype(np.float32, copy=False),
        major_call=major_call.astype(bool, copy=False),
        multiplicity_call=multiplicity_call.astype(np.float32, copy=False),
        multiplicity_estimated_mask=data.multiplicity_estimation_mask.astype(bool, copy=False),
        loglik=float(-fit_loss),
        summary_loglik=float(-summary_fit_loss),
        penalized_objective=float(objective),
        lambda_value=float(lambda_value),
        n_clusters=int(cluster_centers.shape[0]),
        iterations=int(iterations),
        converged=bool(converged),
        device=runtime.device_name,
        graph_name=str(graph.name),
        history=[float(value) for value in history],
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
    device: str | None = "auto",
    verbose: bool = False,
) -> FusionFitArtifacts:
    runtime = resolve_runtime(device)
    effective_graph = coerce_graph(data.num_mutations, graph)
    torch_data = to_torch_tumor_data(data, runtime)

    exact_pilot = compute_exact_observed_data_pilot(
        data,
        runtime=runtime,
        major_prior=major_prior,
        eps=eps,
        tol=max(tol, 1e-6),
        max_iter=max(inner_max_iter, 16),
    )
    pooled_start = compute_pooled_observed_data_start(
        data,
        runtime=runtime,
        major_prior=major_prior,
        eps=eps,
        tol=max(tol, 1e-6),
        max_iter=max(inner_max_iter, 16),
        beta_hints=exact_pilot,
    )

    start_bank = []
    if phi_start is not None:
        start_bank.append(np.asarray(phi_start, dtype=np.float64))
    start_bank.extend([exact_pilot, pooled_start])
    start_bank = _deduplicate_starts(start_bank)

    best_artifacts: FusionFitArtifacts | None = None
    for start in start_bank:
        artifacts = _fit_from_start(
            data,
            torch_data=torch_data,
            runtime=runtime,
            graph=effective_graph,
            lambda_value=lambda_value,
            major_prior=major_prior,
            eps=eps,
            outer_max_iter=outer_max_iter,
            inner_max_iter=inner_max_iter,
            tol=tol,
            phi_start=start,
            verbose=verbose,
        )
        if best_artifacts is None or artifacts.penalized_objective < best_artifacts.penalized_objective - 1e-8:
            best_artifacts = artifacts

    if best_artifacts is None:
        raise RuntimeError("No valid start produced a fusion fit.")
    return best_artifacts
