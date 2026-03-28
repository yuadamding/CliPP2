from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar
import torch

from ..io.data import PatientData
from .graph import GraphData


@dataclass
class FitOptions:
    lambda_value: float
    em_max_iter: int = 8
    em_tol: float = 1e-4
    admm_max_iter: int = 20
    admm_tol: float = 5e-3
    admm_rho: float = 2.0
    inner_steps: int = 2
    inner_lr: float = 5e-2
    cg_max_iter: int = 30
    cg_tol: float = 1e-4
    curvature_floor: float = 1e-4
    major_prior: float = 0.5
    eps: float = 1e-6
    fused_tol: float = 1e-3
    center_merge_tol: float = 8e-2
    refit_cluster_centers: bool = False
    device: str = "auto"
    verbose: bool = False


@dataclass
class FitResult:
    phi: np.ndarray
    phi_clustered: np.ndarray
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    major_probability: np.ndarray
    major_call: np.ndarray
    multiplicity_call: np.ndarray
    multiplicity_estimated_mask: np.ndarray
    loglik: float
    penalized_objective: float
    lambda_value: float
    n_clusters: int
    iterations: int
    converged: bool
    device: str
    z_norm: np.ndarray
    history: list[float] = field(default_factory=list)
    bic: float | None = None
    classic_bic: float | None = None
    extended_bic: float | None = None
    selection_score_name: str | None = None


@dataclass
class RawFitResult:
    phi: np.ndarray
    z_norm: np.ndarray
    lambda_value: float
    iterations: int
    converged: bool
    device: str
    history: list[float] = field(default_factory=list)


@dataclass
class _TorchFitContext:
    device: torch.device
    alt_counts: torch.Tensor
    total_counts: torch.Tensor
    scaling: torch.Tensor
    minor_cn: torch.Tensor
    major_cn: torch.Tensor
    minor_scale: torch.Tensor
    major_scale: torch.Tensor
    phi_upper: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_weight: torch.Tensor
    degree: torch.Tensor
    multiplicity_estimation_mask: torch.Tensor
    fixed_major_probability: torch.Tensor
    num_edges: int


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _make_torch_context(data: PatientData, graph: GraphData, device: torch.device) -> _TorchFitContext:
    alt_counts = torch.as_tensor(data.alt_counts, dtype=torch.float32, device=device)
    total_counts = torch.as_tensor(data.total_counts, dtype=torch.float32, device=device)
    scaling = torch.as_tensor(data.scaling, dtype=torch.float32, device=device)
    minor_cn = torch.as_tensor(data.minor_cn, dtype=torch.float32, device=device)
    major_cn = torch.as_tensor(data.major_cn, dtype=torch.float32, device=device)
    minor_scale = scaling * minor_cn
    major_scale = scaling * major_cn
    phi_upper = torch.as_tensor(data.phi_upper, dtype=torch.float32, device=device)

    edge_src = torch.as_tensor(graph.src, dtype=torch.long, device=device)
    edge_dst = torch.as_tensor(graph.dst, dtype=torch.long, device=device)
    edge_weight = torch.as_tensor(graph.weight, dtype=torch.float32, device=device).view(-1, 1)
    degree = torch.zeros((data.num_mutations, 1), dtype=torch.float32, device=device)
    multiplicity_estimation_mask = torch.as_tensor(
        data.multiplicity_estimation_mask,
        dtype=torch.bool,
        device=device,
    )
    fixed_major_probability = torch.where(
        multiplicity_estimation_mask,
        torch.zeros_like(alt_counts),
        torch.ones_like(alt_counts),
    )
    if graph.num_edges > 0:
        ones = torch.ones((graph.num_edges, 1), dtype=torch.float32, device=device)
        degree.index_add_(0, edge_src, ones)
        degree.index_add_(0, edge_dst, ones)

    return _TorchFitContext(
        device=device,
        alt_counts=alt_counts,
        total_counts=total_counts,
        scaling=scaling,
        minor_cn=minor_cn,
        major_cn=major_cn,
        minor_scale=minor_scale,
        major_scale=major_scale,
        phi_upper=phi_upper,
        edge_src=edge_src,
        edge_dst=edge_dst,
        edge_weight=edge_weight,
        degree=degree,
        multiplicity_estimation_mask=multiplicity_estimation_mask,
        fixed_major_probability=fixed_major_probability,
        num_edges=int(graph.num_edges),
    )


def make_fit_context(
    data: PatientData,
    graph: GraphData,
    device: str | torch.device = "auto",
) -> _TorchFitContext:
    resolved_device = device if isinstance(device, torch.device) else _resolve_device(str(device))
    return _make_torch_context(data=data, graph=graph, device=resolved_device)


def _torch_e_step(
    phi: torch.Tensor,
    alt_counts: torch.Tensor,
    total_counts: torch.Tensor,
    scaling: torch.Tensor,
    minor_cn: torch.Tensor,
    major_cn: torch.Tensor,
    multiplicity_estimation_mask: torch.Tensor | None,
    fixed_major_probability: torch.Tensor | None,
    major_prior: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    log_prior_major = float(np.log(max(major_prior, eps)))
    log_prior_minor = float(np.log(max(1.0 - major_prior, eps)))

    p_minor = torch.clamp(scaling * minor_cn * phi, eps, 1.0 - eps)
    p_major = torch.clamp(scaling * major_cn * phi, eps, 1.0 - eps)

    ll_minor = alt_counts * torch.log(p_minor) + (total_counts - alt_counts) * torch.log1p(-p_minor) + log_prior_minor
    ll_major = alt_counts * torch.log(p_major) + (total_counts - alt_counts) * torch.log1p(-p_major) + log_prior_major

    stacked = torch.stack((ll_minor, ll_major), dim=-1)
    log_norm = torch.logsumexp(stacked, dim=-1, keepdim=True)
    responsibilities = torch.exp(stacked - log_norm)
    if multiplicity_estimation_mask is None or fixed_major_probability is None:
        major_probability = responsibilities[..., 1]
        minor_probability = responsibilities[..., 0]
        marginal_terms = log_norm.squeeze(-1)
    else:
        estimate_mask = multiplicity_estimation_mask
        fixed_major = fixed_major_probability
        major_probability = torch.where(estimate_mask, responsibilities[..., 1], fixed_major)
        minor_probability = 1.0 - major_probability
        fixed_loglik = torch.where(fixed_major >= 0.5, ll_major, ll_minor)
        marginal_terms = torch.where(estimate_mask, log_norm.squeeze(-1), fixed_loglik)
    marginal_loglik = float(marginal_terms.sum().detach().cpu().item())
    return minor_probability, major_probability, marginal_loglik


def _numpy_e_step(phi: np.ndarray, data: PatientData, options: FitOptions) -> tuple[np.ndarray, np.ndarray, float]:
    phi_t = torch.as_tensor(phi, dtype=torch.float32)
    alt_t = torch.as_tensor(data.alt_counts, dtype=torch.float32)
    total_t = torch.as_tensor(data.total_counts, dtype=torch.float32)
    scaling_t = torch.as_tensor(data.scaling, dtype=torch.float32)
    minor_t = torch.as_tensor(data.minor_cn, dtype=torch.float32)
    major_t = torch.as_tensor(data.major_cn, dtype=torch.float32)
    estimate_mask_t = torch.as_tensor(data.multiplicity_estimation_mask, dtype=torch.bool)
    fixed_major_t = torch.where(
        estimate_mask_t,
        torch.zeros_like(alt_t),
        torch.ones_like(alt_t),
    )
    minor_prob, major_prob, marginal_loglik = _torch_e_step(
        phi=phi_t,
        alt_counts=alt_t,
        total_counts=total_t,
        scaling=scaling_t,
        minor_cn=minor_t,
        major_cn=major_t,
        multiplicity_estimation_mask=estimate_mask_t,
        fixed_major_probability=fixed_major_t,
        major_prior=options.major_prior,
        eps=options.eps,
    )
    return (
        minor_prob.numpy(),
        major_prob.numpy(),
        marginal_loglik,
    )


def _penalty_value(phi: np.ndarray, graph: GraphData, lambda_value: float) -> float:
    if graph.num_edges == 0:
        return 0.0
    diff = phi[graph.src] - phi[graph.dst]
    norms = np.linalg.norm(diff, axis=1)
    return float(lambda_value * np.sum(graph.weight * norms))


def _difference_transpose_torch(
    values: torch.Tensor,
    num_nodes: int,
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros((num_nodes, values.shape[1]), dtype=values.dtype, device=values.device)
    if src.numel() == 0:
        return out
    out.index_add_(0, src, values)
    out.index_add_(0, dst, -values)
    return out


def _laplacian_matvec_torch(
    x: torch.Tensor,
    diagonal: torch.Tensor,
    rho: float,
    src: torch.Tensor,
    dst: torch.Tensor,
) -> torch.Tensor:
    out = diagonal * x
    if src.numel() == 0:
        return out
    diff = x[src] - x[dst]
    lap = torch.zeros_like(x)
    lap.index_add_(0, src, diff)
    lap.index_add_(0, dst, -diff)
    return out + float(rho) * lap


def _group_soft_threshold_torch(
    value: torch.Tensor,
    threshold: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if value.numel() == 0:
        return value
    norms = torch.linalg.norm(value, dim=1, keepdim=True)
    scale = torch.clamp(1.0 - threshold / torch.clamp(norms, min=eps), min=0.0)
    return scale * value


def _weighted_surrogate_grad_hessian_torch(
    phi: torch.Tensor,
    alt_counts: torch.Tensor,
    total_counts: torch.Tensor,
    minor_scale: torch.Tensor,
    major_scale: torch.Tensor,
    minor_probability: torch.Tensor,
    major_probability: torch.Tensor,
    eps: float,
    curvature_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    one_minus_alt = total_counts - alt_counts

    p_minor = torch.clamp(minor_scale * phi, eps, 1.0 - eps)
    p_major = torch.clamp(major_scale * phi, eps, 1.0 - eps)

    grad_minor = minor_probability * minor_scale * (one_minus_alt / (1.0 - p_minor) - alt_counts / p_minor)
    grad_major = major_probability * major_scale * (one_minus_alt / (1.0 - p_major) - alt_counts / p_major)

    hess_minor = minor_probability * minor_scale.square() * (
        alt_counts / p_minor.square() + one_minus_alt / (1.0 - p_minor).square()
    )
    hess_major = major_probability * major_scale.square() * (
        alt_counts / p_major.square() + one_minus_alt / (1.0 - p_major).square()
    )

    grad = grad_minor + grad_major
    hess = torch.clamp(hess_minor + hess_major, min=curvature_floor)
    return grad, hess


def _cg_solve_graph_quadratic_torch(
    initial: torch.Tensor,
    diagonal: torch.Tensor,
    target: torch.Tensor,
    rhs_shift: torch.Tensor,
    degree: torch.Tensor,
    phi_upper: torch.Tensor,
    rho: float,
    src: torch.Tensor,
    dst: torch.Tensor,
    cg_max_iter: int,
    cg_tol: float,
    eps: float,
) -> torch.Tensor:
    rhs = diagonal * target + float(rho) * rhs_shift
    x = initial.clone()
    residual = rhs - _laplacian_matvec_torch(x, diagonal, rho, src, dst)
    preconditioner = 1.0 / torch.clamp(diagonal + float(rho) * degree, min=eps)
    z = preconditioner * residual
    direction = z.clone()
    rz_old = torch.sum(residual * z)

    if float(torch.sqrt(torch.clamp(rz_old, min=0.0)).item()) <= cg_tol:
        return torch.minimum(torch.clamp(x, min=eps), phi_upper)

    rhs_norm = max(float(torch.linalg.norm(rhs).item()), 1.0)
    for _ in range(max(cg_max_iter, 1)):
        matvec = _laplacian_matvec_torch(direction, diagonal, rho, src, dst)
        denom = torch.clamp(torch.sum(direction * matvec), min=eps)
        alpha = rz_old / denom
        x = x + alpha * direction
        residual = residual - alpha * matvec

        if float(torch.linalg.norm(residual).item()) <= cg_tol * rhs_norm:
            break

        z = preconditioner * residual
        rz_new = torch.sum(residual * z)
        beta = rz_new / torch.clamp(rz_old, min=eps)
        direction = z + beta * direction
        rz_old = rz_new

    return torch.minimum(torch.clamp(x, min=eps), phi_upper)


def _m_step_admm_torch(
    context: _TorchFitContext,
    phi_start: torch.Tensor,
    minor_probability: torch.Tensor,
    major_probability: torch.Tensor,
    options: FitOptions,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.set_float32_matmul_precision("high")

    phi = torch.minimum(torch.clamp(phi_start, min=options.eps), context.phi_upper)

    if context.num_edges > 0:
        z = torch.zeros((context.num_edges, context.alt_counts.shape[1]), dtype=torch.float32, device=context.device)
        u = torch.zeros_like(z)
    else:
        z = torch.zeros((0, context.alt_counts.shape[1]), dtype=torch.float32, device=context.device)
        u = torch.zeros_like(z)

    for admm_iter in range(options.admm_max_iter):
        for _ in range(max(options.inner_steps, 1)):
            grad, hess = _weighted_surrogate_grad_hessian_torch(
                phi=phi,
                alt_counts=context.alt_counts,
                total_counts=context.total_counts,
                minor_scale=context.minor_scale,
                major_scale=context.major_scale,
                minor_probability=minor_probability,
                major_probability=major_probability,
                eps=options.eps,
                curvature_floor=options.curvature_floor,
            )
            target = torch.minimum(torch.clamp(phi - grad / hess, min=options.eps), context.phi_upper)
            if context.num_edges == 0:
                phi = target
                continue

            rhs_shift = _difference_transpose_torch(
                values=z - u,
                num_nodes=context.alt_counts.shape[0],
                src=context.edge_src,
                dst=context.edge_dst,
            )
            phi = _cg_solve_graph_quadratic_torch(
                initial=phi,
                diagonal=hess,
                target=target,
                rhs_shift=rhs_shift,
                degree=context.degree,
                phi_upper=context.phi_upper,
                rho=options.admm_rho,
                src=context.edge_src,
                dst=context.edge_dst,
                cg_max_iter=options.cg_max_iter,
                cg_tol=options.cg_tol,
                eps=options.eps,
            )

        if context.num_edges == 0:
            break

        diff = phi[context.edge_src] - phi[context.edge_dst]
        v = diff + u
        z_new = _group_soft_threshold_torch(
            value=v,
            threshold=(options.lambda_value * context.edge_weight) / max(options.admm_rho, options.eps),
            eps=options.eps,
        )
        primal_residual = torch.linalg.norm(diff - z_new).item()
        dual_residual = (options.admm_rho * torch.linalg.norm(z_new - z)).item()
        u = u + diff - z_new
        z = z_new

        if options.verbose:
            print(
                f"  ADMM iter {admm_iter + 1:02d} "
                f"| primal={primal_residual:.4e} dual={dual_residual:.4e}"
            )

        if primal_residual < options.admm_tol and dual_residual < options.admm_tol:
            break

    return phi, z


def _connected_components(num_nodes: int, src: np.ndarray, dst: np.ndarray, fused_mask: np.ndarray) -> np.ndarray:
    parent = np.arange(num_nodes, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i, keep in enumerate(fused_mask):
        if keep:
            union(int(src[i]), int(dst[i]))

    raw_labels = np.array([find(i) for i in range(num_nodes)], dtype=np.int64)
    _, relabeled = np.unique(raw_labels, return_inverse=True)
    return relabeled.astype(np.int64)


def _cluster_profiles(phi: np.ndarray, graph: GraphData, z_norm: np.ndarray, fused_tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if graph.num_edges == 0:
        labels = np.arange(phi.shape[0], dtype=np.int64)
        centers = phi.copy()
        return labels, centers, phi.copy()

    fused_mask = z_norm <= fused_tol
    labels = _connected_components(phi.shape[0], graph.src, graph.dst, fused_mask)
    num_clusters = int(labels.max()) + 1
    centers = np.zeros((num_clusters, phi.shape[1]), dtype=np.float32)

    for label in range(num_clusters):
        members = labels == label
        centers[label] = phi[members].mean(axis=0)

    clustered_phi = centers[labels]
    return labels, centers, clustered_phi


def _merge_cluster_profiles(
    phi: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
    merge_tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if merge_tol <= 0 or cluster_centers.shape[0] <= 1:
        return cluster_labels, cluster_centers, cluster_centers[cluster_labels]

    num_centers = cluster_centers.shape[0]
    parent = np.arange(num_centers, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(num_centers):
        for j in range(i + 1, num_centers):
            diff = cluster_centers[i] - cluster_centers[j]
            rms_distance = float(np.sqrt(np.mean(diff * diff)))
            if rms_distance <= merge_tol:
                union(i, j)

    center_groups = np.array([find(i) for i in range(num_centers)], dtype=np.int64)
    _, relabeled_centers = np.unique(center_groups, return_inverse=True)
    merged_labels = relabeled_centers[cluster_labels]
    merged_count = int(merged_labels.max()) + 1
    merged_centers = np.zeros((merged_count, phi.shape[1]), dtype=np.float32)

    for label in range(merged_count):
        merged_centers[label] = phi[merged_labels == label].mean(axis=0)

    merged_phi = merged_centers[merged_labels]
    return merged_labels.astype(np.int64), merged_centers, merged_phi


def _small_cluster_threshold(num_mutations: int, num_samples: int) -> int:
    adaptive = int(np.round(0.01 * float(num_mutations) / max(np.sqrt(float(max(num_samples, 1))), 1.0)))
    return int(np.clip(max(adaptive, 2), 2, 25))


def _recompute_cluster_centers(
    phi: np.ndarray,
    cluster_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, relabeled = np.unique(cluster_labels.astype(np.int64), return_inverse=True)
    relabeled = relabeled.astype(np.int64)
    num_clusters = int(relabeled.max()) + 1
    centers = np.zeros((num_clusters, phi.shape[1]), dtype=np.float32)
    for label in range(num_clusters):
        centers[label] = phi[relabeled == label].mean(axis=0)
    return relabeled, centers, centers[relabeled]


def _cleanup_small_clusters(
    phi: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
    *,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cluster_centers.shape[0] <= 1:
        return cluster_labels, cluster_centers, cluster_centers[cluster_labels]

    labels = cluster_labels.astype(np.int64, copy=True)
    min_size = _small_cluster_threshold(phi.shape[0], num_samples=num_samples)

    for _ in range(4):
        cluster_sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
        small_labels = np.where(cluster_sizes <= min_size)[0]
        if small_labels.size == 0 or cluster_sizes.size <= 1:
            break

        large_labels = np.where(cluster_sizes > min_size)[0]
        if large_labels.size == 0:
            break

        centers = np.zeros((cluster_sizes.size, phi.shape[1]), dtype=np.float32)
        for label in range(cluster_sizes.size):
            centers[label] = phi[labels == label].mean(axis=0)

        changed = False
        for label in small_labels:
            if cluster_sizes[label] == 0:
                continue

            candidate_labels = large_labels[large_labels != label]
            if candidate_labels.size == 0:
                candidate_labels = np.where(cluster_sizes > 0)[0]
                candidate_labels = candidate_labels[candidate_labels != label]
            if candidate_labels.size == 0:
                continue

            diffs = centers[candidate_labels] - centers[label]
            rms_distances = np.sqrt(np.mean(diffs * diffs, axis=1))
            target_label = int(candidate_labels[int(np.argmin(rms_distances))])
            labels[labels == label] = target_label
            changed = True

        labels, centers, clustered_phi = _recompute_cluster_centers(phi, labels)
        if not changed:
            return labels, centers, clustered_phi

    labels, centers, clustered_phi = _recompute_cluster_centers(phi, labels)
    return labels, centers, clustered_phi


def _cluster_sample_loglik(
    theta: float,
    *,
    alt_counts: np.ndarray,
    total_counts: np.ndarray,
    scaling: np.ndarray,
    minor_cn: np.ndarray,
    major_cn: np.ndarray,
    estimate_mask: np.ndarray,
    fixed_multiplicity: np.ndarray,
    major_prior: float,
    eps: float,
) -> float:
    theta = float(theta)
    total_loglik = 0.0

    if estimate_mask.any():
        p_minor = np.clip(scaling[estimate_mask] * minor_cn[estimate_mask] * theta, eps, 1.0 - eps)
        p_major = np.clip(scaling[estimate_mask] * major_cn[estimate_mask] * theta, eps, 1.0 - eps)
        alt = alt_counts[estimate_mask]
        total = total_counts[estimate_mask]
        ll_minor = alt * np.log(p_minor) + (total - alt) * np.log1p(-p_minor) + np.log(max(1.0 - major_prior, eps))
        ll_major = alt * np.log(p_major) + (total - alt) * np.log1p(-p_major) + np.log(max(major_prior, eps))
        total_loglik += float(np.logaddexp(ll_minor, ll_major).sum())

    fixed_mask = ~estimate_mask
    if fixed_mask.any():
        p_fixed = np.clip(scaling[fixed_mask] * fixed_multiplicity[fixed_mask] * theta, eps, 1.0 - eps)
        alt = alt_counts[fixed_mask]
        total = total_counts[fixed_mask]
        total_loglik += float((alt * np.log(p_fixed) + (total - alt) * np.log1p(-p_fixed)).sum())

    return total_loglik


def _refit_partition_cluster_centers(
    data: PatientData,
    cluster_labels: np.ndarray,
    cluster_centers: np.ndarray,
    options: FitOptions,
) -> tuple[np.ndarray, np.ndarray]:
    refit_centers = np.zeros_like(cluster_centers, dtype=np.float32)
    estimate_mask = data.multiplicity_estimation_mask.astype(bool, copy=False)
    fixed_multiplicity = data.fixed_multiplicity.astype(np.float32, copy=False)

    for label in range(cluster_centers.shape[0]):
        members = cluster_labels == label
        if not np.any(members):
            continue

        for sample_idx in range(data.num_samples):
            alt = data.alt_counts[members, sample_idx].astype(np.float64, copy=False)
            total = data.total_counts[members, sample_idx].astype(np.float64, copy=False)
            scaling = data.scaling[members, sample_idx].astype(np.float64, copy=False)
            minor_cn = data.minor_cn[members, sample_idx].astype(np.float64, copy=False)
            major_cn = data.major_cn[members, sample_idx].astype(np.float64, copy=False)
            sample_estimate_mask = estimate_mask[members, sample_idx]
            sample_fixed = fixed_multiplicity[members, sample_idx].astype(np.float64, copy=False)

            lower = float(options.eps)
            upper = float(np.clip(np.min(data.phi_upper[members, sample_idx]), lower, 1.0))
            start = float(np.clip(cluster_centers[label, sample_idx], lower, upper))

            if upper <= lower + 1e-8:
                refit_centers[label, sample_idx] = lower
                continue

            def objective(theta: float) -> float:
                return -_cluster_sample_loglik(
                    theta,
                    alt_counts=alt,
                    total_counts=total,
                    scaling=scaling,
                    minor_cn=minor_cn,
                    major_cn=major_cn,
                    estimate_mask=sample_estimate_mask,
                    fixed_multiplicity=sample_fixed,
                    major_prior=options.major_prior,
                    eps=options.eps,
                )

            candidate_thetas = np.array([lower, start, upper], dtype=np.float64)
            candidate_values = np.array([objective(value) for value in candidate_thetas], dtype=np.float64)
            best_theta = float(candidate_thetas[int(np.argmin(candidate_values))])

            result = minimize_scalar(
                objective,
                bounds=(lower, upper),
                method="bounded",
                options={"xatol": 1e-4, "maxiter": 64},
            )
            if result.success and np.isfinite(result.fun):
                best_theta = float(np.clip(result.x, lower, upper))

            refit_centers[label, sample_idx] = best_theta

    refit_phi = refit_centers[cluster_labels]
    return refit_centers.astype(np.float32), refit_phi.astype(np.float32)


def fit_single_stage_em(
    data: PatientData,
    graph: GraphData,
    options: FitOptions,
    phi_start: np.ndarray | None = None,
) -> FitResult:
    raw_fit = fit_single_stage_em_raw(
        data=data,
        graph=graph,
        options=options,
        phi_start=phi_start,
    )
    return finalize_raw_fit(
        data=data,
        graph=graph,
        raw_fit=raw_fit,
        options=options,
    )


def fit_single_stage_em_raw(
    data: PatientData,
    graph: GraphData,
    options: FitOptions,
    phi_start: np.ndarray | None = None,
) -> RawFitResult:
    device = _resolve_device(options.device)
    context = _make_torch_context(data=data, graph=graph, device=device)

    if phi_start is None:
        phi = data.phi_init.copy()
    else:
        phi = np.asarray(phi_start, dtype=np.float32).copy()
    phi = np.clip(phi, options.eps, data.phi_upper)
    history: list[float] = []
    converged = False
    phi_t = torch.as_tensor(phi, dtype=torch.float32, device=device)
    z_t = torch.zeros((graph.num_edges, data.num_samples), dtype=torch.float32, device=device)

    previous_loglik = -np.inf
    with torch.inference_mode():
        for iteration in range(options.em_max_iter):
            minor_probability_t, major_probability_t, marginal_loglik = _torch_e_step(
                phi=phi_t,
                alt_counts=context.alt_counts,
                total_counts=context.total_counts,
                scaling=context.scaling,
                minor_cn=context.minor_cn,
                major_cn=context.major_cn,
                multiplicity_estimation_mask=context.multiplicity_estimation_mask,
                fixed_major_probability=context.fixed_major_probability,
                major_prior=options.major_prior,
                eps=options.eps,
            )
            history.append(marginal_loglik)

            if options.verbose:
                print(
                    f"EM iter {iteration + 1:02d} | "
                    f"loglik={marginal_loglik:.3f} | lambda={options.lambda_value:.3f}"
                )

            if iteration > 0 and abs(marginal_loglik - previous_loglik) < options.em_tol * (1.0 + abs(previous_loglik)):
                converged = True
                break

            previous_loglik = marginal_loglik
            phi_t, z_t = _m_step_admm_torch(
                context=context,
                phi_start=phi_t,
                minor_probability=minor_probability_t,
                major_probability=major_probability_t,
                options=options,
            )

        phi = phi_t.detach().cpu().numpy().astype(np.float32)
        z = z_t.detach().cpu().numpy().astype(np.float32)

    z_norm = np.linalg.norm(z, axis=1) if graph.num_edges > 0 else np.zeros(0, dtype=np.float32)
    return RawFitResult(
        phi=phi.astype(np.float32),
        z_norm=z_norm.astype(np.float32),
        lambda_value=float(options.lambda_value),
        iterations=len(history),
        converged=converged,
        device=str(device),
        history=history,
    )


def finalize_raw_fit(
    data: PatientData,
    graph: GraphData,
    raw_fit: RawFitResult,
    options: FitOptions,
    context: _TorchFitContext | None = None,
) -> FitResult:
    if context is None:
        device = _resolve_device(options.device)
        context = _make_torch_context(data=data, graph=graph, device=device)
    else:
        device = context.device

    phi = raw_fit.phi.astype(np.float32, copy=True)
    z_norm = raw_fit.z_norm.astype(np.float32, copy=True)
    cluster_labels, cluster_centers, phi_clustered = _cluster_profiles(
        phi=phi,
        graph=graph,
        z_norm=z_norm,
        fused_tol=options.fused_tol,
    )
    cluster_labels, cluster_centers, phi_clustered = _merge_cluster_profiles(
        phi=phi,
        cluster_labels=cluster_labels,
        cluster_centers=cluster_centers,
        merge_tol=options.center_merge_tol,
    )
    cluster_labels, cluster_centers, phi_clustered = _cleanup_small_clusters(
        phi=phi,
        cluster_labels=cluster_labels,
        cluster_centers=cluster_centers,
        num_samples=data.num_samples,
    )
    cluster_labels, cluster_centers, phi_clustered = _merge_cluster_profiles(
        phi=phi,
        cluster_labels=cluster_labels,
        cluster_centers=cluster_centers,
        merge_tol=options.center_merge_tol,
    )
    if options.refit_cluster_centers:
        cluster_centers, phi_clustered = _refit_partition_cluster_centers(
            data=data,
            cluster_labels=cluster_labels,
            cluster_centers=cluster_centers,
            options=options,
        )

    phi_clustered_t = torch.as_tensor(phi_clustered, dtype=torch.float32, device=device)
    _, major_probability_t, loglik = _torch_e_step(
        phi=phi_clustered_t,
        alt_counts=context.alt_counts,
        total_counts=context.total_counts,
        scaling=context.scaling,
        minor_cn=context.minor_cn,
        major_cn=context.major_cn,
        multiplicity_estimation_mask=context.multiplicity_estimation_mask,
        fixed_major_probability=context.fixed_major_probability,
        major_prior=options.major_prior,
        eps=options.eps,
    )
    major_probability = major_probability_t.detach().cpu().numpy()
    major_call = major_probability >= 0.5
    multiplicity_call = np.where(
        data.multiplicity_estimation_mask,
        np.where(major_call, data.major_cn, data.minor_cn),
        data.fixed_multiplicity,
    ).astype(np.float32)
    penalty_value = _penalty_value(phi_clustered, graph, options.lambda_value)
    penalized_objective = float(loglik - penalty_value)

    return FitResult(
        phi=phi.astype(np.float32),
        phi_clustered=phi_clustered.astype(np.float32),
        cluster_labels=cluster_labels.astype(np.int64),
        cluster_centers=cluster_centers.astype(np.float32),
        major_probability=major_probability.astype(np.float32),
        major_call=major_call.astype(bool),
        multiplicity_call=multiplicity_call.astype(np.float32),
        multiplicity_estimated_mask=data.multiplicity_estimation_mask.astype(bool),
        loglik=float(loglik),
        penalized_objective=penalized_objective,
        lambda_value=float(options.lambda_value),
        n_clusters=int(cluster_centers.shape[0]),
        iterations=int(raw_fit.iterations),
        converged=bool(raw_fit.converged),
        device=str(raw_fit.device),
        z_norm=z_norm.astype(np.float32),
        history=list(raw_fit.history),
        selection_score_name=None,
    )
