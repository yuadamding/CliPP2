from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...io.data import TumorData
from .types import PairwiseFusionGraph, TorchRuntime


@dataclass(frozen=True)
class TorchTumorData:
    alt: torch.Tensor
    total: torch.Tensor
    nonalt: torch.Tensor
    phi_upper: torch.Tensor
    ambiguous: torch.Tensor
    b_minus: torch.Tensor
    b_plus: torch.Tensor
    b_fixed: torch.Tensor


@dataclass(frozen=True)
class TorchCellTerms:
    loss: torch.Tensor
    grad: torch.Tensor
    hess_upper: torch.Tensor
    gamma_major: torch.Tensor


def _binary_entropy_offset_torch(weight: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(weight, min=0.0, max=1.0)
    positive = clipped > 0.0
    one_minus = 1.0 - clipped
    positive_complement = one_minus > 0.0
    term = torch.zeros_like(clipped)
    term = torch.where(positive, term + clipped * torch.log(clipped), term)
    term = torch.where(positive_complement, term + one_minus * torch.log(one_minus), term)
    return term


def resolve_runtime(device: str | None) -> TorchRuntime:
    requested = "auto" if device is None else str(device).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    runtime_device = torch.device(requested)
    runtime_dtype = torch.float32 if runtime_device.type == "cuda" else torch.float64
    device_name = runtime_device.type if runtime_device.index is None else f"{runtime_device.type}:{runtime_device.index}"
    return TorchRuntime(device=runtime_device, device_name=device_name, dtype=runtime_dtype)


def to_torch_tumor_data(data: TumorData, runtime: TorchRuntime) -> TorchTumorData:
    dtype = runtime.dtype
    device = runtime.device
    scaling = np.asarray(data.scaling, dtype=np.float64)
    return TorchTumorData(
        alt=torch.as_tensor(data.alt_counts, dtype=dtype, device=device),
        total=torch.as_tensor(data.total_counts, dtype=dtype, device=device),
        nonalt=torch.as_tensor(data.total_counts - data.alt_counts, dtype=dtype, device=device),
        phi_upper=torch.as_tensor(data.phi_upper, dtype=dtype, device=device),
        ambiguous=torch.as_tensor(data.multiplicity_estimation_mask, dtype=torch.bool, device=device),
        b_minus=torch.as_tensor(scaling * np.asarray(data.minor_cn, dtype=np.float64), dtype=dtype, device=device),
        b_plus=torch.as_tensor(scaling * np.asarray(data.major_cn, dtype=np.float64), dtype=dtype, device=device),
        b_fixed=torch.as_tensor(
            scaling * np.asarray(data.fixed_multiplicity, dtype=np.float64),
            dtype=dtype,
            device=device,
        ),
    )


def _as_loss_shape(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = mask
    while result.ndim < target.ndim:
        result = result.unsqueeze(-1)
    return result


def clip_probability_and_slope(beta: torch.Tensor, scale: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    linear = scale * beta
    prob = torch.clamp(linear, min=float(eps), max=float(1.0 - eps))
    interior = (linear > float(eps)) & (linear < float(1.0 - eps))
    slope = torch.where(interior, scale, torch.zeros_like(scale))
    return prob, slope


def state_log_kernel_grad_and_curvature(
    *,
    alt: torch.Tensor,
    nonalt: torch.Tensor,
    prob: torch.Tensor,
    slope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad = slope * (alt / prob - nonalt / (1.0 - prob))
    curvature = torch.square(slope) * (alt / torch.square(prob) + nonalt / torch.square(1.0 - prob))
    return grad, curvature


def cell_loss_grid_torch(
    beta_grid: torch.Tensor,
    *,
    alt: torch.Tensor,
    total: torch.Tensor,
    b_minus: torch.Tensor,
    b_plus: torch.Tensor,
    b_fixed: torch.Tensor,
    ambiguous: torch.Tensor,
    major_prior: float,
    eps: float,
) -> torch.Tensor:
    beta = beta_grid
    nonalt = total - alt
    prob_fixed = torch.clamp(beta * b_fixed, min=float(eps), max=float(1.0 - eps))
    fixed_loss = -(alt * torch.log(prob_fixed) + nonalt * torch.log1p(-prob_fixed))

    prob_minus = torch.clamp(beta * b_minus, min=float(eps), max=float(1.0 - eps))
    prob_plus = torch.clamp(beta * b_plus, min=float(eps), max=float(1.0 - eps))
    log_minor = alt * torch.log(prob_minus) + nonalt * torch.log1p(-prob_minus) + float(np.log(max(1.0 - major_prior, eps)))
    log_major = alt * torch.log(prob_plus) + nonalt * torch.log1p(-prob_plus) + float(np.log(max(major_prior, eps)))
    amb_loss = -torch.logaddexp(log_minor, log_major)
    return torch.where(_as_loss_shape(ambiguous, beta), amb_loss, fixed_loss)


def cell_terms_torch(
    data: TorchTumorData,
    phi: torch.Tensor,
    *,
    major_prior: float,
    eps: float,
) -> TorchCellTerms:
    phi = torch.clamp(phi, min=float(eps), max=1.0)
    phi = torch.minimum(phi, data.phi_upper)
    alt = data.alt
    nonalt = data.nonalt
    amb = data.ambiguous

    loss = torch.zeros_like(phi)
    grad = torch.zeros_like(phi)
    hess_upper = torch.zeros_like(phi)
    gamma_major = torch.ones_like(phi)

    fixed = ~amb
    if torch.any(fixed):
        prob_fixed, slope_fixed = clip_probability_and_slope(phi[fixed], data.b_fixed[fixed], eps)
        loss_fixed = -(alt[fixed] * torch.log(prob_fixed) + nonalt[fixed] * torch.log1p(-prob_fixed))
        grad_fixed, curvature_fixed = state_log_kernel_grad_and_curvature(
            alt=alt[fixed],
            nonalt=nonalt[fixed],
            prob=prob_fixed,
            slope=slope_fixed,
        )
        loss[fixed] = loss_fixed
        grad[fixed] = -grad_fixed
        hess_upper[fixed] = curvature_fixed

    if torch.any(amb):
        prob_minus, slope_minus = clip_probability_and_slope(phi[amb], data.b_minus[amb], eps)
        prob_plus, slope_plus = clip_probability_and_slope(phi[amb], data.b_plus[amb], eps)
        log_minor = alt[amb] * torch.log(prob_minus) + nonalt[amb] * torch.log1p(-prob_minus) + float(np.log(max(1.0 - major_prior, eps)))
        log_major = alt[amb] * torch.log(prob_plus) + nonalt[amb] * torch.log1p(-prob_plus) + float(np.log(max(major_prior, eps)))
        norm = torch.logaddexp(log_minor, log_major)
        gamma = torch.exp(log_major - norm)
        gamma_major[amb] = gamma
        loss[amb] = -norm
        grad_minus, curvature_minus = state_log_kernel_grad_and_curvature(
            alt=alt[amb],
            nonalt=nonalt[amb],
            prob=prob_minus,
            slope=slope_minus,
        )
        grad_plus, curvature_plus = state_log_kernel_grad_and_curvature(
            alt=alt[amb],
            nonalt=nonalt[amb],
            prob=prob_plus,
            slope=slope_plus,
        )
        grad[amb] = -((1.0 - gamma) * grad_minus + gamma * grad_plus)
        hess_upper[amb] = (1.0 - gamma) * curvature_minus + gamma * curvature_plus

    return TorchCellTerms(
        loss=loss,
        grad=grad,
        hess_upper=torch.clamp(hess_upper, min=1e-8),
        gamma_major=gamma_major,
    )


def em_surrogate_terms_torch(
    data: TorchTumorData,
    phi: torch.Tensor,
    *,
    omega_major: torch.Tensor,
    major_prior: float,
    eps: float,
) -> TorchCellTerms:
    phi = torch.clamp(phi, min=float(eps), max=1.0)
    phi = torch.minimum(phi, data.phi_upper)
    alt = data.alt
    nonalt = data.nonalt
    amb = data.ambiguous

    loss = torch.zeros_like(phi)
    grad = torch.zeros_like(phi)
    hess_upper = torch.zeros_like(phi)
    gamma_major = torch.ones_like(phi)

    fixed = ~amb
    if torch.any(fixed):
        prob_fixed, slope_fixed = clip_probability_and_slope(phi[fixed], data.b_fixed[fixed], eps)
        loss_fixed = -(alt[fixed] * torch.log(prob_fixed) + nonalt[fixed] * torch.log1p(-prob_fixed))
        grad_fixed, curvature_fixed = state_log_kernel_grad_and_curvature(
            alt=alt[fixed],
            nonalt=nonalt[fixed],
            prob=prob_fixed,
            slope=slope_fixed,
        )
        loss[fixed] = loss_fixed
        grad[fixed] = -grad_fixed
        hess_upper[fixed] = curvature_fixed

    if torch.any(amb):
        omega = torch.clamp(omega_major[amb], min=0.0, max=1.0)
        prob_minus, slope_minus = clip_probability_and_slope(phi[amb], data.b_minus[amb], eps)
        prob_plus, slope_plus = clip_probability_and_slope(phi[amb], data.b_plus[amb], eps)
        log_minor = alt[amb] * torch.log(prob_minus) + nonalt[amb] * torch.log1p(-prob_minus) + float(np.log(max(1.0 - major_prior, eps)))
        log_major = alt[amb] * torch.log(prob_plus) + nonalt[amb] * torch.log1p(-prob_plus) + float(np.log(max(major_prior, eps)))
        gamma_major[amb] = omega

        grad_minus, curvature_minus = state_log_kernel_grad_and_curvature(
            alt=alt[amb],
            nonalt=nonalt[amb],
            prob=prob_minus,
            slope=slope_minus,
        )
        grad_plus, curvature_plus = state_log_kernel_grad_and_curvature(
            alt=alt[amb],
            nonalt=nonalt[amb],
            prob=prob_plus,
            slope=slope_plus,
        )

        loss_major = -log_major
        loss_minor = -log_minor
        entropy_offset = _binary_entropy_offset_torch(omega)
        loss[amb] = (1.0 - omega) * loss_minor + omega * loss_major + entropy_offset
        grad[amb] = -((1.0 - omega) * grad_minus + omega * grad_plus)
        hess_upper[amb] = (1.0 - omega) * curvature_minus + omega * curvature_plus

    return TorchCellTerms(
        loss=loss,
        grad=grad,
        hess_upper=torch.clamp(hess_upper, min=1e-8),
        gamma_major=gamma_major,
    )


def pairwise_penalty_torch(
    phi: torch.Tensor,
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:
    if lambda_value <= 0.0 or edge_u.numel() == 0:
        return torch.zeros((), dtype=phi.dtype, device=phi.device)
    diffs = phi.index_select(0, edge_u) - phi.index_select(0, edge_v)
    return torch.as_tensor(float(lambda_value), dtype=phi.dtype, device=phi.device) * torch.sum(edge_w * torch.linalg.norm(diffs, dim=1))


def objective_value_torch(
    data: TorchTumorData,
    phi: torch.Tensor,
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
    major_prior: float,
    eps: float,
) -> tuple[float, float, float, torch.Tensor]:
    terms = cell_terms_torch(data, phi, major_prior=major_prior, eps=eps)
    fit_loss = torch.sum(terms.loss)
    penalty = pairwise_penalty_torch(phi, edge_u=edge_u, edge_v=edge_v, edge_w=edge_w, lambda_value=lambda_value)
    objective = fit_loss + penalty
    return float(fit_loss.item()), float(penalty.item()), float(objective.item()), terms.gamma_major


def stationarity_residual_torch(
    *,
    total_grad: torch.Tensor,
    phi: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    atol: float,
) -> torch.Tensor:
    lower_active = phi <= lower + float(atol)
    upper_active = phi >= upper - float(atol)
    frozen = upper <= lower + float(atol)
    interior = ~(lower_active | upper_active | frozen)
    residual = torch.zeros_like(total_grad)
    residual[interior] = total_grad[interior]
    residual[lower_active] = torch.minimum(total_grad[lower_active], torch.zeros_like(total_grad[lower_active]))
    residual[upper_active] = torch.maximum(total_grad[upper_active], torch.zeros_like(total_grad[upper_active]))
    residual[frozen] = 0.0
    return residual


def inner_kkt_residual_torch(
    *,
    phi: torch.Tensor,
    dual: torch.Tensor,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    atol: float,
) -> float:
    if edge_u.numel() == 0 or lambda_value <= 0.0:
        total_grad = h * (phi - U)
        stat = stationarity_residual_torch(total_grad=total_grad, phi=phi, lower=lower, upper=upper, atol=atol)
        return float((torch.linalg.norm(stat) / (1.0 + torch.linalg.norm(phi))).item())

    adj = torch.zeros_like(phi)
    adj.index_add_(0, edge_u, dual)
    adj.index_add_(0, edge_v, -dual)
    total_grad = h * (phi - U) + adj
    stat = stationarity_residual_torch(total_grad=total_grad, phi=phi, lower=lower, upper=upper, atol=atol)
    stat_resid = float((torch.linalg.norm(stat) / (1.0 + torch.linalg.norm(phi))).item())

    diff = phi.index_select(0, edge_u) - phi.index_select(0, edge_v)
    diff_norm = torch.linalg.norm(diff, dim=1)
    dual_norm = torch.linalg.norm(dual, dim=1)
    radius = float(lambda_value) * edge_w
    edge_resid = torch.zeros_like(diff_norm)
    active = diff_norm > float(atol)
    if torch.any(active):
        target = radius[active, None] * diff[active] / diff_norm[active, None].clamp_min(float(atol))
        edge_resid[active] = torch.linalg.norm(dual[active] - target, dim=1)
    if torch.any(~active):
        edge_resid[~active] = torch.clamp(dual_norm[~active] - radius[~active], min=0.0)
    denom = 1.0 + float(torch.max(radius).item()) if radius.numel() else 1.0
    edge_resid_value = float((torch.max(edge_resid) / denom).item()) if edge_resid.numel() else 0.0
    return max(stat_resid, edge_resid_value)


def solve_majorized_subproblem_pdhg_torch(
    *,
    runtime: TorchRuntime,
    num_mutations: int,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    degree_bound: int,
    tol: float,
    max_iter: int,
    phi_start: torch.Tensor,
    dual_start: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int, bool, float]:
    phi = torch.minimum(torch.maximum(phi_start.to(dtype=runtime.dtype, device=runtime.device), lower), upper)
    if lambda_value <= 0.0 or edge_u.numel() == 0:
        projected = torch.minimum(torch.maximum(U, lower), upper)
        total_grad = h * (projected - U)
        stat = stationarity_residual_torch(total_grad=total_grad, phi=projected, lower=lower, upper=upper, atol=tol)
        residual = float((torch.linalg.norm(stat) / (1.0 + torch.linalg.norm(projected))).item())
        return projected, torch.zeros((0, phi.shape[1]), dtype=runtime.dtype, device=runtime.device), 1, residual <= tol, residual

    if dual_start is not None and tuple(dual_start.shape) == (int(edge_u.numel()), int(phi.shape[1])):
        dual = dual_start.to(dtype=runtime.dtype, device=runtime.device)
    else:
        dual = torch.zeros((int(edge_u.numel()), int(phi.shape[1])), dtype=runtime.dtype, device=runtime.device)
    bar = phi.clone()
    degree = max(int(degree_bound), 1)
    step = 0.99 / np.sqrt(2.0 * float(degree))
    radius = float(lambda_value) * edge_w

    converged = False
    iterations = 0
    last_residual = np.inf
    for inner_iter in range(max(int(max_iter), 10)):
        iterations = inner_iter + 1
        edge_diff = bar.index_select(0, edge_u) - bar.index_select(0, edge_v)
        dual_trial = dual + step * edge_diff
        norms = torch.linalg.norm(dual_trial, dim=1)
        scale = torch.ones_like(norms)
        positive_radius = radius > 0.0
        if torch.any(positive_radius):
            scale[positive_radius] = torch.maximum(
                torch.ones_like(norms[positive_radius]),
                norms[positive_radius] / radius[positive_radius],
            )
        if torch.any(~positive_radius):
            scale[~positive_radius] = torch.inf
        dual_new = dual_trial / scale[:, None]
        if torch.any(~positive_radius):
            dual_new[~positive_radius] = 0.0

        adj = torch.zeros_like(phi)
        adj.index_add_(0, edge_u, dual_new)
        adj.index_add_(0, edge_v, -dual_new)
        primal_base = phi - step * adj
        phi_new = (primal_base + step * h * U) / (1.0 + step * h)
        phi_new = torch.minimum(torch.maximum(phi_new, lower), upper)
        bar = phi_new + (phi_new - phi)

        primal_delta = float((torch.linalg.norm(phi_new - phi) / (1.0 + torch.linalg.norm(phi))).item())
        dual_delta = float((torch.linalg.norm(dual_new - dual) / (1.0 + torch.linalg.norm(dual))).item())
        phi = phi_new
        dual = dual_new

        last_residual = inner_kkt_residual_torch(
            phi=phi,
            dual=dual,
            U=U,
            h=h,
            lower=lower,
            upper=upper,
            lambda_value=lambda_value,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            atol=max(tol, 1e-8),
        )
        if primal_delta <= tol and dual_delta <= tol and last_residual <= 5.0 * tol:
            converged = True
            break

    return phi, dual, iterations, converged, float(last_residual)


def _complete_graph_isotropic_box_qp_torch(
    *,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    rho: float,
    q: torch.Tensor,
    max_iter: int,
) -> torch.Tensor:
    num_mutations = int(U.shape[0])
    rho_t = torch.as_tensor(float(rho), dtype=U.dtype, device=U.device)
    rhs = h * U + rho_t * q
    denom = h + rho_t * float(num_mutations)

    lo = torch.sum(lower, dim=0)
    hi = torch.sum(upper, dim=0)
    mid = 0.5 * (lo + hi)
    for _ in range(max(int(max_iter), 16)):
        mid = 0.5 * (lo + hi)
        x_mid = torch.minimum(
            torch.maximum((rhs + rho_t * mid.unsqueeze(0)) / denom, lower),
            upper,
        )
        residual = torch.sum(x_mid, dim=0) - mid
        move_right = residual > 0.0
        lo = torch.where(move_right, mid, lo)
        hi = torch.where(move_right, hi, mid)

    return torch.minimum(
        torch.maximum((rhs + rho_t * mid.unsqueeze(0)) / denom, lower),
        upper,
    )


def solve_majorized_subproblem_alm_torch(
    *,
    runtime: TorchRuntime,
    num_mutations: int,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    tol: float,
    max_iter: int,
    phi_start: torch.Tensor,
    dual_start: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int, bool, float]:
    phi = torch.minimum(torch.maximum(phi_start.to(dtype=runtime.dtype, device=runtime.device), lower), upper)
    if lambda_value <= 0.0 or edge_u.numel() == 0:
        projected = torch.minimum(torch.maximum(U, lower), upper)
        total_grad = h * (projected - U)
        stat = stationarity_residual_torch(total_grad=total_grad, phi=projected, lower=lower, upper=upper, atol=tol)
        residual = float((torch.linalg.norm(stat) / (1.0 + torch.linalg.norm(projected))).item())
        return projected, torch.zeros((0, phi.shape[1]), dtype=runtime.dtype, device=runtime.device), 1, residual <= tol, residual

    if dual_start is not None and tuple(dual_start.shape) == (int(edge_u.numel()), int(phi.shape[1])):
        scaled_dual = dual_start.to(dtype=runtime.dtype, device=runtime.device)
    else:
        scaled_dual = torch.zeros((int(edge_u.numel()), int(phi.shape[1])), dtype=runtime.dtype, device=runtime.device)

    rho = float(torch.clamp(torch.median(h), min=1e-3, max=1e3).item())
    shrink_radius = (float(lambda_value) * edge_w) / rho

    converged = False
    iterations = 0
    last_residual = np.inf
    z = phi.index_select(0, edge_u) - phi.index_select(0, edge_v)

    for inner_iter in range(max(int(max_iter), 10)):
        iterations = inner_iter + 1
        edge_diff = phi.index_select(0, edge_u) - phi.index_select(0, edge_v)
        z_argument = edge_diff + scaled_dual
        z_norm = torch.linalg.norm(z_argument, dim=1, keepdim=True)
        shrink = torch.clamp(1.0 - shrink_radius[:, None] / z_norm.clamp_min(1e-12), min=0.0)
        z_new = shrink * z_argument

        rhs_edge = z_new - scaled_dual
        q = torch.zeros_like(phi)
        q.index_add_(0, edge_u, rhs_edge)
        q.index_add_(0, edge_v, -rhs_edge)
        phi_new = _complete_graph_isotropic_box_qp_torch(
            U=U,
            h=h,
            lower=lower,
            upper=upper,
            rho=rho,
            q=q,
            max_iter=max(max_iter, 20),
        )

        edge_diff_new = phi_new.index_select(0, edge_u) - phi_new.index_select(0, edge_v)
        primal_residual = edge_diff_new - z_new
        scaled_dual_new = scaled_dual + primal_residual

        primal_norm = float(
            (torch.linalg.norm(primal_residual) / (1.0 + torch.linalg.norm(z_new))).item()
        )
        dual_residual_vec = torch.zeros_like(phi)
        dual_residual_vec.index_add_(0, edge_u, z_new - z)
        dual_residual_vec.index_add_(0, edge_v, -(z_new - z))
        dual_norm = float(
            (rho * torch.linalg.norm(dual_residual_vec) / (1.0 + torch.linalg.norm(phi_new))).item()
        )
        actual_dual = rho * scaled_dual_new
        last_residual = inner_kkt_residual_torch(
            phi=phi_new,
            dual=actual_dual,
            U=U,
            h=h,
            lower=lower,
            upper=upper,
            lambda_value=lambda_value,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            atol=max(tol, 1e-8),
        )

        phi = phi_new
        z = z_new
        scaled_dual = scaled_dual_new

        if primal_norm <= tol and dual_norm <= tol and last_residual <= 5.0 * tol:
            converged = True
            break

    return phi, scaled_dual, iterations, converged, float(last_residual)
