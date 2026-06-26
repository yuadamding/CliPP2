from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...io.data import TumorData
from .graph_ops import graph_adjoint_edges, graph_forward_edges, project_dual_ball
from .types import ObjectiveTerms, TorchRuntime


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


DEFAULT_INNER_KKT_CHECK_EVERY = 8
DEFAULT_BOX_PHI_ATOL = 1e-8
DEFAULT_BOX_MAX_ITER = 32


def validate_lambda_value(lambda_value: float) -> float:
    value = float(lambda_value)
    if not np.isfinite(value):
        raise ValueError("lambda_value must be finite.")
    if value < 0.0:
        raise ValueError("lambda_value must be nonnegative.")
    return value


def _box_qp_sweeps_for_atol(
    phi_atol: float = DEFAULT_BOX_PHI_ATOL,
    *,
    max_iter: int = DEFAULT_BOX_MAX_ITER,
) -> int:
    atol = float(phi_atol)
    if not np.isfinite(atol) or atol <= 0.0:
        atol = DEFAULT_BOX_PHI_ATOL
    requested = int(np.ceil(np.log2(1.0 / min(max(atol, np.finfo(float).tiny), 1.0)))) + 1
    return max(16, min(max(int(max_iter), 16), requested))


def _inner_kkt_audit_due(
    *,
    iteration: int,
    max_iter: int,
    kkt_check_every: int,
    cheap_converged: bool,
) -> bool:
    if bool(cheap_converged):
        return True
    if int(iteration) >= int(max_iter):
        return True
    check_every = max(int(kkt_check_every), 1)
    return int(iteration) % check_every == 0


def _binary_entropy_offset_torch(weight: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(weight, min=0.0, max=1.0)
    positive = clipped > 0.0
    one_minus = 1.0 - clipped
    positive_complement = one_minus > 0.0
    term = torch.zeros_like(clipped)
    term = torch.where(positive, term + clipped * torch.log(clipped), term)
    term = torch.where(positive_complement, term + one_minus * torch.log(one_minus), term)
    return term


def resolve_runtime(device: str | None, *, dtype: str | None = None) -> TorchRuntime:
    requested = "auto" if device is None else str(device).strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested Torch device {device!r}, but CUDA is not available. "
            "Use device='cpu' or device='auto' to permit CPU execution."
        )
    runtime_device = torch.device(requested)
    requested_dtype = "auto" if dtype is None else str(dtype).strip().lower()
    if requested_dtype == "auto":
        runtime_dtype = torch.float32 if runtime_device.type == "cuda" else torch.float64
    elif requested_dtype == "float32":
        runtime_dtype = torch.float32
    elif requested_dtype == "float64":
        runtime_dtype = torch.float64
    else:
        raise ValueError(f"Unknown runtime dtype: {dtype}")
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


def _major_prior_log_tensors(
    major_prior: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prior = float(major_prior)
    if not np.isfinite(prior) or not (0.0 < prior < 1.0):
        raise ValueError("major_prior must lie strictly in (0, 1).")
    prior_tensor = torch.as_tensor(prior, dtype=dtype, device=device)
    return torch.log1p(-prior_tensor), torch.log(prior_tensor)


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
    log_prior_minor, log_prior_major = _major_prior_log_tensors(
        major_prior,
        dtype=beta.dtype,
        device=beta.device,
    )
    prob_fixed = torch.clamp(beta * b_fixed, min=float(eps), max=float(1.0 - eps))
    fixed_loss = -(alt * torch.log(prob_fixed) + nonalt * torch.log1p(-prob_fixed))

    prob_minus = torch.clamp(beta * b_minus, min=float(eps), max=float(1.0 - eps))
    prob_plus = torch.clamp(beta * b_plus, min=float(eps), max=float(1.0 - eps))
    log_minor = alt * torch.log(prob_minus) + nonalt * torch.log1p(-prob_minus) + log_prior_minor
    log_major = alt * torch.log(prob_plus) + nonalt * torch.log1p(-prob_plus) + log_prior_major
    amb_loss = -torch.logaddexp(log_minor, log_major)
    return torch.where(_as_loss_shape(ambiguous, beta), amb_loss, fixed_loss)


def cell_terms_torch(
    data: TorchTumorData,
    phi: torch.Tensor,
    *,
    major_prior: float,
    eps: float,
) -> TorchCellTerms:
    alt = data.alt
    nonalt = data.nonalt
    amb = data.ambiguous
    log_prior_minor, log_prior_major = _major_prior_log_tensors(
        major_prior,
        dtype=phi.dtype,
        device=phi.device,
    )

    prob_fixed, slope_fixed = clip_probability_and_slope(phi, data.b_fixed, eps)
    loss_fixed = -(alt * torch.log(prob_fixed) + nonalt * torch.log1p(-prob_fixed))
    grad_fixed_state, curvature_fixed = state_log_kernel_grad_and_curvature(
        alt=alt,
        nonalt=nonalt,
        prob=prob_fixed,
        slope=slope_fixed,
    )

    prob_minus, slope_minus = clip_probability_and_slope(phi, data.b_minus, eps)
    prob_plus, slope_plus = clip_probability_and_slope(phi, data.b_plus, eps)
    log_minor = alt * torch.log(prob_minus) + nonalt * torch.log1p(-prob_minus) + log_prior_minor
    log_major = alt * torch.log(prob_plus) + nonalt * torch.log1p(-prob_plus) + log_prior_major
    norm = torch.logaddexp(log_minor, log_major)
    gamma = torch.sigmoid(log_major - log_minor)
    grad_minus, curvature_minus = state_log_kernel_grad_and_curvature(
        alt=alt,
        nonalt=nonalt,
        prob=prob_minus,
        slope=slope_minus,
    )
    grad_plus, curvature_plus = state_log_kernel_grad_and_curvature(
        alt=alt,
        nonalt=nonalt,
        prob=prob_plus,
        slope=slope_plus,
    )
    amb_loss = -norm
    amb_grad = -((1.0 - gamma) * grad_minus + gamma * grad_plus)
    amb_curvature = (1.0 - gamma) * curvature_minus + gamma * curvature_plus

    loss = torch.where(amb, amb_loss, loss_fixed)
    grad = torch.where(amb, amb_grad, -grad_fixed_state)
    hess_upper = torch.where(amb, amb_curvature, curvature_fixed)
    gamma_major = torch.where(amb, gamma, torch.ones_like(gamma))

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
    alt = data.alt
    nonalt = data.nonalt
    amb = data.ambiguous
    log_prior_minor, log_prior_major = _major_prior_log_tensors(
        major_prior,
        dtype=phi.dtype,
        device=phi.device,
    )

    prob_fixed, slope_fixed = clip_probability_and_slope(phi, data.b_fixed, eps)
    loss_fixed = -(alt * torch.log(prob_fixed) + nonalt * torch.log1p(-prob_fixed))
    grad_fixed_state, curvature_fixed = state_log_kernel_grad_and_curvature(
        alt=alt,
        nonalt=nonalt,
        prob=prob_fixed,
        slope=slope_fixed,
    )

    omega = torch.clamp(omega_major, min=0.0, max=1.0)
    prob_minus, slope_minus = clip_probability_and_slope(phi, data.b_minus, eps)
    prob_plus, slope_plus = clip_probability_and_slope(phi, data.b_plus, eps)
    log_minor = alt * torch.log(prob_minus) + nonalt * torch.log1p(-prob_minus) + log_prior_minor
    log_major = alt * torch.log(prob_plus) + nonalt * torch.log1p(-prob_plus) + log_prior_major
    grad_minus, curvature_minus = state_log_kernel_grad_and_curvature(
        alt=alt,
        nonalt=nonalt,
        prob=prob_minus,
        slope=slope_minus,
    )
    grad_plus, curvature_plus = state_log_kernel_grad_and_curvature(
        alt=alt,
        nonalt=nonalt,
        prob=prob_plus,
        slope=slope_plus,
    )

    loss_major = -log_major
    loss_minor = -log_minor
    entropy_offset = _binary_entropy_offset_torch(omega)
    amb_loss = (1.0 - omega) * loss_minor + omega * loss_major + entropy_offset
    amb_grad = -((1.0 - omega) * grad_minus + omega * grad_plus)
    amb_curvature = (1.0 - omega) * curvature_minus + omega * curvature_plus

    loss = torch.where(amb, amb_loss, loss_fixed)
    grad = torch.where(amb, amb_grad, -grad_fixed_state)
    hess_upper = torch.where(amb, amb_curvature, curvature_fixed)
    gamma_major = torch.where(amb, omega, torch.ones_like(omega))

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
    lambda_value = validate_lambda_value(lambda_value)
    if lambda_value <= 0.0 or edge_u.numel() == 0:
        return torch.zeros((), dtype=phi.dtype, device=phi.device)
    diffs = graph_forward_edges(phi, edge_u=edge_u, edge_v=edge_v)
    return torch.as_tensor(float(lambda_value), dtype=phi.dtype, device=phi.device) * torch.sum(edge_w * torch.linalg.norm(diffs, dim=1))


def objective_terms_torch(
    data: TorchTumorData,
    phi: torch.Tensor,
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
    major_prior: float,
    eps: float,
) -> ObjectiveTerms:
    terms = cell_terms_torch(data, phi, major_prior=major_prior, eps=eps)
    fit_loss = torch.sum(terms.loss)
    penalty = pairwise_penalty_torch(
        phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
    )
    return ObjectiveTerms(
        fit=fit_loss,
        penalty=penalty,
        total=fit_loss + penalty,
        gamma_major=terms.gamma_major,
    )


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
    terms = objective_terms_torch(
        data,
        phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        major_prior=major_prior,
        eps=eps,
    )
    return float(terms.fit.item()), float(terms.penalty.item()), float(terms.total.item()), terms.gamma_major


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


def graph_fusion_kkt_residual_from_grad_torch(
    *,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    dual_kkt: torch.Tensor | None,
    lower: torch.Tensor,
    upper: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
    atol: float,
) -> dict[str, float]:
    lambda_value = validate_lambda_value(lambda_value)
    if dual_kkt is None or tuple(dual_kkt.shape) != (int(edge_u.numel()), int(phi.shape[1])):
        dual = torch.zeros((int(edge_u.numel()), int(phi.shape[1])), dtype=phi.dtype, device=phi.device)
    else:
        dual = dual_kkt.to(dtype=phi.dtype, device=phi.device)

    adj = torch.zeros_like(phi)
    if edge_u.numel() > 0 and lambda_value > 0.0:
        adj = graph_adjoint_edges(
            dual,
            edge_u=edge_u,
            edge_v=edge_v,
            num_nodes=int(phi.shape[0]),
        )
    total_grad = grad_smooth + adj
    stat = stationarity_residual_torch(total_grad=total_grad, phi=phi, lower=lower, upper=upper, atol=atol)
    smooth_gradient_norm = float(torch.linalg.norm(grad_smooth).item())
    fusion_adjustment_norm = float(torch.linalg.norm(adj).item())
    projected_stationarity_norm = float(torch.linalg.norm(stat).item())
    stationarity_normalizer = float(1.0 + smooth_gradient_norm + fusion_adjustment_norm)
    stationarity_residual = float(projected_stationarity_norm / max(stationarity_normalizer, 1e-300))
    lower_active = phi <= lower + float(atol)
    upper_active = phi >= upper - float(atol)
    frozen = upper <= lower + float(atol)
    interior = ~(lower_active | upper_active | frozen)
    diagnostic_upper_active = upper_active & ~frozen
    diagnostic_lower_active = lower_active & ~upper_active & ~frozen
    box_violation = torch.maximum(torch.clamp(lower - phi, min=0.0), torch.clamp(phi - upper, min=0.0))
    box_primal_violation = float(torch.max(box_violation).item()) if box_violation.numel() else 0.0
    box_scale = 1.0 + max(
        float(torch.max(torch.abs(lower)).item()) if lower.numel() else 0.0,
        float(torch.max(torch.abs(upper)).item()) if upper.numel() else 0.0,
    )
    box_residual = box_primal_violation / max(box_scale, 1e-300)

    if edge_u.numel() == 0 or lambda_value <= 0.0:
        edge_subgradient_residual = 0.0
        dual_ball_residual = 0.0
    else:
        diff = graph_forward_edges(phi, edge_u=edge_u, edge_v=edge_v)
        diff_norm = torch.linalg.norm(diff, dim=1)
        dual_norm = torch.linalg.norm(dual, dim=1)
        radius = float(lambda_value) * edge_w
        edge_resid = torch.zeros_like(diff_norm)
        active = diff_norm > float(atol)
        if torch.any(active):
            target = radius[active, None] * diff[active] / diff_norm[active, None].clamp_min(float(atol))
            edge_resid[active] = torch.linalg.norm(dual[active] - target, dim=1)

        ball_resid = torch.clamp(dual_norm - radius, min=0.0)
        if torch.any(~active):
            edge_resid[~active] = ball_resid[~active]

        denom = 1.0 + float(torch.max(radius).item()) if radius.numel() else 1.0
        edge_subgradient_residual = float((torch.max(edge_resid) / denom).item()) if edge_resid.numel() else 0.0
        dual_ball_residual = float((torch.max(ball_resid) / denom).item()) if ball_resid.numel() else 0.0

    return {
        "stationarity_residual": stationarity_residual,
        "projected_stationarity_residual": stationarity_residual,
        "projected_stationarity_norm": projected_stationarity_norm,
        "stationarity_normalizer": stationarity_normalizer,
        "smooth_gradient_norm": smooth_gradient_norm,
        "fusion_adjustment_norm": fusion_adjustment_norm,
        "edge_subgradient_residual": edge_subgradient_residual,
        "dual_ball_residual": dual_ball_residual,
        "box_primal_violation": box_primal_violation,
        "num_interior_coordinates": int(torch.sum(interior).item()),
        "num_lower_active_coordinates": int(torch.sum(diagnostic_lower_active).item()),
        "num_upper_active_coordinates": int(torch.sum(diagnostic_upper_active).item()),
        "num_frozen_coordinates": int(torch.sum(frozen).item()),
        "box_residual": float(box_residual),
        "kkt_residual": max(
            stationarity_residual,
            edge_subgradient_residual,
            dual_ball_residual,
            float(box_residual),
        ),
    }


def refine_graph_fusion_dual_certificate_torch(
    *,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    dual_kkt: torch.Tensor | None,
    lower: torch.Tensor,
    upper: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
    atol: float,
    max_iter: int = 96,
) -> dict[str, object]:
    lambda_value = validate_lambda_value(lambda_value)
    before_diag = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad_smooth,
        dual_kkt=dual_kkt,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        atol=atol,
    )
    if edge_u.numel() == 0 or lambda_value <= 0.0:
        dual = torch.zeros((int(edge_u.numel()), int(phi.shape[1])), dtype=phi.dtype, device=phi.device)
        after_diag = graph_fusion_kkt_residual_from_grad_torch(
            phi=phi,
            grad_smooth=grad_smooth,
            dual_kkt=dual,
            lower=lower,
            upper=upper,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            lambda_value=lambda_value,
            atol=atol,
        )
        return {
            "dual": dual,
            "diag": after_diag,
            "status": "zero_penalty_no_dual_needed",
            "dual_refined": False,
            "fused_edges": 0,
            "nonzero_edges": 0,
            "stationarity_before": float(before_diag["stationarity_residual"]),
            "stationarity_after": float(after_diag["stationarity_residual"]),
        }

    diff = graph_forward_edges(phi, edge_u=edge_u, edge_v=edge_v)
    diff_norm = torch.linalg.norm(diff, dim=1)
    radius = float(lambda_value) * edge_w
    active = diff_norm > float(atol)
    fused = ~active
    dual = torch.zeros((int(edge_u.numel()), int(phi.shape[1])), dtype=phi.dtype, device=phi.device)
    if torch.any(active):
        dual[active] = radius[active, None] * diff[active] / diff_norm[active, None].clamp_min(float(atol))
    if torch.any(fused) and dual_kkt is not None and tuple(dual_kkt.shape) == tuple(dual.shape):
        dual[fused] = dual_kkt.to(dtype=phi.dtype, device=phi.device)[fused]
        fused_norm = torch.linalg.norm(dual[fused], dim=1)
        fused_radius = radius[fused]
        dual[fused] = project_dual_ball(dual[fused], fused_radius)

    best_dual = dual.clone()
    best_diag = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad_smooth,
        dual_kkt=dual,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        atol=atol,
    )
    best_residual = float(best_diag["kkt_residual"])
    if torch.any(fused):
        degree = torch.bincount(
            torch.cat([edge_u, edge_v]),
            minlength=int(phi.shape[0]),
        ).max()
        step = 0.25 / max(float(degree.item()), 1.0)
        for _ in range(max(int(max_iter), 1)):
            adj = graph_adjoint_edges(
                dual,
                edge_u=edge_u,
                edge_v=edge_v,
                num_nodes=int(phi.shape[0]),
            )
            total_grad = grad_smooth + adj
            stat = stationarity_residual_torch(
                total_grad=total_grad,
                phi=phi,
                lower=lower,
                upper=upper,
                atol=atol,
            )
            dual[fused] = dual[fused] - float(step) * (
                graph_forward_edges(stat, edge_u=edge_u, edge_v=edge_v)[fused]
            )
            fused_radius = radius[fused]
            dual[fused] = project_dual_ball(dual[fused], fused_radius)
            diag = graph_fusion_kkt_residual_from_grad_torch(
                phi=phi,
                grad_smooth=grad_smooth,
                dual_kkt=dual,
                lower=lower,
                upper=upper,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                lambda_value=lambda_value,
                atol=atol,
            )
            residual = float(diag["kkt_residual"])
            if residual < best_residual:
                best_residual = residual
                best_diag = diag
                best_dual = dual.clone()

    return {
        "dual": best_dual,
        "diag": best_diag,
        "status": "refined_fused_edge_dual" if torch.any(fused) else "analytic_nonfused_dual",
        "dual_refined": bool(torch.any(fused)),
        "fused_edges": int(torch.sum(fused).item()),
        "nonzero_edges": int(torch.sum(active).item()),
        "stationarity_before": float(before_diag["stationarity_residual"]),
        "stationarity_after": float(best_diag["stationarity_residual"]),
    }


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
    lambda_value = validate_lambda_value(lambda_value)
    diag = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=h * (phi - U),
        dual_kkt=dual,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        atol=atol,
    )
    return float(diag["kkt_residual"])


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
    kkt_check_every: int = DEFAULT_INNER_KKT_CHECK_EVERY,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool, float]:
    lambda_value = validate_lambda_value(lambda_value)
    phi = torch.minimum(torch.maximum(phi_start.to(dtype=runtime.dtype, device=runtime.device), lower), upper)
    if lambda_value <= 0.0 or edge_u.numel() == 0:
        projected = torch.minimum(torch.maximum(U, lower), upper)
        total_grad = h * (projected - U)
        stat = stationarity_residual_torch(total_grad=total_grad, phi=projected, lower=lower, upper=upper, atol=tol)
        residual = float((torch.linalg.norm(stat) / (1.0 + torch.linalg.norm(projected))).item())
        empty_dual = torch.zeros((0, phi.shape[1]), dtype=runtime.dtype, device=runtime.device)
        return projected, empty_dual, empty_dual, 1, residual <= tol, residual

    if dual_start is not None and tuple(dual_start.shape) == (int(edge_u.numel()), int(phi.shape[1])):
        dual = dual_start.to(dtype=runtime.dtype, device=runtime.device)
    else:
        dual = torch.zeros((int(edge_u.numel()), int(phi.shape[1])), dtype=runtime.dtype, device=runtime.device)
    bar = phi.clone()
    del degree_bound
    eta = 0.99
    node_degree = torch.bincount(
        torch.cat([edge_u, edge_v]),
        minlength=int(num_mutations),
    ).to(dtype=runtime.dtype, device=runtime.device)
    tau_node = (eta / node_degree.clamp_min(1.0))[:, None]
    sigma_edge = eta / 2.0
    radius = float(lambda_value) * edge_w

    converged = False
    iterations = 0
    last_residual = np.inf
    actual_max_iter = max(int(max_iter), 10)
    for inner_iter in range(actual_max_iter):
        iterations = inner_iter + 1
        edge_diff = graph_forward_edges(bar, edge_u=edge_u, edge_v=edge_v)
        dual_trial = dual + sigma_edge * edge_diff
        dual_new = project_dual_ball(dual_trial, radius)

        adj = graph_adjoint_edges(
            dual_new,
            edge_u=edge_u,
            edge_v=edge_v,
            num_nodes=int(phi.shape[0]),
        )
        primal_base = phi - tau_node * adj
        phi_new = (primal_base + tau_node * h * U) / (1.0 + tau_node * h)
        phi_new = torch.minimum(torch.maximum(phi_new, lower), upper)
        bar = phi_new + (phi_new - phi)

        audit_due = _inner_kkt_audit_due(
            iteration=iterations,
            max_iter=actual_max_iter,
            kkt_check_every=kkt_check_every,
            cheap_converged=False,
        )
        if audit_due:
            primal_delta = float((torch.linalg.norm(phi_new - phi) / (1.0 + torch.linalg.norm(phi))).item())
            dual_delta = float((torch.linalg.norm(dual_new - dual) / (1.0 + torch.linalg.norm(dual))).item())
        phi = phi_new
        dual = dual_new

        if audit_due:
            cheap_converged = bool(primal_delta <= tol and dual_delta <= tol)
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
                atol=tol,
            )
            if cheap_converged and last_residual <= 5.0 * tol:
                converged = True
                break

    return phi, dual, dual, iterations, converged, float(last_residual)


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

    mid = 0.5 * (lo + hi)
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
    dual_start_is_actual: bool = False,
    kkt_check_every: int = DEFAULT_INNER_KKT_CHECK_EVERY,
    box_phi_atol: float = DEFAULT_BOX_PHI_ATOL,
    box_max_iter: int = DEFAULT_BOX_MAX_ITER,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool, float]:
    lambda_value = validate_lambda_value(lambda_value)
    phi = torch.minimum(torch.maximum(phi_start.to(dtype=runtime.dtype, device=runtime.device), lower), upper)
    if lambda_value <= 0.0 or edge_u.numel() == 0:
        projected = torch.minimum(torch.maximum(U, lower), upper)
        total_grad = h * (projected - U)
        stat = stationarity_residual_torch(total_grad=total_grad, phi=projected, lower=lower, upper=upper, atol=tol)
        residual = float((torch.linalg.norm(stat) / (1.0 + torch.linalg.norm(projected))).item())
        empty_dual = torch.zeros((0, phi.shape[1]), dtype=runtime.dtype, device=runtime.device)
        return projected, empty_dual, empty_dual, 1, residual <= tol, residual

    rho = float(torch.clamp(torch.median(h), min=1e-3, max=1e3).item())
    radius = float(lambda_value) * edge_w
    if dual_start is not None and tuple(dual_start.shape) == (int(edge_u.numel()), int(phi.shape[1])):
        initial_dual = dual_start.to(dtype=runtime.dtype, device=runtime.device)
        if bool(dual_start_is_actual):
            initial_dual = project_dual_ball(initial_dual, radius)
            scaled_dual = initial_dual / rho
        else:
            scaled_dual = initial_dual
    else:
        scaled_dual = torch.zeros((int(edge_u.numel()), int(phi.shape[1])), dtype=runtime.dtype, device=runtime.device)

    shrink_radius = radius / rho

    converged = False
    iterations = 0
    last_residual = np.inf
    z = graph_forward_edges(phi, edge_u=edge_u, edge_v=edge_v)
    actual_dual = rho * scaled_dual
    actual_max_iter = max(int(max_iter), 10)
    box_iter = _box_qp_sweeps_for_atol(box_phi_atol, max_iter=box_max_iter)

    for inner_iter in range(actual_max_iter):
        iterations = inner_iter + 1
        edge_diff = graph_forward_edges(phi, edge_u=edge_u, edge_v=edge_v)
        z_argument = edge_diff + scaled_dual
        z_norm = torch.linalg.norm(z_argument, dim=1, keepdim=True)
        shrink = torch.clamp(1.0 - shrink_radius[:, None] / z_norm.clamp_min(1e-12), min=0.0)
        z_new = shrink * z_argument

        rhs_edge = z_new - scaled_dual
        q = graph_adjoint_edges(
            rhs_edge,
            edge_u=edge_u,
            edge_v=edge_v,
            num_nodes=int(phi.shape[0]),
        )
        phi_new = _complete_graph_isotropic_box_qp_torch(
            U=U,
            h=h,
            lower=lower,
            upper=upper,
            rho=rho,
            q=q,
            max_iter=box_iter,
        )

        edge_diff_new = graph_forward_edges(phi_new, edge_u=edge_u, edge_v=edge_v)
        primal_residual = edge_diff_new - z_new
        scaled_dual_new = scaled_dual + primal_residual

        actual_dual = rho * scaled_dual_new

        audit_due = _inner_kkt_audit_due(
            iteration=iterations,
            max_iter=actual_max_iter,
            kkt_check_every=kkt_check_every,
            cheap_converged=False,
        )
        if audit_due:
            primal_norm = float(
                (torch.linalg.norm(primal_residual) / (1.0 + torch.linalg.norm(z_new))).item()
            )
            dual_residual_vec = graph_adjoint_edges(
                z_new - z,
                edge_u=edge_u,
                edge_v=edge_v,
                num_nodes=int(phi.shape[0]),
            )
            dual_norm = float(
                (rho * torch.linalg.norm(dual_residual_vec) / (1.0 + torch.linalg.norm(phi_new))).item()
            )
            cheap_converged = bool(primal_norm <= tol and dual_norm <= tol)

        phi = phi_new
        z = z_new
        scaled_dual = scaled_dual_new

        if audit_due:
            last_residual = inner_kkt_residual_torch(
                phi=phi,
                dual=actual_dual,
                U=U,
                h=h,
                lower=lower,
                upper=upper,
                lambda_value=lambda_value,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                atol=tol,
            )
            if cheap_converged and last_residual <= 5.0 * tol:
                converged = True
                break

    return phi, scaled_dual, actual_dual, iterations, converged, float(last_residual)
