from __future__ import annotations

import numpy as np
import torch

from ...io.data import TumorData
from .torch_backend import TorchRuntime, cell_loss_grid_torch


def _golden_section_minimize(
    objective,
    *,
    left: float,
    right: float,
    tol: float,
    max_iter: int,
) -> tuple[float, float]:
    if right <= left + 1e-12:
        value = float(objective(np.asarray([left], dtype=np.float64))[0])
        return float(left), value

    ratio = 0.5 * (np.sqrt(5.0) - 1.0)
    x1 = right - ratio * (right - left)
    x2 = left + ratio * (right - left)
    f1 = float(objective(np.asarray([x1], dtype=np.float64))[0])
    f2 = float(objective(np.asarray([x2], dtype=np.float64))[0])

    for _ in range(max(int(max_iter), 8)):
        if abs(right - left) <= tol * (1.0 + abs(left) + abs(right)):
            break
        if f1 <= f2:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - ratio * (right - left)
            f1 = float(objective(np.asarray([x1], dtype=np.float64))[0])
        else:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + ratio * (right - left)
            f2 = float(objective(np.asarray([x2], dtype=np.float64))[0])

    if f1 <= f2:
        return float(x1), float(f1)
    return float(x2), float(f2)


def _cell_loss_grid_numpy(
    beta_values: np.ndarray,
    *,
    alt: float,
    total: float,
    b_minus: float,
    b_plus: float,
    b_fixed: float,
    ambiguous: bool,
    major_prior: float,
    eps: float,
) -> np.ndarray:
    beta = np.asarray(beta_values, dtype=np.float64)
    alt = float(alt)
    total = float(total)
    nonalt = total - alt
    if not ambiguous:
        prob = np.clip(beta * float(b_fixed), eps, 1.0 - eps)
        return -(alt * np.log(prob) + nonalt * np.log1p(-prob))

    prob_minus = np.clip(beta * float(b_minus), eps, 1.0 - eps)
    prob_plus = np.clip(beta * float(b_plus), eps, 1.0 - eps)
    log_minor = alt * np.log(prob_minus) + nonalt * np.log1p(-prob_minus) + float(np.log(max(1.0 - major_prior, eps)))
    log_major = alt * np.log(prob_plus) + nonalt * np.log1p(-prob_plus) + float(np.log(max(major_prior, eps)))
    return -np.logaddexp(log_minor, log_major)


def _sample_loss_grid_numpy(
    beta_values: np.ndarray,
    *,
    alt: np.ndarray,
    total: np.ndarray,
    b_minus: np.ndarray,
    b_plus: np.ndarray,
    b_fixed: np.ndarray,
    ambiguous: np.ndarray,
    major_prior: float,
    eps: float,
) -> np.ndarray:
    beta = np.asarray(beta_values, dtype=np.float64)
    losses = np.zeros(beta.shape, dtype=np.float64)
    for idx in range(int(alt.shape[0])):
        losses += _cell_loss_grid_numpy(
            beta,
            alt=float(alt[idx]),
            total=float(total[idx]),
            b_minus=float(b_minus[idx]),
            b_plus=float(b_plus[idx]),
            b_fixed=float(b_fixed[idx]),
            ambiguous=bool(ambiguous[idx]),
            major_prior=major_prior,
            eps=eps,
        )
    return losses


def _batched_log_grid(
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    num_points: int,
    runtime: TorchRuntime,
) -> torch.Tensor:
    lower_t = torch.as_tensor(np.asarray(lower, dtype=np.float64), dtype=runtime.dtype, device=runtime.device)
    upper_t = torch.as_tensor(np.asarray(upper, dtype=np.float64), dtype=runtime.dtype, device=runtime.device)
    t = torch.linspace(0.0, 1.0, steps=int(num_points), dtype=runtime.dtype, device=runtime.device)
    return torch.exp(torch.log(lower_t).unsqueeze(-1) + (torch.log(upper_t) - torch.log(lower_t)).unsqueeze(-1) * t)


def compute_exact_observed_data_pilot(
    data: TumorData,
    *,
    runtime: TorchRuntime,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    alt = np.asarray(data.alt_counts, dtype=np.float64).reshape(-1)
    total = np.asarray(data.total_counts, dtype=np.float64).reshape(-1)
    b_minus = (np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.minor_cn, dtype=np.float64)).reshape(-1)
    b_plus = (np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.major_cn, dtype=np.float64)).reshape(-1)
    b_fixed = (np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.fixed_multiplicity, dtype=np.float64)).reshape(-1)
    ambiguous = np.asarray(data.multiplicity_estimation_mask, dtype=bool).reshape(-1)
    lower = np.full_like(alt, float(eps), dtype=np.float64)
    upper = np.asarray(data.phi_upper, dtype=np.float64).reshape(-1)
    hint = np.clip(np.asarray(data.phi_init, dtype=np.float64).reshape(-1), lower, upper)
    local_left = np.maximum(lower, hint / 3.0)
    local_right = np.minimum(upper, hint * 3.0)
    collapsed = local_right <= local_left + 1e-12
    local_left[collapsed] = lower[collapsed]
    local_right[collapsed] = upper[collapsed]

    grid = _batched_log_grid(local_left, local_right, num_points=25, runtime=runtime)
    losses = cell_loss_grid_torch(
        grid,
        alt=torch.as_tensor(alt[:, None], dtype=runtime.dtype, device=runtime.device),
        total=torch.as_tensor(total[:, None], dtype=runtime.dtype, device=runtime.device),
        b_minus=torch.as_tensor(b_minus[:, None], dtype=runtime.dtype, device=runtime.device),
        b_plus=torch.as_tensor(b_plus[:, None], dtype=runtime.dtype, device=runtime.device),
        b_fixed=torch.as_tensor(b_fixed[:, None], dtype=runtime.dtype, device=runtime.device),
        ambiguous=torch.as_tensor(ambiguous[:, None], dtype=torch.bool, device=runtime.device),
        major_prior=major_prior,
        eps=eps,
    ).detach().cpu().numpy()
    grid_np = grid.detach().cpu().numpy()
    best_index = np.argmin(losses, axis=1)
    best_beta = grid_np[np.arange(grid_np.shape[0]), best_index]
    left = np.where(best_index == 0, lower, grid_np[np.arange(grid_np.shape[0]), np.maximum(best_index - 1, 0)])
    right = np.where(
        best_index == grid_np.shape[1] - 1,
        upper,
        grid_np[np.arange(grid_np.shape[0]), np.minimum(best_index + 1, grid_np.shape[1] - 1)],
    )

    refined = np.zeros_like(best_beta, dtype=np.float64)
    for idx in range(best_beta.shape[0]):
        objective = lambda values, idx=idx: _cell_loss_grid_numpy(
            values,
            alt=float(alt[idx]),
            total=float(total[idx]),
            b_minus=float(b_minus[idx]),
            b_plus=float(b_plus[idx]),
            b_fixed=float(b_fixed[idx]),
            ambiguous=bool(ambiguous[idx]),
            major_prior=major_prior,
            eps=eps,
        )
        refined_beta, refined_value = _golden_section_minimize(
            objective,
            left=float(left[idx]),
            right=float(right[idx]),
            tol=tol,
            max_iter=max_iter,
        )
        best_value = float(_cell_loss_grid_numpy(
            np.asarray([best_beta[idx]], dtype=np.float64),
            alt=float(alt[idx]),
            total=float(total[idx]),
            b_minus=float(b_minus[idx]),
            b_plus=float(b_plus[idx]),
            b_fixed=float(b_fixed[idx]),
            ambiguous=bool(ambiguous[idx]),
            major_prior=major_prior,
            eps=eps,
        )[0])
        refined[idx] = refined_beta if refined_value <= best_value else best_beta[idx]
    return np.clip(refined.reshape(data.phi_init.shape), eps, np.asarray(data.phi_upper, dtype=np.float64))


def compute_pooled_observed_data_start(
    data: TumorData,
    *,
    runtime: TorchRuntime,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    beta_hints: np.ndarray | None = None,
) -> np.ndarray:
    ambiguous = np.asarray(data.multiplicity_estimation_mask, dtype=bool)
    b_minus = np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.minor_cn, dtype=np.float64)
    b_plus = np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.major_cn, dtype=np.float64)
    b_fixed = np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.fixed_multiplicity, dtype=np.float64)
    pooled = np.zeros((data.num_samples,), dtype=np.float64)

    for sample_idx in range(data.num_samples):
        lower = float(eps)
        upper = float(np.min(np.asarray(data.phi_upper[:, sample_idx], dtype=np.float64)))
        hint = None
        if beta_hints is not None:
            hint = float(np.median(np.asarray(beta_hints[:, sample_idx], dtype=np.float64)))
        local_left = max(lower, lower if hint is None else hint / 3.0)
        local_right = min(upper, upper if hint is None else hint * 3.0)
        if local_right <= local_left + 1e-12:
            local_left, local_right = lower, upper

        grid = _batched_log_grid(
            np.asarray([local_left], dtype=np.float64),
            np.asarray([local_right], dtype=np.float64),
            num_points=25,
            runtime=runtime,
        )[0]
        losses = cell_loss_grid_torch(
            grid.unsqueeze(0).expand(data.num_mutations, -1),
            alt=torch.as_tensor(np.asarray(data.alt_counts[:, sample_idx], dtype=np.float64)[:, None], dtype=runtime.dtype, device=runtime.device),
            total=torch.as_tensor(np.asarray(data.total_counts[:, sample_idx], dtype=np.float64)[:, None], dtype=runtime.dtype, device=runtime.device),
            b_minus=torch.as_tensor(b_minus[:, sample_idx][:, None], dtype=runtime.dtype, device=runtime.device),
            b_plus=torch.as_tensor(b_plus[:, sample_idx][:, None], dtype=runtime.dtype, device=runtime.device),
            b_fixed=torch.as_tensor(b_fixed[:, sample_idx][:, None], dtype=runtime.dtype, device=runtime.device),
            ambiguous=torch.as_tensor(ambiguous[:, sample_idx][:, None], dtype=torch.bool, device=runtime.device),
            major_prior=major_prior,
            eps=eps,
        )
        losses = torch.sum(losses, dim=0).detach().cpu().numpy()
        grid_np = grid.detach().cpu().numpy()
        best_index = int(np.argmin(losses))
        best_beta = float(grid_np[best_index])
        left = float(lower if best_index == 0 else grid_np[best_index - 1])
        right = float(upper if best_index == grid_np.shape[0] - 1 else grid_np[best_index + 1])

        objective = lambda values: _sample_loss_grid_numpy(
            values,
            alt=np.asarray(data.alt_counts[:, sample_idx], dtype=np.float64),
            total=np.asarray(data.total_counts[:, sample_idx], dtype=np.float64),
            b_minus=b_minus[:, sample_idx],
            b_plus=b_plus[:, sample_idx],
            b_fixed=b_fixed[:, sample_idx],
            ambiguous=ambiguous[:, sample_idx],
            major_prior=major_prior,
            eps=eps,
        )
        refined_beta, refined_value = _golden_section_minimize(
            objective,
            left=left,
            right=right,
            tol=tol,
            max_iter=max_iter,
        )
        best_value = float(objective(np.asarray([best_beta], dtype=np.float64))[0])
        pooled[sample_idx] = refined_beta if refined_value <= best_value else best_beta

    tiled = np.tile(pooled[None, :], (data.num_mutations, 1))
    return np.clip(tiled, eps, np.asarray(data.phi_upper, dtype=np.float64))

