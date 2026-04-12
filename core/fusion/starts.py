from __future__ import annotations

import numpy as np
import torch

from ...io.data import TumorData
from .torch_backend import TorchRuntime, cell_loss_grid_torch


_ROOT_SCAN_POINTS = 65


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


def _project_to_interval(value: float, left: float, right: float) -> float:
    return float(min(max(float(value), float(left)), float(right)))


def _interval_endpoints(
    *,
    left: float,
    right: float,
    include_hint: float | None = None,
) -> list[float]:
    candidates = [float(left), float(right)]
    if include_hint is not None and np.isfinite(include_hint):
        candidates.append(_project_to_interval(float(include_hint), float(left), float(right)))
    return [float(value) for value in np.unique(np.round(np.asarray(candidates, dtype=np.float64), 15))]


def _scaled_sum(sign_left: float, logabs_left: float, sign_right: float, logabs_right: float) -> float:
    if sign_left == 0.0:
        return float(sign_right)
    if sign_right == 0.0:
        return float(sign_left)
    anchor = max(float(logabs_left), float(logabs_right))
    return float(sign_left * np.exp(float(logabs_left) - anchor) + sign_right * np.exp(float(logabs_right) - anchor))


def _ambiguous_middle_stationarity(
    beta: float,
    *,
    alt: float,
    total: float,
    b_minus: float,
    b_plus: float,
    major_prior: float,
) -> float:
    tiny = np.finfo(np.float64).tiny
    beta = float(beta)
    alt = float(alt)
    total = float(total)
    nonalt = float(total - alt)

    base_minus = max(1.0 - float(b_minus) * beta, tiny)
    base_plus = max(1.0 - float(b_plus) * beta, tiny)
    delta_minus = alt - total * float(b_minus) * beta
    delta_plus = alt - total * float(b_plus) * beta

    sign_minus = float(np.sign(delta_minus))
    sign_plus = float(np.sign(delta_plus))
    if sign_minus == 0.0 and sign_plus == 0.0:
        return 0.0

    logabs_minus = (
        np.log(max(1.0 - float(major_prior), tiny))
        + alt * np.log(max(float(b_minus), tiny))
        + (nonalt - 1.0) * np.log(base_minus)
        + (np.log(max(abs(delta_minus), tiny)) if sign_minus != 0.0 else -np.inf)
    )
    logabs_plus = (
        np.log(max(float(major_prior), tiny))
        + alt * np.log(max(float(b_plus), tiny))
        + (nonalt - 1.0) * np.log(base_plus)
        + (np.log(max(abs(delta_plus), tiny)) if sign_plus != 0.0 else -np.inf)
    )
    return _scaled_sum(sign_minus, logabs_minus, sign_plus, logabs_plus)


def _bisect_root(
    function,
    *,
    left: float,
    right: float,
    tol: float,
    max_iter: int,
) -> float | None:
    left = float(left)
    right = float(right)
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        return None

    f_left = float(function(left))
    f_right = float(function(right))
    if not np.isfinite(f_left) or not np.isfinite(f_right):
        return None
    if abs(f_left) <= 1e-12:
        return left
    if abs(f_right) <= 1e-12:
        return right
    if f_left * f_right > 0.0:
        return None

    mid = 0.5 * (left + right)
    for _ in range(max(int(max_iter), 32)):
        mid = 0.5 * (left + right)
        f_mid = float(function(mid))
        if not np.isfinite(f_mid):
            return None
        if abs(f_mid) <= 1e-12 or abs(right - left) <= tol * (1.0 + abs(mid)):
            return float(mid)
        if f_left * f_mid <= 0.0:
            right = mid
            f_right = f_mid
        else:
            left = mid
            f_left = f_mid
    return float(mid)


def _middle_regime_roots(
    *,
    left: float,
    right: float,
    alt: float,
    total: float,
    b_minus: float,
    b_plus: float,
    major_prior: float,
    tol: float,
    max_iter: int,
) -> list[float]:
    left = float(left)
    right = float(right)
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        return []

    left_probe = np.nextafter(left, right)
    right_probe = np.nextafter(right, left)
    if not np.isfinite(left_probe) or not np.isfinite(right_probe) or right_probe <= left_probe:
        return []

    def stationarity(value: float) -> float:
        return _ambiguous_middle_stationarity(
            value,
            alt=alt,
            total=total,
            b_minus=b_minus,
            b_plus=b_plus,
            major_prior=major_prior,
        )

    probe = np.linspace(left_probe, right_probe, num=_ROOT_SCAN_POINTS, dtype=np.float64)
    values = np.asarray([stationarity(float(beta)) for beta in probe], dtype=np.float64)

    roots: list[float] = []
    for idx, current in enumerate(values):
        if not np.isfinite(current):
            continue
        if abs(float(current)) <= 1e-10:
            roots.append(float(probe[idx]))

    for left_beta, right_beta, left_value, right_value in zip(probe[:-1], probe[1:], values[:-1], values[1:]):
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            continue
        if left_value == 0.0 or right_value == 0.0:
            continue
        if left_value * right_value > 0.0:
            continue
        root = _bisect_root(
            stationarity,
            left=float(left_beta),
            right=float(right_beta),
            tol=tol,
            max_iter=max_iter,
        )
        if root is not None:
            roots.append(float(root))

    if not roots:
        return []
    return [float(value) for value in np.unique(np.round(np.asarray(roots, dtype=np.float64), 12))]


def _cell_candidate_betas(
    *,
    alt: float,
    total: float,
    b_minus: float,
    b_plus: float,
    b_fixed: float,
    ambiguous: bool,
    lower: float,
    upper: float,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    hint: float | None,
) -> np.ndarray:
    lower = float(lower)
    upper = float(upper)
    hint = None if hint is None else float(hint)
    if upper <= lower + 1e-12:
        return np.asarray([float(lower)], dtype=np.float64)

    if not ambiguous:
        if float(total) <= 0.0 or float(b_fixed) <= 0.0:
            return np.asarray([_project_to_interval(lower if hint is None else hint, lower, upper)], dtype=np.float64)
        p_hat = _project_to_interval(float(alt) / float(total), float(eps), float(1.0 - eps))
        return np.asarray([_project_to_interval(p_hat / float(b_fixed), lower, upper)], dtype=np.float64)

    if float(total) <= 0.0:
        return np.asarray([_project_to_interval(lower if hint is None else hint, lower, upper)], dtype=np.float64)

    b_low = float(min(b_minus, b_plus))
    b_high = float(max(b_minus, b_plus))
    if b_low <= 0.0 or b_high <= 0.0:
        return np.asarray([_project_to_interval(lower if hint is None else hint, lower, upper)], dtype=np.float64)

    r1 = float(eps) / b_high
    r2 = float(eps) / b_low
    r3 = float(1.0 - float(eps)) / b_high
    r4 = float(1.0 - float(eps)) / b_low

    candidates = _interval_endpoints(left=lower, right=upper, include_hint=hint)
    for kink in (r1, r2, r3, r4):
        if lower <= kink <= upper:
            candidates.append(float(kink))

    if float(total) > 0.0:
        beta_high = float(alt) / (float(total) * b_high)
        region2_left = max(lower, r1)
        region2_right = min(upper, r2)
        if region2_left < beta_high < region2_right:
            candidates.append(float(beta_high))

        beta_low = float(alt) / (float(total) * b_low)
        region4_left = max(lower, r3)
        region4_right = min(upper, r4)
        if region4_left < beta_low < region4_right:
            candidates.append(float(beta_low))

    middle_left = max(lower, r2)
    middle_right = min(upper, r3)
    if middle_right > middle_left + 1e-12:
        candidates.extend(
            _middle_regime_roots(
                left=middle_left,
                right=middle_right,
                alt=float(alt),
                total=float(total),
                b_minus=b_low,
                b_plus=b_high,
                major_prior=major_prior,
                tol=tol,
                max_iter=max_iter,
            )
        )

    candidate_array = np.unique(np.round(np.asarray(candidates, dtype=np.float64), 12))
    candidate_array = candidate_array[(candidate_array >= lower - 1e-12) & (candidate_array <= upper + 1e-12)]
    candidate_array = np.clip(candidate_array, lower, upper)
    if candidate_array.size == 0:
        candidate_array = np.asarray([_project_to_interval(lower if hint is None else hint, lower, upper)], dtype=np.float64)
    return candidate_array


def _exact_pilot_cell_beta(
    *,
    alt: float,
    total: float,
    b_minus: float,
    b_plus: float,
    b_fixed: float,
    ambiguous: bool,
    lower: float,
    upper: float,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    hint: float | None,
) -> float:
    candidate_array = _cell_candidate_betas(
        alt=alt,
        total=total,
        b_minus=b_minus,
        b_plus=b_plus,
        b_fixed=b_fixed,
        ambiguous=ambiguous,
        lower=lower,
        upper=upper,
        major_prior=major_prior,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
        hint=hint,
    )
    losses = _cell_loss_grid_numpy(
        candidate_array,
        alt=float(alt),
        total=float(total),
        b_minus=float(b_minus),
        b_plus=float(b_plus),
        b_fixed=float(b_fixed),
        ambiguous=bool(ambiguous),
        major_prior=major_prior,
        eps=eps,
    )
    best_loss = float(np.min(losses))
    best_mask = losses <= best_loss + 1e-12
    best_candidates = candidate_array[best_mask]
    if best_candidates.size == 1 or hint is None or not np.isfinite(hint):
        return float(best_candidates[0])
    hint_projected = _project_to_interval(hint, lower, upper)
    best_idx = int(np.argmin(np.abs(best_candidates - hint_projected)))
    return float(best_candidates[best_idx])


def _local_minimum_representatives(
    candidate_array: np.ndarray,
    losses: np.ndarray,
    *,
    hint: float | None,
    loss_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    if candidate_array.size == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty

    order = np.argsort(candidate_array, kind="stable")
    beta_sorted = candidate_array[order].astype(np.float64, copy=False)
    loss_sorted = losses[order].astype(np.float64, copy=False)

    blocks: list[tuple[int, int]] = []
    block_start = 0
    for idx in range(1, int(beta_sorted.size)):
        if abs(float(loss_sorted[idx]) - float(loss_sorted[idx - 1])) > loss_tol:
            blocks.append((block_start, idx))
            block_start = idx
    blocks.append((block_start, int(beta_sorted.size)))

    block_losses = np.asarray(
        [float(np.min(loss_sorted[start:stop])) for start, stop in blocks],
        dtype=np.float64,
    )
    representatives: list[float] = []
    representative_losses: list[float] = []
    projected_hint = None if hint is None or not np.isfinite(hint) else float(hint)

    for block_idx, (start, stop) in enumerate(blocks):
        current_loss = float(block_losses[block_idx])
        left_loss = float(block_losses[block_idx - 1]) if block_idx > 0 else float("inf")
        right_loss = float(block_losses[block_idx + 1]) if block_idx + 1 < len(blocks) else float("inf")
        if current_loss > left_loss + loss_tol or current_loss > right_loss + loss_tol:
            continue

        beta_block = beta_sorted[start:stop]
        if beta_block.size == 0:
            continue
        if projected_hint is None:
            representative = float(beta_block[len(beta_block) // 2])
        else:
            representative = float(beta_block[int(np.argmin(np.abs(beta_block - projected_hint)))])
        representatives.append(representative)
        representative_losses.append(current_loss)

    if not representatives:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty
    return (
        np.asarray(representatives, dtype=np.float64),
        np.asarray(representative_losses, dtype=np.float64),
    )


def _cell_best_two_betas(
    *,
    alt: float,
    total: float,
    b_minus: float,
    b_plus: float,
    b_fixed: float,
    ambiguous: bool,
    lower: float,
    upper: float,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    hint: float | None,
) -> tuple[float, float | None]:
    candidate_array = _cell_candidate_betas(
        alt=alt,
        total=total,
        b_minus=b_minus,
        b_plus=b_plus,
        b_fixed=b_fixed,
        ambiguous=ambiguous,
        lower=lower,
        upper=upper,
        major_prior=major_prior,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
        hint=hint,
    )
    losses = _cell_loss_grid_numpy(
        candidate_array,
        alt=float(alt),
        total=float(total),
        b_minus=float(b_minus),
        b_plus=float(b_plus),
        b_fixed=float(b_fixed),
        ambiguous=bool(ambiguous),
        major_prior=major_prior,
        eps=eps,
    )
    loss_tol = max(float(tol) * 10.0, 1e-10)
    local_betas, local_losses = _local_minimum_representatives(
        candidate_array,
        losses,
        hint=hint,
        loss_tol=loss_tol,
    )
    if local_betas.size == 0:
        order = np.argsort(losses, kind="stable")
        ordered_beta = candidate_array[order]
        ordered_losses = losses[order]
    else:
        if hint is None or not np.isfinite(hint):
            local_order = np.lexsort((local_betas, local_losses))
        else:
            local_order = np.lexsort((np.abs(local_betas - float(hint)), local_losses))
        ordered_beta = local_betas[local_order]
        ordered_losses = local_losses[local_order]

    primary = float(ordered_beta[0])
    beta_tol = max(float(tol) * 10.0, 1e-8 * max(1.0, abs(primary)))
    secondary: float | None = None
    primary_loss = float(ordered_losses[0])
    for beta, loss in zip(ordered_beta[1:], ordered_losses[1:]):
        if abs(float(loss) - primary_loss) <= loss_tol and abs(float(beta) - primary) <= beta_tol:
            continue
        if abs(float(beta) - primary) > beta_tol:
            secondary = float(beta)
            break
    return primary, secondary


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


def _cell_grad_numpy(
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
    nonalt = float(total - alt)

    if not ambiguous:
        linear = float(b_fixed) * beta
        prob = np.clip(linear, eps, 1.0 - eps)
        slope = np.where((linear > eps) & (linear < 1.0 - eps), float(b_fixed), 0.0)
        state_grad = slope * (alt / prob - nonalt / (1.0 - prob))
        return -state_grad

    linear_minus = float(b_minus) * beta
    linear_plus = float(b_plus) * beta
    prob_minus = np.clip(linear_minus, eps, 1.0 - eps)
    prob_plus = np.clip(linear_plus, eps, 1.0 - eps)
    slope_minus = np.where((linear_minus > eps) & (linear_minus < 1.0 - eps), float(b_minus), 0.0)
    slope_plus = np.where((linear_plus > eps) & (linear_plus < 1.0 - eps), float(b_plus), 0.0)

    grad_minus = slope_minus * (alt / prob_minus - nonalt / (1.0 - prob_minus))
    grad_plus = slope_plus * (alt / prob_plus - nonalt / (1.0 - prob_plus))

    log_minor = alt * np.log(prob_minus) + nonalt * np.log1p(-prob_minus) + float(np.log(max(1.0 - major_prior, eps)))
    log_major = alt * np.log(prob_plus) + nonalt * np.log1p(-prob_plus) + float(np.log(max(major_prior, eps)))
    gamma = np.exp(log_major - np.logaddexp(log_minor, log_major))
    return -((1.0 - gamma) * grad_minus + gamma * grad_plus)


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


def _cell_breakpoints(
    *,
    lower: float,
    upper: float,
    b_minus: float,
    b_plus: float,
    b_fixed: float,
    ambiguous: bool,
    eps: float,
) -> np.ndarray:
    points = [float(lower), float(upper)]
    if ambiguous:
        b_low = float(min(b_minus, b_plus))
        b_high = float(max(b_minus, b_plus))
        if b_low > 0.0 and b_high > 0.0:
            points.extend(
                [
                    float(eps) / b_high,
                    float(eps) / b_low,
                    float(1.0 - float(eps)) / b_high,
                    float(1.0 - float(eps)) / b_low,
                ]
            )
    elif float(b_fixed) > 0.0:
        points.extend(
            [
                float(eps) / float(b_fixed),
                float(1.0 - float(eps)) / float(b_fixed),
            ]
        )
    point_array = np.unique(np.round(np.asarray(points, dtype=np.float64), 12))
    point_array = point_array[(point_array >= float(lower) - 1e-12) & (point_array <= float(upper) + 1e-12)]
    return np.clip(point_array, float(lower), float(upper))


def _scan_roots(
    function,
    *,
    left: float,
    right: float,
    tol: float,
    max_iter: int,
    num_points: int = _ROOT_SCAN_POINTS,
) -> list[float]:
    left = float(left)
    right = float(right)
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        return []

    left_probe = np.nextafter(left, right)
    right_probe = np.nextafter(right, left)
    if not np.isfinite(left_probe) or not np.isfinite(right_probe) or right_probe <= left_probe:
        return []

    probe = np.linspace(left_probe, right_probe, num=max(int(num_points), 9), dtype=np.float64)
    values = np.asarray([float(function(beta)) for beta in probe], dtype=np.float64)

    roots: list[float] = []
    for idx, current in enumerate(values):
        if not np.isfinite(current):
            continue
        if abs(float(current)) <= 1e-10:
            roots.append(float(probe[idx]))

    for left_beta, right_beta, left_value, right_value in zip(probe[:-1], probe[1:], values[:-1], values[1:]):
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            continue
        if left_value == 0.0 or right_value == 0.0 or left_value * right_value > 0.0:
            continue
        root = _bisect_root(
            function,
            left=float(left_beta),
            right=float(right_beta),
            tol=tol,
            max_iter=max_iter,
        )
        if root is not None:
            roots.append(float(root))

    if not roots:
        return []
    return [float(value) for value in np.unique(np.round(np.asarray(roots, dtype=np.float64), 12))]


def _screen_hull_from_gradient_strip(
    *,
    alt: float,
    total: float,
    b_minus: float,
    b_plus: float,
    b_fixed: float,
    ambiguous: bool,
    lower: float,
    upper: float,
    major_prior: float,
    eps: float,
    threshold: float,
    tol: float,
    max_iter: int,
) -> tuple[float, float]:
    lower = float(lower)
    upper = float(upper)
    threshold = float(threshold)
    if upper <= lower + 1e-12 or threshold < 0.0:
        return lower, upper

    breakpoints = _cell_breakpoints(
        lower=lower,
        upper=upper,
        b_minus=b_minus,
        b_plus=b_plus,
        b_fixed=b_fixed,
        ambiguous=ambiguous,
        eps=eps,
    )
    if breakpoints.size < 2:
        return lower, upper

    def grad_fn(values: np.ndarray) -> np.ndarray:
        return _cell_grad_numpy(
            values,
            alt=alt,
            total=total,
            b_minus=b_minus,
            b_plus=b_plus,
            b_fixed=b_fixed,
            ambiguous=ambiguous,
            major_prior=major_prior,
            eps=eps,
        )

    all_points = list(breakpoints.tolist())
    for left_point, right_point in zip(breakpoints[:-1], breakpoints[1:]):
        if right_point <= left_point + 1e-12:
            continue
        all_points.extend(
            _scan_roots(
                lambda beta: float(grad_fn(np.asarray([beta], dtype=np.float64))[0] - threshold),
                left=float(left_point),
                right=float(right_point),
                tol=tol,
                max_iter=max_iter,
            )
        )
        all_points.extend(
            _scan_roots(
                lambda beta: float(grad_fn(np.asarray([beta], dtype=np.float64))[0] + threshold),
                left=float(left_point),
                right=float(right_point),
                tol=tol,
                max_iter=max_iter,
            )
        )

    partition = np.unique(np.round(np.asarray(all_points, dtype=np.float64), 12))
    partition = partition[(partition >= lower - 1e-12) & (partition <= upper + 1e-12)]
    partition = np.clip(partition, lower, upper)
    if partition.size == 0:
        return lower, upper

    admissible_left = float("inf")
    admissible_right = float("-inf")
    point_values = grad_fn(partition)
    point_mask = np.isfinite(point_values) & (np.abs(point_values) <= threshold + 1e-10)
    if np.any(point_mask):
        admissible_left = min(admissible_left, float(np.min(partition[point_mask])))
        admissible_right = max(admissible_right, float(np.max(partition[point_mask])))

    for left_point, right_point in zip(partition[:-1], partition[1:]):
        if right_point <= left_point + 1e-12:
            continue
        probe_left = np.nextafter(left_point, right_point)
        probe_right = np.nextafter(right_point, left_point)
        if not np.isfinite(probe_left) or not np.isfinite(probe_right) or probe_right <= probe_left:
            continue
        midpoint = 0.5 * (probe_left + probe_right)
        grad_mid = float(grad_fn(np.asarray([midpoint], dtype=np.float64))[0])
        if np.isfinite(grad_mid) and abs(grad_mid) <= threshold + 1e-10:
            admissible_left = min(admissible_left, float(left_point))
            admissible_right = max(admissible_right, float(right_point))

    if not np.isfinite(admissible_left) or not np.isfinite(admissible_right) or admissible_right < admissible_left:
        return lower, upper
    return max(lower, admissible_left), min(upper, admissible_right)


def compute_exact_observed_data_pilot(
    data: TumorData,
    *,
    runtime: TorchRuntime,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    primary, _, _ = compute_scalar_cell_wells(
        data,
        major_prior=major_prior,
        eps=eps,
        tol=tol,
        max_iter=max_iter,
    )
    return np.clip(primary, eps, np.asarray(data.phi_upper, dtype=np.float64))


def compute_scalar_cell_wells(
    data: TumorData,
    *,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alt = np.asarray(data.alt_counts, dtype=np.float64).reshape(-1)
    total = np.asarray(data.total_counts, dtype=np.float64).reshape(-1)
    b_minus = (np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.minor_cn, dtype=np.float64)).reshape(-1)
    b_plus = (np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.major_cn, dtype=np.float64)).reshape(-1)
    b_fixed = (np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.fixed_multiplicity, dtype=np.float64)).reshape(-1)
    ambiguous = np.asarray(data.multiplicity_estimation_mask, dtype=bool).reshape(-1)
    lower = np.full_like(alt, float(eps), dtype=np.float64)
    upper = np.asarray(data.phi_upper, dtype=np.float64).reshape(-1)
    hint = np.clip(np.asarray(data.phi_init, dtype=np.float64).reshape(-1), lower, upper)
    refined = np.zeros_like(hint, dtype=np.float64)
    secondary = np.full_like(hint, np.nan, dtype=np.float64)
    valid_secondary = np.zeros_like(hint, dtype=bool)

    fixed_mask = ~ambiguous
    if np.any(fixed_mask):
        fixed_total = total[fixed_mask]
        fixed_valid = (fixed_total > 0.0) & (b_fixed[fixed_mask] > 0.0)
        fixed_solution = hint[fixed_mask].copy()
        if np.any(fixed_valid):
            p_hat = np.clip(alt[fixed_mask][fixed_valid] / fixed_total[fixed_valid], eps, 1.0 - eps)
            beta_hat = p_hat / np.clip(b_fixed[fixed_mask][fixed_valid], eps, None)
            fixed_solution[fixed_valid] = np.minimum(
                np.maximum(beta_hat, lower[fixed_mask][fixed_valid]),
                upper[fixed_mask][fixed_valid],
            )
        refined[fixed_mask] = np.minimum(np.maximum(fixed_solution, lower[fixed_mask]), upper[fixed_mask])

    amb_indices = np.flatnonzero(ambiguous)
    for idx in amb_indices:
        primary, alternate = _cell_best_two_betas(
            alt=float(alt[idx]),
            total=float(total[idx]),
            b_minus=float(b_minus[idx]),
            b_plus=float(b_plus[idx]),
            b_fixed=float(b_fixed[idx]),
            ambiguous=True,
            lower=float(lower[idx]),
            upper=float(upper[idx]),
            major_prior=major_prior,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
            hint=float(hint[idx]),
        )
        refined[idx] = float(primary)
        if alternate is not None:
            secondary[idx] = float(alternate)
            valid_secondary[idx] = abs(float(alternate) - float(primary)) > max(float(tol) * 10.0, 1e-8)

    shape = data.phi_init.shape
    return (
        np.clip(refined.reshape(shape), eps, np.asarray(data.phi_upper, dtype=np.float64)),
        secondary.reshape(shape),
        valid_secondary.reshape(shape),
    )


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
    alt_t = torch.as_tensor(np.asarray(data.alt_counts, dtype=np.float64), dtype=runtime.dtype, device=runtime.device)
    total_t = torch.as_tensor(np.asarray(data.total_counts, dtype=np.float64), dtype=runtime.dtype, device=runtime.device)
    b_minus_t = torch.as_tensor(b_minus, dtype=runtime.dtype, device=runtime.device)
    b_plus_t = torch.as_tensor(b_plus, dtype=runtime.dtype, device=runtime.device)
    b_fixed_t = torch.as_tensor(b_fixed, dtype=runtime.dtype, device=runtime.device)
    ambiguous_t = torch.as_tensor(ambiguous, dtype=torch.bool, device=runtime.device)
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
            alt=alt_t[:, sample_idx : sample_idx + 1],
            total=total_t[:, sample_idx : sample_idx + 1],
            b_minus=b_minus_t[:, sample_idx : sample_idx + 1],
            b_plus=b_plus_t[:, sample_idx : sample_idx + 1],
            b_fixed=b_fixed_t[:, sample_idx : sample_idx + 1],
            ambiguous=ambiguous_t[:, sample_idx : sample_idx + 1],
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


def compute_scalar_well_start_bank(
    data: TumorData,
    *,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    exact_pilot: np.ndarray,
    secondary_wells: np.ndarray | None = None,
    valid_secondary: np.ndarray | None = None,
    max_sample_flips: int = 4,
) -> list[np.ndarray]:
    exact_pilot = np.asarray(exact_pilot, dtype=np.float64)
    if secondary_wells is None or valid_secondary is None:
        _, secondary, valid_secondary = compute_scalar_cell_wells(
            data,
            major_prior=major_prior,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
        )
    secondary = np.asarray(secondary_wells if secondary_wells is not None else secondary, dtype=np.float64)
    valid_secondary = np.asarray(valid_secondary, dtype=bool)

    starts: list[np.ndarray] = [exact_pilot.copy()]
    if not np.any(valid_secondary):
        return starts

    global_alternate = exact_pilot.copy()
    global_alternate[valid_secondary] = secondary[valid_secondary]
    starts.append(global_alternate)

    sample_delta = np.where(valid_secondary, np.abs(secondary - exact_pilot), 0.0)
    sample_scores = [
        (float(score), int(sample_idx))
        for sample_idx, score in enumerate(np.sum(sample_delta, axis=0))
        if score > 0.0
    ]
    sample_scores.sort(reverse=True)

    for _, sample_idx in sample_scores[: max(0, int(max_sample_flips))]:
        sample_start = exact_pilot.copy()
        mask = valid_secondary[:, sample_idx]
        sample_start[mask, sample_idx] = secondary[mask, sample_idx]
        starts.append(sample_start)

    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for start in starts:
        signature = np.round(np.asarray(start), decimals=8).astype(np.float32, copy=False).tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(start.astype(np.float64, copy=False))
    return unique


def compute_stationary_screen_box(
    data: TumorData,
    *,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_w: np.ndarray,
    lambda_value: float,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full_like(np.asarray(data.phi_upper, dtype=np.float64), float(eps), dtype=np.float64)
    upper = np.asarray(data.phi_upper, dtype=np.float64).copy()
    if float(lambda_value) <= 0.0 or edge_u.size == 0:
        return lower, upper

    row_weight_sum = np.zeros((data.num_mutations,), dtype=np.float64)
    np.add.at(row_weight_sum, np.asarray(edge_u, dtype=np.int64), np.asarray(edge_w, dtype=np.float64))
    np.add.at(row_weight_sum, np.asarray(edge_v, dtype=np.int64), np.asarray(edge_w, dtype=np.float64))

    alt = np.asarray(data.alt_counts, dtype=np.float64)
    total = np.asarray(data.total_counts, dtype=np.float64)
    b_minus = np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.minor_cn, dtype=np.float64)
    b_plus = np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.major_cn, dtype=np.float64)
    b_fixed = np.asarray(data.scaling, dtype=np.float64) * np.asarray(data.fixed_multiplicity, dtype=np.float64)
    ambiguous = np.asarray(data.multiplicity_estimation_mask, dtype=bool)

    screen_lower = lower.copy()
    screen_upper = upper.copy()
    for mutation_idx in range(data.num_mutations):
        threshold = float(lambda_value) * float(row_weight_sum[mutation_idx])
        if threshold <= 0.0:
            continue
        for sample_idx in range(data.num_samples):
            cell_lower, cell_upper = _screen_hull_from_gradient_strip(
                alt=float(alt[mutation_idx, sample_idx]),
                total=float(total[mutation_idx, sample_idx]),
                b_minus=float(b_minus[mutation_idx, sample_idx]),
                b_plus=float(b_plus[mutation_idx, sample_idx]),
                b_fixed=float(b_fixed[mutation_idx, sample_idx]),
                ambiguous=bool(ambiguous[mutation_idx, sample_idx]),
                lower=float(lower[mutation_idx, sample_idx]),
                upper=float(upper[mutation_idx, sample_idx]),
                major_prior=major_prior,
                eps=eps,
                threshold=threshold,
                tol=tol,
                max_iter=max_iter,
            )
            if cell_upper >= cell_lower:
                screen_lower[mutation_idx, sample_idx] = float(cell_lower)
                screen_upper[mutation_idx, sample_idx] = float(cell_upper)
    return screen_lower, screen_upper
