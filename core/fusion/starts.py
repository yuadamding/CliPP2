from __future__ import annotations

import numpy as np
import torch

from ...io.data import TumorData
from ..objective import ObservedModel, compile_observed_model
from ..scalar import (
    ScalarProblem,
    scalar_breakpoints,
    scalar_loss,
    scalar_loss_and_gradient,
    scalar_problem_from_model,
)
from .torch_backend import (
    TorchTumorData,
    mutation_region_loss_grid_torch,
    path_internal_breakpoints_torch,
)


_ROOT_SCAN_POINTS = 65


def _fixed_linear_path_scale_torch(torch_data: TorchTumorData) -> torch.Tensor | None:
    """Return the shared per-unit path slope when the categorical model is fixed."""

    path = torch_data.path_likelihood
    if path is None:
        return None
    first_valid_index = torch.argmax(path.valid.to(dtype=torch.int64), dim=-1)
    reference = torch.gather(
        path.first_scale,
        -1,
        first_valid_index.unsqueeze(-1),
    )
    fixed = (~path.valid) | (
        (path.first_scale == path.second_scale) & (path.first_scale == reference)
    )
    if not bool(torch.all(fixed).item()):
        return None
    return reference.squeeze(-1)


def _source_observed_model(torch_data: TorchTumorData) -> ObservedModel:
    """Return the immutable float64 source required by host scalar searches."""

    if torch_data.source_model is None:
        raise ValueError("Host scalar searches require an immutable source model.")
    return torch_data.source_model


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


def _scaled_sum(
    sign_left: float, logabs_left: float, sign_right: float, logabs_right: float
) -> float:
    if sign_left == 0.0:
        return float(sign_right)
    if sign_right == 0.0:
        return float(sign_left)
    anchor = max(float(logabs_left), float(logabs_right))
    return float(
        sign_left * np.exp(float(logabs_left) - anchor)
        + sign_right * np.exp(float(logabs_right) - anchor)
    )


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
    prior = float(major_prior)
    if not np.isfinite(prior) or not 0.0 < prior < 1.0:
        raise ValueError("major_prior must lie strictly in (0, 1).")
    log_prior_minor = float(np.log1p(-prior))
    log_prior_major = float(np.log(prior))

    logabs_minus = (
        log_prior_minor
        + alt * np.log(max(float(b_minus), tiny))
        + (nonalt - 1.0) * np.log(base_minus)
        + (np.log(max(abs(delta_minus), tiny)) if sign_minus != 0.0 else -np.inf)
    )
    logabs_plus = (
        log_prior_major
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
    if (
        not np.isfinite(left_probe)
        or not np.isfinite(right_probe)
        or right_probe <= left_probe
    ):
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

    probe = np.linspace(
        left_probe, right_probe, num=_ROOT_SCAN_POINTS, dtype=np.float64
    )
    values = np.asarray([stationarity(float(beta)) for beta in probe], dtype=np.float64)

    roots: list[float] = []
    for idx, current in enumerate(values):
        if not np.isfinite(current):
            continue
        if abs(float(current)) <= 1e-10:
            roots.append(float(probe[idx]))

    for left_beta, right_beta, left_value, right_value in zip(
        probe[:-1], probe[1:], values[:-1], values[1:]
    ):
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
    return [
        float(value)
        for value in np.unique(np.round(np.asarray(roots, dtype=np.float64), 12))
    ]


def _ambiguous_candidate_betas_numpy(
    *,
    alt: float,
    total: float,
    b_minus: float,
    b_plus: float,
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

    if float(total) <= 0.0:
        return np.asarray(
            [_project_to_interval(lower if hint is None else hint, lower, upper)],
            dtype=np.float64,
        )

    b_low = float(min(b_minus, b_plus))
    b_high = float(max(b_minus, b_plus))
    if b_low <= 0.0 or b_high <= 0.0:
        return np.asarray(
            [_project_to_interval(lower if hint is None else hint, lower, upper)],
            dtype=np.float64,
        )

    r1 = float(eps) / b_high
    r2 = float(eps) / b_low
    r3 = float(1.0 - float(eps)) / b_high
    r4 = float(1.0 - float(eps)) / b_low

    candidates = [lower, upper]
    if hint is not None and np.isfinite(hint):
        candidates.append(_project_to_interval(hint, lower, upper))
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
    candidate_array = candidate_array[
        (candidate_array >= lower - 1e-12) & (candidate_array <= upper + 1e-12)
    ]
    candidate_array = np.clip(candidate_array, lower, upper)
    if candidate_array.size == 0:
        candidate_array = np.asarray(
            [_project_to_interval(lower if hint is None else hint, lower, upper)],
            dtype=np.float64,
        )
    return candidate_array


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
        left_loss = (
            float(block_losses[block_idx - 1]) if block_idx > 0 else float("inf")
        )
        right_loss = (
            float(block_losses[block_idx + 1])
            if block_idx + 1 < len(blocks)
            else float("inf")
        )
        if current_loss > left_loss + loss_tol or current_loss > right_loss + loss_tol:
            continue

        beta_block = beta_sorted[start:stop]
        if beta_block.size == 0:
            continue
        if projected_hint is None:
            representative = float(beta_block[len(beta_block) // 2])
        else:
            representative = float(
                beta_block[int(np.argmin(np.abs(beta_block - projected_hint)))]
            )
        representatives.append(representative)
        representative_losses.append(current_loss)

    if not representatives:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty
    return (
        np.asarray(representatives, dtype=np.float64),
        np.asarray(representative_losses, dtype=np.float64),
    )


def _best_two_candidate_wells_numpy(
    problem: ScalarProblem,
    candidate_array: np.ndarray,
    *,
    tol: float,
    hint: float | None,
    decimals: int = 12,
) -> tuple[float, float | None]:
    candidate_array = np.asarray(candidate_array, dtype=np.float64)
    candidate_array = candidate_array[np.isfinite(candidate_array)]
    candidate_array = np.unique(np.round(candidate_array, int(decimals)))
    candidate_array = candidate_array[
        (candidate_array >= problem.lower - 1e-12)
        & (candidate_array <= problem.upper + 1e-12)
    ]
    candidate_array = np.clip(candidate_array, problem.lower, problem.upper)
    if candidate_array.size == 0:
        fallback = (
            problem.lower
            if hint is None
            else _project_to_interval(float(hint), problem.lower, problem.upper)
        )
        candidate_array = np.asarray([fallback], dtype=np.float64)

    losses = np.asarray(scalar_loss(problem, candidate_array), dtype=np.float64)
    finite = np.isfinite(losses)
    if not np.any(finite):
        fallback = problem.lower if hint is None else float(hint)
        return _project_to_interval(fallback, problem.lower, problem.upper), None
    candidate_array = candidate_array[finite]
    losses = losses[finite]
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
        if (
            abs(float(loss) - primary_loss) <= loss_tol
            and abs(float(beta) - primary) <= beta_tol
        ):
            continue
        if abs(float(beta) - primary) > beta_tol:
            secondary = float(beta)
            break
    return primary, secondary


def _scatter_min_by_row_block(
    values: torch.Tensor,
    *,
    valid: torch.Tensor,
    block_id: torch.Tensor,
) -> torch.Tensor:
    num_rows, num_cols = values.shape
    row_index = torch.arange(num_rows, dtype=torch.long, device=values.device)[:, None]
    safe_block = torch.clamp(block_id, min=0)
    flat_index = (row_index * num_cols + safe_block).reshape(-1)
    flat_valid = valid.reshape(-1)
    output = torch.full(
        (num_rows * num_cols,), float("inf"), dtype=values.dtype, device=values.device
    )
    if flat_valid.numel() == 0:
        return output.reshape(num_rows, num_cols)
    output.scatter_reduce_(
        0,
        flat_index[flat_valid],
        values.reshape(-1)[flat_valid],
        reduce="amin",
        include_self=True,
    )
    return output.reshape(num_rows, num_cols)


def _select_lexicographic_rows_torch(
    beta: torch.Tensor,
    loss: torch.Tensor,
    valid: torch.Tensor,
    tie_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inf = torch.full_like(loss, float("inf"))
    masked_loss = torch.where(valid, loss, inf)
    best_loss = torch.min(masked_loss, dim=1).values
    near_best = valid & (loss == best_loss[:, None])
    best_tie = torch.min(torch.where(near_best, tie_value, inf), dim=1).values
    near_best_tie = near_best & (tie_value == best_tie[:, None])
    best_index = torch.argmax(near_best_tie.to(torch.long), dim=1)
    best_beta = torch.gather(beta, 1, best_index[:, None]).squeeze(1)
    best_beta = torch.where(
        torch.isfinite(best_loss), best_beta, torch.full_like(best_beta, float("nan"))
    )
    return best_beta, best_loss


def _select_secondary_lexicographic_rows_torch(
    beta: torch.Tensor,
    loss: torch.Tensor,
    valid: torch.Tensor,
    tie_value: torch.Tensor,
    *,
    primary: torch.Tensor,
    beta_tol: torch.Tensor,
) -> torch.Tensor:
    inf = torch.full_like(loss, float("inf"))
    far_from_primary = torch.abs(beta - primary[:, None]) > beta_tol[:, None]
    eligible = valid & far_from_primary
    masked_loss = torch.where(eligible, loss, inf)
    best_loss = torch.min(masked_loss, dim=1).values
    near_best = eligible & (loss == best_loss[:, None])
    best_tie = torch.min(torch.where(near_best, tie_value, inf), dim=1).values
    near_best_tie = near_best & (tie_value == best_tie[:, None])
    best_index = torch.argmax(near_best_tie.to(torch.long), dim=1)
    secondary = torch.gather(beta, 1, best_index[:, None]).squeeze(1)
    return torch.where(
        torch.isfinite(best_loss), secondary, torch.full_like(secondary, float("nan"))
    )


def _ambiguous_best_two_from_candidate_grid_torch(
    candidate_grid: torch.Tensor,
    *,
    alt: torch.Tensor,
    total: torch.Tensor,
    b_minus: torch.Tensor,
    b_plus: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    hint: torch.Tensor,
    major_prior: float,
    eps: float,
    tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidates = torch.round(
        candidate_grid.to(dtype=alt.dtype, device=alt.device), decimals=12
    )
    lower = lower.to(dtype=alt.dtype, device=alt.device)
    upper = upper.to(dtype=alt.dtype, device=alt.device)
    hint = hint.to(dtype=alt.dtype, device=alt.device)
    valid = (
        torch.isfinite(candidates)
        & (candidates >= lower[:, None] - 1e-12)
        & (candidates <= upper[:, None] + 1e-12)
    )
    candidates = torch.minimum(
        torch.maximum(candidates, lower[:, None]), upper[:, None]
    )

    sorted_beta = torch.sort(
        torch.where(valid, candidates, torch.full_like(candidates, float("inf"))), dim=1
    ).values
    sorted_valid = torch.isfinite(sorted_beta)
    previous_beta = torch.cat(
        [
            torch.full(
                (sorted_beta.shape[0], 1),
                float("nan"),
                dtype=sorted_beta.dtype,
                device=sorted_beta.device,
            ),
            sorted_beta[:, :-1],
        ],
        dim=1,
    )
    column_index = torch.arange(
        sorted_beta.shape[1], dtype=torch.long, device=sorted_beta.device
    )[None, :]
    unique = sorted_valid & ((column_index == 0) | (sorted_beta != previous_beta))
    beta = torch.sort(
        torch.where(unique, sorted_beta, torch.full_like(sorted_beta, float("inf"))),
        dim=1,
    ).values
    valid = torch.isfinite(beta)
    empty_rows = ~torch.any(valid, dim=1)
    if beta.shape[1] > 0:
        fallback = torch.minimum(torch.maximum(hint, lower), upper)
        beta[:, 0] = torch.where(empty_rows, fallback, beta[:, 0])
        valid[:, 0] = valid[:, 0] | empty_rows

    loss = mutation_region_loss_grid_torch(
        beta,
        alt=alt[:, None],
        total=total[:, None],
        b_minus=b_minus[:, None],
        b_plus=b_plus[:, None],
        b_fixed=b_plus[:, None],
        ambiguous=torch.ones_like(beta, dtype=torch.bool),
        major_prior=major_prior,
        eps=eps,
    )
    loss = torch.where(valid, loss, torch.full_like(loss, float("inf")))
    loss_tol = max(float(tol) * 10.0, 1e-10)

    previous_valid = torch.cat(
        [
            torch.zeros((valid.shape[0], 1), dtype=torch.bool, device=valid.device),
            valid[:, :-1],
        ],
        dim=1,
    )
    previous_loss = torch.cat(
        [
            torch.full(
                (loss.shape[0], 1), float("inf"), dtype=loss.dtype, device=loss.device
            ),
            loss[:, :-1],
        ],
        dim=1,
    )
    loss_delta = torch.where(
        valid & previous_valid,
        torch.abs(loss - previous_loss),
        torch.full_like(loss, float("inf")),
    )
    block_start = valid & (~previous_valid | (loss_delta > loss_tol))
    block_id = torch.cumsum(block_start.to(torch.long), dim=1) - 1
    block_id = torch.where(valid, block_id, torch.full_like(block_id, -1))

    block_loss = _scatter_min_by_row_block(loss, valid=valid, block_id=block_id)
    left_block_loss = torch.cat(
        [
            torch.full(
                (block_loss.shape[0], 1),
                float("inf"),
                dtype=block_loss.dtype,
                device=block_loss.device,
            ),
            block_loss[:, :-1],
        ],
        dim=1,
    )
    right_block_loss = torch.cat(
        [
            block_loss[:, 1:],
            torch.full(
                (block_loss.shape[0], 1),
                float("inf"),
                dtype=block_loss.dtype,
                device=block_loss.device,
            ),
        ],
        dim=1,
    )
    local_block = (
        torch.isfinite(block_loss)
        & (block_loss <= left_block_loss + loss_tol)
        & (block_loss <= right_block_loss + loss_tol)
    )

    distance_to_hint = torch.where(
        valid, torch.abs(beta - hint[:, None]), torch.full_like(beta, float("inf"))
    )
    representative_distance = _scatter_min_by_row_block(
        distance_to_hint, valid=valid, block_id=block_id
    )
    safe_block = torch.clamp(block_id, min=0)
    distance_at_block = torch.gather(representative_distance, 1, safe_block)
    representative_candidate = (
        valid
        & torch.isfinite(distance_to_hint)
        & (distance_to_hint == distance_at_block)
    )

    num_rows, num_cols = beta.shape
    row_index = torch.arange(num_rows, dtype=torch.long, device=beta.device)[:, None]
    flat_index = (row_index * num_cols + safe_block).reshape(-1)
    col_index = torch.arange(num_cols, dtype=torch.long, device=beta.device)[
        None, :
    ].expand(num_rows, -1)
    selected_col = torch.full(
        (num_rows * num_cols,), num_cols, dtype=torch.long, device=beta.device
    )
    flat_candidate = representative_candidate.reshape(-1)
    selected_col.scatter_reduce_(
        0,
        flat_index[flat_candidate],
        col_index.reshape(-1)[flat_candidate],
        reduce="amin",
        include_self=True,
    )
    selected_col = selected_col.reshape(num_rows, num_cols)
    gathered_beta = torch.gather(
        beta, 1, torch.clamp(selected_col, max=max(num_cols - 1, 0))
    )
    local_beta = torch.where(
        selected_col < num_cols, gathered_beta, torch.full_like(beta, float("nan"))
    )

    local_loss = torch.where(
        local_block, block_loss, torch.full_like(block_loss, float("inf"))
    )
    local_valid = torch.isfinite(local_loss) & torch.isfinite(local_beta)
    local_tie = torch.where(
        local_valid,
        torch.abs(local_beta - hint[:, None]),
        torch.full_like(local_loss, float("inf")),
    )
    has_local = torch.any(local_valid, dim=1)

    local_primary, _ = _select_lexicographic_rows_torch(
        local_beta, local_loss, local_valid, local_tie
    )
    fallback_primary, _ = _select_lexicographic_rows_torch(beta, loss, valid, beta)
    primary = torch.where(has_local, local_primary, fallback_primary)

    secondary_beta_tol = torch.maximum(
        torch.full_like(primary, loss_tol),
        1e-8 * torch.maximum(torch.ones_like(primary), torch.abs(primary)),
    )
    local_secondary = _select_secondary_lexicographic_rows_torch(
        local_beta,
        local_loss,
        local_valid,
        local_tie,
        primary=primary,
        beta_tol=secondary_beta_tol,
    )
    fallback_secondary = _select_secondary_lexicographic_rows_torch(
        beta,
        loss,
        valid,
        beta,
        primary=primary,
        beta_tol=secondary_beta_tol,
    )
    secondary = torch.where(has_local, local_secondary, fallback_secondary)
    valid_secondary = torch.isfinite(secondary) & (
        torch.abs(secondary - primary) > max(float(tol) * 10.0, 1e-8)
    )
    return primary, secondary, valid_secondary


def _pooled_sample_loss_grid_torch(
    torch_data: TorchTumorData,
    beta_by_sample: torch.Tensor,
    *,
    major_prior: float,
    eps: float,
) -> torch.Tensor:
    if beta_by_sample.ndim == 1:
        beta_grid = beta_by_sample[:, None]
        squeeze = True
    else:
        beta_grid = beta_by_sample
        squeeze = False
    num_mutations = int(torch_data.alt.shape[0])
    beta = beta_grid.unsqueeze(0).expand(num_mutations, -1, -1)
    losses = mutation_region_loss_grid_torch(
        beta,
        alt=torch_data.alt.unsqueeze(-1),
        total=torch_data.total.unsqueeze(-1),
        b_minus=torch_data.b_minus.unsqueeze(-1),
        b_plus=torch_data.b_plus.unsqueeze(-1),
        b_fixed=torch_data.b_fixed.unsqueeze(-1),
        ambiguous=torch_data.ambiguous.unsqueeze(-1),
        major_prior=major_prior,
        eps=eps,
    )
    sample_losses = torch.sum(losses, dim=0)
    return sample_losses.squeeze(-1) if squeeze else sample_losses


def _golden_section_minimize_samples_torch(
    objective,
    *,
    left: torch.Tensor,
    right: torch.Tensor,
    tol: float,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = 0.5 * (np.sqrt(5.0) - 1.0)
    ratio_t = torch.as_tensor(ratio, dtype=left.dtype, device=left.device)
    left_current = left
    right_current = right

    for _ in range(max(int(max_iter), 8)):
        width = right_current - left_current
        x1 = right_current - ratio_t * width
        x2 = left_current + ratio_t * width
        f1 = objective(x1)
        f2 = objective(x2)
        active = torch.abs(width) > float(tol) * (
            1.0 + torch.abs(left_current) + torch.abs(right_current)
        )
        shrink_right = active & (f1 <= f2)
        shrink_left = active & ~shrink_right
        left_current = torch.where(shrink_left, x1, left_current)
        right_current = torch.where(shrink_right, x2, right_current)

    width = right_current - left_current
    x1 = right_current - ratio_t * width
    x2 = left_current + ratio_t * width
    f1 = objective(x1)
    f2 = objective(x2)
    use_x1 = f1 <= f2
    return torch.where(use_x1, x1, x2), torch.where(use_x1, f1, f2)


def _ambiguous_middle_stationarity_torch(
    beta: torch.Tensor,
    *,
    alt: torch.Tensor,
    total: torch.Tensor,
    b_low: torch.Tensor,
    b_high: torch.Tensor,
    major_prior: float,
) -> torch.Tensor:
    while alt.ndim < beta.ndim:
        alt = alt.unsqueeze(-1)
        total = total.unsqueeze(-1)
        b_low = b_low.unsqueeze(-1)
        b_high = b_high.unsqueeze(-1)

    tiny = torch.finfo(beta.dtype).tiny
    nonalt = total - alt
    base_low = torch.clamp(1.0 - b_low * beta, min=tiny)
    base_high = torch.clamp(1.0 - b_high * beta, min=tiny)
    delta_low = alt - total * b_low * beta
    delta_high = alt - total * b_high * beta

    sign_low = torch.sign(delta_low)
    sign_high = torch.sign(delta_high)
    log_prior_low = torch.log1p(beta.new_tensor(-float(major_prior)))
    log_prior_high = torch.log(beta.new_tensor(float(major_prior)))
    neg_inf = torch.full_like(beta, float("-inf"))

    logabs_low = (
        log_prior_low
        + alt * torch.log(torch.clamp(b_low, min=tiny))
        + (nonalt - 1.0) * torch.log(base_low)
        + torch.where(
            sign_low != 0.0,
            torch.log(torch.clamp(torch.abs(delta_low), min=tiny)),
            neg_inf,
        )
    )
    logabs_high = (
        log_prior_high
        + alt * torch.log(torch.clamp(b_high, min=tiny))
        + (nonalt - 1.0) * torch.log(base_high)
        + torch.where(
            sign_high != 0.0,
            torch.log(torch.clamp(torch.abs(delta_high), min=tiny)),
            neg_inf,
        )
    )
    anchor = torch.maximum(logabs_low, logabs_high)
    finite_anchor = torch.isfinite(anchor)
    safe_anchor = torch.where(finite_anchor, anchor, torch.zeros_like(anchor))
    low_term = sign_low * torch.exp(logabs_low - safe_anchor)
    high_term = sign_high * torch.exp(logabs_high - safe_anchor)
    low_term = torch.where(
        (sign_low != 0.0) & finite_anchor, low_term, torch.zeros_like(low_term)
    )
    high_term = torch.where(
        (sign_high != 0.0) & finite_anchor, high_term, torch.zeros_like(high_term)
    )
    return low_term + high_term


def _ambiguous_candidate_grid_torch(
    *,
    alt: torch.Tensor,
    total: torch.Tensor,
    b_minus: torch.Tensor,
    b_plus: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    hint: torch.Tensor,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
) -> torch.Tensor:
    b_low = torch.minimum(b_minus, b_plus)
    b_high = torch.maximum(b_minus, b_plus)
    valid = (upper > lower + 1e-12) & (total > 0.0) & (b_low > 0.0) & (b_high > 0.0)
    fallback = torch.where(upper <= lower + 1e-12, lower, hint)
    nan = torch.full_like(lower, float("nan"))

    def column(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, values, nan)

    columns = [
        torch.where(valid, lower, fallback),
        column(upper, valid),
        column(hint, valid),
    ]

    r1 = float(eps) / b_high
    r2 = float(eps) / b_low
    r3 = (1.0 - float(eps)) / b_high
    r4 = (1.0 - float(eps)) / b_low
    for kink in (r1, r2, r3, r4):
        columns.append(column(kink, valid & (kink >= lower) & (kink <= upper)))

    beta_high = alt / (total * b_high)
    region2_left = torch.maximum(lower, r1)
    region2_right = torch.minimum(upper, r2)
    columns.append(
        column(
            beta_high, valid & (region2_left < beta_high) & (beta_high < region2_right)
        )
    )

    beta_low = alt / (total * b_low)
    region4_left = torch.maximum(lower, r3)
    region4_right = torch.minimum(upper, r4)
    columns.append(
        column(beta_low, valid & (region4_left < beta_low) & (beta_low < region4_right))
    )

    middle_left = torch.maximum(lower, r2)
    middle_right = torch.minimum(upper, r3)
    middle_mask = valid & (middle_right > middle_left + 1e-12)
    num_mutation_regions = int(alt.numel())
    zero_roots = torch.full(
        (num_mutation_regions, _ROOT_SCAN_POINTS),
        float("nan"),
        dtype=alt.dtype,
        device=alt.device,
    )
    interval_roots = torch.full(
        (num_mutation_regions, _ROOT_SCAN_POINTS - 1),
        float("nan"),
        dtype=alt.dtype,
        device=alt.device,
    )

    if bool(torch.any(middle_mask).item()):
        left_probe = torch.nextafter(middle_left, middle_right)
        right_probe = torch.nextafter(middle_right, middle_left)
        t = torch.linspace(
            0.0, 1.0, steps=_ROOT_SCAN_POINTS, dtype=alt.dtype, device=alt.device
        )
        probe = left_probe[:, None] + (right_probe - left_probe)[:, None] * t
        probe = torch.where(
            middle_mask[:, None], probe, torch.full_like(probe, float("nan"))
        )
        values = _ambiguous_middle_stationarity_torch(
            probe,
            alt=alt,
            total=total,
            b_low=b_low,
            b_high=b_high,
            major_prior=major_prior,
        )

        zero_mask = torch.isfinite(values) & (torch.abs(values) <= 1e-10)
        zero_roots = torch.where(zero_mask, probe, zero_roots)

        left_values = values[:, :-1]
        right_values = values[:, 1:]
        interval_mask = (
            torch.isfinite(left_values)
            & torch.isfinite(right_values)
            & (left_values != 0.0)
            & (right_values != 0.0)
            & (left_values * right_values <= 0.0)
        )
        flat_interval_indices = torch.nonzero(
            interval_mask.reshape(-1), as_tuple=False
        ).flatten()
        if int(flat_interval_indices.numel()) > 0:
            flat_left = probe[:, :-1].reshape(-1)
            flat_right = probe[:, 1:].reshape(-1)
            left_current = flat_left[flat_interval_indices]
            right_current = flat_right[flat_interval_indices]
            f_left = left_values.reshape(-1)[flat_interval_indices]
            mutation_region_ids = torch.arange(
                num_mutation_regions, dtype=torch.long, device=alt.device
            ).repeat_interleave(_ROOT_SCAN_POINTS - 1)
            mutation_region_ids = mutation_region_ids[flat_interval_indices]

            roots = 0.5 * (left_current + right_current)
            active = torch.ones_like(roots, dtype=torch.bool)
            invalid = torch.zeros_like(roots, dtype=torch.bool)
            for _ in range(max(int(max_iter), 32)):
                midpoint = 0.5 * (left_current + right_current)
                f_mid = _ambiguous_middle_stationarity_torch(
                    midpoint,
                    alt=alt[mutation_region_ids],
                    total=total[mutation_region_ids],
                    b_low=b_low[mutation_region_ids],
                    b_high=b_high[mutation_region_ids],
                    major_prior=major_prior,
                )
                finite_mid = torch.isfinite(f_mid)
                invalid = invalid | (active & ~finite_mid)
                converged = (
                    active
                    & finite_mid
                    & (
                        (torch.abs(f_mid) <= 1e-12)
                        | (
                            torch.abs(right_current - left_current)
                            <= float(tol) * (1.0 + torch.abs(midpoint))
                        )
                    )
                )
                roots = torch.where(converged, midpoint, roots)
                active = active & finite_mid & ~converged
                keep_left_interval = active & (f_left * f_mid <= 0.0)
                move_left = active & ~keep_left_interval
                right_current = torch.where(keep_left_interval, midpoint, right_current)
                left_current = torch.where(move_left, midpoint, left_current)
                f_left = torch.where(move_left, f_mid, f_left)

            roots = torch.where(
                active & ~invalid, 0.5 * (left_current + right_current), roots
            )
            roots = torch.where(invalid, torch.full_like(roots, float("nan")), roots)
            interval_roots.reshape(-1)[flat_interval_indices] = roots

    columns.extend([zero_roots, interval_roots])
    return torch.cat([col[:, None] if col.ndim == 1 else col for col in columns], dim=1)


def _path_unit_base_points_numpy(
    problem: ScalarProblem,
    *,
    hint: float,
) -> tuple[np.ndarray, np.ndarray]:
    if problem.alt.size != 1:
        raise ValueError("Path-well base points require one mutation-region row.")
    lower, upper = problem.lower, problem.upper
    hard = scalar_breakpoints(problem, observed_only=False)
    points: list[float] = [*hard.tolist(), float(hint)]
    first = problem.first_scale[0]
    second = problem.second_scale[0]
    switch = problem.switch[0]
    valid = problem.valid[0]
    total = float(problem.alt[0] + problem.nonalt[0])
    smoothed_vaf = (float(problem.alt[0]) + 0.5) / (total + 1.0)

    def add_probability_inverse(target: float) -> None:
        with np.errstate(divide="ignore", invalid="ignore"):
            left = float(target) / first
            right = switch + (float(target) - first * switch) / second
        left_ok = (
            valid
            & (first > 0.0)
            & np.isfinite(left)
            & (left >= lower)
            & (left <= np.minimum(switch, upper))
        )
        right_ok = (
            valid
            & (second > 0.0)
            & np.isfinite(right)
            & (right >= np.maximum(switch, lower))
            & (right <= upper)
        )
        values = np.concatenate((left[left_ok], right[right_ok]))
        points.extend(values.tolist())

    add_probability_inverse(smoothed_vaf)
    point_array = np.unique(np.round(np.asarray(points, dtype=np.float64), 14))
    hard_array = np.unique(np.round(hard, 14))
    keep = (point_array >= lower - 1e-12) & (point_array <= upper + 1e-12)
    hard_keep = (hard_array >= lower - 1e-12) & (hard_array <= upper + 1e-12)
    return (
        np.clip(point_array[keep], lower, upper),
        np.clip(hard_array[hard_keep], lower, upper),
    )


def _path_unit_best_two_betas_numpy(
    problem: ScalarProblem,
    *,
    hint: float,
    tol: float,
    max_iter: int,
) -> tuple[float, float | None]:
    base_points, hard_points = _path_unit_base_points_numpy(
        problem,
        hint=hint,
    )
    candidates: list[float] = base_points.tolist()

    def evaluate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        loss, gradient = scalar_loss_and_gradient(problem, values)
        return (
            np.asarray(loss, dtype=np.float64),
            np.asarray(gradient, dtype=np.float64),
        )

    for left, right in zip(hard_points[:-1], hard_points[1:]):
        if right <= left + 1e-12:
            continue
        scan_left = np.nextafter(float(left), float(right))
        scan_right = np.nextafter(float(right), float(left))
        scan = np.linspace(
            scan_left,
            scan_right,
            num=_ROOT_SCAN_POINTS,
            dtype=np.float64,
        )
        _, gradient = evaluate(scan)
        finite = np.isfinite(gradient)
        candidates.extend(scan[finite].tolist())
        near_zero = finite & (np.abs(gradient) <= 1e-10)
        candidates.extend(scan[near_zero].tolist())
        for index in range(scan.size - 1):
            if not finite[index] or not finite[index + 1]:
                continue
            left_gradient = float(gradient[index])
            right_gradient = float(gradient[index + 1])
            if left_gradient == 0.0 or right_gradient == 0.0:
                continue
            if left_gradient * right_gradient > 0.0:
                continue
            root_left = float(scan[index])
            root_right = float(scan[index + 1])
            root_gradient = left_gradient
            for _ in range(max(int(max_iter), 32)):
                midpoint = 0.5 * (root_left + root_right)
                _, midpoint_gradient_array = evaluate(
                    np.asarray([midpoint], dtype=np.float64)
                )
                midpoint_gradient = float(midpoint_gradient_array[0])
                if not np.isfinite(midpoint_gradient):
                    break
                if abs(midpoint_gradient) <= 1e-12 or root_right - root_left <= float(
                    tol
                ) * (1.0 + abs(midpoint)):
                    root_left = midpoint
                    root_right = midpoint
                    break
                if root_gradient * midpoint_gradient <= 0.0:
                    root_right = midpoint
                else:
                    root_left = midpoint
                    root_gradient = midpoint_gradient
            candidates.append(0.5 * (root_left + root_right))

    return _best_two_candidate_wells_numpy(
        problem,
        np.asarray(candidates, dtype=np.float64),
        tol=tol,
        hint=hint,
        decimals=14,
    )


def _path_scalar_wells_from_model(
    model: ObservedModel,
    *,
    phi_init: np.ndarray,
    eps: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = model.shape
    flat_upper = model.upper.reshape(-1)
    flat_hint = np.clip(
        np.asarray(phi_init, dtype=np.float64).reshape(-1),
        float(eps),
        flat_upper,
    )
    observed = (model.observed & ((model.alt + model.nonalt) > 0.0)).reshape(-1)
    primary = flat_hint.copy()
    secondary = np.full_like(primary, np.nan)
    valid_secondary = np.zeros_like(primary, dtype=bool)
    for index in np.flatnonzero(observed):
        mutation_index, region_index = np.unravel_index(index, shape)
        problem = scalar_problem_from_model(
            model,
            np.asarray([mutation_index], dtype=np.int64),
            int(region_index),
            lower=float(eps),
            upper=float(flat_upper[index]),
            eps=float(eps),
        )
        best, alternate = _path_unit_best_two_betas_numpy(
            problem,
            hint=float(flat_hint[index]),
            tol=float(tol),
            max_iter=int(max_iter),
        )
        primary[index] = best
        if alternate is not None:
            secondary[index] = alternate
            valid_secondary[index] = True
    return (
        primary.reshape(shape),
        secondary.reshape(shape),
        valid_secondary.reshape(shape),
    )


def _path_pooled_start_from_model(
    model: ObservedModel,
    *,
    beta_hints: np.ndarray,
    eps: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    hints = np.asarray(beta_hints, dtype=np.float64)
    num_mutations, num_regions = model.shape
    pooled = np.empty((num_regions,), dtype=np.float64)

    for region_index in range(num_regions):
        lower = float(eps)
        upper = float(np.min(model.upper[:, region_index]))
        if upper <= lower + 1e-12:
            pooled[region_index] = lower
            continue
        region_observed = model.observed[:, region_index] & (
            (model.alt[:, region_index] + model.nonalt[:, region_index]) > 0.0
        )
        if not np.any(region_observed):
            pooled[region_index] = 0.5 * (lower + upper)
            continue
        switches = model.switch[region_observed, region_index, :][
            model.valid[region_observed, region_index, :]
        ]
        hard = np.unique(
            np.round(
                np.concatenate(
                    (
                        np.asarray([lower, upper], dtype=np.float64),
                        switches,
                    )
                ),
                14,
            )
        )
        hard = hard[(hard >= lower - 1e-12) & (hard <= upper + 1e-12)]
        hint_values = np.clip(hints[region_observed, region_index], lower, upper)
        if hint_values.size > 33:
            hint_values = np.quantile(hint_values, np.linspace(0.0, 1.0, num=33))
        grid = np.unique(
            np.round(
                np.concatenate(
                    (
                        hard,
                        hint_values,
                        np.geomspace(max(lower, float(eps)), upper, num=257),
                        np.linspace(lower, upper, num=129),
                    )
                ),
                14,
            )
        )
        grid = grid[(grid >= lower - 1e-12) & (grid <= upper + 1e-12)]

        member_indices = np.flatnonzero(region_observed)
        problem = scalar_problem_from_model(
            model,
            member_indices,
            region_index,
            lower=lower,
            upper=upper,
            eps=float(eps),
        )

        def objective(values: np.ndarray) -> np.ndarray:
            return np.asarray(scalar_loss(problem, values), dtype=np.float64)

        losses = objective(grid)
        finite_indices = np.flatnonzero(np.isfinite(losses))
        if finite_indices.size == 0:
            pooled[region_index] = float(np.median(hint_values))
            continue
        local_indices = [
            int(index)
            for index in finite_indices
            if (
                (index == 0 or losses[index] <= losses[index - 1] + 1e-10)
                and (
                    index + 1 == losses.size
                    or losses[index] <= losses[index + 1] + 1e-10
                )
            )
        ]
        if not local_indices:
            local_indices = [int(finite_indices[np.argmin(losses[finite_indices])])]
        best_index = int(finite_indices[np.argmin(losses[finite_indices])])
        best_beta = float(grid[best_index])
        best_loss = float(losses[best_index])
        for index in sorted(local_indices, key=lambda item: float(losses[item]))[:16]:
            center = float(grid[index])
            at_switch = bool(np.any(np.isclose(center, hard, rtol=0.0, atol=1e-12)))
            intervals = (
                [
                    (float(grid[max(index - 1, 0)]), center),
                    (center, float(grid[min(index + 1, grid.size - 1)])),
                ]
                if at_switch
                else [
                    (
                        float(grid[max(index - 1, 0)]),
                        float(grid[min(index + 1, grid.size - 1)]),
                    )
                ]
            )
            for left, right in intervals:
                if right <= left + 1e-12:
                    continue
                refined_beta, refined_loss = _golden_section_minimize(
                    objective,
                    left=left,
                    right=right,
                    tol=float(tol),
                    max_iter=max(int(max_iter), 32),
                )
                if np.isfinite(refined_loss) and refined_loss < best_loss:
                    best_beta = float(refined_beta)
                    best_loss = float(refined_loss)
        pooled[region_index] = float(np.clip(best_beta, lower, upper))
    return np.tile(pooled[None, :], (num_mutations, 1))


def compute_scalar_mutation_region_wells(
    data: TumorData,
    *,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model = compile_observed_model(data, major_prior=major_prior, eps=eps)
    path_spec = getattr(data, "path_likelihood", None)
    if path_spec is not None:
        return _path_scalar_wells_from_model(
            model,
            phi_init=np.asarray(data.phi_init, dtype=np.float64),
            eps=float(eps),
            tol=float(tol),
            max_iter=int(max_iter),
        )
    alt = model.alt.reshape(-1)
    total = (model.alt + model.nonalt).reshape(-1)
    b_minus = model.first_scale[..., 0].reshape(-1)
    b_plus = model.first_scale[..., 1].reshape(-1)
    b_fixed = model.first_scale[..., 0].reshape(-1)
    ambiguous = model.valid[..., 1].reshape(-1)
    lower = np.full_like(alt, float(eps), dtype=np.float64)
    upper = model.upper.reshape(-1)
    hint = np.clip(
        np.asarray(data.phi_init, dtype=np.float64).reshape(-1), lower, upper
    )
    refined = np.zeros_like(hint, dtype=np.float64)
    secondary = np.full_like(hint, np.nan, dtype=np.float64)
    valid_secondary = np.zeros_like(hint, dtype=bool)

    fixed_mask = ~ambiguous
    if np.any(fixed_mask):
        fixed_total = total[fixed_mask]
        fixed_valid = (fixed_total > 0.0) & (b_fixed[fixed_mask] > 0.0)
        fixed_solution = hint[fixed_mask].copy()
        if np.any(fixed_valid):
            p_hat = np.clip(
                alt[fixed_mask][fixed_valid] / fixed_total[fixed_valid], eps, 1.0 - eps
            )
            beta_hat = p_hat / np.clip(b_fixed[fixed_mask][fixed_valid], eps, None)
            fixed_solution[fixed_valid] = np.minimum(
                np.maximum(beta_hat, lower[fixed_mask][fixed_valid]),
                upper[fixed_mask][fixed_valid],
            )
        refined[fixed_mask] = np.minimum(
            np.maximum(fixed_solution, lower[fixed_mask]), upper[fixed_mask]
        )

    amb_indices = np.flatnonzero(ambiguous)
    for idx in amb_indices:
        mutation_index, region_index = np.unravel_index(idx, model.shape)
        problem = scalar_problem_from_model(
            model,
            np.asarray([mutation_index], dtype=np.int64),
            int(region_index),
            lower=float(lower[idx]),
            upper=float(upper[idx]),
            eps=float(eps),
            # Historical exact pilots use count information even where the
            # fit mask is false; those coordinates still define the graph.
            respect_observed=False,
        )
        primary, alternate = _best_two_candidate_wells_numpy(
            problem,
            _ambiguous_candidate_betas_numpy(
                alt=float(alt[idx]),
                total=float(total[idx]),
                b_minus=float(b_minus[idx]),
                b_plus=float(b_plus[idx]),
                lower=float(lower[idx]),
                upper=float(upper[idx]),
                major_prior=major_prior,
                eps=eps,
                tol=tol,
                max_iter=max_iter,
                hint=float(hint[idx]),
            ),
            tol=tol,
            hint=float(hint[idx]),
        )
        refined[idx] = float(primary)
        if alternate is not None:
            secondary[idx] = float(alternate)
            valid_secondary[idx] = abs(float(alternate) - float(primary)) > max(
                float(tol) * 10.0, 1e-8
            )

    shape = model.shape
    return (
        np.clip(refined.reshape(shape), eps, model.upper),
        secondary.reshape(shape),
        valid_secondary.reshape(shape),
    )


def compute_scalar_mutation_region_wells_torch(
    torch_data: TorchTumorData,
    *,
    phi_init: torch.Tensor | np.ndarray,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch_data.alt.dtype
    device = torch_data.alt.device
    shape = tuple(torch_data.alt.shape)
    fixed_path_scale = _fixed_linear_path_scale_torch(torch_data)
    if torch_data.path_likelihood is not None and fixed_path_scale is None:
        primary, secondary, valid_secondary = _path_scalar_wells_from_model(
            _source_observed_model(torch_data),
            phi_init=(
                phi_init.detach().cpu().numpy()
                if torch.is_tensor(phi_init)
                else np.asarray(phi_init)
            ),
            eps=float(eps),
            tol=float(tol),
            max_iter=int(max_iter),
        )
        return (
            torch.as_tensor(primary, dtype=dtype, device=device),
            torch.as_tensor(secondary, dtype=dtype, device=device),
            torch.as_tensor(valid_secondary, dtype=torch.bool, device=device),
        )
    lower = torch.full_like(torch_data.phi_upper, float(eps))
    upper = torch_data.phi_upper
    hint = torch.as_tensor(phi_init, dtype=dtype, device=device).reshape(shape)
    hint = torch.minimum(torch.maximum(hint, lower), upper)

    refined = torch.zeros_like(hint)
    secondary = torch.full_like(hint, float("nan"))
    valid_secondary = torch.zeros(shape, dtype=torch.bool, device=device)

    effective_ambiguous = (
        torch.zeros_like(torch_data.ambiguous)
        if fixed_path_scale is not None
        else torch_data.ambiguous
    )
    effective_b_fixed = (
        fixed_path_scale if fixed_path_scale is not None else torch_data.b_fixed
    )
    fixed_mask = ~effective_ambiguous
    if bool(torch.any(fixed_mask).item()):
        fixed_valid = fixed_mask & (torch_data.total > 0.0) & (effective_b_fixed > 0.0)
        fixed_solution = torch.where(fixed_mask, hint, refined)
        if bool(torch.any(fixed_valid).item()):
            p_hat = torch.clamp(
                torch_data.alt
                / torch.clamp(torch_data.total, min=torch.finfo(dtype).tiny),
                min=float(eps),
                max=1.0 - float(eps),
            )
            beta_hat = p_hat / torch.clamp(effective_b_fixed, min=float(eps))
            fixed_solution = torch.where(fixed_valid, beta_hat, fixed_solution)
        fixed_solution = torch.minimum(torch.maximum(fixed_solution, lower), upper)
        refined = torch.where(fixed_mask, fixed_solution, refined)

    ambiguous_mask = effective_ambiguous
    if bool(torch.any(ambiguous_mask).item()):
        flat_mask = ambiguous_mask.reshape(-1)
        alt = torch_data.alt.reshape(-1)[flat_mask]
        total = torch_data.total.reshape(-1)[flat_mask]
        b_minus = torch_data.b_minus.reshape(-1)[flat_mask]
        b_plus = torch_data.b_plus.reshape(-1)[flat_mask]
        lower_flat = lower.reshape(-1)[flat_mask]
        upper_flat = upper.reshape(-1)[flat_mask]
        hint_flat = hint.reshape(-1)[flat_mask]
        candidate_grid = _ambiguous_candidate_grid_torch(
            alt=alt,
            total=total,
            b_minus=b_minus,
            b_plus=b_plus,
            lower=lower_flat,
            upper=upper_flat,
            hint=hint_flat,
            major_prior=major_prior,
            eps=eps,
            tol=tol,
            max_iter=max_iter,
        )

        primary, alternate, valid_alternate = (
            _ambiguous_best_two_from_candidate_grid_torch(
                candidate_grid,
                alt=alt,
                total=total,
                b_minus=b_minus,
                b_plus=b_plus,
                lower=lower_flat,
                upper=upper_flat,
                hint=hint_flat,
                major_prior=major_prior,
                eps=eps,
                tol=tol,
            )
        )

        refined_flat = refined.reshape(-1)
        secondary_flat = secondary.reshape(-1)
        valid_secondary_flat = valid_secondary.reshape(-1)
        refined_flat[flat_mask] = primary
        secondary_flat[flat_mask] = alternate
        valid_secondary_flat[flat_mask] = valid_alternate

    return (
        torch.minimum(torch.maximum(refined, lower), upper),
        secondary,
        valid_secondary,
    )


def compute_pooled_observed_data_start_torch(
    torch_data: TorchTumorData,
    *,
    major_prior: float,
    eps: float,
    tol: float,
    max_iter: int,
    beta_hints: torch.Tensor | np.ndarray | None = None,
) -> torch.Tensor:
    dtype = torch_data.alt.dtype
    device = torch_data.alt.device
    num_mutations = int(torch_data.alt.shape[0])
    num_regions = int(torch_data.alt.shape[1])
    if torch_data.path_likelihood is not None:
        if beta_hints is None:
            hints = 0.5 * (float(eps) + torch_data.phi_upper.detach().cpu().numpy())
        else:
            hints = (
                beta_hints.detach().cpu().numpy()
                if torch.is_tensor(beta_hints)
                else np.asarray(beta_hints)
            )
        pooled = _path_pooled_start_from_model(
            _source_observed_model(torch_data),
            beta_hints=hints,
            eps=float(eps),
            tol=float(tol),
            max_iter=int(max_iter),
        )
        return torch.as_tensor(pooled, dtype=dtype, device=device)
    lower = torch.full((num_regions,), float(eps), dtype=dtype, device=device)
    upper = torch.min(torch_data.phi_upper, dim=0).values

    if beta_hints is None:
        local_left = lower
        local_right = upper
    else:
        hints = torch.as_tensor(beta_hints, dtype=dtype, device=device)
        hints = torch.minimum(
            torch.maximum(hints, torch_data.phi_upper.new_full((), float(eps))),
            torch_data.phi_upper,
        )
        hint = torch.median(hints, dim=0).values
        local_left = torch.maximum(lower, hint / 3.0)
        local_right = torch.minimum(upper, hint * 3.0)
        invalid_local = local_right <= local_left + 1e-12
        local_left = torch.where(invalid_local, lower, local_left)
        local_right = torch.where(invalid_local, upper, local_right)

    t = torch.linspace(0.0, 1.0, steps=25, dtype=dtype, device=device)
    grid = torch.exp(
        torch.log(local_left).unsqueeze(-1)
        + (torch.log(local_right) - torch.log(local_left)).unsqueeze(-1) * t
    )
    losses = _pooled_sample_loss_grid_torch(
        torch_data,
        grid,
        major_prior=major_prior,
        eps=eps,
    )
    best_index = torch.argmin(losses, dim=1)
    best_beta = torch.gather(grid, 1, best_index[:, None]).squeeze(1)
    best_value = torch.gather(losses, 1, best_index[:, None]).squeeze(1)
    left_index = torch.clamp(best_index - 1, min=0)
    right_index = torch.clamp(best_index + 1, max=grid.shape[1] - 1)
    left = torch.gather(grid, 1, left_index[:, None]).squeeze(1)
    right = torch.gather(grid, 1, right_index[:, None]).squeeze(1)
    left = torch.where(best_index == 0, lower, left)
    right = torch.where(best_index == grid.shape[1] - 1, upper, right)

    def objective(values):
        return _pooled_sample_loss_grid_torch(
            torch_data,
            values,
            major_prior=major_prior,
            eps=eps,
        )

    refined_beta, refined_value = _golden_section_minimize_samples_torch(
        objective,
        left=left,
        right=right,
        tol=float(tol),
        max_iter=max_iter,
    )
    pooled = torch.where(refined_value <= best_value, refined_beta, best_beta)
    tiled = pooled.unsqueeze(0).expand(num_mutations, -1)
    return torch.minimum(
        torch.maximum(tiled, torch_data.phi_upper.new_full((), float(eps))),
        torch_data.phi_upper,
    )


def _deduplicate_tensor_starts(starts: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    unique: list[torch.Tensor] = []
    for start in starts:
        candidate = start.detach()
        if any(
            torch.allclose(candidate, retained, rtol=0.0, atol=1e-8)
            for retained in unique
        ):
            continue
        unique.append(start)
    return tuple(unique)


def compute_scalar_well_start_bank_torch(
    torch_data: TorchTumorData,
    *,
    eps: float,
    exact_pilot: torch.Tensor,
    secondary_wells: torch.Tensor | np.ndarray | None = None,
    valid_secondary: torch.Tensor | np.ndarray | None = None,
    max_region_flips: int = 4,
) -> tuple[torch.Tensor, ...]:
    dtype = torch_data.alt.dtype
    device = torch_data.alt.device
    lower = torch.full_like(torch_data.phi_upper, float(eps))
    pilot = exact_pilot.to(dtype=dtype, device=device)
    pilot = torch.minimum(torch.maximum(pilot, lower), torch_data.phi_upper)

    starts: list[torch.Tensor] = [pilot]
    if torch_data.path_likelihood is not None:
        path = torch_data.path_likelihood
        breakpoints, breakpoint_valid = path_internal_breakpoints_torch(
            path, eps=float(eps)
        )
        interior = (
            breakpoint_valid
            & (breakpoints > float(eps))
            & (breakpoints < torch_data.phi_upper.unsqueeze(-1))
        )
        distance = torch.where(
            interior,
            torch.abs(breakpoints - pilot.unsqueeze(-1)),
            torch.full_like(breakpoints, float("inf")),
        )
        closest_index = torch.argmin(distance, dim=-1, keepdim=True)
        closest = torch.gather(breakpoints, -1, closest_index).squeeze(-1)
        has_switch = torch.any(interior, dim=-1)
        offset = max(float(eps) * 10.0, 1e-6)
        for direction in (-1.0, 1.0):
            switch_start = torch.where(
                has_switch,
                closest + float(direction) * offset,
                pilot,
            )
            starts.append(
                torch.minimum(
                    torch.maximum(switch_start, lower),
                    torch_data.phi_upper,
                )
            )
    if secondary_wells is None or valid_secondary is None:
        return _deduplicate_tensor_starts(starts)

    secondary = torch.as_tensor(secondary_wells, dtype=dtype, device=device)
    valid = torch.as_tensor(valid_secondary, dtype=torch.bool, device=device)
    valid = valid & torch.isfinite(secondary)
    if not bool(torch.any(valid).item()):
        return _deduplicate_tensor_starts(starts)

    global_alternate = torch.where(valid, secondary, pilot)
    starts.append(
        torch.minimum(torch.maximum(global_alternate, lower), torch_data.phi_upper)
    )

    region_delta = torch.where(
        valid, torch.abs(secondary - pilot), torch.zeros_like(pilot)
    )
    region_scores = torch.sum(region_delta, dim=0).detach().cpu().numpy()
    region_order = [
        (float(score), int(region_idx))
        for region_idx, score in enumerate(region_scores)
        if float(score) > 0.0
    ]
    region_order.sort(reverse=True)

    for _, region_idx in region_order[: max(0, int(max_region_flips))]:
        region_start = pilot.clone()
        mask = valid[:, region_idx]
        region_start[:, region_idx] = torch.where(
            mask,
            secondary[:, region_idx],
            region_start[:, region_idx],
        )
        starts.append(
            torch.minimum(torch.maximum(region_start, lower), torch_data.phi_upper)
        )

    return _deduplicate_tensor_starts(starts)
