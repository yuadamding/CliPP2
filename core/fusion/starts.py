from __future__ import annotations

import numpy as np
import torch

from ..objective import (
    ObservedModel,
    TorchObservedModel,
    observed_terms_torch,
)
from ..scalar import (
    ScalarGlobalMinimumCertificate,
    ScalarProblem,
    certify_scalar_minimum,
    scalar_breakpoints,
    scalar_loss,
    scalar_loss_and_gradient,
    scalar_problem_from_model,
)
from .types import TorchRuntime


_ROOT_SCAN_POINTS = 65


def initialize_marginal_phi(model: ObservedModel, *, eps: float) -> np.ndarray:
    """Canonical marginalized scalar initializer; never a hard component fit."""
    primary, _secondary, _valid = _scalar_wells_from_model(
        model, phi_init=np.clip(np.full(model.shape, 0.5, dtype=np.float64), eps, model.upper),
        eps=eps, tol=1e-10, max_iter=256,
    )
    return np.clip(np.asarray(primary, dtype=np.float64), eps, model.upper)


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


def _unit_base_points_numpy(
    problem: ScalarProblem, *, hint: float,
) -> tuple[np.ndarray, np.ndarray]:
    if problem.alt.size != 1:
        raise ValueError("Scalar-well base points require one mutation-region row.")
    lower, upper = problem.lower, problem.upper
    hard = scalar_breakpoints(problem, observed_only=False)
    total = float(problem.alt[0] + problem.nonalt[0])
    smoothed_vaf = (float(problem.alt[0]) + 0.5) / (total + 1.0)
    slope = problem.slope[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = smoothed_vaf / slope
    eligible = problem.valid[0] & (slope > 0.0) & np.isfinite(inverse)
    points = np.concatenate((hard, [float(hint)], inverse[eligible]))
    point_array = np.unique(np.round(points, 14))
    hard_array = np.unique(np.round(hard, 14))
    keep = (point_array >= lower - 1e-12) & (point_array <= upper + 1e-12)
    hard_keep = (hard_array >= lower - 1e-12) & (hard_array <= upper + 1e-12)
    return np.clip(point_array[keep], lower, upper), np.clip(hard_array[hard_keep], lower, upper)


def _unit_best_two_betas_numpy(
    problem: ScalarProblem,
    *,
    hint: float,
    tol: float,
    max_iter: int,
) -> tuple[float, float | None]:
    base_points, hard_points = _unit_base_points_numpy(
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


def _scalar_wells_from_model(
    model: ObservedModel,
    *,
    phi_init: np.ndarray,
    eps: float,
    tol: float,
    max_iter: int,
    certificates: list[ScalarGlobalMinimumCertificate] | None = None,
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
    for index in range(primary.size):
        if not observed[index]:
            if certificates is not None:
                certificates.append(ScalarGlobalMinimumCertificate(
                    float(primary[index]), 0.0, 0.0, 0.0, True,
                    "uninformative_scalar_coordinate_v1", 0,
                ))
            continue
        mutation_index, region_index = np.unravel_index(index, shape)
        problem = scalar_problem_from_model(
            model,
            np.asarray([mutation_index], dtype=np.int64),
            int(region_index),
            lower=float(eps),
            upper=float(flat_upper[index]),
            eps=float(eps),
        )
        best, alternate = _unit_best_two_betas_numpy(
            problem,
            hint=float(flat_hint[index]),
            tol=float(tol),
            max_iter=int(max_iter),
        )
        if certificates is not None:
            certificate = certify_scalar_minimum(
                problem,
                tolerance=max(1e-10, min(float(tol), 1e-6)),
                max_intervals=max(256, min(int(max_iter) * 16, 4096)),
                hint=best,
            )
            certificates.append(certificate)
            best = certificate.argmin
        primary[index] = best
        if alternate is not None:
            secondary[index] = alternate
            valid_secondary[index] = True
    return (
        primary.reshape(shape),
        secondary.reshape(shape),
        valid_secondary.reshape(shape),
    )


def _pooled_start_from_model(
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
        member_indices = np.flatnonzero(region_observed)
        problem = scalar_problem_from_model(
            model, member_indices, region_index, lower=lower, upper=upper, eps=float(eps),
        )
        hard = scalar_breakpoints(problem)
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
            at_breakpoint = bool(np.any(np.isclose(center, hard, rtol=0.0, atol=1e-12)))
            intervals = (
                [
                    (float(grid[max(index - 1, 0)]), center),
                    (center, float(grid[min(index + 1, grid.size - 1)])),
                ]
                if at_breakpoint
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


def _linear_candidate_starts_torch(
    model: TorchObservedModel, *, pilot: torch.Tensor, eps: float
) -> tuple[torch.Tensor, ...]:
    """Bounded candidate seeds, each refined against the *full* mixture.

    One matrix per candidate index avoids enumerating cross-mutation state
    assignments. Safeguarded gradient steps never select a hard component or
    introduce multiplicity as optimization state. These are local starts, not
    global scalar certificates.
    """

    if int(model.candidate_shape[-1]) <= 2:
        return ()
    lower, upper = model.lower, model.upper
    informed = model.observed & (model.total > 0.0)
    vaf = (model.alt + 0.5) / (model.total + 1.0)
    starts: list[torch.Tensor] = []
    for candidate_index in range(min(int(model.candidate_shape[-1]), 6)):
        slope = model.slope[..., candidate_index]
        eligible = informed & model.valid[..., candidate_index] & (slope > 0.0)
        seed = torch.where(
            eligible, vaf / torch.clamp(slope, min=torch.finfo(slope.dtype).tiny),
            pilot,
        )
        current = torch.minimum(torch.maximum(seed, lower), upper)
        for _ in range(12):
            terms = observed_terms_torch(model, current, eps=eps)
            step = terms.gradient / torch.clamp(terms.hessian_upper, min=1e-8)
            accepted = torch.zeros_like(eligible)
            next_phi = current
            for backtrack in range(8):
                trial = torch.minimum(torch.maximum(
                    current - (0.5 ** backtrack) * step, lower
                ), upper)
                trial_terms = observed_terms_torch(model, trial, eps=eps)
                take = eligible & ~accepted & torch.isfinite(trial_terms.loss) & (
                    trial_terms.loss <= terms.loss
                )
                next_phi = torch.where(take, trial, next_phi)
                accepted |= take
                if bool(torch.all(accepted | ~eligible).item()):
                    break
            current = next_phi
        starts.append(current)
    return _deduplicate_tensor_starts(starts)


def compute_scalar_well_start_bank_torch(
    model: TorchObservedModel,
    *,
    eps: float,
    exact_pilot: torch.Tensor,
    secondary_wells: torch.Tensor | np.ndarray | None = None,
    valid_secondary: torch.Tensor | np.ndarray | None = None,
    max_region_flips: int = 4,
) -> tuple[torch.Tensor, ...]:
    dtype = model.alt.dtype
    device = model.alt.device
    lower = torch.full_like(model.upper, float(eps))
    pilot = exact_pilot.to(dtype=dtype, device=device)
    pilot = torch.minimum(torch.maximum(pilot, lower), model.upper)

    starts: list[torch.Tensor] = [pilot]
    starts.extend(_linear_candidate_starts_torch(
        model, pilot=pilot, eps=float(eps)
    ))
    if secondary_wells is None or valid_secondary is None:
        return _deduplicate_tensor_starts(starts)

    secondary = torch.as_tensor(secondary_wells, dtype=dtype, device=device)
    valid = torch.as_tensor(valid_secondary, dtype=torch.bool, device=device)
    valid = valid & torch.isfinite(secondary)
    if not bool(torch.any(valid).item()):
        return _deduplicate_tensor_starts(starts)

    global_alternate = torch.where(valid, secondary, pilot)
    starts.append(
        torch.minimum(torch.maximum(global_alternate, lower), model.upper)
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
            torch.minimum(torch.maximum(region_start, lower), model.upper)
        )

    return _deduplicate_tensor_starts(starts)


def compute_scalar_mutation_region_wells_torch(
    model: ObservedModel, runtime: TorchRuntime, *, phi_init: torch.Tensor | np.ndarray,
    eps: float, tol: float, max_iter: int,
    certificates: list[ScalarGlobalMinimumCertificate] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype, device = runtime.dtype, runtime.device
    primary, secondary, valid = _scalar_wells_from_model(
        model,
        phi_init=phi_init.detach().cpu().numpy() if torch.is_tensor(phi_init) else np.asarray(phi_init),
        eps=float(eps), tol=float(tol), max_iter=int(max_iter), certificates=certificates,
    )
    return (
        torch.as_tensor(primary, dtype=dtype, device=device),
        torch.as_tensor(secondary, dtype=dtype, device=device),
        torch.as_tensor(valid, dtype=torch.bool, device=device),
    )


def compute_pooled_observed_data_start_torch(
    model: ObservedModel, runtime: TorchRuntime, *, eps: float, tol: float, max_iter: int,
    beta_hints: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    hints = beta_hints.detach().cpu().numpy() if torch.is_tensor(beta_hints) else np.asarray(beta_hints)
    pooled = _pooled_start_from_model(
        model, beta_hints=hints,
        eps=float(eps), tol=float(tol), max_iter=int(max_iter),
    )
    return torch.as_tensor(pooled, dtype=runtime.dtype, device=runtime.device)
