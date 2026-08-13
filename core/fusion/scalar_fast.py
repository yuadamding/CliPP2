"""Bounded vectorized scalar likelihood search for approximate profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...io.data import TumorData


@dataclass(frozen=True, slots=True)
class ApproximateScalarMinimum:
    argmin: float
    attained_value: float
    grid_points_evaluated: int
    final_grid_spacing: float
    best_second_loss_gap: float
    method: str = "vectorized_grid_local_v1"


def _observed_mask(data: TumorData) -> np.ndarray:
    observed = np.asarray(data.total_counts, dtype=np.float64) > 0.0
    if data.count_observed is not None:
        observed &= np.asarray(data.count_observed, dtype=bool)
    return observed


def evaluate_tumor_scalar_loss_grid(
    data: TumorData,
    mutation_indices: np.ndarray,
    region_index: int,
    beta_grid: np.ndarray,
    *,
    major_prior: float,
    eps: float,
) -> np.ndarray:
    """Evaluate a shared scalar grid for one cluster-region coordinate."""

    rows = np.asarray(mutation_indices, dtype=np.int64).reshape(-1)
    region = int(region_index)
    grid = np.asarray(beta_grid, dtype=np.float64).reshape(-1)
    observed = _observed_mask(data)[rows, region]
    rows = rows[observed]
    if rows.size == 0:
        return np.zeros(grid.size, dtype=np.float64)
    alt = np.asarray(data.alt_counts, dtype=np.float64)[rows, region]
    total = np.asarray(data.total_counts, dtype=np.float64)[rows, region]
    nonalt = total - alt
    beta = grid[None, :, None]
    path = data.path_likelihood
    if path is not None:
        scale = np.asarray(data.scaling, dtype=np.float64)[rows, region, None]
        first = scale * np.asarray(path.first_copy, dtype=np.float64)[rows, region]
        second = scale * np.asarray(path.second_copy, dtype=np.float64)[rows, region]
        switch = np.asarray(path.switch_fraction, dtype=np.float64)[rows, region]
        log_prior = np.asarray(path.log_prior, dtype=np.float64)[rows, region]
        valid = np.asarray(path.valid, dtype=bool)[rows, region]
    else:
        prior = float(major_prior)
        if not np.isfinite(prior) or not 0.0 < prior < 1.0:
            raise ValueError("major_prior must lie strictly in (0, 1).")
        scale = np.asarray(data.scaling, dtype=np.float64)[rows, region]
        fixed = scale * np.asarray(data.fixed_multiplicity, dtype=np.float64)[
            rows, region
        ]
        minor = scale * np.asarray(data.minor_cn, dtype=np.float64)[rows, region]
        major = scale * np.asarray(data.major_cn, dtype=np.float64)[rows, region]
        ambiguous = np.asarray(data.multiplicity_estimation_mask, dtype=bool)[
            rows, region
        ]
        first = np.column_stack((np.where(ambiguous, minor, fixed), major))
        second = first
        switch = np.zeros_like(first)
        valid = np.column_stack((np.ones(rows.size, dtype=bool), ambiguous))
        log_prior = np.column_stack(
            (
                np.where(ambiguous, np.log1p(-prior), 0.0),
                np.full(rows.size, np.log(prior), dtype=np.float64),
            )
        )
        log_prior = np.where(valid, log_prior, -np.inf)
    mass = first[:, None, :] * np.minimum(beta, switch[:, None, :])
    mass += second[:, None, :] * np.maximum(
        beta - switch[:, None, :], 0.0
    )
    probability = np.clip(mass, float(eps), 1.0 - float(eps))
    joint = (
        alt[:, None, None] * np.log(probability)
        + nonalt[:, None, None] * np.log1p(-probability)
        + log_prior[:, None, :]
    )
    joint = np.where(valid[:, None, :], joint, -np.inf)
    return -np.sum(np.logaddexp.reduce(joint, axis=-1), axis=0)


def _path_breakpoints(
    data: TumorData,
    rows: np.ndarray,
    region: int,
    *,
    lower: float,
    upper: float,
    eps: float,
) -> np.ndarray:
    path = data.path_likelihood
    if path is None:
        return np.empty(0, dtype=np.float64)
    scale = np.asarray(data.scaling, dtype=np.float64)[rows, region, None]
    first = scale * np.asarray(path.first_copy, dtype=np.float64)[rows, region]
    second = scale * np.asarray(path.second_copy, dtype=np.float64)[rows, region]
    switch = np.asarray(path.switch_fraction, dtype=np.float64)[rows, region]
    valid = np.asarray(path.valid, dtype=bool)[rows, region]
    points: list[float] = []
    for row, path_index in np.argwhere(valid):
        first_scale = float(first[row, path_index])
        second_scale = float(second[row, path_index])
        knot = float(switch[row, path_index])
        if lower < knot < upper:
            points.append(knot)
        for target in (float(eps), 1.0 - float(eps)):
            if first_scale > 0.0:
                value = target / first_scale
                if lower < value <= min(knot, upper):
                    points.append(value)
            if second_scale > 0.0:
                value = knot + (target - first_scale * knot) / second_scale
                if max(knot, lower) <= value < upper:
                    points.append(value)
    return np.asarray(points, dtype=np.float64)


def approximate_tumor_scalar_minimum(
    data: TumorData,
    mutation_indices: np.ndarray,
    region_index: int,
    *,
    lower: float,
    upper: float,
    major_prior: float,
    eps: float,
    grid_points: int,
    local_steps: int,
    hint: float | None = None,
) -> ApproximateScalarMinimum:
    """Deterministic bounded grid search with local bracket refinement."""

    lower = float(lower)
    upper = float(upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper < lower:
        raise ValueError("Require a finite scalar interval with lower <= upper.")
    if int(grid_points) < 3:
        raise ValueError("grid_points must be at least three.")
    if int(local_steps) < 0:
        raise ValueError("local_steps must be nonnegative.")
    rows = np.asarray(mutation_indices, dtype=np.int64).reshape(-1)
    initial = np.linspace(lower, upper, num=int(grid_points), dtype=np.float64)
    extras = _path_breakpoints(
        data,
        rows,
        int(region_index),
        lower=lower,
        upper=upper,
        eps=float(eps),
    )
    if hint is not None and np.isfinite(float(hint)):
        extras = np.append(extras, np.clip(float(hint), lower, upper))
    grid = np.unique(np.clip(np.concatenate((initial, extras)), lower, upper))
    evaluated: dict[float, float] = {}

    def evaluate(values: np.ndarray) -> None:
        unique = np.asarray(
            [float(value) for value in values if float(value) not in evaluated],
            dtype=np.float64,
        )
        if unique.size == 0:
            return
        losses = evaluate_tumor_scalar_loss_grid(
            data,
            rows,
            int(region_index),
            unique,
            major_prior=float(major_prior),
            eps=float(eps),
        )
        evaluated.update(zip(unique.tolist(), losses.tolist()))

    evaluate(grid)
    final_spacing = float(upper - lower)
    for _ in range(int(local_steps)):
        ordered = np.asarray(sorted(evaluated), dtype=np.float64)
        losses = np.asarray([evaluated[float(value)] for value in ordered])
        best = int(np.nanargmin(losses))
        left = float(ordered[max(best - 1, 0)])
        right = float(ordered[min(best + 1, ordered.size - 1)])
        if right <= left:
            break
        local = np.linspace(left, right, num=5, dtype=np.float64)
        final_spacing = float((right - left) / 4.0)
        evaluate(local)
    ordered = np.asarray(sorted(evaluated), dtype=np.float64)
    losses = np.asarray([evaluated[float(value)] for value in ordered])
    finite = np.flatnonzero(np.isfinite(losses))
    if finite.size == 0:
        return ApproximateScalarMinimum(
            argmin=float(np.clip(hint if hint is not None else lower, lower, upper)),
            attained_value=float("inf"),
            grid_points_evaluated=int(len(evaluated)),
            final_grid_spacing=float(final_spacing),
            best_second_loss_gap=float("inf"),
        )
    ranked = finite[np.argsort(losses[finite], kind="stable")]
    best_index = int(ranked[0])
    gap = (
        float(losses[int(ranked[1])] - losses[best_index])
        if ranked.size > 1
        else float("inf")
    )
    return ApproximateScalarMinimum(
        argmin=float(ordered[best_index]),
        attained_value=float(losses[best_index]),
        grid_points_evaluated=int(len(evaluated)),
        final_grid_spacing=float(final_spacing),
        best_second_loss_gap=max(gap, 0.0),
    )


__all__ = [
    "ApproximateScalarMinimum",
    "approximate_tumor_scalar_minimum",
    "evaluate_tumor_scalar_loss_grid",
]
