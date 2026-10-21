"""Conditional CCF-coordinate refits shared by every regional cluster."""
# Adapted from CliPP1.5; see ../NOTICE.

import numpy as np
from scipy.optimize import minimize_scalar


def _grid(view, bounds):
    if bounds is None:
        lower, upper = view.bounds()
        bounds = (float(lower.max()), float(upper.min()))
    lower, upper = bounds
    if not (np.isfinite(lower) and np.isfinite(upper) and 0 <= lower <= upper <= 1):
        raise ValueError("Invalid conditional CCF box")
    return np.unique(np.r_[np.linspace(lower, upper, 513),
                           np.clip(view.component_modes(), lower, upper)]), (lower, upper)


def _brackets(grid, values):
    best_index = int(np.argmax(values))
    best = (float(values[best_index]), float(grid[best_index]))
    maxima = (
        np.flatnonzero(
            (values[1:-1] >= values[:-2])
            & (values[1:-1] >= values[2:])
            & ((values[1:-1] > values[:-2]) | (values[1:-1] > values[2:]))
        )
        + 1
    )
    # A maximum just inside a finite boundary can beat that endpoint while
    # every interior grid point is worse. It then has no grid-local maximum
    # to bracket. Search both edge intervals even when neither is bracketed;
    # the explicit grid candidates above still preserve exact endpoints.
    intervals = set() if len(grid) == 1 else {(float(grid[0]), float(grid[1])), (float(grid[-2]), float(grid[-1]))}
    intervals.update((float(grid[index - 1]), float(grid[index + 1])) for index in maxima)
    return best, sorted(intervals)


def _tolerance(view, bounds):
    lower, upper = bounds
    return 1e-12 / view.purity if view.standard and lower == 0 and upper == 1 else 1e-12


def refit_center(view, bounds=None):
    """Unchanged scalar reference: all multimodal brackets and both endpoints."""
    grid, bounds = _grid(view, bounds)
    best, intervals = _brackets(grid, view.grid_log_likelihood(grid))
    # Preserve the standard policy's physical resolution after CP -> CCF.
    tolerance = _tolerance(view, bounds)
    for lower, upper in intervals:
        fit = minimize_scalar(
            lambda ccf: -float(view.log_likelihood(ccf).sum()),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": tolerance, "maxiter": 500},
        )
        if not fit.success or not np.isfinite(fit.fun):
            raise RuntimeError("Marginal-likelihood center refit failed: %s" % fit.message)
        candidate = (-float(fit.fun), float(fit.x))
        if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
            best = candidate
    if not np.isfinite(best[0]):
        raise RuntimeError("No finite marginal likelihood in cluster refit")
    return best[1], best[0]


def refit_centers(tasks, workspace=None):
    """Batch independent cluster-region searches, retaining every original bracket.

    Each task is (RegionalView, bounds). Only numeric likelihood work is batched;
    per-interval Brent decisions/tolerances/limits remain the scalar reference's.
    No common grid, unimodal replacement, approximate pruning or tie prior is used.
    """
    if workspace is None:
        return [refit_center(view, bounds) for view, bounds in tasks]
    from ._bounded import bounded_steps

    grids_bounds = [_grid(view, bounds) for view, bounds in tasks]
    values = workspace.task_grid_values([(view, gb[0])
                                        for (view, _), gb in zip(tasks, grids_bounds)])
    if len(values) != len(tasks):
        raise RuntimeError("Batched refit grid count does not match tasks")
    best, pending = [], []
    for index, ((view, _), (grid, bounds), value) in enumerate(zip(tasks, grids_bounds, values)):
        value = np.asarray(value)
        if value.shape != grid.shape or np.isnan(value).any() or np.isposinf(value).any():
            raise RuntimeError("Invalid batched refit grid shape or values")
        candidate, intervals = _brackets(grid, value)
        best.append(candidate)
        for interval in intervals:
            iterator = bounded_steps(interval, _tolerance(view, bounds))
            pending.append((index, iterator, next(iterator)))
    while pending:
        losses = -np.asarray(workspace.task_values(
            [tasks[index][0] for index, _, _ in pending], [x for _, _, x in pending]))
        if losses.shape != (len(pending),):
            raise RuntimeError("Batched refit loss shape does not match tasks")
        remaining = []
        for (index, iterator, _), loss in zip(pending, losses):
            try:
                remaining.append((index, iterator, iterator.send(float(loss))))
            except StopIteration as finished:
                center, value, success = finished.value
                if not success or not np.isfinite(value):
                    raise RuntimeError("Marginal-likelihood center refit failed in batched interval")
                candidate = (-value, center)
                old = best[index]
                if candidate[0] > old[0] or (candidate[0] == old[0] and candidate[1] < old[1]):
                    best[index] = candidate
        pending = remaining
    if any(not np.isfinite(value) for value, _ in best):
        raise RuntimeError("No finite marginal likelihood in cluster refit")
    return [(center, value) for value, center in best]
