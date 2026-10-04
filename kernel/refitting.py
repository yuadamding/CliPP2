"""CliPP2 conditional scalar multimode refits (not global certificates)."""
# Adapted from CliPP1.5; see ../NOTICE.

import numpy as np
from scipy.optimize import minimize_scalar


def refit_center(model):
    """Refine bracketed modes and edge intervals, retaining physical endpoints."""
    component_modes = (model.r / model.n)[:, None] / model.scale
    grid = np.unique(
        np.concatenate(
            (np.linspace(0, model.purity, 513), np.clip(component_modes[model.valid], 0, model.purity))
        )
    )
    values = model.grid_log_likelihood(grid)
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
    intervals = {(float(grid[0]), float(grid[1])), (float(grid[-2]), float(grid[-1]))}
    intervals.update((float(grid[index - 1]), float(grid[index + 1])) for index in maxima)
    for lower, upper in sorted(intervals):
        fit = minimize_scalar(
            lambda cp: -float(model.log_likelihood(cp).sum()),
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 500},
        )
        if not fit.success or not np.isfinite(fit.fun):
            raise RuntimeError("Marginal-likelihood center refit failed: %s" % fit.message)
        candidate = (-float(fit.fun), float(fit.x))
        if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
            best = candidate
    if not np.isfinite(best[0]):
        raise RuntimeError("No finite marginal likelihood in cluster refit")
    return best[1], best[0]
