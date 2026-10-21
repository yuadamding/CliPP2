"""Conditional CCF-coordinate refits shared by every regional cluster."""
# Adapted from CliPP1.5; see ../NOTICE.

import numpy as np
from scipy.optimize import minimize_scalar


def refit_center(view, bounds=None):
    """Refine bracketed modes and edge intervals within the membership box.

    This is one scalar-coordinate optimization inside the regional estimator,
    not a second scalar model or a global-optimality certificate.
    """
    if bounds is None:
        lower, upper = view.bounds()
        bounds = (float(lower.max()), float(upper.min()))
    lower, upper = bounds
    if not (np.isfinite(lower) and np.isfinite(upper) and 0 <= lower <= upper <= 1):
        raise ValueError("Invalid conditional CCF box")
    grid = np.unique(np.r_[np.linspace(lower, upper, 513),
                           np.clip(view.component_modes(), lower, upper)])
    values = view.grid_log_likelihood(grid)
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
    # Preserve the standard policy's physical resolution after CP -> CCF.
    tolerance = 1e-12 / view.purity if view.standard and lower == 0 and upper == 1 else 1e-12
    for lower, upper in sorted(intervals):
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
