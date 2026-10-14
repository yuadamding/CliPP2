"""One occupied clonal center: projection helpers, never a synthetic tree node."""
import numpy as np


def project_anchor(x, eligible):
    """Nearest complete CCF vector; canonical row order breaks exact ties."""
    x = np.asarray(x, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if x.ndim != 2 or eligible.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError("Anchor projection needs finite N by R coordinates and an N mask")
    rows = np.flatnonzero(eligible)
    if not len(rows):
        raise ValueError("clonal_infeasible_no_eligible_mutation")
    distances = np.sum((x[rows]-1.)**2, axis=1)
    index = int(np.argmin(distances))
    return int(rows[index]), float(np.sqrt(distances[index]))


def anchor_penalty(x, witness, penalty):
    return .5 * penalty * float(np.sum((x[witness]-1.)**2))


def anchored_quadratic(x, gradient, curvature, witness, penalty):
    """Eliminate a fixed node at one; form the original RHS before augmentation."""
    h = np.maximum(curvature, 1e-8).copy()
    rhs = h*x-gradient
    h[witness] += penalty
    rhs[witness] += penalty
    return h, rhs
