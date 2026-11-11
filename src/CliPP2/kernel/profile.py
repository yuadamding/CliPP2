"""Exact-support switch preparation for branch-complete scalar profile fitting."""
import numpy as np


def switch_grid(alt, depth, scaling, minimum, maximum, lower, upper):
    """Adjacent switches, with floating-point brackets; never tolerance-merge.

    Higher precision is used only to prepare breakpoints, not likelihoods.
    Both neighbors of each rounded switch are retained, so a rounding error
    cannot drop an arbitrarily narrow branch. Endpoints are always evaluated.
    """
    points = [lower, upper]
    y, d, a = map(np.longdouble, (alt, depth, scaling))
    if 0 < y < d:
        for m in range(int(minimum), int(maximum)):
            t = -np.expm1(-(y / (d-y)) * np.log1p(1 / np.longdouble(m)))
            switch = t / (a * (1 + m*t))
            c = float(switch)
            if not lower < c < upper:
                continue
            lo, hi = np.nextafter(c, -np.inf), np.nextafter(c, np.inf)
            # Stable adjacent log-mass difference must bracket its root.
            def difference(value):
                v = np.longdouble(value)
                return y*np.log1p(1/np.longdouble(m)) + (d-y)*np.log1p(-a*v/(1-a*m*v))
            eps = 64*np.finfo(np.longdouble).eps*max(1, float(y+d))
            if difference(lo) < -eps or difference(hi) > eps:
                raise ArithmeticError("Adjacent profile switch could not be bracketed")
            points.extend((max(lower, lo), c, min(upper, hi)))
    # Clipped all-ALT components have a plateau after p=1.
    if y == d:
        for m in range(int(minimum), int(maximum)+1):
            c = float(1/(a*m))
            if lower < c < upper:
                points.append(c)
    return np.unique(points)
