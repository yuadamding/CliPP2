"""The sole additional prior: an occupied CCF-one vector in every region.

Eligibility is defined on the original float64 box, never rounded working
bounds, counts, a pilot, or a distance-to-one threshold. No other center is
restricted and no loss or allocation term is added by these helpers.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np

from ..io.data import readonly_array

CLONAL_CONSTRAINT_ID = "occupied_all_regions_ccf_one_v1"


class ClonalConstraintInfeasibleError(ValueError):
    """No retained mutation's original domain permits an all-one CCF vector."""


class ClonalPartitionInfeasibleError(ValueError):
    """A proposed partition has no occupied block feasible at the clonal center."""


def _box(lower, upper) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if (lo.ndim != 2 or lo.shape != hi.shape or not all(lo.shape)
            or not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or np.any(lo > hi)):
        raise ValueError("CCF bounds must be aligned, finite, nonempty 2-D intervals.")
    return lo, hi


def clonal_eligible_rows(lower, upper) -> np.ndarray:
    """Rows whose original box contains exactly one in every region."""
    lo, hi = _box(lower, upper)
    return np.all((lo <= 1.0) & (hi >= 1.0), axis=1)


def require_clonal_eligible_rows(lower, upper) -> np.ndarray:
    eligible = clonal_eligible_rows(lower, upper)
    if not np.any(eligible):
        raise ClonalConstraintInfeasibleError(
            "No retained mutation permits CCF = 1 in every region under "
            "the current compiled CCF bounds."
        )
    return eligible


def clonal_members(phi) -> np.ndarray:
    """Exact joint all-region membership; separate regional maxima do not count."""
    values = np.asarray(phi, dtype=np.float64)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("CCFs must be a finite, nonempty mutation-region matrix.")
    return np.all(values == 1.0, axis=1)


def make_clonal_witness_bounds(lower, upper, witness_index) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = _box(lower, upper)
    if (isinstance(witness_index, bool) or not isinstance(witness_index, Integral)
            or not 0 <= witness_index < len(lo)):
        raise ValueError("Clonal witness must identify one existing mutation row.")
    if not clonal_eligible_rows(lo, hi)[witness_index]:
        raise ClonalConstraintInfeasibleError("The chosen witness cannot reach CCF 1 in every region.")
    lo, hi = lo.copy(), hi.copy()
    lo[witness_index, :] = hi[witness_index, :] = 1.0
    return readonly_array(lo, dtype=np.float64), readonly_array(hi, dtype=np.float64)


def validate_clonal_feasibility(phi, lower, upper) -> np.ndarray:
    """Validate the original domain and return the occupied exact clonal group."""
    lo, hi = _box(lower, upper)
    values = np.asarray(phi, dtype=np.float64)
    members = clonal_members(values)
    if values.shape != lo.shape or np.any(values < lo) or np.any(values > hi):
        raise ValueError("CCFs violate the original compiled CCF bounds.")
    if not np.any(members):
        raise ValueError("An occupied cluster at CCF 1 in every region is required.")
    return members
