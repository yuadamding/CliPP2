"""Source-bound scalar bookkeeping for the unchanged clonal-witness search."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ...io.data import readonly_array
from ..clonal import clonal_eligible_rows
from ..objective import ObservedModel, _observed_reduction_numpy
from ..scalar import ScalarGlobalMinimumCertificate
from .types import PreparedProblem

_EPS = np.finfo(np.float64).eps
_CACHE_TAG = "clonal_witness_scalar_data_v1"


def _gamma(count: int) -> float:
    # Using eps rather than unit roundoff also guards arithmetic that constructs
    # the allowance. Refuse pruning if the standard summation bound is vacuous.
    error = count * _EPS
    return error / (1.0 - error) if error < 0.5 else float("inf")


def _vectorized_lower_bounds(bounds: np.ndarray, clonal_loss: np.ndarray):
    """B - b_j + c_j, with allowance for both new and former reductions.

    The old implementation replaced a row and summed the entire matrix for
    every witness. Its near-boundary decisions must not become more aggressive
    merely because subtraction or reduction order changed. The returned error
    covers the distance from the real sum *and* from that former floating sum.
    All work here is O(MR), once, rather than once per witness.
    """
    count, regions = bounds.size, bounds.shape[1]
    full_gamma, row_gamma = _gamma(count), _gamma(regions)
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(np.sum(bounds, dtype=np.float64))
        rows = np.sum(bounds, axis=1, dtype=np.float64)
        clonal = np.sum(clonal_loss, axis=1, dtype=np.float64)
        lower = total - rows + clonal
        absolute_total = np.nextafter(
            np.sum(np.abs(bounds), dtype=np.float64) / (1.0 - full_gamma), np.inf,
        )
        absolute_rows = np.nextafter(
            np.sum(np.abs(bounds), axis=1, dtype=np.float64) / (1.0 - row_gamma), np.inf,
        )
        absolute_clonal = np.nextafter(
            np.sum(np.abs(clonal_loss), axis=1, dtype=np.float64) / (1.0 - row_gamma), np.inf,
        )
        # Deliberately retain the removed row in the absolute-sum upper bound:
        # no cancellation-sensitive subtraction is needed for this allowance.
        absolute_terms = np.nextafter(
            (absolute_total + absolute_clonal) * (1.0 + full_gamma), np.inf,
        )
        uncertainty = (
            full_gamma * absolute_total
            + row_gamma * (absolute_rows + absolute_clonal)
            + _gamma(2) * (abs(total) + np.abs(rows) + np.abs(clonal))
            + full_gamma * absolute_terms
        )
        uncertainty = np.nextafter(
            uncertainty * (1.0 + 16.0 * _EPS)
            + (count + regions + 8) * np.finfo(np.float64).tiny, np.inf,
        )
    return tuple(readonly_array(value) for value in (lower, uncertainty, absolute_terms))


@dataclass(frozen=True, slots=True)
class WitnessScalarData:
    """Immutable likelihood bounds; never an attained-minimum surrogate.

    These quantities do not depend on lambda, working dtype, graph, or the
    effective witness box. They do depend on the original source/epsilon and
    the exact immutable scalar-certificate tuple, retained as identity owners.
    """

    eligible: tuple[int, ...]
    scalar_bounds: np.ndarray | None
    clonal_loss: np.ndarray
    lower_bounds: np.ndarray | None
    reduction_uncertainty: np.ndarray | None
    absolute_terms_upper: np.ndarray | None
    _source: ObservedModel = field(repr=False)
    _certificates: tuple[ScalarGlobalMinimumCertificate, ...] = field(repr=False)
    _epsilon: str = field(repr=False)
    _source_fingerprint: str = field(repr=False)

    def assert_bound_to(self, problem: PreparedProblem) -> None:
        if (
            problem.source_model is not self._source
            or problem.source_model.fingerprint != self._source_fingerprint
            or problem.eps.hex() != self._epsilon
            or problem.scalar_pilot_certificates is not self._certificates
        ):
            raise ValueError("Witness scalar cache has a changed source, epsilon, or pilot.")

    def can_prune(self, index: int, incumbent_objective: float) -> bool:
        """Decline pruning whenever rounding could change the former decision."""
        if self.lower_bounds is None or not np.isfinite(incumbent_objective):
            return False
        lower = float(self.lower_bounds[index])
        uncertainty = float(self.reduction_uncertainty[index])
        # Retain the previous two-sided 256*eps margin, using an upper bound
        # on its absolute sum, and add the reduction-change allowance.
        slack = 256.0 * _EPS * (
            1.0 + float(self.absolute_terms_upper[index]) + abs(incumbent_objective)
        )
        return bool(
            np.isfinite(lower) and np.isfinite(uncertainty) and np.isfinite(slack)
            and lower - uncertainty - slack > incumbent_objective + slack
        )


def witness_scalar_data(problem: PreparedProblem) -> WitnessScalarData:
    """Reuse one immutable source/pilot record across starts and lambdas.

    The prepared problem's full external validation remains the caller's
    responsibility. Version checks prevent reuse after runtime tensor edits.
    Namespacing keeps this host record separate from per-box float64 audits.
    """
    problem.assert_runtime_unchanged()
    source, certificates = problem.source_model, problem.scalar_pilot_certificates
    epsilon = problem.eps.hex()
    key = (_CACHE_TAG, source.fingerprint, epsilon, str(id(source)), str(id(certificates)))
    cached = problem.audit_context_cache.get(key)
    if cached is not None:
        if not isinstance(cached, WitnessScalarData):
            raise TypeError("PreparedProblem witness scalar cache is corrupted.")
        cached.assert_bound_to(problem)
        return cached
    eligible = tuple(int(index) for index in np.flatnonzero(
        clonal_eligible_rows(source.lower, source.upper),
    ))
    clonal_loss = readonly_array(_observed_reduction_numpy(
        source, np.ones(source.shape, dtype=np.float64), eps=problem.eps, output="loss",
    ))
    bounds = lower = uncertainty = absolute_terms = None
    if len(certificates) == int(np.prod(source.shape)) and all(
        item.globally_certified
        and np.isfinite(item.global_lower_bound)
        and np.isfinite(item.attained_value)
        and item.global_lower_bound <= item.attained_value
        for item in certificates
    ):
        bounds = readonly_array(np.asarray([
            item.global_lower_bound for item in certificates
        ], dtype=np.float64).reshape(source.shape))
        lower, uncertainty, absolute_terms = _vectorized_lower_bounds(bounds, clonal_loss)
    record = WitnessScalarData(
        eligible, bounds, clonal_loss, lower, uncertainty, absolute_terms,
        source, certificates, epsilon, source.fingerprint,
    )
    problem.audit_context_cache[key] = record
    return record


def separable_zero_witness(
    problem: PreparedProblem, data: WitnessScalarData,
) -> tuple[int, np.ndarray] | None:
    """Retain the exact zero-gap/representability shortcut, using loss only."""
    data.assert_bound_to(problem)
    certificates, source = problem.scalar_pilot_certificates, problem.source_model
    if not data.eligible or len(certificates) != int(np.prod(source.shape)) or not all(
        item.globally_certified and item.optimality_gap == 0.0
        and np.isfinite(item.argmin) and np.isfinite(item.attained_value)
        and item.global_lower_bound == item.attained_value
        for item in certificates
    ):
        return None
    phi = np.asarray([item.argmin for item in certificates], dtype=np.float64).reshape(source.shape)
    if np.any(phi < source.lower) or np.any(phi > source.upper):
        return None
    if problem.runtime.dtype == torch.float32 and not np.array_equal(phi, phi.astype(np.float32)):
        return None
    free_loss = _observed_reduction_numpy(source, phi, eps=problem.eps, output="loss")
    attained = np.asarray([item.attained_value for item in certificates]).reshape(source.shape)
    if not np.array_equal(free_loss, attained):
        return None
    deltas = np.sum(data.clonal_loss - free_loss, axis=1)
    if not np.all(np.isfinite(deltas[list(data.eligible)])):
        return None
    witness = min(data.eligible, key=lambda index: (float(deltas[index]), index))
    phi[witness] = 1.0
    return witness, phi
