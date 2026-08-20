"""Compatibility adapters for the canonical certified scalar solver."""

from __future__ import annotations

import numpy as np

from ...io.data import TumorData
from ..scalar import (
    ScalarGlobalMinimumCertificate,
    certify_scalar_minimum,
    scalar_loss,
)
from .scalar_fast import _problem


def evaluate_tumor_scalar_loss(
    data: TumorData,
    mutation_indices: np.ndarray,
    region_index: int,
    beta: float,
    *,
    major_prior: float,
    eps: float,
) -> float:
    problem = _problem(
        data,
        mutation_indices,
        region_index,
        lower=float(eps),
        upper=1.0,
        major_prior=major_prior,
        eps=eps,
    )
    return float(scalar_loss(problem, beta))


def certify_tumor_scalar_minimum(
    data: TumorData,
    mutation_indices: np.ndarray,
    region_index: int,
    *,
    lower: float,
    upper: float,
    major_prior: float,
    eps: float,
    tolerance: float,
    max_intervals: int,
    hint: float | None = None,
) -> ScalarGlobalMinimumCertificate:
    """Run the canonical interval solver through the historical API."""

    return certify_scalar_minimum(
        _problem(
            data,
            mutation_indices,
            region_index,
            lower=lower,
            upper=upper,
            major_prior=major_prior,
            eps=eps,
        ),
        tolerance=tolerance,
        max_intervals=max_intervals,
        hint=hint,
    )


__all__ = [
    "ScalarGlobalMinimumCertificate",
    "certify_tumor_scalar_minimum",
    "evaluate_tumor_scalar_loss",
]
