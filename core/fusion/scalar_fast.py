"""Compatibility adapters for the canonical approximate scalar solver."""

from __future__ import annotations

import numpy as np

from ...io.data import TumorData
from ..objective import compile_observed_model
from ..scalar import (
    ApproximateScalarMinimum,
    approximate_scalar_minimum,
    scalar_loss,
    scalar_problem_from_model,
)


def _problem(
    data: TumorData,
    mutation_indices: np.ndarray,
    region_index: int,
    *,
    lower: float,
    upper: float,
    major_prior: float,
    eps: float,
):
    model = compile_observed_model(data, major_prior=major_prior, eps=eps)
    return scalar_problem_from_model(
        model,
        mutation_indices,
        region_index,
        lower=lower,
        upper=upper,
        eps=eps,
    )


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

    grid = np.asarray(beta_grid, dtype=np.float64).reshape(-1)
    problem = _problem(
        data,
        mutation_indices,
        region_index,
        lower=float(eps),
        upper=1.0,
        major_prior=major_prior,
        eps=eps,
    )
    return np.asarray(scalar_loss(problem, grid), dtype=np.float64)


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
    """Run the canonical grid-local solver through the historical API."""

    problem = _problem(
        data,
        mutation_indices,
        region_index,
        lower=lower,
        upper=upper,
        major_prior=major_prior,
        eps=eps,
    )
    return approximate_scalar_minimum(
        problem,
        grid_points=grid_points,
        local_steps=local_steps,
        hint=hint,
        # Historical grid-local legacy fits did not add analytical knots.
        include_breakpoints=data.path_likelihood is not None,
    )


__all__ = [
    "ApproximateScalarMinimum",
    "approximate_tumor_scalar_minimum",
    "evaluate_tumor_scalar_loss_grid",
]
