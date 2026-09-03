"""Single source of truth for model-selection score primitives.

This is a leaf module: it depends only on numpy and ``io.data.TumorData`` so that
both the ``core.fusion`` layer (partition refits / candidate BIC) and the
``model_selection`` / ``runners`` layers can import it *downward*. Keeping the
BIC arithmetic in one place avoids the correctness-drift risk of re-deriving
``-2*loglik + df*log(n)`` (and the observed-mutation_region count) in several modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..io.data import TumorData

if TYPE_CHECKING:
    from .scalar import PartitionFit


@dataclass(frozen=True, slots=True)
class SelectionScore:
    """Immutable value returned by the fixed-partition score evaluators."""

    name: Literal["fixed_partition_bic"]
    value: float
    loglik: float
    penalty: float
    degrees_of_freedom: int
    n_eff: int
    partition_signature: str
    numerical_uncertainty: float = 0.0

    def validate_against(self, refit: "PartitionFit") -> None:
        """Validate this score against its authoritative fixed-label refit."""

        if self.partition_signature != refit.partition_signature:
            raise AssertionError("Selection score does not match raw partition.")
        if self.name != "fixed_partition_bic":
            raise AssertionError("Production partition selection requires BIC.")
        tolerance = 1e-10 * (1.0 + abs(float(self.value)))
        if not np.isclose(
            float(self.loglik),
            float(refit.loglik),
            rtol=0.0,
            atol=tolerance,
        ):
            raise AssertionError("Score likelihood differs from stored refit likelihood.")
        expected_penalty, expected_value = _bic_components(
            float(refit.loglik),
            float(self.degrees_of_freedom),
            float(self.n_eff),
        )
        if not np.isclose(
            float(self.penalty), expected_penalty, rtol=0.0, atol=tolerance
        ):
            raise AssertionError("Stored selection penalty is not reconstructible.")
        if not np.isclose(
            float(self.value), expected_value, rtol=0.0, atol=tolerance
        ):
            raise AssertionError("Stored score is not reconstructible.")
        minimum_uncertainty = 2.0 * max(float(refit.global_optimality_gap), 0.0)
        if (
            refit.global_optimum_certified
            and float(self.numerical_uncertainty) + tolerance < minimum_uncertainty
        ):
            raise AssertionError(
                "Score uncertainty does not cover the refit certificate gap."
            )


def _observed_positive_depth_mask(data: TumorData) -> np.ndarray:
    """Boolean (M, S) mask of mutation_regions that contribute to the likelihood.

    A mutation_region counts only if it has positive sequencing depth *and* is flagged
    observed. This is the single definition shared by the BIC denominator and
    by the partition refit numerator so the two never disagree.
    """
    mask = np.asarray(data.total_counts, dtype=np.float64) > 0.0
    return mask & np.asarray(data.objective_inclusion_mask(), dtype=bool)


def effective_bic_mutation_region_count(data: TumorData) -> int:
    return max(int(np.sum(_observed_positive_depth_mask(data))), 1)


def bic_degrees_of_freedom(num_clusters: int, data: TumorData) -> int:
    """Nominal center degrees of freedom for an unanchored K-block model."""

    return max(int(num_clusters), 0) * int(data.num_regions)


def compute_bic_with_df(
    loglik: float, degrees_of_freedom: float, num_observations: float
) -> float:
    return _bic_components(loglik, degrees_of_freedom, num_observations)[1]


def _bic_components(
    loglik: float,
    degrees_of_freedom: float,
    num_observations: float,
) -> tuple[float, float]:
    penalty = float(
        float(degrees_of_freedom) * np.log(max(float(num_observations), 1.0))
    )
    return penalty, float(-2.0 * float(loglik) + penalty)


def compute_classic_bic(loglik: float, num_clusters: int, data: TumorData) -> float:
    num_observations = effective_bic_mutation_region_count(data)
    degrees_of_freedom = bic_degrees_of_freedom(num_clusters, data)
    return compute_bic_with_df(loglik, degrees_of_freedom, num_observations)


def fixed_partition_bic(
    *,
    loglik: float,
    num_clusters: int,
    data: TumorData,
    partition_signature: str,
    labels: np.ndarray | None = None,
    loglik_uncertainty: float = 0.0,
) -> SelectionScore:
    """Return the explicitly named BIC for one immutable partition refit."""

    degrees_of_freedom = int(num_clusters) * int(data.num_regions)
    if labels is not None:
        labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_array.size != int(data.num_mutations):
            raise ValueError("BIC labels must contain one value per mutation.")
    n_eff = effective_bic_mutation_region_count(data)
    penalty, value = _bic_components(loglik, degrees_of_freedom, n_eff)
    likelihood_uncertainty = max(float(loglik_uncertainty), 0.0)
    arithmetic_uncertainty = 16.0 * np.finfo(np.float64).eps * (1.0 + abs(value))
    return SelectionScore(
        name="fixed_partition_bic",
        value=value,
        loglik=float(loglik),
        penalty=penalty,
        degrees_of_freedom=int(degrees_of_freedom),
        n_eff=int(n_eff),
        partition_signature=str(partition_signature),
        numerical_uncertainty=float(
            2.0 * likelihood_uncertainty + arithmetic_uncertainty
        ),
    )


__all__ = [
    "bic_degrees_of_freedom",
    "compute_bic_with_df",
    "compute_classic_bic",
    "effective_bic_mutation_region_count",
    "fixed_partition_bic",
    "SelectionScore",
]
