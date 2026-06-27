"""Single source of truth for BIC primitives and lambda-grid mode helpers.

This is a leaf module: it depends only on numpy and ``io.data.TumorData`` so that
both the ``core.fusion`` layer (partition refits / candidate BIC) and the
``model_selection`` / ``runners`` layers can import it *downward*. Keeping the
BIC arithmetic in one place avoids the correctness-drift risk of re-deriving
``-2*loglik + df*log(n)`` (and the observed-mutation_region count) in several modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.data import TumorData


ADAPTIVE_LAMBDA_GRID_MODES = ("adaptive_bic",)
LAMBDA_GRID_MODES = ADAPTIVE_LAMBDA_GRID_MODES


@dataclass(frozen=True)
class LambdaBracket:
    lambda_min: float
    lambda_eq: float
    lambda_full: float
    anchors: list[float]
    diagnostics: dict[str, float]


def is_adaptive_lambda_grid_mode(mode: str) -> bool:
    return str(mode).strip().lower() in ADAPTIVE_LAMBDA_GRID_MODES


def _observed_positive_depth_mask(data: TumorData) -> np.ndarray:
    """Boolean (M, S) mask of mutation_regions that contribute to the likelihood.

    A mutation_region counts only if it has positive sequencing depth *and* is flagged
    observed. This is the single definition shared by the BIC denominator and
    by the partition refit numerator so the two never disagree.
    """
    mask = np.asarray(data.total_counts, dtype=np.float64) > 0.0
    count_observed = getattr(data, "count_observed", None)
    if count_observed is not None:
        mask = mask & np.asarray(count_observed, dtype=bool)
    return mask


def effective_bic_mutation_region_count(data: TumorData) -> int:
    return max(int(np.sum(_observed_positive_depth_mask(data))), 1)


def effective_bic_depth_count(data: TumorData) -> float:
    depth = np.asarray(data.total_counts, dtype=np.float64)
    return float(max(float(np.sum(depth[_observed_positive_depth_mask(data)])), 1.0))


def bic_degrees_of_freedom(num_clusters: int, data: TumorData) -> int:
    """Nominal BIC degrees of freedom: K * S.

    Upper bound; active df (parameters not at a boundary) is typically smaller.
    Both should be reported when a refit result is available.
    """
    return max(int(num_clusters), 1) * int(data.num_regions)


def compute_bic_with_df(loglik: float, degrees_of_freedom: float, num_observations: float) -> float:
    return float(-2.0 * float(loglik) + float(degrees_of_freedom) * np.log(max(float(num_observations), 1.0)))


def compute_classic_bic(loglik: float, num_clusters: int, data: TumorData) -> float:
    num_observations = effective_bic_mutation_region_count(data)
    degrees_of_freedom = bic_degrees_of_freedom(num_clusters, data)
    return compute_bic_with_df(loglik, degrees_of_freedom, num_observations)


def compute_classic_bic_depth_n(loglik: float, num_clusters: int, data: TumorData) -> float:
    num_observations = effective_bic_depth_count(data)
    degrees_of_freedom = bic_degrees_of_freedom(num_clusters, data)
    return compute_bic_with_df(loglik, degrees_of_freedom, num_observations)


def compute_extended_bic(
    loglik: float,
    num_clusters: int,
    data: TumorData,
    bic_df_scale: float,
    bic_cluster_penalty: float,
) -> float:
    num_observations = effective_bic_mutation_region_count(data)
    cluster_count = max(int(num_clusters), 1)
    cp_degrees_of_freedom = bic_degrees_of_freedom(cluster_count, data)
    cluster_complexity = cluster_count
    return float(
        -2.0 * loglik
        + bic_df_scale * cp_degrees_of_freedom * np.log(num_observations)
        + bic_cluster_penalty * cluster_complexity * np.log(max(data.num_mutations, 2))
    )


__all__ = [
    "ADAPTIVE_LAMBDA_GRID_MODES",
    "LAMBDA_GRID_MODES",
    "LambdaBracket",
    "bic_degrees_of_freedom",
    "compute_bic_with_df",
    "compute_classic_bic",
    "compute_classic_bic_depth_n",
    "compute_extended_bic",
    "effective_bic_mutation_region_count",
    "effective_bic_depth_count",
    "is_adaptive_lambda_grid_mode",
]
