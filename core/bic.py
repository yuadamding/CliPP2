"""Single source of truth for BIC primitives and lambda-grid mode helpers.

This is a leaf module: it depends only on numpy and ``io.data.TumorData`` so that
both the ``core.fusion`` layer (partition refits / candidate BIC) and the
``model_selection`` / ``runners`` layers can import it *downward*. Keeping the
BIC arithmetic in one place avoids the correctness-drift risk of re-deriving
``-2*loglik + df*log(n)`` (and the observed-mutation_region count) in several modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, lgamma

import numpy as np

from ..io.data import TumorData


PARTITION_GUIDED_LAMBDA_GRID_MODES = ("partition_guided_admm",)
ADAPTIVE_LAMBDA_GRID_MODES = ("adaptive_bic",)
LAMBDA_GRID_MODES = PARTITION_GUIDED_LAMBDA_GRID_MODES + ADAPTIVE_LAMBDA_GRID_MODES


@dataclass(frozen=True)
class LambdaBracket:
    lambda_min: float
    lambda_eq: float
    lambda_full: float
    anchors: list[float]
    diagnostics: dict[str, float]


def is_adaptive_lambda_grid_mode(mode: str) -> bool:
    return str(mode).strip().lower() in ADAPTIVE_LAMBDA_GRID_MODES


def is_partition_guided_lambda_grid_mode(mode: str) -> bool:
    return str(mode).strip().lower() in PARTITION_GUIDED_LAMBDA_GRID_MODES


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


def cluster_sizes_from_labels(labels: np.ndarray) -> np.ndarray:
    """Return occupied-cluster sizes, invariant to the numeric label names."""
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("Partition labels must be a one-dimensional array.")
    if values.size == 0:
        raise ValueError("A partition must contain at least one mutation label.")
    if not np.issubdtype(values.dtype, np.integer):
        numeric = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.round(numeric)):
            raise ValueError("Partition labels must be finite integers.")
        values = numeric.astype(np.int64)
    _, counts = np.unique(values, return_counts=True)
    return counts.astype(np.int64, copy=False)


def _validated_cluster_sizes(cluster_sizes: np.ndarray) -> np.ndarray:
    raw = np.asarray(cluster_sizes)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("cluster_sizes must be a non-empty one-dimensional array.")
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("cluster_sizes must contain finite positive integers.") from exc
    if (
        not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
        or not np.all(values == np.round(values))
    ):
        raise ValueError("cluster_sizes must contain finite positive integers.")
    return values.astype(np.int64)


def compute_unlabeled_dirichlet_partition_log_evidence(
    cluster_sizes: np.ndarray,
    *,
    alpha: float = 1.0,
) -> float:
    """Integrated log probability of an unlabeled occupied partition.

    Mixing proportions have a symmetric ``Dirichlet(alpha, ..., alpha)``
    prior over the ``K`` occupied clusters.  Integrating them out gives the
    probability of a particular labeled allocation.  Adding ``log(K!)``
    converts that allocation probability to the corresponding unlabeled set
    partition, so arbitrary permutations of cluster names are not charged as
    distinct models.
    """
    sizes = _validated_cluster_sizes(cluster_sizes)
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("Dirichlet alpha must be positive and finite.")
    num_clusters = int(sizes.size)
    num_mutations = int(np.sum(sizes))
    log_labeled_evidence = (
        lgamma(float(num_clusters) * alpha)
        - lgamma(float(num_mutations) + float(num_clusters) * alpha)
        + fsum(lgamma(float(size) + alpha) - lgamma(alpha) for size in sizes)
    )
    return float(log_labeled_evidence + lgamma(float(num_clusters) + 1.0))


def compute_partition_icl(
    loglik: float,
    cluster_sizes: np.ndarray,
    data: TumorData,
    *,
    alpha: float = 1.0,
) -> float:
    """Classic center BIC plus an integrated assignment-code deviance."""
    sizes = _validated_cluster_sizes(cluster_sizes)
    if int(np.sum(sizes)) != int(data.num_mutations):
        raise ValueError(
            "Partition cluster sizes must sum to the number of tumor mutations "
            f"({int(data.num_mutations)})."
        )
    classic_bic = compute_classic_bic(float(loglik), int(sizes.size), data)
    log_partition_evidence = compute_unlabeled_dirichlet_partition_log_evidence(
        sizes,
        alpha=float(alpha),
    )
    return float(classic_bic - 2.0 * log_partition_evidence)


__all__ = [
    "ADAPTIVE_LAMBDA_GRID_MODES",
    "LAMBDA_GRID_MODES",
    "PARTITION_GUIDED_LAMBDA_GRID_MODES",
    "LambdaBracket",
    "bic_degrees_of_freedom",
    "cluster_sizes_from_labels",
    "compute_bic_with_df",
    "compute_classic_bic",
    "compute_classic_bic_depth_n",
    "compute_extended_bic",
    "compute_partition_icl",
    "compute_unlabeled_dirichlet_partition_log_evidence",
    "effective_bic_mutation_region_count",
    "effective_bic_depth_count",
    "is_adaptive_lambda_grid_mode",
    "is_partition_guided_lambda_grid_mode",
]
