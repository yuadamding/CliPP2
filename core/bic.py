"""Single source of truth for model-selection score primitives.

This is a leaf module: it depends only on numpy and ``io.data.TumorData`` so that
both the ``core.fusion`` layer (partition refits / candidate BIC) and the
``model_selection`` / ``runners`` layers can import it *downward*. Keeping the
BIC arithmetic in one place avoids the correctness-drift risk of re-deriving
``-2*loglik + df*log(n)`` (and the observed-mutation_region count) in several modules.
"""

from __future__ import annotations

from math import fsum, lgamma
from typing import TYPE_CHECKING

import numpy as np

from ..io.data import TumorData

if TYPE_CHECKING:
    from ..model_selection.types import SelectionScore


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
    """Nominal BIC degrees of freedom under the clonal-anchored restriction.

    Model selection requires one mutation coordinate pinned at its feasible
    clonal center inside the raw pairwise-fusion objective. The immutable raw
    partition determines its clonal block, and the fixed-label refit preserves
    that block at its common feasible clonal center. Those pinned centers are
    constants, so a K-cluster model estimates (K - 1) * S center parameters.
    Upper bound; active df is typically smaller.
    """
    return max(int(num_clusters) - 1, 0) * int(data.num_regions)


def compute_bic_with_df(
    loglik: float, degrees_of_freedom: float, num_observations: float
) -> float:
    return float(
        -2.0 * float(loglik)
        + float(degrees_of_freedom) * np.log(max(float(num_observations), 1.0))
    )


def compute_classic_bic(loglik: float, num_clusters: int, data: TumorData) -> float:
    num_observations = effective_bic_mutation_region_count(data)
    degrees_of_freedom = bic_degrees_of_freedom(num_clusters, data)
    return compute_bic_with_df(loglik, degrees_of_freedom, num_observations)


def fixed_partition_bic(
    *,
    loglik: float,
    num_clusters: int,
    data: TumorData,
    anchor_mode: str,
    partition_signature: str,
    anchor_block_signature: str = "none",
    labels: np.ndarray | None = None,
    anchor_cluster: int | None = None,
) -> "SelectionScore":
    """Return the explicitly named BIC for one immutable partition refit."""

    normalized_anchor = str(anchor_mode).strip().lower()
    if normalized_anchor == "none":
        degrees_of_freedom = int(num_clusters) * int(data.num_regions)
        score_name = "fixed_partition_bic"
    elif normalized_anchor == "clonal_required":
        degrees_of_freedom = max(int(num_clusters) - 1, 0) * int(data.num_regions)
        score_name = "clonal_fixed_partition_bic"
    else:
        raise ValueError("anchor_mode must be either 'none' or 'clonal_required'.")
    if labels is not None:
        labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_array.size != int(data.num_mutations):
            raise ValueError("BIC labels must contain one value per mutation.")
        observed = _observed_positive_depth_mask(data)
        identifiable_df = 0
        for cluster in range(int(num_clusters)):
            if normalized_anchor == "clonal_required" and cluster == anchor_cluster:
                continue
            members = labels_array == int(cluster)
            identifiable_df += sum(
                bool(np.any(observed[members, region]))
                for region in range(int(data.num_regions))
            )
        degrees_of_freedom = int(identifiable_df)
    n_eff = effective_bic_mutation_region_count(data)
    penalty = float(degrees_of_freedom * np.log(max(int(n_eff), 1)))
    value = float(-2.0 * float(loglik) + penalty)
    # Imported lazily to keep this score-primitives module usable by the lower
    # fusion layer without introducing a module-import cycle.
    from ..model_selection.types import SelectionScore

    return SelectionScore(
        name=score_name,
        value=value,
        loglik=float(loglik),
        penalty=penalty,
        degrees_of_freedom=int(degrees_of_freedom),
        n_eff=int(n_eff),
        partition_signature=str(partition_signature),
        anchor_block_signature=str(anchor_block_signature),
    )


def compute_classic_bic_depth_n(
    loglik: float, num_clusters: int, data: TumorData
) -> float:
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
        raise ValueError(
            "cluster_sizes must contain finite positive integers."
        ) from exc
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


# Scale on the assignment-code deviance in compute_partition_icl. The full
# code cost (1.0) grows as ~2*M*H(cluster fractions) and systematically merges
# weakly separated subclones; 0.0 is plain BIC, which over-splits everything.
# 0.7 was selected by every fold of a 5-fold cross-validation over a 252-tumor
# stratified truth ladder (held-out exact-K 82.5% +/- 5.2% vs 76.2% +/- 3.4%
# at 1.0; K=1 detection unchanged at 100%, true K=3/K=4 recovery +8/+20 pts).
PARTITION_ICL_CODE_WEIGHT = 0.7


def compute_partition_icl(
    loglik: float,
    cluster_sizes: np.ndarray,
    data: TumorData,
    *,
    alpha: float = 1.0,
    code_weight: float = PARTITION_ICL_CODE_WEIGHT,
) -> float:
    """Classic center BIC plus a scaled integrated assignment-code deviance."""
    sizes = _validated_cluster_sizes(cluster_sizes)
    if int(np.sum(sizes)) != int(data.num_mutations):
        raise ValueError(
            "Partition cluster sizes must sum to the number of tumor mutations "
            f"({int(data.num_mutations)})."
        )
    weight = float(code_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("code_weight must be nonnegative and finite.")
    classic_bic = compute_classic_bic(float(loglik), int(sizes.size), data)
    log_partition_evidence = compute_unlabeled_dirichlet_partition_log_evidence(
        sizes,
        alpha=float(alpha),
    )
    return float(classic_bic - 2.0 * weight * log_partition_evidence)


__all__ = [
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
    "fixed_partition_bic",
]
