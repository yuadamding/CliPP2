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
    """Nominal center degrees of freedom for an unanchored K-block model."""

    return max(int(num_clusters), 0) * int(data.num_regions)


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
    partition_signature: str,
    labels: np.ndarray | None = None,
    loglik_uncertainty: float = 0.0,
    selection_contract_id: str = "hybrid-ward-cem-v1",
) -> "SelectionScore":
    """Return the explicitly named BIC for one immutable partition refit."""

    degrees_of_freedom = int(num_clusters) * int(data.num_regions)
    if labels is not None:
        labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
        if labels_array.size != int(data.num_mutations):
            raise ValueError("BIC labels must contain one value per mutation.")
    n_eff = effective_bic_mutation_region_count(data)
    penalty = float(degrees_of_freedom * np.log(max(int(n_eff), 1)))
    value = float(-2.0 * float(loglik) + penalty)
    likelihood_uncertainty = max(float(loglik_uncertainty), 0.0)
    arithmetic_uncertainty = 16.0 * np.finfo(np.float64).eps * (1.0 + abs(value))
    # Imported lazily to keep this score-primitives module usable by the lower
    # fusion layer without introducing a module-import cycle.
    from ..model_selection.types import SelectionScore

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
        selection_contract_id=str(selection_contract_id),
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


DIRICHLET_EXACT_PARTITION_MODEL_ID = "symmetric_dirichlet_integrated_exact_partition_v1"
# Preserve the production allocation-code weight selected before the clonal
# anchor was removed.  Removing a fixed CCF-one block changes the symmetry and
# center degrees of freedom; it must not silently retune the independent
# assignment-code contribution.
PARTITION_DIRICHLET_SCORE_WEIGHT = 0.7
PARTITION_DIRICHLET_ALPHA = 1.0


def _dirichlet_exact_partition_log_mass_and_uncertainty(
    cluster_sizes: np.ndarray,
    *,
    alpha: float,
) -> tuple[float, float]:
    """Return exact-partition log mass and a conservative arithmetic bound."""

    sizes = _validated_cluster_sizes(cluster_sizes)
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("Dirichlet alpha must be positive and finite.")
    num_clusters = int(sizes.size)
    num_mutations = int(np.sum(sizes))
    terms = [
        lgamma(float(num_clusters) * alpha),
        -lgamma(float(num_mutations) + float(num_clusters) * alpha),
    ]
    for size in sizes:
        terms.extend((lgamma(float(size) + alpha), -lgamma(alpha)))
    # Every block is exchangeable in the unanchored model, so integrate over
    # all K! equivalent component labelings.
    terms.append(lgamma(float(num_clusters) + 1.0))
    value = float(fsum(terms))
    magnitude = float(fsum(abs(term) for term in terms))
    arithmetic_uncertainty = float(
        32.0 * np.finfo(np.float64).eps * float(len(terms) + 1) * (1.0 + magnitude)
    )
    return value, arithmetic_uncertainty


def compute_dirichlet_exact_partition_log_mass(
    cluster_sizes: np.ndarray,
    *,
    alpha: float = PARTITION_DIRICHLET_ALPHA,
) -> float:
    """Integrated log mass of one exact occupied set partition.

    A symmetric ``Dirichlet(alpha, ..., alpha)`` prior is placed on the mixing
    proportions of the ``K`` occupied blocks. Integrating those proportions
    gives the probability of a particular labeled allocation. The result is
    then summed over all ``K!`` equivalent block labelings.

    With ``alpha=1`` and no distinguished block this is

    ``log(K! (K-1)! prod_k n_k! / (n+K-1)!)``.

    This is an exact-allocation prior mass, not a posterior classification-
    entropy term. Consequently, for fixed ``n`` and ``K`` it assigns more mass
    to imbalanced partitions than to balanced partitions.
    """

    value, _ = _dirichlet_exact_partition_log_mass_and_uncertainty(
        cluster_sizes,
        alpha=float(alpha),
    )
    return value


def fixed_partition_dirichlet_score(
    *,
    loglik: float,
    num_clusters: int,
    data: TumorData,
    partition_signature: str,
    labels: np.ndarray,
    loglik_uncertainty: float = 0.0,
    alpha: float = PARTITION_DIRICHLET_ALPHA,
    code_weight: float = PARTITION_DIRICHLET_SCORE_WEIGHT,
    selection_contract_id: str = "hybrid-ward-cem-v1",
) -> "SelectionScore":
    """Return BIC plus a Dirichlet-integrated exact-partition deviance.

    The likelihood and center degrees of freedom are exactly those of
    :func:`fixed_partition_bic`. The additional term is invariant to arbitrary
    names of exchangeable blocks and is scaled by ``code_weight`` (0.7 by
    default). All K blocks are exchangeable. This is
    deliberately named a Dirichlet exact-partition score rather than
    posterior-entropy ICL.
    """

    labels_array = np.asarray(labels, dtype=np.int64).reshape(-1)
    sizes = cluster_sizes_from_labels(labels_array)
    if int(sizes.size) != int(num_clusters):
        raise ValueError(
            "Dirichlet-score cluster count does not match the partition labels."
        )
    if int(np.sum(sizes)) != int(data.num_mutations):
        raise ValueError("Dirichlet-score labels must contain one value per mutation.")
    alpha = float(alpha)
    weight = float(code_weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("code_weight must be nonnegative and finite.")
    base = fixed_partition_bic(
        loglik=loglik,
        num_clusters=num_clusters,
        data=data,
        partition_signature=partition_signature,
        labels=labels_array,
        loglik_uncertainty=loglik_uncertainty,
        selection_contract_id=str(selection_contract_id),
    )
    log_evidence, log_evidence_uncertainty = (
        _dirichlet_exact_partition_log_mass_and_uncertainty(
            sizes,
            alpha=alpha,
        )
    )
    assignment_penalty = float(-2.0 * weight * log_evidence)
    assignment_arithmetic_uncertainty = float(
        2.0 * abs(weight) * log_evidence_uncertainty
    )
    score_value = float(base.value + assignment_penalty)
    addition_uncertainty = float(
        16.0
        * np.finfo(np.float64).eps
        * (1.0 + abs(float(base.value)) + abs(assignment_penalty))
    )
    from ..model_selection.types import SelectionScore

    return SelectionScore(
        name="fixed_partition_dirichlet_score",
        value=score_value,
        loglik=float(base.loglik),
        penalty=float(base.penalty + assignment_penalty),
        degrees_of_freedom=int(base.degrees_of_freedom),
        n_eff=int(base.n_eff),
        partition_signature=str(base.partition_signature),
        numerical_uncertainty=float(
            base.numerical_uncertainty
            + assignment_arithmetic_uncertainty
            + addition_uncertainty
        ),
        assignment_log_evidence=float(log_evidence),
        assignment_code_weight=weight,
        assignment_penalty=assignment_penalty,
        assignment_dirichlet_alpha=alpha,
        assignment_model_id=DIRICHLET_EXACT_PARTITION_MODEL_ID,
        assignment_symmetry_mode="all_blocks_exchangeable",
        assignment_arithmetic_uncertainty=assignment_arithmetic_uncertainty,
        selection_contract_id=str(selection_contract_id),
    )


def compute_partition_dirichlet_score(
    loglik: float,
    cluster_sizes: np.ndarray,
    data: TumorData,
    *,
    alpha: float = PARTITION_DIRICHLET_ALPHA,
    code_weight: float = PARTITION_DIRICHLET_SCORE_WEIGHT,
) -> float:
    """Unanchored center BIC plus exact-partition Dirichlet deviance."""
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
    log_partition_evidence = compute_dirichlet_exact_partition_log_mass(
        sizes,
        alpha=float(alpha),
    )
    return float(classic_bic - 2.0 * weight * log_partition_evidence)


__all__ = [
    "bic_degrees_of_freedom",
    "cluster_sizes_from_labels",
    "compute_bic_with_df",
    "compute_classic_bic",
    "compute_dirichlet_exact_partition_log_mass",
    "compute_partition_dirichlet_score",
    "DIRICHLET_EXACT_PARTITION_MODEL_ID",
    "effective_bic_mutation_region_count",
    "effective_bic_depth_count",
    "fixed_partition_bic",
    "fixed_partition_dirichlet_score",
    "PARTITION_DIRICHLET_ALPHA",
    "PARTITION_DIRICHLET_SCORE_WEIGHT",
]
