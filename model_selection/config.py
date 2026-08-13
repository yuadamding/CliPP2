from __future__ import annotations

SELECTION_SCORE_NAMES = (
    "fixed_partition_dirichlet_score",
    "fixed_partition_bic",
)
# A mild superlinear degree correction keeps the adaptive guide traversable
# while reducing shrinkage transmitted between clearly separated groups.
PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT = 1.05
LIKELIHOOD_PARTITION_K_MAX = 50
