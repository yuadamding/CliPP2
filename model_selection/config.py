from __future__ import annotations

SELECTION_SCORE_NAMES = ("marginal_bic", "partition_icl", "bic", "extended_bic")
# Classification-EM stabilization budget per candidate before the marginal
# mixture log-likelihood is evaluated (cluster-emptying E-steps are rejected,
# so the candidate's K is preserved).
MARGINAL_BIC_MAX_CEM_ITERATIONS = 6
# Ward-ladder candidates built on the best fused phi after the lambda path.
# Path partitions inherit fusion's membership commitments; the ladder supplies
# independently drawn memberships per K, which the marginal criterion needs
# (its scores are far more sensitive to membership quality than ICL's).
# Override per run with --kmax (0 disables the ladder).
FINAL_PHI_WARD_LADDER_KMAX = 30
PARTITION_ICL_DIRICHLET_ALPHA = 1.0

PARTITION_GUIDED_ADMM_MAX_UNIQUE_LAMBDAS = 12
PARTITION_GUIDED_ADMM_MAX_SOLVER_RETRIES_PER_LAMBDA = 4
# A mild superlinear degree correction keeps the adaptive guide traversable
# while reducing shrinkage transmitted between clearly separated groups.
PARTITION_GUIDED_ADAPTIVE_NOISE_DEGREE_EXPONENT = 1.05
LIKELIHOOD_PARTITION_K_MAX = 50
LIKELIHOOD_PARTITION_K_ANCHORS = (*range(1, 16), 20, 25, 30, 40, 50)
LIKELIHOOD_PARTITION_MAX_CANDIDATES_PER_K = 5
LIKELIHOOD_PARTITION_CEM_MAX_ITER = 8
LIKELIHOOD_PARTITION_REFIT_MAX_ITER = 32
LIKELIHOOD_PARTITION_SENTINEL_LAMBDA = -1.0
