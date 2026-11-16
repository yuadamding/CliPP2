"""CliPP2-owned regional kernels, with algorithmic ancestry in CliPP1.5."""

from ..config import SELECTION_CRITERION

KERNEL_VERSION = "clipp2_regional_forest_v2"
ALGORITHM = "regional_tree_free_marginal_mixture_v1"
LEGACY_SEARCH_ALGORITHMS = ("regional_recurring_general_search_v8", "regional_bounded_search_v7")
COMPATIBLE_SEARCH_ALGORITHMS = (ALGORITHM, *LEGACY_SEARCH_ALGORITHMS)
SCORE_DEFINITION = SELECTION_CRITERION
