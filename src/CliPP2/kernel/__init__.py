"""CliPP2-owned regional kernels, with algorithmic ancestry in CliPP1.5."""

from ..config import SELECTION_CRITERION

KERNEL_VERSION = "clipp2_regional_forest_v2"
ALGORITHM = "regional_A_initialized_marginal_mixture_v1"
INITIALIZER_ALGORITHM = "regional_recurring_general_search_v8"
SCORE_DEFINITION = SELECTION_CRITERION
