"""CliPP2-owned numerical kernels; no sibling repository is needed at runtime.

Algorithm ancestry and license attribution are recorded in ../NOTICE.
"""

from ..config import MAX_CLUSTERS as MAX_CLUSTERS
from .objectives import CLASSIFICATION_SCORE_DEFINITION, score_definition_for as score_definition_for

KERNEL_VERSION = "clipp2_occupied_clonal_chain_tree_abi3_classification_v3"
ALGORITHM = "occupied_clonal_explicit_objective_five_family_cumulative_search_v5"
SCORE_DEFINITION = CLASSIFICATION_SCORE_DEFINITION
LEGACY_SCORING_VERSION = "occupied_clonal_multiplicity_2K_minus_2_v2"
SCORING_VERSION = CLASSIFICATION_SCORE_DEFINITION
CANDIDATE_SEARCH_VERSION = "clipp2_five_family_cumulative_v2"
