"""Backward-compatible re-export of the BIC primitives.

The implementations now live in :mod:`CliPP2.core.bic` (a leaf module so that
``core.fusion`` can import them without a ``core -> runners`` back-edge). This
module is kept so existing import paths
(``from CliPP2.runners.selection import effective_bic_mutation_region_count`` etc.) keep
working.
"""

from __future__ import annotations

from ..core.bic import (
    ADAPTIVE_LAMBDA_GRID_MODES,
    LAMBDA_GRID_MODES,
    PARTITION_GUIDED_LAMBDA_GRID_MODES,
    LambdaBracket,
    bic_degrees_of_freedom,
    compute_bic_with_df,
    compute_classic_bic,
    compute_classic_bic_depth_n,
    compute_extended_bic,
    effective_bic_mutation_region_count,
    effective_bic_depth_count,
    is_adaptive_lambda_grid_mode,
    is_partition_guided_lambda_grid_mode,
)

__all__ = [
    "ADAPTIVE_LAMBDA_GRID_MODES",
    "LAMBDA_GRID_MODES",
    "PARTITION_GUIDED_LAMBDA_GRID_MODES",
    "LambdaBracket",
    "bic_degrees_of_freedom",
    "compute_bic_with_df",
    "compute_classic_bic",
    "compute_classic_bic_depth_n",
    "compute_extended_bic",
    "effective_bic_mutation_region_count",
    "effective_bic_depth_count",
    "is_adaptive_lambda_grid_mode",
    "is_partition_guided_lambda_grid_mode",
]
