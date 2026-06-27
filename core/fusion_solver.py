"""Deprecated compatibility shim. Remove in v0.3."""

from .fusion.solver import fit_observed_data_pairwise_fusion as fit
from .fusion.types import PairwiseFusionGraph

__all__ = ["PairwiseFusionGraph", "fit"]
