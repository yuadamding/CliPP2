"""Compact facade for the fixed pairwise-fusion estimator internals."""

from .solver import fit_observed_data_pairwise_fusion as fit
from .types import PairwiseFusionGraph

__all__ = ["PairwiseFusionGraph", "fit"]
