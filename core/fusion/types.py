from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PairwiseFusionGraph:
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_w: np.ndarray
    name: str = "complete_uniform"


@dataclass(frozen=True)
class FusionFitArtifacts:
    phi: np.ndarray
    phi_clustered: np.ndarray
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    gamma_major: np.ndarray
    major_probability: np.ndarray
    major_call: np.ndarray
    multiplicity_call: np.ndarray
    multiplicity_estimated_mask: np.ndarray
    loglik: float
    summary_loglik: float
    penalized_objective: float
    lambda_value: float
    n_clusters: int
    iterations: int
    converged: bool
    device: str
    graph_name: str
    history: list[float]


@dataclass(frozen=True)
class TorchRuntime:
    device: torch.device
    device_name: str
    dtype: torch.dtype
