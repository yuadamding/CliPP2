from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..io.data import TumorData
from .fusion_solver import PairwiseFusionGraph, fit_observed_data_pairwise_fusion


@dataclass
class FitOptions:
    lambda_value: float
    outer_max_iter: int = 8
    inner_max_iter: int = 30
    tol: float = 1e-4
    major_prior: float = 0.5
    eps: float = 1e-6
    graph: PairwiseFusionGraph | None = None
    device: str = "auto"
    verbose: bool = False


@dataclass
class FitResult:
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
    history: list[float] = field(default_factory=list)
    bic: float | None = None
    classic_bic: float | None = None
    extended_bic: float | None = None
    selection_score_name: str | None = None


def fit_single_stage_em(
    data: TumorData,
    options: FitOptions,
    phi_start: np.ndarray | None = None,
) -> FitResult:
    artifacts = fit_observed_data_pairwise_fusion(
        data=data,
        lambda_value=float(options.lambda_value),
        major_prior=float(options.major_prior),
        eps=float(options.eps),
        outer_max_iter=max(int(options.outer_max_iter), 1),
        inner_max_iter=max(int(options.inner_max_iter), 16),
        tol=max(float(options.tol), 1e-6),
        phi_start=None if phi_start is None else np.asarray(phi_start, dtype=np.float32),
        graph=options.graph,
        device=str(options.device),
        verbose=bool(options.verbose),
    )
    return FitResult(
        phi=artifacts.phi.astype(np.float32, copy=False),
        phi_clustered=artifacts.phi_clustered.astype(np.float32, copy=False),
        cluster_labels=artifacts.cluster_labels.astype(np.int64, copy=False),
        cluster_centers=artifacts.cluster_centers.astype(np.float32, copy=False),
        gamma_major=artifacts.gamma_major.astype(np.float32, copy=False),
        major_probability=artifacts.major_probability.astype(np.float32, copy=False),
        major_call=artifacts.major_call.astype(bool, copy=False),
        multiplicity_call=artifacts.multiplicity_call.astype(np.float32, copy=False),
        multiplicity_estimated_mask=artifacts.multiplicity_estimated_mask.astype(bool, copy=False),
        loglik=float(artifacts.loglik),
        summary_loglik=float(artifacts.summary_loglik),
        penalized_objective=float(artifacts.penalized_objective),
        lambda_value=float(artifacts.lambda_value),
        n_clusters=int(artifacts.n_clusters),
        iterations=int(artifacts.iterations),
        converged=bool(artifacts.converged),
        device=str(artifacts.device),
        graph_name=str(artifacts.graph_name),
        history=list(artifacts.history),
    )


__all__ = [
    "FitOptions",
    "FitResult",
    "fit_single_stage_em",
]
