from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class PairwiseFusionGraph:
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_w: np.ndarray
    name: str = "complete_uniform"
    degree_bound: int = 1
    torch_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def clear_torch_cache(self) -> None:
        self.torch_cache.clear()


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
    dtype: str
    graph_name: str
    summary_tol: float
    history: list[float]
    inner_kkt_residual: float
    accepted_inner_kkt_residual: float
    last_attempted_inner_kkt_residual: float
    best_attempted_inner_kkt_residual: float
    last_attempted_objective_gap: float
    best_attempted_objective_gap: float
    last_attempted_surrogate_gap: float
    best_attempted_surrogate_gap: float
    last_attempted_inner_model_gap: float
    best_attempted_inner_model_gap: float
    last_attempted_em_envelope_gap: float
    best_attempted_em_envelope_gap: float
    outer_stationarity_residual: float
    outer_edge_subgradient_residual: float
    outer_dual_ball_residual: float
    outer_box_residual: float
    fixed_objective_kkt_residual: float
    converged_inner: bool
    converged_outer: bool
    final_relative_objective_change: float
    final_step_residual: float
    accepted_outer_steps: int
    accepted_full_steps: int
    accepted_damped_steps: int
    attempted_outer_steps: int
    failed_majorization_checks: int
    failed_inner_model_checks: int
    failed_em_envelope_checks: int
    failed_descent_checks: int
    failed_nonfinite_checks: int
    mm_consistency_violations: int
    accepted_step_type: str
    last_reject_reason: str
    failure_reason: str
    selection_eligible: bool


@dataclass(frozen=True)
class TorchRuntime:
    device: torch.device
    device_name: str
    dtype: torch.dtype
