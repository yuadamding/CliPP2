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
    degree_bound: int = 1

    def clear_torch_cache(self) -> None:
        # Backward-compatible no-op. Device tensors are owned by SolverContext
        # or per-run TensorFusionGraph objects, not cached on this host graph.
        return None


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
    outer_projected_stationarity_residual: float
    outer_projected_stationarity_norm: float
    outer_stationarity_normalizer: float
    outer_smooth_gradient_norm: float
    outer_fusion_adjustment_norm: float
    outer_edge_subgradient_residual: float
    outer_dual_ball_residual: float
    outer_box_primal_violation: float
    outer_num_interior_coordinates: int
    outer_num_lower_active_coordinates: int
    outer_num_upper_active_coordinates: int
    outer_num_frozen_coordinates: int
    outer_box_residual: float
    fixed_objective_kkt_residual: float
    outer_kkt_certificate_status: str
    outer_kkt_dual_refined: bool
    outer_kkt_fused_edges: int
    outer_kkt_nonzero_edges: int
    outer_stationarity_residual_before_dual_refine: float
    outer_stationarity_residual_after_dual_refine: float
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


@dataclass(frozen=True, slots=True)
class TensorProblem:
    alt: torch.Tensor
    total: torch.Tensor
    nonalt: torch.Tensor
    phi_upper: torch.Tensor
    ambiguous: torch.Tensor
    b_minus: torch.Tensor
    b_plus: torch.Tensor
    b_fixed: torch.Tensor
    eps: float
    major_prior: float
    log_prior_minor: torch.Tensor
    log_prior_major: torch.Tensor


@dataclass(frozen=True, slots=True)
class TensorFusionGraph:
    edge_index: torch.Tensor
    weight: torch.Tensor
    degree: torch.Tensor
    num_nodes: int
    is_complete: bool
    is_uniform: bool
    name: str

    @property
    def edge_u(self) -> torch.Tensor:
        return self.edge_index[0]

    @property
    def edge_v(self) -> torch.Tensor:
        return self.edge_index[1]


@dataclass(frozen=True, slots=True)
class SolverContext:
    problem: TensorProblem
    graph: TensorFusionGraph
    graph_spec: PairwiseFusionGraph
    exact_pilot: torch.Tensor
    pooled_start: torch.Tensor
    scalar_well_starts: tuple[torch.Tensor, ...]
    lower: torch.Tensor
    upper: torch.Tensor
    runtime: TorchRuntime


@dataclass(slots=True)
class SolverState:
    phi: torch.Tensor
    dual: torch.Tensor | None
    split: torch.Tensor | None
    curvature: torch.Tensor | None
    previous_lambda: float


@dataclass(frozen=True, slots=True)
class ObjectiveTerms:
    fit: torch.Tensor
    penalty: torch.Tensor
    total: torch.Tensor
    gamma_major: torch.Tensor


@dataclass(frozen=True, slots=True)
class InnerDiagnostics:
    iterations: int
    kkt_residual: float
    primal_delta: float
    dual_delta: float
    converged: bool


@dataclass(frozen=True, slots=True)
class OuterDiagnostics:
    iterations: int
    objective_history: tuple[float, ...]
    stationarity_residual: float
    majorization_failures: int
    accepted_full_steps: int
    accepted_damped_steps: int
    converged: bool


@dataclass(frozen=True, slots=True)
class TorchFitResult:
    phi_raw: torch.Tensor
    gamma_major: torch.Tensor
    dual: torch.Tensor | None
    fit_loss: torch.Tensor
    fusion_penalty: torch.Tensor
    objective: torch.Tensor
    inner: InnerDiagnostics
    outer: OuterDiagnostics
    graph_name: str
