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
    adaptive_weight_gamma: float = 1.0
    adaptive_weight_floor: float = 1e-6
    adaptive_weight_baseline: float = 1.0
    device: str = "cuda"
    dtype: str = "float64"
    summary_tol: float | None = 1e-4
    bic_partition_tol: float | None = 1e-4
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
    dtype: str
    graph_name: str
    summary_tol: float
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
    history: list[float] = field(default_factory=list)
    bic: float | None = None
    classic_bic: float | None = None
    extended_bic: float | None = None
    classic_bic_depth_n: float | None = None
    bic_loglik: float | None = None
    bic_loglik_source: str | None = None
    bic_df: float | None = None
    bic_active_df: float | None = None
    bic_n_eff: float | None = None
    bic_depth_n_eff: float | None = None
    bic_partition_tol: float | None = None
    bic_refit_boundary_count: int | None = None
    bic_refit_converged: bool | None = None
    bic_refit_phi: np.ndarray | None = None
    bic_refit_cluster_centers: np.ndarray | None = None
    bic_partition_labels: np.ndarray | None = None
    selection_score_name: str | None = None


def fit_single_stage_em(
    data: TumorData,
    options: FitOptions,
    phi_start: np.ndarray | None = None,
    exact_pilot: np.ndarray | None = None,
    pooled_start: np.ndarray | None = None,
    scalar_well_starts: list[np.ndarray] | None = None,
    start_mode: str = "full",
    runtime=None,
    torch_data=None,
    compute_summary: bool = True,
) -> FitResult:
    artifacts = fit_observed_data_pairwise_fusion(
        data=data,
        lambda_value=float(options.lambda_value),
        major_prior=float(options.major_prior),
        eps=float(options.eps),
        outer_max_iter=max(int(options.outer_max_iter), 1),
        inner_max_iter=max(int(options.inner_max_iter), 16),
        tol=float(options.tol),
        phi_start=None if phi_start is None else np.asarray(phi_start),
        graph=options.graph,
        adaptive_weight_gamma=float(options.adaptive_weight_gamma),
        adaptive_weight_floor=float(options.adaptive_weight_floor),
        adaptive_weight_baseline=float(options.adaptive_weight_baseline),
        exact_pilot=None if exact_pilot is None else np.asarray(exact_pilot),
        pooled_start=None if pooled_start is None else np.asarray(pooled_start),
        scalar_well_starts=None
        if scalar_well_starts is None
        else [np.asarray(start) for start in scalar_well_starts],
        start_mode=str(start_mode),
        device=str(options.device),
        dtype=str(options.dtype),
        summary_tol=options.summary_tol,
        runtime=runtime,
        torch_data=torch_data,
        compute_summary=bool(compute_summary),
        verbose=bool(options.verbose),
    )
    return FitResult(
        phi=artifacts.phi,
        phi_clustered=artifacts.phi_clustered,
        cluster_labels=artifacts.cluster_labels.astype(np.int64, copy=False),
        cluster_centers=artifacts.cluster_centers,
        gamma_major=artifacts.gamma_major,
        major_probability=artifacts.major_probability,
        major_call=artifacts.major_call.astype(bool, copy=False),
        multiplicity_call=artifacts.multiplicity_call,
        multiplicity_estimated_mask=artifacts.multiplicity_estimated_mask.astype(bool, copy=False),
        loglik=float(artifacts.loglik),
        summary_loglik=float(artifacts.summary_loglik),
        penalized_objective=float(artifacts.penalized_objective),
        lambda_value=float(artifacts.lambda_value),
        n_clusters=int(artifacts.n_clusters),
        iterations=int(artifacts.iterations),
        converged=bool(artifacts.converged),
        device=str(artifacts.device),
        dtype=str(artifacts.dtype),
        graph_name=str(artifacts.graph_name),
        summary_tol=float(artifacts.summary_tol),
        inner_kkt_residual=float(artifacts.inner_kkt_residual),
        accepted_inner_kkt_residual=float(artifacts.accepted_inner_kkt_residual),
        last_attempted_inner_kkt_residual=float(artifacts.last_attempted_inner_kkt_residual),
        best_attempted_inner_kkt_residual=float(artifacts.best_attempted_inner_kkt_residual),
        last_attempted_objective_gap=float(artifacts.last_attempted_objective_gap),
        best_attempted_objective_gap=float(artifacts.best_attempted_objective_gap),
        last_attempted_surrogate_gap=float(artifacts.last_attempted_surrogate_gap),
        best_attempted_surrogate_gap=float(artifacts.best_attempted_surrogate_gap),
        last_attempted_inner_model_gap=float(artifacts.last_attempted_inner_model_gap),
        best_attempted_inner_model_gap=float(artifacts.best_attempted_inner_model_gap),
        last_attempted_em_envelope_gap=float(artifacts.last_attempted_em_envelope_gap),
        best_attempted_em_envelope_gap=float(artifacts.best_attempted_em_envelope_gap),
        outer_stationarity_residual=float(artifacts.outer_stationarity_residual),
        outer_projected_stationarity_residual=float(artifacts.outer_projected_stationarity_residual),
        outer_projected_stationarity_norm=float(artifacts.outer_projected_stationarity_norm),
        outer_stationarity_normalizer=float(artifacts.outer_stationarity_normalizer),
        outer_smooth_gradient_norm=float(artifacts.outer_smooth_gradient_norm),
        outer_fusion_adjustment_norm=float(artifacts.outer_fusion_adjustment_norm),
        outer_edge_subgradient_residual=float(artifacts.outer_edge_subgradient_residual),
        outer_dual_ball_residual=float(artifacts.outer_dual_ball_residual),
        outer_box_primal_violation=float(artifacts.outer_box_primal_violation),
        outer_num_interior_coordinates=int(artifacts.outer_num_interior_coordinates),
        outer_num_lower_active_coordinates=int(artifacts.outer_num_lower_active_coordinates),
        outer_num_upper_active_coordinates=int(artifacts.outer_num_upper_active_coordinates),
        outer_num_frozen_coordinates=int(artifacts.outer_num_frozen_coordinates),
        outer_box_residual=float(artifacts.outer_box_residual),
        fixed_objective_kkt_residual=float(artifacts.fixed_objective_kkt_residual),
        outer_kkt_certificate_status=str(artifacts.outer_kkt_certificate_status),
        outer_kkt_dual_refined=bool(artifacts.outer_kkt_dual_refined),
        outer_kkt_fused_edges=int(artifacts.outer_kkt_fused_edges),
        outer_kkt_nonzero_edges=int(artifacts.outer_kkt_nonzero_edges),
        outer_stationarity_residual_before_dual_refine=float(artifacts.outer_stationarity_residual_before_dual_refine),
        outer_stationarity_residual_after_dual_refine=float(artifacts.outer_stationarity_residual_after_dual_refine),
        converged_inner=bool(artifacts.converged_inner),
        converged_outer=bool(artifacts.converged_outer),
        final_relative_objective_change=float(artifacts.final_relative_objective_change),
        final_step_residual=float(artifacts.final_step_residual),
        accepted_outer_steps=int(artifacts.accepted_outer_steps),
        accepted_full_steps=int(artifacts.accepted_full_steps),
        accepted_damped_steps=int(artifacts.accepted_damped_steps),
        attempted_outer_steps=int(artifacts.attempted_outer_steps),
        failed_majorization_checks=int(artifacts.failed_majorization_checks),
        failed_inner_model_checks=int(artifacts.failed_inner_model_checks),
        failed_em_envelope_checks=int(artifacts.failed_em_envelope_checks),
        failed_descent_checks=int(artifacts.failed_descent_checks),
        failed_nonfinite_checks=int(artifacts.failed_nonfinite_checks),
        mm_consistency_violations=int(artifacts.mm_consistency_violations),
        accepted_step_type=str(artifacts.accepted_step_type),
        last_reject_reason=str(artifacts.last_reject_reason),
        failure_reason=str(artifacts.failure_reason),
        selection_eligible=bool(artifacts.selection_eligible),
        history=list(artifacts.history),
    )


__all__ = [
    "FitOptions",
    "FitResult",
    "fit_single_stage_em",
]
