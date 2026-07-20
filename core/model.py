from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..io.data import TumorData
from .fusion.defaults import (
    DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
    DEFAULT_CERTIFICATE_MAX_ITER,
    DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS,
    DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    DEFAULT_DENSE_FALLBACK_POLICY,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_INNER_BACKEND,
    DEFAULT_WORKSET_ADD_BATCH,
    DEFAULT_WORKSET_MAX_BYTES,
    DEFAULT_WORKSET_MAX_EXPANSIONS,
)
from .fusion.solver import fit_observed_data_pairwise_fusion
from .fusion.types import (
    ExactFusionProvenance,
    PairwiseFusionGraph,
    SolverContext,
    SolverState,
)


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
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    summary_tol: float | None = 1e-4
    bic_partition_tol: float | None = 1e-4
    objective_shape: str = "unimodal"
    inner_backend: str = DEFAULT_INNER_BACKEND
    workset_max_bytes: int = DEFAULT_WORKSET_MAX_BYTES
    compressed_cache_max_bytes: int = DEFAULT_COMPRESSED_CACHE_MAX_BYTES
    dense_fallback_policy: str = DEFAULT_DENSE_FALLBACK_POLICY
    workset_add_batch: int = DEFAULT_WORKSET_ADD_BATCH
    workset_max_expansions: int = DEFAULT_WORKSET_MAX_EXPANSIONS
    certificate_max_iter: int = DEFAULT_CERTIFICATE_MAX_ITER
    certificate_refinement_rounds: int = DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS
    certificate_column_tol_scale: float = DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE
    allow_heuristic_structure_splits: bool = True
    materialize_full_dual: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class Problem:
    data: TumorData
    graph: PairwiseFusionGraph
    lambda_value: float
    major_prior: float = 0.5
    eps: float = 1e-6


@dataclass(frozen=True)
class SolverOptions:
    outer_max_iter: int = 8
    inner_max_iter: int = 30
    tol: float = 1e-4
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    compute_summary: bool = False
    summary_tol: float | None = 1e-4
    bic_partition_tol: float | None = 1e-4
    inner_backend: str = DEFAULT_INNER_BACKEND
    workset_max_bytes: int = DEFAULT_WORKSET_MAX_BYTES
    compressed_cache_max_bytes: int = DEFAULT_COMPRESSED_CACHE_MAX_BYTES
    dense_fallback_policy: str = DEFAULT_DENSE_FALLBACK_POLICY
    workset_add_batch: int = DEFAULT_WORKSET_ADD_BATCH
    workset_max_expansions: int = DEFAULT_WORKSET_MAX_EXPANSIONS
    certificate_max_iter: int = DEFAULT_CERTIFICATE_MAX_ITER
    certificate_refinement_rounds: int = DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS
    certificate_column_tol_scale: float = DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE
    allow_heuristic_structure_splits: bool = True
    materialize_full_dual: bool = False
    verbose: bool = False

    def to_fit_options(self, problem: Problem) -> FitOptions:
        return FitOptions(
            lambda_value=float(problem.lambda_value),
            outer_max_iter=int(self.outer_max_iter),
            inner_max_iter=int(self.inner_max_iter),
            tol=float(self.tol),
            major_prior=float(problem.major_prior),
            eps=float(problem.eps),
            graph=problem.graph,
            device=str(self.device),
            dtype=str(self.dtype),
            summary_tol=self.summary_tol,
            bic_partition_tol=self.bic_partition_tol,
            inner_backend=str(self.inner_backend),
            workset_max_bytes=int(self.workset_max_bytes),
            compressed_cache_max_bytes=int(self.compressed_cache_max_bytes),
            dense_fallback_policy=str(self.dense_fallback_policy),
            workset_add_batch=int(self.workset_add_batch),
            workset_max_expansions=int(self.workset_max_expansions),
            certificate_max_iter=int(self.certificate_max_iter),
            certificate_refinement_rounds=int(self.certificate_refinement_rounds),
            certificate_column_tol_scale=float(self.certificate_column_tol_scale),
            allow_heuristic_structure_splits=bool(
                self.allow_heuristic_structure_splits
            ),
            materialize_full_dual=bool(self.materialize_full_dual),
            verbose=bool(self.verbose),
        )


@dataclass(frozen=True)
class Estimate:
    phi: np.ndarray
    objective: float
    loglik: float
    lambda_value: float
    graph_name: str


@dataclass(frozen=True)
class Diagnostics:
    converged: bool
    outer_iterations: int
    inner_iterations: int
    objective_history: tuple[float, ...]
    fixed_objective_kkt_residual: float | None = None
    inner_kkt_residual: float | None = None
    failure_reason: str | None = None
    admm_iterations: int = 0
    inner_solver: str = "unknown"
    full_kkt_certified: bool = False
    certificate_scope: str = "unknown"
    certificate_gradient_scope: str = "unknown"


@dataclass(frozen=True)
class Summary:
    cluster_labels: np.ndarray | None = None
    cluster_centers: np.ndarray | None = None
    major_probability: np.ndarray | None = None
    multiplicity_call: np.ndarray | None = None


@dataclass
class FitResult:
    phi: np.ndarray
    phi_clustered: np.ndarray
    cluster_labels: np.ndarray
    cluster_centers: np.ndarray
    cluster_diameters: np.ndarray
    max_cluster_diameter: float
    cluster_diameter_exact: bool
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
    summary_available: bool
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
    stationarity_certified: bool = False  # True if KKT residual is below tolerance
    global_optimality_certified: bool = False
    global_optimality_basis: str = "none"
    number_of_starts: int = 1
    number_of_finite_starts: int = 1
    best_start_objective: float = float("nan")
    second_best_start_objective: float = float("nan")
    objective_spread_across_starts: float = float("nan")
    selected_start_objective_rank: int = 1
    history: list[float] = field(default_factory=list)
    solver_state: SolverState | None = None
    # ``iterations`` is retained as the outer-MM count for compatibility.
    # These fields report the actual accumulated inner-solver work and its
    # implementation identity.
    inner_iterations: int = 0
    admm_iterations: int = 0
    inner_solver: str = "unknown"
    inner_backend: str = "unknown"
    backend_iterations: int = 0
    quotient_iterations: int = 0
    workset_iterations: int = 0
    workset_expansions: int = 0
    streamed_edge_passes: int = 0
    dense_iterations: int = 0
    certificate_iterations: int = 0
    activity_passes: int = 0
    analytic_adjoint_passes: int = 0
    column_scan_passes: int = 0
    full_certificate_audit_passes: int = 0
    fallback_reason: str = ""
    exactness_provenance_version: int = 0
    estimator_role: str = "raw_fused_lambda_path"
    objective_faithful: bool = False
    objective_spec_hash: str = ""
    original_graph_hash: str = ""
    certificate_problem_hash: str = ""
    certificate_scope: str = "unknown"
    certificate_gradient_scope: str = "unknown"
    full_kkt_certified: bool = False
    full_kkt_certificate_status: str = "not_audited"
    full_kkt_tolerance: float = 0.0
    exactness_provenance: ExactFusionProvenance | None = None

    @property
    def estimate(self) -> Estimate:
        return Estimate(
            phi=self.phi,
            objective=float(self.penalized_objective),
            loglik=float(self.loglik),
            lambda_value=float(self.lambda_value),
            graph_name=str(self.graph_name),
        )

    @property
    def diagnostics(self) -> Diagnostics:
        return Diagnostics(
            converged=bool(self.converged),
            outer_iterations=int(self.iterations),
            inner_iterations=int(self.inner_iterations),
            objective_history=tuple(float(value) for value in self.history),
            fixed_objective_kkt_residual=float(self.fixed_objective_kkt_residual),
            inner_kkt_residual=float(self.inner_kkt_residual),
            failure_reason=str(self.failure_reason or ""),
            admm_iterations=int(self.admm_iterations),
            inner_solver=str(self.inner_solver),
            full_kkt_certified=bool(self.full_kkt_certified),
            certificate_scope=str(self.certificate_scope),
            certificate_gradient_scope=str(self.certificate_gradient_scope),
        )

    @property
    def summary(self) -> Summary | None:
        if not self.summary_available:
            return None
        if self.cluster_labels is None and self.cluster_centers is None:
            return None
        return Summary(
            cluster_labels=self.cluster_labels,
            cluster_centers=self.cluster_centers,
            major_probability=self.major_probability,
            multiplicity_call=self.multiplicity_call,
        )


def fit_fixed_objective(
    data: TumorData,
    options: FitOptions,
    phi_start: np.ndarray | torch.Tensor | None = None,
    exact_pilot: np.ndarray | torch.Tensor | None = None,
    pooled_start: np.ndarray | torch.Tensor | None = None,
    scalar_well_starts: list[np.ndarray | torch.Tensor] | None = None,
    start_mode: str = "full",
    runtime=None,
    torch_data=None,
    solver_context: SolverContext | None = None,
    solver_state: SolverState | None = None,
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
        phi_start=phi_start,
        graph=options.graph,
        adaptive_weight_gamma=float(options.adaptive_weight_gamma),
        adaptive_weight_floor=float(options.adaptive_weight_floor),
        adaptive_weight_baseline=float(options.adaptive_weight_baseline),
        exact_pilot=exact_pilot,
        pooled_start=pooled_start,
        scalar_well_starts=scalar_well_starts,
        start_mode=str(start_mode),
        device=str(options.device),
        dtype=str(options.dtype),
        summary_tol=options.summary_tol,
        objective_shape=str(options.objective_shape),
        inner_backend=str(options.inner_backend),
        workset_max_bytes=int(options.workset_max_bytes),
        compressed_cache_max_bytes=int(options.compressed_cache_max_bytes),
        dense_fallback_policy=str(options.dense_fallback_policy),
        workset_add_batch=int(options.workset_add_batch),
        workset_max_expansions=int(options.workset_max_expansions),
        certificate_max_iter=int(options.certificate_max_iter),
        certificate_refinement_rounds=int(options.certificate_refinement_rounds),
        certificate_column_tol_scale=float(options.certificate_column_tol_scale),
        allow_heuristic_structure_splits=bool(options.allow_heuristic_structure_splits),
        runtime=runtime,
        torch_data=torch_data,
        solver_context=solver_context,
        solver_state=solver_state,
        compute_summary=bool(compute_summary),
        verbose=bool(options.verbose),
    )
    provenance = artifacts.exactness_provenance
    return FitResult(
        phi=artifacts.phi,
        phi_clustered=artifacts.phi_clustered,
        cluster_labels=artifacts.cluster_labels.astype(np.int64, copy=False),
        cluster_centers=artifacts.cluster_centers,
        cluster_diameters=artifacts.cluster_diameters.astype(np.float64, copy=False),
        max_cluster_diameter=float(artifacts.max_cluster_diameter),
        cluster_diameter_exact=bool(artifacts.cluster_diameter_exact),
        gamma_major=artifacts.gamma_major,
        major_probability=artifacts.major_probability,
        major_call=artifacts.major_call.astype(bool, copy=False),
        multiplicity_call=artifacts.multiplicity_call,
        multiplicity_estimated_mask=artifacts.multiplicity_estimated_mask.astype(
            bool, copy=False
        ),
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
        summary_available=bool(compute_summary),
        inner_kkt_residual=float(artifacts.inner_kkt_residual),
        accepted_inner_kkt_residual=float(artifacts.accepted_inner_kkt_residual),
        last_attempted_inner_kkt_residual=float(
            artifacts.last_attempted_inner_kkt_residual
        ),
        best_attempted_inner_kkt_residual=float(
            artifacts.best_attempted_inner_kkt_residual
        ),
        last_attempted_objective_gap=float(artifacts.last_attempted_objective_gap),
        best_attempted_objective_gap=float(artifacts.best_attempted_objective_gap),
        last_attempted_surrogate_gap=float(artifacts.last_attempted_surrogate_gap),
        best_attempted_surrogate_gap=float(artifacts.best_attempted_surrogate_gap),
        last_attempted_inner_model_gap=float(artifacts.last_attempted_inner_model_gap),
        best_attempted_inner_model_gap=float(artifacts.best_attempted_inner_model_gap),
        last_attempted_em_envelope_gap=float(artifacts.last_attempted_em_envelope_gap),
        best_attempted_em_envelope_gap=float(artifacts.best_attempted_em_envelope_gap),
        outer_stationarity_residual=float(artifacts.outer_stationarity_residual),
        outer_projected_stationarity_residual=float(
            artifacts.outer_projected_stationarity_residual
        ),
        outer_projected_stationarity_norm=float(
            artifacts.outer_projected_stationarity_norm
        ),
        outer_stationarity_normalizer=float(artifacts.outer_stationarity_normalizer),
        outer_smooth_gradient_norm=float(artifacts.outer_smooth_gradient_norm),
        outer_fusion_adjustment_norm=float(artifacts.outer_fusion_adjustment_norm),
        outer_edge_subgradient_residual=float(
            artifacts.outer_edge_subgradient_residual
        ),
        outer_dual_ball_residual=float(artifacts.outer_dual_ball_residual),
        outer_box_primal_violation=float(artifacts.outer_box_primal_violation),
        outer_num_interior_coordinates=int(artifacts.outer_num_interior_coordinates),
        outer_num_lower_active_coordinates=int(
            artifacts.outer_num_lower_active_coordinates
        ),
        outer_num_upper_active_coordinates=int(
            artifacts.outer_num_upper_active_coordinates
        ),
        outer_num_frozen_coordinates=int(artifacts.outer_num_frozen_coordinates),
        outer_box_residual=float(artifacts.outer_box_residual),
        fixed_objective_kkt_residual=float(artifacts.fixed_objective_kkt_residual),
        outer_kkt_certificate_status=str(artifacts.outer_kkt_certificate_status),
        outer_kkt_dual_refined=bool(artifacts.outer_kkt_dual_refined),
        outer_kkt_fused_edges=int(artifacts.outer_kkt_fused_edges),
        outer_kkt_nonzero_edges=int(artifacts.outer_kkt_nonzero_edges),
        outer_stationarity_residual_before_dual_refine=float(
            artifacts.outer_stationarity_residual_before_dual_refine
        ),
        outer_stationarity_residual_after_dual_refine=float(
            artifacts.outer_stationarity_residual_after_dual_refine
        ),
        converged_inner=bool(artifacts.converged_inner),
        converged_outer=bool(artifacts.converged_outer),
        final_relative_objective_change=float(
            artifacts.final_relative_objective_change
        ),
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
        stationarity_certified=bool(
            getattr(artifacts, "stationarity_certified", artifacts.selection_eligible)
        ),
        global_optimality_certified=bool(
            getattr(artifacts, "global_optimality_certified", False)
        ),
        global_optimality_basis=str(
            getattr(artifacts, "global_optimality_basis", "none")
        ),
        number_of_starts=int(getattr(artifacts, "number_of_starts", 1)),
        number_of_finite_starts=int(getattr(artifacts, "number_of_finite_starts", 1)),
        best_start_objective=float(
            getattr(artifacts, "best_start_objective", artifacts.penalized_objective)
        ),
        second_best_start_objective=float(
            getattr(artifacts, "second_best_start_objective", float("nan"))
        ),
        objective_spread_across_starts=float(
            getattr(artifacts, "objective_spread_across_starts", float("nan"))
        ),
        selected_start_objective_rank=int(
            getattr(artifacts, "selected_start_objective_rank", 1)
        ),
        history=list(artifacts.history),
        solver_state=artifacts.solver_state,
        inner_iterations=int(getattr(artifacts, "inner_iterations", 0)),
        admm_iterations=int(getattr(artifacts, "admm_iterations", 0)),
        inner_solver=str(getattr(artifacts, "inner_solver", "unknown")),
        inner_backend=str(
            provenance.backend_name
            if provenance is not None
            else getattr(artifacts, "inner_solver", "unknown")
        ),
        backend_iterations=int(
            provenance.backend_iterations
            if provenance is not None
            else getattr(artifacts, "inner_iterations", 0)
        ),
        quotient_iterations=int(
            provenance.quotient_iterations if provenance is not None else 0
        ),
        workset_iterations=int(
            provenance.workset_iterations if provenance is not None else 0
        ),
        workset_expansions=int(
            provenance.workset_expansions if provenance is not None else 0
        ),
        streamed_edge_passes=int(
            provenance.streamed_edge_passes if provenance is not None else 0
        ),
        dense_iterations=int(
            provenance.dense_iterations if provenance is not None else 0
        ),
        certificate_iterations=int(
            provenance.certificate_iterations if provenance is not None else 0
        ),
        activity_passes=int(
            provenance.activity_passes if provenance is not None else 0
        ),
        analytic_adjoint_passes=int(
            provenance.analytic_adjoint_passes if provenance is not None else 0
        ),
        column_scan_passes=int(
            provenance.column_scan_passes if provenance is not None else 0
        ),
        full_certificate_audit_passes=int(
            provenance.full_certificate_audit_passes if provenance is not None else 0
        ),
        fallback_reason=str(
            provenance.fallback_reason if provenance is not None else ""
        ),
        exactness_provenance_version=int(
            provenance.schema_version if provenance is not None else 0
        ),
        estimator_role=str(
            provenance.estimator_role
            if provenance is not None
            else "raw_fused_lambda_path"
        ),
        objective_faithful=bool(
            provenance.objective_faithful if provenance is not None else False
        ),
        objective_spec_hash=str(
            provenance.objective_spec_hash if provenance is not None else ""
        ),
        original_graph_hash=str(
            provenance.original_graph_hash if provenance is not None else ""
        ),
        certificate_problem_hash=str(
            provenance.certificate_problem_hash if provenance is not None else ""
        ),
        certificate_scope=str(
            provenance.certificate_scope if provenance is not None else "unknown"
        ),
        certificate_gradient_scope=str(
            provenance.gradient_scope if provenance is not None else "unknown"
        ),
        full_kkt_certified=bool(
            provenance.full_kkt_certified if provenance is not None else False
        ),
        full_kkt_certificate_status=str(
            provenance.status if provenance is not None else "not_audited"
        ),
        full_kkt_tolerance=float(
            provenance.tolerance if provenance is not None else 0.0
        ),
        exactness_provenance=provenance,
    )


def fit(problem: Problem, options: SolverOptions | None = None) -> FitResult:
    solver_options = SolverOptions() if options is None else options
    return fit_fixed_objective(
        problem.data,
        solver_options.to_fit_options(problem),
        compute_summary=bool(solver_options.compute_summary),
    )


__all__ = [
    "Diagnostics",
    "Estimate",
    "FitOptions",
    "FitResult",
    "Problem",
    "SolverOptions",
    "Summary",
    "fit",
    "fit_fixed_objective",
]
