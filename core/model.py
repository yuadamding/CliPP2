from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch

from ..io.data import TumorData
from .fusion.defaults import (
    DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
    DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    DEFAULT_DENSE_FALLBACK_POLICY,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_OPTIMIZATION_TOLERANCE,
    DEFAULT_WORKSET_ADD_BATCH,
    DEFAULT_WORKSET_MAX_BYTES,
    DEFAULT_WORKSET_MAX_EXPANSIONS,
)
from .fusion.profiles import (
    DEFAULT_COMPUTATION_PROFILE,
    get_computation_profile,
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
    outer_max_iter: int = 6
    inner_max_iter: int = 25
    tol: float = DEFAULT_OPTIMIZATION_TOLERANCE
    major_prior: float = 0.5
    eps: float = 1e-6
    graph: PairwiseFusionGraph | None = None
    adaptive_weight_gamma: float = 1.0
    adaptive_weight_floor: float = 1e-6
    adaptive_weight_baseline: float = 1.0
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    summary_tol: float | None = 2e-4
    selection_score: str = "fixed_partition_dirichlet_score"
    selection_partition_tol: float = 2e-4
    selection_refit_tol: float = 1e-5
    selection_refit_max_iter: int = 64
    selection_contract: str = "hybrid-ward-cem-v1"
    selection_dirichlet_alpha: float = 1.0
    selection_dirichlet_code_weight: float = 0.7
    objective_shape: str = "unimodal"
    workset_max_bytes: int = DEFAULT_WORKSET_MAX_BYTES
    compressed_cache_max_bytes: int = DEFAULT_COMPRESSED_CACHE_MAX_BYTES
    dense_fallback_policy: str = DEFAULT_DENSE_FALLBACK_POLICY
    workset_add_batch: int = DEFAULT_WORKSET_ADD_BATCH
    workset_max_expansions: int = DEFAULT_WORKSET_MAX_EXPANSIONS
    certificate_max_iter: int = 128
    certificate_refinement_rounds: int = 1
    certificate_column_tol_scale: float = DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE
    verbose: bool = False
    computation_profile: str = DEFAULT_COMPUTATION_PROFILE

    def __post_init__(self) -> None:
        from ..model_selection.contracts import get_selection_contract

        profile = get_computation_profile(self.computation_profile)
        self.computation_profile = profile.name
        contract = get_selection_contract(self.selection_contract)
        self.selection_contract = contract.contract_id
        self.selection_dirichlet_alpha = float(
            contract.partition_config.classification_alpha
        )
        self.selection_dirichlet_code_weight = float(
            contract.partition_config.classification_code_weight
        )
        if contract.force_float64:
            self.dtype = "float64"
        score = str(self.selection_score).strip().lower().replace("-", "_")
        if score not in {
            "fixed_partition_bic",
            "fixed_partition_dirichlet_score",
        }:
            raise ValueError("Unknown fixed-partition selection score.")
        self.selection_score = score
        finite_positive = {
            "tol": self.tol,
            "eps": self.eps,
            "selection_partition_tol": self.selection_partition_tol,
            "selection_refit_tol": self.selection_refit_tol,
            "certificate_column_tol_scale": self.certificate_column_tol_scale,
            "selection_dirichlet_alpha": self.selection_dirichlet_alpha,
        }
        for name, value in finite_positive.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if not np.isfinite(float(self.lambda_value)) or float(self.lambda_value) < 0.0:
            raise ValueError("lambda_value must be finite and nonnegative.")
        if not 0.0 < float(self.major_prior) < 1.0:
            raise ValueError("major_prior must lie strictly in (0, 1).")
        if int(self.selection_refit_max_iter) < 1:
            raise ValueError("selection_refit_max_iter must be positive.")
        if (
            not np.isfinite(float(self.selection_dirichlet_code_weight))
            or float(self.selection_dirichlet_code_weight) < 0.0
        ):
            raise ValueError(
                "selection_dirichlet_code_weight must be nonnegative and finite."
            )


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
    base_fusion_objective_hash: str = ""
    original_graph_hash: str = ""
    certificate_problem_hash: str = ""
    certificate_scope: str = "unknown"
    certificate_gradient_scope: str = "unknown"
    full_kkt_certified: bool = False
    full_kkt_certificate_status: str = "not_audited"
    full_kkt_tolerance: float = 0.0
    exactness_provenance: ExactFusionProvenance | None = None
    path_posterior: np.ndarray | None = None
    likelihood_model_id: str = "clipp2_legacy_major_minor_v1"
    likelihood_eps: float = 1e-6


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
        workset_max_bytes=int(options.workset_max_bytes),
        compressed_cache_max_bytes=int(options.compressed_cache_max_bytes),
        dense_fallback_policy=str(options.dense_fallback_policy),
        workset_add_batch=int(options.workset_add_batch),
        workset_max_expansions=int(options.workset_max_expansions),
        certificate_max_iter=int(options.certificate_max_iter),
        certificate_refinement_rounds=int(options.certificate_refinement_rounds),
        certificate_column_tol_scale=float(options.certificate_column_tol_scale),
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
        base_fusion_objective_hash=(
            ""
            if solver_context is None
            else str(solver_context.base_fusion_objective_hash)
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
        path_posterior=getattr(artifacts, "path_posterior", None),
        likelihood_model_id=str(
            getattr(
                artifacts,
                "likelihood_model_id",
                "clipp2_legacy_major_minor_v1",
            )
        ),
        likelihood_eps=float(options.eps),
    )


__all__ = [
    "FitOptions",
    "FitResult",
    "fit_fixed_objective",
]
