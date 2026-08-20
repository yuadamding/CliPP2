from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
import torch

from ..config import FitConfig, FitOptions
from ..io.data import TumorData
from .fusion.solver import fit_observed_data_pairwise_fusion
from .fusion.types import (
    RawFit,
    SolverContext,
    SolverState,
)


def _raw_property(*path: str) -> property:
    def get(result: FitResult):
        value = result.raw
        for name in path:
            value = getattr(value, name)
        return value

    return property(get)


@dataclass(slots=True)
class FitResult:
    """Temporary public adapter around the immutable, nested raw-fit result."""

    raw: RawFit

    phi = _raw_property("phi")
    objective = _raw_property("objective")
    certificate_result = _raw_property("certificate")
    certificate = _raw_property("certificate", "witness")
    convergence = _raw_property("convergence")
    work = _raw_property("work")
    multistart = _raw_property("multistart")
    provenance = _raw_property("provenance")

    @property
    def state(self) -> SolverState | None:
        return self.raw.state

    @state.setter
    def state(self, value: SolverState | None) -> None:
        self.raw = replace(self.raw, state=value)

    solver_state = state

    @property
    def admm_iterations(self) -> int:
        return (
            int(self.raw.work.dense_iterations)
            if self.raw.provenance.inner_solver == "admm_complete_graph"
            else 0
        )


# Explicit compatibility names live only on the public adapter. RawFit itself
# intentionally exposes only its eight nested fields.
_COMPATIBILITY_PATHS = {
    "loglik": ("objective", "loglik"),
    "penalized_objective": ("objective", "total"),
    "lambda_value": ("provenance", "lambda_value"),
    "n_clusters": ("multistart", "threshold_component_count"),
    "iterations": ("convergence", "iterations"),
    "converged": ("convergence", "converged"),
    "device": ("provenance", "device"),
    "dtype": ("provenance", "dtype"),
    "graph_name": ("provenance", "graph_name"),
    "converged_inner": ("convergence", "inner_converged"),
    "converged_outer": ("convergence", "outer_converged"),
    "final_relative_objective_change": ("convergence", "relative_objective_change"),
    "final_step_residual": ("convergence", "step_residual"),
    "selection_eligible": ("certificate", "admissible"),
    "stationarity_certified": ("certificate", "stationary"),
    "global_optimality_certified": ("certificate", "global_optimum"),
    "global_optimality_basis": ("provenance", "global_optimality_basis"),
    "best_start_objective": ("multistart", "best_objective"),
    "second_best_start_objective": ("multistart", "second_best_objective"),
    "objective_spread_across_starts": ("multistart", "objective_spread"),
    "selected_start_objective_rank": ("multistart", "selected_objective_rank"),
    "inner_solver": ("provenance", "inner_solver"),
    "inner_backend": ("certificate", "backend_name"),
    "backend_iterations": ("work", "inner_iterations"),
    "fallback_reason": ("certificate", "fallback_reason"),
    "exactness_provenance_version": ("certificate", "schema_version"),
    "estimator_role": ("certificate", "estimator_role"),
    "objective_faithful": ("certificate", "objective_faithful"),
    "objective_spec_hash": ("provenance", "objective_spec_hash"),
    "base_fusion_objective_hash": ("provenance", "base_fusion_objective_hash"),
    "original_graph_hash": ("provenance", "original_graph_hash"),
    "certificate_problem_hash": ("provenance", "certificate_problem_hash"),
    "certificate_scope": ("certificate", "scope"),
    "certificate_gradient_scope": ("certificate", "gradient_scope"),
    "full_kkt_certified": ("certificate", "certified"),
    "full_kkt_certificate_status": ("certificate", "status"),
    "full_kkt_tolerance": ("certificate", "tolerance"),
    "working_precision_kkt_residual": ("certificate", "working_residual"),
    "working_dtype": ("certificate", "working_dtype"),
    "certificate_audit_dtype": ("certificate", "audit_dtype"),
    "precision_polish_applied": ("certificate", "precision_polished"),
    "precision_polish_max_abs_phi_delta": ("certificate", "precision_polish_delta"),
    "exactness_provenance": ("certificate",),
    "likelihood_eps": ("provenance", "likelihood_eps"),
    "fixed_objective_kkt_residual": ("certificate", "components", "residual"),
    "outer_backward_error_stationarity_residual": ("certificate", "components", "stationarity"),
    "outer_backward_error_edge_subgradient_residual": ("certificate", "components", "edge_subgradient"),
    "outer_backward_error_dual_ball_residual": ("certificate", "components", "dual_ball"),
    "outer_kkt_certificate_status": ("certificate", "status"),
    "outer_kkt_fused_edges": ("certificate", "fused_edges"),
    "outer_kkt_nonzero_edges": ("certificate", "nonzero_edges"),
    "outer_stationarity_residual_before_dual_refine": ("certificate", "stationarity_before"),
    "outer_stationarity_residual_after_dual_refine": ("certificate", "stationarity_after"),
}
_COMPATIBILITY_PATHS.update(
    {
        name: ("certificate", "progress", target)
        for name, target in {
            "outer_stationarity_residual": "stationarity_residual",
            "outer_projected_stationarity_norm": "projected_stationarity_norm",
            "outer_stationarity_normalizer": "stationarity_normalizer",
            "outer_smooth_gradient_norm": "smooth_gradient_norm",
            "outer_fusion_adjustment_norm": "fusion_adjustment_norm",
            "outer_edge_subgradient_residual": "edge_subgradient_residual",
            "outer_dual_ball_residual": "dual_ball_residual",
            "outer_box_primal_violation": "box_primal_violation",
            "outer_num_frozen_coordinates": "num_frozen_coordinates",
            "outer_box_residual": "box_residual",
        }.items()
    }
)
for _group, _names in {
    "convergence": (
        "accepted_outer_steps", "accepted_full_steps", "accepted_damped_steps",
        "attempted_outer_steps", "failed_majorization_checks",
        "failed_inner_model_checks", "failed_em_envelope_checks",
        "failed_descent_checks", "failed_nonfinite_checks",
        "mm_consistency_violations", "failure_reason",
    ),
    "work": (
        "inner_iterations", "workset_iterations", "workset_expansions",
        "streamed_edge_passes", "dense_iterations", "certificate_iterations",
        "activity_passes", "analytic_adjoint_passes", "column_scan_passes",
        "full_certificate_audit_passes",
    ),
    "multistart": ("number_of_starts", "number_of_finite_starts"),
}.items():
    _COMPATIBILITY_PATHS.update({name: (_group, name) for name in _names})
for _name, _path in _COMPATIBILITY_PATHS.items():
    setattr(FitResult, _name, _raw_property(*_path))
del _group, _name, _names, _path


def fit_fixed_objective(
    data: TumorData,
    options: FitOptions | FitConfig,
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
    if isinstance(options, FitConfig):
        options = options.to_options()
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
    return FitResult(raw=artifacts)


__all__ = [
    "FitOptions",
    "FitResult",
    "RawFit",
    "fit_fixed_objective",
]
