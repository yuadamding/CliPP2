from __future__ import annotations

from dataclasses import dataclass, fields
import numpy as np
import torch

from ..config import FitConfig, FitOptions
from ..io.data import TumorData
from .fusion.solver import fit_observed_data_pairwise_fusion
from .fusion.types import (
    FusionFitArtifacts,
    SolverContext,
    SolverState,
)


@dataclass
class FitResult(FusionFitArtifacts):
    """Public solver result plus flattened exactness-provenance conveniences."""

    base_fusion_objective_hash: str = ""
    likelihood_eps: float = 1e-6

    def __getattr__(self, name: str):
        aliases = {
            "inner_backend": "backend_name",
            "exactness_provenance_version": "schema_version",
            "certificate_gradient_scope": "gradient_scope",
            "full_kkt_certificate_status": "status",
            "full_kkt_tolerance": "tolerance",
            "working_precision_kkt_residual": "working_precision_residual",
        }
        defaults = {
            "inner_backend": self.inner_solver,
            "backend_iterations": self.inner_iterations,
            "workset_iterations": 0,
            "workset_expansions": 0,
            "streamed_edge_passes": 0,
            "dense_iterations": 0,
            "certificate_iterations": 0,
            "activity_passes": 0,
            "analytic_adjoint_passes": 0,
            "column_scan_passes": 0,
            "full_certificate_audit_passes": 0,
            "fallback_reason": "",
            "exactness_provenance_version": 0,
            "estimator_role": "raw_fused_lambda_path",
            "objective_faithful": False,
            "objective_spec_hash": "",
            "original_graph_hash": "",
            "certificate_problem_hash": "",
            "certificate_scope": "unknown",
            "certificate_gradient_scope": "unknown",
            "full_kkt_certified": False,
            "full_kkt_certificate_status": "not_audited",
            "full_kkt_tolerance": 0.0,
            "working_precision_kkt_residual": self.fixed_objective_kkt_residual,
            "working_dtype": self.dtype,
            "certificate_audit_dtype": self.dtype,
            "precision_polish_applied": False,
            "precision_polish_max_abs_phi_delta": 0.0,
        }
        if name not in defaults:
            raise AttributeError(name)
        provenance = self.exactness_provenance
        return (
            defaults[name]
            if provenance is None
            else getattr(provenance, aliases.get(name, name))
        )


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
    values = {
        item.name: getattr(artifacts, item.name) for item in fields(FusionFitArtifacts)
    }
    values.update(
        cluster_labels=artifacts.cluster_labels.astype(np.int64, copy=False),
        cluster_diameters=artifacts.cluster_diameters.astype(np.float64, copy=False),
        major_call=artifacts.major_call.astype(bool, copy=False),
        multiplicity_estimated_mask=artifacts.multiplicity_estimated_mask.astype(
            bool, copy=False
        ),
        history=list(artifacts.history),
    )
    return FitResult(
        **values,
        base_fusion_objective_hash=(
            str(solver_context.base_fusion_objective_hash) if solver_context else ""
        ),
        likelihood_eps=float(options.eps),
    )


__all__ = [
    "FitOptions",
    "FitResult",
    "fit_fixed_objective",
]
