"""Public fixed-objective fit boundary."""

from __future__ import annotations

import numpy as np
import torch

from ..config import FitConfig
from ..io.data import TumorData
from .fusion.solver import fit_observed_data_pairwise_fusion
from .fusion.types import RawFit, SolverContext, SolverState


def fit_fixed_objective(
    data: TumorData,
    config: FitConfig,
    phi_start: np.ndarray | torch.Tensor | None = None,
    exact_pilot: np.ndarray | torch.Tensor | None = None,
    pooled_start: np.ndarray | torch.Tensor | None = None,
    scalar_well_starts: list[np.ndarray | torch.Tensor] | None = None,
    start_mode: str = "full",
    append_default_nonconvex_starts: bool | None = None,
    runtime=None,
    torch_data=None,
    solver_context: SolverContext | None = None,
    solver_state: SolverState | None = None,
) -> RawFit:
    """Fit the immutable observed objective described by ``config``."""

    solver = config.solver
    resources = solver.resources
    certificate = solver.certificate
    graph = config.graph
    return fit_observed_data_pairwise_fusion(
        data=data,
        lambda_value=float(config.lambda_value),
        major_prior=float(config.major_prior),
        eps=float(config.eps),
        outer_max_iter=max(int(solver.outer_max_iter), 1),
        inner_max_iter=max(int(solver.inner_max_iter), 16),
        tol=float(solver.tolerance),
        certification_tol=(
            float(solver.tolerance)
            if solver.certification_tolerance is None
            else float(solver.certification_tolerance)
        ),
        use_backward_error_progress=bool(solver.use_backward_error_progress),
        phi_start=phi_start,
        graph=graph.graph,
        adaptive_weight_gamma=float(graph.adaptive_weight_gamma),
        adaptive_weight_floor=float(graph.adaptive_weight_floor),
        adaptive_weight_baseline=float(graph.adaptive_weight_baseline),
        exact_pilot=exact_pilot,
        pooled_start=pooled_start,
        scalar_well_starts=scalar_well_starts,
        start_mode=str(start_mode),
        append_default_nonconvex_starts=append_default_nonconvex_starts,
        device=str(config.runtime.device),
        dtype=str(config.runtime.dtype),
        objective_shape=str(solver.objective_shape),
        workset_max_bytes=int(resources.workset_max_bytes),
        compressed_cache_max_bytes=int(resources.compressed_cache_max_bytes),
        dense_fallback_policy=str(config.runtime.fallback),
        workset_add_batch=int(resources.workset_add_batch),
        workset_max_expansions=int(resources.workset_max_expansions),
        certificate_max_iter=int(certificate.max_iter),
        certificate_refinement_rounds=int(certificate.refinement_rounds),
        certificate_column_tol_scale=float(certificate.column_tolerance_scale),
        runtime=runtime,
        torch_data=torch_data,
        solver_context=solver_context,
        solver_state=solver_state,
        verbose=bool(config.runtime.verbose),
    )


FitResult = RawFit

__all__ = ["FitConfig", "FitResult", "RawFit", "fit_fixed_objective"]
