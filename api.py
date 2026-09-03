"""Lightweight public entry points for fixed-objective fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import FitConfig
    from .core.fusion.interface import SolveBudget, SolverInit
    from .core.fusion.types import RawFit
    from .io.data import TumorData


def fit_fixed_objective(
    data: TumorData,
    config: FitConfig,
    *,
    lambda_value: float,
    init: SolverInit | None = None,
    budget: SolveBudget | None = None,
) -> RawFit:
    """Fit the immutable observed objective described by ``config``.

    Numerical modules are imported only when a fit is requested, keeping the
    package and configuration surfaces inexpensive to import.
    """

    from .core.fusion.solver import fit_observed_data_pairwise_fusion
    from .core.fusion.interface import (
        FusionProblem,
        SolveBudget,
        SolvePlan,
        SolverInit,
    )

    solver = config.solver
    resources = solver.resources
    certificate = solver.certificate
    graph = config.graph
    return fit_observed_data_pairwise_fusion(
        FusionProblem(
            data=data,
            lambda_value=float(lambda_value),
            major_prior=float(config.major_prior),
            eps=float(config.eps),
            graph=graph.graph,
            adaptive_weight_gamma=float(graph.adaptive_weight_gamma),
            adaptive_weight_floor=float(graph.adaptive_weight_floor),
            adaptive_weight_baseline=float(graph.adaptive_weight_baseline),
            objective_shape=str(solver.objective_shape),
        ),
        SolvePlan(
            outer_max_iter=max(int(solver.outer_max_iter), 1),
            inner_max_iter=max(int(solver.inner_max_iter), 16),
            tol=float(solver.tolerance),
            certification_tol=(
                float(solver.tolerance)
                if solver.certification_tolerance is None
                else float(solver.certification_tolerance)
            ),
            use_backward_error_progress=bool(solver.use_backward_error_progress),
            stagnation_audit_patience=int(solver.stagnation_audit_patience),
            device=str(config.runtime.device),
            dtype=str(config.runtime.dtype),
            workset_max_bytes=int(resources.workset_max_bytes),
            compressed_cache_max_bytes=int(resources.compressed_cache_max_bytes),
            dense_fallback_policy=str(config.runtime.fallback),
            workset_add_batch=int(resources.workset_add_batch),
            workset_max_expansions=int(resources.workset_max_expansions),
            certificate_max_iter=int(certificate.max_iter),
            certificate_refinement_rounds=int(certificate.refinement_rounds),
            certificate_column_tol_scale=float(certificate.column_tolerance_scale),
            verbose=bool(config.runtime.verbose),
        ),
        SolverInit() if init is None else init,
        SolveBudget() if budget is None else budget,
    )


__all__ = ["fit_fixed_objective"]
