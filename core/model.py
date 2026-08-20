from __future__ import annotations

from dataclasses import dataclass, fields
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
    FusionFitArtifacts,
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
