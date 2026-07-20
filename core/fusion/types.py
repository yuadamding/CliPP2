from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, TypeAlias

import numpy as np
import torch

from .defaults import (
    DEFAULT_CERTIFICATE_MAX_ITER,
    DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS,
    DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    DEFAULT_WORKSET_ADD_BATCH,
    DEFAULT_WORKSET_MAX_BYTES,
    DEFAULT_WORKSET_MAX_EXPANSIONS,
    DenseFallbackPolicy as DenseFallbackPolicy,
    InnerBackend as InnerBackend,
)


SmoothGradientScope: TypeAlias = Literal[
    "mm_surrogate",
    "observed_objective",
]
CertificateScope: TypeAlias = Literal["full_original_graph"]
CertificateStatus: TypeAlias = Literal[
    "certified",
    "workset_incomplete",
    "not_certified",
    "resource_limit",
    "dense_fallback",
]


class ExactSolverResourceLimit(MemoryError):
    """No configured exact backend can fit or fallback under its resource policy."""


@dataclass(frozen=True, slots=True)
class WorksetMemoryOptions:
    max_workset_bytes: int = DEFAULT_WORKSET_MAX_BYTES
    max_compressed_cache_bytes: int = DEFAULT_COMPRESSED_CACHE_MAX_BYTES
    allow_heuristic_split_before_dense_fallback: bool = True

    def __post_init__(self) -> None:
        if int(self.max_workset_bytes) <= 0:
            raise ValueError("max_workset_bytes must be positive.")
        if int(self.max_compressed_cache_bytes) <= 0:
            raise ValueError("max_compressed_cache_bytes must be positive.")


@dataclass(frozen=True, slots=True)
class CertificateOptions:
    max_iter: int = DEFAULT_CERTIFICATE_MAX_ITER
    refinement_rounds: int = DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS
    max_expansions: int = DEFAULT_WORKSET_MAX_EXPANSIONS
    add_batch: int = DEFAULT_WORKSET_ADD_BATCH
    mapping_tolerance: float = 1e-6
    column_tolerance: float = 1e-6
    memory: WorksetMemoryOptions = WorksetMemoryOptions()

    def __post_init__(self) -> None:
        if int(self.max_iter) <= 0:
            raise ValueError("certificate max_iter must be positive.")
        if int(self.refinement_rounds) < 0:
            raise ValueError("certificate refinement_rounds must be nonnegative.")
        if int(self.max_expansions) <= 0:
            raise ValueError("certificate max_expansions must be positive.")
        if int(self.add_batch) <= 0:
            raise ValueError("certificate add_batch must be positive.")
        if float(self.mapping_tolerance) <= 0.0:
            raise ValueError("certificate mapping_tolerance must be positive.")
        if float(self.column_tolerance) <= 0.0:
            raise ValueError("certificate column_tolerance must be positive.")


@dataclass(frozen=True, slots=True)
class KKTDiagnostics:
    """Backend-neutral normalized graph-fusion KKT diagnostics."""

    stationarity_residual: float
    projected_stationarity_residual: float
    projected_stationarity_norm: float
    stationarity_normalizer: float
    smooth_gradient_norm: float
    fusion_adjustment_norm: float
    edge_subgradient_residual: float
    dual_ball_residual: float
    box_primal_violation: float
    num_interior_coordinates: int
    num_lower_active_coordinates: int
    num_upper_active_coordinates: int
    num_frozen_coordinates: int
    box_residual: float
    kkt_residual: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, float | int]) -> "KKTDiagnostics":
        return cls(
            stationarity_residual=float(values["stationarity_residual"]),
            projected_stationarity_residual=float(
                values["projected_stationarity_residual"]
            ),
            projected_stationarity_norm=float(values["projected_stationarity_norm"]),
            stationarity_normalizer=float(values["stationarity_normalizer"]),
            smooth_gradient_norm=float(values["smooth_gradient_norm"]),
            fusion_adjustment_norm=float(values["fusion_adjustment_norm"]),
            edge_subgradient_residual=float(values["edge_subgradient_residual"]),
            dual_ball_residual=float(values["dual_ball_residual"]),
            box_primal_violation=float(values["box_primal_violation"]),
            num_interior_coordinates=int(values["num_interior_coordinates"]),
            num_lower_active_coordinates=int(values["num_lower_active_coordinates"]),
            num_upper_active_coordinates=int(values["num_upper_active_coordinates"]),
            num_frozen_coordinates=int(values["num_frozen_coordinates"]),
            box_residual=float(values["box_residual"]),
            kkt_residual=float(values["kkt_residual"]),
        )

    def as_dict(self) -> dict[str, float | int]:
        return {
            "stationarity_residual": self.stationarity_residual,
            "projected_stationarity_residual": self.projected_stationarity_residual,
            "projected_stationarity_norm": self.projected_stationarity_norm,
            "stationarity_normalizer": self.stationarity_normalizer,
            "smooth_gradient_norm": self.smooth_gradient_norm,
            "fusion_adjustment_norm": self.fusion_adjustment_norm,
            "edge_subgradient_residual": self.edge_subgradient_residual,
            "dual_ball_residual": self.dual_ball_residual,
            "box_primal_violation": self.box_primal_violation,
            "num_interior_coordinates": self.num_interior_coordinates,
            "num_lower_active_coordinates": self.num_lower_active_coordinates,
            "num_upper_active_coordinates": self.num_upper_active_coordinates,
            "num_frozen_coordinates": self.num_frozen_coordinates,
            "box_residual": self.box_residual,
            "kkt_residual": self.kkt_residual,
        }


@dataclass(frozen=True, slots=True)
class DenseEdgeCertificate:
    dual: torch.Tensor
    graph_hash: str
    gradient_scope: SmoothGradientScope
    certificate_scope: CertificateScope = "full_original_graph"


@dataclass(frozen=True, slots=True)
class CompressedEdgeCertificate:
    labels: torch.Tensor
    centers: torch.Tensor
    internal_edge_ids: torch.Tensor
    internal_dual: torch.Tensor
    graph_hash: str
    gradient_scope: SmoothGradientScope
    nonfused_dual_mode: Literal["analytic_streamed"] = "analytic_streamed"
    certificate_scope: CertificateScope = "full_original_graph"


GraphFusionCertificate: TypeAlias = DenseEdgeCertificate | CompressedEdgeCertificate


@dataclass(frozen=True, slots=True)
class DenseWarmState:
    phi: torch.Tensor
    dual: torch.Tensor | None
    previous_lambda: float
    graph_hash: str


@dataclass(frozen=True, slots=True)
class QuotientWorksetWarmState:
    phi: torch.Tensor
    labels: torch.Tensor
    centers: torch.Tensor
    quotient_dual: torch.Tensor | None
    internal_edge_ids: torch.Tensor
    internal_dual: torch.Tensor
    graph_hash: str
    previous_lambda: float


@dataclass(frozen=True, slots=True)
class PrimalOnlyWarmState:
    phi: torch.Tensor
    structure_hint: torch.Tensor | None = None
    certificate_hint: GraphFusionCertificate | None = None
    structure_hint_is_heuristic: bool = True


BackendWarmState: TypeAlias = (
    DenseWarmState | QuotientWorksetWarmState | PrimalOnlyWarmState
)


@dataclass(frozen=True, slots=True)
class QuotientFailureProvenance:
    lambda_value: float
    graph_hash: str
    reason: str


@dataclass(frozen=True, slots=True)
class BackendWorkCounters:
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


@dataclass(frozen=True, slots=True)
class InnerSolveResult:
    phi: torch.Tensor
    backend_name: str
    warm_state: BackendWarmState
    surrogate_certificate: GraphFusionCertificate | None
    surrogate_kkt: KKTDiagnostics
    converged: bool
    inner_iterations: int
    backend_iterations: int
    work_counters: BackendWorkCounters
    fallback_reason: str = ""


QuotientAttemptStatus: TypeAlias = Literal[
    "certified",
    "not_certified",
    "workset_incomplete",
    "resource_limit",
    "quotient_unconverged",
]


@dataclass(frozen=True, slots=True)
class QuotientAttemptResult:
    status: QuotientAttemptStatus
    phi_candidate: torch.Tensor
    warm_state: QuotientWorksetWarmState | PrimalOnlyWarmState
    certificate_hint: CompressedEdgeCertificate | None
    exact_inner_objective: float
    work_counters: BackendWorkCounters
    reason: str
    certified_result: InnerSolveResult | None = None


@dataclass(frozen=True, slots=True)
class ExactFusionProvenance:
    """Evidence used to decide fixed-objective candidate eligibility.

    Solver identity and iteration counts are deliberately diagnostic.  Exact
    eligibility is carried only by the objective/graph scopes and the normalized
    terminal KKT certificate.
    """

    schema_version: int = 1
    estimator_role: str = "raw_fused_lambda_path"
    objective_faithful: bool = True
    objective_spec_hash: str = ""
    original_graph_hash: str = ""
    certificate_problem_hash: str = ""
    certificate_scope: str = "full_original_graph"
    gradient_scope: str = "observed_objective"
    full_kkt_certified: bool = False
    status: str = "not_audited"
    residual: float = float("inf")
    tolerance: float = 0.0
    backend_name: str = "unknown"
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
    stationarity_certified: bool
    global_optimality_certified: bool
    global_optimality_basis: str
    number_of_starts: int
    number_of_finite_starts: int
    best_start_objective: float
    second_best_start_objective: float
    objective_spread_across_starts: float
    selected_start_objective_rank: int
    solver_state: SolverState | None = None
    torch_result: TorchFitResult | None = None
    # Backward-compatible solver provenance. ``iterations`` above remains the
    # number of outer MM iterations; this is the accumulated work performed by
    # the inner convex solver for the selected start.
    inner_iterations: int = 0
    admm_iterations: int = 0
    inner_solver: str = "unknown"
    certificate: GraphFusionCertificate | None = None
    exactness_provenance: ExactFusionProvenance | None = None


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
    count_observed: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class TensorFusionGraph:
    edge_index: torch.Tensor
    weight: torch.Tensor
    degree: torch.Tensor
    pdhg_tau_node: torch.Tensor
    num_nodes: int
    is_complete: bool
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
    data_fingerprint: str
    graph_hash: str = ""
    objective_spec_hash: str = ""
    resource_fallback: str | None = None


@dataclass(slots=True)
class SolverState:
    phi: torch.Tensor
    dual: torch.Tensor | None
    previous_lambda: float
    warm_state: BackendWarmState | None = None
    certificate: GraphFusionCertificate | None = None
    quotient_failure: QuotientFailureProvenance | None = None


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
    admm_iterations: int = 0
    inner_solver: str = "unknown"
    certificate: GraphFusionCertificate | None = None
    exactness_provenance: ExactFusionProvenance | None = None
