from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
from typing import TYPE_CHECKING, Literal, Mapping, TypeAlias

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
)

if TYPE_CHECKING:
    from ..objective import (
        BaseObjectiveKey,
        LambdaObjectiveKey,
        ObservedModel,
        TorchObservedModel,
    )


_GRAPH_FINGERPRINT_SCHEMA = "clipp2.pairwise-fusion-graph.v1"


def _graph_source_fingerprint(
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_w: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(_GRAPH_FINGERPRINT_SCHEMA.encode("ascii"))
    for name, value in (("edge_u", edge_u), ("edge_v", edge_v), ("edge_w", edge_w)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


SmoothGradientScope: TypeAlias = Literal[
    "mm_surrogate",
    "observed_objective",
    "clarke_piecewise_observed_objective_subgradient",
]
CertificateScope: TypeAlias = Literal["full_original_graph"]


class ExactSolverResourceLimit(MemoryError):
    """No configured exact backend can fit or fallback under its resource policy."""


@dataclass(frozen=True, slots=True)
class WorksetMemoryOptions:
    max_workset_bytes: int = DEFAULT_WORKSET_MAX_BYTES
    max_compressed_cache_bytes: int = DEFAULT_COMPRESSED_CACHE_MAX_BYTES

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
    # Scale-stable full-certificate diagnostics.  The historical fields above
    # remain solver-progress diagnostics; terminal raw-candidate admission uses
    # the backward-error residual under exactness-provenance schema v2.
    backward_error_stationarity_residual: float = float("inf")
    backward_error_edge_subgradient_residual: float = float("inf")
    backward_error_dual_ball_residual: float = float("inf")
    backward_error_kkt_residual: float = float("inf")

    @classmethod
    def from_mapping(cls, values: Mapping[str, float | int]) -> "KKTDiagnostics":
        integer_fields = {
            "num_interior_coordinates",
            "num_lower_active_coordinates",
            "num_upper_active_coordinates",
            "num_frozen_coordinates",
        }
        fail_closed_fields = {
            "backward_error_stationarity_residual",
            "backward_error_edge_subgradient_residual",
            "backward_error_dual_ball_residual",
            "backward_error_kkt_residual",
        }
        return cls(
            **{
                item.name: (int if item.name in integer_fields else float)(
                    values.get(item.name, float("inf"))
                    if item.name in fail_closed_fields
                    else values[item.name]
                )
                for item in fields(cls)
            }
        )

    def as_dict(self) -> dict[str, float | int]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


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
    certificate_scope: CertificateScope = "full_original_graph"


GraphFusionCertificate: TypeAlias = DenseEdgeCertificate | CompressedEdgeCertificate


@dataclass(frozen=True, slots=True)
class DenseWarmState:
    phi: torch.Tensor
    dual: torch.Tensor | None
    previous_lambda: float
    graph_hash: str


@dataclass(frozen=True, slots=True)
class PrimalOnlyWarmState:
    phi: torch.Tensor
    structure_hint: torch.Tensor | None = None
    certificate_hint: GraphFusionCertificate | None = None


BackendWarmState: TypeAlias = DenseWarmState | PrimalOnlyWarmState


@dataclass(frozen=True, slots=True)
class WorkCounters:
    outer_iterations: int = 0
    inner_iterations: int = 0
    workset_iterations: int = 0
    workset_expansions: int = 0
    streamed_edge_passes: int = 0
    dense_iterations: int = 0
    certificate_iterations: int = 0
    activity_passes: int = 0
    analytic_adjoint_passes: int = 0
    column_scan_passes: int = 0
    full_certificate_audit_passes: int = 0

    def __add__(self, other: "WorkCounters") -> "WorkCounters":
        if not isinstance(other, WorkCounters):
            return NotImplemented
        return WorkCounters(
            **{
                item.name: int(getattr(self, item.name))
                + int(getattr(other, item.name))
                for item in fields(self)
            }
        )

    @classmethod
    def from_attributes(cls, value: object) -> "WorkCounters":
        return cls(
            **{
                item.name: int(getattr(value, item.name, 0))
                for item in fields(cls)
            }
        )

    def as_dict(self) -> dict[str, int]:
        return {item.name: int(getattr(self, item.name)) for item in fields(self)}


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
    work_counters: WorkCounters
    fallback_reason: str = ""


@dataclass(frozen=True, slots=True)
class PairwiseFusionGraph:
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_w: np.ndarray
    name: str = "complete_uniform"
    degree_bound: int = 1
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        edge_u = np.array(self.edge_u, dtype=np.int32, copy=True, order="C")
        edge_v = np.array(self.edge_v, dtype=np.int32, copy=True, order="C")
        edge_w = np.array(self.edge_w, dtype=np.float64, copy=True, order="C")
        if edge_u.ndim != 1 or edge_v.ndim != 1 or edge_w.ndim != 1:
            raise ValueError("PairwiseFusionGraph edge arrays must be one-dimensional.")
        if edge_u.shape != edge_v.shape or edge_u.shape != edge_w.shape:
            raise ValueError("PairwiseFusionGraph edge arrays must have identical shapes.")
        if np.any(edge_u < 0) or np.any(edge_v < 0):
            raise ValueError("PairwiseFusionGraph edge indices must be nonnegative.")
        if np.any(edge_u == edge_v):
            raise ValueError("PairwiseFusionGraph may not contain self-loops.")
        if np.any(~np.isfinite(edge_w)) or np.any(edge_w < 0.0):
            raise ValueError(
                "PairwiseFusionGraph weights must be finite and nonnegative."
            )
        edge_u.setflags(write=False)
        edge_v.setflags(write=False)
        edge_w.setflags(write=False)
        object.__setattr__(self, "edge_u", edge_u)
        object.__setattr__(self, "edge_v", edge_v)
        object.__setattr__(self, "edge_w", edge_w)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "degree_bound", max(int(self.degree_bound), 1))
        object.__setattr__(
            self,
            "fingerprint",
            _graph_source_fingerprint(edge_u, edge_v, edge_w),
        )

@dataclass(frozen=True)
class TorchRuntime:
    device: torch.device
    device_name: str
    dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class TensorProblem:
    observed_model: TorchObservedModel
    eps: float
    major_prior: float
    source_model: ObservedModel | None = None

    @property
    def count_observed(self) -> torch.Tensor:
        return self.observed_model.observed


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
    base_fusion_objective_hash: str = ""
    base_objective_key: BaseObjectiveKey | None = None
    resource_fallback: str | None = None


@dataclass(slots=True)
class SolverState:
    phi: torch.Tensor
    dual: torch.Tensor | None
    previous_lambda: float
    warm_state: BackendWarmState | None = None
    certificate: GraphFusionCertificate | None = None
    objective_spec_hash: str = ""


@dataclass(frozen=True, slots=True)
class ObjectiveValue:
    """Observed fit loss and lambda-weighted fusion objective."""

    fit_loss: float
    fusion_penalty: float
    total: float

    @property
    def loglik(self) -> float:
        return float(-self.fit_loss)


@dataclass(frozen=True, slots=True)
class KKTComponents:
    """Scale-stable componentwise terminal backward error."""

    stationarity: float
    edge_subgradient: float
    dual_ball: float
    box: float

    @property
    def residual(self) -> float:
        return float(
            max(self.stationarity, self.edge_subgradient, self.dual_ball, self.box)
        )


@dataclass(frozen=True, slots=True)
class CertificateResult:
    """Terminal full-objective certificate and exactness provenance."""

    components: KKTComponents
    certified: bool
    admissible: bool
    global_optimum: bool
    status: str
    tolerance: float
    scope: str
    gradient_scope: str
    directional_admissible: bool
    witness: GraphFusionCertificate | None
    working_residual: float
    working_dtype: str
    audit_dtype: str
    precision_polished: bool
    precision_polish_delta: float
    residual_method: str
    fallback_reason: str

    @property
    def schema_version(self) -> int:
        return 2

    @property
    def estimator_role(self) -> str:
        return "raw_fused_lambda_path"

    @property
    def objective_faithful(self) -> bool:
        return True

    @property
    def full_kkt_certified(self) -> bool:
        return bool(self.certified)

    @property
    def residual(self) -> float:
        return self.components.residual

    @property
    def certificate_scope(self) -> str:
        return str(self.scope)

    @property
    def directional_kink_admissible(self) -> bool:
        return bool(self.directional_admissible)

    @property
    def working_precision_residual(self) -> float:
        return float(self.working_residual)

    @property
    def certificate_audit_dtype(self) -> str:
        return str(self.audit_dtype)

    @property
    def precision_polish_applied(self) -> bool:
        return bool(self.precision_polished)

    @property
    def precision_polish_max_abs_phi_delta(self) -> float:
        return float(self.precision_polish_delta)


@dataclass(frozen=True, slots=True)
class ConvergenceResult:
    converged: bool
    mm_consistency_violations: int
    failure_reason: str


@dataclass(frozen=True, slots=True)
class MultiStartResult:
    number_of_starts: int
    number_of_finite_starts: int
    best_objective: float
    second_best_objective: float
    objective_spread: float
    selected_objective_rank: int


@dataclass(frozen=True, slots=True)
class FitProvenance:
    objective_key: LambdaObjectiveKey
    graph_name: str
    device: str
    dtype: str
    inner_solver: str
    global_optimality_basis: str
    likelihood_eps: float

    @property
    def lambda_value(self) -> float:
        return float.fromhex(str(self.objective_key.lambda_hex))

    @property
    def objective_spec_hash(self) -> str:
        return str(self.objective_key.base.fingerprint)

    @property
    def base_fusion_objective_hash(self) -> str:
        return str(self.objective_key.base.fingerprint)

    @property
    def original_graph_hash(self) -> str:
        return str(self.objective_key.base.graph_hash)

    @property
    def certificate_problem_hash(self) -> str:
        return str(self.objective_key.fingerprint)


@dataclass(frozen=True, slots=True)
class RawFit:
    """Compact raw fixed-objective fit; partitions remain a secondary layer."""

    phi: np.ndarray
    objective: ObjectiveValue
    certificate: CertificateResult
    convergence: ConvergenceResult
    work: WorkCounters
    multistart: MultiStartResult
    state: SolverState | None
    provenance: FitProvenance

    def __post_init__(self) -> None:
        phi = np.array(self.phi, copy=True, order="C")
        if phi.ndim != 2 or not np.all(np.isfinite(phi)):
            raise ValueError("RawFit.phi must be a finite mutation-by-region matrix.")
        object.__setattr__(self, "phi", phi)
