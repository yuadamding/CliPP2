from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import torch

from ...io.data import ImmutableArrayRecord, TumorData, readonly_array, tumor_data_fingerprint
from ..objective import compile_observed_model, make_base_objective_key
from ...config import (
    DEFAULT_CERTIFICATE_MAX_ITER,
    DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS,
    DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    DEFAULT_WORKSET_ADD_BATCH,
    DEFAULT_WORKSET_MAX_BYTES,
    DEFAULT_WORKSET_MAX_EXPANSIONS,
    DenseFallbackPolicy as DenseFallbackPolicy,
)

if TYPE_CHECKING:
    from ..scalar import ScalarGlobalMinimumCertificate
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


def _residual_max(*values: float) -> float:
    """Aggregate nonnegative residuals without hiding invalid components."""
    normalized = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in normalized):
        return math.inf
    return max(normalized, default=0.0)


@dataclass(frozen=True, slots=True)
class KKTDiagnostics:
    """Backend-neutral normalized graph-fusion KKT diagnostics."""

    stationarity_residual: float
    edge_subgradient_residual: float
    dual_ball_residual: float
    box_residual: float
    # Scale-stable full-certificate diagnostics.  The historical fields above
    # remain solver-progress diagnostics; terminal raw-candidate admission uses
    # the backward-error residual under exactness-provenance schema v2.
    backward_error_stationarity_residual: float = float("inf")
    backward_error_edge_subgradient_residual: float = float("inf")
    backward_error_dual_ball_residual: float = float("inf")

    @property
    def kkt_residual(self) -> float:
        return _residual_max(
            self.stationarity_residual, self.edge_subgradient_residual,
            self.dual_ball_residual, self.box_residual,
        )

    @property
    def backward_error_kkt_residual(self) -> float:
        return _residual_max(
            self.backward_error_stationarity_residual,
            self.backward_error_edge_subgradient_residual,
            self.backward_error_dual_ball_residual, self.box_residual,
        )

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
    full_certificate_audit_passes: int = 0

    def __add__(self, other: "WorkCounters") -> "WorkCounters":
        if not isinstance(other, WorkCounters):
            return NotImplemented
        return WorkCounters(
            full_certificate_audit_passes=(
                int(self.full_certificate_audit_passes)
                + int(other.full_certificate_audit_passes)
            )
        )


@dataclass(frozen=True, slots=True)
class InnerSolveResult:
    phi: torch.Tensor
    backend_name: str
    warm_state: BackendWarmState
    surrogate_certificate: GraphFusionCertificate | None
    surrogate_kkt: KKTDiagnostics
    converged: bool
    fallback_reason: str = ""
    iterations: int = 0


@dataclass(frozen=True, slots=True)
class PairwiseFusionGraph(ImmutableArrayRecord):
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_w: np.ndarray
    name: str = "complete_uniform"
    degree_bound: int = 1
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        edge_u = readonly_array(self.edge_u, dtype=np.int32)
        edge_v = readonly_array(self.edge_v, dtype=np.int32)
        edge_w = readonly_array(self.edge_w, dtype=np.float64)
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
class PreparedProblem:
    source_data: TumorData
    source_model: ObservedModel
    model: TorchObservedModel
    graph: TensorFusionGraph
    graph_spec: PairwiseFusionGraph
    exact_pilot: torch.Tensor
    pooled_start: torch.Tensor
    scalar_well_starts: tuple[torch.Tensor, ...]
    runtime: TorchRuntime
    data_fingerprint: str
    base_objective_key: BaseObjectiveKey
    resource_fallback: str | None = None
    fallback_policy: str = "cpu_allowed"
    verbose: bool = False
    adaptive_graph_options: tuple[float, float, float] | None = None
    # Float64 scalar source results, in mutation-major/region-minor order.
    # The legacy exact_pilot tensor may be a float32 view of their argmins;
    # unresolved bounds never imply globally certified scalar minima.
    scalar_pilot_certificates: tuple[ScalarGlobalMinimumCertificate, ...] = ()
    audit_context_cache: dict[tuple[str, str, str], object] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    # Preserve this evidence across dataclasses.replace: changing a runtime
    # view must not silently establish a new baseline under old source hashes.
    _tensor_snapshot: tuple = field(default=(), compare=False, repr=False)

    @property
    def eps(self) -> float:
        return float.fromhex(self.base_objective_key.eps_hex)

    @property
    def lower(self) -> torch.Tensor:
        return self.model.lower

    @property
    def upper(self) -> torch.Tensor:
        return self.model.upper

    @property
    def graph_hash(self) -> str:
        return self.base_objective_key.graph_hash

    @property
    def objective_spec_hash(self) -> str:
        return self.base_objective_key.fingerprint

    @property
    def base_fusion_objective_hash(self) -> str:
        return self.base_objective_key.fingerprint

    def __post_init__(self) -> None:
        if self.base_objective_key is None:
            raise ValueError("PreparedProblem requires a typed base-objective key.")
        if self._tensor_snapshot:
            self.assert_runtime_unchanged()
            return
        snapshot = []
        for name, tensor in self._runtime_tensors():
            if not torch.is_tensor(tensor):
                raise ValueError(f"Prepared runtime {name} must be a Tensor.")
            try:
                version = tensor._version
            except RuntimeError as error:
                raise ValueError("Prepared runtime requires version-tracked tensors.") from error
            snapshot.append((name, tensor, version, tuple(tensor.shape), tensor.dtype, tensor.device))
        object.__setattr__(self, "_tensor_snapshot", tuple(snapshot))

    def _runtime_tensors(self):
        model = self.model
        for name in ("alt", "nonalt", "observed", "lower", "upper", "slope", "log_prior", "valid"):
            yield f"model.{name}", getattr(model, name)
        for name in ("edge_index", "weight", "degree", "pdhg_tau_node"):
            yield f"graph.{name}", getattr(self.graph, name)
        for name in ("exact_pilot", "pooled_start"):
            yield name, getattr(self, name)
        for index, tensor in enumerate(self.scalar_well_starts):
            yield f"scalar_well_starts[{index}]", tensor

    def assert_runtime_unchanged(self) -> None:
        """Reject ordinary in-place tensor edits without GPU synchronization.

        Runtime views are private implementation state, not writable buffers.
        As with PyTorch autograd version checks, unsafe writes through ``.data``
        or foreign memory aliases are outside this interface's contract.
        """
        current = tuple(self._runtime_tensors())
        if len(current) != len(self._tensor_snapshot):
            raise ValueError("Prepared runtime tensor set changed; prepare a new problem.")
        for (name, tensor), (original_name, original, version, shape, dtype, device) in zip(
            current, self._tensor_snapshot,
        ):
            if (
                name != original_name or tensor is not original
                or tensor._version != version or tuple(tensor.shape) != shape
                or tensor.dtype != dtype or tensor.device != device
            ):
                raise ValueError(f"Prepared runtime tensor {name} changed; prepare a new problem.")

    def validate(self, *, allow_deferred_graph: bool = False) -> None:
        """Validate frozen source identity without accepting a competing request."""
        self.assert_runtime_unchanged()
        data = self.source_data
        if self.data_fingerprint != tumor_data_fingerprint(data):
            raise ValueError("Prepared problem data fingerprint is inconsistent.")
        source = compile_observed_model(data, eps=self.eps, multiplicity_policy=self.model.support_policy)
        if (
            self.source_model is None
            or self.source_model.fingerprint != source.fingerprint
            or self.model.source_fingerprint != source.fingerprint
            or self.model.coupling != source.coupling
            or self.model.support_policy != source.support_policy
        ):
            raise ValueError("Prepared problem likelihood or epsilon identity is inconsistent.")
        if self.graph_spec.name == "deferred_likelihood_pilot" and not allow_deferred_graph:
            raise ValueError("A deferred likelihood pilot is not a prepared fusion graph.")
        if self.graph_hash != self.graph_spec.fingerprint:
            raise ValueError("Prepared problem graph identity is inconsistent.")
        key = make_base_objective_key(
            source, graph_hash=self.graph_hash, eps=self.eps,
            lower=source.lower, upper=source.upper,
        )
        if self.base_objective_key != key:
            raise ValueError("Prepared problem objective identity is inconsistent.")


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
    """Lambda-weighted observed fusion objective."""

    total: float


@dataclass(frozen=True, slots=True)
class KKTComponents:
    """Scale-stable componentwise terminal backward error."""

    stationarity: float
    edge_subgradient: float
    dual_ball: float
    box: float

    @property
    def residual(self) -> float:
        return _residual_max(self.stationarity, self.edge_subgradient, self.dual_ball, self.box)


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


@dataclass(frozen=True, slots=True)
class ConvergenceResult:
    converged: bool
    mm_consistency_violations: int
    stage_outer_iterations: int = 0
    stage_outer_max_iter: int = 0
    stage_inner_iterations: int = 0
    stage_inner_max_iter: int = 0
    stage_inner_solve_calls: int = 0
    stop_reason: str = "not_recorded"
    progress_residual_method: str = "not_recorded"
    solve_tolerance: float = float("nan")
    legacy_stop_kkt_residual: float = float("inf")
    componentwise_stop_kkt_residual: float = float("inf")
    accepted_full_steps: int = 0
    accepted_damped_steps: int = 0
    rejected_outer_steps: int = 0


@dataclass(frozen=True, slots=True)
class FitProvenance:
    objective_key: LambdaObjectiveKey
    source_data_hash: str
    device: str
    dtype: str
    inner_solver: str
    global_optimality_basis: str
    scalar_pilot_certificates: tuple[ScalarGlobalMinimumCertificate, ...] = ()
    multiplicity_policy: str = "independent_broad"

    @property
    def likelihood_eps(self) -> float:
        return float.fromhex(self.objective_key.base.eps_hex)

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
class RawFit(ImmutableArrayRecord):
    """Compact raw fixed-objective fit; partitions remain a secondary layer."""

    phi: np.ndarray
    objective: ObjectiveValue
    certificate: CertificateResult
    convergence: ConvergenceResult
    work: WorkCounters
    state: SolverState | None
    provenance: FitProvenance

    def __post_init__(self) -> None:
        phi = readonly_array(self.phi)
        if phi.ndim != 2 or not np.all(np.isfinite(phi)):
            raise ValueError("RawFit.phi must be a finite mutation-by-region matrix.")
        object.__setattr__(self, "phi", phi)
