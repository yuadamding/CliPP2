from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import torch

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
    edge_subgradient_residual: float
    dual_ball_residual: float
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
    def infinite(cls) -> "KKTDiagnostics":
        """Return fail-closed diagnostics before any KKT audit."""

        return cls(
            stationarity_residual=float("inf"),
            edge_subgradient_residual=float("inf"),
            dual_ball_residual=float("inf"),
            box_residual=float("inf"),
            kkt_residual=float("inf"),
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
    """Deterministic optimizer-work accounting.

    ``edge_region_visits`` is exact realized graph work: each primitive visit
    to one edge-region coordinate contributes one.  Full dense and streamed
    traversals therefore contribute ``|E| * S``; compressed workset
    traversals contribute ``|E_work| * S``.

    ``edge_pass_equivalents`` is the conservative integer cap unit.  A full
    dense or streamed traversal contributes one, while each partial workset
    traversal is rounded up to one so prospective budget checks remain safe.
    It counts logical work, not kernel launches or wall-clock time.
    ``certificate_full_graph_passes`` counts full-graph certificate traversals;
    certificate status, rather than a work counter, records whether a compressed
    representation was complete enough for an authoritative KKT audit.
    """

    inner_iterations: int = 0
    inner_stationarity_checks: int = 0
    inner_full_kkt_audits: int = 0
    outer_kkt_audits: int = 0
    certificate_iterations: int = 0
    certificate_full_graph_passes: int = 0
    partition_refit_coordinates: int = 0
    partition_refit_objective_evaluations: int = 0
    edge_pass_equivalents: int = 0
    edge_region_visits: int = 0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = int(getattr(self, item.name))
            if value < 0:
                raise ValueError(f"Work counter {item.name} must be nonnegative.")
            object.__setattr__(self, item.name, value)

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


@dataclass(slots=True)
class WorkLedger:
    """Single authority for accumulating deterministic optimizer work."""

    total: WorkCounters = field(default_factory=WorkCounters)

    def __post_init__(self) -> None:
        if not isinstance(self.total, WorkCounters):
            raise TypeError("WorkLedger total must be WorkCounters.")

    def charge(self, work: WorkCounters) -> None:
        """Accumulate one immutable work charge."""

        if not isinstance(work, WorkCounters):
            raise TypeError("WorkLedger charges must be WorkCounters.")
        self.total = self.total + work

    def charge_edge_passes(
        self,
        *,
        edge_count: int,
        num_regions: int,
        passes: int = 1,
    ) -> None:
        """Charge complete or partial logical edge traversals exactly once."""

        edge_count = int(edge_count)
        num_regions = int(num_regions)
        passes = int(passes)
        if edge_count < 0 or num_regions < 0 or passes < 0:
            raise ValueError("Edge-work dimensions and passes must be nonnegative.")
        if edge_count == 0 or num_regions == 0:
            passes = 0
        self.charge(
            WorkCounters(
                edge_pass_equivalents=passes,
                edge_region_visits=edge_count * num_regions * passes,
            )
        )

    def charge_inner_work(
        self,
        *,
        iterations: int = 0,
        stationarity_checks: int = 0,
        full_kkt_audits: int = 0,
    ) -> None:
        """Charge inner iterations and their diagnostic checks."""

        self.charge(
            WorkCounters(
                inner_iterations=iterations,
                inner_stationarity_checks=stationarity_checks,
                inner_full_kkt_audits=full_kkt_audits,
            )
        )

    def charge_outer_kkt_audits(self, count: int = 1) -> None:
        """Charge full observed-objective outer KKT audits."""

        self.charge(WorkCounters(outer_kkt_audits=count))

    def charge_certificate_work(
        self,
        *,
        iterations: int = 0,
        full_graph_passes: int = 0,
    ) -> None:
        """Charge certificate optimization and full-graph audit work."""

        self.charge(
            WorkCounters(
                certificate_iterations=iterations,
                certificate_full_graph_passes=full_graph_passes,
            )
        )

    def charge_partition_refit_work(
        self,
        *,
        coordinates: int = 0,
        objective_evaluations: int = 0,
    ) -> None:
        """Charge fixed-partition refit coordinate and objective work."""

        self.charge(
            WorkCounters(
                partition_refit_coordinates=coordinates,
                partition_refit_objective_evaluations=objective_evaluations,
            )
        )


@dataclass(frozen=True, slots=True)
class KKTAudit:
    """Typed KKT diagnostics with their work and edge-activity metadata."""

    diagnostics: KKTDiagnostics
    work: WorkCounters = WorkCounters()
    fused_edges: int | None = None
    nonzero_edges: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.diagnostics, KKTDiagnostics):
            raise TypeError("KKTAudit diagnostics must be KKTDiagnostics.")
        if not isinstance(self.work, WorkCounters):
            raise TypeError("KKTAudit work must be WorkCounters.")
        if (self.fused_edges is None) != (self.nonzero_edges is None):
            raise ValueError(
                "KKTAudit fused and nonzero edge counts must be present together."
            )
        if self.fused_edges is None:
            return
        fused_edges = int(self.fused_edges)
        nonzero_edges = int(self.nonzero_edges)
        if fused_edges < 0 or nonzero_edges < 0:
            raise ValueError("KKTAudit edge counts must be nonnegative.")
        object.__setattr__(self, "fused_edges", fused_edges)
        object.__setattr__(self, "nonzero_edges", nonzero_edges)

    def require_activity(self, *, edge_count: int) -> tuple[int, int]:
        """Return measured activity, failing closed when it is unavailable."""

        if self.fused_edges is None or self.nonzero_edges is None:
            raise ValueError("KKTAudit does not contain measured edge activity.")
        edge_count = int(edge_count)
        if edge_count < 0:
            raise ValueError("edge_count must be nonnegative.")
        if int(self.fused_edges) + int(self.nonzero_edges) != edge_count:
            raise ValueError("KKTAudit edge activity does not match the graph.")
        return int(self.fused_edges), int(self.nonzero_edges)


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
    work: WorkCounters = WorkCounters()


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
            raise ValueError(
                "PairwiseFusionGraph edge arrays must have identical shapes."
            )
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
    source_graph_fingerprint: str = ""

    @property
    def edge_u(self) -> torch.Tensor:
        return self.edge_index[0]

    @property
    def edge_v(self) -> torch.Tensor:
        return self.edge_index[1]


@dataclass(frozen=True, slots=True)
class SolverContext:
    source_model: ObservedModel
    observed_model: TorchObservedModel
    eps: float
    graph: TensorFusionGraph
    graph_spec: PairwiseFusionGraph
    exact_pilot: torch.Tensor
    pooled_start: torch.Tensor
    scalar_well_starts: tuple[torch.Tensor, ...]
    runtime: TorchRuntime
    data_fingerprint: str
    base_objective_key: BaseObjectiveKey
    resource_fallback: str | None = None
    audit_context_cache: dict[tuple[str, str, str], object] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        from ..objective import observed_box_fingerprint

        if self.observed_model.source_fingerprint != self.source_model.fingerprint:
            raise ValueError("Runtime and source observed models do not match.")
        if self.base_objective_key.likelihood_hash != (
            self.source_model.likelihood_fingerprint
        ):
            raise ValueError("SolverContext likelihood identity is inconsistent.")
        if self.base_objective_key.graph_hash != self.graph_spec.fingerprint:
            raise ValueError("SolverContext graph identity is inconsistent.")
        if self.graph.source_graph_fingerprint != self.graph_spec.fingerprint:
            raise ValueError("SolverContext runtime graph identity is inconsistent.")
        if self.base_objective_key.box_hash != observed_box_fingerprint(
            self.source_model
        ):
            raise ValueError("SolverContext objective-box identity is inconsistent.")
        if self.base_objective_key.eps_hex != float(self.eps).hex():
            raise ValueError("SolverContext epsilon identity is inconsistent.")
        if not str(self.data_fingerprint):
            raise ValueError("SolverContext data fingerprint must be nonempty.")
        if self.observed_model.alt.dtype != self.runtime.dtype:
            raise ValueError("SolverContext observed model has the wrong dtype.")
        if self.observed_model.alt.device != self.runtime.device:
            raise ValueError("SolverContext observed model has the wrong device.")
        if self.graph.weight.dtype != self.runtime.dtype:
            raise ValueError("SolverContext graph has the wrong dtype.")
        if self.graph.weight.device != self.runtime.device:
            raise ValueError("SolverContext graph has the wrong device.")

    @property
    def graph_hash(self) -> str:
        return str(self.graph_spec.fingerprint)

    @property
    def objective_spec_hash(self) -> str:
        return str(self.base_objective_key.fingerprint)

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

    @classmethod
    def from_diagnostics(cls, diagnostics: KKTDiagnostics) -> KKTComponents:
        """Project one terminal audit into the persisted component authority."""

        return cls(
            stationarity=float(
                diagnostics.backward_error_stationarity_residual
            ),
            edge_subgradient=float(
                diagnostics.backward_error_edge_subgradient_residual
            ),
            dual_ball=float(diagnostics.backward_error_dual_ball_residual),
            box=float(diagnostics.box_residual),
        )

    @property
    def residual(self) -> float:
        values = (
            float(self.stationarity),
            float(self.edge_subgradient),
            float(self.dual_ball),
            float(self.box),
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            return float("inf")
        return float(max(values))


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
    state: SolverState | None
    provenance: FitProvenance

    def __post_init__(self) -> None:
        phi = np.array(self.phi, copy=True, order="C")
        if phi.ndim != 2 or not np.all(np.isfinite(phi)):
            raise ValueError("RawFit.phi must be a finite mutation-by-region matrix.")
        object.__setattr__(self, "phi", phi)
