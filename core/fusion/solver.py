from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import torch

from ...io.data import (
    TumorData,
    tumor_data_fingerprint,
)
from ..objective import (
    ObservedModel,
    TorchObservedModel,
    TorchObservedTerms,
    compile_observed_model,
    default_phi_initialization,
    make_base_objective_key,
    make_lambda_objective_key,
    model_to_torch,
    observed_box_fingerprint,
    observed_internal_breakpoints_torch,
    observed_one_sided_gradients_torch,
)
from ...config import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    normalize_dense_fallback_policy,
)
from .certificates import (
    CertificateProblem,
    build_certificate_gradient,
    certify,
)
from .graph import resolve_pairwise_fusion_graph
from .interface import FusionProblem, SolveBudget, SolvePlan, SolverInit
from .graph_ops import (
    build_complete_adaptive_tensor_graph,
    dense_complete_solver_memory_preflight,
    graph_adjoint_edges,
    project_dual_ball,
    tensor_graph_to_pairwise_graph,
    tensorize_graph,
)
from .policy import (
    NextAction,
    PolicyState,
    decide_next_action,
    record_attempt,
)
from .starts import (
    compute_pooled_observed_data_start_torch,
    compute_scalar_mutation_region_wells_torch,
    compute_scalar_well_start_bank_torch,
)
from .torch_backend import (
    CudaUnavailableError,
    DEFAULT_INNER_KKT_CHECK_EVERY,
    as_runtime_tensor,
    mutation_region_terms_torch,
    dtype_name,
    em_surrogate_terms_torch,
    graph_adjoint_edges_in_dtype,
    pairwise_penalty_torch,
    resolve_runtime,
    solve_majorized_subproblem_alm_torch,
    solve_majorized_subproblem_pdhg_torch,
    validate_lambda_value,
)
from .types import (
    CertificateResult,
    CertificateOptions,
    CompressedEdgeCertificate,
    ConvergenceResult,
    DenseEdgeCertificate,
    DenseWarmState,
    ExactSolverResourceLimit,
    FitProvenance,
    GraphFusionCertificate,
    InnerSolveResult,
    KKTComponents,
    KKTDiagnostics,
    ObjectiveValue,
    PairwiseFusionGraph,
    PrimalOnlyWarmState,
    RawFit,
    SolverContext,
    SolverState,
    TensorFusionGraph,
    TorchRuntime,
    WorkCounters,
    WorkLedger,
    WorksetMemoryOptions,
)


@dataclass(frozen=True, slots=True)
class _Float64AuditContext:
    runtime: TorchRuntime
    observed_model: TorchObservedModel
    graph: TensorFusionGraph
    lower: torch.Tensor
    upper: torch.Tensor


def _float64_audit_context(
    *,
    source_model: ObservedModel,
    graph_spec: PairwiseFusionGraph,
    graph_hash: str,
    device: torch.device,
    cache: dict[tuple[str, str, str], object] | None,
) -> _Float64AuditContext:
    """Return the immutable float64 audit tensors for one tumor/graph/device."""

    device = torch.device(device)
    key = (str(source_model.fingerprint), str(graph_hash), str(device))
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            if not isinstance(cached, _Float64AuditContext):
                raise TypeError("SolverContext float64 audit cache is corrupted.")
            return cached
    runtime = TorchRuntime(
        device=device,
        device_name=str(device),
        dtype=torch.float64,
    )
    context = _Float64AuditContext(
        runtime=runtime,
        observed_model=model_to_torch(source_model, runtime),
        graph=tensorize_graph(
            graph_spec,
            runtime,
            num_nodes=int(source_model.shape[0]),
        ),
        lower=torch.as_tensor(
            np.array(source_model.lower, copy=True),
            dtype=torch.float64,
            device=device,
        ),
        upper=torch.as_tensor(
            np.array(source_model.upper, copy=True),
            dtype=torch.float64,
            device=device,
        ),
    )
    if cache is not None:
        cache[key] = context
    return context


def _terminal_backward_error_audit_float64(
    *,
    source_model: ObservedModel,
    phi: torch.Tensor,
    certificate: GraphFusionCertificate | None,
    graph_spec: PairwiseFusionGraph,
    graph_hash: str,
    lambda_value: float,
    eps: float,
    tol: float,
    audit_context_cache: dict[tuple[str, str, str], object] | None = None,
    return_work: bool = False,
) -> (
    tuple[KKTDiagnostics, str, bool, float]
    | tuple[KKTDiagnostics, str, bool, float, WorkCounters]
):
    """Audit the unchanged terminal witness with float64 backward error."""

    audit = _float64_audit_context(
        source_model=source_model,
        graph_spec=graph_spec,
        graph_hash=graph_hash,
        device=phi.device,
        cache=audit_context_cache,
    )
    model64 = audit.observed_model
    graph64 = audit.graph
    lower64 = audit.lower
    upper64 = audit.upper
    phi64 = phi.to(dtype=torch.float64, device=audit.runtime.device)
    terms64 = mutation_region_terms_torch(
        model64,
        phi64,
        eps=float(eps),
    )
    gradient = build_certificate_gradient(
        model64,
        phi=phi64,
        smooth_gradient=terms64.gradient,
        lower=lower64,
        upper=upper64,
        eps=float(eps),
        tol=float(tol),
    )
    dense_dual = getattr(certificate, "dual", None)
    work = WorkLedger()
    if torch.is_tensor(dense_dual) and bool(torch.any(gradient.at_breakpoint).item()):
        adjustment = graph_adjoint_edges_in_dtype(
            dense_dual,
            edge_u=graph64.edge_u,
            edge_v=graph64.edge_v,
            num_nodes=int(phi64.shape[0]),
            dtype=torch.float64,
            device=audit.runtime.device,
        )
        work.charge_edge_passes(
            edge_count=int(graph64.edge_u.numel()),
            num_regions=int(phi64.shape[1]),
        )
        gradient = build_certificate_gradient(
            model64,
            phi=phi64,
            smooth_gradient=terms64.gradient,
            lower=lower64,
            upper=upper64,
            eps=float(eps),
            tol=float(tol),
            fusion_adjoint=adjustment,
        )
    result = certify(
        problem=CertificateProblem(
            graph=graph64,
            graph_hash=str(graph_hash),
            lower=lower64,
            upper=upper64,
            lambda_value=float(lambda_value),
            atol=float(tol),
        ),
        phi=phi64,
        gradient=gradient,
        witness=certificate,
        refine=False,
    )
    penalty64, penalty_work = _evaluate_pairwise_penalty_torch(
        phi64,
        edge_u=graph64.edge_u,
        edge_v=graph64.edge_v,
        edge_w=graph64.weight,
        lambda_value=float(lambda_value),
    )
    work.charge(result.work_counters)
    work.charge(penalty_work)
    _, _, objective64 = _objective_value_from_mutation_region_terms_torch(
        terms64,
        penalty_tensor=penalty64,
    )
    result_values = (
        result.diagnostics,
        gradient.scope,
        gradient.directional_admissible,
        float(objective64),
    )
    if return_work:
        return (*result_values, work.total)
    return result_values


def _deduplicate_starts(
    starts: list[np.ndarray | torch.Tensor],
    *,
    runtime,
    atol: float = 1e-8,
) -> list[np.ndarray | torch.Tensor]:
    unique: list[np.ndarray | torch.Tensor] = []
    unique_tensors: list[torch.Tensor] = []
    for start in starts:
        start_tensor = as_runtime_tensor(start, runtime).detach()
        duplicate = any(
            torch.allclose(start_tensor, retained, rtol=0.0, atol=float(atol))
            for retained in unique_tensors
        )
        if duplicate:
            continue
        unique_tensors.append(start_tensor)
        unique.append(start)
    return unique


def _inner_model_value_torch(
    phi: torch.Tensor,
    *,
    U: torch.Tensor,
    h: torch.Tensor,
    penalty_tensor: torch.Tensor,
) -> torch.Tensor:
    quad = 0.5 * torch.sum(h * torch.square(phi - U))
    return quad + penalty_tensor


def _evaluate_pairwise_penalty_torch(
    phi: torch.Tensor,
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
) -> tuple[torch.Tensor, WorkCounters]:
    """Evaluate one penalty and charge its budget unit and exact visits."""

    penalty = pairwise_penalty_torch(
        phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
    )
    edge_passes = _pairwise_penalty_edge_passes(
        edge_u=edge_u,
        lambda_value=lambda_value,
    )
    work = WorkLedger()
    work.charge_edge_passes(
        edge_count=int(edge_u.numel()),
        num_regions=int(phi.shape[1]),
        passes=edge_passes,
    )
    return penalty, work.total


def _pairwise_penalty_edge_passes(
    *,
    edge_u: torch.Tensor,
    lambda_value: float,
) -> int:
    return int(float(lambda_value) > 0.0 and int(edge_u.numel()) > 0)


def _nonterminal_edge_work_fits(
    work: WorkCounters,
    limit: int | None,
    *,
    edge_passes: int,
) -> bool:
    """Reserve the documented terminal allowance before optional graph work."""

    if limit is None or int(edge_passes) <= 0:
        return True
    return bool(
        int(work.edge_pass_equivalents)
        + int(edge_passes)
        + _MANDATORY_TERMINAL_EDGE_PASS_ALLOWANCE
        <= int(limit)
    )


def _objective_value_from_mutation_region_terms_torch(
    mutation_region_terms: TorchObservedTerms,
    *,
    penalty_tensor: torch.Tensor,
) -> tuple[float, float, float]:
    fit_loss_tensor = torch.sum(mutation_region_terms.loss)
    objective_tensor = fit_loss_tensor + penalty_tensor
    fit_loss, penalty, objective = (
        float(value)
        for value in torch.stack(
            [
                fit_loss_tensor.detach(),
                penalty_tensor.detach(),
                objective_tensor.detach(),
            ]
        ).cpu()
    )
    return fit_loss, penalty, objective


_MISSING_SURROGATE_CURVATURE = 1e-6
_OUTER_KKT_CHECK_EVERY = 4
_PERIODIC_CERTIFICATE_MAX_ITER = 96
_FULL_STEP_MAX_CURVATURE_ATTEMPTS = 24
_RECOVERY_STAGNATION_MIN_OUTER_ITER = 16
_RECOVERY_STAGNATION_AUDIT_WINDOW = 4
_RECOVERY_STAGNATION_MIN_RESIDUAL_GAIN = float(np.log(1.05))
_RECOVERY_STAGNATION_OBJECTIVE_GAIN_SCALE = 0.1
_RECOVERY_STAGNATION_REJECTED_STEPS = 2
_FINAL_CERTIFICATE_PROBE_MAX_ITER = 32
_FINAL_CERTIFICATE_DEEPEN_GATE_MULTIPLIER = 10.0
_MIN_INNER_ITERATIONS = 10
# A capped solve may exceed its soft limit only inside this mandatory terminal
# allowance.  Refinement is skipped fail-closed when its exact bound plus the
# authoritative terminal audit would cross the hard C+10 boundary.
_MANDATORY_TERMINAL_EDGE_PASS_ALLOWANCE = 10
_CONVEX_GLOBAL_OPTIMALITY_BASIS = "convex_fixed_linear_objective_plus_kkt"
OBJECTIVE_SHAPE_AUTO = "auto"
PATH_OBJECTIVE_SHAPE = "generic_nonconvex"


@dataclass(frozen=True, slots=True)
class _RecoveryProgressSnapshot:
    outer_iteration: int
    best_objective: float
    best_kkt_residual: float
    step_residual: float
    rejected_since_previous_audit: int
    dominant_component: str


@dataclass(slots=True)
class _RecoveryProgressMonitor:
    """Detect a recovery-only plateau without making a convergence claim."""

    tolerance: float
    certification_tolerance: float
    audit_patience: int = _RECOVERY_STAGNATION_AUDIT_WINDOW
    snapshots: list[_RecoveryProgressSnapshot] = field(default_factory=list)
    best_objective: float = float("inf")
    best_kkt_residual: float = float("inf")
    best_dominant_component: str = "stationarity"
    previous_rejected_outer_steps: int = 0

    def observe(
        self,
        *,
        outer_iteration: int,
        objective: float,
        diagnostics: KKTDiagnostics,
        step_residual: float,
        rejected_outer_steps: int,
    ) -> str | None:
        """Return a typed stagnation reason, never a certificate decision."""

        objective_value = float(objective)
        if np.isfinite(objective_value):
            self.best_objective = min(float(self.best_objective), objective_value)

        components = {
            "stationarity": float(diagnostics.backward_error_stationarity_residual),
            "edge_subgradient": float(
                diagnostics.backward_error_edge_subgradient_residual
            ),
            "dual_ball": float(diagnostics.backward_error_dual_ball_residual),
        }
        residual = float(diagnostics.backward_error_kkt_residual)
        if (
            np.isfinite(residual)
            and residual >= 0.0
            and residual < float(self.best_kkt_residual)
        ):
            self.best_kkt_residual = residual
            finite_components = {
                name: value
                for name, value in components.items()
                if np.isfinite(value) and value >= 0.0
            }
            if finite_components:
                self.best_dominant_component = max(
                    finite_components, key=finite_components.get
                )

        rejected_total = max(int(rejected_outer_steps), 0)
        rejected_increment = max(
            rejected_total - int(self.previous_rejected_outer_steps), 0
        )
        self.previous_rejected_outer_steps = rejected_total
        self.snapshots.append(
            _RecoveryProgressSnapshot(
                outer_iteration=int(outer_iteration),
                best_objective=float(self.best_objective),
                best_kkt_residual=float(self.best_kkt_residual),
                step_residual=float(step_residual),
                rejected_since_previous_audit=int(rejected_increment),
                dominant_component=str(self.best_dominant_component),
            )
        )
        patience = max(int(self.audit_patience), 1)
        if len(self.snapshots) > patience:
            del self.snapshots[:-patience]

        if (
            int(outer_iteration) < _RECOVERY_STAGNATION_MIN_OUTER_ITER
            or len(self.snapshots) < patience
        ):
            return None
        first = self.snapshots[0]
        last = self.snapshots[-1]
        gate = 5.0 * _validate_solver_tolerance(self.certification_tolerance)
        if not (
            np.isfinite(last.best_objective)
            and np.isfinite(first.best_objective)
            and np.isfinite(last.best_kkt_residual)
            and np.isfinite(first.best_kkt_residual)
            and last.best_kkt_residual > gate
        ):
            return None
        objective_gain = max(
            float(first.best_objective) - float(last.best_objective), 0.0
        ) / (1.0 + abs(float(first.best_objective)))
        residual_gain = float(
            np.log(
                max(float(first.best_kkt_residual), np.finfo(np.float64).tiny)
                / max(float(last.best_kkt_residual), np.finfo(np.float64).tiny)
            )
        )
        objective_stalled = bool(
            objective_gain
            < _RECOVERY_STAGNATION_OBJECTIVE_GAIN_SCALE
            * _validate_solver_tolerance(self.tolerance)
        )
        residual_stalled = bool(residual_gain < _RECOVERY_STAGNATION_MIN_RESIDUAL_GAIN)
        step_stalled = bool(
            np.isfinite(last.step_residual)
            and last.step_residual
            <= max(1e-8, float(np.sqrt(_validate_solver_tolerance(self.tolerance))))
        )
        rejected_in_window = sum(
            int(item.rejected_since_previous_audit) for item in self.snapshots
        )
        repeatedly_rejected = bool(
            rejected_in_window >= _RECOVERY_STAGNATION_REJECTED_STEPS
        )
        if not (
            objective_stalled
            and residual_stalled
            and (step_stalled or repeatedly_rejected)
        ):
            return None
        if repeatedly_rejected:
            return "solver_stagnation_rejected_mm_steps"
        return f"solver_stagnation_{last.dominant_component}"


@dataclass(frozen=True, slots=True)
class _MMState:
    """One accepted MM iterate and its trajectory-coupled solver state."""

    phi: torch.Tensor
    dual: torch.Tensor | None
    certificate: GraphFusionCertificate | None
    warm_state: DenseWarmState | PrimalOnlyWarmState
    terms: TorchObservedTerms
    penalty_tensor: torch.Tensor
    fit_loss: float
    objective: float
    inner_solver: str
    dual_start_is_actual: bool
    inner_converged: bool = False

    @property
    def penalty(self) -> float:
        return float(self.objective - self.fit_loss)


@dataclass(frozen=True, slots=True)
class _AuditedSolverSnapshot:
    """Self-consistent primal/dual state from one full observed KKT audit."""

    phi: torch.Tensor
    dual: torch.Tensor | None
    certificate: GraphFusionCertificate | None
    warm_state: DenseWarmState | PrimalOnlyWarmState
    inner_solver: str
    dual_start_is_actual: bool
    objective: float
    fit_loss: float
    diagnostics: KKTDiagnostics
    current_inner_converged: bool
    certified: bool

    @property
    def kkt_residual(self) -> float:
        return float(self.diagnostics.backward_error_kkt_residual)


def _snapshot_audited_solver_state(
    *,
    state: _MMState,
    graph_hash: str,
    lambda_value: float,
    diagnostics: KKTDiagnostics,
    certified: bool,
    reuse: _AuditedSolverSnapshot | None = None,
) -> _AuditedSolverSnapshot:
    """Detach a full audit, reusing incumbent buffers when shapes permit."""

    def retained_copy(
        source: torch.Tensor,
        destination: torch.Tensor | None,
    ) -> torch.Tensor:
        detached = source.detach()
        if (
            torch.is_tensor(destination)
            and tuple(destination.shape) == tuple(detached.shape)
            and destination.dtype == detached.dtype
            and destination.device == detached.device
        ):
            with torch.no_grad():
                destination.copy_(detached)
            return destination
        return detached.clone()

    phi_copy = retained_copy(state.phi, None if reuse is None else reuse.phi)
    if isinstance(state.certificate, DenseEdgeCertificate):
        reusable_dual = (
            reuse.dual
            if reuse is not None and isinstance(reuse.certificate, DenseEdgeCertificate)
            else None
        )
        dual_copy = retained_copy(state.certificate.dual, reusable_dual)
        certificate_copy: GraphFusionCertificate | None = replace(
            state.certificate,
            dual=dual_copy,
        )
        warm_state: DenseWarmState | PrimalOnlyWarmState = DenseWarmState(
            phi=phi_copy,
            dual=dual_copy,
            previous_lambda=float(lambda_value),
            graph_hash=str(graph_hash),
        )
        dual_start_is_actual = True
    elif isinstance(state.certificate, CompressedEdgeCertificate):
        reusable_certificate = (
            reuse.certificate
            if reuse is not None
            and isinstance(reuse.certificate, CompressedEdgeCertificate)
            else None
        )
        certificate_copy = replace(
            state.certificate,
            labels=retained_copy(
                state.certificate.labels,
                None if reusable_certificate is None else reusable_certificate.labels,
            ),
            centers=retained_copy(
                state.certificate.centers,
                None if reusable_certificate is None else reusable_certificate.centers,
            ),
            internal_edge_ids=retained_copy(
                state.certificate.internal_edge_ids,
                (
                    None
                    if reusable_certificate is None
                    else reusable_certificate.internal_edge_ids
                ),
            ),
            internal_dual=retained_copy(
                state.certificate.internal_dual,
                (
                    None
                    if reusable_certificate is None
                    else reusable_certificate.internal_dual
                ),
            ),
        )
        dual_copy = None
        warm_state = PrimalOnlyWarmState(
            phi=phi_copy,
            structure_hint=certificate_copy.labels,
            certificate_hint=certificate_copy,
        )
        dual_start_is_actual = False
    else:
        certificate_copy = None
        dual_copy = None
        warm_state = PrimalOnlyWarmState(phi=phi_copy)
        dual_start_is_actual = False
    return _AuditedSolverSnapshot(
        phi=phi_copy,
        dual=dual_copy,
        certificate=certificate_copy,
        warm_state=warm_state,
        inner_solver=str(state.inner_solver),
        dual_start_is_actual=bool(dual_start_is_actual),
        objective=float(state.objective),
        fit_loss=float(state.fit_loss),
        diagnostics=diagnostics,
        current_inner_converged=bool(state.inner_converged),
        certified=bool(certified),
    )


def _prefer_audited_solver_snapshot(
    candidate: _AuditedSolverSnapshot,
    incumbent: _AuditedSolverSnapshot | None,
) -> bool:
    """Rank audited recovery states without weakening certificate admission."""
    return _prefer_audited_solver_metadata(
        certified=bool(candidate.certified),
        objective=float(candidate.objective),
        diagnostics=candidate.diagnostics,
        incumbent=incumbent,
    )


def _prefer_audited_solver_metadata(
    *,
    certified: bool,
    objective: float,
    diagnostics: KKTDiagnostics,
    incumbent: _AuditedSolverSnapshot | None,
) -> bool:
    """Rank scalar audit metadata before any potentially dense state copy."""

    if incumbent is None:
        return True
    if bool(certified) != bool(incumbent.certified):
        return bool(certified)
    candidate_objective = float(objective)
    incumbent_objective = float(incumbent.objective)
    if certified:
        return (not np.isfinite(incumbent_objective)) or (
            np.isfinite(candidate_objective)
            and candidate_objective < incumbent_objective
        )
    candidate_residual = float(diagnostics.backward_error_kkt_residual)
    incumbent_residual = float(incumbent.kkt_residual)
    if np.isfinite(candidate_residual) != np.isfinite(incumbent_residual):
        return bool(np.isfinite(candidate_residual))
    if candidate_residual != incumbent_residual:
        return bool(candidate_residual < incumbent_residual)
    return (not np.isfinite(incumbent_objective)) or (
        np.isfinite(candidate_objective) and candidate_objective < incumbent_objective
    )


def _retain_audited_solver_state_if_better(
    *,
    incumbent: _AuditedSolverSnapshot | None,
    state: _MMState,
    graph_hash: str,
    lambda_value: float,
    diagnostics: KKTDiagnostics,
    certified: bool,
) -> _AuditedSolverSnapshot | None:
    """Rank cheap scalar metadata, materializing only an improving state."""

    if not _prefer_audited_solver_metadata(
        certified=certified,
        objective=state.objective,
        diagnostics=diagnostics,
        incumbent=incumbent,
    ):
        return incumbent
    return _snapshot_audited_solver_state(
        state=state,
        graph_hash=graph_hash,
        lambda_value=lambda_value,
        diagnostics=diagnostics,
        certified=certified,
        reuse=incumbent,
    )


def _remaining_edge_pass_budget(
    work: WorkCounters,
    limit: int | None,
) -> int | None:
    if limit is None:
        return None
    return max(int(limit) - int(work.edge_pass_equivalents), 0)


def _remaining_terminal_edge_pass_budget(
    work: WorkCounters,
    limit: int | None,
) -> int | None:
    """Return the hard C+10 terminal boundary rather than the soft cap C."""

    if limit is None:
        return None
    return max(
        int(limit)
        + _MANDATORY_TERMINAL_EDGE_PASS_ALLOWANCE
        - int(work.edge_pass_equivalents),
        0,
    )


def _terminal_float64_audit_edge_pass_bound(
    *,
    runtime_dtype: torch.dtype,
    certificate: GraphFusionCertificate | None,
    edge_count: int,
) -> int:
    """Worst-case conservative EPE owed after working-precision refinement."""

    if runtime_dtype == torch.float64 or int(edge_count) <= 0:
        return 0
    # One float64 objective penalty.  Dense witnesses need an adjoint, graph
    # forward, and edgewise KKT reduction; a breakpoint interval may need one
    # extra adjoint.
    if isinstance(certificate, DenseEdgeCertificate):
        return 5
    if isinstance(certificate, CompressedEdgeCertificate):
        # The compressed audit fuses its edge operations into one traversal
        # and does not support interval-dual adjustment.
        return 2
    # Refinement can materialize a dense witness from a missing witness before
    # this audit runs. Reserve that reachable dense/breakpoint worst case; the
    # witness-free three-pass audit applies only if refinement is skipped.
    return 5


def _inner_edge_pass_bound(
    iterations: int,
    *,
    use_alm: bool,
    spectral_rho: bool,
) -> int:
    """Conservative EPE bound for one low-level inner invocation."""

    count = max(int(iterations), 0)
    if count == 0:
        return 0
    audits = int(np.ceil(count / max(int(DEFAULT_INNER_KKT_CHECK_EVERY), 1)))
    audit_passes = 2 if use_alm else 3
    return (
        2 * count
        + audit_passes * audits
        + (1 if use_alm else 0)
        + (count // 10 if use_alm and spectral_rho else 0)
    )


def _budgeted_inner_max_iter(
    requested: int,
    *,
    remaining: int | None,
    use_alm: bool,
    spectral_rho: bool,
) -> int | None:
    desired = max(int(requested), _MIN_INNER_ITERATIONS)
    if remaining is None:
        return desired
    available = max(int(remaining), 0)
    if (
        _inner_edge_pass_bound(
            _MIN_INNER_ITERATIONS,
            use_alm=use_alm,
            spectral_rho=spectral_rho,
        )
        > available
    ):
        return None
    lower = _MIN_INNER_ITERATIONS
    upper = desired
    while lower < upper:
        midpoint = (lower + upper + 1) // 2
        if (
            _inner_edge_pass_bound(
                midpoint,
                use_alm=use_alm,
                spectral_rho=spectral_rho,
            )
            <= available
        ):
            lower = midpoint
        else:
            upper = midpoint - 1
    return lower


def _budgeted_certificate_parameters(
    *,
    certificate: GraphFusionCertificate | None,
    options: CertificateOptions,
    requested_max_iter: int,
    remaining: int | None,
    mandatory: bool,
) -> tuple[int, CertificateOptions] | None:
    """Bound one refinement by remaining EPE using a conservative cost model."""

    desired_iter = max(int(requested_max_iter), 1)
    if remaining is None:
        return desired_iter, replace(options, max_iter=desired_iter)
    available = max(int(remaining), 0)
    compressed = isinstance(certificate, CompressedEdgeCertificate)
    desired_expansions = int(options.max_expansions) if compressed else 1

    def bound(iterations: int, expansions: int) -> int:
        if not compressed:
            # The streamed dense-certificate backend is the worst case: an
            # incoming audit, analytic construction, analytic audit, then six
            # complete-edge primitives per streamed projected-dual iteration.
            # Edge activity is collected inside the incoming audit, making
            # this the dense backend's conservative worst case.
            return 6 * int(iterations) + 7
        # Each expansion can consume max_iter projected-dual iterations, one
        # missing-column scan, and one terminal full-graph diagnostic.  The
        # inherited-certificate fast path contributes one further audit.
        return 2 * int(expansions) * int(iterations) + 4 * int(expansions) + 2

    if bound(desired_iter, desired_expansions) <= available:
        return desired_iter, replace(options, max_iter=desired_iter)
    for expansions in range(desired_expansions, 0, -1):
        if compressed:
            residual = available - (4 * expansions + 2)
            divisor = 2 * expansions
        else:
            residual = available - 7
            divisor = 6
        iterations = min(desired_iter, residual // divisor)
        if iterations >= 1 and bound(iterations, expansions) <= available:
            return int(iterations), replace(
                options,
                max_iter=int(iterations),
                max_expansions=int(expansions),
            )
    # The caller substitutes a non-refining fail-closed audit when even one
    # refinement iteration would cross the hard C+10 terminal boundary.
    return None


def _certificate_probe_action(
    *,
    residual_before: float,
    probe_residual: float,
    certification_tolerance: float,
    recovery_stagnated: bool,
) -> str:
    """Choose whether a terminal certificate probe deserves deeper work."""

    gate = 5.0 * _validate_solver_tolerance(certification_tolerance)
    after = float(probe_residual)
    if np.isfinite(after) and after >= 0.0 and after <= gate:
        return "certified"
    if not recovery_stagnated:
        return "deepen"
    before = float(residual_before)
    if not (
        np.isfinite(before)
        and before >= 0.0
        and np.isfinite(after)
        and after > _FINAL_CERTIFICATE_DEEPEN_GATE_MULTIPLIER * gate
    ):
        return "deepen"
    gain = float(
        np.log(
            max(before, np.finfo(np.float64).tiny)
            / max(after, np.finfo(np.float64).tiny)
        )
    )
    return "plateau" if gain < _RECOVERY_STAGNATION_MIN_RESIDUAL_GAIN else "deepen"


def _major_prior_for_model(model: ObservedModel) -> float:
    """Return the prior encoded in the hashed model for fast scalar formulas."""

    encoded = model.binary_linear_mixture_prior
    return 0.5 if encoded is None else float(encoded)


def objective_shape_for_data(data: TumorData, requested: str) -> str:
    """Return the only solver shape declaration valid for this likelihood.

    Competing or genuinely piecewise paths can be multimodal and therefore
    always use the generic route. A path specification whose valid candidates
    all reduce to the same fixed linear emission reuses the existing scalar
    route without discarding its path provenance.
    """

    model = compile_observed_model(data, major_prior=0.5, eps=1e-6)
    return objective_shape_for_model(model, requested)


def objective_shape_for_model(model: ObservedModel, requested: str) -> str:
    """Resolve solver shape from canonical emission structure alone."""

    normalized = _normalize_objective_shape(requested)
    if model.requires_generic_path_solver:
        return PATH_OBJECTIVE_SHAPE
    return "unimodal" if normalized == OBJECTIVE_SHAPE_AUTO else normalized


def _path_smooth_interval_bounds(
    model: TorchObservedModel,
    phi: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict one MM trial to the current smooth path interval.

    The path kernel returns the left derivative at an exact breakpoint, so an
    exact breakpoint is treated as the upper end of its left interval.  The
    nonconvex start bank separately seeds both sides of nearby occupancy
    switches.
    """

    points, valid = observed_internal_breakpoints_torch(
        model,
        eps=float(eps),
    )
    valid = (
        valid
        & torch.isfinite(points)
        & (points > lower.unsqueeze(-1))
        & (points < upper.unsqueeze(-1))
    )
    # Ordering, rather than an approximate equality test, is essential here:
    # a point infinitesimally to the right of a kink is still on the right
    # smooth branch.  Projection onto a breakpoint reuses the exact tensor
    # value, so true breakpoint iterates compare equal.
    below = valid & (points < phi.unsqueeze(-1))
    above = valid & ~below
    local_lower = torch.max(
        torch.where(
            below,
            points,
            lower.unsqueeze(-1).expand_as(points),
        ),
        dim=-1,
    ).values
    local_upper = torch.min(
        torch.where(
            above,
            points,
            upper.unsqueeze(-1).expand_as(points),
        ),
        dim=-1,
    ).values
    local_lower = torch.minimum(local_lower, phi)
    local_upper = torch.maximum(local_upper, phi)
    return local_lower, local_upper


def _safe_surrogate_curvature_and_gradient(
    surrogate_terms: TorchObservedTerms,
    likelihood_included: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_base = torch.clamp(
        surrogate_terms.hessian_upper,
        min=_MISSING_SURROGATE_CURVATURE,
    )
    surrogate_grad = surrogate_terms.gradient
    if likelihood_included is None:
        return h_base, surrogate_grad

    included = likelihood_included
    h_base = torch.where(
        included,
        h_base,
        torch.full_like(h_base, _MISSING_SURROGATE_CURVATURE),
    )
    surrogate_grad = torch.where(
        included, surrogate_grad, torch.zeros_like(surrogate_grad)
    )
    return h_base, surrogate_grad


def _safe_majorized_center(
    phi: torch.Tensor,
    *,
    surrogate_grad: torch.Tensor,
    h: torch.Tensor,
    likelihood_included: torch.Tensor | None,
) -> torch.Tensor:
    U_raw = phi - surrogate_grad / h
    if likelihood_included is None:
        return U_raw
    return torch.where(likelihood_included, U_raw, phi)


def _validate_solver_tolerance(tol: float) -> float:
    value = float(tol)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("Solver tolerance must be a positive finite value.")
    return value


def _prefer_multistart_fit(
    candidate: RawFit,
    incumbent: RawFit,
) -> bool:
    """Rank finite starts by the observed objective they all optimize.

    Certification remains mandatory downstream. It cannot change the target
    function here: if the best observed-objective basin is unfinished, the
    candidate must enter same-lambda recovery or fail closed rather than be
    replaced by a materially worse stationary basin.
    """

    candidate_finite = bool(np.isfinite(candidate.objective.total))
    incumbent_finite = bool(np.isfinite(incumbent.objective.total))
    if candidate_finite != incumbent_finite:
        return candidate_finite
    if candidate_finite:
        objective_delta = float(candidate.objective.total) - float(
            incumbent.objective.total
        )
        if abs(objective_delta) > 1e-8:
            return bool(objective_delta < 0.0)
    candidate_status = (
        bool(candidate.certificate.admissible),
        bool(candidate.convergence.converged),
    )
    incumbent_status = (
        bool(incumbent.certificate.admissible),
        bool(incumbent.convergence.converged),
    )
    return candidate_status > incumbent_status


def _normalize_objective_shape(objective_shape: str) -> str:
    normalized = str(objective_shape).strip().lower()
    if normalized not in {
        OBJECTIVE_SHAPE_AUTO,
        "unimodal",
        "unimodal_full_step_backtracking",
        "generic_nonconvex",
    }:
        raise ValueError(
            "objective_shape must be 'auto', 'unimodal', "
            "'unimodal_full_step_backtracking', or 'generic_nonconvex'."
        )
    return normalized


def _validate_prebuilt_tensor_graph(
    graph: PairwiseFusionGraph,
    tensor_graph: TensorFusionGraph,
    *,
    runtime,
    num_nodes: int,
) -> TensorFusionGraph:
    """Validate cheap invariants for an already paired host/device graph."""

    if int(tensor_graph.num_nodes) != int(num_nodes):
        raise ValueError("prebuilt_tensor_graph has the wrong number of nodes.")
    if str(tensor_graph.name) != str(graph.name):
        raise ValueError("prebuilt_tensor_graph name does not match graph.")
    if tensor_graph.edge_index.ndim != 2 or int(tensor_graph.edge_index.shape[0]) != 2:
        raise ValueError("prebuilt_tensor_graph edge_index must have shape (2, E).")
    edge_count = int(tensor_graph.edge_index.shape[1])
    if edge_count != int(np.asarray(graph.edge_u).size):
        raise ValueError("prebuilt_tensor_graph edge count does not match graph.")
    if tensor_graph.weight.ndim != 1 or int(tensor_graph.weight.numel()) != edge_count:
        raise ValueError("prebuilt_tensor_graph weights must have shape (E,).")
    if tensor_graph.edge_index.dtype != torch.long:
        raise ValueError("prebuilt_tensor_graph edge indices must use torch.long.")
    if tensor_graph.weight.dtype != runtime.dtype:
        raise ValueError(
            "prebuilt_tensor_graph weights do not match the runtime dtype."
        )
    if tuple(tensor_graph.degree.shape) != (int(num_nodes),):
        raise ValueError("prebuilt_tensor_graph degree has the wrong shape.")
    if tuple(tensor_graph.pdhg_tau_node.shape) != (int(num_nodes), 1):
        raise ValueError(
            "prebuilt_tensor_graph PDHG preconditioner has the wrong shape."
        )
    for value in (
        tensor_graph.edge_index,
        tensor_graph.weight,
        tensor_graph.degree,
        tensor_graph.pdhg_tau_node,
    ):
        if value.device.type != runtime.device.type or (
            runtime.device.index is not None
            and value.device.index != runtime.device.index
        ):
            raise ValueError("prebuilt_tensor_graph is not on the runtime device.")
    source_fingerprint = str(tensor_graph.source_graph_fingerprint)
    if source_fingerprint:
        if source_fingerprint != graph.fingerprint:
            raise ValueError("prebuilt_tensor_graph content does not match graph.")
        return tensor_graph
    # The host graph is the authority for objective/certificate hashes.  Exact
    # equality prevents an unrelated device graph with the same name and edge
    # count from being run under false provenance.  This is a D2H validation,
    # not an H2D re-upload; guided construction already materializes this host
    # spec for stable outputs.
    tensor_edge_u = (
        tensor_graph.edge_u.detach().cpu().numpy().astype(np.int64, copy=False)
    )
    tensor_edge_v = (
        tensor_graph.edge_v.detach().cpu().numpy().astype(np.int64, copy=False)
    )
    tensor_weight_native = tensor_graph.weight.detach().cpu().numpy()
    if not np.array_equal(tensor_edge_u, np.asarray(graph.edge_u, dtype=np.int64)):
        raise ValueError("prebuilt_tensor_graph edge_u does not match graph.")
    if not np.array_equal(tensor_edge_v, np.asarray(graph.edge_v, dtype=np.int64)):
        raise ValueError("prebuilt_tensor_graph edge_v does not match graph.")
    # The tensor values are the numeric objective actually optimized.  Compare
    # the host specification after applying the same runtime-dtype rounding;
    # comparing rounded float32 values with their float64 source bit-for-bit
    # incorrectly rejects a correctly paired graph.
    expected_weight = np.asarray(graph.edge_w, dtype=tensor_weight_native.dtype)
    if not np.array_equal(tensor_weight_native, expected_weight):
        raise ValueError("prebuilt_tensor_graph weights do not match graph.")
    return replace(
        tensor_graph,
        source_graph_fingerprint=graph.fingerprint,
    )


def _project_state_dual(
    state: SolverState | None,
    *,
    runtime,
    edge_w: torch.Tensor,
    lambda_value: float,
    num_edges: int,
    num_regions: int,
) -> torch.Tensor | None:
    if state is None or state.dual is None:
        return None
    if tuple(state.dual.shape) != (int(num_edges), int(num_regions)):
        return None
    dual = state.dual.to(dtype=runtime.dtype, device=runtime.device)
    if int(num_edges) == 0:
        return torch.zeros(
            (0, int(num_regions)), dtype=runtime.dtype, device=runtime.device
        )
    radius = float(lambda_value) * edge_w.to(dtype=runtime.dtype, device=runtime.device)
    return project_dual_ball(dual, radius)


def _invalidate_damped_trial_state(
    *,
    phi: torch.Tensor,
    trial_warm_state: DenseWarmState | PrimalOnlyWarmState,
) -> PrimalOnlyWarmState:
    """Create the primal-only warm state for a damped MM endpoint."""

    structure_hint = None
    if isinstance(trial_warm_state, DenseWarmState) and torch.is_tensor(
        trial_warm_state.dual
    ):
        certificate_hint = DenseEdgeCertificate(
            dual=trial_warm_state.dual,
            graph_hash=trial_warm_state.graph_hash,
            gradient_scope="mm_surrogate",
        )
    elif isinstance(trial_warm_state, PrimalOnlyWarmState):
        certificate_hint = trial_warm_state.certificate_hint
    else:
        certificate_hint = None
    return PrimalOnlyWarmState(
        phi=phi,
        structure_hint=structure_hint,
        certificate_hint=certificate_hint,
    )


def _compressed_certificate_for_primal(
    phi: torch.Tensor,
    *,
    graph_hash: str,
    gradient_scope: str,
) -> CompressedEdgeCertificate:
    """Represent exact-equal primal rows without materializing a full dual."""

    _, labels = torch.unique(phi, dim=0, sorted=True, return_inverse=True)
    num_blocks = int(torch.max(labels).item()) + 1 if labels.numel() else 0
    roots = torch.full(
        (num_blocks,),
        int(phi.shape[0]),
        dtype=torch.long,
        device=phi.device,
    )
    if labels.numel():
        nodes = torch.arange(int(labels.numel()), device=phi.device)
        roots.scatter_reduce_(0, labels, nodes, reduce="amin", include_self=True)
    return CompressedEdgeCertificate(
        labels=labels,
        centers=phi.index_select(0, roots),
        internal_edge_ids=torch.empty(0, dtype=torch.long, device=phi.device),
        internal_dual=torch.empty(
            (0, int(phi.shape[1])), dtype=phi.dtype, device=phi.device
        ),
        graph_hash=str(graph_hash),
        gradient_scope=gradient_scope,
    )


def _rebase_certificate_hint(
    hint: GraphFusionCertificate | None,
    *,
    phi: torch.Tensor,
    graph: TensorFusionGraph,
    graph_hash: str,
    lambda_value: float,
) -> GraphFusionCertificate | None:
    """Make a non-authoritative warm hint structurally valid at ``phi``."""

    if hint is None or hint.graph_hash != str(graph_hash):
        return None
    if isinstance(hint, DenseEdgeCertificate):
        return hint
    rebased = _compressed_certificate_for_primal(
        phi,
        graph_hash=graph_hash,
        gradient_scope="observed_objective",
    )
    edge_ids = hint.internal_edge_ids.to(device=phi.device, dtype=torch.long)
    dual = hint.internal_dual.to(device=phi.device, dtype=phi.dtype)
    num_edges = int(graph.edge_u.numel())
    if (
        edge_ids.ndim != 1
        or tuple(dual.shape) != (int(edge_ids.numel()), int(phi.shape[1]))
        or bool(torch.any((edge_ids < 0) | (edge_ids >= num_edges)).item())
    ):
        return rebased
    if edge_ids.numel():
        edge_u = graph.edge_u.index_select(0, edge_ids)
        edge_v = graph.edge_v.index_select(0, edge_ids)
        internal = rebased.labels.index_select(
            0, edge_u
        ) == rebased.labels.index_select(0, edge_v)
        edge_ids = edge_ids[internal]
        dual = dual[internal]
        if edge_ids.numel():
            radius = float(lambda_value) * graph.weight.index_select(0, edge_ids)
            dual = project_dual_ball(dual, radius)
    return CompressedEdgeCertificate(
        labels=rebased.labels,
        centers=rebased.centers,
        internal_edge_ids=edge_ids,
        internal_dual=dual,
        graph_hash=graph_hash,
        gradient_scope="observed_objective",
    )


def promote_solver_context_dtype(
    context: SolverContext,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
    start_override: np.ndarray | torch.Tensor | None = None,
) -> SolverContext:
    """Rebuild one frozen objective from its immutable host sources."""

    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("Promoted solver contexts require float32 or float64.")
    target_device = context.runtime.device if device is None else torch.device(device)
    if (
        context.runtime.dtype == dtype
        and context.runtime.device == target_device
        and start_override is None
    ):
        return context
    source_model = context.source_model
    runtime = replace(
        context.runtime,
        dtype=dtype,
        device=target_device,
        device_name=str(target_device),
    )
    graph = tensorize_graph(
        context.graph_spec,
        runtime,
        num_nodes=int(source_model.shape[0]),
    )
    override = (
        None
        if start_override is None
        else as_runtime_tensor(start_override, runtime).detach()
    )
    if override is None:
        exact = context.exact_pilot.to(dtype=dtype, device=target_device)
        pooled = context.pooled_start.to(dtype=dtype, device=target_device)
        wells = tuple(
            start.to(dtype=dtype, device=target_device)
            for start in context.scalar_well_starts
        )
    else:
        exact = pooled = override
        wells = ()
    return replace(
        context,
        observed_model=model_to_torch(source_model, runtime),
        graph=graph,
        exact_pilot=exact,
        pooled_start=pooled,
        scalar_well_starts=wells,
        runtime=runtime,
    )


def _require_dense_memory(
    data: TumorData,
    runtime: TorchRuntime,
    *,
    operation: str,
    limit_name: str,
    cause: BaseException | None = None,
) -> None:
    fits, required, limit = dense_complete_solver_memory_preflight(
        num_nodes=data.num_mutations,
        num_regions=data.num_regions,
        runtime=runtime,
    )
    if fits:
        return
    error = ExactSolverResourceLimit(
        f"exact_solver_resource_limit: {operation} needs approximately "
        f"{required} bytes (available {limit_name}: {limit})."
    )
    if cause is not None:
        raise error from cause
    raise error


def _float64_context(
    data: TumorData,
    context: SolverContext,
    *,
    device: torch.device | None = None,
    cause: BaseException | None = None,
) -> SolverContext:
    target = context.runtime.device if device is None else torch.device(device)
    runtime = replace(
        context.runtime,
        dtype=torch.float64,
        device=target,
        device_name=str(target),
    )
    if context.graph.is_complete:
        prefix = (
            "CPU " if target.type == "cpu" and target != context.runtime.device else ""
        )
        _require_dense_memory(
            data,
            runtime,
            operation=f"{prefix}float64 fixed-objective precision polish",
            limit_name="policy limit",
            cause=cause,
        )
    return promote_solver_context_dtype(context, dtype=torch.float64, device=target)


def _finalize_precision_polish(
    polished: RawFit,
    working: RawFit,
    source_context: SolverContext,
    *,
    on_cpu: bool,
) -> RawFit:
    if (
        polished.provenance.objective_spec_hash != source_context.objective_spec_hash
        or polished.provenance.original_graph_hash != source_context.graph_hash
    ):
        raise AssertionError("Precision polishing changed estimator identity.")
    objective = working.objective.total
    # The working objective carries the working dtype's evaluation error, so
    # the non-increase check must use that precision's scale, not float64's.
    working_eps = (
        float(np.finfo(np.float32).eps)
        if str(working.provenance.dtype) == "float32"
        else float(np.finfo(np.float64).eps)
    )
    slack = max(
        1e-10 * (1.0 + abs(objective)),
        64.0 * working_eps * (1.0 + abs(objective)),
    )
    if (
        not np.isfinite(polished.objective.total)
        or polished.objective.total > objective + slack
    ):
        raise AssertionError(
            "Float64 fixed-objective polishing increased the objective."
        )
    delta = float(
        np.max(
            np.abs(
                np.asarray(polished.phi, dtype=np.float64)
                - np.asarray(working.phi, dtype=np.float64)
            )
        )
    )
    reason = "float64_fixed_objective_precision_polish"
    backend = None
    if on_cpu:
        reason += ";float64_precision_polish_cpu_after_cuda_resource_limit"
        backend = "admm_complete_graph_cpu_precision_polish"
    polished = record_attempt(
        polished,
        attempted=working,
        reason=reason,
        backend_name=backend,
    )
    return replace(
        polished,
        certificate=replace(
            polished.certificate,
            working_residual=working.certificate.working_residual,
            working_dtype=working.provenance.dtype,
            audit_dtype="float64",
            precision_polished=True,
            precision_polish_delta=delta,
        ),
    )


def escape_path_breakpoint_solver_state(
    state: SolverState | None,
    *,
    context: SolverContext,
    tol: float,
) -> tuple[SolverState | None, int, WorkCounters]:
    """Nudge a failed dense-certificate state off exact path breakpoints.

    The dense certificate supplies the fusion adjoint.  At each exact
    breakpoint, the one-sided observed gradients plus that adjoint choose a
    retry side.  A changed primal invalidates every dual/certificate warm
    object because none remains valid at the new point.
    """

    certificate = None if state is None else state.certificate
    model = context.source_model
    if (
        state is None
        or not model.has_internal_switches
        or not isinstance(certificate, DenseEdgeCertificate)
        or certificate.certificate_scope != "full_original_graph"
        or certificate.gradient_scope == "mm_surrogate"
        or certificate.graph_hash != str(context.graph_hash)
        or not torch.is_tensor(certificate.dual)
    ):
        return state, 0, WorkCounters()

    tolerance = _validate_solver_tolerance(tol)
    phi = as_runtime_tensor(state.phi, context.runtime)
    expected_shape = model.shape
    if tuple(phi.shape) != expected_shape or not bool(torch.all(torch.isfinite(phi))):
        return state, 0, WorkCounters()
    dual = as_runtime_tensor(certificate.dual, context.runtime)
    expected_dual_shape = (
        int(context.graph.edge_u.numel()),
        int(phi.shape[1]),
    )
    if tuple(dual.shape) != expected_dual_shape or not bool(
        torch.all(torch.isfinite(dual))
    ):
        return state, 0, WorkCounters()

    with torch.no_grad():
        fusion_adjustment = graph_adjoint_edges(
            dual,
            edge_u=context.graph.edge_u,
            edge_v=context.graph.edge_v,
            num_nodes=int(phi.shape[0]),
        )
        work_ledger = WorkLedger()
        work_ledger.charge_edge_passes(
            edge_count=int(context.graph.edge_u.numel()),
            num_regions=int(phi.shape[1]),
        )
        work = work_ledger.total
        gradient_left, gradient_right, at_breakpoint = (
            observed_one_sided_gradients_torch(
                context.observed_model,
                phi,
                eps=float(context.eps),
            )
        )
        left_total = gradient_left + fusion_adjustment
        right_total = gradient_right + fusion_adjustment
        dtype_epsilon = float(torch.finfo(phi.dtype).eps)
        numerical_threshold = (
            64.0
            * dtype_epsilon
            * (
                1.0
                + torch.abs(gradient_left)
                + torch.abs(gradient_right)
                + torch.abs(fusion_adjustment)
            )
        )
        direction_threshold = torch.maximum(
            torch.full_like(phi, 1e-3 * tolerance),
            numerical_threshold,
        )
        left_descends = (
            at_breakpoint
            & (phi > context.observed_model.lower)
            & (left_total > direction_threshold)
        )
        right_descends = (
            at_breakpoint
            & (phi < context.observed_model.upper)
            & (right_total < -direction_threshold)
        )
        choose_right = right_descends & (~left_descends | (-right_total >= left_total))
        choose_left = left_descends & ~choose_right
        base_offset = max(
            10.0 * float(context.eps),
            0.2 * tolerance,
        )
        offset = torch.maximum(
            torch.full_like(phi, base_offset),
            64.0 * dtype_epsilon * (1.0 + torch.abs(phi)),
        )
        escaped = torch.where(
            choose_right,
            phi + offset,
            torch.where(
                choose_left,
                phi - offset,
                phi,
            ),
        )
        escaped = torch.minimum(
            torch.maximum(escaped, context.observed_model.lower),
            context.observed_model.upper,
        )
        changed = escaped != phi
        changed_count = int(torch.count_nonzero(changed).item())
        if changed_count == 0:
            return state, 0, work

    return (
        replace(
            state,
            phi=escaped.detach(),
            dual=None,
            warm_state=None,
            certificate=None,
        ),
        changed_count,
        work,
    )


def prepare_torch_problem(
    data: TumorData,
    *,
    major_prior: float,
    eps: float,
    tol: float,
    inner_max_iter: int,
    graph: PairwiseFusionGraph | None = None,
    prebuilt_tensor_graph: TensorFusionGraph | None = None,
    adaptive_weight_gamma: float = 1.0,
    adaptive_weight_floor: float = 1e-6,
    adaptive_weight_baseline: float = 1.0,
    exact_pilot: np.ndarray | torch.Tensor | None = None,
    pooled_start: np.ndarray | torch.Tensor | None = None,
    scalar_well_starts: list[np.ndarray | torch.Tensor]
    | tuple[np.ndarray | torch.Tensor, ...]
    | None = None,
    device: str | None = DEFAULT_DEVICE,
    dtype: str | None = DEFAULT_DTYPE,
    runtime=None,
    objective_shape: str = OBJECTIVE_SHAPE_AUTO,
    defer_graph: bool = False,
) -> SolverContext:
    tol = _validate_solver_tolerance(tol)
    effective_runtime = (
        resolve_runtime(device, dtype=dtype) if runtime is None else runtime
    )
    source_model = compile_observed_model(
        data,
        major_prior=float(major_prior),
        eps=float(eps),
    )
    major_prior = _major_prior_for_model(source_model)
    phi_initialization = default_phi_initialization(source_model, eps=float(eps))
    objective_shape = objective_shape_for_model(source_model, objective_shape)
    use_unimodal_objective = objective_shape.startswith("unimodal")
    observed_model = model_to_torch(source_model, effective_runtime)
    data_fingerprint = tumor_data_fingerprint(data)

    if exact_pilot is None:
        exact_pilot_tensor, secondary_wells, valid_secondary = (
            compute_scalar_mutation_region_wells_torch(
                observed_model,
                source_model,
                phi_init=phi_initialization,
                major_prior=float(major_prior),
                eps=float(eps),
                tol=tol,
                max_iter=max(int(inner_max_iter), 16),
            )
        )
    else:
        exact_pilot_tensor = as_runtime_tensor(exact_pilot, effective_runtime)
        if scalar_well_starts is None and not use_unimodal_objective:
            _, secondary_wells, valid_secondary = (
                compute_scalar_mutation_region_wells_torch(
                    observed_model,
                    source_model,
                    phi_init=phi_initialization,
                    major_prior=float(major_prior),
                    eps=float(eps),
                    tol=tol,
                    max_iter=max(int(inner_max_iter), 16),
                )
            )
        else:
            secondary_wells = None
            valid_secondary = None

    if defer_graph:
        if graph is not None or prebuilt_tensor_graph is not None:
            raise ValueError(
                "defer_graph=True does not accept a resolved or prebuilt graph."
            )
        # Guided selection needs the likelihood pilot, bounds, and Torch data
        # before its observed-curvature graph is known.  Do not build and copy
        # an O(M^2) adaptive graph that would be discarded immediately.
        effective_graph = PairwiseFusionGraph(
            edge_u=np.zeros((0,), dtype=np.int32),
            edge_v=np.zeros((0,), dtype=np.int32),
            edge_w=np.zeros((0,), dtype=np.float64),
            name="deferred_likelihood_pilot",
            degree_bound=1,
        )
        tensor_graph = tensorize_graph(
            effective_graph,
            effective_runtime,
            num_nodes=data.num_mutations,
        )
    elif prebuilt_tensor_graph is not None:
        if graph is None:
            raise ValueError("prebuilt_tensor_graph requires its host graph spec.")
        effective_graph = resolve_pairwise_fusion_graph(
            data.num_mutations,
            graph=graph,
            pilot_phi=None,
            gamma=float(adaptive_weight_gamma),
            tau=max(float(adaptive_weight_floor), float(eps)),
            baseline=float(adaptive_weight_baseline),
        )
        tensor_graph = _validate_prebuilt_tensor_graph(
            effective_graph,
            prebuilt_tensor_graph,
            runtime=effective_runtime,
            num_nodes=data.num_mutations,
        )
    elif graph is None:
        working_tensor_graph = build_complete_adaptive_tensor_graph(
            exact_pilot_tensor,
            effective_runtime,
            likelihood_included=observed_model.observed,
            gamma=float(adaptive_weight_gamma),
            tau=max(float(adaptive_weight_floor), float(eps)),
            baseline=float(adaptive_weight_baseline),
        )
        # Adaptive construction intentionally follows working-runtime
        # arithmetic. Freeze that one result as the immutable host source, then
        # recreate every runtime view from it.
        effective_graph = tensor_graph_to_pairwise_graph(working_tensor_graph)
        tensor_graph = tensorize_graph(
            effective_graph,
            effective_runtime,
            num_nodes=data.num_mutations,
        )
    else:
        effective_graph = resolve_pairwise_fusion_graph(
            data.num_mutations,
            graph=graph,
            pilot_phi=None,
            gamma=float(adaptive_weight_gamma),
            tau=max(float(adaptive_weight_floor), float(eps)),
            baseline=float(adaptive_weight_baseline),
        )
        tensor_graph = tensorize_graph(
            effective_graph, effective_runtime, num_nodes=data.num_mutations
        )

    if use_unimodal_objective and pooled_start is None:
        pooled_start_tensor = exact_pilot_tensor
    elif pooled_start is None:
        pooled_start_tensor = compute_pooled_observed_data_start_torch(
            observed_model,
            source_model,
            major_prior=float(major_prior),
            eps=float(eps),
            tol=tol,
            max_iter=max(int(inner_max_iter), 16),
            beta_hints=exact_pilot_tensor,
        )
    else:
        pooled_start_tensor = as_runtime_tensor(pooled_start, effective_runtime)

    if use_unimodal_objective and scalar_well_starts is None:
        scalar_well_starts_seq = ()
    elif scalar_well_starts is None:
        scalar_well_starts_seq = compute_scalar_well_start_bank_torch(
            observed_model,
            eps=float(eps),
            exact_pilot=exact_pilot_tensor,
            secondary_wells=secondary_wells,
            valid_secondary=valid_secondary,
        )
    else:
        scalar_well_starts_seq = list(scalar_well_starts)

    graph_hash = effective_graph.fingerprint
    base_objective_key = make_base_objective_key(
        source_model,
        graph_hash=graph_hash,
        eps=float(eps),
        lower=source_model.lower,
        upper=source_model.upper,
    )
    return SolverContext(
        source_model=source_model,
        observed_model=observed_model,
        eps=float(eps),
        graph=tensor_graph,
        graph_spec=effective_graph,
        exact_pilot=exact_pilot_tensor,
        pooled_start=pooled_start_tensor,
        scalar_well_starts=tuple(
            as_runtime_tensor(start, effective_runtime)
            for start in scalar_well_starts_seq
        ),
        runtime=effective_runtime,
        data_fingerprint=data_fingerprint,
        base_objective_key=base_objective_key,
    )


def prepare_torch_problem_with_resource_policy(
    data: TumorData,
    *,
    dense_fallback_policy: str,
    inherited_resource_fallback: str | None = None,
    **prepare_kwargs,
) -> SolverContext:
    """Prepare an immutable context under the same typed fallback policy as fits."""
    normalized_policy = normalize_dense_fallback_policy(dense_fallback_policy)
    kwargs = dict(prepare_kwargs)
    supplied_prebuilt_tensor_graph = kwargs.pop("prebuilt_tensor_graph", None)
    supplied_runtime = kwargs.pop("runtime", None)
    requested_device = kwargs.pop("device", "cuda")
    requested_dtype = kwargs.pop("dtype", "float64")
    resolved_by_cpu_fallback = False
    requested_runtime = supplied_runtime
    try:
        if requested_runtime is None:
            requested_runtime = resolve_runtime(requested_device, dtype=requested_dtype)
    except CudaUnavailableError:
        if normalized_policy != "cpu_allowed":
            raise
        try:
            requested_runtime = resolve_runtime("cpu", dtype=requested_dtype)
        except RuntimeError as exc:
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: the requested runtime is unavailable "
                "and dense CPU fallback does not support the requested dtype."
            ) from exc
        resolved_by_cpu_fallback = True

    def prepare_on_runtime() -> SolverContext:
        reusable_tensor_graph = supplied_prebuilt_tensor_graph
        graph_runtime = (
            None
            if reusable_tensor_graph is None
            else (
                reusable_tensor_graph.weight.device,
                reusable_tensor_graph.weight.dtype,
            )
        )
        if graph_runtime != (requested_runtime.device, requested_runtime.dtype):
            reusable_tensor_graph = None
        context = prepare_torch_problem(
            data,
            device=requested_runtime.device_name,
            dtype=dtype_name(requested_runtime.dtype),
            runtime=requested_runtime,
            prebuilt_tensor_graph=reusable_tensor_graph,
            **kwargs,
        )
        fallback = (
            "dense_cpu" if resolved_by_cpu_fallback else inherited_resource_fallback
        )
        return replace(context, resource_fallback=fallback)

    while True:
        if resolved_by_cpu_fallback:
            _require_dense_memory(
                data,
                runtime=requested_runtime,
                operation="dense CPU fallback",
                limit_name="host limit",
            )
        try:
            return prepare_on_runtime()
        except (MemoryError, torch.OutOfMemoryError) as exc:
            action = decide_next_action(
                PolicyState(
                    phase="working",
                    resource_error=exc,
                    runtime_device_type=requested_runtime.device.type,
                    fallback_policy=normalized_policy,
                )
            )
            if action is not NextAction.CPU_FALLBACK:
                raise ExactSolverResourceLimit(
                    "exact_solver_resource_limit: exact problem or graph "
                    f"construction exhausted memory on "
                    f"{requested_runtime.device_name}."
                ) from exc
            try:
                requested_runtime = resolve_runtime(
                    "cpu", dtype=dtype_name(requested_runtime.dtype)
                )
            except RuntimeError as cpu_exc:
                raise ExactSolverResourceLimit(
                    "exact_solver_resource_limit: dense CPU fallback does not "
                    f"support dtype {dtype_name(requested_runtime.dtype)}."
                ) from cpu_exc
            resolved_by_cpu_fallback = True


def _initial_outer_diag() -> KKTDiagnostics:
    """Fail-closed residuals used until the first outer KKT audit."""
    return KKTDiagnostics.infinite()


def _backward_error_kkt_within_gate(
    diagnostics: KKTDiagnostics,
    *,
    certification_tol: float,
) -> bool:
    """Apply the immutable full-KKT gate to the componentwise residual.

    The legacy globally normalized L2 residual remains available for progress
    reporting, but its dimension-dependent scaling cannot terminate an outer
    solve whose authoritative componentwise backward error is still too high.
    Missing and nonfinite schema-v2 diagnostics fail closed.
    """

    residual = float(diagnostics.backward_error_kkt_residual)
    return bool(
        np.isfinite(residual)
        and residual >= 0.0
        and residual <= 5.0 * _validate_solver_tolerance(certification_tol)
    )


def _uses_backward_error_progress(
    *,
    requested: bool,
    runtime_dtype: torch.dtype,
) -> bool:
    """Use schema-v2 progress only for float64 certification recovery.

    Ordinary fits retain the legacy progress signal and their required float64
    terminal audit. Explicit recovery mode survives equal solve/certification
    tolerances (for example strict) but remains disabled if float64 context
    promotion failed.
    """

    return bool(requested and runtime_dtype == torch.float64)


def _solve_inner_subproblem(
    *,
    use_alm: bool,
    runtime,
    num_mutations: int,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    degree_bound: int,
    tol: float,
    inner_max_iter: int,
    phi: torch.Tensor,
    dual,
    dual_start_is_actual: bool,
    spectral_rho: bool,
    use_backward_error_stopping: bool,
    pdhg_tau_node,
    backend_name: str,
    graph_hash: str,
) -> InnerSolveResult:
    """Dispatch the majorized inner subproblem to the ALM (complete-graph) or PDHG
    solver and wrap its legacy tuple in a representation-aware result."""
    if use_alm:
        dense_fits, dense_bytes, dense_limit = dense_complete_solver_memory_preflight(
            num_nodes=num_mutations,
            num_regions=int(U.shape[1]),
            runtime=runtime,
        )
        if not dense_fits:
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: dense complete-graph solve needs "
                f"approximately {dense_bytes} bytes (available policy limit: "
                f"{dense_limit})."
            )
    if use_alm:
        (
            phi_trial,
            dual_trial,
            dual_kkt_trial,
            _inner_iterations,
            inner_ok,
            _inner_residual,
            surrogate_audit,
        ) = solve_majorized_subproblem_alm_torch(
            runtime=runtime,
            num_mutations=num_mutations,
            U=U,
            h=h,
            lower=lower,
            upper=upper,
            lambda_value=lambda_value,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            tol=tol,
            max_iter=max(inner_max_iter, _MIN_INNER_ITERATIONS),
            phi_start=phi,
            dual_start=dual,
            dual_start_is_actual=dual_start_is_actual,
            spectral_rho=bool(spectral_rho),
            use_backward_error_stopping=bool(use_backward_error_stopping),
        )
    else:
        (
            phi_trial,
            dual_trial,
            dual_kkt_trial,
            _inner_iterations,
            inner_ok,
            _inner_residual,
            surrogate_audit,
        ) = solve_majorized_subproblem_pdhg_torch(
            runtime=runtime,
            num_mutations=num_mutations,
            U=U,
            h=h,
            lower=lower,
            upper=upper,
            lambda_value=lambda_value,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            degree_bound=degree_bound,
            tol=tol,
            max_iter=max(inner_max_iter, _MIN_INNER_ITERATIONS),
            phi_start=phi,
            dual_start=dual,
            tau_node=pdhg_tau_node,
            use_backward_error_stopping=bool(use_backward_error_stopping),
        )
    if use_alm:
        # The outer MM loop carries the rho-invariant actual multiplier y.
        # Drop the low-level scaled-u return here so a second complete
        # edge-by-region tensor does not remain live through outer scoring and
        # certificate refinement.
        dual_trial = dual_kkt_trial
    inner_iterations = max(int(_inner_iterations), 0)
    certificate = (
        DenseEdgeCertificate(
            dual=dual_kkt_trial,
            graph_hash=str(graph_hash),
            gradient_scope="mm_surrogate",
        )
        if torch.is_tensor(dual_kkt_trial)
        else None
    )
    return InnerSolveResult(
        phi=phi_trial,
        backend_name=str(backend_name),
        warm_state=DenseWarmState(
            phi=phi_trial,
            dual=dual_trial if torch.is_tensor(dual_trial) else None,
            previous_lambda=float(lambda_value),
            graph_hash=str(graph_hash),
        ),
        surrogate_certificate=certificate,
        surrogate_kkt=surrogate_audit.diagnostics,
        converged=bool(inner_ok),
        iterations=inner_iterations,
        work=surrogate_audit.work,
    )


def _fit_from_start(
    data: TumorData,
    *,
    context: SolverContext,
    lambda_value: float,
    eps: float,
    outer_max_iter: int,
    inner_max_iter: int,
    tol: float,
    certification_tol: float,
    use_backward_error_progress: bool,
    stagnation_audit_patience: int = _RECOVERY_STAGNATION_AUDIT_WINDOW,
    phi_start: np.ndarray | torch.Tensor,
    solver_state: SolverState | None,
    objective_shape: str,
    workset_max_bytes: int,
    compressed_cache_max_bytes: int,
    workset_add_batch: int,
    workset_max_expansions: int,
    max_edge_pass_equivalents: int | None = None,
    certificate_max_iter: int,
    certificate_refinement_rounds: int,
    certificate_column_tol_scale: float,
    verbose: bool,
    audit_context_cache: dict[tuple[str, str, str], object] | None = None,
) -> RawFit:
    source_model = context.source_model
    observed_model = context.observed_model
    runtime = context.runtime
    graph = context.graph_spec
    tensor_graph = context.graph
    graph_hash = context.graph_hash
    base_objective_key = context.base_objective_key
    objective_spec_hash = context.objective_spec_hash
    lower = observed_model.lower
    upper = observed_model.upper
    tol = _validate_solver_tolerance(tol)
    cert_tol = _validate_solver_tolerance(certification_tol)
    if base_objective_key.fingerprint != str(objective_spec_hash):
        raise ValueError("Base objective key does not match objective_spec_hash.")
    if (
        solver_state is not None
        and str(solver_state.objective_spec_hash)
        and str(solver_state.objective_spec_hash) != str(objective_spec_hash)
    ):
        raise ValueError("Solver warm state belongs to a different raw objective.")
    objective_shape = objective_shape_for_model(source_model, objective_shape)
    certificate_options = CertificateOptions(
        max_iter=max(int(certificate_max_iter), 1),
        refinement_rounds=max(int(certificate_refinement_rounds), 0),
        max_expansions=max(int(workset_max_expansions), 1),
        add_batch=max(int(workset_add_batch), 1),
        mapping_tolerance=max(
            0.1 * float(cert_tol), float(torch.finfo(runtime.dtype).eps)
        ),
        column_tolerance=max(
            float(certificate_column_tol_scale) * float(cert_tol),
            float(torch.finfo(runtime.dtype).eps),
        ),
        memory=WorksetMemoryOptions(
            max_workset_bytes=int(workset_max_bytes),
            max_compressed_cache_bytes=int(compressed_cache_max_bytes),
        ),
    )
    certificate_problem = CertificateProblem(
        graph=tensor_graph,
        graph_hash=str(graph_hash),
        lower=lower,
        upper=upper,
        lambda_value=float(lambda_value),
        atol=float(cert_tol),
    )
    use_unimodal_objective = objective_shape.startswith("unimodal")
    require_full_step_backtracking = (
        objective_shape == "unimodal_full_step_backtracking"
    )
    # This option is internal to fixed-objective certification recovery.  Keep
    # its requested value as the recovery-only stagnation marker even if a
    # failed float64 promotion requires legacy arithmetic for solver stopping.
    recovery_stagnation_enabled = bool(use_backward_error_progress)
    use_backward_error_progress = _uses_backward_error_progress(
        requested=use_backward_error_progress,
        runtime_dtype=runtime.dtype,
    )
    use_alm = bool(
        tensor_graph.is_complete
        and int(graph.degree_bound) == max(int(data.num_mutations) - 1, 1)
    )
    edge_u, edge_v, edge_w = (
        tensor_graph.edge_u,
        tensor_graph.edge_v,
        tensor_graph.weight,
    )
    if lambda_value <= 0.0 or int(edge_u.numel()) == 0:
        dense_inner_solver = "closed_form_projection"
    elif use_alm:
        # The complete-graph ALM backend is the scaled-dual ADMM algorithm:
        # group shrinkage, constrained phi update, then dual ascent.
        dense_inner_solver = "admm_complete_graph"
    else:
        dense_inner_solver = "pdhg"
    if (
        solver_state is not None
        and solver_state.phi is not None
        and tuple(solver_state.phi.shape) == tuple(observed_model.upper.shape)
    ):
        initial_phi = solver_state.phi.to(dtype=runtime.dtype, device=runtime.device)
    else:
        initial_phi = as_runtime_tensor(phi_start, runtime)
    initial_phi = torch.minimum(torch.maximum(initial_phi, lower), upper)

    state_dual = _project_state_dual(
        solver_state,
        runtime=runtime,
        edge_w=edge_w,
        lambda_value=lambda_value,
        num_edges=int(edge_u.numel()),
        num_regions=int(initial_phi.shape[1]),
    )
    initial_warm_state = (
        solver_state.warm_state
        if solver_state is not None and solver_state.warm_state is not None
        else DenseWarmState(
            phi=initial_phi,
            dual=state_dual,
            previous_lambda=float(lambda_value),
            graph_hash=str(graph_hash),
        )
    )
    initial_certificate = (
        solver_state.certificate
        if (
            solver_state is not None
            and solver_state.certificate is not None
            and getattr(solver_state.certificate, "graph_hash", None) == graph_hash
        )
        else (
            DenseEdgeCertificate(
                dual=state_dual,
                graph_hash=graph_hash,
                gradient_scope="observed_objective",
            )
            if torch.is_tensor(state_dual)
            else None
        )
    )
    initial_dual_start_is_actual = bool(use_alm and state_dual is not None)
    converged = False
    converged_outer = False
    iterations = 0
    work = WorkLedger()
    final_outer_diag = _initial_outer_diag()
    outer_kkt_certificate_status = "not_audited"
    mm_consistency_violations = 0
    inner_solve_calls = 0
    accepted_full_steps = 0
    accepted_damped_steps = 0
    rejected_outer_steps = 0
    outer_stop_reason = "outer_iteration_limit"
    legacy_stop_kkt_residual = float("inf")
    componentwise_stop_kkt_residual = float("inf")
    recovery_progress_monitor = (
        _RecoveryProgressMonitor(
            tolerance=float(tol),
            certification_tolerance=float(cert_tol),
            audit_patience=max(int(stagnation_audit_patience), 1),
        )
        if recovery_stagnation_enabled
        else None
    )
    recovery_stagnated = False
    best_audited_state: _AuditedSolverSnapshot | None = None
    work_budget_reached = False
    progress_residual_method = (
        "componentwise_box_cone_backward_error_v1"
        if use_backward_error_progress
        else "legacy_global_l2_progress_v1"
    )
    full_step_curvature_multiplier = torch.ones_like(initial_phi)

    initial_terms = mutation_region_terms_torch(observed_model, initial_phi, eps=eps)
    initial_penalty_tensor, penalty_work = _evaluate_pairwise_penalty_torch(
        initial_phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
    )
    work.charge(penalty_work)
    initial_fit_loss, _, initial_objective = (
        _objective_value_from_mutation_region_terms_torch(
            initial_terms,
            penalty_tensor=initial_penalty_tensor,
        )
    )
    state = _MMState(
        phi=initial_phi,
        dual=state_dual,
        certificate=initial_certificate,
        warm_state=initial_warm_state,
        terms=initial_terms,
        penalty_tensor=initial_penalty_tensor,
        fit_loss=initial_fit_loss,
        objective=initial_objective,
        inner_solver=dense_inner_solver,
        dual_start_is_actual=initial_dual_start_is_actual,
    )
    for outer_iter in range(max(int(outer_max_iter), 1)):
        remaining_budget = _remaining_edge_pass_budget(
            work.total,
            max_edge_pass_equivalents,
        )
        if (
            remaining_budget is not None
            and remaining_budget <= _MANDATORY_TERMINAL_EDGE_PASS_ALLOWANCE
        ):
            outer_stop_reason = "solver_work_budget_reached"
            work_budget_reached = True
            break
        iterations = outer_iter + 1
        previous_phi = state.phi.clone()
        previous_objective = state.objective
        if use_unimodal_objective:
            surrogate_terms = state.terms
            surrogate_fit_loss = float(state.fit_loss)
        else:
            responsibilities = state.terms.posterior
            surrogate_terms = em_surrogate_terms_torch(
                observed_model,
                state.phi,
                responsibilities=responsibilities,
                eps=eps,
            )
            surrogate_fit_loss = float(torch.sum(surrogate_terms.loss).item())
        h_base, surrogate_grad = _safe_surrogate_curvature_and_gradient(
            surrogate_terms,
            observed_model.observed,
        )
        if use_unimodal_objective:
            smooth_lower, smooth_upper = lower, upper
        else:
            smooth_lower, smooth_upper = _path_smooth_interval_bounds(
                observed_model,
                state.phi,
                lower=lower,
                upper=upper,
                eps=float(eps),
            )
        if require_full_step_backtracking:
            remaining_budget = _remaining_edge_pass_budget(
                work.total,
                max_edge_pass_equivalents,
            )
            if (
                remaining_budget is not None
                and remaining_budget < _MANDATORY_TERMINAL_EDGE_PASS_ALLOWANCE + 2
            ):
                outer_stop_reason = "solver_work_budget_reached"
                work_budget_reached = True
                break
            forcing_certificate = state.certificate
            if forcing_certificate is None:
                forcing_certificate = _compressed_certificate_for_primal(
                    state.phi,
                    graph_hash=graph_hash,
                    gradient_scope="observed_objective",
                )
            forcing_gradient = build_certificate_gradient(
                observed_model,
                phi=state.phi,
                smooth_gradient=state.terms.gradient,
                lower=lower,
                upper=upper,
                eps=eps,
                tol=cert_tol,
            )
            forcing_attempt = certify(
                problem=certificate_problem,
                phi=state.phi,
                gradient=forcing_gradient,
                witness=forcing_certificate,
                refine=False,
            )
            work.charge(forcing_attempt.work_counters)
            forcing_diag = forcing_attempt.diagnostics
            forcing_residual = (
                forcing_diag.backward_error_kkt_residual
                if use_backward_error_progress
                else forcing_diag.kkt_residual
            )
            inner_progress_tolerance = max(
                5.0 * tol,
                min(
                    float(np.sqrt(tol)),
                    0.9 * float(forcing_residual),
                ),
            )
        else:
            inner_progress_tolerance = 5.0 * tol
        scale = 1.0
        curvature_multiplier = full_step_curvature_multiplier
        accepted = False
        candidate_state = state

        curvature_attempts = (
            _FULL_STEP_MAX_CURVATURE_ATTEMPTS
            if require_full_step_backtracking
            else (1 if use_unimodal_objective else 10)
        )
        for _curvature_attempt in range(curvature_attempts):
            remaining_budget = _remaining_edge_pass_budget(
                work.total,
                max_edge_pass_equivalents,
            )
            if (
                remaining_budget is not None
                and remaining_budget <= _MANDATORY_TERMINAL_EDGE_PASS_ALLOWANCE
            ):
                work_budget_reached = True
                break
            h = (
                h_base * curvature_multiplier
                if require_full_step_backtracking
                else h_base * scale
            )
            U = _safe_majorized_center(
                state.phi,
                surrogate_grad=surrogate_grad,
                h=h,
                likelihood_included=observed_model.observed,
            )
            if use_unimodal_objective and not require_full_step_backtracking:
                q_current = None
            else:
                q_current = _inner_model_value_torch(
                    state.phi,
                    U=U,
                    h=h,
                    penalty_tensor=state.penalty_tensor,
                )
            recovery_inner_model_tol = (
                max(
                    64.0 * float(torch.finfo(state.phi.dtype).eps),
                    float(tol) ** 2,
                )
                * (1.0 + abs(float(q_current.item())))
                if require_full_step_backtracking
                else 0.0
            )
            inner_phi_start = state.phi
            inner_dual_start = state.dual
            inner_dual_start_is_actual = state.dual_start_is_actual
            batch_penalty_tensor: torch.Tensor | None = None
            inner_batch_limit = 8 if require_full_step_backtracking else 1
            for _inner_batch in range(inner_batch_limit):
                remaining_budget = _remaining_edge_pass_budget(
                    work.total,
                    max_edge_pass_equivalents,
                )
                inner_iteration_limit = _budgeted_inner_max_iter(
                    inner_max_iter,
                    remaining=(
                        None
                        if remaining_budget is None
                        else remaining_budget - _MANDATORY_TERMINAL_EDGE_PASS_ALLOWANCE
                    ),
                    use_alm=use_alm,
                    spectral_rho=bool(
                        require_full_step_backtracking
                        or (use_alm and use_backward_error_progress)
                    ),
                )
                if inner_iteration_limit is None:
                    work_budget_reached = True
                    break
                inner_result = _solve_inner_subproblem(
                    use_alm=use_alm,
                    runtime=runtime,
                    num_mutations=data.num_mutations,
                    U=U,
                    h=h,
                    lower=smooth_lower,
                    upper=smooth_upper,
                    lambda_value=lambda_value,
                    edge_u=edge_u,
                    edge_v=edge_v,
                    edge_w=edge_w,
                    degree_bound=int(graph.degree_bound),
                    tol=tol,
                    inner_max_iter=inner_iteration_limit,
                    phi=inner_phi_start,
                    dual=inner_dual_start,
                    dual_start_is_actual=inner_dual_start_is_actual,
                    spectral_rho=bool(
                        require_full_step_backtracking
                        or (use_alm and use_backward_error_progress)
                    ),
                    use_backward_error_stopping=use_backward_error_progress,
                    pdhg_tau_node=tensor_graph.pdhg_tau_node,
                    backend_name=dense_inner_solver,
                    graph_hash=graph_hash,
                )
                inner_solve_calls += 1
                work.charge(inner_result.work)
                phi_trial = inner_result.phi
                dense_warm_state = inner_result.warm_state
                dual_trial = getattr(dense_warm_state, "dual", None)
                surrogate_certificate = inner_result.surrogate_certificate
                dual_kkt_trial = getattr(surrogate_certificate, "dual", None)
                inner_ok = bool(inner_result.converged)
                inner_residual = float(
                    (
                        inner_result.surrogate_kkt.backward_error_kkt_residual
                        if use_backward_error_progress
                        else inner_result.surrogate_kkt.kkt_residual
                    )
                )
                batch_inner_certified = bool(inner_ok)
                if require_full_step_backtracking:
                    penalty_edge_passes = _pairwise_penalty_edge_passes(
                        edge_u=edge_u,
                        lambda_value=lambda_value,
                    )
                    if not _nonterminal_edge_work_fits(
                        work.total,
                        max_edge_pass_equivalents,
                        edge_passes=penalty_edge_passes,
                    ):
                        work_budget_reached = True
                        break
                    batch_penalty_tensor, batch_penalty_work = (
                        _evaluate_pairwise_penalty_torch(
                            phi_trial,
                            edge_u=edge_u,
                            edge_v=edge_v,
                            edge_w=edge_w,
                            lambda_value=lambda_value,
                        )
                    )
                    work.charge(batch_penalty_work)
                    batch_inner_certified = bool(
                        np.isfinite(float(inner_residual))
                        and float(inner_residual) <= inner_progress_tolerance
                    )
                    batch_q_trial = _inner_model_value_torch(
                        phi_trial,
                        U=U,
                        h=h,
                        penalty_tensor=batch_penalty_tensor,
                    )
                    batch_inner_model_gap = float((batch_q_trial - q_current).item())
                    batch_inner_certified = bool(
                        batch_inner_certified
                        and np.isfinite(batch_inner_model_gap)
                        and batch_inner_model_gap <= recovery_inner_model_tol
                    )
                if batch_inner_certified:
                    inner_ok = True
                    break
                inner_phi_start = phi_trial
                inner_dual_start = dual_kkt_trial if use_alm else dual_trial
                inner_dual_start_is_actual = bool(use_alm)
            if work_budget_reached:
                break
            delta = phi_trial - state.phi
            trial_mutation_region_terms = mutation_region_terms_torch(
                observed_model, phi_trial, eps=eps
            )
            if batch_penalty_tensor is None:
                penalty_edge_passes = _pairwise_penalty_edge_passes(
                    edge_u=edge_u,
                    lambda_value=lambda_value,
                )
                if not _nonterminal_edge_work_fits(
                    work.total,
                    max_edge_pass_equivalents,
                    edge_passes=penalty_edge_passes,
                ):
                    work_budget_reached = True
                    break
                trial_penalty_tensor, trial_penalty_work = (
                    _evaluate_pairwise_penalty_torch(
                        phi_trial,
                        edge_u=edge_u,
                        edge_v=edge_v,
                        edge_w=edge_w,
                        lambda_value=lambda_value,
                    )
                )
                work.charge(trial_penalty_work)
            else:
                trial_penalty_tensor = batch_penalty_tensor
            trial_fit_loss, _, trial_objective = (
                _objective_value_from_mutation_region_terms_torch(
                    trial_mutation_region_terms,
                    penalty_tensor=trial_penalty_tensor,
                )
            )
            objective_gap = float(trial_objective - previous_objective)
            audit_quadratic_majorizer = bool(
                require_full_step_backtracking or not use_unimodal_objective
            )
            if not audit_quadratic_majorizer:
                inner_model_gap = 0.0
                surrogate_gap = 0.0
                em_envelope_gap = 0.0
            else:
                quadratic_gap = float(
                    torch.sum(
                        surrogate_terms.gradient * delta
                        + 0.5 * h * torch.square(delta)
                    ).item()
                )
                majorizer_rhs = surrogate_fit_loss + quadratic_gap
                q_trial = (
                    batch_q_trial
                    if require_full_step_backtracking
                    else _inner_model_value_torch(
                        phi_trial,
                        U=U,
                        h=h,
                        penalty_tensor=trial_penalty_tensor,
                    )
                )
                inner_model_gap = float((q_trial - q_current).item())
                if use_unimodal_objective:
                    # Recovery uses an exact smooth-loss majorization check.
                    # This makes the accepted full ADMM endpoint a valid
                    # proximal-MM update with a matching actual dual.
                    surrogate_gap = float(trial_fit_loss - majorizer_rhs)
                    em_envelope_gap = 0.0
                else:
                    trial_surrogate_terms = em_surrogate_terms_torch(
                        observed_model,
                        phi_trial,
                        responsibilities=responsibilities,
                        eps=eps,
                    )
                    trial_surrogate_loss = float(
                        torch.sum(trial_surrogate_terms.loss).item()
                    )
                    surrogate_gap = float(trial_surrogate_loss - majorizer_rhs)
                    em_envelope_gap = float(
                        (trial_fit_loss - state.fit_loss)
                        - (trial_surrogate_loss - surrogate_fit_loss)
                    )
            finite_attempt = all(
                np.isfinite(value)
                for value in [
                    inner_model_gap,
                    surrogate_gap,
                    em_envelope_gap,
                    objective_gap,
                    trial_fit_loss,
                    trial_objective,
                ]
            )
            if require_full_step_backtracking:
                numerical_factor = 64.0 * float(torch.finfo(state.phi.dtype).eps)
                inner_model_tol = max(numerical_factor, float(tol) ** 2) * (
                    1.0 + abs(float(q_current.item()))
                )
                majorization_tol = max(numerical_factor, float(tol) ** 2) * (
                    1.0 + abs(surrogate_fit_loss)
                )
                objective_tol = numerical_factor * (1.0 + abs(previous_objective))
            else:
                inner_model_tol = (
                    1e-8 * (1.0 + abs(float(q_current.item())))
                    if audit_quadratic_majorizer
                    else 0.0
                )
                majorization_tol = 1e-8 * (1.0 + abs(surrogate_fit_loss))
                objective_tol = 1e-8 * (1.0 + abs(previous_objective))
            envelope_tol = 1e-8 * (1.0 + abs(state.fit_loss))
            if not finite_attempt:
                scale *= 2.0
                if require_full_step_backtracking:
                    curvature_multiplier = torch.clamp(
                        2.0 * curvature_multiplier,
                        max=1e12,
                    )
                    full_step_curvature_multiplier = curvature_multiplier
                continue
            if audit_quadratic_majorizer and inner_model_gap > inner_model_tol:
                if require_full_step_backtracking:
                    break
                scale *= 2.0
                continue
            if (
                audit_quadratic_majorizer
                and not require_full_step_backtracking
                and surrogate_gap > majorization_tol
            ):
                scale *= 2.0
                continue
            if not use_unimodal_objective and em_envelope_gap > envelope_tol:
                scale *= 2.0
                continue
            if require_full_step_backtracking and not (
                np.isfinite(float(inner_residual))
                and float(inner_residual) <= inner_progress_tolerance
            ):
                break
            recovery_armijo_rhs = (
                1e-4 * min(float(inner_model_gap), 0.0) + objective_tol
                if require_full_step_backtracking
                else objective_tol
            )
            if objective_gap <= recovery_armijo_rhs:
                accepted = True
                accepted_full_steps += 1
                # The complete-graph ADMM backend also returns the actual KKT
                # multiplier y=rho*u. Carry y, not the rho-dependent scaled u,
                # across outer MM subproblems because curvature changes rho.
                trial_inner_converged = bool(
                    (
                        np.isfinite(float(inner_residual))
                        and float(inner_residual) <= 5.0 * tol
                    )
                    if require_full_step_backtracking
                    else (
                        inner_ok
                        and np.isfinite(float(inner_residual))
                        and float(inner_residual) <= 5.0 * tol
                    )
                )
                candidate_state = _MMState(
                    phi=phi_trial,
                    dual=dual_kkt_trial if use_alm else dual_trial,
                    certificate=surrogate_certificate,
                    warm_state=inner_result.warm_state,
                    terms=trial_mutation_region_terms,
                    penalty_tensor=trial_penalty_tensor,
                    fit_loss=trial_fit_loss,
                    objective=trial_objective,
                    inner_solver=inner_result.backend_name,
                    dual_start_is_actual=bool(use_alm),
                    inner_converged=trial_inner_converged,
                )
                if require_full_step_backtracking:
                    # Retain coordinate-wise curvature evidence while trying a
                    # less conservative metric at the next accepted iterate.
                    full_step_curvature_multiplier = torch.clamp(
                        0.5 * curvature_multiplier,
                        min=1.0,
                        max=1e12,
                    )
                break
            if require_full_step_backtracking:
                # A damped primal point does not share the full subproblem's
                # dual certificate. Enlarge the persistent majorizing
                # curvature and accept only a full proximal-MM/ADMM endpoint.
                # If the resource limit is exhausted, leave this outer iterate
                # unchanged and uncertified rather than interpolating phi.
                delta_square = torch.square(delta)
                resolution = torch.finfo(state.phi.dtype).eps * (
                    1.0 + torch.square(state.phi)
                )
                secant_remainder = (
                    trial_mutation_region_terms.loss
                    - state.terms.loss
                    - surrogate_grad * delta
                )
                required_h = torch.where(
                    delta_square > resolution,
                    2.0
                    * torch.clamp(secant_remainder, min=0.0)
                    / torch.clamp(
                        delta_square,
                        min=torch.finfo(state.phi.dtype).tiny,
                    ),
                    h,
                )
                target_h = torch.maximum(h, 1.25 * required_h)
                proposed_multiplier = torch.clamp(
                    target_h
                    / torch.clamp(
                        h_base,
                        min=_MISSING_SURROGATE_CURVATURE,
                    ),
                    min=1.0,
                    max=1e12,
                )
                changed = bool(
                    torch.any(
                        proposed_multiplier
                        > curvature_multiplier
                        * (1.0 + 64.0 * torch.finfo(state.phi.dtype).eps)
                    ).item()
                )
                if changed:
                    curvature_multiplier = proposed_multiplier
                else:
                    curvature_multiplier = torch.clamp(
                        2.0 * curvature_multiplier,
                        max=1e12,
                    )
                full_step_curvature_multiplier = curvature_multiplier
                continue
            if (
                not use_unimodal_objective
                and inner_ok
                and inner_model_gap <= inner_model_tol
                and surrogate_gap <= majorization_tol
                and em_envelope_gap <= envelope_tol
                and objective_gap > max(1e-5, objective_tol)
            ):
                mm_consistency_violations += 1
            theta = 0.5
            damped_accepted = False
            for _line_search_iter in range(12):
                phi_theta = state.phi + theta * delta
                theta_mutation_region_terms = mutation_region_terms_torch(
                    observed_model, phi_theta, eps=eps
                )
                penalty_edge_passes = _pairwise_penalty_edge_passes(
                    edge_u=edge_u,
                    lambda_value=lambda_value,
                )
                if not _nonterminal_edge_work_fits(
                    work.total,
                    max_edge_pass_equivalents,
                    edge_passes=penalty_edge_passes,
                ):
                    work_budget_reached = True
                    break
                theta_penalty_tensor, theta_penalty_work = (
                    _evaluate_pairwise_penalty_torch(
                        phi_theta,
                        edge_u=edge_u,
                        edge_v=edge_v,
                        edge_w=edge_w,
                        lambda_value=lambda_value,
                    )
                )
                work.charge(theta_penalty_work)
                theta_fit_loss, _, theta_objective = (
                    _objective_value_from_mutation_region_terms_torch(
                        theta_mutation_region_terms,
                        penalty_tensor=theta_penalty_tensor,
                    )
                )
                if (
                    np.isfinite(theta_objective)
                    and theta_objective <= previous_objective + objective_tol
                ):
                    accepted = True
                    damped_accepted = True
                    accepted_damped_steps += 1
                    trial_warm_state = _invalidate_damped_trial_state(
                        phi=phi_theta,
                        trial_warm_state=inner_result.warm_state,
                    )
                    candidate_state = _MMState(
                        phi=phi_theta,
                        dual=None,
                        certificate=None,
                        warm_state=trial_warm_state,
                        terms=theta_mutation_region_terms,
                        penalty_tensor=theta_penalty_tensor,
                        fit_loss=theta_fit_loss,
                        objective=theta_objective,
                        inner_solver=inner_result.backend_name,
                        dual_start_is_actual=False,
                    )
                    break
                theta *= 0.5
            if damped_accepted:
                break
            if work_budget_reached:
                break
            scale *= 2.0

        if work_budget_reached:
            outer_stop_reason = "solver_work_budget_reached"

        if not accepted:
            rejected_outer_steps += 1
        state = candidate_state
        if verbose:
            print(
                f"[pairwise-fusion:{runtime.device_name}] iter={iterations:02d} objective={state.objective:.6f} "
                f"fit={state.fit_loss:.6f} penalty={state.penalty:.6f}"
            )

        rel_change = abs(previous_objective - state.objective) / (
            1.0 + abs(previous_objective)
        )
        step_residual = float(
            (
                torch.linalg.norm(state.phi - previous_phi)
                / (1.0 + torch.linalg.norm(previous_phi))
            ).item()
        )
        cheap_outer_converged = bool(
            rel_change <= 10.0 * tol and step_residual <= max(1e-8, np.sqrt(tol))
        )
        do_outer_kkt_audit = bool(
            cheap_outer_converged
            or iterations >= max(int(outer_max_iter), 1)
            or iterations % _OUTER_KKT_CHECK_EVERY == 0
            or not np.isfinite(state.objective)
        )
        remaining_budget = _remaining_edge_pass_budget(
            work.total,
            max_edge_pass_equivalents,
        )
        if (
            do_outer_kkt_audit
            and remaining_budget is not None
            and remaining_budget < _MANDATORY_TERMINAL_EDGE_PASS_ALLOWANCE + 2
        ):
            do_outer_kkt_audit = False
            work_budget_reached = True
        outer_diag = final_outer_diag
        outer_converged = False
        if do_outer_kkt_audit:
            outer_terms = state.terms
            observed_start = state.certificate
            if observed_start is None and isinstance(
                state.warm_state, PrimalOnlyWarmState
            ):
                observed_start = _rebase_certificate_hint(
                    state.warm_state.certificate_hint,
                    phi=state.phi,
                    graph=tensor_graph,
                    graph_hash=graph_hash,
                    lambda_value=lambda_value,
                )
            should_refine = bool(
                state.certificate is None
                or isinstance(observed_start, CompressedEdgeCertificate)
            )
            periodic_gradient = build_certificate_gradient(
                observed_model,
                state.phi,
                smooth_gradient=outer_terms.gradient,
                lower=lower,
                upper=upper,
                eps=eps,
                tol=cert_tol,
            )
            periodic_limit = min(
                int(certificate_options.max_iter),
                _PERIODIC_CERTIFICATE_MAX_ITER,
            )
            periodic_parameters = _budgeted_certificate_parameters(
                certificate=observed_start,
                options=certificate_options,
                requested_max_iter=periodic_limit,
                remaining=(
                    None
                    if remaining_budget is None
                    else remaining_budget - _MANDATORY_TERMINAL_EDGE_PASS_ALLOWANCE
                ),
                mandatory=False,
            )
            if should_refine and periodic_parameters is None:
                do_outer_kkt_audit = False
                work_budget_reached = True
            periodic_max_iter, periodic_options = (
                (periodic_limit, certificate_options)
                if periodic_parameters is None
                else periodic_parameters
            )
        if do_outer_kkt_audit:
            work.charge_outer_kkt_audits()
            observed_refinement = certify(
                problem=certificate_problem,
                phi=state.phi,
                gradient=periodic_gradient,
                witness=observed_start,
                refine=should_refine,
                max_iter=periodic_max_iter,
                options=(
                    periodic_options
                    if isinstance(observed_start, CompressedEdgeCertificate)
                    else None
                ),
            )
            work.charge(observed_refinement.work_counters)
            state = replace(state, certificate=observed_refinement.certificate)
            outer_diag = observed_refinement.diagnostics
            legacy_stop_kkt_residual = float(outer_diag.kkt_residual)
            componentwise_stop_kkt_residual = float(
                outer_diag.backward_error_kkt_residual
            )
            outer_converged = bool(
                _backward_error_kkt_within_gate(
                    outer_diag,
                    certification_tol=cert_tol,
                )
                if use_backward_error_progress
                else float(outer_diag.kkt_residual) <= 5.0 * cert_tol
            )
            if recovery_progress_monitor is not None:
                audited_certified = bool(
                    outer_converged and periodic_gradient.directional_admissible
                )
                best_audited_state = _retain_audited_solver_state_if_better(
                    incumbent=best_audited_state,
                    state=state,
                    graph_hash=graph_hash,
                    lambda_value=lambda_value,
                    diagnostics=outer_diag,
                    certified=audited_certified,
                )
        if do_outer_kkt_audit:
            final_outer_diag = outer_diag
        converged_outer = bool(outer_converged)
        if (
            rel_change <= tol
            and step_residual <= np.sqrt(tol)
            and state.inner_converged
            and outer_converged
        ):
            converged = True
            outer_stop_reason = (
                "componentwise_backward_error_converged"
                if use_backward_error_progress
                else "legacy_progress_converged"
            )
            break
        if do_outer_kkt_audit and recovery_progress_monitor is not None:
            stagnation_reason = recovery_progress_monitor.observe(
                outer_iteration=int(iterations),
                objective=float(state.objective),
                diagnostics=outer_diag,
                step_residual=float(step_residual),
                rejected_outer_steps=int(rejected_outer_steps),
            )
            if stagnation_reason is not None:
                outer_stop_reason = str(stagnation_reason)
                recovery_stagnated = True
                break
        if work_budget_reached:
            outer_stop_reason = "solver_work_budget_reached"
            break
        if max_edge_pass_equivalents is not None and int(
            work.total.edge_pass_equivalents
        ) >= int(max_edge_pass_equivalents):
            outer_stop_reason = "solver_work_budget_reached"
            break

    if (recovery_stagnated or work_budget_reached) and best_audited_state is not None:
        # Terminal certification must start from the best state that actually
        # passed a full observed-objective audit, not merely the last state in
        # a stalled trajectory.  All work spent reaching later states remains
        # charged above.
        restored_phi = best_audited_state.phi
        restored_penalty_tensor = torch.as_tensor(
            best_audited_state.objective - best_audited_state.fit_loss,
            dtype=restored_phi.dtype,
            device=restored_phi.device,
        )
        final_outer_diag = best_audited_state.diagnostics
        legacy_stop_kkt_residual = float(final_outer_diag.kkt_residual)
        componentwise_stop_kkt_residual = float(
            final_outer_diag.backward_error_kkt_residual
        )
        converged_outer = best_audited_state.certified
        restored_terms = mutation_region_terms_torch(
            observed_model,
            restored_phi,
            eps=eps,
        )
        state = _MMState(
            phi=restored_phi,
            dual=best_audited_state.dual,
            certificate=best_audited_state.certificate,
            warm_state=best_audited_state.warm_state,
            terms=restored_terms,
            penalty_tensor=restored_penalty_tensor,
            fit_loss=best_audited_state.fit_loss,
            objective=best_audited_state.objective,
            inner_solver=best_audited_state.inner_solver,
            dual_start_is_actual=best_audited_state.dual_start_is_actual,
            inner_converged=best_audited_state.current_inner_converged,
        )

    final_terms = state.terms
    if state.certificate is None and isinstance(state.warm_state, PrimalOnlyWarmState):
        state = replace(
            state,
            certificate=_rebase_certificate_hint(
                state.warm_state.certificate_hint,
                phi=state.phi,
                graph=tensor_graph,
                graph_hash=graph_hash,
                lambda_value=lambda_value,
            ),
        )
    certificate_gradient = build_certificate_gradient(
        observed_model,
        state.phi,
        smooth_gradient=final_terms.gradient,
        lower=lower,
        upper=upper,
        eps=eps,
        tol=cert_tol,
    )

    final_refinements = []
    certificate_needs_final_pass = False
    last_refined_certificate_gradient = certificate_gradient
    residual_before_probe = float(final_outer_diag.backward_error_kkt_residual)

    def terminal_refinement(
        requested_max_iter: int,
        *,
        mandatory: bool,
    ):
        nonlocal state, last_refined_certificate_gradient
        remaining = _remaining_terminal_edge_pass_budget(
            work.total,
            max_edge_pass_equivalents,
        )
        # Float32 fits still owe objective evaluation and the authoritative
        # float64 full-graph audit. Reserve their conservative worst case.
        audit_reserve = _terminal_float64_audit_edge_pass_bound(
            runtime_dtype=runtime.dtype,
            certificate=state.certificate,
            edge_count=int(edge_u.numel()),
        )
        refinement_budget = None if remaining is None else remaining - audit_reserve
        parameters = _budgeted_certificate_parameters(
            certificate=state.certificate,
            options=certificate_options,
            requested_max_iter=requested_max_iter,
            remaining=refinement_budget,
            mandatory=mandatory,
        )
        if parameters is None:
            if not mandatory:
                return None
            # A terminal audit remains mandatory, but refinement is optional
            # under a hard work cap.  This path cannot fabricate convergence:
            # it audits the current witness and preserves fail-closed status.
            refinement = certify(
                problem=certificate_problem,
                phi=state.phi,
                gradient=certificate_gradient,
                witness=state.certificate,
                refine=False,
            )
            final_refinements.append(refinement)
            work.charge(refinement.work_counters)
            state = replace(state, certificate=refinement.certificate)
            last_refined_certificate_gradient = certificate_gradient
            return refinement
        refinement_max_iter, refinement_options = parameters
        refinement = certify(
            problem=certificate_problem,
            phi=state.phi,
            gradient=certificate_gradient,
            witness=state.certificate,
            refine=True,
            max_iter=refinement_max_iter,
            options=(
                refinement_options
                if isinstance(state.certificate, CompressedEdgeCertificate)
                else None
            ),
        )
        final_refinements.append(refinement)
        work.charge(refinement.work_counters)
        state = replace(state, certificate=refinement.certificate)
        last_refined_certificate_gradient = certificate_gradient
        return refinement

    residual_before_round = residual_before_probe
    for refinement_round in range(4):
        if refinement_round == 0:
            probe_iter = min(
                _FINAL_CERTIFICATE_PROBE_MAX_ITER,
                int(certificate_options.max_iter),
            )
            final_certificate_refinement = terminal_refinement(
                probe_iter,
                mandatory=True,
            )
            assert final_certificate_refinement is not None
            probe_action = _certificate_probe_action(
                residual_before=residual_before_round,
                probe_residual=float(
                    final_certificate_refinement.diagnostics.backward_error_kkt_residual
                ),
                certification_tolerance=float(cert_tol),
                recovery_stagnated=bool(recovery_stagnated),
            )
            remaining_iter = int(certificate_options.max_iter) - probe_iter
            if probe_action == "deepen" and remaining_iter > 0:
                deepened = terminal_refinement(
                    remaining_iter,
                    mandatory=False,
                )
                if deepened is None:
                    work_budget_reached = True
                else:
                    final_certificate_refinement = deepened
                    probe_action = _certificate_probe_action(
                        residual_before=residual_before_round,
                        probe_residual=float(
                            final_certificate_refinement.diagnostics.backward_error_kkt_residual
                        ),
                        certification_tolerance=float(cert_tol),
                        recovery_stagnated=bool(recovery_stagnated),
                    )
        else:
            next_refinement = terminal_refinement(
                int(certificate_options.max_iter),
                mandatory=False,
            )
            if next_refinement is None:
                work_budget_reached = True
                certificate_gradient = last_refined_certificate_gradient
                certificate_needs_final_pass = False
                break
            final_certificate_refinement = next_refinement
            probe_action = _certificate_probe_action(
                residual_before=residual_before_round,
                probe_residual=float(
                    final_certificate_refinement.diagnostics.backward_error_kkt_residual
                ),
                certification_tolerance=float(cert_tol),
                recovery_stagnated=bool(recovery_stagnated),
            )
        residual_before_round = float(
            final_certificate_refinement.diagnostics.backward_error_kkt_residual
        )
        certificate_needs_final_pass = False
        if not bool(torch.any(certificate_gradient.at_breakpoint).item()):
            if probe_action == "plateau":
                outer_stop_reason = "certificate_refinement_plateau"
            break
        interval_dual = getattr(state.certificate, "dual", None)
        if not torch.is_tensor(interval_dual):
            # A selected endpoint gradient is already a valid member of the
            # subgradient interval; compressed certificates simply cannot
            # improve a false negative by alternating the interval choice.
            if probe_action == "plateau":
                outer_stop_reason = "certificate_refinement_plateau"
            break
        remaining = _remaining_terminal_edge_pass_budget(
            work.total,
            max_edge_pass_equivalents,
        )
        audit_reserve = _terminal_float64_audit_edge_pass_bound(
            runtime_dtype=runtime.dtype,
            certificate=state.certificate,
            edge_count=int(edge_u.numel()),
        )
        if remaining is not None and remaining <= audit_reserve:
            work_budget_reached = True
            break
        fusion_adjustment = graph_adjoint_edges(
            interval_dual,
            edge_u=edge_u,
            edge_v=edge_v,
            num_nodes=int(state.phi.shape[0]),
        )
        work.charge_edge_passes(
            edge_count=int(edge_u.numel()),
            num_regions=int(state.phi.shape[1]),
        )
        next_gradient = build_certificate_gradient(
            observed_model,
            state.phi,
            smooth_gradient=final_terms.gradient,
            lower=lower,
            upper=upper,
            eps=eps,
            tol=cert_tol,
            fusion_adjoint=fusion_adjustment,
        )
        if torch.allclose(
            next_gradient.value,
            certificate_gradient.value,
            rtol=0.0,
            atol=max(float(cert_tol) * 0.1, 1e-12),
        ):
            if probe_action == "plateau":
                outer_stop_reason = "certificate_refinement_plateau"
            break
        certificate_gradient = next_gradient
        certificate_needs_final_pass = True

    if certificate_needs_final_pass:
        reconciled_refinement = terminal_refinement(
            int(certificate_options.max_iter),
            mandatory=False,
        )
        if reconciled_refinement is None:
            work_budget_reached = True
            certificate_gradient = last_refined_certificate_gradient
        else:
            final_certificate_refinement = reconciled_refinement
    if work_budget_reached:
        outer_stop_reason = "solver_work_budget_reached"
    final_outer_diag = final_certificate_refinement.diagnostics
    working_precision_kkt_residual = float(
        final_outer_diag.backward_error_kkt_residual
    )
    certificate_audit_dtype = dtype_name(runtime.dtype)
    authoritative_objective = float(state.objective)
    gradient_scope = certificate_gradient.scope
    directional_kink_admissible = certificate_gradient.directional_admissible
    if runtime.dtype == torch.float64:
        admission_diagnostics = final_certificate_refinement.diagnostics
    else:
        (
            admission_diagnostics,
            audit_gradient_scope,
            audit_directional_admissible,
            authoritative_objective,
            terminal_audit_work,
        ) = _terminal_backward_error_audit_float64(
            source_model=source_model,
            phi=state.phi,
            certificate=state.certificate,
            graph_spec=graph,
            graph_hash=graph_hash,
            lambda_value=lambda_value,
            eps=eps,
            tol=cert_tol,
            audit_context_cache=audit_context_cache,
            return_work=True,
        )
        certificate_audit_dtype = "float64"
        gradient_scope = audit_gradient_scope
        directional_kink_admissible = bool(
            directional_kink_admissible and audit_directional_admissible
        )
        work.charge(terminal_audit_work)
    if (
        max_edge_pass_equivalents is not None
        and int(work.total.edge_pass_equivalents)
        > int(max_edge_pass_equivalents) + _MANDATORY_TERMINAL_EDGE_PASS_ALLOWANCE
    ):
        raise AssertionError(
            "Edge-pass budget exceeded the configured cap plus terminal allowance."
        )
    terminal_components = KKTComponents.from_diagnostics(
        admission_diagnostics
    )
    authoritative_kkt_residual = terminal_components.residual
    if not np.isfinite(float(authoritative_objective)):
        outer_stop_reason = "nonfinite_objective"
    final_dual = getattr(state.certificate, "dual", None)
    outer_kkt_certificate_status = str(final_certificate_refinement.status)
    converged_outer = bool(authoritative_kkt_residual <= 5.0 * cert_tol)
    valid_dual_certificate = outer_kkt_certificate_status in {
        "zero_penalty_no_dual_needed",
        "analytic_nonfused_dual",
        "refined_fused_edge_dual",
        "input_dual_retained",
        "certified",
    }
    if converged_outer and directional_kink_admissible:
        # The full-original-graph backward-error audit is the certificate;
        # compressed workset stopping labels cannot invalidate that stronger
        # terminal result.
        outer_kkt_certificate_status = "certified"
        valid_dual_certificate = True
    selection_eligible = bool(
        np.isfinite(float(authoritative_objective))
        and converged_outer
        and valid_dual_certificate
        and mm_consistency_violations == 0
        and directional_kink_admissible
    )
    full_kkt_certified = bool(
        np.isfinite(authoritative_kkt_residual)
        and converged_outer
        and valid_dual_certificate
        and directional_kink_admissible
    )
    global_optimality_certified = bool(
        selection_eligible
        and use_unimodal_objective
        and source_model.has_fixed_linear_emission
    )
    global_optimality_basis = (
        _CONVEX_GLOBAL_OPTIMALITY_BASIS
        if global_optimality_certified
        else "not_certified"
    )
    if use_unimodal_objective and global_optimality_certified:
        converged = True

    phi_np = state.phi.detach().cpu().numpy()
    if isinstance(state.certificate, CompressedEdgeCertificate):
        terminal_warm_state = PrimalOnlyWarmState(
            phi=state.phi.detach(),
            structure_hint=state.certificate.labels.detach(),
            certificate_hint=state.certificate,
        )
    else:
        terminal_warm_state = DenseWarmState(
            phi=state.phi.detach(),
            dual=final_dual.detach() if torch.is_tensor(final_dual) else None,
            previous_lambda=float(lambda_value),
            graph_hash=str(graph_hash),
        )
    solver_state_out = SolverState(
        phi=state.phi.detach(),
        dual=final_dual.detach() if torch.is_tensor(final_dual) else None,
        previous_lambda=float(lambda_value),
        warm_state=terminal_warm_state,
        certificate=state.certificate,
        objective_spec_hash=str(objective_spec_hash),
    )
    return RawFit(
        phi=phi_np.astype(phi_np.dtype, copy=False),
        objective=ObjectiveValue(total=float(authoritative_objective)),
        certificate=CertificateResult(
            components=terminal_components,
            certified=bool(full_kkt_certified),
            admissible=bool(selection_eligible),
            global_optimum=bool(global_optimality_certified),
            status=str(outer_kkt_certificate_status),
            tolerance=5.0 * float(cert_tol),
            scope="full_original_graph",
            gradient_scope=str(gradient_scope),
            directional_admissible=bool(directional_kink_admissible),
            witness=state.certificate,
            working_residual=float(working_precision_kkt_residual),
            working_dtype=dtype_name(runtime.dtype),
            audit_dtype=str(certificate_audit_dtype),
            precision_polished=False,
            precision_polish_delta=0.0,
            residual_method="componentwise_box_cone_backward_error_v1",
            fallback_reason="",
        ),
        convergence=ConvergenceResult(
            converged=bool(converged),
            mm_consistency_violations=int(mm_consistency_violations),
            stage_outer_iterations=int(iterations),
            stage_outer_max_iter=max(int(outer_max_iter), 1),
            stage_inner_iterations=int(work.total.inner_iterations),
            stage_inner_max_iter=max(int(inner_max_iter), 10),
            stage_inner_solve_calls=int(inner_solve_calls),
            stop_reason=str(outer_stop_reason),
            progress_residual_method=str(progress_residual_method),
            solve_tolerance=float(tol),
            legacy_stop_kkt_residual=float(legacy_stop_kkt_residual),
            componentwise_stop_kkt_residual=float(componentwise_stop_kkt_residual),
            accepted_full_steps=int(accepted_full_steps),
            accepted_damped_steps=int(accepted_damped_steps),
            rejected_outer_steps=int(rejected_outer_steps),
        ),
        work=work.total,
        state=solver_state_out,
        provenance=FitProvenance(
            objective_key=make_lambda_objective_key(
                base_objective_key,
                lambda_value=float(lambda_value),
            ),
            device=runtime.device_name,
            dtype=dtype_name(runtime.dtype),
            inner_solver=str(state.inner_solver),
            global_optimality_basis=str(global_optimality_basis),
            likelihood_eps=float(eps),
        ),
    )


def fit_observed_data_pairwise_fusion(
    problem: FusionProblem,
    plan: SolvePlan,
    init: SolverInit | None = None,
    budget: SolveBudget | None = None,
) -> RawFit:
    data = problem.data
    lambda_value = problem.lambda_value
    major_prior = problem.major_prior
    eps = problem.eps
    graph = problem.graph
    adaptive_weight_gamma = problem.adaptive_weight_gamma
    adaptive_weight_floor = problem.adaptive_weight_floor
    adaptive_weight_baseline = problem.adaptive_weight_baseline
    objective_shape = problem.objective_shape
    outer_max_iter = plan.outer_max_iter
    inner_max_iter = plan.inner_max_iter
    tol = plan.tol
    certification_tol = plan.certification_tol
    use_backward_error_progress = plan.use_backward_error_progress
    stagnation_audit_patience = plan.stagnation_audit_patience
    device = plan.device
    dtype = plan.dtype
    workset_max_bytes = plan.workset_max_bytes
    compressed_cache_max_bytes = plan.compressed_cache_max_bytes
    dense_fallback_policy = plan.dense_fallback_policy
    workset_add_batch = plan.workset_add_batch
    workset_max_expansions = plan.workset_max_expansions
    certificate_max_iter = plan.certificate_max_iter
    certificate_refinement_rounds = plan.certificate_refinement_rounds
    certificate_column_tol_scale = plan.certificate_column_tol_scale
    verbose = plan.verbose
    init = SolverInit() if init is None else init
    phi_start = init.phi_start
    exact_pilot = init.exact_pilot
    pooled_start = init.pooled_start
    scalar_well_starts = init.scalar_well_starts
    runtime = init.runtime
    solver_context = init.solver_context
    solver_state = init.solver_state
    start_mode = init.start_mode
    append_default_nonconvex_starts = init.append_default_nonconvex_starts
    budget = SolveBudget() if budget is None else budget
    max_edge_pass_equivalents = budget.max_edge_pass_equivalents
    tol = _validate_solver_tolerance(tol)
    certification_tol = _validate_solver_tolerance(
        tol if certification_tol is None else certification_tol
    )
    lambda_value = validate_lambda_value(lambda_value)
    if solver_context is not None:
        objective_shape = objective_shape_for_model(
            solver_context.source_model,
            objective_shape,
        )
    major_prior = float(major_prior)
    normalized_fallback_policy = normalize_dense_fallback_policy(dense_fallback_policy)
    if solver_context is None:
        solver_context = prepare_torch_problem_with_resource_policy(
            data,
            dense_fallback_policy=normalized_fallback_policy,
            major_prior=float(major_prior),
            eps=float(eps),
            tol=tol,
            inner_max_iter=int(inner_max_iter),
            graph=graph,
            adaptive_weight_gamma=float(adaptive_weight_gamma),
            adaptive_weight_floor=float(adaptive_weight_floor),
            adaptive_weight_baseline=float(adaptive_weight_baseline),
            exact_pilot=exact_pilot,
            pooled_start=pooled_start,
            scalar_well_starts=scalar_well_starts,
            device=device,
            dtype=dtype,
            runtime=runtime,
            objective_shape=objective_shape,
        )
    else:
        expected_data_fingerprint = tumor_data_fingerprint(data)
        if solver_context.data_fingerprint != expected_data_fingerprint:
            raise ValueError(
                "SolverContext data fingerprint does not match the requested TumorData."
            )
        requested_model = compile_observed_model(
            data,
            major_prior=float(major_prior),
            eps=float(eps),
        )
        if (
            requested_model.fingerprint != solver_context.source_model.fingerprint
            or abs(float(solver_context.eps) - float(eps)) > 0.0
        ):
            raise ValueError(
                "SolverContext major_prior/eps do not match the requested fit options."
            )
        if (
            graph is not None
            and graph.fingerprint != solver_context.graph_spec.fingerprint
        ):
            raise ValueError(
                "SolverContext graph does not match the requested FusionProblem."
            )
        if observed_box_fingerprint(requested_model) != (
            solver_context.base_objective_key.box_hash
        ):
            raise ValueError(
                "SolverContext objective box does not match the requested "
                "FusionProblem."
            )

    source_model = solver_context.source_model
    objective_shape = objective_shape_for_model(source_model, objective_shape)
    major_prior = _major_prior_for_model(source_model)

    context_prepared_by_cpu_fallback = bool(
        solver_context.resource_fallback == "dense_cpu"
    )
    effective_runtime = solver_context.runtime
    effective_exact_pilot = solver_context.exact_pilot
    effective_pooled_start = solver_context.pooled_start
    effective_scalar_well_starts = solver_context.scalar_well_starts
    requires_generic_path_solver = bool(source_model.requires_generic_path_solver)

    normalized_start_mode = str(start_mode).strip().lower()
    if normalized_start_mode not in {"full", "warm_plus_pilot", "warm_only"}:
        raise ValueError(f"Unknown start_mode: {start_mode}")
    append_defaults = normalized_start_mode == "full"
    if append_default_nonconvex_starts is None:
        append_defaults = bool(append_defaults or requires_generic_path_solver)
    else:
        append_defaults = bool(append_default_nonconvex_starts)

    if objective_shape.startswith("unimodal"):
        start_bank = [phi_start] if phi_start is not None else [effective_exact_pilot]
    else:
        start_bank: list[np.ndarray | torch.Tensor] = []
        if phi_start is not None:
            start_bank.append(phi_start)
        if append_defaults:
            start_bank.extend(effective_scalar_well_starts)
            start_bank.append(effective_pooled_start)
    start_bank = _deduplicate_starts(start_bank, runtime=effective_runtime)

    attempted_work = WorkLedger()

    def remaining_attempt_budget() -> int | None:
        return _remaining_edge_pass_budget(
            attempted_work.total,
            max_edge_pass_equivalents,
        )

    def can_launch_another_solver_attempt() -> bool:
        remaining = remaining_attempt_budget()
        return bool(
            remaining is None or remaining > _MANDATORY_TERMINAL_EDGE_PASS_ALLOWANCE
        )

    def solve_start_once(
        *,
        context: SolverContext,
        start: np.ndarray | torch.Tensor,
        state: SolverState | None,
        polish: bool = False,
    ) -> RawFit:
        result = _fit_from_start(
            data,
            context=context,
            lambda_value=lambda_value,
            eps=eps,
            outer_max_iter=1 if polish else outer_max_iter,
            inner_max_iter=inner_max_iter,
            tol=tol,
            certification_tol=certification_tol,
            use_backward_error_progress=use_backward_error_progress,
            stagnation_audit_patience=stagnation_audit_patience,
            phi_start=start,
            solver_state=state,
            objective_shape=objective_shape,
            workset_max_bytes=workset_max_bytes,
            compressed_cache_max_bytes=compressed_cache_max_bytes,
            workset_add_batch=workset_add_batch,
            workset_max_expansions=workset_max_expansions,
            max_edge_pass_equivalents=remaining_attempt_budget(),
            certificate_max_iter=certificate_max_iter,
            certificate_refinement_rounds=certificate_refinement_rounds,
            certificate_column_tol_scale=certificate_column_tol_scale,
            verbose=verbose,
            audit_context_cache=context.audit_context_cache,
        )
        attempted_work.charge(result.work)
        return result

    cpu_fallback_context: SolverContext | None = None
    best_artifacts: RawFit | None = None
    best_artifacts_index = -1
    start_artifacts: list[RawFit] = []
    start_contexts: list[SolverContext] = []
    for start in start_bank:
        if start_artifacts and not can_launch_another_solver_attempt():
            break
        state_for_start = (
            solver_state
            if (solver_state is not None and start is start_bank[0])
            else None
        )
        cpu_seed = state_for_start.phi if state_for_start is not None else start
        attempted_artifacts: RawFit | None = None
        attempt_context = solver_context
        attempt_start = start
        attempt_state = state_for_start
        fallback_reason = ""
        fallback_backend: str | None = None
        policy_state = PolicyState(
            phase="working",
            runtime_device_type=attempt_context.runtime.device.type,
            fallback_policy=normalized_fallback_policy,
        )
        while True:
            action = decide_next_action(policy_state)
            match action:
                case NextAction.RETRY_SAME_RUNTIME:
                    try:
                        artifacts = solve_start_once(
                            context=attempt_context,
                            start=attempt_start,
                            state=attempt_state,
                        )
                    except (MemoryError, torch.OutOfMemoryError) as exc:
                        policy_state.result = None
                        policy_state.resource_error = exc
                        continue
                    policy_state.result = record_attempt(
                        artifacts,
                        attempted=attempted_artifacts,
                        reason=fallback_reason,
                        backend_name=fallback_backend,
                    )
                    policy_state.resource_error = None
                case NextAction.DENSE_CURRENT_DEVICE:
                    attempted_artifacts = policy_state.result
                    if attempted_artifacts is None:
                        raise AssertionError("Dense retry lacks an attempted fit.")
                    if not can_launch_another_solver_attempt():
                        artifacts = attempted_artifacts
                        artifacts_context = attempt_context
                        break
                    cpu_seed = (
                        attempted_artifacts.state.phi
                        if attempted_artifacts.state is not None
                        else attempted_artifacts.phi
                    )
                    attempt_start = cpu_seed
                    attempt_state = None
                    fallback_reason = (
                        "dense_current_device_after_compressed_not_certified"
                    )
                    fallback_backend = None
                    policy_state.result = None
                    policy_state.representation_retry_done = True
                case NextAction.CPU_FALLBACK:
                    resource_exc = policy_state.resource_error
                    if resource_exc is None:
                        raise AssertionError("CPU fallback lacks a resource failure.")
                    if cpu_fallback_context is None:
                        cpu_runtime = resolve_runtime(
                            "cpu", dtype=dtype_name(effective_runtime.dtype)
                        )
                        _require_dense_memory(
                            data,
                            cpu_runtime,
                            operation="dense CPU fallback",
                            limit_name="host limit",
                            cause=resource_exc,
                        )
                        try:
                            cpu_fallback_context = promote_solver_context_dtype(
                                solver_context,
                                dtype=cpu_runtime.dtype,
                                device=cpu_runtime.device,
                                start_override=cpu_seed,
                            )
                        except (MemoryError, torch.OutOfMemoryError) as cpu_exc:
                            raise ExactSolverResourceLimit(
                                "exact_solver_resource_limit: exact problem or graph "
                                "construction exhausted host memory during dense CPU "
                                "fallback."
                            ) from cpu_exc
                        cpu_start = cpu_fallback_context.exact_pilot
                    else:
                        cpu_start = (
                            cpu_seed.detach().cpu()
                            if torch.is_tensor(cpu_seed)
                            else np.asarray(cpu_seed)
                        )
                    attempt_context = cpu_fallback_context
                    attempt_start = cpu_start
                    attempt_state = None
                    fallback_reason = "dense_cpu_after_solver_resource_limit"
                    fallback_backend = "admm_complete_graph_cpu_fallback"
                    policy_state.result = None
                    policy_state.resource_error = None
                    policy_state.runtime_device_type = "cpu"
                    policy_state.representation_retry_done = True
                case NextAction.ACCEPT:
                    artifacts = policy_state.result
                    if artifacts is None:
                        raise AssertionError("Accepted policy state lacks a fit.")
                    if context_prepared_by_cpu_fallback:
                        artifacts = record_attempt(
                            artifacts,
                            reason="dense_cpu_after_context_resource_limit",
                            backend_name="admm_complete_graph_cpu_fallback",
                        )
                    artifacts_context = attempt_context
                    break
                case NextAction.FAIL:
                    if policy_state.resource_error is None:
                        raise ExactSolverResourceLimit(
                            "exact_solver_resource_limit: quotient/workset did not "
                            "produce an accepted terminal observed-objective "
                            "certificate and dense fallback is disabled by policy."
                        )
                    resource_exc = policy_state.resource_error
                    if isinstance(resource_exc, ExactSolverResourceLimit):
                        raise resource_exc
                    raise ExactSolverResourceLimit(
                        "exact_solver_resource_limit: exact solver allocation "
                        f"exhausted memory on {effective_runtime.device_name}."
                    ) from resource_exc
                case NextAction.FLOAT64_POLISH:
                    raise AssertionError("Working-fit policy requested polishing.")
        start_artifacts.append(artifacts)
        start_contexts.append(artifacts_context)
        if best_artifacts is None:
            best_artifacts = artifacts
            best_artifacts_index = len(start_artifacts) - 1
            continue
        if _prefer_multistart_fit(artifacts, best_artifacts):
            best_artifacts = artifacts
            best_artifacts_index = len(start_artifacts) - 1

    if best_artifacts is None:
        raise RuntimeError("No valid start produced a fusion fit.")
    if best_artifacts_index < 0:
        raise AssertionError("Best multistart fit lacks a source context.")
    selected_start_context = start_contexts[best_artifacts_index]
    working_artifacts = best_artifacts
    precision_context: SolverContext | None = None
    precision_on_cpu = False
    policy_state = PolicyState(
        phase="selected",
        result=best_artifacts,
        runtime_device_type=selected_start_context.runtime.device.type,
        fallback_policy=normalized_fallback_policy,
    )
    while True:
        action = decide_next_action(policy_state)
        match action:
            case NextAction.FLOAT64_POLISH:
                if not can_launch_another_solver_attempt():
                    best_artifacts = working_artifacts
                    break
                try:
                    precision_context = _float64_context(data, selected_start_context)
                except (MemoryError, torch.OutOfMemoryError) as polish_exc:
                    policy_state.phase = "precision_polish"
                    policy_state.resource_error = polish_exc
                    continue
                policy_state.phase = "precision_polish"
                policy_state.result = None
                policy_state.resource_error = None
            case NextAction.RETRY_SAME_RUNTIME:
                if precision_context is None:
                    raise AssertionError("Precision retry lacks a promoted context.")
                if not can_launch_another_solver_attempt():
                    best_artifacts = working_artifacts
                    break
                try:
                    polished = solve_start_once(
                        context=precision_context,
                        start=torch.tensor(
                            working_artifacts.phi,
                            dtype=torch.float64,
                            device=precision_context.runtime.device,
                        ),
                        state=working_artifacts.state,
                        polish=True,
                    )
                except (MemoryError, torch.OutOfMemoryError) as polish_exc:
                    policy_state.result = None
                    policy_state.resource_error = polish_exc
                    continue
                polished = _finalize_precision_polish(
                    polished,
                    working_artifacts,
                    selected_start_context,
                    on_cpu=precision_on_cpu,
                )
                policy_state.result = polished
                policy_state.resource_error = None
            case NextAction.CPU_FALLBACK:
                polish_exc = policy_state.resource_error
                if polish_exc is None:
                    raise AssertionError("CPU polish lacks a resource failure.")
                cpu_device = resolve_runtime("cpu", dtype="float64").device
                precision_context = _float64_context(
                    data,
                    selected_start_context,
                    device=cpu_device,
                    cause=polish_exc,
                )
                precision_on_cpu = True
                policy_state.result = None
                policy_state.resource_error = None
                policy_state.runtime_device_type = "cpu"
            case NextAction.ACCEPT:
                if policy_state.result is None:
                    raise AssertionError("Accepted precision state lacks a fit.")
                best_artifacts = policy_state.result
                break
            case NextAction.FAIL:
                if policy_state.resource_error is None:
                    raise AssertionError("Precision policy failed without a cause.")
                raise policy_state.resource_error
            case NextAction.DENSE_CURRENT_DEVICE:
                raise AssertionError("Precision policy requested a dense retry.")
    return replace(best_artifacts, work=attempted_work.total)
