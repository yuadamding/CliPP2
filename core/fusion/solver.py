from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import torch

from ...io.data import (
    TumorData,
    tumor_data_fingerprint,
)
from ..objective import (
    TorchObservedModel,
    TorchObservedTerms,
    compile_observed_model,
    has_proven_convex_observed_loss,
    make_base_objective_key,
    make_lambda_objective_key,
    model_to_torch,
    observed_internal_breakpoints_torch,
    observed_one_sided_gradients_torch,
    observed_em_terms_torch,
    observed_terms_torch,
)
from ...config import (
    CLONAL_INTEGER_MODEL_ID,
    _FitOptions,
    SolverConfig,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
)
from .certificates import (
    CertificateProblem,
    build_certificate_gradient,
    certify,
)
from .graph import resolve_pairwise_fusion_graph
from .graph_ops import (
    build_complete_adaptive_tensor_graph,
    dense_complete_solver_memory_preflight,
    graph_adjoint_edges,
    project_dual_ball,
    tensor_graph_to_pairwise_graph,
    tensorize_graph,
)
from .policy import (
    needs_precision_polish,
    record_attempt,
)
from .starts import (
    compute_pooled_observed_data_start_torch,
    compute_scalar_mutation_region_wells_torch,
    compute_scalar_well_start_bank_torch,
)
from .torch_backend import (
    as_runtime_tensor,
    dtype_name,
    graph_adjoint_edges_in_dtype,
    graph_fusion_kkt_residual_from_grad_torch,
    pairwise_penalty_torch,
    resolve_runtime,
    solve_majorized_subproblem_alm_torch,
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
    PreparedProblem,
    SolverState,
    TensorFusionGraph,
    TorchRuntime,
    WorkCounters,
    WorksetMemoryOptions,
)


@dataclass(frozen=True, slots=True)
class _Float64AuditContext:
    runtime: TorchRuntime
    model: TorchObservedModel
    graph: TensorFusionGraph

    @property
    def lower(self) -> torch.Tensor:
        return self.model.lower

    @property
    def upper(self) -> torch.Tensor:
        return self.model.upper


def _float64_audit_context(
    problem: PreparedProblem,
) -> _Float64AuditContext:
    """Return the immutable float64 audit tensors for one tumor/graph/device."""

    problem.assert_runtime_unchanged()
    source_model = problem.source_model
    if source_model is None:
        raise ValueError("Float64 audit requires an immutable observed-model source.")
    device = problem.runtime.device
    key = (str(source_model.fingerprint), str(problem.graph_hash), str(device))
    cached = problem.audit_context_cache.get(key)
    if cached is not None:
        if not isinstance(cached, _Float64AuditContext):
            raise TypeError("PreparedProblem float64 audit cache is corrupted.")
        return cached
    runtime = TorchRuntime(
        device=device,
        device_name=str(device),
        dtype=torch.float64,
    )
    context = _Float64AuditContext(
        runtime=runtime,
        model=model_to_torch(source_model, runtime, eps=problem.eps),
        graph=tensorize_graph(
            problem.graph_spec,
            runtime,
            num_nodes=int(source_model.shape[0]),
        ),
    )
    problem.audit_context_cache[key] = context
    return context


def _terminal_backward_error_audit_float64(
    *,
    problem: PreparedProblem,
    phi: torch.Tensor,
    certificate: GraphFusionCertificate | None,
    lambda_value: float,

    tol: float,
) -> tuple[KKTDiagnostics, str, bool, float]:
    """Audit the unchanged terminal witness with float64 backward error."""

    audit = _float64_audit_context(problem)
    eps, graph_hash = problem.eps, problem.graph_hash
    data64 = audit.model
    graph64 = audit.graph
    lower64 = audit.lower
    upper64 = audit.upper
    phi64 = phi.to(dtype=torch.float64, device=audit.runtime.device)
    terms64 = observed_terms_torch(
        data64,
        phi64,
        eps=float(eps),
    )
    gradient = build_certificate_gradient(
        data64,
        phi=phi64,
        smooth_gradient=terms64.gradient,
        lower=lower64,
        upper=upper64,
        eps=float(eps),
        tol=float(tol),
    )
    dense_dual = getattr(certificate, "dual", None)
    if torch.is_tensor(dense_dual) and bool(
        torch.any(gradient.at_breakpoint).item()
    ):
        adjustment = graph_adjoint_edges_in_dtype(
            dense_dual,
            edge_u=graph64.edge_u,
            edge_v=graph64.edge_v,
            num_nodes=int(phi64.shape[0]),
            dtype=torch.float64,
            device=audit.runtime.device,
        )
        gradient = build_certificate_gradient(
            data64,
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
    _, _, objective64 = (
        _objective_value_from_mutation_region_terms_torch(
            terms64,
            phi64,
            edge_u=graph64.edge_u,
            edge_v=graph64.edge_v,
            edge_w=graph64.weight,
            lambda_value=float(lambda_value),
        )
    )
    return (
        result.diagnostics,
        gradient.scope,
        gradient.directional_admissible,
        float(objective64),
    )


@dataclass(frozen=True, slots=True)
class _StartAttempt:
    phi: np.ndarray | torch.Tensor
    warm_state: SolverState | None
    source: str


def _deduplicate_start_attempts(
    attempts: list[_StartAttempt],
    *,
    runtime,
    shape: tuple[int, ...],
    atol: float = 1e-8,
) -> list[_StartAttempt]:
    unique: list[_StartAttempt] = []
    unique_tensors: list[torch.Tensor] = []
    for attempt in attempts:
        start_tensor = as_runtime_tensor(attempt.phi, runtime).detach()
        if tuple(start_tensor.shape) != shape or not bool(torch.isfinite(start_tensor).all()):
            raise ValueError(f"Invalid {attempt.source} start: expected finite CCFs with shape {shape}.")
        duplicate = any(
            attempt.warm_state is previous.warm_state
            and torch.allclose(start_tensor, retained, rtol=0.0, atol=float(atol))
            for previous, retained in zip(unique, unique_tensors)
        )
        if duplicate:
            continue
        unique_tensors.append(start_tensor)
        unique.append(attempt)
    return unique


def _clipped_singleton_start_needs_pilot(
    problem: PreparedProblem, start: np.ndarray | torch.Tensor,
) -> bool:
    """Detect a flat clipping start with feasible downhill likelihood beyond it."""
    model = problem.source_model
    if model is None:
        raise ValueError("PreparedProblem lacks an immutable observed-model source.")
    working_phi = as_runtime_tensor(start, problem.runtime).detach()
    working_phi = torch.minimum(torch.maximum(working_phi, problem.lower), problem.upper)
    working_mass = problem.model.slope[..., 0] * working_phi
    phi = working_phi.cpu().numpy()
    phi = np.clip(phi, model.lower, model.upper)
    slope = model.slope[..., 0]
    mass = slope * phi
    eps = float(problem.eps)
    upper_clip = 1.0 - eps
    active = model.observed & ((model.alt > 0) | (model.nonalt > 0))
    active &= np.sum(model.valid, axis=-1) == 1
    # Rounded working slopes can land on a clipping plateau even when the
    # float64 source mass is just outside it. Compare both views; only starts
    # are added, never a shifted clipping threshold or a different objective.
    lower_clipped = (mass <= eps) | (working_mass <= eps).cpu().numpy()
    upper_clipped = (mass >= upper_clip) | (working_mass >= upper_clip).cpu().numpy()
    lower_escape = (lower_clipped & (slope * model.upper > eps)
                    & (model.alt * (1.0 - eps) > eps * model.nonalt))
    upper_escape = (upper_clipped & (slope * model.lower < upper_clip)
                    & (model.nonalt * upper_clip > (1.0 - upper_clip) * model.alt))
    return bool(np.any(active & (lower_escape | upper_escape)))


def _inner_model_value_torch(
    phi: torch.Tensor,
    *,
    U: torch.Tensor,
    h: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:
    quad = 0.5 * torch.sum(h * torch.square(phi - U))
    penalty = pairwise_penalty_torch(
        phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
    )
    return quad + penalty


def _objective_value_from_mutation_region_terms_torch(
    mutation_region_terms,
    phi: torch.Tensor,
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
) -> tuple[float, float, float]:
    fit_loss_tensor = torch.sum(mutation_region_terms.loss)
    penalty_tensor = pairwise_penalty_torch(
        phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
    )
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
_CONVEX_GLOBAL_OPTIMALITY_BASIS = "convex_clipped_singleton_objective_plus_kkt_v1"
OBJECTIVE_SHAPE_AUTO = "auto"
INTEGER_MIXTURE_OBJECTIVE_SHAPE = "generic_nonconvex"


def has_multiplicity_ambiguity(data: TumorData) -> bool:
    """Structural candidate ambiguity, not a convexity assertion."""
    return bool(np.any(np.asarray(data.major_cn) > 1))


def objective_shape_for_data(data: TumorData, requested: str) -> str:
    """Choose the numerical update route, independently of convexity proof.

    Competing integer emissions can be multimodal and use the generic route.
    Singleton emissions keep their scalar update route, but clipping can still
    make their loss nonconvex. Only the frozen-model proof authorizes a global
    certificate; a requested shape cannot authorize one.
    """

    normalized = _normalize_objective_shape(requested)
    if has_multiplicity_ambiguity(data):
        return INTEGER_MIXTURE_OBJECTIVE_SHAPE
    return "unimodal" if normalized == OBJECTIVE_SHAPE_AUTO else normalized


def _clipping_smooth_interval_bounds(
    model: TorchObservedModel,
    phi: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict one MM trial to the current smooth clipping interval.

    The kernel returns the left derivative at an exact breakpoint, so an
    exact breakpoint is treated as the upper end of its left interval.
    The start bank separately seeds both sides of nearby clipping kinks.
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
    surrogate_terms,
    count_observed: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    h_base = torch.clamp(surrogate_terms.hessian_upper, min=_MISSING_SURROGATE_CURVATURE)
    surrogate_grad = surrogate_terms.gradient
    if count_observed is None:
        return h_base, surrogate_grad

    observed = count_observed
    h_base = torch.where(
        observed,
        h_base,
        torch.full_like(h_base, _MISSING_SURROGATE_CURVATURE),
    )
    surrogate_grad = torch.where(
        observed, surrogate_grad, torch.zeros_like(surrogate_grad)
    )
    return h_base, surrogate_grad


def _safe_majorized_center(
    phi: torch.Tensor,
    *,
    surrogate_grad: torch.Tensor,
    h: torch.Tensor,
    count_observed: torch.Tensor | None,
) -> torch.Tensor:
    U_raw = phi - surrogate_grad / h
    if count_observed is None:
        return U_raw
    return torch.where(count_observed, U_raw, phi)


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
) -> None:
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
    for value in (
        tensor_graph.edge_index,
        tensor_graph.weight,
        tensor_graph.degree,
    ):
        if value.device.type != runtime.device.type or (
            runtime.device.index is not None
            and value.device.index != runtime.device.index
        ):
            raise ValueError("prebuilt_tensor_graph is not on the runtime device.")
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
) -> tuple[None, None, None, PrimalOnlyWarmState, bool]:
    """Create the only state that may be promoted for a damped MM endpoint."""

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
    return (
        None,
        None,
        None,
        PrimalOnlyWarmState(
            phi=phi,
            structure_hint=structure_hint,
            certificate_hint=certificate_hint,
        ),
        False,
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


def transfer_scalar_pilot_certificates(
    source: PreparedProblem,
    target: PreparedProblem,
) -> PreparedProblem:
    """Preserve scalar evidence only for the identical model and pilot."""

    if not source.scalar_pilot_certificates:
        return target
    source_model = source.source_model
    target_model = target.source_model
    if (
        source_model is None
        or target_model is None
        or source_model.fingerprint != target_model.fingerprint
        or source_model.fingerprint != source.model.source_fingerprint
        or target_model.fingerprint != target.model.source_fingerprint
        or float(source.eps) != float(target.eps)

        or not torch.equal(
            source.exact_pilot.detach().cpu().double(),
            target.exact_pilot.detach().cpu().double(),
        )
    ):
        raise ValueError("Cannot transfer scalar certificates to a changed pilot/model.")
    return replace(target, scalar_pilot_certificates=source.scalar_pilot_certificates)


def promote_solver_context_dtype(
    context: PreparedProblem,
    *,
    dtype: torch.dtype,
) -> PreparedProblem:
    """Rebuild one frozen objective from its immutable host sources."""

    context.assert_runtime_unchanged()
    if dtype not in {torch.float32, torch.float64}:
        raise ValueError("Promoted solver contexts require float32 or float64.")
    target_device = context.runtime.device
    if context.runtime.dtype == dtype:
        return context
    source_model = context.source_model
    if source_model is None:
        raise ValueError("PreparedProblem lacks an immutable observed-model source.")
    runtime = replace(
        context.runtime,
        dtype=dtype,
        device=target_device,
        device_name=str(target_device),
    )
    promoted_model = model_to_torch(source_model, runtime, eps=context.eps)
    graph = tensorize_graph(
        context.graph_spec,
        runtime,
        num_nodes=int(source_model.shape[0]),
    )
    exact = context.exact_pilot.to(dtype=dtype, device=target_device)
    pooled = context.pooled_start.to(dtype=dtype, device=target_device)
    wells = tuple(start.to(dtype=dtype, device=target_device) for start in context.scalar_well_starts)
    return replace(
        context,
        _tensor_snapshot=(),
        model=promoted_model,
        graph=graph,
        exact_pilot=exact,
        pooled_start=pooled,
        scalar_well_starts=wells,
        scalar_pilot_certificates=context.scalar_pilot_certificates,
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


def _float64_context(data: TumorData, context: PreparedProblem) -> PreparedProblem:
    runtime = replace(context.runtime, dtype=torch.float64)
    if context.graph.is_complete:
        _require_dense_memory(data, runtime, operation="float64 fixed-objective precision polish",
                              limit_name="policy limit")
    return promote_solver_context_dtype(context, dtype=torch.float64)


def _finalize_precision_polish(
    polished: RawFit,
    working: RawFit,
    source_context: PreparedProblem,
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
    if not np.isfinite(polished.objective.total) or polished.objective.total > objective + slack:
        raise AssertionError("Float64 fixed-objective polishing increased the objective.")
    delta = float(
        np.max(
            np.abs(
                np.asarray(polished.phi, dtype=np.float64)
                - np.asarray(working.phi, dtype=np.float64)
            )
        )
    )
    reason = "float64_fixed_objective_precision_polish"
    polished = record_attempt(
        polished,
        attempted=working,
        reason=reason,
    )
    return replace(
        polished,
        provenance=replace(
            polished.provenance,
            scalar_pilot_certificates=working.provenance.scalar_pilot_certificates,
        ),
        certificate=replace(
            polished.certificate,
            working_residual=working.certificate.working_residual,
            working_dtype=working.provenance.dtype,
            audit_dtype="float64",
            precision_polished=True,
            precision_polish_delta=delta,
        ),
    )


def escape_emission_breakpoint_solver_state(
    state: SolverState | None,
    *,
    context: PreparedProblem,
    tol: float,
) -> tuple[SolverState | None, int]:
    """Nudge a failed dense-certificate state off exact path breakpoints.

    The dense certificate supplies the fusion adjoint.  At each exact
    breakpoint, the one-sided observed gradients plus that adjoint choose a
    retry side.  A changed primal invalidates every dual/certificate warm
    object because none remains valid at the new point.
    """

    certificate = None if state is None else state.certificate
    model = context.model
    if (
        state is None

        or not isinstance(certificate, DenseEdgeCertificate)
        or certificate.certificate_scope != "full_original_graph"
        or certificate.gradient_scope == "mm_surrogate"
        or certificate.graph_hash != str(context.graph_hash)
        or not torch.is_tensor(certificate.dual)
    ):
        return state, 0

    tolerance = _validate_solver_tolerance(tol)
    phi = as_runtime_tensor(state.phi, context.runtime)
    expected_shape = model.shape
    if tuple(phi.shape) != expected_shape or not bool(torch.all(torch.isfinite(phi))):
        return state, 0
    dual = as_runtime_tensor(certificate.dual, context.runtime)
    expected_dual_shape = (
        int(context.graph.edge_u.numel()),
        int(phi.shape[1]),
    )
    if tuple(dual.shape) != expected_dual_shape or not bool(
        torch.all(torch.isfinite(dual))
    ):
        return state, 0

    with torch.no_grad():
        fusion_adjustment = graph_adjoint_edges(
            dual,
            edge_u=context.graph.edge_u,
            edge_v=context.graph.edge_v,
            num_nodes=int(phi.shape[0]),
        )
        gradient_left, gradient_right, at_breakpoint = (
            observed_one_sided_gradients_torch(
                model,
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
            at_breakpoint & (phi > context.lower) & (left_total > direction_threshold)
        )
        right_descends = (
            at_breakpoint & (phi < context.upper) & (right_total < -direction_threshold)
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
        escaped = torch.minimum(torch.maximum(escaped, context.lower), context.upper)
        changed = escaped != phi
        changed_count = int(torch.count_nonzero(changed).item())
        if changed_count == 0:
            return state, 0

    return (
        replace(
            state,
            phi=escaped.detach(),
            dual=None,
            warm_state=None,
            certificate=None,
        ),
        changed_count,
    )


def prepare_torch_problem(
    data: TumorData,
    *,

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
    verbose: bool = False,
) -> PreparedProblem:
    tol = _validate_solver_tolerance(tol)
    objective_shape = objective_shape_for_data(data, objective_shape)

    use_unimodal_objective = objective_shape.startswith("unimodal")
    effective_runtime = (
        resolve_runtime(device, dtype=dtype) if runtime is None else runtime
    )
    source_model = compile_observed_model(
        data,
        eps=float(eps),
    )
    runtime_model = model_to_torch(source_model, effective_runtime, eps=float(eps))
    data_fingerprint = tumor_data_fingerprint(data)

    pilot_certificates = (
        [] if exact_pilot is None and source_model.model_id == CLONAL_INTEGER_MODEL_ID
        else None
    )
    if exact_pilot is None:
        exact_pilot_tensor, secondary_wells, valid_secondary = (
            compute_scalar_mutation_region_wells_torch(
                source_model, effective_runtime,
                phi_init=data.phi_init,
                eps=float(eps),
                tol=tol,
                max_iter=max(int(inner_max_iter), 16),
                certificates=pilot_certificates,
            )
        )
    else:
        exact_pilot_tensor = as_runtime_tensor(exact_pilot, effective_runtime)
        if scalar_well_starts is None and not use_unimodal_objective:
            _, secondary_wells, valid_secondary = (
                compute_scalar_mutation_region_wells_torch(
                    source_model, effective_runtime,
                    phi_init=data.phi_init,
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
        _validate_prebuilt_tensor_graph(
            effective_graph,
            prebuilt_tensor_graph,
            runtime=effective_runtime,
            num_nodes=data.num_mutations,
        )
        tensor_graph = prebuilt_tensor_graph
    elif graph is None:
        working_tensor_graph = build_complete_adaptive_tensor_graph(
            exact_pilot_tensor,
            effective_runtime,
            count_observed=runtime_model.observed,
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
            source_model, effective_runtime,
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
            runtime_model,
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
    return PreparedProblem(
        source_data=data,
        source_model=source_model,
        model=runtime_model,
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
        verbose=bool(verbose),
        scalar_pilot_certificates=tuple(pilot_certificates or ()),
        adaptive_graph_options=(
            (float(adaptive_weight_gamma), float(adaptive_weight_floor),
             float(adaptive_weight_baseline))
            if graph is None and not defer_graph else None
        ),
    )


def prepare_torch_problem_with_resource_policy(
    data: TumorData, options: _FitOptions, **prepare_overrides,
) -> PreparedProblem:
    """Prepare on the requested device; resource failure never changes devices."""
    kwargs = dict(
        eps=options.eps, tol=options.solver.tolerance,
        inner_max_iter=max(options.solver.inner_max_iter, 16),
        graph=options.graph.graph,
        adaptive_weight_gamma=options.graph.adaptive_weight_gamma,
        adaptive_weight_floor=options.graph.adaptive_weight_floor,
        adaptive_weight_baseline=options.graph.adaptive_weight_baseline,
        device=options.runtime.device, dtype=options.runtime.dtype,
        objective_shape=options.solver.objective_shape,
        verbose=options.runtime.verbose,
    )
    kwargs.update(prepare_overrides)
    runtime = kwargs.pop("runtime", None)
    if runtime is None:
        runtime = resolve_runtime(kwargs["device"], dtype=kwargs["dtype"])
    try:
        return prepare_torch_problem(data, runtime=runtime, **kwargs)
    except (MemoryError, torch.OutOfMemoryError) as exc:
        if isinstance(exc, ExactSolverResourceLimit):
            raise
        raise ExactSolverResourceLimit(
            "exact_solver_resource_limit: exact problem or graph construction "
            f"exhausted memory on {runtime.device_name}."
        ) from exc


def _backward_error_kkt_within_gate(
    diagnostics: KKTDiagnostics,
    *,
    certification_tol: float,
) -> bool:
    """Apply the immutable full-KKT gate to the componentwise residual.

    The legacy globally normalized L2 residual remains available for progress
    reporting, but its dimension-dependent scaling cannot terminate an outer
    solve whose authoritative componentwise backward error is still too high.
    Unset and nonfinite schema-v2 diagnostics fail closed.
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
    tolerances but remains disabled if float64 context
    promotion failed.
    """

    return bool(requested and runtime_dtype == torch.float64)


def _solve_inner_subproblem(
    *,
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
    tol: float,
    inner_max_iter: int,
    phi: torch.Tensor,
    dual,
    dual_start_is_actual: bool,
    spectral_rho: bool,
    use_backward_error_stopping: bool,
    backend_name: str,
    graph_hash: str,
) -> InnerSolveResult:
    """Solve the complete-graph ALM subproblem and retain its actual multiplier."""
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
    (
        phi_trial,
        dual_kkt_trial,
        _inner_iterations,
        inner_ok,
        surrogate_diag,
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
        max_iter=max(inner_max_iter, 10),
        phi_start=phi,
        dual_start=dual,
        dual_start_is_actual=dual_start_is_actual,
        spectral_rho=bool(spectral_rho),
        use_backward_error_stopping=bool(use_backward_error_stopping),
    )
    if surrogate_diag is None:
        surrogate_diag = graph_fusion_kkt_residual_from_grad_torch(
            phi=phi_trial,
            grad_smooth=h * (phi_trial - U),
            dual_kkt=dual_kkt_trial,
            lower=lower,
            upper=upper,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            lambda_value=lambda_value,
            atol=tol,
        )
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
            dual=dual_kkt_trial if torch.is_tensor(dual_kkt_trial) else None,
            previous_lambda=float(lambda_value),
            graph_hash=str(graph_hash),
        ),
        surrogate_certificate=certificate,
        surrogate_kkt=surrogate_diag,
        converged=bool(inner_ok),
        iterations=int(_inner_iterations),
    )


@lru_cache(maxsize=32)
def _certificate_options(options: SolverConfig, dtype: torch.dtype) -> CertificateOptions:
    """Resolve one immutable certificate configuration per solve config/dtype."""
    cert_tol = _validate_solver_tolerance(
        options.tolerance if options.certification_tolerance is None
        else options.certification_tolerance
    )
    resources, certificate = options.resources, options.certificate
    return CertificateOptions(
        max_iter=max(int(certificate.max_iter), 1),
        refinement_rounds=max(int(certificate.refinement_rounds), 0),
        max_expansions=max(int(resources.workset_max_expansions), 1),
        add_batch=max(int(resources.workset_add_batch), 1),
        mapping_tolerance=max(0.1 * cert_tol, float(torch.finfo(dtype).eps)),
        column_tolerance=max(
            float(certificate.column_tolerance_scale) * cert_tol,
            float(torch.finfo(dtype).eps),
        ),
        memory=WorksetMemoryOptions(
            max_workset_bytes=int(resources.workset_max_bytes),
            max_compressed_cache_bytes=int(resources.compressed_cache_max_bytes),
        ),
    )


@dataclass(slots=True)
class _IterateState:
    """A primal point and only the dual/certificate state valid at that point.

    Records hold existing tensors, never clone dense edge arrays. A rejected
    proposal retains the old record; damping creates a primal-only record.
    """
    phi: torch.Tensor
    dual: torch.Tensor | None
    dual_kkt: torch.Tensor | None
    certificate: DenseEdgeCertificate | CompressedEdgeCertificate | None
    warm_state: DenseWarmState | PrimalOnlyWarmState
    backend_name: str
    dual_start_is_actual: bool
    objective: float
    fit_loss: float
    terms: TorchObservedTerms


def _fit_from_start(
    problem: PreparedProblem,
    lambda_value: float,
    options: SolverConfig,
    attempt: _StartAttempt,
) -> RawFit:
    data, model, runtime = problem.source_data, problem.model, problem.runtime
    tensor_graph = problem.graph
    lower, upper = problem.lower, problem.upper
    graph_hash, objective_spec_hash = problem.graph_hash, problem.objective_spec_hash
    base_objective_key = problem.base_objective_key
    eps, verbose = float(problem.eps), bool(problem.verbose)
    phi_start, solver_state = attempt.phi, attempt.warm_state
    tol = _validate_solver_tolerance(options.tolerance)
    cert_tol = _validate_solver_tolerance(
        tol if options.certification_tolerance is None else options.certification_tolerance
    )
    outer_max_iter = max(int(options.outer_max_iter), 1)
    inner_max_iter = max(int(options.inner_max_iter), 16)
    if (
        solver_state is not None
        and str(solver_state.objective_spec_hash)
        and str(solver_state.objective_spec_hash) != objective_spec_hash
    ):
        raise ValueError("Solver warm state belongs to a different raw objective.")
    objective_shape = objective_shape_for_data(data, options.objective_shape)
    certificate_options = _certificate_options(options, runtime.dtype)
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
    use_backward_error_progress = _uses_backward_error_progress(
        requested=bool(options.use_backward_error_progress),
        runtime_dtype=runtime.dtype,
    )
    edge_u, edge_v, edge_w = (
        tensor_graph.edge_u,
        tensor_graph.edge_v,
        tensor_graph.weight,
    )
    if lambda_value <= 0.0 or int(edge_u.numel()) == 0:
        dense_inner_solver = "closed_form_projection"
    else:
        dense_inner_solver = "admm_complete_graph"
    inner_solver = dense_inner_solver
    if (
        solver_state is not None
        and solver_state.phi is not None
        and tuple(solver_state.phi.shape) == tuple(model.upper.shape)
    ):
        phi = solver_state.phi.to(dtype=runtime.dtype, device=runtime.device)
    else:
        phi = as_runtime_tensor(phi_start, runtime)
    phi = torch.minimum(torch.maximum(phi, lower), upper)

    state_dual = _project_state_dual(
        solver_state,
        runtime=runtime,
        edge_w=edge_w,
        lambda_value=lambda_value,
        num_edges=int(edge_u.numel()),
        num_regions=int(phi.shape[1]),
    )
    dual = state_dual
    dual_kkt = state_dual
    warm_state = (
        solver_state.warm_state
        if solver_state is not None and solver_state.warm_state is not None
        else DenseWarmState(
            phi=phi,
            dual=state_dual,
            previous_lambda=float(lambda_value),
            graph_hash=str(graph_hash),
        )
    )
    certificate = (
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
    dual_start_is_actual = bool(state_dual is not None)
    converged = False
    converged_outer = False
    iterations = 0
    work_counters = WorkCounters()
    current_inner_converged = False
    outer_kkt_certificate_status = "not_audited"
    mm_consistency_violations = 0
    total_inner_iterations = 0
    inner_solve_calls = 0
    accepted_full_steps = 0
    accepted_damped_steps = 0
    rejected_outer_steps = 0
    outer_stop_reason = "outer_iteration_limit"
    legacy_stop_kkt_residual = float("inf")
    componentwise_stop_kkt_residual = float("inf")
    progress_residual_method = (
        "componentwise_box_cone_backward_error_v1"
        if use_backward_error_progress
        else "legacy_global_l2_progress_v1"
    )
    full_step_curvature_multiplier = torch.ones_like(phi)

    current_mutation_region_terms = observed_terms_torch(
        model, phi, eps=eps
    )
    fit_loss, penalty, objective = (
        _objective_value_from_mutation_region_terms_torch(
            current_mutation_region_terms,
            phi,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            lambda_value=lambda_value,
        )
    )
    iterate = _IterateState(
        phi, dual, dual_kkt, certificate, warm_state, inner_solver,
        dual_start_is_actual, objective, fit_loss, current_mutation_region_terms,
    )
    del phi, dual, dual_kkt, certificate, warm_state, inner_solver
    del dual_start_is_actual, objective, fit_loss, current_mutation_region_terms
    for outer_iter in range(max(int(outer_max_iter), 1)):
        iterations = outer_iter + 1
        previous_phi = iterate.phi.clone()
        previous_objective = iterate.objective
        if use_unimodal_objective:
            surrogate_terms = iterate.terms
            surrogate_fit_loss = float(iterate.fit_loss)
        else:
            responsibilities = iterate.terms.posterior
            if responsibilities is None:
                raise AssertionError("Observed terms lack candidate responsibilities.")
            surrogate_terms = observed_em_terms_torch(
                model,
                iterate.phi,
                responsibilities=responsibilities,
                eps=eps,
            )
            surrogate_fit_loss = float(torch.sum(surrogate_terms.loss).item())
        h_base, surrogate_grad = _safe_surrogate_curvature_and_gradient(
            surrogate_terms,
            model.observed,
        )
        if use_unimodal_objective:
            smooth_lower, smooth_upper = lower, upper
        else:
            smooth_lower, smooth_upper = _clipping_smooth_interval_bounds(
                model,
                iterate.phi,
                lower=lower,
                upper=upper,
                eps=float(eps),
            )
        if require_full_step_backtracking:
            forcing_certificate = iterate.certificate
            if forcing_certificate is None:
                forcing_certificate = _compressed_certificate_for_primal(
                    iterate.phi,
                    graph_hash=graph_hash,
                    gradient_scope="observed_objective",
                )
            forcing_gradient = build_certificate_gradient(
                model,
                phi=iterate.phi,
                smooth_gradient=iterate.terms.gradient,
                lower=lower,
                upper=upper,
                eps=eps,
                tol=cert_tol,
            )
            forcing_diag = certify(
                problem=certificate_problem,
                phi=iterate.phi,
                gradient=forcing_gradient,
                witness=forcing_certificate,
                refine=False,
            ).diagnostics
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
        candidate = iterate
        inner_converged = False

        curvature_attempts = (
            _FULL_STEP_MAX_CURVATURE_ATTEMPTS
            if require_full_step_backtracking
            else (1 if use_unimodal_objective else 10)
        )
        for _curvature_attempt in range(curvature_attempts):
            h = (
                h_base * curvature_multiplier
                if require_full_step_backtracking
                else h_base * scale
            )
            U = _safe_majorized_center(
                iterate.phi,
                surrogate_grad=surrogate_grad,
                h=h,
                count_observed=model.observed,
            )
            if use_unimodal_objective and not require_full_step_backtracking:
                q_current = None
            else:
                q_current = _inner_model_value_torch(
                    iterate.phi,
                    U=U,
                    h=h,
                    edge_u=edge_u,
                    edge_v=edge_v,
                    edge_w=edge_w,
                    lambda_value=lambda_value,
                )
            recovery_inner_model_tol = (
                max(
                    64.0 * float(torch.finfo(iterate.phi.dtype).eps),
                    float(tol) ** 2,
                )
                * (1.0 + abs(float(q_current.item())))
                if require_full_step_backtracking
                else 0.0
            )
            inner_phi_start = iterate.phi
            inner_dual_start = iterate.dual
            inner_dual_start_is_actual = iterate.dual_start_is_actual
            inner_batch_limit = 8 if require_full_step_backtracking else 1
            for _inner_batch in range(inner_batch_limit):
                inner_result = _solve_inner_subproblem(
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
                    tol=tol,
                    inner_max_iter=inner_max_iter,
                    phi=inner_phi_start,
                    dual=inner_dual_start,
                    dual_start_is_actual=inner_dual_start_is_actual,
                    spectral_rho=bool(require_full_step_backtracking),
                    use_backward_error_stopping=use_backward_error_progress,
                    backend_name=dense_inner_solver,
                    graph_hash=graph_hash,
                )
                inner_solve_calls += 1
                total_inner_iterations += int(inner_result.iterations)
                phi_trial = inner_result.phi
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
                    batch_inner_certified = bool(
                        np.isfinite(float(inner_residual))
                        and float(inner_residual) <= inner_progress_tolerance
                    )
                    batch_q_trial = _inner_model_value_torch(
                        phi_trial,
                        U=U,
                        h=h,
                        edge_u=edge_u,
                        edge_v=edge_v,
                        edge_w=edge_w,
                        lambda_value=lambda_value,
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
                inner_dual_start = dual_kkt_trial
                inner_dual_start_is_actual = True
            delta = phi_trial - iterate.phi
            trial_mutation_region_terms = observed_terms_torch(
                model, phi_trial, eps=eps
            )
            trial_fit_loss, _, trial_objective = (
                _objective_value_from_mutation_region_terms_torch(
                    trial_mutation_region_terms,
                    phi_trial,
                    edge_u=edge_u,
                    edge_v=edge_v,
                    edge_w=edge_w,
                    lambda_value=lambda_value,
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
                        surrogate_terms.gradient * delta + 0.5 * h * torch.square(delta)
                    ).item()
                )
                majorizer_rhs = surrogate_fit_loss + quadratic_gap
                q_trial = _inner_model_value_torch(
                    phi_trial,
                    U=U,
                    h=h,
                    edge_u=edge_u,
                    edge_v=edge_v,
                    edge_w=edge_w,
                    lambda_value=lambda_value,
                )
                inner_model_gap = float((q_trial - q_current).item())
                if use_unimodal_objective:
                    # Recovery uses an exact smooth-loss majorization check.
                    # This makes the accepted full ADMM endpoint a valid
                    # proximal-MM update with a matching actual dual.
                    surrogate_gap = float(trial_fit_loss - majorizer_rhs)
                    em_envelope_gap = 0.0
                else:
                    trial_surrogate_terms = observed_em_terms_torch(
                        model,
                        phi_trial,
                        responsibilities=responsibilities,
                        eps=eps,
                    )
                    trial_surrogate_loss = float(
                        torch.sum(trial_surrogate_terms.loss).item()
                    )
                    surrogate_gap = float(trial_surrogate_loss - majorizer_rhs)
                    em_envelope_gap = float(
                        (trial_fit_loss - iterate.fit_loss)
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
                numerical_factor = 64.0 * float(torch.finfo(iterate.phi.dtype).eps)
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
            envelope_tol = 1e-8 * (1.0 + abs(iterate.fit_loss))
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
            if objective_gap <= recovery_armijo_rhs and (
                not audit_quadratic_majorizer or surrogate_gap <= majorization_tol
            ):
                accepted = True
                accepted_full_steps += 1
                # ALM returns actual y=rho*u; retain it across curvature changes.
                candidate = _IterateState(
                    phi_trial, dual_kkt_trial, dual_kkt_trial, surrogate_certificate,
                    inner_result.warm_state, inner_result.backend_name, True,
                    trial_objective, trial_fit_loss, trial_mutation_region_terms,
                )
                inner_converged = bool(
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
                # A failed majorization or Armijo check requires a new full
                # subproblem. A damped primal point does not share its
                # dual certificate. Enlarge the persistent majorizing
                # curvature and accept only a full proximal-MM/ADMM endpoint.
                # If the resource limit is exhausted, leave this outer iterate
                # unchanged and uncertified rather than interpolating phi.
                delta_square = torch.square(delta)
                resolution = torch.finfo(iterate.phi.dtype).eps * (1.0 + torch.square(iterate.phi))
                secant_remainder = (
                    trial_mutation_region_terms.loss
                    - iterate.terms.loss
                    - surrogate_grad * delta
                )
                required_h = torch.where(
                    delta_square > resolution,
                    2.0
                    * torch.clamp(secant_remainder, min=0.0)
                    / torch.clamp(
                        delta_square,
                        min=torch.finfo(iterate.phi.dtype).tiny,
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
                        * (1.0 + 64.0 * torch.finfo(iterate.phi.dtype).eps)
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
                phi_theta = iterate.phi + theta * delta
                theta_mutation_region_terms = observed_terms_torch(
                    model, phi_theta, eps=eps
                )
                theta_fit_loss, _, theta_objective = (
                    _objective_value_from_mutation_region_terms_torch(
                        theta_mutation_region_terms,
                        phi_theta,
                        edge_u=edge_u,
                        edge_v=edge_v,
                        edge_w=edge_w,
                        lambda_value=lambda_value,
                    )
                )
                if (
                    np.isfinite(theta_objective)
                    and theta_objective <= previous_objective + objective_tol
                ):
                    accepted = True
                    damped_accepted = True
                    accepted_damped_steps += 1
                    damped_state = _invalidate_damped_trial_state(
                        phi=phi_theta, trial_warm_state=inner_result.warm_state,
                    )
                    candidate = _IterateState(
                        phi_theta, *damped_state[:4], inner_result.backend_name,
                        damped_state[4], theta_objective, theta_fit_loss,
                        theta_mutation_region_terms,
                    )
                    inner_converged = False
                    break
                theta *= 0.5
            if damped_accepted:
                break
            scale *= 2.0

        if not accepted:
            rejected_outer_steps += 1
        iterate = candidate
        penalty = iterate.objective - iterate.fit_loss
        if verbose:
            print(
                f"[pairwise-fusion:{runtime.device_name}] iter={iterations:02d} objective={iterate.objective:.6f} "
                f"fit={iterate.fit_loss:.6f} penalty={penalty:.6f}"
            )

        rel_change = abs(previous_objective - iterate.objective) / (
            1.0 + abs(previous_objective)
        )
        step_residual = float(
            (
                torch.linalg.norm(iterate.phi - previous_phi)
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
            or not np.isfinite(iterate.objective)
        )
        outer_converged = False
        if do_outer_kkt_audit:
            outer_terms = iterate.terms
            observed_start = iterate.certificate
            if observed_start is None and isinstance(iterate.warm_state, PrimalOnlyWarmState):
                observed_start = _rebase_certificate_hint(
                    iterate.warm_state.certificate_hint,
                    phi=iterate.phi,
                    graph=tensor_graph,
                    graph_hash=graph_hash,
                    lambda_value=lambda_value,
                )
            should_refine = bool(
                iterate.certificate is None
                or isinstance(observed_start, CompressedEdgeCertificate)
            )
            periodic_gradient = build_certificate_gradient(
                model,
                iterate.phi,
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
            observed_refinement = certify(
                problem=certificate_problem,
                phi=iterate.phi,
                gradient=periodic_gradient,
                witness=observed_start,
                refine=should_refine,
                max_iter=periodic_limit,
                options=(
                    replace(certificate_options, max_iter=periodic_limit)
                    if isinstance(observed_start, CompressedEdgeCertificate)
                    else None
                ),
            )
            if should_refine:
                work_counters = work_counters + observed_refinement.work_counters
            iterate.certificate = observed_refinement.certificate
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
        if accepted:
            current_inner_converged = bool(inner_converged)
        converged_outer = bool(outer_converged)
        if (
            rel_change <= tol
            and step_residual <= np.sqrt(tol)
            and current_inner_converged
            and outer_converged
        ):
            converged = True
            outer_stop_reason = (
                "componentwise_backward_error_converged"
                if use_backward_error_progress
                else "legacy_progress_converged"
            )
            break

    final_terms = iterate.terms
    if iterate.certificate is None and isinstance(iterate.warm_state, PrimalOnlyWarmState):
        iterate.certificate = _rebase_certificate_hint(
            iterate.warm_state.certificate_hint,
            phi=iterate.phi,
            graph=tensor_graph,
            graph_hash=graph_hash,
            lambda_value=lambda_value,
        )
    certificate_gradient = build_certificate_gradient(
        model,
        iterate.phi,
        smooth_gradient=final_terms.gradient,
        lower=lower,
        upper=upper,
        eps=eps,
        tol=cert_tol,
    )

    certificate_needs_final_pass = False
    for _ in range(4):
        final_certificate_refinement = certify(
            problem=certificate_problem,
            phi=iterate.phi,
            gradient=certificate_gradient,
            witness=iterate.certificate,
            refine=True,
            max_iter=int(certificate_options.max_iter),
            options=(
                certificate_options
                if isinstance(iterate.certificate, CompressedEdgeCertificate)
                else None
            ),
        )
        work_counters = work_counters + final_certificate_refinement.work_counters
        iterate.certificate = final_certificate_refinement.certificate
        certificate_needs_final_pass = False
        if not bool(torch.any(certificate_gradient.at_breakpoint).item()):
            break
        interval_dual = getattr(iterate.certificate, "dual", None)
        if not torch.is_tensor(interval_dual):
            # A selected endpoint gradient is already a valid member of the
            # subgradient interval; compressed certificates simply cannot
            # improve a false negative by alternating the interval choice.
            break
        fusion_adjustment = graph_adjoint_edges(
            interval_dual,
            edge_u=edge_u,
            edge_v=edge_v,
            num_nodes=int(iterate.phi.shape[0]),
        )
        del interval_dual
        next_gradient = build_certificate_gradient(
            model,
            iterate.phi,
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
            break
        certificate_gradient = next_gradient
        certificate_needs_final_pass = True

    if certificate_needs_final_pass:
        final_certificate_refinement = certify(
            problem=certificate_problem,
            phi=iterate.phi,
            gradient=certificate_gradient,
            witness=iterate.certificate,
            refine=True,
            max_iter=int(certificate_options.max_iter),
            options=(
                certificate_options
                if isinstance(iterate.certificate, CompressedEdgeCertificate)
                else None
            ),
        )
        work_counters = work_counters + final_certificate_refinement.work_counters
    iterate.certificate = final_certificate_refinement.certificate
    working_precision_kkt_residual = float(
        final_certificate_refinement.diagnostics.backward_error_kkt_residual
    )
    certificate_audit_dtype = dtype_name(runtime.dtype)
    authoritative_objective = float(iterate.objective)
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
        ) = _terminal_backward_error_audit_float64(
            problem=problem,
            phi=iterate.phi,
            certificate=iterate.certificate,
            lambda_value=lambda_value,
            tol=cert_tol,
        )
        certificate_audit_dtype = "float64"
        gradient_scope = audit_gradient_scope
        directional_kink_admissible = bool(
            directional_kink_admissible
            and audit_directional_admissible
        )
        work_counters = work_counters + WorkCounters(
            full_certificate_audit_passes=1
        )
    authoritative_kkt_residual = float(
        admission_diagnostics.backward_error_kkt_residual
    )
    if not np.isfinite(float(authoritative_objective)):
        outer_stop_reason = "nonfinite_objective"
    final_dual = getattr(iterate.certificate, "dual", None)
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
        and has_proven_convex_observed_loss(problem.source_model, eps=eps)
    )
    global_optimality_basis = (
        _CONVEX_GLOBAL_OPTIMALITY_BASIS
        if global_optimality_certified
        else "not_certified"
    )
    if use_unimodal_objective and global_optimality_certified:
        converged = True

    phi_np = iterate.phi.detach().cpu().numpy()
    if isinstance(iterate.certificate, CompressedEdgeCertificate):
        terminal_warm_state = PrimalOnlyWarmState(
            phi=iterate.phi.detach(),
            structure_hint=iterate.certificate.labels.detach(),
            certificate_hint=iterate.certificate,
        )
    else:
        terminal_warm_state = DenseWarmState(
            phi=iterate.phi.detach(),
            dual=final_dual.detach() if torch.is_tensor(final_dual) else None,
            previous_lambda=float(lambda_value),
            graph_hash=str(graph_hash),
        )
    solver_state_out = SolverState(
        phi=iterate.phi.detach(),
        dual=final_dual.detach() if torch.is_tensor(final_dual) else None,
        previous_lambda=float(lambda_value),
        warm_state=terminal_warm_state,
        certificate=iterate.certificate,
        objective_spec_hash=str(objective_spec_hash),
    )
    terminal_components = KKTComponents(
        stationarity=float(admission_diagnostics.backward_error_stationarity_residual),
        edge_subgradient=float(
            admission_diagnostics.backward_error_edge_subgradient_residual
        ),
        dual_ball=float(admission_diagnostics.backward_error_dual_ball_residual),
        # Box feasibility is enforced by every primal update. The normalized
        # stationarity component already incorporates the box normal cone.
        box=0.0,
    )
    if not np.isclose(
        terminal_components.residual,
        authoritative_kkt_residual,
        rtol=0.0,
        atol=8.0 * np.finfo(np.float64).eps * (1.0 + authoritative_kkt_residual),
    ):
        raise AssertionError("Terminal KKT components do not reproduce the audit.")
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
            witness=iterate.certificate,
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
            stage_inner_iterations=int(total_inner_iterations),
            stage_inner_max_iter=max(int(inner_max_iter), 10),
            stage_inner_solve_calls=int(inner_solve_calls),
            stop_reason=str(outer_stop_reason),
            progress_residual_method=str(progress_residual_method),
            solve_tolerance=float(tol),
            legacy_stop_kkt_residual=float(legacy_stop_kkt_residual),
            componentwise_stop_kkt_residual=float(
                componentwise_stop_kkt_residual
            ),
            accepted_full_steps=int(accepted_full_steps),
            accepted_damped_steps=int(accepted_damped_steps),
            rejected_outer_steps=int(rejected_outer_steps),
        ),
        work=work_counters,
        state=solver_state_out,
        provenance=FitProvenance(
            objective_key=make_lambda_objective_key(
                base_objective_key,
                lambda_value=float(lambda_value),
            ),
            source_data_hash=problem.data_fingerprint,
            device=runtime.device_name,
            dtype=dtype_name(runtime.dtype),
            inner_solver=str(iterate.backend_name),
            global_optimality_basis=str(global_optimality_basis),
            scalar_pilot_certificates=problem.scalar_pilot_certificates,
            multiplicity_policy=problem.source_model.support_policy,
        ),
    )


def fit_prepared(
    problem: PreparedProblem,
    lambda_value: float,
    solver_options: SolverConfig,
    *,
    warm_state: SolverState | None = None,
    phi_start: np.ndarray | torch.Tensor | None = None,
    include_default_starts: bool = True,
) -> RawFit:
    """Solve a lambda on one frozen objective; no competing data/graph/runtime.

    Warm state and starts may change numerical effort, never the compiled
    likelihood, adaptive graph, epsilon, or float64 source authority.
    """
    problem.validate()
    data = problem.source_data
    _validate_solver_tolerance(solver_options.tolerance)
    _certificate_options(solver_options, problem.runtime.dtype)
    lambda_value = validate_lambda_value(lambda_value)
    objective_shape = objective_shape_for_data(data, solver_options.objective_shape)
    effective_runtime = problem.runtime

    attempts: list[_StartAttempt] = []
    if warm_state is not None:
        attempts.append(_StartAttempt(warm_state.phi, warm_state, "warm"))
    if phi_start is not None:
        attempts.append(_StartAttempt(phi_start, None, "explicit"))
    if not attempts and not include_default_starts:
        raise ValueError("No start supplied: provide warm_state or phi_start, or enable default starts.")
    singleton_route = objective_shape.startswith("unimodal")
    if include_default_starts:
        if singleton_route:
            if not attempts:
                attempts.append(_StartAttempt(problem.exact_pilot, None, "scalar_pilot"))
        else:
            attempts.extend(_StartAttempt(start, None, "scalar_well")
                            for start in problem.scalar_well_starts)
            attempts.append(_StartAttempt(problem.pooled_start, None, "pooled"))
    attempts = _deduplicate_start_attempts(
        attempts, runtime=effective_runtime, shape=tuple(problem.lower.shape),
    )
    if not has_multiplicity_ambiguity(data) and any(
        _clipped_singleton_start_needs_pilot(problem, item.phi) for item in attempts
    ):
        # This mandatory safeguard compares feasible points of the same frozen
        # objective. It does not move the box or turn local KKT into a global proof.
        attempts = _deduplicate_start_attempts(
            [*attempts, _StartAttempt(problem.exact_pilot, None, "clipping_escape_pilot")],
            runtime=effective_runtime, shape=tuple(problem.lower.shape), atol=0.0,
        )

    if data.num_mutations > 1 and (
        not problem.graph.is_complete
        or problem.graph_spec.degree_bound != data.num_mutations - 1
    ):
        raise ValueError("Raw fitting requires the frozen complete graph.")
    best_artifacts: RawFit | None = None
    for attempt in attempts:
        try:
            artifacts = _fit_from_start(problem, lambda_value, solver_options, attempt)
        except (MemoryError, torch.OutOfMemoryError) as exc:
            if isinstance(exc, ExactSolverResourceLimit):
                raise
            raise ExactSolverResourceLimit(
                "exact_solver_resource_limit: exact solver allocation "
                f"exhausted memory on {effective_runtime.device_name}."
            ) from exc
        if best_artifacts is None or _prefer_multistart_fit(artifacts, best_artifacts):
            best_artifacts = artifacts
        del artifacts
    if best_artifacts is None:
        raise RuntimeError("No valid start produced a fusion fit.")
    if not needs_precision_polish(best_artifacts):
        return best_artifacts
    precision_context = _float64_context(data, problem)
    polished = _fit_from_start(
        precision_context, lambda_value, replace(solver_options, outer_max_iter=1),
        _StartAttempt(
            torch.tensor(best_artifacts.phi, dtype=torch.float64,
                         device=precision_context.runtime.device),
            best_artifacts.state, "precision_polish",
        ),
    )
    return _finalize_precision_polish(polished, best_artifacts, problem)
