from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from ...io.data import (
    TumorData,
    tumor_data_fingerprint,
)
from ..objective import (
    BaseObjectiveKey,
    compile_observed_model,
    make_base_objective_key,
    make_lambda_objective_key,
    model_to_torch,
    observed_internal_breakpoints_torch,
    observed_one_sided_gradients_torch,
)
from .defaults import (
    DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
    DEFAULT_CERTIFICATE_MAX_ITER,
    DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS,
    DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    DEFAULT_DENSE_FALLBACK_POLICY,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_WORKSET_ADD_BATCH,
    DEFAULT_WORKSET_MAX_BYTES,
    DEFAULT_WORKSET_MAX_EXPANSIONS,
    normalize_dense_fallback_policy,
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
    TorchTumorData,
    as_runtime_tensor,
    copy_torch_tumor_data,
    mutation_region_terms_torch,
    dtype_name,
    em_surrogate_terms_torch,
    graph_adjoint_edges_in_dtype,
    graph_fusion_kkt_residual_from_grad_torch,
    pairwise_penalty_torch,
    resolve_runtime,
    solve_majorized_subproblem_alm_torch,
    solve_majorized_subproblem_pdhg_torch,
    to_torch_tumor_data,
    validate_torch_tumor_data,
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
    TensorProblem,
    TorchRuntime,
    WorkCounters,
    WorksetMemoryOptions,
)


def _terminal_backward_error_audit_float64(
    *,
    torch_data: TorchTumorData,
    phi: torch.Tensor,
    certificate: GraphFusionCertificate | None,
    graph_spec: PairwiseFusionGraph,
    graph_hash: str,
    lambda_value: float,
    major_prior: float,
    eps: float,
    tol: float,
) -> tuple[KKTDiagnostics, str, bool, float]:
    """Audit the unchanged terminal witness with float64 backward error."""

    device = phi.device
    source_model = torch_data.source_model
    if source_model is None:
        raise ValueError("Float64 audit requires an immutable observed-model source.")
    audit_runtime = TorchRuntime(
        device=device,
        device_name=str(device),
        dtype=torch.float64,
    )
    data64 = copy_torch_tumor_data(
        torch_data,
        dtype=torch.float64,
        device=device,
    )
    graph64 = tensorize_graph(
        graph_spec,
        audit_runtime,
        num_nodes=int(source_model.shape[0]),
    )
    phi64 = phi.to(dtype=torch.float64, device=device)
    lower64 = torch.as_tensor(
        np.array(source_model.lower, copy=True),
        dtype=torch.float64,
        device=device,
    )
    upper64 = torch.as_tensor(
        np.array(source_model.upper, copy=True),
        dtype=torch.float64,
        device=device,
    )
    terms64 = mutation_region_terms_torch(
        data64,
        phi64,
        major_prior=float(major_prior),
        eps=float(eps),
    )
    gradient = build_certificate_gradient(
        data64,
        phi=phi64,
        smooth_gradient=terms64.grad,
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
            device=device,
        )
        gradient = build_certificate_gradient(
            data64,
            phi=phi64,
            smooth_gradient=terms64.grad,
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
_CONVEX_GLOBAL_OPTIMALITY_BASIS = "convex_fixed_linear_objective_plus_kkt"
OBJECTIVE_SHAPE_AUTO = "auto"
PATH_OBJECTIVE_SHAPE = "generic_nonconvex"


def uses_explicit_path_likelihood(data: TumorData) -> bool:
    """Whether ``data`` carries an explicit categorical occupancy-path model."""

    return getattr(data, "path_likelihood", None) is not None


def uses_nonconvex_path_likelihood(data: TumorData) -> bool:
    """Whether an explicit path family requires generic nonconvex handling."""

    path = getattr(data, "path_likelihood", None)
    return bool(
        path is not None and not bool(getattr(path, "has_fixed_linear_emission", False))
    )


def uses_nonconvex_observed_likelihood(data: TumorData) -> bool:
    """Whether the observed-data likelihood can contain competing wells.

    A legacy major/minor mixture is no more globally unimodal than an explicit
    occupancy-path mixture.  Only a fixed linear emission is known to retain
    the convex binomial-loss contract used by the global KKT claim.
    """

    legacy_mixture = bool(
        getattr(data, "path_likelihood", None) is None
        and np.any(np.asarray(data.multiplicity_estimation_mask, dtype=bool))
    )
    return bool(legacy_mixture or uses_nonconvex_path_likelihood(data))


def _effective_major_prior(data: TumorData, major_prior: float) -> float:
    """Canonicalize a legacy option that is absent from fixed-prior path models."""

    if uses_explicit_path_likelihood(data):
        return 0.5
    return float(major_prior)


def objective_shape_for_data(data: TumorData, requested: str) -> str:
    """Return the only solver shape declaration valid for this likelihood.

    Competing or genuinely piecewise paths can be multimodal and therefore
    always use the generic route. A path specification whose valid candidates
    all reduce to the same fixed linear emission reuses the existing scalar
    route without discarding its path provenance.
    """

    normalized = _normalize_objective_shape(requested)
    if uses_nonconvex_path_likelihood(data):
        return PATH_OBJECTIVE_SHAPE
    return "unimodal" if normalized == OBJECTIVE_SHAPE_AUTO else normalized


def _path_smooth_interval_bounds(
    torch_data: TorchTumorData,
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
        torch_data.observed_model,
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
    h_base = torch.clamp(surrogate_terms.hess_upper, min=_MISSING_SURROGATE_CURVATURE)
    surrogate_grad = surrogate_terms.grad
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


def _tensor_problem_from_torch_data(
    torch_data: TorchTumorData,
    *,
    major_prior: float,
    eps: float,
) -> TensorProblem:
    prior = float(major_prior)
    if not np.isfinite(prior) or not (0.0 < prior < 1.0):
        raise ValueError("major_prior must lie strictly in (0, 1).")
    return TensorProblem(
        observed_model=torch_data.observed_model,
        eps=float(eps),
        major_prior=prior,
        source_model=torch_data.source_model,
    )


def torch_data_from_context(context: SolverContext) -> TorchTumorData:
    problem = context.problem
    return TorchTumorData(
        observed_model=problem.observed_model,
        data_fingerprint=context.data_fingerprint,
        source_model=problem.source_model,
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
    source_model = context.problem.source_model
    if source_model is None:
        raise ValueError("SolverContext lacks an immutable observed-model source.")
    promoted_data = copy_torch_tumor_data(
        torch_data_from_context(context),
        dtype=dtype,
        device=target_device,
    )
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
    problem = _tensor_problem_from_torch_data(
        promoted_data,
        major_prior=float(context.problem.major_prior),
        eps=float(context.problem.eps),
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
        problem=problem,
        graph=graph,
        exact_pilot=exact,
        pooled_start=pooled,
        scalar_well_starts=wells,
        lower=torch.as_tensor(
            np.array(source_model.lower, copy=True),
            dtype=dtype,
            device=target_device,
        ),
        upper=torch.as_tensor(
            np.array(source_model.upper, copy=True),
            dtype=dtype,
            device=target_device,
        ),
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
        prefix = "CPU " if target.type == "cpu" and target != context.runtime.device else ""
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
) -> tuple[SolverState | None, int]:
    """Nudge a failed dense-certificate state off exact path breakpoints.

    The dense certificate supplies the fusion adjoint.  At each exact
    breakpoint, the one-sided observed gradients plus that adjoint choose a
    retry side.  A changed primal invalidates every dual/certificate warm
    object because none remains valid at the new point.
    """

    certificate = None if state is None else state.certificate
    model = context.problem.observed_model
    if (
        state is None
        or model.model_id == "legacy_major_low_as_paths_v2"
        or not isinstance(certificate, DenseEdgeCertificate)
        or certificate.scope != "full_original_graph"
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

    torch_data = torch_data_from_context(context)
    with torch.no_grad():
        fusion_adjustment = graph_adjoint_edges(
            dual,
            edge_u=context.graph.edge_u,
            edge_v=context.graph.edge_v,
            num_nodes=int(phi.shape[0]),
        )
        gradient_left, gradient_right, at_breakpoint = (
            observed_one_sided_gradients_torch(
                torch_data.observed_model,
                phi,
                eps=float(context.problem.eps),
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
            10.0 * float(context.problem.eps),
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
    torch_data: TorchTumorData | None = None,
    objective_shape: str = OBJECTIVE_SHAPE_AUTO,
    defer_graph: bool = False,
) -> SolverContext:
    tol = _validate_solver_tolerance(tol)
    objective_shape = objective_shape_for_data(data, objective_shape)
    major_prior = _effective_major_prior(data, major_prior)
    use_unimodal_objective = objective_shape.startswith("unimodal")
    effective_runtime = (
        resolve_runtime(device, dtype=dtype) if runtime is None else runtime
    )
    source_model = compile_observed_model(
        data,
        major_prior=float(major_prior),
        eps=float(eps),
    )
    if torch_data is None:
        effective_torch_data = to_torch_tumor_data(
            data,
            effective_runtime,
            source_model=source_model,
        )
        data_fingerprint = effective_torch_data.data_fingerprint
    else:
        effective_torch_data = replace(
            torch_data,
            source_model=source_model,
            observed_model=model_to_torch(source_model, effective_runtime),
        )
        data_fingerprint = tumor_data_fingerprint(data)
        validate_torch_tumor_data(
            effective_torch_data,
            data=data,
            runtime=effective_runtime,
            expected_fingerprint=data_fingerprint,
        )

    if exact_pilot is None:
        exact_pilot_tensor, secondary_wells, valid_secondary = (
            compute_scalar_mutation_region_wells_torch(
                effective_torch_data,
                phi_init=data.phi_init,
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
                    effective_torch_data,
                    phi_init=data.phi_init,
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
            count_observed=effective_torch_data.count_observed,
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
            effective_torch_data,
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
            effective_torch_data,
            eps=float(eps),
            exact_pilot=exact_pilot_tensor,
            secondary_wells=secondary_wells,
            valid_secondary=valid_secondary,
        )
    else:
        scalar_well_starts_seq = list(scalar_well_starts)

    lower = torch.as_tensor(
        np.array(source_model.lower, copy=True),
        dtype=effective_runtime.dtype,
        device=effective_runtime.device,
    )
    upper = torch.as_tensor(
        np.array(source_model.upper, copy=True),
        dtype=effective_runtime.dtype,
        device=effective_runtime.device,
    )
    problem = _tensor_problem_from_torch_data(
        effective_torch_data,
        major_prior=float(major_prior),
        eps=float(eps),
    )
    graph_hash = effective_graph.fingerprint
    base_objective_key = make_base_objective_key(
        source_model,
        graph_hash=graph_hash,
        eps=float(eps),
        lower=source_model.lower,
        upper=source_model.upper,
    )
    base_fusion_objective_hash = base_objective_key.fingerprint
    objective_spec_hash = base_fusion_objective_hash
    return SolverContext(
        problem=problem,
        graph=tensor_graph,
        graph_spec=effective_graph,
        exact_pilot=exact_pilot_tensor,
        pooled_start=pooled_start_tensor,
        scalar_well_starts=tuple(
            as_runtime_tensor(start, effective_runtime)
            for start in scalar_well_starts_seq
        ),
        lower=lower,
        upper=upper,
        runtime=effective_runtime,
        data_fingerprint=data_fingerprint,
        graph_hash=graph_hash,
        objective_spec_hash=objective_spec_hash,
        base_fusion_objective_hash=base_fusion_objective_hash,
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
    supplied_torch_data = kwargs.pop("torch_data", None)
    requested_device = kwargs.pop("device", "cuda")
    requested_dtype = kwargs.pop("dtype", "float64")
    resolved_by_cpu_fallback = False
    requested_runtime = supplied_runtime
    try:
        if requested_runtime is None:
            requested_runtime = resolve_runtime(
                requested_device, dtype=requested_dtype
            )
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

    def prepare_on_runtime(*, retain_torch_data: bool) -> SolverContext:
        reusable_tensor_graph = supplied_prebuilt_tensor_graph
        graph_runtime = None if reusable_tensor_graph is None else (
            reusable_tensor_graph.weight.device,
            reusable_tensor_graph.weight.dtype,
        )
        if graph_runtime != (requested_runtime.device, requested_runtime.dtype):
            reusable_tensor_graph = None
        context = prepare_torch_problem(
            data,
            device=requested_runtime.device_name,
            dtype=dtype_name(requested_runtime.dtype),
            runtime=requested_runtime,
            torch_data=supplied_torch_data if retain_torch_data else None,
            prebuilt_tensor_graph=reusable_tensor_graph,
            **kwargs,
        )
        fallback = "dense_cpu" if resolved_by_cpu_fallback else inherited_resource_fallback
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
            return prepare_on_runtime(retain_torch_data=not resolved_by_cpu_fallback)
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
            supplied_torch_data = None


def _initial_outer_diag() -> dict[str, float | int]:
    """Fail-closed residuals used until the first outer KKT audit."""
    return {
        "stationarity_residual": np.inf,
        "edge_subgradient_residual": np.inf,
        "dual_ball_residual": np.inf,
        "box_residual": np.inf,
        "kkt_residual": np.inf,
    }


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
    surrogate_diag_values: dict[str, float | int] = {}
    if use_alm:
        (
            phi_trial,
            dual_trial,
            dual_kkt_trial,
            _inner_iterations,
            inner_ok,
            _inner_residual,
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
            diagnostics_out=surrogate_diag_values,
        )
    else:
        (
            phi_trial,
            dual_trial,
            dual_kkt_trial,
            _inner_iterations,
            inner_ok,
            _inner_residual,
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
            max_iter=max(inner_max_iter, 10),
            phi_start=phi,
            dual_start=dual,
            tau_node=pdhg_tau_node,
        )
    if use_alm:
        # The outer MM loop carries the rho-invariant actual multiplier y.
        # Drop the low-level scaled-u return here so a second complete
        # edge-by-region tensor does not remain live through outer scoring and
        # certificate refinement.
        dual_trial = dual_kkt_trial
    if surrogate_diag_values:
        surrogate_diag = surrogate_diag_values
    else:
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
            dual=dual_trial if torch.is_tensor(dual_trial) else None,
            previous_lambda=float(lambda_value),
            graph_hash=str(graph_hash),
        ),
        surrogate_certificate=certificate,
        surrogate_kkt=KKTDiagnostics.from_mapping(surrogate_diag),
        converged=bool(inner_ok),
    )


def _fit_from_start(
    data: TumorData,
    *,
    torch_data,
    runtime,
    graph: PairwiseFusionGraph,
    tensor_graph: TensorFusionGraph,
    graph_hash: str,
    base_objective_key: BaseObjectiveKey,
    objective_spec_hash: str,
    lambda_value: float,
    major_prior: float,
    eps: float,
    outer_max_iter: int,
    inner_max_iter: int,
    tol: float,
    phi_start: np.ndarray | torch.Tensor,
    solver_state: SolverState | None,
    lower: torch.Tensor,
    upper: torch.Tensor,
    objective_shape: str,
    workset_max_bytes: int,
    compressed_cache_max_bytes: int,
    workset_add_batch: int,
    workset_max_expansions: int,
    certificate_max_iter: int,
    certificate_refinement_rounds: int,
    certificate_column_tol_scale: float,
    verbose: bool,
) -> RawFit:
    tol = _validate_solver_tolerance(tol)
    if base_objective_key.fingerprint != str(objective_spec_hash):
        raise ValueError("Base objective key does not match objective_spec_hash.")
    if (
        solver_state is not None
        and str(solver_state.objective_spec_hash)
        and str(solver_state.objective_spec_hash) != str(objective_spec_hash)
    ):
        raise ValueError(
            "Solver warm state belongs to a different raw objective."
        )
    objective_shape = objective_shape_for_data(data, objective_shape)
    certificate_options = CertificateOptions(
        max_iter=max(int(certificate_max_iter), 1),
        refinement_rounds=max(int(certificate_refinement_rounds), 0),
        max_expansions=max(int(workset_max_expansions), 1),
        add_batch=max(int(workset_add_batch), 1),
        mapping_tolerance=max(0.1 * float(tol), float(torch.finfo(runtime.dtype).eps)),
        column_tolerance=max(
            float(certificate_column_tol_scale) * float(tol),
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
        atol=float(tol),
    )
    use_unimodal_objective = objective_shape.startswith("unimodal")
    require_full_step_backtracking = (
        objective_shape == "unimodal_full_step_backtracking"
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
    inner_solver = dense_inner_solver
    if (
        solver_state is not None
        and solver_state.phi is not None
        and tuple(solver_state.phi.shape) == tuple(torch_data.phi_upper.shape)
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
    dual_start_is_actual = bool(use_alm and state_dual is not None)
    converged = False
    converged_outer = False
    iterations = 0
    work_counters = WorkCounters()
    current_inner_converged = False
    final_outer_diag = _initial_outer_diag()
    outer_kkt_certificate_status = "not_audited"
    mm_consistency_violations = 0
    full_step_curvature_multiplier = torch.ones_like(phi)

    current_mutation_region_terms = mutation_region_terms_torch(
        torch_data, phi, major_prior=major_prior, eps=eps
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
    for outer_iter in range(max(int(outer_max_iter), 1)):
        iterations = outer_iter + 1
        previous_phi = phi.clone()
        previous_objective = objective
        if use_unimodal_objective:
            surrogate_terms = current_mutation_region_terms
            surrogate_fit_loss = float(fit_loss)
        else:
            responsibilities = current_mutation_region_terms.path_posterior
            if responsibilities is None:
                raise AssertionError("Observed terms lack path responsibilities.")
            surrogate_terms = em_surrogate_terms_torch(
                torch_data,
                phi,
                responsibilities=responsibilities,
                eps=eps,
            )
            surrogate_fit_loss = float(torch.sum(surrogate_terms.loss).item())
        h_base, surrogate_grad = _safe_surrogate_curvature_and_gradient(
            surrogate_terms,
            torch_data.count_observed,
        )
        if use_unimodal_objective:
            smooth_lower, smooth_upper = lower, upper
        else:
            smooth_lower, smooth_upper = _path_smooth_interval_bounds(
                torch_data,
                phi,
                lower=lower,
                upper=upper,
                eps=float(eps),
            )
        if require_full_step_backtracking:
            forcing_certificate = certificate
            if forcing_certificate is None:
                forcing_certificate = _compressed_certificate_for_primal(
                    phi,
                    graph_hash=graph_hash,
                    gradient_scope="observed_objective",
                )
            forcing_gradient = build_certificate_gradient(
                torch_data,
                phi=phi,
                smooth_gradient=current_mutation_region_terms.grad,
                lower=lower,
                upper=upper,
                eps=eps,
                tol=tol,
            )
            forcing_diag = certify(
                problem=certificate_problem,
                phi=phi,
                gradient=forcing_gradient,
                witness=forcing_certificate,
                refine=False,
            ).diagnostics.as_dict()
            inner_progress_tolerance = max(
                5.0 * tol,
                min(
                    float(np.sqrt(tol)),
                    0.9 * float(forcing_diag["kkt_residual"]),
                ),
            )
        else:
            inner_progress_tolerance = 5.0 * tol
        scale = 1.0
        curvature_multiplier = full_step_curvature_multiplier
        accepted = False
        candidate_phi = phi
        candidate_dual = dual
        candidate_dual_kkt = dual_kkt
        candidate_certificate = certificate
        candidate_warm_state = warm_state
        candidate_backend_name = inner_solver
        candidate_dual_start_is_actual = dual_start_is_actual
        candidate_objective = objective
        candidate_fit_loss = fit_loss
        candidate_mutation_region_terms = current_mutation_region_terms
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
                phi,
                surrogate_grad=surrogate_grad,
                h=h,
                count_observed=torch_data.count_observed,
            )
            if use_unimodal_objective and not require_full_step_backtracking:
                q_current = None
            else:
                q_current = _inner_model_value_torch(
                    phi,
                    U=U,
                    h=h,
                    edge_u=edge_u,
                    edge_v=edge_v,
                    edge_w=edge_w,
                    lambda_value=lambda_value,
                )
            recovery_inner_model_tol = (
                max(
                    64.0 * float(torch.finfo(phi.dtype).eps),
                    float(tol) ** 2,
                )
                * (1.0 + abs(float(q_current.item())))
                if require_full_step_backtracking
                else 0.0
            )
            inner_phi_start = phi
            inner_dual_start = dual
            inner_dual_start_is_actual = dual_start_is_actual
            inner_batch_limit = 8 if require_full_step_backtracking else 1
            for _inner_batch in range(inner_batch_limit):
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
                    inner_max_iter=inner_max_iter,
                    phi=inner_phi_start,
                    dual=inner_dual_start,
                    dual_start_is_actual=inner_dual_start_is_actual,
                    spectral_rho=bool(require_full_step_backtracking),
                    pdhg_tau_node=tensor_graph.pdhg_tau_node,
                    backend_name=dense_inner_solver,
                    graph_hash=graph_hash,
                )
                phi_trial = inner_result.phi
                dense_warm_state = inner_result.warm_state
                dual_trial = getattr(dense_warm_state, "dual", None)
                surrogate_certificate = inner_result.surrogate_certificate
                dual_kkt_trial = getattr(surrogate_certificate, "dual", None)
                inner_ok = bool(inner_result.converged)
                inner_residual = float(inner_result.surrogate_kkt.kkt_residual)
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
                inner_dual_start = dual_kkt_trial if use_alm else dual_trial
                inner_dual_start_is_actual = bool(use_alm)
            delta = phi_trial - phi
            trial_mutation_region_terms = mutation_region_terms_torch(
                torch_data, phi_trial, major_prior=major_prior, eps=eps
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
                        surrogate_terms.grad * delta + 0.5 * h * torch.square(delta)
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
                    trial_surrogate_terms = em_surrogate_terms_torch(
                        torch_data,
                        phi_trial,
                        responsibilities=responsibilities,
                        eps=eps,
                    )
                    trial_surrogate_loss = float(
                        torch.sum(trial_surrogate_terms.loss).item()
                    )
                    surrogate_gap = float(trial_surrogate_loss - majorizer_rhs)
                    em_envelope_gap = float(
                        (trial_fit_loss - fit_loss)
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
                numerical_factor = 64.0 * float(torch.finfo(phi.dtype).eps)
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
            envelope_tol = 1e-8 * (1.0 + abs(fit_loss))
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
                candidate_phi = phi_trial
                # The complete-graph ADMM backend also returns the actual KKT
                # multiplier y=rho*u. Carry y, not the rho-dependent scaled u,
                # across outer MM subproblems because curvature changes rho.
                candidate_dual = dual_kkt_trial if use_alm else dual_trial
                candidate_dual_kkt = dual_kkt_trial
                candidate_certificate = surrogate_certificate
                candidate_warm_state = inner_result.warm_state
                candidate_backend_name = inner_result.backend_name
                candidate_dual_start_is_actual = bool(use_alm)
                candidate_objective = trial_objective
                candidate_fit_loss = trial_fit_loss
                candidate_mutation_region_terms = trial_mutation_region_terms
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
                # A damped primal point does not share the full subproblem's
                # dual certificate. Enlarge the persistent majorizing
                # curvature and accept only a full proximal-MM/ADMM endpoint.
                # If the resource limit is exhausted, leave this outer iterate
                # unchanged and uncertified rather than interpolating phi.
                delta_square = torch.square(delta)
                resolution = torch.finfo(phi.dtype).eps * (1.0 + torch.square(phi))
                secant_remainder = (
                    trial_mutation_region_terms.loss
                    - current_mutation_region_terms.loss
                    - surrogate_grad * delta
                )
                required_h = torch.where(
                    delta_square > resolution,
                    2.0
                    * torch.clamp(secant_remainder, min=0.0)
                    / torch.clamp(
                        delta_square,
                        min=torch.finfo(phi.dtype).tiny,
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
                        * (1.0 + 64.0 * torch.finfo(phi.dtype).eps)
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
                phi_theta = phi + theta * delta
                theta_mutation_region_terms = mutation_region_terms_torch(
                    torch_data, phi_theta, major_prior=major_prior, eps=eps
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
                    candidate_phi = phi_theta
                    (
                        candidate_dual,
                        candidate_dual_kkt,
                        candidate_certificate,
                        candidate_warm_state,
                        candidate_dual_start_is_actual,
                    ) = _invalidate_damped_trial_state(
                        phi=phi_theta,
                        trial_warm_state=inner_result.warm_state,
                    )
                    candidate_backend_name = inner_result.backend_name
                    candidate_objective = theta_objective
                    candidate_fit_loss = theta_fit_loss
                    candidate_mutation_region_terms = theta_mutation_region_terms
                    inner_converged = False
                    break
                theta *= 0.5
            if damped_accepted:
                break
            scale *= 2.0

        if not accepted:
            candidate_phi = phi
            candidate_dual = dual
            candidate_dual_kkt = dual_kkt
            candidate_certificate = certificate
            candidate_warm_state = warm_state
            candidate_backend_name = inner_solver
            candidate_dual_start_is_actual = dual_start_is_actual
            candidate_objective = objective
            candidate_fit_loss = fit_loss
            candidate_mutation_region_terms = current_mutation_region_terms
        phi = candidate_phi
        dual = candidate_dual
        dual_kkt = candidate_dual_kkt
        certificate = candidate_certificate
        warm_state = candidate_warm_state
        inner_solver = candidate_backend_name
        dual_start_is_actual = candidate_dual_start_is_actual
        objective = candidate_objective
        fit_loss = candidate_fit_loss
        current_mutation_region_terms = candidate_mutation_region_terms
        penalty = objective - fit_loss
        if verbose:
            print(
                f"[pairwise-fusion:{runtime.device_name}] iter={iterations:02d} objective={objective:.6f} "
                f"fit={fit_loss:.6f} penalty={penalty:.6f}"
            )

        rel_change = abs(previous_objective - objective) / (
            1.0 + abs(previous_objective)
        )
        step_residual = float(
            (
                torch.linalg.norm(phi - previous_phi)
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
            or not np.isfinite(objective)
        )
        outer_diag = final_outer_diag
        outer_converged = False
        if do_outer_kkt_audit:
            outer_terms = current_mutation_region_terms
            observed_start = certificate
            if observed_start is None and isinstance(warm_state, PrimalOnlyWarmState):
                observed_start = _rebase_certificate_hint(
                    warm_state.certificate_hint,
                    phi=phi,
                    graph=tensor_graph,
                    graph_hash=graph_hash,
                    lambda_value=lambda_value,
                )
            should_refine = bool(
                certificate is None
                or isinstance(observed_start, CompressedEdgeCertificate)
            )
            periodic_gradient = build_certificate_gradient(
                torch_data,
                phi,
                smooth_gradient=outer_terms.grad,
                lower=lower,
                upper=upper,
                eps=eps,
                tol=tol,
            )
            periodic_limit = min(
                int(certificate_options.max_iter),
                _PERIODIC_CERTIFICATE_MAX_ITER,
            )
            observed_refinement = certify(
                problem=certificate_problem,
                phi=phi,
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
            certificate = observed_refinement.certificate
            outer_diag = observed_refinement.diagnostics.as_dict()
            outer_converged = bool(outer_diag["kkt_residual"] <= 5.0 * tol)
        if accepted:
            current_inner_converged = bool(inner_converged)
        if do_outer_kkt_audit:
            final_outer_diag = outer_diag
        converged_outer = bool(outer_converged)
        if (
            rel_change <= tol
            and step_residual <= np.sqrt(tol)
            and current_inner_converged
            and outer_converged
        ):
            converged = True
            break

    final_terms = current_mutation_region_terms
    if certificate is None and isinstance(warm_state, PrimalOnlyWarmState):
        certificate = _rebase_certificate_hint(
            warm_state.certificate_hint,
            phi=phi,
            graph=tensor_graph,
            graph_hash=graph_hash,
            lambda_value=lambda_value,
        )
    certificate_gradient = build_certificate_gradient(
        torch_data,
        phi,
        smooth_gradient=final_terms.grad,
        lower=lower,
        upper=upper,
        eps=eps,
        tol=tol,
    )

    final_refinements = []
    certificate_needs_final_pass = False
    for _ in range(4):
        final_certificate_refinement = certify(
            problem=certificate_problem,
            phi=phi,
            gradient=certificate_gradient,
            witness=certificate,
            refine=True,
            max_iter=int(certificate_options.max_iter),
            options=(
                certificate_options
                if isinstance(certificate, CompressedEdgeCertificate)
                else None
            ),
        )
        final_refinements.append(final_certificate_refinement)
        certificate = final_certificate_refinement.certificate
        certificate_needs_final_pass = False
        if not bool(torch.any(certificate_gradient.at_breakpoint).item()):
            break
        interval_dual = getattr(certificate, "dual", None)
        if not torch.is_tensor(interval_dual):
            # A selected endpoint gradient is already a valid member of the
            # subgradient interval; compressed certificates simply cannot
            # improve a false negative by alternating the interval choice.
            break
        fusion_adjustment = graph_adjoint_edges(
            interval_dual,
            edge_u=edge_u,
            edge_v=edge_v,
            num_nodes=int(phi.shape[0]),
        )
        next_gradient = build_certificate_gradient(
            torch_data,
            phi,
            smooth_gradient=final_terms.grad,
            lower=lower,
            upper=upper,
            eps=eps,
            tol=tol,
            fusion_adjoint=fusion_adjustment,
        )
        if torch.allclose(
            next_gradient.value,
            certificate_gradient.value,
            rtol=0.0,
            atol=max(float(tol) * 0.1, 1e-12),
        ):
            break
        certificate_gradient = next_gradient
        certificate_needs_final_pass = True

    if certificate_needs_final_pass:
        final_certificate_refinement = certify(
            problem=certificate_problem,
            phi=phi,
            gradient=certificate_gradient,
            witness=certificate,
            refine=True,
            max_iter=int(certificate_options.max_iter),
            options=(
                certificate_options
                if isinstance(certificate, CompressedEdgeCertificate)
                else None
            ),
        )
        final_refinements.append(final_certificate_refinement)

    for refinement in final_refinements:
        work_counters = work_counters + refinement.work_counters
    certificate = final_certificate_refinement.certificate
    final_outer_diag = final_certificate_refinement.diagnostics.as_dict()
    working_precision_kkt_residual = float(
        final_outer_diag["backward_error_kkt_residual"]
    )
    certificate_audit_dtype = dtype_name(runtime.dtype)
    authoritative_objective = float(objective)
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
            torch_data=torch_data,
            phi=phi,
            certificate=certificate,
            graph_spec=graph,
            graph_hash=graph_hash,
            lambda_value=lambda_value,
            major_prior=major_prior,
            eps=eps,
            tol=tol,
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
    admission_diag = admission_diagnostics.as_dict()
    for key in (
        "backward_error_stationarity_residual",
        "backward_error_edge_subgradient_residual",
        "backward_error_dual_ball_residual",
        "backward_error_kkt_residual",
    ):
        final_outer_diag[key] = admission_diag[key]
    authoritative_kkt_residual = float(
        admission_diag["backward_error_kkt_residual"]
    )
    final_dual = getattr(certificate, "dual", None)
    outer_kkt_certificate_status = str(final_certificate_refinement.status)
    converged_outer = bool(authoritative_kkt_residual <= 5.0 * tol)
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
        and not uses_nonconvex_observed_likelihood(data)
    )
    global_optimality_basis = (
        _CONVEX_GLOBAL_OPTIMALITY_BASIS
        if global_optimality_certified
        else "not_certified"
    )
    if use_unimodal_objective and global_optimality_certified:
        converged = True

    phi_np = phi.detach().cpu().numpy()
    if isinstance(certificate, CompressedEdgeCertificate):
        terminal_warm_state = PrimalOnlyWarmState(
            phi=phi.detach(),
            structure_hint=certificate.labels.detach(),
            certificate_hint=certificate,
        )
    else:
        terminal_warm_state = DenseWarmState(
            phi=phi.detach(),
            dual=final_dual.detach() if torch.is_tensor(final_dual) else None,
            previous_lambda=float(lambda_value),
            graph_hash=str(graph_hash),
        )
    solver_state_out = SolverState(
        phi=phi.detach(),
        dual=final_dual.detach() if torch.is_tensor(final_dual) else None,
        previous_lambda=float(lambda_value),
        warm_state=terminal_warm_state,
        certificate=certificate,
        objective_spec_hash=str(objective_spec_hash),
    )
    terminal_components = KKTComponents(
        stationarity=float(admission_diag["backward_error_stationarity_residual"]),
        edge_subgradient=float(
            admission_diag["backward_error_edge_subgradient_residual"]
        ),
        dual_ball=float(admission_diag["backward_error_dual_ball_residual"]),
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
            tolerance=5.0 * float(tol),
            scope="full_original_graph",
            gradient_scope=str(gradient_scope),
            directional_admissible=bool(directional_kink_admissible),
            witness=certificate,
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
        ),
        work=work_counters,
        state=solver_state_out,
        provenance=FitProvenance(
            objective_key=make_lambda_objective_key(
                base_objective_key,
                lambda_value=float(lambda_value),
            ),
            device=runtime.device_name,
            dtype=dtype_name(runtime.dtype),
            inner_solver=str(inner_solver),
            global_optimality_basis=str(global_optimality_basis),
            likelihood_eps=float(eps),
        ),
    )


def fit_observed_data_pairwise_fusion(
    data: TumorData,
    *,
    lambda_value: float,
    major_prior: float,
    eps: float,
    outer_max_iter: int,
    inner_max_iter: int,
    tol: float,
    phi_start: np.ndarray | torch.Tensor | None = None,
    graph: PairwiseFusionGraph | None = None,
    adaptive_weight_gamma: float = 1.0,
    adaptive_weight_floor: float = 1e-6,
    adaptive_weight_baseline: float = 1.0,
    exact_pilot: np.ndarray | torch.Tensor | None = None,
    pooled_start: np.ndarray | torch.Tensor | None = None,
    scalar_well_starts: list[np.ndarray | torch.Tensor]
    | tuple[np.ndarray | torch.Tensor, ...]
    | None = None,
    start_mode: str = "full",
    device: str | None = DEFAULT_DEVICE,
    dtype: str | None = DEFAULT_DTYPE,
    runtime=None,
    torch_data=None,
    solver_context: SolverContext | None = None,
    solver_state: SolverState | None = None,
    objective_shape: str = OBJECTIVE_SHAPE_AUTO,
    workset_max_bytes: int = DEFAULT_WORKSET_MAX_BYTES,
    compressed_cache_max_bytes: int = DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    dense_fallback_policy: str = DEFAULT_DENSE_FALLBACK_POLICY,
    workset_add_batch: int = DEFAULT_WORKSET_ADD_BATCH,
    workset_max_expansions: int = DEFAULT_WORKSET_MAX_EXPANSIONS,
    certificate_max_iter: int = DEFAULT_CERTIFICATE_MAX_ITER,
    certificate_refinement_rounds: int = DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS,
    certificate_column_tol_scale: float = DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
    verbose: bool = False,
) -> RawFit:
    tol = _validate_solver_tolerance(tol)
    lambda_value = validate_lambda_value(lambda_value)
    objective_shape = objective_shape_for_data(data, objective_shape)
    major_prior = _effective_major_prior(data, major_prior)
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
            torch_data=torch_data,
            objective_shape=objective_shape,
        )
    else:
        expected_data_fingerprint = tumor_data_fingerprint(data)
        if (
            getattr(solver_context, "data_fingerprint", None)
            != expected_data_fingerprint
        ):
            raise ValueError(
                "SolverContext data fingerprint does not match the requested TumorData."
            )
        if (
            abs(float(solver_context.problem.major_prior) - float(major_prior)) > 0.0
            or abs(float(solver_context.problem.eps) - float(eps)) > 0.0
        ):
            raise ValueError(
                "SolverContext major_prior/eps do not match the requested fit options."
            )

    context_prepared_by_cpu_fallback = bool(
        solver_context.resource_fallback == "dense_cpu"
    )
    effective_runtime = solver_context.runtime
    effective_exact_pilot = (
        solver_context.exact_pilot if exact_pilot is None else exact_pilot
    )
    effective_pooled_start = (
        solver_context.pooled_start if pooled_start is None else pooled_start
    )
    effective_scalar_well_starts = (
        solver_context.scalar_well_starts
        if scalar_well_starts is None
        else tuple(scalar_well_starts)
    )
    if (
        uses_explicit_path_likelihood(data)
        and not effective_scalar_well_starts
        and solver_context.scalar_well_starts
    ):
        effective_scalar_well_starts = solver_context.scalar_well_starts

    normalized_start_mode = str(start_mode).strip().lower()
    if normalized_start_mode not in {"full", "warm_plus_pilot", "warm_only"}:
        raise ValueError(f"Unknown start_mode: {start_mode}")
    if uses_explicit_path_likelihood(data):
        normalized_start_mode = "full"

    if objective_shape.startswith("unimodal"):
        start_bank = [phi_start] if phi_start is not None else [effective_exact_pilot]
    else:
        start_bank: list[np.ndarray | torch.Tensor] = []
        if phi_start is not None:
            start_bank.append(phi_start)
        if normalized_start_mode == "full":
            start_bank.extend(effective_scalar_well_starts)
            start_bank.append(effective_pooled_start)
    start_bank = _deduplicate_starts(start_bank, runtime=effective_runtime)

    def solve_start_once(
        *,
        context: SolverContext,
        start: np.ndarray | torch.Tensor,
        state: SolverState | None,
        polish: bool = False,
    ) -> RawFit:
        if context.base_objective_key is None:
            raise ValueError("SolverContext lacks a typed base-objective key.")
        return _fit_from_start(
            data,
            torch_data=torch_data_from_context(context),
            runtime=context.runtime,
            graph=context.graph_spec,
            tensor_graph=context.graph,
            graph_hash=str(context.graph_hash),
            base_objective_key=context.base_objective_key,
            objective_spec_hash=str(context.objective_spec_hash),
            lambda_value=lambda_value,
            major_prior=major_prior,
            eps=eps,
            outer_max_iter=1 if polish else outer_max_iter,
            inner_max_iter=inner_max_iter,
            tol=tol,
            phi_start=start,
            solver_state=state,
            lower=context.lower,
            upper=context.upper,
            objective_shape=objective_shape,
            workset_max_bytes=workset_max_bytes,
            compressed_cache_max_bytes=compressed_cache_max_bytes,
            workset_add_batch=workset_add_batch,
            workset_max_expansions=workset_max_expansions,
            certificate_max_iter=certificate_max_iter,
            certificate_refinement_rounds=certificate_refinement_rounds,
            certificate_column_tol_scale=certificate_column_tol_scale,
            verbose=verbose,
        )

    cpu_fallback_context: SolverContext | None = None
    best_artifacts: RawFit | None = None
    best_artifacts_index = -1
    start_artifacts: list[RawFit] = []
    start_contexts: list[SolverContext] = []
    for start in start_bank:
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
                try:
                    precision_context = _float64_context(
                        data, selected_start_context
                    )
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
    return best_artifacts
