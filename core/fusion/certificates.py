from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import torch

from ..objective import TorchObservedModel, observed_one_sided_gradients_torch
from .torch_backend import (
    DEFAULT_EDGE_WORK_BYTES,
    downward_kink_mask_torch,
    edge_chunk_size,
    edge_kkt_maxima_from_diff_torch,
    edge_slices,
    edge_tensor_nbytes,
    graph_fusion_kkt_diagnostics_from_components_torch,
    graph_fusion_kkt_residual_from_grad_torch,
    project_stationarity_cone_torch,
    stationarity_residual_torch,
    streaming_graph_adjoint_edges_torch,
    validate_lambda_value,
)
from .graph_ops import graph_adjoint_edges, graph_forward_edges, project_dual_ball
from .types import (
    CertificateOptions,
    CompressedEdgeCertificate,
    DenseEdgeCertificate,
    GraphFusionCertificate,
    KKTAudit,
    KKTDiagnostics,
    SmoothGradientScope,
    TensorFusionGraph,
    WorkCounters,
    WorkLedger,
)

# Column generation assumes that the retained-edge subproblem has itself been
# solved.  Repeatedly enlarging an unconverged workset is both non-authoritative
# and can multiply ``max_iter`` by every configured expansion.  Three attempts
# leave room for the added columns to improve conditioning before failing closed
# to the caller's configured dense-fallback policy.
_MAX_CONSECUTIVE_UNCONVERGED_WORKSETS = 3
_CERTIFICATE_KKT_ATOL_SCALE = 5.0
_CERTIFICATE_PLATEAU_PATIENCE = 8
_CERTIFICATE_MOVING_PLATEAU_PATIENCE = 16
_CERTIFICATE_PLATEAU_ATOL_SCALE = 0.01
_CERTIFICATE_PLATEAU_EPS_SCALE = 32.0


def _update_refinement_plateau(
    *,
    anchor_residual: float,
    best_residual: float,
    mapping_delta: float,
    stalled_iterations: int,
    atol: float,
    dtype: torch.dtype,
) -> tuple[float, int, bool]:
    """Stop only after both residual progress and dual motion have stalled."""

    scale = max(1.0, abs(float(anchor_residual)))
    progress_floor = max(
        _CERTIFICATE_PLATEAU_ATOL_SCALE * float(atol),
        _CERTIFICATE_PLATEAU_EPS_SCALE * float(torch.finfo(dtype).eps) * scale,
    )
    if np.isfinite(best_residual) and (
        not np.isfinite(anchor_residual)
        or float(anchor_residual) - float(best_residual) > progress_floor
    ):
        return float(best_residual), 0, False
    stalled = int(stalled_iterations) + 1
    mapping_stalled = bool(
        np.isfinite(mapping_delta) and float(mapping_delta) <= progress_floor
    )
    patience = (
        _CERTIFICATE_PLATEAU_PATIENCE
        if mapping_stalled
        else _CERTIFICATE_MOVING_PLATEAU_PATIENCE
    )
    return float(anchor_residual), stalled, stalled >= patience


@dataclass(frozen=True, slots=True)
class CertificateProblem:
    """Fixed graph-fusion objective surface used by every certificate pass."""

    graph: TensorFusionGraph
    graph_hash: str
    lower: torch.Tensor
    upper: torch.Tensor
    lambda_value: float
    atol: float

    def __post_init__(self) -> None:
        if not str(self.graph_hash):
            raise ValueError("Certificate graph hash must be nonempty.")
        if self.lower.ndim != 2 or tuple(self.lower.shape) != tuple(self.upper.shape):
            raise ValueError("Certificate bounds must have one identical 2-D shape.")
        if int(self.lower.shape[0]) != int(self.graph.num_nodes):
            raise ValueError("Certificate bounds must have shape (M, S).")
        if float(self.lambda_value) < 0.0 or not math.isfinite(
            float(self.lambda_value)
        ):
            raise ValueError("Certificate lambda must be finite and nonnegative.")
        if float(self.atol) < 0.0 or not math.isfinite(float(self.atol)):
            raise ValueError("Certificate tolerance must be finite and nonnegative.")


@dataclass(frozen=True, slots=True)
class CertificateGradient:
    """One observed-objective generalized gradient and its kink provenance."""

    value: torch.Tensor
    scope: SmoothGradientScope
    directional_admissible: bool
    at_breakpoint: torch.Tensor


@dataclass(frozen=True, slots=True)
class CertificateAttempt:
    certificate: GraphFusionCertificate | None
    diagnostics: KKTDiagnostics
    status: str
    work_counters: WorkCounters = WorkCounters()


@dataclass(frozen=True, slots=True)
class DualRefinementResult:
    """Explicit-edge dual refinement before witness provenance is attached."""

    dual: torch.Tensor
    status: str
    dual_refined: bool
    audit: KKTAudit
    stationarity_before: float
    stationarity_after: float
    refinement_iterations: int


def _inadmissible_downward_kink_mask(
    downward_kink: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Mask downward kinks only where a coordinate can actually move."""

    resolution = (
        8.0
        * torch.finfo(phi.dtype).eps
        * (1.0 + torch.maximum(torch.abs(lower), torch.abs(upper)))
    )
    return downward_kink & ((upper - lower) > resolution)


def build_certificate_gradient(
    model: TorchObservedModel,
    phi: torch.Tensor,
    *,
    smooth_gradient: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    eps: float,
    tol: float,
    fusion_adjoint: torch.Tensor | None = None,
) -> CertificateGradient:
    """Build the sole generalized-gradient representation used by audits.

    The observed kernel supplies the left derivative. At any canonical
    breakpoint, a supplied fusion adjoint selects the interval member closest
    to stationarity; without one, the left derivative is retained exactly.
    Downward kinks remain selection-ineligible independently of that choice.
    """

    expected_shape = tuple(phi.shape)
    for name, value in (
        ("smooth_gradient", smooth_gradient),
        ("lower", lower),
        ("upper", upper),
    ):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}.")
    if fusion_adjoint is not None and tuple(fusion_adjoint.shape) != expected_shape:
        raise ValueError(f"fusion_adjoint must have shape {expected_shape}.")

    gradient_left, gradient_right, at_breakpoint = observed_one_sided_gradients_torch(
        model, phi, eps=float(eps)
    )
    gradient_lower = torch.where(
        at_breakpoint,
        torch.minimum(gradient_left, gradient_right),
        smooth_gradient,
    )
    gradient_upper = torch.where(
        at_breakpoint,
        torch.maximum(gradient_left, gradient_right),
        smooth_gradient,
    )
    target = smooth_gradient if fusion_adjoint is None else -fusion_adjoint
    value = torch.where(
        at_breakpoint,
        torch.minimum(torch.maximum(target, gradient_lower), gradient_upper),
        smooth_gradient,
    )
    downward_kink = downward_kink_mask_torch(
        gradient_left,
        gradient_right,
        at_breakpoint,
        tol=float(tol),
    )
    directional_admissible = not bool(
        torch.any(
            _inadmissible_downward_kink_mask(
                downward_kink,
                lower,
                upper,
                phi,
            )
        ).item()
    )
    return CertificateGradient(
        value=value,
        scope=(
            "clarke_piecewise_observed_objective_subgradient"
            if bool(torch.any(at_breakpoint).item())
            else "observed_objective"
        ),
        directional_admissible=directional_admissible,
        at_breakpoint=at_breakpoint,
    )


def _analytic_nonfused_adjoint(
    *,
    phi: torch.Tensor,
    labels: torch.Tensor,
    graph: TensorFusionGraph,
    lambda_value: float,
) -> torch.Tensor:
    adj = torch.zeros_like(phi)
    num_edges = int(graph.edge_u.numel())
    chunk_size = _compressed_edge_chunk_size(
        num_edges=num_edges,
        num_regions=int(phi.shape[1]),
        dtype=phi.dtype,
    )
    if lambda_value <= 0.0:
        return adj
    for start in range(0, num_edges, chunk_size):
        stop = min(start + chunk_size, num_edges)
        edge_u = graph.edge_u[start:stop]
        edge_v = graph.edge_v[start:stop]
        between = labels.index_select(0, edge_u) != labels.index_select(0, edge_v)
        if not bool(torch.any(between).item()):
            continue
        diff = phi.index_select(0, edge_u) - phi.index_select(0, edge_v)
        diff_norm = torch.linalg.vector_norm(diff, dim=1)
        active = between & (diff_norm > 0.0)
        if not bool(torch.any(active).item()):
            continue
        dual = (
            float(lambda_value)
            * graph.weight[start:stop][active, None]
            * diff[active]
            / diff_norm[active, None]
        )
        active_u = edge_u[active]
        active_v = edge_v[active]
        adj.index_add_(0, active_u, dual)
        adj.index_add_(0, active_v, dual, alpha=-1.0)
    return adj


def _initial_internal_tree_ids(
    *,
    labels: torch.Tensor,
    graph: TensorFusionGraph,
    dtype: torch.dtype,
) -> torch.Tensor:
    if labels.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=labels.device)
    num_blocks = int(torch.max(labels).item()) + 1
    if graph.is_complete:
        # Complete tensor graphs use canonical torch.triu_indices ordering, so
        # the full-graph edge ID can be computed directly.  A balanced binary
        # tree keeps the workset degree at most three; the former root-star had
        # degree |C|-1 and forced a 0.49/(|C|-1) projected-gradient step.
        num_nodes = int(labels.numel())
        node_ids = torch.arange(num_nodes, dtype=torch.long, device=labels.device)
        order = torch.argsort(labels, stable=True)
        sorted_labels = labels.index_select(0, order)
        counts = torch.bincount(labels, minlength=num_blocks)
        block_starts = torch.cumsum(counts, dim=0) - counts
        sorted_starts = block_starts.index_select(0, sorted_labels)
        local_rank = node_ids - sorted_starts
        child_positions = node_ids[local_rank > 0]
        if child_positions.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=labels.device)
        child_rank = local_rank.index_select(0, child_positions)
        parent_positions = (
            sorted_starts.index_select(0, child_positions) + (child_rank - 1) // 2
        )
        child_nodes = order.index_select(0, child_positions)
        parent_nodes = order.index_select(0, parent_positions)
        edge_u = torch.minimum(parent_nodes, child_nodes)
        edge_v = torch.maximum(parent_nodes, child_nodes)
        edge_ids = edge_u * (2 * num_nodes - edge_u - 1) // 2 + edge_v - edge_u - 1
        return torch.sort(edge_ids).values

    # Defensive fallback for non-complete graphs.  The compressed quotient
    # backend currently requires a complete graph, but certificate utilities
    # remain usable independently.
    roots = torch.full(
        (num_blocks,),
        int(labels.numel()),
        dtype=torch.long,
        device=labels.device,
    )
    node_ids = torch.arange(int(labels.numel()), device=labels.device)
    roots.scatter_reduce_(0, labels, node_ids, reduce="amin", include_self=True)
    num_edges = int(graph.edge_u.numel())
    chunk_size = _compressed_edge_chunk_size(
        num_edges=num_edges,
        num_regions=1,
        dtype=dtype,
    )
    selected: list[torch.Tensor] = []
    for start in range(0, num_edges, chunk_size):
        stop = min(start + chunk_size, num_edges)
        edge_u = graph.edge_u[start:stop]
        edge_v = graph.edge_v[start:stop]
        label_u = labels.index_select(0, edge_u)
        label_v = labels.index_select(0, edge_v)
        same = label_u == label_v
        block_root = roots.index_select(0, label_u)
        star = same & ((edge_u == block_root) | (edge_v == block_root))
        if bool(torch.any(star).item()):
            selected.append(torch.arange(start, stop, device=labels.device)[star])
    if not selected:
        return torch.empty(0, dtype=torch.long, device=labels.device)
    return torch.cat(selected)


def _merge_internal_support(
    *,
    inherited_ids: torch.Tensor,
    inherited_dual: torch.Tensor,
    added_ids: torch.Tensor,
    num_regions: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_ids = torch.unique(
        torch.cat(
            [
                inherited_ids.to(device=device, dtype=torch.long),
                added_ids.to(device=device, dtype=torch.long),
            ]
        ),
        sorted=True,
    )
    dual = torch.zeros(
        (int(all_ids.numel()), int(num_regions)), dtype=dtype, device=device
    )
    if inherited_ids.numel():
        positions = torch.searchsorted(all_ids, inherited_ids.to(device=device))
        dual.index_copy_(0, positions, inherited_dual.to(device=device, dtype=dtype))
    return all_ids, dual


def _workset_storage_bytes(
    *, edge_count: int, num_regions: int, dtype: torch.dtype
) -> int:
    value_bytes = int(torch.empty((), dtype=dtype).element_size())
    # During projected-gradient refinement the retained dual can overlap the
    # projected iterate, edge gradient, mapping, and a merge destination.
    peak_value_arrays = 5
    return int(edge_count) * (
        peak_value_arrays * int(num_regions) * value_bytes + 3 * 8 + 3 * value_bytes
    )


def _resource_limit_diagnostics(
    *,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    atol: float,
) -> KKTDiagnostics:
    """Fail-closed diagnostics when a certificate cannot be loaded safely."""

    box_violation = torch.maximum(
        torch.clamp(lower - phi, min=0.0), torch.clamp(phi - upper, min=0.0)
    )
    box_primal_violation = (
        float(torch.max(box_violation).item()) if box_violation.numel() else 0.0
    )
    box_scale = 1.0 + max(
        float(torch.max(torch.abs(lower)).item()) if lower.numel() else 0.0,
        float(torch.max(torch.abs(upper)).item()) if upper.numel() else 0.0,
    )
    return KKTDiagnostics(
        stationarity_residual=float("inf"),
        edge_subgradient_residual=float("inf"),
        dual_ball_residual=float("inf"),
        box_residual=box_primal_violation / max(box_scale, 1e-300),
        kkt_residual=float("inf"),
    )


def _workset_residual(
    *,
    base_grad: torch.Tensor,
    dual: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    phi: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adj = torch.zeros_like(base_grad)
    if dual.numel():
        adj.index_add_(0, edge_u, dual)
        adj.index_add_(0, edge_v, dual, alpha=-1.0)
    total_grad = base_grad + adj
    cone_projection = project_stationarity_cone_torch(
        total_grad,
        phi=phi,
        lower=lower,
        upper=upper,
    )
    return total_grad - cone_projection, adj, total_grad


def _optimize_internal_workset(
    *,
    base_grad: torch.Tensor,
    dual_start: torch.Tensor,
    edge_ids: torch.Tensor,
    graph: TensorFusionGraph,
    phi: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    options: CertificateOptions,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    int,
    WorkCounters,
]:
    if edge_ids.numel() == 0:
        residual, _, _ = _workset_residual(
            base_grad=base_grad,
            dual=dual_start,
            edge_u=graph.edge_u[:0],
            edge_v=graph.edge_v[:0],
            phi=phi,
            lower=lower,
            upper=upper,
        )
        return (
            dual_start,
            residual,
            torch.zeros_like(phi),
            0.0,
            0,
            WorkCounters(),
        )
    edge_u = graph.edge_u.index_select(0, edge_ids)
    edge_v = graph.edge_v.index_select(0, edge_ids)
    radius = float(lambda_value) * graph.weight.index_select(0, edge_ids)
    degree = torch.zeros(int(phi.shape[0]), dtype=phi.dtype, device=phi.device)
    ones = torch.ones(int(edge_ids.numel()), dtype=phi.dtype, device=phi.device)
    degree.index_add_(0, edge_u, ones)
    degree.index_add_(0, edge_v, ones)
    d_max = max(float(torch.max(degree).item()), 1.0)
    step = 0.49 / d_max
    dual = project_dual_ball(dual_start, radius)
    mapping_scale = 1.0 + float(torch.max(radius).item())
    # A scalar read synchronizes the CUDA stream.  Checking every projected
    # gradient step made small compressed worksets host-latency bound; batched
    # checks retain the same stopping certificate and merely perform a few
    # extra device-resident iterations after convergence.
    check_every = 16 if phi.device.type == "cuda" else 1
    mapping_residual = float("inf")
    residual = torch.zeros_like(phi)
    adj = torch.zeros_like(phi)
    iterations = 0
    work = WorkLedger()
    for iteration in range(int(options.max_iter)):
        iterations = iteration + 1
        residual, adj, _ = _workset_residual(
            base_grad=base_grad,
            dual=dual,
            edge_u=edge_u,
            edge_v=edge_v,
            phi=phi,
            lower=lower,
            upper=upper,
        )
        work.charge_edge_passes(
            edge_count=int(edge_ids.numel()),
            num_regions=int(phi.shape[1]),
        )
        edge_gradient = residual.index_select(0, edge_u) - residual.index_select(
            0, edge_v
        )
        work.charge_edge_passes(
            edge_count=int(edge_ids.numel()),
            num_regions=int(phi.shape[1]),
        )
        projected = project_dual_ball(dual - step * edge_gradient, radius)
        mapping = (dual - projected) / step
        dual = projected
        if iterations % check_every == 0 or iterations >= int(options.max_iter):
            mapping_residual = (
                float(torch.max(torch.linalg.vector_norm(mapping, dim=1)).item())
                / mapping_scale
            )
            if mapping_residual <= float(options.mapping_tolerance):
                break
    residual, adj, _ = _workset_residual(
        base_grad=base_grad,
        dual=dual,
        edge_u=edge_u,
        edge_v=edge_v,
        phi=phi,
        lower=lower,
        upper=upper,
    )
    work.charge_edge_passes(
        edge_count=int(edge_ids.numel()),
        num_regions=int(phi.shape[1]),
    )
    return (
        dual,
        residual,
        adj,
        mapping_residual,
        iterations,
        work.total,
    )


def _scan_omitted_internal_edges(
    *,
    residual: torch.Tensor,
    labels: torch.Tensor,
    support_ids: torch.Tensor,
    graph: TensorFusionGraph,
    scale: float,
    add_batch: int,
) -> tuple[float, torch.Tensor]:
    num_edges = int(graph.edge_u.numel())
    chunk_size = _compressed_edge_chunk_size(
        num_edges=num_edges,
        num_regions=int(residual.shape[1]),
        dtype=residual.dtype,
    )
    best_scores: list[torch.Tensor] = []
    best_ids: list[torch.Tensor] = []
    maximum = 0.0
    for start in range(0, num_edges, chunk_size):
        stop = min(start + chunk_size, num_edges)
        edge_u = graph.edge_u[start:stop]
        edge_v = graph.edge_v[start:stop]
        internal = labels.index_select(0, edge_u) == labels.index_select(0, edge_v)
        chunk_ids = torch.arange(start, stop, device=residual.device)
        if support_ids.numel():
            positions = torch.searchsorted(support_ids, chunk_ids)
            safe_positions = positions.clamp(max=int(support_ids.numel()) - 1)
            included = (positions < int(support_ids.numel())) & (
                support_ids.index_select(0, safe_positions) == chunk_ids
            )
            internal &= ~included
        if not bool(torch.any(internal).item()):
            continue
        gradients = residual.index_select(0, edge_u[internal]) - residual.index_select(
            0, edge_v[internal]
        )
        scores = torch.linalg.vector_norm(gradients, dim=1) / max(float(scale), 1e-300)
        maximum = max(maximum, float(torch.max(scores).item()))
        count = min(int(add_batch), int(scores.numel()))
        values, positions = torch.topk(scores, k=count, largest=True, sorted=False)
        best_scores.append(values)
        best_ids.append(chunk_ids[internal].index_select(0, positions))
    if not best_scores:
        return maximum, torch.empty(0, dtype=torch.long, device=residual.device)
    scores = torch.cat(best_scores)
    ids = torch.cat(best_ids)
    count = min(int(add_batch), int(scores.numel()))
    _values, positions = torch.topk(scores, k=count, largest=True, sorted=True)
    return maximum, ids.index_select(0, positions)


def _refine_compressed_certificate(
    *,
    certificate: CompressedEdgeCertificate,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    gradient_scope: SmoothGradientScope,
    graph: TensorFusionGraph,
    graph_hash: str,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    atol: float,
    options: CertificateOptions,
) -> CertificateAttempt:
    if certificate.graph_hash != str(graph_hash):
        raise ValueError("Compressed certificate graph hash does not match the graph.")
    raw_edge_ids = certificate.internal_edge_ids
    raw_dual = certificate.internal_dual
    if raw_edge_ids.ndim != 1 or tuple(raw_dual.shape) != (
        int(raw_edge_ids.numel()),
        int(phi.shape[1]),
    ):
        raise ValueError("Compressed internal dual must have shape (W, S).")
    if _workset_storage_bytes(
        edge_count=int(raw_edge_ids.numel()),
        num_regions=int(phi.shape[1]),
        dtype=phi.dtype,
    ) > int(options.memory.max_workset_bytes):
        diag = _resource_limit_diagnostics(
            phi=phi,
            grad_smooth=grad_smooth,
            lower=lower,
            upper=upper,
            atol=atol,
        )
        return CertificateAttempt(
            certificate=certificate,
            diagnostics=diag,
            status="resource_limit",
        )
    labels, centers, inherited_ids, inherited_dual = _validated_compressed_tensors(
        certificate,
        phi=phi,
        graph=graph,
        graph_hash=graph_hash,
    )
    tree_ids = _initial_internal_tree_ids(
        labels=labels,
        graph=graph,
        dtype=phi.dtype,
    )
    combined_support_count = int(
        torch.unique(torch.cat([inherited_ids, tree_ids])).numel()
    )
    if _workset_storage_bytes(
        edge_count=combined_support_count,
        num_regions=int(phi.shape[1]),
        dtype=phi.dtype,
    ) > int(options.memory.max_workset_bytes):
        diag = _resource_limit_diagnostics(
            phi=phi,
            grad_smooth=grad_smooth,
            lower=lower,
            upper=upper,
            atol=atol,
        )
        return CertificateAttempt(
            certificate=certificate,
            diagnostics=diag,
            status="resource_limit",
        )
    support_ids, dual = _merge_internal_support(
        inherited_ids=inherited_ids,
        inherited_dual=inherited_dual,
        added_ids=tree_ids,
        num_regions=int(phi.shape[1]),
        dtype=phi.dtype,
        device=phi.device,
    )
    between_adj = _analytic_nonfused_adjoint(
        phi=phi,
        labels=labels,
        graph=graph,
        lambda_value=lambda_value,
    )
    base_grad = grad_smooth + between_adj
    num_edges = int(graph.edge_u.numel())
    num_regions = int(phi.shape[1])
    work = WorkLedger()
    work.charge_edge_passes(
        edge_count=num_edges,
        num_regions=num_regions,
        passes=int(num_edges > 0 and float(lambda_value) > 0.0),
    )
    if not bool(graph.is_complete) and num_edges > 0:
        # _initial_internal_tree_ids scans a non-complete graph to identify a
        # deterministic star support.  Complete graphs use direct edge IDs.
        work.charge_edge_passes(
            edge_count=num_edges,
            num_regions=num_regions,
        )
    full_kkt_audited = False
    # A nonempty inherited support may already be authoritative, so give it one
    # full-graph fast-path audit. Fresh proposals go directly to the cheap
    # workset/missing-column gates and pay for a full audit only if those pass.
    has_inherited_fast_path = bool(inherited_ids.numel())
    before = _resource_limit_diagnostics(
        phi=phi,
        grad_smooth=grad_smooth,
        lower=lower,
        upper=upper,
        atol=atol,
    )
    if has_inherited_fast_path:
        before_audit = _compressed_graph_fusion_kkt(
            certificate=certificate,
            phi=phi,
            grad_smooth=grad_smooth,
            graph=graph,
            graph_hash=graph_hash,
            lower=lower,
            upper=upper,
            lambda_value=lambda_value,
            atol=atol,
        )
        before = before_audit.diagnostics
        work.charge(before_audit.work)
        full_kkt_audited = True
        work.charge_certificate_work(full_graph_passes=1)
    if has_inherited_fast_path and before.kkt_residual <= 5.0 * float(atol):
        # The inherited compressed state has already passed a full
        # original-graph audit.  Re-optimizing its workset cannot strengthen
        # that certificate and was the dominant CUDA cost on favorable warm
        # starts.
        certified = CompressedEdgeCertificate(
            labels=certificate.labels,
            centers=certificate.centers,
            internal_edge_ids=certificate.internal_edge_ids,
            internal_dual=certificate.internal_dual,
            graph_hash=certificate.graph_hash,
            gradient_scope=gradient_scope,
        )
        return CertificateAttempt(
            certificate=certified,
            diagnostics=before,
            status="certified",
            work_counters=work.total,
        )
    status = "not_certified"
    force_rounds = 0
    unconverged_worksets = 0
    final_diag = before
    for _expansion in range(int(options.max_expansions)):
        dual, residual, work_adj, mapping_residual, iterations, workset_work = (
            _optimize_internal_workset(
                base_grad=base_grad,
                dual_start=dual,
                edge_ids=support_ids,
                graph=graph,
                phi=phi,
                lower=lower,
                upper=upper,
                lambda_value=lambda_value,
                options=options,
            )
        )
        work.charge_certificate_work(iterations=int(iterations))
        work.charge(workset_work)
        current = CompressedEdgeCertificate(
            labels=labels,
            centers=centers,
            internal_edge_ids=support_ids,
            internal_dual=dual,
            graph_hash=graph_hash,
            gradient_scope=gradient_scope,
        )
        # Workset optimization changes the terminal dual, so any audit of an
        # inherited predecessor no longer certifies this representation.
        full_kkt_audited = False
        mapping_ready = bool(
            math.isfinite(mapping_residual)
            and mapping_residual <= float(options.mapping_tolerance)
        )
        if not mapping_ready:
            unconverged_worksets += 1
            if unconverged_worksets >= _MAX_CONSECUTIVE_UNCONVERGED_WORKSETS:
                status = "workset_incomplete"
                certificate = current
                break
            # More columns cannot certify an unresolved retained-edge problem.
            continue
        unconverged_worksets = 0

        scale = (
            1.0
            + float(torch.linalg.norm(grad_smooth).item())
            + float(torch.linalg.norm(between_adj + work_adj).item())
        )
        column_residual, proposed_ids = _scan_omitted_internal_edges(
            residual=residual,
            labels=labels,
            support_ids=support_ids,
            graph=graph,
            scale=scale,
            add_batch=int(options.add_batch),
        )
        work.charge_certificate_work(full_graph_passes=1)
        work.charge_edge_passes(
            edge_count=num_edges,
            num_regions=num_regions,
        )
        column_ready = column_residual <= float(options.column_tolerance)
        should_expand = not column_ready
        if column_ready:
            final_audit = _compressed_graph_fusion_kkt(
                certificate=current,
                phi=phi,
                grad_smooth=grad_smooth,
                graph=graph,
                graph_hash=graph_hash,
                lower=lower,
                upper=upper,
                lambda_value=lambda_value,
                atol=atol,
            )
            final_diag = final_audit.diagnostics
            work.charge(final_audit.work)
            full_kkt_audited = True
            work.charge_certificate_work(full_graph_passes=1)
            if final_diag.kkt_residual <= 5.0 * float(atol):
                status = "certified"
                certificate = current
                break
            if force_rounds < int(options.refinement_rounds):
                should_expand = bool(proposed_ids.numel())
                force_rounds += 1
        if not should_expand or not proposed_ids.numel():
            status = "not_certified"
            certificate = current
            break
        new_count = int(torch.unique(torch.cat([support_ids, proposed_ids])).numel())
        if _workset_storage_bytes(
            edge_count=new_count,
            num_regions=int(phi.shape[1]),
            dtype=phi.dtype,
        ) > int(options.memory.max_workset_bytes):
            status = "resource_limit"
            certificate = current
            break
        support_ids, dual = _merge_internal_support(
            inherited_ids=support_ids,
            inherited_dual=dual,
            added_ids=proposed_ids,
            num_regions=int(phi.shape[1]),
            dtype=phi.dtype,
            device=phi.device,
        )
    else:
        status = "workset_incomplete"
        certificate = CompressedEdgeCertificate(
            labels=labels,
            centers=centers,
            internal_edge_ids=support_ids,
            internal_dual=dual,
            graph_hash=graph_hash,
            gradient_scope=gradient_scope,
        )
    if status == "not_certified" and not full_kkt_audited:
        # ``not_certified`` is reserved for a complete original-graph KKT
        # audit that misses tolerance. Representation-limited exits remain
        # explicit so retry policy never depends on work-accounting metadata.
        status = "workset_incomplete"
    return CertificateAttempt(
        certificate=certificate,
        diagnostics=final_diag,
        status=status,
        work_counters=work.total,
    )


def _compressed_edge_chunk_size(
    *,
    num_edges: int,
    num_regions: int,
    dtype: torch.dtype,
    work_bytes: int = 64 * 1024 * 1024,
) -> int:
    if num_edges <= 0:
        return 1
    value_bytes = torch.empty((), dtype=dtype).element_size()
    # diff, dual, prox input, and residual temporaries can coexist.
    bytes_per_edge = max(6 * int(num_regions) * int(value_bytes) + 32, 1)
    proposed = max(1, min(int(num_edges), int(work_bytes) // bytes_per_edge))
    # A compressed path should not accidentally materialize an E-by-S chunk.
    if num_edges > 1:
        proposed = min(proposed, num_edges - 1)
    return proposed


def _validated_compressed_tensors(
    certificate: CompressedEdgeCertificate,
    *,
    phi: torch.Tensor,
    graph: TensorFusionGraph,
    graph_hash: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if certificate.graph_hash != str(graph_hash):
        raise ValueError("Compressed certificate graph hash does not match the graph.")
    labels = certificate.labels.to(device=phi.device, dtype=torch.long)
    centers = certificate.centers.to(device=phi.device, dtype=phi.dtype)
    edge_ids = certificate.internal_edge_ids.to(device=phi.device, dtype=torch.long)
    dual = certificate.internal_dual.to(device=phi.device, dtype=phi.dtype)
    if labels.ndim != 1 or int(labels.numel()) != int(graph.num_nodes):
        raise ValueError("Compressed certificate labels must have shape (M,).")
    if centers.ndim != 2 or int(centers.shape[1]) != int(phi.shape[1]):
        raise ValueError("Compressed certificate centers must have shape (K, S).")
    if labels.numel() and (
        bool(torch.any(labels < 0).item())
        or bool(torch.any(labels >= int(centers.shape[0])).item())
    ):
        raise ValueError("Compressed certificate labels are outside the center range.")
    lifted = centers.index_select(0, labels)
    if tuple(lifted.shape) != tuple(phi.shape) or not torch.equal(lifted, phi):
        raise ValueError(
            "Compressed certificate is stale for the supplied primal point."
        )
    if edge_ids.ndim != 1:
        raise ValueError("Compressed internal edge IDs must be one-dimensional.")
    if tuple(dual.shape) != (int(edge_ids.numel()), int(phi.shape[1])):
        raise ValueError("Compressed internal dual must have shape (W, S).")
    num_edges = int(graph.edge_u.numel())
    if edge_ids.numel() and (
        bool(torch.any(edge_ids < 0).item())
        or bool(torch.any(edge_ids >= num_edges).item())
    ):
        raise ValueError("Compressed internal edge IDs are outside the graph range.")
    if edge_ids.numel() > 1 and not bool(
        torch.all(edge_ids[1:] > edge_ids[:-1]).item()
    ):
        raise ValueError("Compressed internal edge IDs must be sorted and unique.")
    if edge_ids.numel():
        support_u = graph.edge_u.index_select(0, edge_ids)
        support_v = graph.edge_v.index_select(0, edge_ids)
        if not bool(
            torch.all(
                labels.index_select(0, support_u) == labels.index_select(0, support_v)
            ).item()
        ):
            raise ValueError(
                "Explicit compressed support must contain only internal edges."
            )
    return labels, centers, edge_ids, dual


def _compressed_graph_fusion_kkt(
    *,
    certificate: CompressedEdgeCertificate,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    graph: TensorFusionGraph,
    graph_hash: str,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    atol: float,
) -> KKTAudit:
    labels, _centers, support_ids, support_dual = _validated_compressed_tensors(
        certificate,
        phi=phi,
        graph=graph,
        graph_hash=graph_hash,
    )
    num_edges = int(graph.edge_u.numel())
    chunk_size = _compressed_edge_chunk_size(
        num_edges=num_edges,
        num_regions=int(phi.shape[1]),
        dtype=phi.dtype,
    )
    adj = torch.zeros_like(phi)
    max_edge_residual = 0.0
    max_ball_residual = 0.0
    max_radius = 0.0
    max_scaled_edge_residual = 0.0
    max_scaled_ball_residual = 0.0
    nonzero_edge_count = 0
    for start in range(0, num_edges, chunk_size):
        stop = min(start + chunk_size, num_edges)
        edge_u = graph.edge_u[start:stop]
        edge_v = graph.edge_v[start:stop]
        diff = phi.index_select(0, edge_u) - phi.index_select(0, edge_v)
        radius = float(lambda_value) * graph.weight[start:stop].to(dtype=phi.dtype)
        dual_chunk = torch.zeros_like(diff)
        if lambda_value > 0.0:
            same = labels.index_select(0, edge_u) == labels.index_select(0, edge_v)
            diff_norm = torch.linalg.vector_norm(diff, dim=1)
            nonzero_edge_count += int(
                torch.count_nonzero(diff_norm > float(atol)).item()
            )
            nonfused = (~same) & (diff_norm > 0.0)
            if bool(torch.any(nonfused).item()):
                dual_chunk[nonfused] = (
                    radius[nonfused, None] * diff[nonfused] / diff_norm[nonfused, None]
                )
            if support_ids.numel():
                chunk_ids = torch.arange(start, stop, device=phi.device)
                positions = torch.searchsorted(support_ids, chunk_ids)
                safe_positions = positions.clamp(max=int(support_ids.numel()) - 1)
                included = (positions < int(support_ids.numel())) & (
                    support_ids.index_select(0, safe_positions) == chunk_ids
                )
                if bool(torch.any(included).item()):
                    dual_chunk[included] = support_dual.index_select(
                        0, safe_positions[included]
                    )
        adj.index_add_(0, edge_u, dual_chunk)
        adj.index_add_(0, edge_v, dual_chunk, alpha=-1.0)

        if num_edges > 0 and lambda_value > 0.0:
            (
                edge_residual,
                ball_residual,
                radius_max,
                scaled_edge_residual,
                scaled_ball_residual,
            ) = edge_kkt_maxima_from_diff_torch(
                diff=diff,
                dual=dual_chunk,
                radius=radius,
            )
            max_edge_residual = max(
                max_edge_residual,
                float(edge_residual.item()),
            )
            max_ball_residual = max(
                max_ball_residual,
                float(ball_residual.item()),
            )
            max_radius = max(
                max_radius,
                float(radius_max.item()),
            )
            max_scaled_edge_residual = max(
                max_scaled_edge_residual, float(scaled_edge_residual.item())
            )
            max_scaled_ball_residual = max(
                max_scaled_ball_residual, float(scaled_ball_residual.item())
            )

    diagnostics = graph_fusion_kkt_diagnostics_from_components_torch(
        phi=phi,
        grad_smooth=grad_smooth,
        adj=adj,
        lower=lower,
        upper=upper,
        atol=atol,
        max_edge_residual=max_edge_residual,
        max_ball_residual=max_ball_residual,
        max_radius=max_radius,
        max_scaled_edge_residual=max_scaled_edge_residual,
        max_scaled_ball_residual=max_scaled_ball_residual,
    )
    work = WorkLedger()
    work.charge_edge_passes(
        edge_count=num_edges,
        num_regions=int(phi.shape[1]),
    )
    return KKTAudit(
        diagnostics=diagnostics,
        work=work.total,
        fused_edges=(num_edges - nonzero_edge_count if lambda_value > 0.0 else None),
        nonzero_edges=(nonzero_edge_count if lambda_value > 0.0 else None),
    )


def _refine_graph_fusion_dual_certificate_streaming_torch(
    *,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    dual_kkt: torch.Tensor | None,
    lower: torch.Tensor,
    upper: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
    atol: float,
    max_iter: int,
    before_audit: KKTAudit,
    edge_work_bytes: int | None,
) -> DualRefinementResult:
    """Memory-bounded counterpart of the final dual-certificate refinement."""
    num_edges = int(edge_u.numel())
    num_regions = int(phi.shape[1])
    num_nodes = int(phi.shape[0])
    chunk_size = edge_chunk_size(
        num_edges=num_edges,
        num_regions=num_regions,
        dtype=phi.dtype,
        work_bytes=edge_work_bytes,
    )
    incoming_valid = bool(
        dual_kkt is not None and tuple(dual_kkt.shape) == (num_edges, num_regions)
    )
    incoming = (
        dual_kkt.to(dtype=phi.dtype, device=phi.device) if incoming_valid else None
    )

    work = WorkLedger(before_audit.work)
    fused_edges, nonzero_edges = before_audit.require_activity(edge_count=num_edges)

    dual = torch.zeros(
        (num_edges, num_regions),
        dtype=phi.dtype,
        device=phi.device,
    )
    for edge_slice in edge_slices(num_edges, chunk_size):
        diff = graph_forward_edges(
            phi,
            edge_u=edge_u[edge_slice],
            edge_v=edge_v[edge_slice],
        )
        diff_norm = torch.linalg.vector_norm(diff, dim=1)
        active = diff_norm > float(atol)
        radius = float(lambda_value) * edge_w[edge_slice]
        analytic_chunk = (
            radius[:, None] * diff / diff_norm[:, None].clamp_min(float(atol))
        )
        if incoming is None:
            dual[edge_slice].copy_(
                torch.where(
                    active[:, None],
                    analytic_chunk,
                    torch.zeros_like(analytic_chunk),
                )
            )
        else:
            projected_incoming = project_dual_ball(
                incoming[edge_slice],
                radius,
            )
            dual[edge_slice].copy_(
                torch.where(
                    active[:, None],
                    analytic_chunk,
                    projected_incoming,
                )
            )
    if num_edges > 0:
        work.charge_edge_passes(
            edge_count=num_edges,
            num_regions=num_regions,
        )

    analytic_audit = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad_smooth,
        dual_kkt=dual,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        atol=atol,
        edge_work_bytes=edge_work_bytes,
    )
    work.charge(analytic_audit.work)
    best_diag = analytic_audit.diagnostics
    best_residual = float(analytic_audit.diagnostics.kkt_residual)
    best_source = "analytic"
    best_dual: torch.Tensor = dual.clone()
    if incoming is not None:
        incoming_residual = float(before_audit.diagnostics.kkt_residual)
        if np.isfinite(incoming_residual) and incoming_residual <= best_residual:
            best_diag = before_audit.diagnostics
            best_residual = incoming_residual
            best_source = "incoming"
            best_dual = incoming

    refinement_iterations = 0
    kkt_target = _CERTIFICATE_KKT_ATOL_SCALE * float(atol)
    if fused_edges > 0 and best_residual > kkt_target:
        degree = torch.bincount(edge_u, minlength=num_nodes) + torch.bincount(
            edge_v,
            minlength=num_nodes,
        )
        step = 0.25 / max(float(torch.max(degree).item()), 1.0)
        plateau_anchor = best_residual
        stalled_iterations = 0
        for _ in range(max(int(max_iter), 1)):
            refinement_iterations += 1
            mapping_delta = 0.0
            adj = streaming_graph_adjoint_edges_torch(
                dual,
                edge_u=edge_u,
                edge_v=edge_v,
                num_nodes=num_nodes,
                dtype=phi.dtype,
                device=phi.device,
                scale=1.0,
                chunk_size=chunk_size,
            )
            work.charge_edge_passes(
                edge_count=num_edges,
                num_regions=num_regions,
            )
            stat = stationarity_residual_torch(
                total_grad=grad_smooth + adj,
                phi=phi,
                lower=lower,
                upper=upper,
                atol=atol,
            )
            stationarity_edge_count = 0
            for edge_slice in edge_slices(num_edges, chunk_size):
                diff = graph_forward_edges(
                    phi,
                    edge_u=edge_u[edge_slice],
                    edge_v=edge_v[edge_slice],
                )
                fused = torch.linalg.vector_norm(diff, dim=1) <= float(atol)
                if not bool(torch.any(fused).item()):
                    continue
                stat_diff = graph_forward_edges(
                    stat,
                    edge_u=edge_u[edge_slice],
                    edge_v=edge_v[edge_slice],
                )
                stationarity_edge_count += int(edge_u[edge_slice].numel())
                radius = float(lambda_value) * edge_w[edge_slice]
                projected = project_dual_ball(
                    dual[edge_slice] - float(step) * stat_diff,
                    radius,
                )
                mapping_delta = max(
                    mapping_delta,
                    float(
                        torch.max(
                            torch.abs(projected[fused] - dual[edge_slice][fused])
                        ).item()
                    ),
                )
                dual[edge_slice].copy_(
                    torch.where(
                        fused[:, None],
                        projected,
                        dual[edge_slice],
                    )
                )
            # The first forward covers every edge.  The stationarity forward
            # is evaluated only on chunks containing fused edges; charge its
            # logical certificate sweep as one EPE as well.
            work.charge_edge_passes(
                edge_count=num_edges,
                num_regions=num_regions,
            )
            work.charge_edge_passes(
                edge_count=stationarity_edge_count,
                num_regions=num_regions,
            )
            iteration_audit = graph_fusion_kkt_residual_from_grad_torch(
                phi=phi,
                grad_smooth=grad_smooth,
                dual_kkt=dual,
                lower=lower,
                upper=upper,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                lambda_value=lambda_value,
                atol=atol,
                edge_work_bytes=edge_work_bytes,
            )
            work.charge(iteration_audit.work)
            residual = float(iteration_audit.diagnostics.kkt_residual)
            if residual < best_residual:
                best_residual = residual
                best_diag = iteration_audit.diagnostics
                if best_source == "incoming":
                    best_dual = dual.clone()
                else:
                    best_dual.copy_(dual)
                best_source = "refined"
            if residual <= kkt_target:
                break
            plateau_anchor, stalled_iterations, plateaued = (
                _update_refinement_plateau(
                    anchor_residual=plateau_anchor,
                    best_residual=best_residual,
                    mapping_delta=mapping_delta,
                    stalled_iterations=stalled_iterations,
                    atol=atol,
                    dtype=phi.dtype,
                )
            )
            if plateaued:
                break

    if best_source == "incoming":
        status = "input_dual_retained"
    elif fused_edges > 0:
        status = "refined_fused_edge_dual"
    else:
        status = "analytic_nonfused_dual"
    return DualRefinementResult(
        dual=best_dual,
        status=status,
        dual_refined=bool(best_source != "incoming"),
        audit=KKTAudit(
            diagnostics=best_diag,
            work=work.total,
            fused_edges=fused_edges,
            nonzero_edges=nonzero_edges,
        ),
        stationarity_before=float(before_audit.diagnostics.stationarity_residual),
        stationarity_after=float(best_diag.stationarity_residual),
        refinement_iterations=int(refinement_iterations),
    )


def refine_graph_fusion_dual_certificate_torch(
    *,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    dual_kkt: torch.Tensor | None,
    lower: torch.Tensor,
    upper: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    lambda_value: float,
    atol: float,
    max_iter: int = 96,
    edge_work_bytes: int | None = None,
) -> DualRefinementResult:
    lambda_value = validate_lambda_value(lambda_value)
    before_audit = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad_smooth,
        dual_kkt=dual_kkt,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        atol=atol,
        edge_work_bytes=edge_work_bytes,
    )
    work = WorkLedger(before_audit.work)
    if edge_u.numel() == 0 or lambda_value <= 0.0:
        dual = torch.zeros(
            (int(edge_u.numel()), int(phi.shape[1])), dtype=phi.dtype, device=phi.device
        )
        after_audit = graph_fusion_kkt_residual_from_grad_torch(
            phi=phi,
            grad_smooth=grad_smooth,
            dual_kkt=dual,
            lower=lower,
            upper=upper,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            lambda_value=lambda_value,
            atol=atol,
            edge_work_bytes=edge_work_bytes,
        )
        work.charge(after_audit.work)
        return DualRefinementResult(
            dual=dual,
            status="zero_penalty_no_dual_needed",
            dual_refined=False,
            audit=KKTAudit(
                diagnostics=after_audit.diagnostics,
                work=work.total,
            ),
            stationarity_before=float(
                before_audit.diagnostics.stationarity_residual
            ),
            stationarity_after=float(after_audit.diagnostics.stationarity_residual),
            refinement_iterations=0,
        )

    budget = (
        DEFAULT_EDGE_WORK_BYTES if edge_work_bytes is None else int(edge_work_bytes)
    )
    incoming_valid = bool(
        dual_kkt is not None
        and tuple(dual_kkt.shape) == (int(edge_u.numel()), int(phi.shape[1]))
    )
    incoming_residual = float(before_audit.diagnostics.kkt_residual)
    if (
        incoming_valid
        and np.isfinite(incoming_residual)
        and incoming_residual <= _CERTIFICATE_KKT_ATOL_SCALE * float(atol)
    ):
        fused_edges, nonzero_edges = before_audit.require_activity(
            edge_count=int(edge_u.numel())
        )
        incoming = dual_kkt.to(dtype=phi.dtype, device=phi.device)
        return DualRefinementResult(
            dual=incoming,
            status="input_dual_retained",
            dual_refined=False,
            audit=KKTAudit(
                diagnostics=before_audit.diagnostics,
                work=work.total,
                fused_edges=fused_edges,
                nonzero_edges=nonzero_edges,
            ),
            stationarity_before=float(
                before_audit.diagnostics.stationarity_residual
            ),
            stationarity_after=float(before_audit.diagnostics.stationarity_residual),
            refinement_iterations=0,
        )
    if (
        edge_tensor_nbytes(
            num_edges=int(edge_u.numel()),
            num_regions=int(phi.shape[1]),
            dtype=phi.dtype,
        )
        > budget
    ):
        return _refine_graph_fusion_dual_certificate_streaming_torch(
            phi=phi,
            grad_smooth=grad_smooth,
            dual_kkt=dual_kkt,
            lower=lower,
            upper=upper,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            lambda_value=lambda_value,
            atol=atol,
            max_iter=max_iter,
            before_audit=before_audit,
            edge_work_bytes=budget,
        )

    diff = graph_forward_edges(phi, edge_u=edge_u, edge_v=edge_v)
    work.charge_edge_passes(
        edge_count=int(edge_u.numel()),
        num_regions=int(phi.shape[1]),
    )
    diff_norm = torch.linalg.norm(diff, dim=1)
    radius = float(lambda_value) * edge_w
    active = diff_norm > float(atol)
    fused = ~active
    dual = torch.zeros(
        (int(edge_u.numel()), int(phi.shape[1])), dtype=phi.dtype, device=phi.device
    )
    if torch.any(active):
        dual[active] = (
            radius[active, None]
            * diff[active]
            / diff_norm[active, None].clamp_min(float(atol))
        )
    if (
        torch.any(fused)
        and dual_kkt is not None
        and tuple(dual_kkt.shape) == tuple(dual.shape)
    ):
        dual[fused] = dual_kkt.to(dtype=phi.dtype, device=phi.device)[fused]
        fused_radius = radius[fused]
        dual[fused] = project_dual_ball(dual[fused], fused_radius)

    analytic_audit = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad_smooth,
        dual_kkt=dual,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        atol=atol,
    )
    work.charge(analytic_audit.work)
    best_dual = dual.clone()
    best_diag = analytic_audit.diagnostics
    best_residual = float(analytic_audit.diagnostics.kkt_residual)
    best_source = "analytic"

    # Reconstructing the analytic active-edge subgradient can improve edge
    # feasibility while worsening stationarity at a finite-accuracy primal
    # iterate.  A certificate-refinement routine must be monotone in the full
    # KKT residual, so retain the incoming actual ADMM multiplier whenever it
    # is the stronger certificate.
    if dual_kkt is not None and tuple(dual_kkt.shape) == tuple(dual.shape):
        incoming_residual = float(before_audit.diagnostics.kkt_residual)
        if np.isfinite(incoming_residual) and incoming_residual <= best_residual:
            best_dual = dual_kkt.to(dtype=phi.dtype, device=phi.device).clone()
            best_diag = before_audit.diagnostics
            best_residual = incoming_residual
            best_source = "incoming"
    refinement_iterations = 0
    kkt_target = _CERTIFICATE_KKT_ATOL_SCALE * float(atol)
    if torch.any(fused) and best_residual > kkt_target:
        degree = torch.bincount(
            torch.cat([edge_u, edge_v]),
            minlength=int(phi.shape[0]),
        ).max()
        step = 0.25 / max(float(degree.item()), 1.0)
        plateau_anchor = best_residual
        stalled_iterations = 0
        for _ in range(max(int(max_iter), 1)):
            refinement_iterations += 1
            adj = graph_adjoint_edges(
                dual,
                edge_u=edge_u,
                edge_v=edge_v,
                num_nodes=int(phi.shape[0]),
            )
            work.charge_edge_passes(
                edge_count=int(edge_u.numel()),
                num_regions=int(phi.shape[1]),
            )
            total_grad = grad_smooth + adj
            stat = stationarity_residual_torch(
                total_grad=total_grad,
                phi=phi,
                lower=lower,
                upper=upper,
                atol=atol,
            )
            fused_before = dual[fused]
            fused_after = (
                fused_before
                - float(step)
                * (graph_forward_edges(stat, edge_u=edge_u, edge_v=edge_v)[fused])
            )
            work.charge_edge_passes(
                edge_count=int(edge_u.numel()),
                num_regions=int(phi.shape[1]),
            )
            fused_radius = radius[fused]
            fused_after = project_dual_ball(fused_after, fused_radius)
            mapping_delta = float(
                torch.max(torch.abs(fused_after - fused_before)).item()
            )
            dual[fused] = fused_after
            iteration_audit = graph_fusion_kkt_residual_from_grad_torch(
                phi=phi,
                grad_smooth=grad_smooth,
                dual_kkt=dual,
                lower=lower,
                upper=upper,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                lambda_value=lambda_value,
                atol=atol,
            )
            work.charge(iteration_audit.work)
            residual = float(iteration_audit.diagnostics.kkt_residual)
            if residual < best_residual:
                best_residual = residual
                best_diag = iteration_audit.diagnostics
                best_dual = dual.clone()
                best_source = "refined"
            if residual <= kkt_target:
                break
            plateau_anchor, stalled_iterations, plateaued = (
                _update_refinement_plateau(
                    anchor_residual=plateau_anchor,
                    best_residual=best_residual,
                    mapping_delta=mapping_delta,
                    stalled_iterations=stalled_iterations,
                    atol=atol,
                    dtype=phi.dtype,
                )
            )
            if plateaued:
                break

    if best_source == "incoming":
        status = "input_dual_retained"
    elif torch.any(fused):
        status = "refined_fused_edge_dual"
    else:
        status = "analytic_nonfused_dual"

    return DualRefinementResult(
        dual=best_dual,
        status=status,
        dual_refined=bool(best_source != "incoming"),
        audit=KKTAudit(
            diagnostics=best_diag,
            work=work.total,
            fused_edges=int(torch.sum(fused).item()),
            nonzero_edges=int(torch.sum(active).item()),
        ),
        stationarity_before=float(before_audit.diagnostics.stationarity_residual),
        stationarity_after=float(best_diag.stationarity_residual),
        refinement_iterations=int(refinement_iterations),
    )



def _dense_dual_for_graph(
    certificate: GraphFusionCertificate | None,
    *,
    graph_hash: str,
) -> torch.Tensor | None:
    if not isinstance(certificate, DenseEdgeCertificate):
        return None
    if certificate.graph_hash != str(graph_hash):
        return None
    return certificate.dual


def _audit_certificate(
    *,
    certificate: GraphFusionCertificate | None,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    problem: CertificateProblem,
) -> KKTAudit:
    if isinstance(certificate, CompressedEdgeCertificate):
        return _compressed_graph_fusion_kkt(
            certificate=certificate,
            phi=phi,
            grad_smooth=grad_smooth,
            graph=problem.graph,
            graph_hash=problem.graph_hash,
            lower=problem.lower,
            upper=problem.upper,
            lambda_value=problem.lambda_value,
            atol=problem.atol,
        )
    return graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad_smooth,
        dual_kkt=_dense_dual_for_graph(certificate, graph_hash=problem.graph_hash),
        lower=problem.lower,
        upper=problem.upper,
        edge_u=problem.graph.edge_u,
        edge_v=problem.graph.edge_v,
        edge_w=problem.graph.weight,
        lambda_value=problem.lambda_value,
        atol=problem.atol,
    )


def _refine_certificate(
    *,
    certificate: GraphFusionCertificate | None,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    gradient_scope: SmoothGradientScope,
    problem: CertificateProblem,
    max_iter: int = 96,
    options: CertificateOptions | None = None,
) -> CertificateAttempt:
    if isinstance(certificate, CompressedEdgeCertificate):
        effective_options = options or CertificateOptions(
            max_iter=max(int(max_iter), 1)
        )
        return _refine_compressed_certificate(
            certificate=certificate,
            phi=phi,
            grad_smooth=grad_smooth,
            gradient_scope=gradient_scope,
            graph=problem.graph,
            graph_hash=problem.graph_hash,
            lower=problem.lower,
            upper=problem.upper,
            lambda_value=problem.lambda_value,
            atol=problem.atol,
            options=effective_options,
        )
    dense = refine_graph_fusion_dual_certificate_torch(
        phi=phi,
        grad_smooth=grad_smooth,
        dual_kkt=_dense_dual_for_graph(certificate, graph_hash=problem.graph_hash),
        lower=problem.lower,
        upper=problem.upper,
        edge_u=problem.graph.edge_u,
        edge_v=problem.graph.edge_v,
        edge_w=problem.graph.weight,
        lambda_value=problem.lambda_value,
        atol=problem.atol,
        max_iter=max_iter,
    )
    refinement_iterations = int(dense.refinement_iterations)
    dense_audit = dense.audit
    work = WorkLedger(dense_audit.work)
    work.charge_certificate_work(
        iterations=refinement_iterations,
        # Keep this category as a count of full diagnostic calls; exact
        # physical/logical edge sweeps come from the backend audit.
        full_graph_passes=refinement_iterations + 2,
    )
    refined_certificate = DenseEdgeCertificate(
        dual=dense.dual,
        graph_hash=str(problem.graph_hash),
        gradient_scope=gradient_scope,
    )
    return CertificateAttempt(
        certificate=refined_certificate,
        diagnostics=dense_audit.diagnostics,
        status=dense.status,
        work_counters=work.total,
    )


def certify(
    *,
    problem: CertificateProblem,
    phi: torch.Tensor,
    gradient: CertificateGradient,
    witness: GraphFusionCertificate | None,
    refine: bool,
    max_iter: int = 96,
    options: CertificateOptions | None = None,
) -> CertificateAttempt:
    """Refine and/or audit one full-original-graph certificate.

    This is the authoritative high-level entry point.  Dense and compressed
    representations remain internal choices and always return the same typed
    result surface.
    """

    expected_shape = tuple(problem.lower.shape)
    if tuple(phi.shape) != expected_shape:
        raise ValueError(f"Certificate phi must have shape {expected_shape}.")
    if tuple(gradient.value.shape) != expected_shape:
        raise ValueError(f"Certificate gradient must have shape {expected_shape}.")
    if tuple(gradient.at_breakpoint.shape) != expected_shape:
        raise ValueError(
            f"Certificate breakpoint mask must have shape {expected_shape}."
        )
    if refine:
        return _refine_certificate(
            certificate=witness,
            phi=phi,
            grad_smooth=gradient.value,
            gradient_scope=gradient.scope,
            problem=problem,
            max_iter=max_iter,
            options=options,
        )
    audit = _audit_certificate(
        certificate=witness,
        phi=phi,
        grad_smooth=gradient.value,
        problem=problem,
    )
    work = WorkLedger(audit.work)
    work.charge_certificate_work(full_graph_passes=1)
    return CertificateAttempt(
        certificate=witness,
        diagnostics=audit.diagnostics,
        status="audited",
        work_counters=work.total,
    )
