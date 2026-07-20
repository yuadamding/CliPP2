from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .torch_backend import (
    graph_fusion_kkt_residual_from_grad_torch,
    project_stationarity_cone_torch,
    refine_graph_fusion_dual_certificate_torch,
    stationarity_residual_torch,
)
from .graph_ops import project_dual_ball
from .types import (
    BackendWorkCounters,
    CertificateOptions,
    CompressedEdgeCertificate,
    DenseEdgeCertificate,
    GraphFusionCertificate,
    KKTDiagnostics,
    SmoothGradientScope,
    TensorFusionGraph,
)


# Column generation assumes that the retained-edge subproblem has itself been
# solved.  Repeatedly enlarging an unconverged workset is both non-authoritative
# and can multiply ``max_iter`` by every configured expansion.  Three attempts
# leave room for the added columns to improve conditioning before failing closed
# to the caller's configured dense-fallback policy.
_MAX_CONSECUTIVE_UNCONVERGED_WORKSETS = 3


@dataclass(frozen=True, slots=True)
class CertificateRefinementResult:
    certificate: GraphFusionCertificate | None
    diagnostics: KKTDiagnostics
    status: str
    dual_refined: bool
    fused_edges: int
    nonzero_edges: int
    stationarity_before: float
    stationarity_after: float
    work_counters: BackendWorkCounters = BackendWorkCounters()
    node_residual: float = float("inf")
    column_residual: float = float("inf")


def _analytic_nonfused_adjoint(
    *,
    phi: torch.Tensor,
    labels: torch.Tensor,
    graph: TensorFusionGraph,
    lambda_value: float,
) -> tuple[torch.Tensor, int]:
    adj = torch.zeros_like(phi)
    num_edges = int(graph.edge_u.numel())
    chunk_size = _compressed_edge_chunk_size(
        num_edges=num_edges,
        num_regions=int(phi.shape[1]),
        dtype=phi.dtype,
    )
    edge_passes = 0
    if lambda_value <= 0.0:
        return adj, edge_passes
    for start in range(0, num_edges, chunk_size):
        stop = min(start + chunk_size, num_edges)
        edge_passes += 1
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
    return adj, edge_passes


def _stream_edge_activity_counts(
    *,
    phi: torch.Tensor,
    graph: TensorFusionGraph,
    atol: float,
) -> tuple[int, int, int]:
    """Count fused/nonzero edges without materializing a full edge matrix."""

    num_edges = int(graph.edge_u.numel())
    chunk_size = _compressed_edge_chunk_size(
        num_edges=num_edges,
        num_regions=int(phi.shape[1]),
        dtype=phi.dtype,
    )
    nonzero_edges = 0
    edge_passes = 0
    for start in range(0, num_edges, chunk_size):
        stop = min(start + chunk_size, num_edges)
        edge_passes += 1
        diff = phi.index_select(0, graph.edge_u[start:stop]) - phi.index_select(
            0, graph.edge_v[start:stop]
        )
        nonzero_edges += int(
            torch.sum(torch.linalg.vector_norm(diff, dim=1) > float(atol)).item()
        )
    return num_edges - nonzero_edges, nonzero_edges, edge_passes


def _initial_internal_tree_ids(
    *,
    labels: torch.Tensor,
    graph: TensorFusionGraph,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, int]:
    if labels.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=labels.device), 0
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
            return torch.empty(0, dtype=torch.long, device=labels.device), 0
        child_rank = local_rank.index_select(0, child_positions)
        parent_positions = (
            sorted_starts.index_select(0, child_positions) + (child_rank - 1) // 2
        )
        child_nodes = order.index_select(0, child_positions)
        parent_nodes = order.index_select(0, parent_positions)
        edge_u = torch.minimum(parent_nodes, child_nodes)
        edge_v = torch.maximum(parent_nodes, child_nodes)
        edge_ids = edge_u * (2 * num_nodes - edge_u - 1) // 2 + edge_v - edge_u - 1
        return torch.sort(edge_ids).values, 0

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
    edge_passes = 0
    for start in range(0, num_edges, chunk_size):
        stop = min(start + chunk_size, num_edges)
        edge_passes += 1
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
        return torch.empty(0, dtype=torch.long, device=labels.device), edge_passes
    return torch.cat(selected), edge_passes


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

    frozen = upper <= lower + float(atol)
    lower_active = phi <= lower + float(atol)
    upper_active = phi >= upper - float(atol)
    diagnostic_lower = lower_active & ~upper_active & ~frozen
    diagnostic_upper = upper_active & ~frozen
    interior = ~(lower_active | upper_active | frozen)
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
        projected_stationarity_residual=float("inf"),
        projected_stationarity_norm=float("inf"),
        stationarity_normalizer=1.0 + float(torch.linalg.norm(grad_smooth).item()),
        smooth_gradient_norm=float(torch.linalg.norm(grad_smooth).item()),
        fusion_adjustment_norm=float("inf"),
        edge_subgradient_residual=float("inf"),
        dual_ball_residual=float("inf"),
        box_primal_violation=box_primal_violation,
        num_interior_coordinates=int(torch.sum(interior).item()),
        num_lower_active_coordinates=int(torch.sum(diagnostic_lower).item()),
        num_upper_active_coordinates=int(torch.sum(diagnostic_upper).item()),
        num_frozen_coordinates=int(torch.sum(frozen).item()),
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
) -> tuple[torch.Tensor, torch.Tensor, float, float, int]:
    if edge_ids.numel() == 0:
        residual, adj, _ = _workset_residual(
            base_grad=base_grad,
            dual=dual_start,
            edge_u=graph.edge_u[:0],
            edge_v=graph.edge_v[:0],
            phi=phi,
            lower=lower,
            upper=upper,
        )
        scale = (
            1.0
            + float(torch.linalg.norm(base_grad).item())
            + float(torch.linalg.norm(adj).item())
        )
        return (
            dual_start,
            residual,
            float(torch.linalg.norm(residual).item()) / scale,
            0.0,
            0,
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
        edge_gradient = residual.index_select(0, edge_u) - residual.index_select(
            0, edge_v
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
    scale = (
        1.0
        + float(torch.linalg.norm(base_grad).item())
        + float(torch.linalg.norm(adj).item())
    )
    node_residual = float(torch.linalg.norm(residual).item()) / max(scale, 1e-300)
    return dual, residual, node_residual, mapping_residual, iterations


def _scan_omitted_internal_edges(
    *,
    residual: torch.Tensor,
    labels: torch.Tensor,
    support_ids: torch.Tensor,
    graph: TensorFusionGraph,
    scale: float,
    add_batch: int,
) -> tuple[float, torch.Tensor, int]:
    num_edges = int(graph.edge_u.numel())
    chunk_size = _compressed_edge_chunk_size(
        num_edges=num_edges,
        num_regions=int(residual.shape[1]),
        dtype=residual.dtype,
    )
    best_scores: list[torch.Tensor] = []
    best_ids: list[torch.Tensor] = []
    maximum = 0.0
    edge_passes = 0
    for start in range(0, num_edges, chunk_size):
        stop = min(start + chunk_size, num_edges)
        edge_passes += 1
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
        return (
            maximum,
            torch.empty(0, dtype=torch.long, device=residual.device),
            edge_passes,
        )
    scores = torch.cat(best_scores)
    ids = torch.cat(best_ids)
    count = min(int(add_batch), int(scores.numel()))
    _values, positions = torch.topk(scores, k=count, largest=True, sorted=True)
    return maximum, ids.index_select(0, positions), edge_passes


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
) -> CertificateRefinementResult:
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
        fused_edges, nonzero_edges, activity_passes = _stream_edge_activity_counts(
            phi=phi,
            graph=graph,
            atol=atol,
        )
        diag = _resource_limit_diagnostics(
            phi=phi,
            grad_smooth=grad_smooth,
            lower=lower,
            upper=upper,
            atol=atol,
        )
        return CertificateRefinementResult(
            certificate=certificate,
            diagnostics=diag,
            status="resource_limit",
            dual_refined=False,
            fused_edges=fused_edges,
            nonzero_edges=nonzero_edges,
            stationarity_before=diag.stationarity_residual,
            stationarity_after=diag.stationarity_residual,
            work_counters=BackendWorkCounters(streamed_edge_passes=activity_passes),
        )
    labels, centers, inherited_ids, inherited_dual = _validated_compressed_tensors(
        certificate,
        phi=phi,
        graph=graph,
        graph_hash=graph_hash,
    )
    tree_ids, tree_passes = _initial_internal_tree_ids(
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
        fused_edges, nonzero_edges, activity_passes = _stream_edge_activity_counts(
            phi=phi,
            graph=graph,
            atol=atol,
        )
        diag = _resource_limit_diagnostics(
            phi=phi,
            grad_smooth=grad_smooth,
            lower=lower,
            upper=upper,
            atol=atol,
        )
        return CertificateRefinementResult(
            certificate=certificate,
            diagnostics=diag,
            status="resource_limit",
            dual_refined=False,
            fused_edges=fused_edges,
            nonzero_edges=nonzero_edges,
            stationarity_before=diag.stationarity_residual,
            stationarity_after=diag.stationarity_residual,
            work_counters=BackendWorkCounters(
                streamed_edge_passes=tree_passes + activity_passes
            ),
        )
    support_ids, dual = _merge_internal_support(
        inherited_ids=inherited_ids,
        inherited_dual=inherited_dual,
        added_ids=tree_ids,
        num_regions=int(phi.shape[1]),
        dtype=phi.dtype,
        device=phi.device,
    )
    fused_edges, nonzero_edges, activity_passes = _stream_edge_activity_counts(
        phi=phi,
        graph=graph,
        atol=atol,
    )
    between_adj, between_passes = _analytic_nonfused_adjoint(
        phi=phi,
        labels=labels,
        graph=graph,
        lambda_value=lambda_value,
    )
    base_grad = grad_smooth + between_adj
    total_iterations = 0
    edge_passes = tree_passes + between_passes + activity_passes
    column_residual = float("inf")
    node_residual = float("inf")
    before = _compressed_graph_fusion_kkt(
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
    edge_passes += _compressed_audit_edge_passes(
        num_edges=int(graph.edge_u.numel()),
        num_regions=int(phi.shape[1]),
        dtype=phi.dtype,
    )
    if before.kkt_residual <= 5.0 * float(atol):
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
        return CertificateRefinementResult(
            certificate=certified,
            diagnostics=before,
            status="certified",
            dual_refined=False,
            fused_edges=fused_edges,
            nonzero_edges=nonzero_edges,
            stationarity_before=before.stationarity_residual,
            stationarity_after=before.stationarity_residual,
            work_counters=BackendWorkCounters(streamed_edge_passes=edge_passes),
            node_residual=before.stationarity_residual,
            column_residual=before.edge_subgradient_residual,
        )
    status = "not_certified"
    expansions = 0
    force_rounds = 0
    unconverged_worksets = 0
    final_diag = before
    for _expansion in range(int(options.max_expansions)):
        dual, residual, node_residual, mapping_residual, iterations = (
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
        total_iterations += iterations
        work_adj = torch.zeros_like(phi)
        if support_ids.numel():
            work_u = graph.edge_u.index_select(0, support_ids)
            work_v = graph.edge_v.index_select(0, support_ids)
            work_adj.index_add_(0, work_u, dual)
            work_adj.index_add_(0, work_v, dual, alpha=-1.0)
        scale = (
            1.0
            + float(torch.linalg.norm(grad_smooth).item())
            + float(torch.linalg.norm(between_adj + work_adj).item())
        )
        column_residual, proposed_ids, scan_passes = _scan_omitted_internal_edges(
            residual=residual,
            labels=labels,
            support_ids=support_ids,
            graph=graph,
            scale=scale,
            add_batch=int(options.add_batch),
        )
        edge_passes += scan_passes
        current = CompressedEdgeCertificate(
            labels=labels,
            centers=centers,
            internal_edge_ids=support_ids,
            internal_dual=dual,
            graph_hash=graph_hash,
            gradient_scope=gradient_scope,
        )
        final_diag = _compressed_graph_fusion_kkt(
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
        edge_passes += max(
            1,
            (
                int(graph.edge_u.numel())
                + _compressed_edge_chunk_size(
                    num_edges=int(graph.edge_u.numel()),
                    num_regions=int(phi.shape[1]),
                    dtype=phi.dtype,
                )
                - 1
            )
            // _compressed_edge_chunk_size(
                num_edges=int(graph.edge_u.numel()),
                num_regions=int(phi.shape[1]),
                dtype=phi.dtype,
            ),
        )
        if final_diag.kkt_residual <= 5.0 * float(atol):
            status = "certified"
            certificate = current
            break
        if not math.isfinite(mapping_residual) or mapping_residual > float(
            options.mapping_tolerance
        ):
            unconverged_worksets += 1
        else:
            unconverged_worksets = 0
        if unconverged_worksets >= _MAX_CONSECUTIVE_UNCONVERGED_WORKSETS:
            # The omitted-edge scan cannot certify optimal columns while the
            # included-edge projected-gradient mapping remains unresolved.
            # Return no authority and let the existing caller select dense
            # fallback, CPU fallback, or an error according to policy.
            status = "workset_incomplete"
            certificate = current
            break
        should_expand = column_residual > float(options.column_tolerance)
        if not should_expand and force_rounds < int(options.refinement_rounds):
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
        expansions += 1
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
    return CertificateRefinementResult(
        certificate=certificate,
        diagnostics=final_diag,
        status=status,
        dual_refined=True,
        fused_edges=fused_edges,
        nonzero_edges=nonzero_edges,
        stationarity_before=before.stationarity_residual,
        stationarity_after=final_diag.stationarity_residual,
        work_counters=BackendWorkCounters(
            workset_iterations=total_iterations,
            workset_expansions=expansions,
            streamed_edge_passes=edge_passes,
        ),
        node_residual=node_residual,
        column_residual=column_residual,
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


def _compressed_audit_edge_passes(
    *, num_edges: int, num_regions: int, dtype: torch.dtype
) -> int:
    if int(num_edges) <= 0:
        return 0
    chunk_size = _compressed_edge_chunk_size(
        num_edges=int(num_edges),
        num_regions=int(num_regions),
        dtype=dtype,
    )
    return (int(num_edges) + chunk_size - 1) // chunk_size


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
) -> KKTDiagnostics:
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
    for start in range(0, num_edges, chunk_size):
        stop = min(start + chunk_size, num_edges)
        edge_u = graph.edge_u[start:stop]
        edge_v = graph.edge_v[start:stop]
        diff = phi.index_select(0, edge_u) - phi.index_select(0, edge_v)
        radius = float(lambda_value) * graph.weight[start:stop]
        dual_chunk = torch.zeros_like(diff)
        if lambda_value > 0.0:
            same = labels.index_select(0, edge_u) == labels.index_select(0, edge_v)
            diff_norm = torch.linalg.vector_norm(diff, dim=1)
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
            prox_input = diff + dual_chunk
            prox_input_norm = torch.linalg.vector_norm(prox_input, dim=1)
            big = prox_input_norm >= radius
            safe_norm = prox_input_norm.clamp_min(1e-300)
            active_residual = (
                -dual_chunk + radius[:, None] * prox_input / safe_norm[:, None]
            )
            edge_residual = torch.where(
                big,
                torch.linalg.vector_norm(active_residual, dim=1),
                torch.linalg.vector_norm(diff, dim=1),
            )
            ball_residual = torch.clamp(
                torch.linalg.vector_norm(dual_chunk, dim=1) - radius,
                min=0.0,
            )
            max_edge_residual = max(
                max_edge_residual,
                float(torch.max(edge_residual).item())
                if edge_residual.numel()
                else 0.0,
            )
            max_ball_residual = max(
                max_ball_residual,
                float(torch.max(ball_residual).item())
                if ball_residual.numel()
                else 0.0,
            )
            max_radius = max(
                max_radius,
                float(torch.max(radius).item()) if radius.numel() else 0.0,
            )

    total_grad = grad_smooth + adj
    stat = stationarity_residual_torch(
        total_grad=total_grad,
        phi=phi,
        lower=lower,
        upper=upper,
        atol=atol,
    )
    smooth_gradient_norm = float(torch.linalg.norm(grad_smooth).item())
    fusion_adjustment_norm = float(torch.linalg.norm(adj).item())
    projected_stationarity_norm = float(torch.linalg.norm(stat).item())
    stationarity_normalizer = 1.0 + smooth_gradient_norm + fusion_adjustment_norm
    stationarity_residual = projected_stationarity_norm / max(
        stationarity_normalizer, 1e-300
    )
    frozen = upper <= lower + float(atol)
    lower_active = phi <= lower + float(atol)
    upper_active = phi >= upper - float(atol)
    interior = ~(lower_active | upper_active | frozen)
    diagnostic_lower = lower_active & ~upper_active & ~frozen
    diagnostic_upper = upper_active & ~frozen
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
    box_residual = box_primal_violation / max(box_scale, 1e-300)
    edge_subgradient_residual = max_edge_residual / (1.0 + max_radius)
    dual_ball_residual = max_ball_residual / (1.0 + max_radius)
    return KKTDiagnostics(
        stationarity_residual=float(stationarity_residual),
        projected_stationarity_residual=float(stationarity_residual),
        projected_stationarity_norm=projected_stationarity_norm,
        stationarity_normalizer=float(stationarity_normalizer),
        smooth_gradient_norm=smooth_gradient_norm,
        fusion_adjustment_norm=fusion_adjustment_norm,
        edge_subgradient_residual=float(edge_subgradient_residual),
        dual_ball_residual=float(dual_ball_residual),
        box_primal_violation=box_primal_violation,
        num_interior_coordinates=int(torch.sum(interior).item()),
        num_lower_active_coordinates=int(torch.sum(diagnostic_lower).item()),
        num_upper_active_coordinates=int(torch.sum(diagnostic_upper).item()),
        num_frozen_coordinates=int(torch.sum(frozen).item()),
        box_residual=float(box_residual),
        kkt_residual=max(
            float(stationarity_residual),
            float(edge_subgradient_residual),
            float(dual_ball_residual),
            float(box_residual),
        ),
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


def audit_graph_fusion_certificate(
    *,
    certificate: GraphFusionCertificate | None,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    graph: TensorFusionGraph,
    graph_hash: str,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    atol: float,
) -> KKTDiagnostics:
    """Audit a certificate without imposing a dense representation contract."""

    if isinstance(certificate, CompressedEdgeCertificate):
        return _compressed_graph_fusion_kkt(
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
    values = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad_smooth,
        dual_kkt=_dense_dual_for_graph(certificate, graph_hash=graph_hash),
        lower=lower,
        upper=upper,
        edge_u=graph.edge_u,
        edge_v=graph.edge_v,
        edge_w=graph.weight,
        lambda_value=lambda_value,
        atol=atol,
    )
    return KKTDiagnostics.from_mapping(values)


def refine_graph_fusion_certificate(
    *,
    certificate: GraphFusionCertificate | None,
    phi: torch.Tensor,
    grad_smooth: torch.Tensor,
    gradient_scope: SmoothGradientScope,
    graph: TensorFusionGraph,
    graph_hash: str,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    atol: float,
    max_iter: int = 96,
    options: CertificateOptions | None = None,
) -> CertificateRefinementResult:
    """Refine and audit a certificate for the supplied fixed smooth gradient."""

    if isinstance(certificate, CompressedEdgeCertificate):
        effective_options = options or CertificateOptions(
            max_iter=max(int(max_iter), 1)
        )
        return _refine_compressed_certificate(
            certificate=certificate,
            phi=phi,
            grad_smooth=grad_smooth,
            gradient_scope=gradient_scope,
            graph=graph,
            graph_hash=graph_hash,
            lower=lower,
            upper=upper,
            lambda_value=lambda_value,
            atol=atol,
            options=effective_options,
        )
    dense = refine_graph_fusion_dual_certificate_torch(
        phi=phi,
        grad_smooth=grad_smooth,
        dual_kkt=_dense_dual_for_graph(certificate, graph_hash=graph_hash),
        lower=lower,
        upper=upper,
        edge_u=graph.edge_u,
        edge_v=graph.edge_v,
        edge_w=graph.weight,
        lambda_value=lambda_value,
        atol=atol,
        max_iter=max_iter,
    )
    dual = dense["dual"]
    refined_certificate = (
        DenseEdgeCertificate(
            dual=dual,
            graph_hash=str(graph_hash),
            gradient_scope=gradient_scope,
        )
        if torch.is_tensor(dual)
        else None
    )
    return CertificateRefinementResult(
        certificate=refined_certificate,
        diagnostics=KKTDiagnostics.from_mapping(dense["diag"]),
        status=str(dense["status"]),
        dual_refined=bool(dense["dual_refined"]),
        fused_edges=int(dense["fused_edges"]),
        nonzero_edges=int(dense["nonzero_edges"]),
        stationarity_before=float(dense["stationarity_before"]),
        stationarity_after=float(dense["stationarity_after"]),
    )
