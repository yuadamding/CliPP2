from __future__ import annotations

from dataclasses import dataclass

import torch

from .certificates import (
    _compressed_audit_edge_passes,
    audit_graph_fusion_certificate,
    refine_graph_fusion_certificate,
)
from .graph_ops import estimate_dense_complete_solver_peak_bytes
from .torch_backend import solve_majorized_subproblem_alm_torch
from .types import (
    BackendWorkCounters,
    CertificateOptions,
    CompressedEdgeCertificate,
    InnerSolveResult,
    KKTDiagnostics,
    PrimalOnlyWarmState,
    QuotientWorksetWarmState,
    SmoothGradientScope,
    TensorFusionGraph,
    TorchRuntime,
)


_QUOTIENT_EDGE_WORK_BYTES = 64 * 1024 * 1024


class QuotientCacheResourceError(MemoryError):
    """The exact quotient cache cannot fit its configured memory budget."""


@dataclass(frozen=True, slots=True)
class ExactQuotientProblem:
    labels: torch.Tensor
    h: torch.Tensor
    center: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    edge_u: torch.Tensor
    edge_v: torch.Tensor
    edge_w: torch.Tensor
    node_to_block_weight: torch.Tensor
    quadratic_constant: torch.Tensor
    common_box_feasible: bool

    @property
    def num_blocks(self) -> int:
        return int(self.h.shape[0])


def estimate_exact_quotient_cache_bytes(
    *,
    num_nodes: int,
    num_blocks: int,
    num_regions: int,
    value_dtype: torch.dtype,
    index_dtype: torch.dtype = torch.long,
    edge_chunk_size: int = 1,
) -> int:
    """Conservatively estimate persistent quotient state plus bounded edge work."""

    m = max(int(num_nodes), 0)
    k = max(int(num_blocks), 0)
    s = max(int(num_regions), 1)
    q = k * max(k - 1, 0) // 2
    value_bytes = int(torch.empty((), dtype=value_dtype).element_size())
    index_bytes = int(torch.empty((), dtype=index_dtype).element_size())
    persistent = (
        6 * k * s * value_bytes
        + k * k * value_bytes
        + m * k * value_bytes
        # triu construction, filtered endpoint copies, masks, and filtered
        # weights can overlap before the final 2-by-Q/value arrays remain.
        + 5 * q * index_bytes
        + 3 * q * value_bytes
        + 8 * q
        + 4 * m * index_bytes
    )
    # The exact quotient is itself solved by complete-graph ADMM.  Its scaled
    # multiplier and split state are Q-by-S, and small problems use the dense
    # loop whose temporary peak is larger still.  These arrays coexist with the
    # aggregated quotient cache, so include the backend peak rather than merely
    # the scalar quotient-edge arrays above.
    quotient_solver_peak = estimate_dense_complete_solver_peak_bytes(
        k,
        num_regions=s,
        dtype=value_dtype,
        include_dual=True,
        include_split=True,
        include_graph=False,
    )
    # Block labels/endpoints, masks, selected endpoints, and weights can coexist.
    work_per_edge = 10 * index_bytes + 3 * value_bytes + 8
    return int(
        persistent + quotient_solver_peak + max(int(edge_chunk_size), 1) * work_per_edge
    )


def canonicalize_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.to(dtype=torch.long)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional.")
    if labels.numel() == 0:
        return labels
    if bool(torch.any(labels < 0).item()):
        raise ValueError("labels must be nonnegative.")
    unique = torch.unique(labels, sorted=True)
    first = torch.full(
        (int(unique.numel()),),
        int(labels.numel()),
        dtype=torch.long,
        device=labels.device,
    )
    positions = torch.searchsorted(unique, labels)
    node_ids = torch.arange(int(labels.numel()), device=labels.device)
    first.scatter_reduce_(0, positions, node_ids, reduce="amin", include_self=True)
    order = torch.argsort(first)
    remap = torch.empty_like(order)
    remap[order] = torch.arange(int(order.numel()), device=labels.device)
    return remap.index_select(0, positions)


def candidate_partition_from_phi(
    phi: torch.Tensor, *, tolerance: float
) -> torch.Tensor:
    """Return a deterministic heuristic equality proposal from a primal point."""

    if phi.ndim != 2:
        raise ValueError("phi must have shape (M, S).")
    if phi.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=phi.device)
    tol = float(tolerance)
    if tol < 0.0:
        raise ValueError("partition tolerance must be nonnegative.")
    keys = phi if tol == 0.0 else torch.round(phi / tol)
    _unique, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    return canonicalize_labels(inverse)


def compressed_certificate_for_primal(
    phi: torch.Tensor,
    *,
    graph_hash: str,
    gradient_scope: SmoothGradientScope,
) -> CompressedEdgeCertificate:
    """Create an empty-support exact equality representation for ``phi``."""

    labels = candidate_partition_from_phi(phi, tolerance=0.0)
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
    centers = phi.index_select(0, roots)
    return CompressedEdgeCertificate(
        labels=labels,
        centers=centers,
        internal_edge_ids=torch.empty(0, dtype=torch.long, device=phi.device),
        internal_dual=torch.empty(
            (0, int(phi.shape[1])), dtype=phi.dtype, device=phi.device
        ),
        graph_hash=str(graph_hash),
        gradient_scope=gradient_scope,
    )


def aggregate_exact_quotient_problem(
    *,
    h: torch.Tensor,
    U: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    labels: torch.Tensor,
    graph: TensorFusionGraph,
    max_cache_bytes: int | None = None,
) -> ExactQuotientProblem:
    """Aggregate the restricted quadratic-fusion problem without approximation."""

    if not (h.shape == U.shape == lower.shape == upper.shape):
        raise ValueError("h, U, lower, and upper must have matching shapes.")
    labels = canonicalize_labels(labels.to(device=h.device))
    if int(labels.numel()) != int(h.shape[0]):
        raise ValueError("labels must contain one entry per primal row.")
    if bool(torch.any(h <= 0.0).item()):
        raise ValueError("Exact quotient aggregation requires positive curvature.")
    num_blocks = int(torch.max(labels).item()) + 1 if labels.numel() else 0
    num_regions = int(h.shape[1])
    num_edges = int(graph.edge_u.numel())
    minimum_bytes = estimate_exact_quotient_cache_bytes(
        num_nodes=int(h.shape[0]),
        num_blocks=num_blocks,
        num_regions=num_regions,
        value_dtype=h.dtype,
        index_dtype=graph.edge_u.dtype,
        edge_chunk_size=1,
    )
    if max_cache_bytes is not None and minimum_bytes > int(max_cache_bytes):
        raise QuotientCacheResourceError(
            "exact_quotient_cache_resource_limit: the minimum exact quotient "
            f"cache needs approximately {minimum_bytes} bytes, exceeding the "
            f"configured limit of {int(max_cache_bytes)} bytes."
        )
    value_bytes = int(torch.empty((), dtype=h.dtype).element_size())
    index_bytes = int(torch.empty((), dtype=graph.edge_u.dtype).element_size())
    work_per_edge = 10 * index_bytes + 3 * value_bytes + 8
    if max_cache_bytes is None:
        edge_chunk_size = max(1, _QUOTIENT_EDGE_WORK_BYTES // work_per_edge)
    else:
        persistent_bytes = minimum_bytes - work_per_edge
        edge_chunk_size = max(
            1,
            (int(max_cache_bytes) - persistent_bytes) // work_per_edge,
        )
    edge_chunk_size = max(1, min(num_edges, edge_chunk_size))
    h_bar = torch.zeros((num_blocks, num_regions), dtype=h.dtype, device=h.device)
    h_center = torch.zeros_like(h_bar)
    h_bar.index_add_(0, labels, h)
    h_center.index_add_(0, labels, h * U)
    center = h_center / h_bar.clamp_min(torch.finfo(h.dtype).tiny)

    expanded_labels = labels[:, None].expand(-1, num_regions)
    lower_bar = torch.full_like(h_bar, -torch.inf)
    upper_bar = torch.full_like(h_bar, torch.inf)
    lower_bar.scatter_reduce_(
        0, expanded_labels, lower, reduce="amax", include_self=True
    )
    upper_bar.scatter_reduce_(
        0, expanded_labels, upper, reduce="amin", include_self=True
    )
    common_box_feasible = bool(torch.all(lower_bar <= upper_bar).item())

    weight_matrix = torch.zeros(
        (num_blocks, num_blocks), dtype=graph.weight.dtype, device=h.device
    )
    node_to_block = torch.zeros(
        (int(h.shape[0]), num_blocks), dtype=graph.weight.dtype, device=h.device
    )
    for start in range(0, num_edges, edge_chunk_size):
        stop = min(start + edge_chunk_size, num_edges)
        edge_u = graph.edge_u[start:stop]
        edge_v = graph.edge_v[start:stop]
        block_u = labels.index_select(0, edge_u)
        block_v = labels.index_select(0, edge_v)
        between_index = torch.nonzero(block_u != block_v, as_tuple=False).flatten()
        if int(between_index.numel()) == 0:
            continue
        between_u = edge_u.index_select(0, between_index)
        between_v = edge_v.index_select(0, between_index)
        between_block_u = block_u.index_select(0, between_index)
        between_block_v = block_v.index_select(0, between_index)
        between_weight = graph.weight[start:stop].index_select(0, between_index)
        lo = torch.minimum(between_block_u, between_block_v)
        hi = torch.maximum(between_block_u, between_block_v)
        quotient_bins = lo * num_blocks + hi
        weight_matrix.view(-1).scatter_add_(0, quotient_bins, between_weight)
        node_to_block.index_put_(
            (between_u, between_block_v), between_weight, accumulate=True
        )
        node_to_block.index_put_(
            (between_v, between_block_u), between_weight, accumulate=True
        )
    quotient_edge_u, quotient_edge_v = torch.triu_indices(
        num_blocks, num_blocks, offset=1, device=h.device
    )
    quotient_edge_w = weight_matrix[quotient_edge_u, quotient_edge_v]
    positive = quotient_edge_w > 0.0
    quotient_edge_u = quotient_edge_u[positive]
    quotient_edge_v = quotient_edge_v[positive]
    quotient_edge_w = quotient_edge_w[positive]

    quadratic_constant = 0.5 * torch.sum(h * torch.square(U)) - 0.5 * torch.sum(
        h_bar * torch.square(center)
    )
    return ExactQuotientProblem(
        labels=labels,
        h=h_bar,
        center=center,
        lower=lower_bar,
        upper=upper_bar,
        edge_u=quotient_edge_u,
        edge_v=quotient_edge_v,
        edge_w=quotient_edge_w,
        node_to_block_weight=node_to_block,
        quadratic_constant=quadratic_constant,
        common_box_feasible=common_box_feasible,
    )


def lifted_inner_objective(
    *,
    phi: torch.Tensor,
    h: torch.Tensor,
    U: torch.Tensor,
    graph: TensorFusionGraph,
    lambda_value: float,
) -> torch.Tensor:
    quadratic = 0.5 * torch.sum(h * torch.square(phi - U))
    diff = phi.index_select(0, graph.edge_u) - phi.index_select(0, graph.edge_v)
    penalty = float(lambda_value) * torch.sum(
        graph.weight * torch.linalg.vector_norm(diff, dim=1)
    )
    return quadratic + penalty


def quotient_inner_objective(
    *,
    centers: torch.Tensor,
    problem: ExactQuotientProblem,
    lambda_value: float,
) -> torch.Tensor:
    quadratic = 0.5 * torch.sum(problem.h * torch.square(centers - problem.center))
    diff = centers.index_select(0, problem.edge_u) - centers.index_select(
        0, problem.edge_v
    )
    penalty = float(lambda_value) * torch.sum(
        problem.edge_w * torch.linalg.vector_norm(diff, dim=1)
    )
    return quadratic + penalty + problem.quadratic_constant


def _coalesced_labels(
    *, labels: torch.Tensor, centers: torch.Tensor, tolerance: float
) -> torch.Tensor:
    center_labels = candidate_partition_from_phi(centers, tolerance=tolerance)
    return canonicalize_labels(center_labels.index_select(0, labels))


def heuristic_residual_split(
    *,
    labels: torch.Tensor,
    residual_signal: torch.Tensor,
) -> torch.Tensor | None:
    """Propose one deterministic singleton split; certification remains authoritative."""

    labels = canonicalize_labels(labels)
    if residual_signal.ndim != 2 or int(residual_signal.shape[0]) != int(
        labels.numel()
    ):
        raise ValueError("residual_signal must have one row per structure label.")
    best_node: int | None = None
    best_score = float("-inf")
    num_blocks = int(torch.max(labels).item()) + 1 if labels.numel() else 0
    for block in range(num_blocks):
        members = torch.nonzero(labels == block, as_tuple=False).flatten()
        if int(members.numel()) <= 1:
            continue
        values = residual_signal.index_select(0, members)
        center = torch.mean(values, dim=0)
        scores = torch.linalg.vector_norm(values - center, dim=1)
        local_position = int(torch.argmax(scores).item())
        node = int(members[local_position].item())
        score = float(scores[local_position].item())
        if score > best_score or (
            score == best_score and (best_node is None or node < best_node)
        ):
            best_score = score
            best_node = node
    if best_node is None:
        return None
    proposed = labels.clone()
    proposed[best_node] = num_blocks
    return canonicalize_labels(proposed)


def solve_majorized_subproblem_quotient_workset_torch(
    *,
    runtime: TorchRuntime,
    U: torch.Tensor,
    h: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    lambda_value: float,
    graph: TensorFusionGraph,
    graph_hash: str,
    tol: float,
    max_iter: int,
    phi_start: torch.Tensor,
    warm_state: QuotientWorksetWarmState | PrimalOnlyWarmState | None,
    certificate_options: CertificateOptions,
    partition_tolerance: float,
    max_structural_rounds: int = 4,
) -> tuple[InnerSolveResult | None, BackendWorkCounters]:
    """Attempt an exact quotient/workset solve and report all attempted work.

    A ``None`` result requests the exact dense fallback.  The counters are
    returned separately so that an unsuccessful quotient/workset attempt is not
    erased from diagnostic provenance when that fallback succeeds.
    """

    quotient_iterations = 0
    workset_iterations = 0
    workset_expansions = 0
    streamed_edge_passes = 0

    def attempted_work() -> BackendWorkCounters:
        return BackendWorkCounters(
            quotient_iterations=int(quotient_iterations),
            workset_iterations=int(workset_iterations),
            workset_expansions=int(workset_expansions),
            streamed_edge_passes=int(streamed_edge_passes),
        )

    if not graph.is_complete or lambda_value <= 0.0:
        return None, attempted_work()
    quotient_warm_state = (
        warm_state if isinstance(warm_state, QuotientWorksetWarmState) else None
    )
    warm_compatible = bool(
        quotient_warm_state is not None
        and quotient_warm_state.graph_hash == str(graph_hash)
    )
    if warm_compatible and quotient_warm_state is not None:
        labels = canonicalize_labels(
            quotient_warm_state.labels.to(device=runtime.device, dtype=torch.long)
        )
    elif (
        isinstance(warm_state, PrimalOnlyWarmState)
        and warm_state.structure_hint_is_heuristic
        and warm_state.structure_hint is not None
        and int(warm_state.structure_hint.numel()) == int(phi_start.shape[0])
    ):
        labels = canonicalize_labels(
            warm_state.structure_hint.to(device=runtime.device, dtype=torch.long)
        )
    else:
        labels = candidate_partition_from_phi(
            phi_start.to(device=runtime.device, dtype=runtime.dtype),
            tolerance=partition_tolerance,
        )
    num_nodes = int(phi_start.shape[0])
    if labels.numel() != num_nodes:
        return None, attempted_work()
    quotient_dual_start = (
        quotient_warm_state.quotient_dual.to(device=runtime.device, dtype=runtime.dtype)
        if (
            warm_compatible
            and quotient_warm_state is not None
            and quotient_warm_state.quotient_dual is not None
        )
        else None
    )
    inherited_certificate = None
    if (
        warm_compatible
        and quotient_warm_state is not None
        and torch.equal(labels, quotient_warm_state.labels.to(device=labels.device))
    ):
        inherited_certificate = CompressedEdgeCertificate(
            labels=labels,
            centers=quotient_warm_state.centers.to(
                device=runtime.device, dtype=runtime.dtype
            ),
            internal_edge_ids=quotient_warm_state.internal_edge_ids.to(
                device=runtime.device, dtype=torch.long
            ),
            internal_dual=quotient_warm_state.internal_dual.to(
                device=runtime.device, dtype=runtime.dtype
            ),
            graph_hash=graph_hash,
            gradient_scope="mm_surrogate",
        )
    if inherited_certificate is not None:
        phi_warm = phi_start.to(device=runtime.device, dtype=runtime.dtype)
        lifted_warm = inherited_certificate.centers.index_select(0, labels)
        if tuple(lifted_warm.shape) == tuple(phi_warm.shape) and torch.equal(
            lifted_warm, phi_warm
        ):
            inherited_kkt = audit_graph_fusion_certificate(
                certificate=inherited_certificate,
                phi=phi_warm,
                grad_smooth=h * (phi_warm - U),
                graph=graph,
                graph_hash=graph_hash,
                lower=lower,
                upper=upper,
                lambda_value=lambda_value,
                atol=tol,
            )
            streamed_edge_passes += _compressed_audit_edge_passes(
                num_edges=int(graph.edge_u.numel()),
                num_regions=int(phi_warm.shape[1]),
                dtype=phi_warm.dtype,
            )
            if inherited_kkt.kkt_residual <= 5.0 * float(tol):
                counters = attempted_work()
                return InnerSolveResult(
                    phi=phi_warm,
                    backend_name="quotient_workset_complete_graph",
                    warm_state=QuotientWorksetWarmState(
                        phi=phi_warm,
                        labels=labels,
                        centers=inherited_certificate.centers,
                        quotient_dual=quotient_dual_start,
                        internal_edge_ids=inherited_certificate.internal_edge_ids,
                        internal_dual=inherited_certificate.internal_dual,
                        graph_hash=graph_hash,
                        previous_lambda=float(lambda_value),
                    ),
                    surrogate_certificate=inherited_certificate,
                    surrogate_kkt=inherited_kkt,
                    converged=True,
                    inner_iterations=0,
                    backend_iterations=0,
                    work_counters=counters,
                ), counters
    allow_heuristic_split = bool(
        certificate_options.memory.allow_heuristic_split_before_dense_fallback
    )
    for _structural_round in range(max(int(max_structural_rounds), 1)):
        num_blocks = int(torch.max(labels).item()) + 1 if labels.numel() else 0
        if num_blocks >= num_nodes:
            return None, attempted_work()
        try:
            quotient = aggregate_exact_quotient_problem(
                h=h,
                U=U,
                lower=lower,
                upper=upper,
                labels=labels,
                graph=graph,
                max_cache_bytes=certificate_options.memory.max_compressed_cache_bytes,
            )
        except QuotientCacheResourceError:
            return None, attempted_work()
        if not quotient.common_box_feasible:
            if not allow_heuristic_split:
                return None, attempted_work()
            split_labels = heuristic_residual_split(
                labels=labels,
                residual_signal=h * (phi_start - U),
            )
            if split_labels is None:
                return None, attempted_work()
            labels = split_labels
            quotient_dual_start = None
            inherited_certificate = None
            continue
        expected_edges = num_blocks * max(num_blocks - 1, 0) // 2
        if int(quotient.edge_u.numel()) != expected_edges:
            return None, attempted_work()
        center_start = quotient.center
        if (
            warm_compatible
            and quotient_warm_state is not None
            and torch.equal(labels, quotient_warm_state.labels.to(device=labels.device))
            and tuple(quotient_warm_state.centers.shape) == tuple(center_start.shape)
        ):
            center_start = quotient_warm_state.centers.to(
                device=runtime.device, dtype=runtime.dtype
            )
        expected_dual_shape = (expected_edges, int(U.shape[1]))
        if quotient_dual_start is not None and tuple(quotient_dual_start.shape) != (
            expected_dual_shape
        ):
            quotient_dual_start = None
        (
            centers,
            _scaled_dual,
            quotient_dual,
            iterations,
            _quotient_converged,
            _quotient_residual,
        ) = solve_majorized_subproblem_alm_torch(
            runtime=runtime,
            num_mutations=num_blocks,
            U=quotient.center,
            h=quotient.h,
            lower=quotient.lower,
            upper=quotient.upper,
            lambda_value=lambda_value,
            edge_u=quotient.edge_u,
            edge_v=quotient.edge_v,
            edge_w=quotient.edge_w,
            tol=tol,
            max_iter=max(max_iter, 10),
            phi_start=center_start,
            dual_start=quotient_dual_start,
            dual_start_is_actual=True,
        )
        quotient_iterations += int(iterations)
        merged_labels = _coalesced_labels(
            labels=labels,
            centers=centers,
            tolerance=max(float(tol), 1e-12),
        )
        if not torch.equal(merged_labels, labels):
            labels = merged_labels
            quotient_dual_start = None
            inherited_certificate = None
            continue
        phi = centers.index_select(0, labels)
        inherited_edge_ids = (
            inherited_certificate.internal_edge_ids
            if inherited_certificate is not None
            else torch.empty(0, dtype=torch.long, device=runtime.device)
        )
        inherited_internal_dual = (
            inherited_certificate.internal_dual
            if inherited_certificate is not None
            else torch.empty(
                (0, int(U.shape[1])), dtype=runtime.dtype, device=runtime.device
            )
        )
        inherited_certificate = CompressedEdgeCertificate(
            labels=labels,
            centers=centers,
            internal_edge_ids=inherited_edge_ids,
            internal_dual=inherited_internal_dual,
            graph_hash=graph_hash,
            gradient_scope="mm_surrogate",
        )
        refinement = refine_graph_fusion_certificate(
            certificate=inherited_certificate,
            phi=phi,
            grad_smooth=h * (phi - U),
            gradient_scope="mm_surrogate",
            graph=graph,
            graph_hash=graph_hash,
            lower=lower,
            upper=upper,
            lambda_value=lambda_value,
            atol=tol,
            options=certificate_options,
        )
        work = refinement.work_counters
        workset_iterations += int(work.workset_iterations)
        workset_expansions += int(work.workset_expansions)
        streamed_edge_passes += int(work.streamed_edge_passes)
        if refinement.status == "not_certified" and allow_heuristic_split:
            split_labels = heuristic_residual_split(
                labels=labels,
                residual_signal=h * (phi - U),
            )
            if split_labels is not None:
                labels = split_labels
                quotient_dual_start = None
                inherited_certificate = None
                continue
        if refinement.status != "certified" or not isinstance(
            refinement.certificate, CompressedEdgeCertificate
        ):
            return None, attempted_work()
        certificate = refinement.certificate
        counters = attempted_work()
        total_iterations = quotient_iterations + workset_iterations
        return InnerSolveResult(
            phi=phi,
            backend_name="quotient_workset_complete_graph",
            warm_state=QuotientWorksetWarmState(
                phi=phi,
                labels=labels,
                centers=centers,
                quotient_dual=quotient_dual,
                internal_edge_ids=certificate.internal_edge_ids,
                internal_dual=certificate.internal_dual,
                graph_hash=graph_hash,
                previous_lambda=float(lambda_value),
            ),
            surrogate_certificate=certificate,
            surrogate_kkt=KKTDiagnostics.from_mapping(refinement.diagnostics.as_dict()),
            converged=True,
            inner_iterations=total_iterations,
            backend_iterations=total_iterations,
            work_counters=counters,
        ), counters
    return None, attempted_work()
