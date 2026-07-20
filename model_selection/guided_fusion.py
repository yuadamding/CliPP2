"""KKT-guided initialization of a pairwise-fusion fit from a partition.

The selected likelihood partition supplies a useful primal start and may pilot
finite, noise-regularized adaptive weights. This module constructs a matching
*actual* edge dual and obtains a nonzero fusion scale from the capacity required
to hold the guide blocks together. It deliberately does not construct a lambda
grid or apply a fixed lambda multiplier.

The closed-form within-block flow below uses the complete graph identity

    q_ij = (a_j - a_i) / n_k,

where ``a`` is the part of the smooth gradient that must be balanced by fusion
inside guide block ``k``.  Consequently, the helper requires a complete graph.
That is the graph used by CliPP2's default adaptive pairwise-fusion fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from ..core.fusion.graph_ops import (
    dense_complete_solver_memory_preflight,
    graph_adjoint_edges,
    project_dual_ball,
)
from ..core.fusion.solver import torch_data_from_context
from ..core.fusion.torch_backend import (
    graph_fusion_kkt_residual_from_grad_torch,
    mutation_region_terms_torch,
)
from ..core.fusion.types import (
    ExactSolverResourceLimit,
    QuotientWorksetWarmState,
    SolverContext,
    SolverState,
)


GuideArray = np.ndarray | torch.Tensor
GuideLabels = Sequence[object] | np.ndarray | torch.Tensor
_GUIDED_FUSION_WORK_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class GuidedFusionDiagnostics:
    """Diagnostics for :func:`build_guided_fusion_initialization`.

    ``required_lambda_without_between_edges`` is the raw, data-derived
    within-block capacity at zero between-block dual. ``lambda_value`` can be
    larger only when the analytically required between-block subgradients add
    further within-block capacity demand. If that fixed-point capacity does not
    exist for heterogeneous adaptive weights, ``capacity_converged`` is false,
    the zero-between block capacity is retained as an initial *scale*, and the
    actual dual is projected into the valid balls. ``kkt_residual`` is allowed
    to be nonzero: the guide is a warm start, not a selected estimator.
    """

    lambda_value: float
    required_lambda_without_between_edges: float
    numerical_lambda_floor: float
    capacity_iterations: int
    capacity_converged: bool
    capacity_status: str
    num_mutations: int
    num_regions: int
    num_clusters: int
    within_edge_count: int
    between_edge_count: int
    zero_separation_between_edge_count: int
    gradient_source: str
    guide_adjustment_max_abs: float
    max_within_cluster_deviation: float
    block_flow_balance_max_abs: float
    max_dual_ball_ratio: float
    max_within_dual_ball_ratio: float
    max_between_dual_ball_ratio: float
    kkt_residual: float
    stationarity_residual: float
    edge_subgradient_residual: float
    dual_ball_residual: float
    box_residual: float
    num_exact_lower_active_coordinates: int
    num_exact_upper_active_coordinates: int
    num_exact_frozen_coordinates: int


@dataclass(frozen=True, slots=True)
class GuidedFusionInitialization:
    """A data-derived lambda and solver-compatible primal/actual-dual state."""

    lambda_value: float
    solver_state: SolverState
    diagnostics: GuidedFusionDiagnostics


def _canonical_labels(labels: GuideLabels, *, num_mutations: int) -> np.ndarray:
    if torch.is_tensor(labels):
        raw = labels.detach().cpu().numpy()
    else:
        raw = np.asarray(labels)
    if raw.ndim != 1 or int(raw.shape[0]) != int(num_mutations):
        raise ValueError(f"guide_labels must have shape ({num_mutations},).")

    canonical = np.empty(int(num_mutations), dtype=np.int64)
    seen: dict[object, int] = {}
    for index, value in enumerate(raw.tolist()):
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("guide_labels must not contain NaN or infinity.")
        try:
            label = seen.get(value)
        except TypeError as exc:
            raise ValueError("guide_labels entries must be hashable.") from exc
        if label is None:
            label = len(seen)
            seen[value] = label
        canonical[index] = int(label)
    if not seen:
        raise ValueError("guide_labels must contain at least one cluster.")
    return canonical


def _validate_complete_graph(context: SolverContext, *, num_mutations: int) -> None:
    graph = context.graph
    expected_edges = int(num_mutations) * max(int(num_mutations) - 1, 0) // 2
    if not bool(graph.is_complete) or int(graph.weight.numel()) != expected_edges:
        raise ValueError(
            "Guided KKT initialization currently requires the complete pairwise-fusion graph."
        )
    if int(graph.num_nodes) != int(num_mutations):
        raise ValueError("SolverContext graph size does not match guide_phi.")
    if not bool(torch.all(torch.isfinite(graph.weight)).item()) or bool(
        torch.any(graph.weight <= 0.0).item()
    ):
        raise ValueError("Complete graph weights must be finite and strictly positive.")


def _canonical_guide_phi(
    guide_phi: GuideArray,
    *,
    labels: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    partition_tolerance: float,
) -> tuple[torch.Tensor, float, float]:
    phi_input = torch.as_tensor(guide_phi, dtype=lower.dtype, device=lower.device)
    if tuple(phi_input.shape) != tuple(lower.shape):
        raise ValueError(f"guide_phi must have shape {tuple(lower.shape)}.")
    if not bool(torch.all(torch.isfinite(phi_input)).item()):
        raise ValueError("guide_phi must contain only finite values.")

    lower_violation = torch.clamp(lower - phi_input, min=0.0)
    upper_violation = torch.clamp(phi_input - upper, min=0.0)
    max_box_violation = float(
        torch.max(torch.maximum(lower_violation, upper_violation)).item()
    )
    if max_box_violation > float(partition_tolerance):
        raise ValueError("guide_phi lies outside the SolverContext box constraints.")
    phi = torch.minimum(torch.maximum(phi_input, lower), upper).clone()

    max_deviation = 0.0
    for cluster in range(int(torch.max(labels).item()) + 1):
        members = torch.nonzero(labels == int(cluster), as_tuple=False).flatten()
        block = phi.index_select(0, members)
        common_lower = torch.max(lower.index_select(0, members), dim=0).values
        common_upper = torch.min(upper.index_select(0, members), dim=0).values
        if bool(torch.any(common_lower > common_upper).item()):
            raise ValueError("A guide block has no common box-feasible center.")
        center = torch.mean(block, dim=0)
        center = torch.minimum(torch.maximum(center, common_lower), common_upper)
        deviation = float(torch.max(torch.abs(block - center)).item())
        scale = 1.0 + float(torch.max(torch.abs(block)).item())
        if deviation > float(partition_tolerance) * scale:
            raise ValueError(
                "guide_phi rows assigned to one cluster must share a center within "
                "partition_tolerance."
            )
        max_deviation = max(max_deviation, deviation)
        phi.index_copy_(0, members, center.expand(int(members.numel()), -1))

    adjustment = float(torch.max(torch.abs(phi - phi_input)).item())
    return phi, adjustment, max_deviation


def _box_stationarity_targets(
    total_grad: torch.Tensor,
    *,
    phi: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Choose block-sum-preserving stationarity targets under box constraints.

    Exact equality is intentional here. A point merely near a boundary is still
    interior for the projected KKT condition and must not absorb a normal-cone
    gradient. Tiny out-of-box guide errors have already been clipped above.
    """

    lower_active = phi == lower
    upper_active = phi == upper
    frozen = lower_active & upper_active
    lower_only = lower_active & ~upper_active
    upper_only = upper_active & ~lower_active
    interior = ~(lower_active | upper_active)

    target = torch.zeros_like(total_grad)
    target = torch.where(frozen, total_grad, target)
    target = torch.where(lower_only, torch.clamp(total_grad, min=0.0), target)
    target = torch.where(upper_only, torch.clamp(total_grad, max=0.0), target)

    num_nodes, num_regions = total_grad.shape
    for region in range(int(num_regions)):
        desired_sum = torch.sum(total_grad[:, region])
        delta = desired_sum - torch.sum(target[:, region])
        delta_value = float(delta.item())
        if delta_value == 0.0:
            continue

        frozen_idx = torch.nonzero(frozen[:, region], as_tuple=False).flatten()
        lower_idx = torch.nonzero(lower_only[:, region], as_tuple=False).flatten()
        upper_idx = torch.nonzero(upper_only[:, region], as_tuple=False).flatten()
        interior_idx = torch.nonzero(interior[:, region], as_tuple=False).flatten()

        # Frozen coordinates can absorb either sign. Lower-active coordinates
        # can absorb an unlimited positive residual; upper-active coordinates
        # can absorb an unlimited negative residual.
        unlimited = frozen_idx
        if int(unlimited.numel()) == 0:
            unlimited = lower_idx if delta_value > 0.0 else upper_idx
        if int(unlimited.numel()) > 0:
            target[unlimited, region] += delta / float(unlimited.numel())
            continue

        # Before violating a box normal cone, use any finite room back toward
        # zero on the opposite active boundary.
        limited = upper_idx if delta_value > 0.0 else lower_idx
        if int(limited.numel()) > 0:
            current = target[limited, region]
            room = torch.abs(current)
            total_room = float(torch.sum(room).item())
            if total_room > 0.0:
                used = min(abs(delta_value), total_room)
                direction = 1.0 if delta_value > 0.0 else -1.0
                target[limited, region] += (
                    direction * float(used) * room / float(total_room)
                )
                delta = desired_sum - torch.sum(target[:, region])
                delta_value = float(delta.item())
                if delta_value == 0.0:
                    continue

        # The guide block is not box-stationary in aggregate. Preserve flow
        # balance and expose the unavoidable projected residual in diagnostics.
        violating = torch.cat([interior_idx, lower_idx, upper_idx])
        if int(violating.numel()) == 0:
            violating = torch.arange(int(num_nodes), device=total_grad.device)
        target[violating, region] += delta / float(violating.numel())

        # Remove the final floating-point sum error so the complete-graph flow
        # has zero demand exactly to working precision.
        final_error = desired_sum - torch.sum(target[:, region])
        target[violating[0], region] += final_error

    return target


def _assemble_actual_dual(
    *,
    lambda_value: float,
    phi: torch.Tensor,
    labels: torch.Tensor,
    grad_smooth: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    _work_memory_bytes: int = _GUIDED_FUSION_WORK_BYTES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_mutations, num_regions = phi.shape
    num_edges = int(edge_u.numel())
    dual = torch.zeros(
        (num_edges, int(num_regions)), dtype=phi.dtype, device=phi.device
    )

    # Keep the historical vectorized path bit-for-bit unchanged when its
    # edge-by-region work fits comfortably in the bounded workspace.  The
    # chunked path below is needed for Sim1K's largest complete graph: at
    # M=4,261 and S=20, one avoidable float64 E-by-S temporary is ~1.45 GB.
    # Six value-sized work vectors plus index/mask overhead is a conservative
    # upper bound for each chunk's live tensors.
    value_bytes_per_edge = 6 * int(num_regions) * int(phi.element_size()) + 4 * int(
        edge_u.element_size()
    )
    chunk_edges = max(
        1,
        min(
            num_edges,
            max(int(_work_memory_bytes), 1) // max(value_bytes_per_edge, 1),
        ),
    )
    if num_edges <= chunk_edges:
        same_block = labels.index_select(0, edge_u) == labels.index_select(0, edge_v)
        between = ~same_block

        if bool(torch.any(between).item()) and float(lambda_value) > 0.0:
            diff = phi.index_select(0, edge_u) - phi.index_select(0, edge_v)
            diff_norm = torch.linalg.vector_norm(diff, dim=1)
            active_between = between & (diff_norm > 0.0)
            if bool(torch.any(active_between).item()):
                radius = float(lambda_value) * edge_w[active_between]
                dual[active_between] = (
                    radius[:, None]
                    * diff[active_between]
                    / diff_norm[active_between, None]
                )

        adjusted_grad = grad_smooth + graph_adjoint_edges(
            dual,
            edge_u=edge_u,
            edge_v=edge_v,
            num_nodes=int(num_mutations),
        )
        flow_demand, stationarity_target, cluster_sizes = _complete_block_flow_terms(
            adjusted_grad,
            phi=phi,
            labels=labels,
            lower=lower,
            upper=upper,
        )

        if bool(torch.any(same_block).item()):
            block_size = cluster_sizes.index_select(
                0, labels.index_select(0, edge_u[same_block])
            )
            dual[same_block] = (
                flow_demand.index_select(0, edge_v[same_block])
                - flow_demand.index_select(0, edge_u[same_block])
            ) / block_size[:, None].to(dtype=phi.dtype)

        return dual, flow_demand, stationarity_target

    if float(lambda_value) > 0.0:
        for start in range(0, num_edges, chunk_edges):
            stop = min(start + chunk_edges, num_edges)
            chunk_u = edge_u[start:stop]
            chunk_v = edge_v[start:stop]
            between_index = torch.nonzero(
                labels.index_select(0, chunk_u) != labels.index_select(0, chunk_v),
                as_tuple=False,
            ).flatten()
            if int(between_index.numel()) == 0:
                continue
            between_u = chunk_u.index_select(0, between_index)
            between_v = chunk_v.index_select(0, between_index)
            diff = phi.index_select(0, between_u) - phi.index_select(0, between_v)
            diff_norm = torch.linalg.vector_norm(diff, dim=1)
            active_index = torch.nonzero(diff_norm > 0.0, as_tuple=False).flatten()
            if int(active_index.numel()) == 0:
                continue
            active_diff = diff.index_select(0, active_index)
            active_radius = float(lambda_value) * edge_w[start:stop].index_select(
                0, between_index
            ).index_select(0, active_index)
            active_diff.mul_(active_radius[:, None])
            active_diff.div_(diff_norm.index_select(0, active_index)[:, None])
            dual[start:stop].index_copy_(
                0,
                between_index.index_select(0, active_index),
                active_diff,
            )

    # Preserve the original adjoint summation order (all u contributions,
    # then all v contributions) while bounding the otherwise full-size
    # ``-dual`` temporary created by graph_adjoint_edges.
    dual_adjoint = torch.zeros_like(grad_smooth)
    for start in range(0, num_edges, chunk_edges):
        stop = min(start + chunk_edges, num_edges)
        dual_adjoint.index_add_(0, edge_u[start:stop], dual[start:stop])
    for start in range(0, num_edges, chunk_edges):
        stop = min(start + chunk_edges, num_edges)
        dual_adjoint.index_add_(0, edge_v[start:stop], -dual[start:stop])
    adjusted_grad = grad_smooth + dual_adjoint
    flow_demand, stationarity_target, cluster_sizes = _complete_block_flow_terms(
        adjusted_grad,
        phi=phi,
        labels=labels,
        lower=lower,
        upper=upper,
    )

    for start in range(0, num_edges, chunk_edges):
        stop = min(start + chunk_edges, num_edges)
        chunk_u = edge_u[start:stop]
        chunk_v = edge_v[start:stop]
        within_index = torch.nonzero(
            labels.index_select(0, chunk_u) == labels.index_select(0, chunk_v),
            as_tuple=False,
        ).flatten()
        if int(within_index.numel()) == 0:
            continue
        within_u = chunk_u.index_select(0, within_index)
        within_v = chunk_v.index_select(0, within_index)
        block_size = cluster_sizes.index_select(0, labels.index_select(0, within_u))
        within_dual = (
            flow_demand.index_select(0, within_v)
            - flow_demand.index_select(0, within_u)
        ) / block_size[:, None].to(dtype=phi.dtype)
        dual[start:stop].index_copy_(0, within_index, within_dual)

    return dual, flow_demand, stationarity_target


def _required_within_lambda(
    dual: torch.Tensor,
    *,
    within: torch.Tensor,
    edge_w: torch.Tensor,
) -> float:
    if not bool(torch.any(within).item()):
        return 0.0
    work_bytes = int(dual.numel()) * int(dual.element_size())
    if work_bytes <= _GUIDED_FUSION_WORK_BYTES:
        ratios = torch.linalg.vector_norm(dual[within], dim=1) / edge_w[within]
        return float(torch.max(ratios).item())
    chunk_edges = max(
        1,
        _GUIDED_FUSION_WORK_BYTES
        // max(3 * int(dual.shape[1]) * int(dual.element_size()), 1),
    )
    maximum = 0.0
    for start in range(0, int(dual.shape[0]), chunk_edges):
        stop = min(start + chunk_edges, int(dual.shape[0]))
        mask = within[start:stop]
        if not bool(torch.any(mask).item()):
            continue
        ratios = (
            torch.linalg.vector_norm(dual[start:stop][mask], dim=1)
            / edge_w[start:stop][mask]
        )
        maximum = max(maximum, float(torch.max(ratios).item()))
    return maximum


def _max_dual_ratio(
    dual: torch.Tensor,
    *,
    mask: torch.Tensor,
    radius: torch.Tensor,
) -> float:
    if not bool(torch.any(mask).item()):
        return 0.0
    work_bytes = int(dual.numel()) * int(dual.element_size())
    if work_bytes <= _GUIDED_FUSION_WORK_BYTES:
        ratio = torch.linalg.vector_norm(dual[mask], dim=1) / radius[mask]
        return float(torch.max(ratio).item())
    chunk_edges = max(
        1,
        _GUIDED_FUSION_WORK_BYTES
        // max(3 * int(dual.shape[1]) * int(dual.element_size()), 1),
    )
    maximum = 0.0
    for start in range(0, int(dual.shape[0]), chunk_edges):
        stop = min(start + chunk_edges, int(dual.shape[0]))
        chunk_mask = mask[start:stop]
        if not bool(torch.any(chunk_mask).item()):
            continue
        ratio = (
            torch.linalg.vector_norm(dual[start:stop][chunk_mask], dim=1)
            / radius[start:stop][chunk_mask]
        )
        maximum = max(maximum, float(torch.max(ratio).item()))
    return maximum


def _project_dual_ball_bounded(
    dual: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    work_bytes = int(dual.numel()) * int(dual.element_size())
    if work_bytes <= _GUIDED_FUSION_WORK_BYTES:
        return project_dual_ball(dual, radius)
    chunk_edges = max(
        1,
        _GUIDED_FUSION_WORK_BYTES
        // max(3 * int(dual.shape[1]) * int(dual.element_size()), 1),
    )
    for start in range(0, int(dual.shape[0]), chunk_edges):
        stop = min(start + chunk_edges, int(dual.shape[0]))
        chunk = dual[start:stop]
        norm = torch.linalg.vector_norm(chunk, dim=1, keepdim=True)
        scale = torch.maximum(
            torch.ones_like(norm),
            norm / radius[start:stop, None].clamp_min(torch.finfo(dual.dtype).tiny),
        )
        chunk.div_(scale)
    return dual


def _zero_separation_between_count(
    phi: torch.Tensor,
    *,
    between: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
) -> int:
    num_edges = int(edge_u.numel())
    work_bytes = num_edges * int(phi.shape[1]) * int(phi.element_size())
    if work_bytes <= _GUIDED_FUSION_WORK_BYTES:
        diff = phi.index_select(0, edge_u) - phi.index_select(0, edge_v)
        return int(
            torch.sum(between & (torch.linalg.vector_norm(diff, dim=1) == 0.0)).item()
        )
    chunk_edges = max(
        1,
        _GUIDED_FUSION_WORK_BYTES
        // max(3 * int(phi.shape[1]) * int(phi.element_size()), 1),
    )
    count = 0
    for start in range(0, num_edges, chunk_edges):
        stop = min(start + chunk_edges, num_edges)
        diff = phi.index_select(0, edge_u[start:stop]) - phi.index_select(
            0, edge_v[start:stop]
        )
        count += int(
            torch.sum(
                between[start:stop] & (torch.linalg.vector_norm(diff, dim=1) == 0.0)
            ).item()
        )
    return count


def _guided_edge_stream_chunk_size(
    *,
    num_edges: int,
    num_regions: int,
    dtype: torch.dtype,
    index_dtype: torch.dtype,
) -> int:
    """Bound edge-region work without ever selecting the full edge set."""

    if int(num_edges) <= 1:
        return 1
    value_bytes = int(torch.empty((), dtype=dtype).element_size())
    index_bytes = int(torch.empty((), dtype=index_dtype).element_size())
    bytes_per_edge = max(8 * int(num_regions) * value_bytes + 4 * index_bytes, 1)
    budget_chunk = max(1, int(_GUIDED_FUSION_WORK_BYTES) // bytes_per_edge)
    return min(int(num_edges) - 1, budget_chunk)


def _edge_stream_slices(num_edges: int, chunk_edges: int):
    for start in range(0, int(num_edges), max(int(chunk_edges), 1)):
        yield slice(start, min(start + int(chunk_edges), int(num_edges)))


def _analytic_between_dual_chunk(
    *,
    lambda_value: float,
    phi: torch.Tensor,
    labels: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
) -> torch.Tensor:
    """Materialize only one bounded chunk of analytic nonfused duals."""

    diff = phi.index_select(0, edge_u) - phi.index_select(0, edge_v)
    dual = torch.zeros_like(diff)
    between = labels.index_select(0, edge_u) != labels.index_select(0, edge_v)
    if float(lambda_value) <= 0.0 or not bool(torch.any(between).item()):
        return dual
    diff_norm = torch.linalg.vector_norm(diff, dim=1)
    active = between & (diff_norm > 0.0)
    if bool(torch.any(active).item()):
        radius = float(lambda_value) * edge_w[active]
        dual[active] = radius[:, None] * diff[active] / diff_norm[active, None]
    return dual


def _stream_analytic_between_adjoint(
    *,
    lambda_value: float,
    phi: torch.Tensor,
    labels: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    chunk_edges: int,
) -> torch.Tensor:
    """Accumulate the analytic inter-block adjoint with bounded edge work."""

    adjoint = torch.zeros_like(phi)
    num_edges = int(edge_u.numel())
    # Match graph_adjoint_edges' summation order: all u contributions, then all v.
    for endpoints, alpha in ((edge_u, 1.0), (edge_v, -1.0)):
        for edge_slice in _edge_stream_slices(num_edges, chunk_edges):
            dual_chunk = _analytic_between_dual_chunk(
                lambda_value=lambda_value,
                phi=phi,
                labels=labels,
                edge_u=edge_u[edge_slice],
                edge_v=edge_v[edge_slice],
                edge_w=edge_w[edge_slice],
            )
            adjoint.index_add_(0, endpoints[edge_slice], dual_chunk, alpha=alpha)
    return adjoint


def _complete_block_flow_terms(
    adjusted_grad: torch.Tensor,
    *,
    phi: torch.Tensor,
    labels: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return complete-block flow demand, targets, and block sizes."""

    flow_demand = torch.zeros_like(adjusted_grad)
    stationarity_target = torch.zeros_like(adjusted_grad)
    num_clusters = int(torch.max(labels).item()) + 1
    cluster_sizes = torch.bincount(labels, minlength=num_clusters)
    for cluster in range(num_clusters):
        members = torch.nonzero(labels == int(cluster), as_tuple=False).flatten()
        block_target = _box_stationarity_targets(
            adjusted_grad.index_select(0, members),
            phi=phi.index_select(0, members),
            lower=lower.index_select(0, members),
            upper=upper.index_select(0, members),
        )
        stationarity_target.index_copy_(0, members, block_target)
        flow_demand.index_copy_(
            0,
            members,
            adjusted_grad.index_select(0, members) - block_target,
        )
    return flow_demand, stationarity_target, cluster_sizes


def _implicit_guided_flow(
    *,
    lambda_value: float,
    phi: torch.Tensor,
    labels: torch.Tensor,
    grad_smooth: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    chunk_edges: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    between_adjoint = _stream_analytic_between_adjoint(
        lambda_value=lambda_value,
        phi=phi,
        labels=labels,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        chunk_edges=chunk_edges,
    )
    adjusted_grad = grad_smooth + between_adjoint
    return _complete_block_flow_terms(
        adjusted_grad,
        phi=phi,
        labels=labels,
        lower=lower,
        upper=upper,
    )


def _required_within_lambda_implicit(
    flow_demand: torch.Tensor,
    *,
    labels: torch.Tensor,
    cluster_sizes: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    chunk_edges: int,
) -> float:
    maximum = 0.0
    num_edges = int(edge_u.numel())
    for edge_slice in _edge_stream_slices(num_edges, chunk_edges):
        chunk_u = edge_u[edge_slice]
        chunk_v = edge_v[edge_slice]
        within = labels.index_select(0, chunk_u) == labels.index_select(0, chunk_v)
        within_index = torch.nonzero(within, as_tuple=False).flatten()
        if int(within_index.numel()) == 0:
            continue
        within_u = chunk_u.index_select(0, within_index)
        within_v = chunk_v.index_select(0, within_index)
        block_size = cluster_sizes.index_select(0, labels.index_select(0, within_u)).to(
            dtype=flow_demand.dtype
        )
        dual = (
            flow_demand.index_select(0, within_v)
            - flow_demand.index_select(0, within_u)
        ) / block_size[:, None]
        ratios = torch.linalg.vector_norm(dual, dim=1) / edge_w[
            edge_slice
        ].index_select(0, within_index)
        maximum = max(maximum, float(torch.max(ratios).item()))
    return maximum


def _implicit_guided_dual_chunk(
    *,
    lambda_value: float,
    phi: torch.Tensor,
    labels: torch.Tensor,
    flow_demand: torch.Tensor,
    cluster_sizes: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
) -> torch.Tensor:
    dual = _analytic_between_dual_chunk(
        lambda_value=lambda_value,
        phi=phi,
        labels=labels,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
    )
    within = labels.index_select(0, edge_u) == labels.index_select(0, edge_v)
    within_index = torch.nonzero(within, as_tuple=False).flatten()
    if int(within_index.numel()) > 0:
        within_u = edge_u.index_select(0, within_index)
        within_v = edge_v.index_select(0, within_index)
        block_size = cluster_sizes.index_select(0, labels.index_select(0, within_u)).to(
            dtype=phi.dtype
        )
        within_dual = (
            flow_demand.index_select(0, within_v)
            - flow_demand.index_select(0, within_u)
        ) / block_size[:, None]
        dual.index_copy_(0, within_index, within_dual)
    radius = float(lambda_value) * edge_w
    return project_dual_ball(dual, radius)


def _implicit_guided_tree_support(
    *,
    lambda_value: float,
    labels: torch.Tensor,
    flow_demand: torch.Tensor,
    cluster_sizes: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build bounded-degree tree flows for the implicit complete-block demand."""

    num_nodes = int(labels.numel())
    if num_nodes == 0:
        return (
            torch.empty(0, dtype=torch.long, device=labels.device),
            torch.empty(
                (0, int(flow_demand.shape[1])),
                dtype=flow_demand.dtype,
                device=flow_demand.device,
            ),
        )

    # Complete tensor graphs use canonical torch.triu_indices ordering. Within
    # each label block, heap-ordering the ascending node IDs gives a
    # deterministic balanced binary tree with maximum degree three.
    nodes = torch.arange(num_nodes, dtype=torch.long, device=labels.device)
    order = torch.argsort(labels, stable=True)
    sorted_labels = labels.index_select(0, order)
    block_starts = torch.cumsum(cluster_sizes, dim=0) - cluster_sizes
    sorted_starts = block_starts.index_select(0, sorted_labels)
    local_rank = nodes - sorted_starts
    child_positions = nodes[local_rank > 0]
    if int(child_positions.numel()) == 0:
        return (
            torch.empty(0, dtype=torch.long, device=labels.device),
            torch.empty(
                (0, int(flow_demand.shape[1])),
                dtype=flow_demand.dtype,
                device=flow_demand.device,
            ),
        )

    child_rank = local_rank.index_select(0, child_positions)
    parent_positions = (
        sorted_starts.index_select(0, child_positions) + (child_rank - 1) // 2
    )
    child_nodes = order.index_select(0, child_positions)
    parent_nodes = order.index_select(0, parent_positions)
    support_u = torch.minimum(parent_nodes, child_nodes)
    support_v = torch.maximum(parent_nodes, child_nodes)
    tree_edge_ids = (
        support_u * (2 * num_nodes - support_u - 1) // 2 + support_v - support_u - 1
    )

    # Fail locally if this internal helper is ever used with a noncanonical
    # graph despite the public complete-graph precondition.
    if not torch.equal(
        edge_u.index_select(0, tree_edge_ids), support_u
    ) or not torch.equal(edge_v.index_select(0, tree_edge_ids), support_v):
        raise ValueError(
            "Balanced guided-tree support requires canonical complete edges."
        )

    # Aggregate demands from leaves to roots one tree level at a time. For an
    # edge from parent to child, its unprojected vector flow is the child's
    # subtree demand; orient it to the canonical (min_node, max_node) edge.
    subtree_demand = flow_demand.clone()
    largest_block = int(torch.max(cluster_sizes).item())
    max_depth = max(largest_block.bit_length() - 1, 0)
    for depth in range(max_depth, 0, -1):
        first_rank = (1 << depth) - 1
        last_rank = (1 << (depth + 1)) - 2
        at_depth = (child_rank >= first_rank) & (child_rank <= last_rank)
        depth_positions = torch.nonzero(at_depth, as_tuple=False).flatten()
        if int(depth_positions.numel()) == 0:
            continue
        depth_children = child_nodes.index_select(0, depth_positions)
        depth_parents = parent_nodes.index_select(0, depth_positions)
        subtree_demand.index_add_(
            0,
            depth_parents,
            subtree_demand.index_select(0, depth_children),
        )

    support_dual = subtree_demand.index_select(0, child_nodes)
    orientation = torch.where(
        parent_nodes == support_u,
        torch.ones_like(parent_nodes),
        -torch.ones_like(parent_nodes),
    ).to(dtype=flow_demand.dtype)
    support_dual = support_dual * orientation[:, None]

    permutation = torch.argsort(tree_edge_ids)
    tree_edge_ids = tree_edge_ids.index_select(0, permutation)
    support_dual = support_dual.index_select(0, permutation)
    radius = float(lambda_value) * edge_w.index_select(0, tree_edge_ids)
    return tree_edge_ids, project_dual_ball(support_dual, radius)


@dataclass(frozen=True, slots=True)
class _ImplicitGuidedAudit:
    diagnostics: dict[str, float | int]
    max_dual_ratio: float
    max_within_dual_ratio: float
    max_between_dual_ratio: float
    zero_separation_between_edge_count: int
    within_edge_count: int
    between_edge_count: int


def _audit_implicit_guided_dual(
    *,
    lambda_value: float,
    phi: torch.Tensor,
    labels: torch.Tensor,
    flow_demand: torch.Tensor,
    cluster_sizes: torch.Tensor,
    grad_smooth: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    atol: float,
    chunk_edges: int,
) -> _ImplicitGuidedAudit:
    """Reproduce full-graph diagnostics while reconstructing each dual chunk."""

    adjoint = torch.zeros_like(phi)
    max_edge_residual = 0.0
    max_ball_residual = 0.0
    max_radius = 0.0
    max_dual_ratio = 0.0
    max_within_ratio = 0.0
    max_between_ratio = 0.0
    zero_between = 0
    within_edge_count = 0
    between_edge_count = 0
    num_edges = int(edge_u.numel())

    for edge_slice in _edge_stream_slices(num_edges, chunk_edges):
        chunk_u = edge_u[edge_slice]
        chunk_v = edge_v[edge_slice]
        chunk_w = edge_w[edge_slice]
        dual = _implicit_guided_dual_chunk(
            lambda_value=lambda_value,
            phi=phi,
            labels=labels,
            flow_demand=flow_demand,
            cluster_sizes=cluster_sizes,
            edge_u=chunk_u,
            edge_v=chunk_v,
            edge_w=chunk_w,
        )
        adjoint.index_add_(0, chunk_u, dual)
        adjoint.index_add_(0, chunk_v, dual, alpha=-1.0)

        diff = phi.index_select(0, chunk_u) - phi.index_select(0, chunk_v)
        diff_norm = torch.linalg.vector_norm(diff, dim=1)
        radius = float(lambda_value) * chunk_w
        dual_norm = torch.linalg.vector_norm(dual, dim=1)
        ratio = dual_norm / radius
        within = labels.index_select(0, chunk_u) == labels.index_select(0, chunk_v)
        between = ~within
        within_edge_count += int(torch.sum(within).item())
        between_edge_count += int(torch.sum(between).item())
        max_dual_ratio = max(max_dual_ratio, float(torch.max(ratio).item()))
        if bool(torch.any(within).item()):
            max_within_ratio = max(
                max_within_ratio, float(torch.max(ratio[within]).item())
            )
        if bool(torch.any(between).item()):
            max_between_ratio = max(
                max_between_ratio, float(torch.max(ratio[between]).item())
            )
            zero_between += int(torch.sum(between & (diff_norm == 0.0)).item())

        prox_input = diff + dual
        prox_input_norm = torch.linalg.vector_norm(prox_input, dim=1)
        big = prox_input_norm >= radius
        active_residual = (
            -dual
            + radius[:, None] * prox_input / prox_input_norm.clamp_min(1e-300)[:, None]
        )
        edge_residual = torch.where(
            big,
            torch.linalg.vector_norm(active_residual, dim=1),
            diff_norm,
        )
        ball_residual = torch.clamp(dual_norm - radius, min=0.0)
        max_edge_residual = max(
            max_edge_residual, float(torch.max(edge_residual).item())
        )
        max_ball_residual = max(
            max_ball_residual, float(torch.max(ball_residual).item())
        )
        max_radius = max(max_radius, float(torch.max(radius).item()))

    total_grad = grad_smooth + adjoint
    projected = torch.minimum(torch.maximum(phi - total_grad, lower), upper)
    stationarity = phi - projected
    smooth_gradient_norm = float(torch.linalg.norm(grad_smooth).item())
    fusion_adjustment_norm = float(torch.linalg.norm(adjoint).item())
    projected_stationarity_norm = float(torch.linalg.norm(stationarity).item())
    stationarity_normalizer = 1.0 + smooth_gradient_norm + fusion_adjustment_norm
    stationarity_residual = projected_stationarity_norm / max(
        stationarity_normalizer, 1e-300
    )

    lower_active = phi <= lower + float(atol)
    upper_active = phi >= upper - float(atol)
    frozen = upper <= lower + float(atol)
    interior = ~(lower_active | upper_active | frozen)
    diagnostic_upper_active = upper_active & ~frozen
    diagnostic_lower_active = lower_active & ~upper_active & ~frozen
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
    diagnostics: dict[str, float | int] = {
        "stationarity_residual": float(stationarity_residual),
        "projected_stationarity_residual": float(stationarity_residual),
        "projected_stationarity_norm": float(projected_stationarity_norm),
        "stationarity_normalizer": float(stationarity_normalizer),
        "smooth_gradient_norm": float(smooth_gradient_norm),
        "fusion_adjustment_norm": float(fusion_adjustment_norm),
        "edge_subgradient_residual": float(edge_subgradient_residual),
        "dual_ball_residual": float(dual_ball_residual),
        "box_primal_violation": float(box_primal_violation),
        "num_interior_coordinates": int(torch.sum(interior).item()),
        "num_lower_active_coordinates": int(torch.sum(diagnostic_lower_active).item()),
        "num_upper_active_coordinates": int(torch.sum(diagnostic_upper_active).item()),
        "num_frozen_coordinates": int(torch.sum(frozen).item()),
        "box_residual": float(box_residual),
        "kkt_residual": max(
            float(stationarity_residual),
            float(edge_subgradient_residual),
            float(dual_ball_residual),
            float(box_residual),
        ),
    }
    return _ImplicitGuidedAudit(
        diagnostics=diagnostics,
        max_dual_ratio=float(max_dual_ratio),
        max_within_dual_ratio=float(max_within_ratio),
        max_between_dual_ratio=float(max_between_ratio),
        zero_separation_between_edge_count=int(zero_between),
        within_edge_count=int(within_edge_count),
        between_edge_count=int(between_edge_count),
    )


def _build_compressed_guided_initialization(
    *,
    phi: torch.Tensor,
    labels: torch.Tensor,
    grad: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    edge_w: torch.Tensor,
    solver_context: SolverContext,
    guide_adjustment: float,
    max_deviation: float,
    gradient_source: str,
    kkt_atol: float,
    max_capacity_iterations: int,
) -> GuidedFusionInitialization:
    """Build the guided state using only node and bounded edge-chunk tensors."""

    num_mutations, num_regions = int(phi.shape[0]), int(phi.shape[1])
    num_edges = int(edge_u.numel())
    chunk_edges = _guided_edge_stream_chunk_size(
        num_edges=num_edges,
        num_regions=num_regions,
        dtype=phi.dtype,
        index_dtype=edge_u.dtype,
    )
    with torch.no_grad():
        zero_flow, _, cluster_sizes = _implicit_guided_flow(
            lambda_value=0.0,
            phi=phi,
            labels=labels,
            grad_smooth=grad,
            lower=lower,
            upper=upper,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            chunk_edges=chunk_edges,
        )
        required_without_between = _required_within_lambda_implicit(
            zero_flow,
            labels=labels,
            cluster_sizes=cluster_sizes,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            chunk_edges=chunk_edges,
        )

        finfo = torch.finfo(phi.dtype)
        gradient_scale = max(
            1.0,
            float(torch.max(torch.linalg.vector_norm(grad, dim=1)).item()),
        )
        weight_scale = (
            float(torch.max(edge_w).item()) if int(edge_w.numel()) > 0 else 1.0
        )
        numerical_floor = float(
            finfo.eps * gradient_scale / max(weight_scale, finfo.tiny)
        )
        if not np.isfinite(numerical_floor) or numerical_floor <= 0.0:
            numerical_floor = float(finfo.eps)
        lambda_value = max(float(required_without_between), numerical_floor)

        capacity_tolerance = 64.0 * float(finfo.eps)
        flow_demand = torch.zeros_like(phi)
        capacity_iterations = 0
        capacity_converged = False
        for capacity_iterations in range(1, int(max_capacity_iterations) + 1):
            flow_demand, _, cluster_sizes = _implicit_guided_flow(
                lambda_value=lambda_value,
                phi=phi,
                labels=labels,
                grad_smooth=grad,
                lower=lower,
                upper=upper,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                chunk_edges=chunk_edges,
            )
            required = _required_within_lambda_implicit(
                flow_demand,
                labels=labels,
                cluster_sizes=cluster_sizes,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                chunk_edges=chunk_edges,
            )
            if required <= lambda_value * (1.0 + capacity_tolerance):
                capacity_converged = True
                break
            next_lambda = float(np.nextafter(required, np.inf))
            if not np.isfinite(next_lambda) or next_lambda <= lambda_value:
                break
            lambda_value = next_lambda

        if not capacity_converged:
            lambda_value = max(float(required_without_between), numerical_floor)
            flow_demand, _, cluster_sizes = _implicit_guided_flow(
                lambda_value=lambda_value,
                phi=phi,
                labels=labels,
                grad_smooth=grad,
                lower=lower,
                upper=upper,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
                chunk_edges=chunk_edges,
            )

        if not np.isfinite(lambda_value) or lambda_value <= 0.0:
            raise RuntimeError(
                "Guided-fusion lambda must be finite and strictly positive."
            )

        implicit_audit = _audit_implicit_guided_dual(
            lambda_value=lambda_value,
            phi=phi,
            labels=labels,
            flow_demand=flow_demand,
            cluster_sizes=cluster_sizes,
            grad_smooth=grad,
            lower=lower,
            upper=upper,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            atol=kkt_atol,
            chunk_edges=chunk_edges,
        )
        audit = implicit_audit.diagnostics

        flow_balance = torch.zeros_like(phi)
        num_clusters = int(torch.max(labels).item()) + 1
        centers = torch.empty(
            (num_clusters, num_regions), dtype=phi.dtype, device=phi.device
        )
        for cluster in range(num_clusters):
            members = torch.nonzero(labels == int(cluster), as_tuple=False).flatten()
            block_sum = torch.sum(flow_demand.index_select(0, members), dim=0)
            flow_balance.index_copy_(
                0, members[:1], block_sum.reshape(1, int(num_regions))
            )
            centers[cluster] = phi[members[0]]
        block_flow_balance = float(torch.max(torch.abs(flow_balance)).item())

        lower_active = phi == lower
        upper_active = phi == upper
        frozen = lower_active & upper_active
        diagnostics = GuidedFusionDiagnostics(
            lambda_value=float(lambda_value),
            required_lambda_without_between_edges=float(required_without_between),
            numerical_lambda_floor=float(numerical_floor),
            capacity_iterations=int(capacity_iterations),
            capacity_converged=bool(capacity_converged),
            capacity_status=(
                "exact_dual_capacity"
                if capacity_converged
                else "projected_zero_between_capacity_scale"
            ),
            num_mutations=int(num_mutations),
            num_regions=int(num_regions),
            num_clusters=int(num_clusters),
            within_edge_count=int(implicit_audit.within_edge_count),
            between_edge_count=int(implicit_audit.between_edge_count),
            zero_separation_between_edge_count=int(
                implicit_audit.zero_separation_between_edge_count
            ),
            gradient_source=str(gradient_source),
            guide_adjustment_max_abs=float(guide_adjustment),
            max_within_cluster_deviation=float(max_deviation),
            block_flow_balance_max_abs=float(block_flow_balance),
            max_dual_ball_ratio=float(implicit_audit.max_dual_ratio),
            max_within_dual_ball_ratio=float(implicit_audit.max_within_dual_ratio),
            max_between_dual_ball_ratio=float(implicit_audit.max_between_dual_ratio),
            kkt_residual=float(audit["kkt_residual"]),
            stationarity_residual=float(audit["stationarity_residual"]),
            edge_subgradient_residual=float(audit["edge_subgradient_residual"]),
            dual_ball_residual=float(audit["dual_ball_residual"]),
            box_residual=float(audit["box_residual"]),
            num_exact_lower_active_coordinates=int(
                torch.sum(lower_active & ~upper_active).item()
            ),
            num_exact_upper_active_coordinates=int(
                torch.sum(upper_active & ~lower_active).item()
            ),
            num_exact_frozen_coordinates=int(torch.sum(frozen).item()),
        )

    tree_edge_ids, tree_dual = _implicit_guided_tree_support(
        lambda_value=lambda_value,
        labels=labels,
        flow_demand=flow_demand,
        cluster_sizes=cluster_sizes,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
    )
    phi_state = phi.detach()
    labels_state = labels.detach()
    centers_state = centers.detach()
    warm_state = QuotientWorksetWarmState(
        phi=phi_state,
        labels=labels_state,
        centers=centers_state,
        quotient_dual=None,
        internal_edge_ids=tree_edge_ids.detach(),
        internal_dual=tree_dual.detach(),
        graph_hash=str(solver_context.graph_hash),
        previous_lambda=float(lambda_value),
    )
    state = SolverState(
        phi=phi_state,
        dual=None,
        previous_lambda=float(lambda_value),
        warm_state=warm_state,
        certificate=None,
    )
    return GuidedFusionInitialization(
        lambda_value=float(lambda_value), solver_state=state, diagnostics=diagnostics
    )


def build_guided_fusion_initialization(
    guide_phi: GuideArray,
    guide_labels: GuideLabels,
    *,
    solver_context: SolverContext,
    grad_smooth: GuideArray | None = None,
    partition_tolerance: float = 1e-8,
    kkt_atol: float = 1e-8,
    max_capacity_iterations: int = 64,
    materialize_dense_dual: bool = True,
) -> GuidedFusionInitialization:
    """Construct a KKT-capacity lambda and warm state from a guide partition.

    Parameters
    ----------
    guide_phi, guide_labels
        Partition-refitted mutation-by-region prevalences and one label per
        mutation. Rows sharing a label are canonicalized to one exactly fused
        center after validation.
    solver_context
        The context for the final fusion fit. Its graph must be complete. The
        adaptive weights may be initialized by ``guide_phi``, but should use a
        finite likelihood-noise floor. Guided mode distributes that floor over
        the complete-graph degree so the proposal informs, but cannot make its
        within-block weights singular.
    grad_smooth
        Optional precomputed smooth observed-loss gradient, mainly useful for
        deterministic tests. When omitted it is evaluated from the context.
    materialize_dense_dual
        Preserve the historical dense actual-dual state when true. When false,
        reconstruct edge duals only in bounded chunks and return a compressed
        quotient/workset warm state with no persistent edge-by-region tensor.

    Notes
    -----
    Box normals are used only at coordinates exactly equal to a bound. This is
    consistent with the solver's projected stationarity residual: a coordinate
    merely near a bound remains interior. The returned dual is in actual KKT
    units, which is the representation expected by ``SolverState.dual``.

    When ``diagnostics.capacity_converged`` is true, the returned lambda
    certifies dual-ball capacity for the proposed within-guide-block flow. For
    a heterogeneous adaptive graph that cannot support that exact guide flow at
    any lambda, the function instead returns the zero-between block capacity as
    a data-derived search scale and projects the dual. In both cases the guide
    is only a warm start; consult ``kkt_residual`` and run ADMM before selection.
    """

    if not np.isfinite(float(partition_tolerance)) or float(partition_tolerance) < 0.0:
        raise ValueError("partition_tolerance must be finite and nonnegative.")
    if not np.isfinite(float(kkt_atol)) or float(kkt_atol) <= 0.0:
        raise ValueError("kkt_atol must be finite and strictly positive.")
    if int(max_capacity_iterations) <= 0:
        raise ValueError("max_capacity_iterations must be positive.")

    lower = solver_context.lower
    upper = solver_context.upper
    if lower.ndim != 2 or tuple(upper.shape) != tuple(lower.shape):
        raise ValueError(
            "SolverContext bounds must be matching two-dimensional tensors."
        )
    num_mutations, num_regions = (int(lower.shape[0]), int(lower.shape[1]))
    if num_mutations <= 0 or num_regions <= 0:
        raise ValueError("Guided fusion requires at least one mutation and one region.")
    _validate_complete_graph(solver_context, num_mutations=num_mutations)

    labels_np = _canonical_labels(guide_labels, num_mutations=num_mutations)
    labels = torch.as_tensor(labels_np, dtype=torch.long, device=lower.device)
    phi, guide_adjustment, max_deviation = _canonical_guide_phi(
        guide_phi,
        labels=labels,
        lower=lower,
        upper=upper,
        partition_tolerance=float(partition_tolerance),
    )

    if grad_smooth is None:
        terms = mutation_region_terms_torch(
            torch_data_from_context(solver_context),
            phi,
            major_prior=float(solver_context.problem.major_prior),
            eps=float(solver_context.problem.eps),
        )
        grad = terms.grad.detach()
        gradient_source = "observed_likelihood"
    else:
        grad = torch.as_tensor(grad_smooth, dtype=phi.dtype, device=phi.device)
        if tuple(grad.shape) != tuple(phi.shape):
            raise ValueError(f"grad_smooth must have shape {tuple(phi.shape)}.")
        if not bool(torch.all(torch.isfinite(grad)).item()):
            raise ValueError("grad_smooth must contain only finite values.")
        grad = grad.detach()
        gradient_source = "provided"

    graph = solver_context.graph
    edge_u, edge_v, edge_w = graph.edge_u, graph.edge_v, graph.weight

    if not bool(materialize_dense_dual):
        return _build_compressed_guided_initialization(
            phi=phi,
            labels=labels,
            grad=grad,
            lower=lower,
            upper=upper,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            solver_context=solver_context,
            guide_adjustment=guide_adjustment,
            max_deviation=max_deviation,
            gradient_source=gradient_source,
            kkt_atol=float(kkt_atol),
            max_capacity_iterations=int(max_capacity_iterations),
        )

    within = labels.index_select(0, edge_u) == labels.index_select(0, edge_v)
    between = ~within
    dense_fits, dense_bytes, dense_limit = dense_complete_solver_memory_preflight(
        num_nodes=num_mutations,
        num_regions=num_regions,
        runtime=solver_context.runtime,
    )
    if not dense_fits:
        raise ExactSolverResourceLimit(
            "exact_solver_resource_limit: dense guided initialization needs "
            f"approximately {dense_bytes} bytes (available device limit: "
            f"{dense_limit}). Use materialize_dense_dual=False or a runtime "
            "with sufficient memory."
        )

    with torch.no_grad():
        zero_dual, _, _ = _assemble_actual_dual(
            lambda_value=0.0,
            phi=phi,
            labels=labels,
            grad_smooth=grad,
            lower=lower,
            upper=upper,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
        )
        required_without_between = _required_within_lambda(
            zero_dual, within=within, edge_w=edge_w
        )

        finfo = torch.finfo(phi.dtype)
        gradient_scale = max(
            1.0,
            float(torch.max(torch.linalg.vector_norm(grad, dim=1)).item()),
        )
        weight_scale = (
            float(torch.max(edge_w).item()) if int(edge_w.numel()) > 0 else 1.0
        )
        numerical_floor = float(
            finfo.eps * gradient_scale / max(weight_scale, finfo.tiny)
        )
        if not np.isfinite(numerical_floor) or numerical_floor <= 0.0:
            numerical_floor = float(finfo.eps)
        lambda_value = max(float(required_without_between), numerical_floor)

        capacity_tolerance = 64.0 * float(finfo.eps)
        del zero_dual
        dual: torch.Tensor | None = None
        flow_demand = torch.zeros_like(phi)
        capacity_iterations = 0
        capacity_converged = False
        for capacity_iterations in range(1, int(max_capacity_iterations) + 1):
            if dual is not None:
                del dual
            dual, flow_demand, _ = _assemble_actual_dual(
                lambda_value=lambda_value,
                phi=phi,
                labels=labels,
                grad_smooth=grad,
                lower=lower,
                upper=upper,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
            )
            required = _required_within_lambda(dual, within=within, edge_w=edge_w)
            if required <= lambda_value * (1.0 + capacity_tolerance):
                capacity_converged = True
                break
            next_lambda = float(np.nextafter(required, np.inf))
            if not np.isfinite(next_lambda) or next_lambda <= lambda_value:
                break
            lambda_value = next_lambda

        if not capacity_converged:
            # Heterogeneous weights can make the between-edge contribution grow
            # faster than the within-block capacity as lambda increases. There
            # is then no exact dual flow that holds the proposed guide fixed.
            # Do not alter the graph to manufacture one: retain the observed
            # zero-between KKT scale and let the online ADMM path move away from
            # the proposal.
            lambda_value = max(float(required_without_between), numerical_floor)
            if dual is not None:
                del dual
            dual, flow_demand, _ = _assemble_actual_dual(
                lambda_value=lambda_value,
                phi=phi,
                labels=labels,
                grad_smooth=grad,
                lower=lower,
                upper=upper,
                edge_u=edge_u,
                edge_v=edge_v,
                edge_w=edge_w,
            )

        if not np.isfinite(lambda_value) or lambda_value <= 0.0:
            raise RuntimeError(
                "Guided-fusion lambda must be finite and strictly positive."
            )

        if dual is None:
            raise RuntimeError("Guided-fusion capacity loop produced no dual state.")
        radius = float(lambda_value) * edge_w
        dual = _project_dual_ball_bounded(dual, radius)
        audit = graph_fusion_kkt_residual_from_grad_torch(
            phi=phi,
            grad_smooth=grad,
            dual_kkt=dual,
            lower=lower,
            upper=upper,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            lambda_value=float(lambda_value),
            atol=float(kkt_atol),
        )

        zero_separation_between_edge_count = _zero_separation_between_count(
            phi,
            between=between,
            edge_u=edge_u,
            edge_v=edge_v,
        )
        flow_balance = torch.zeros_like(phi)
        for cluster in range(int(torch.max(labels).item()) + 1):
            members = torch.nonzero(labels == int(cluster), as_tuple=False).flatten()
            block_sum = torch.sum(flow_demand.index_select(0, members), dim=0)
            flow_balance.index_copy_(
                0, members[:1], block_sum.reshape(1, int(num_regions))
            )
        block_flow_balance = float(torch.max(torch.abs(flow_balance)).item())

        lower_active = phi == lower
        upper_active = phi == upper
        frozen = lower_active & upper_active
        diagnostics = GuidedFusionDiagnostics(
            lambda_value=float(lambda_value),
            required_lambda_without_between_edges=float(required_without_between),
            numerical_lambda_floor=float(numerical_floor),
            capacity_iterations=int(capacity_iterations),
            capacity_converged=bool(capacity_converged),
            capacity_status=(
                "exact_dual_capacity"
                if capacity_converged
                else "projected_zero_between_capacity_scale"
            ),
            num_mutations=int(num_mutations),
            num_regions=int(num_regions),
            num_clusters=int(torch.max(labels).item()) + 1,
            within_edge_count=int(torch.sum(within).item()),
            between_edge_count=int(torch.sum(between).item()),
            zero_separation_between_edge_count=int(zero_separation_between_edge_count),
            gradient_source=str(gradient_source),
            guide_adjustment_max_abs=float(guide_adjustment),
            max_within_cluster_deviation=float(max_deviation),
            block_flow_balance_max_abs=float(block_flow_balance),
            max_dual_ball_ratio=_max_dual_ratio(
                dual, mask=torch.ones_like(within), radius=radius
            ),
            max_within_dual_ball_ratio=_max_dual_ratio(
                dual, mask=within, radius=radius
            ),
            max_between_dual_ball_ratio=_max_dual_ratio(
                dual, mask=between, radius=radius
            ),
            kkt_residual=float(audit["kkt_residual"]),
            stationarity_residual=float(audit["stationarity_residual"]),
            edge_subgradient_residual=float(audit["edge_subgradient_residual"]),
            dual_ball_residual=float(audit["dual_ball_residual"]),
            box_residual=float(audit["box_residual"]),
            num_exact_lower_active_coordinates=int(
                torch.sum(lower_active & ~upper_active).item()
            ),
            num_exact_upper_active_coordinates=int(
                torch.sum(upper_active & ~lower_active).item()
            ),
            num_exact_frozen_coordinates=int(torch.sum(frozen).item()),
        )

    state = SolverState(
        phi=phi.detach(),
        dual=dual.detach(),
        previous_lambda=float(lambda_value),
    )
    return GuidedFusionInitialization(
        lambda_value=float(lambda_value), solver_state=state, diagnostics=diagnostics
    )
