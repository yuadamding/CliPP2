from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import torch

from .types import PairwiseFusionGraph, TensorFusionGraph, TorchRuntime

PDHG_PRECONDITIONER_ETA = 0.99
COMPLETE_GRAPH_MEMORY_SAFETY_FRACTION = 0.80
COMPLETE_GRAPH_MEMORY_LIMIT_ENV = "CLIPP2_MAX_COMPLETE_GRAPH_BYTES"
COMPLETE_ADAPTIVE_WEIGHT_CHUNK_BYTES = 64 * 1024 * 1024
COMPLETE_ADMM_EDGE_WORK_BYTES = 64 * 1024 * 1024


def _complete_graph_weight(num_nodes: int) -> float:
    return 1.0 / float(max(int(num_nodes) - 1, 1))


def _complete_graph_edge_count(num_nodes: int) -> int:
    node_count = max(int(num_nodes), 0)
    return node_count * max(node_count - 1, 0) // 2


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size())


def _adaptive_weight_work_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype == torch.float16 else dtype


def _matches_runtime_device(actual: torch.device, requested: torch.device) -> bool:
    return bool(
        actual.type == requested.type
        and (requested.index is None or actual.index == requested.index)
    )


def _adaptive_weight_chunk_size(*, num_regions: int, dtype: torch.dtype) -> int:
    region_count = max(int(num_regions), 1)
    bytes_per_edge = max(region_count * _dtype_nbytes(dtype), 1)
    return max(1, int(COMPLETE_ADAPTIVE_WEIGHT_CHUNK_BYTES // bytes_per_edge))


def estimate_complete_tensor_graph_bytes(
    num_nodes: int,
    *,
    num_regions: int,
    dtype: torch.dtype,
    adaptive: bool,
) -> int:
    node_count = max(int(num_nodes), 0)
    region_count = max(int(num_regions), 1)
    edge_count = _complete_graph_edge_count(node_count)
    value_bytes = _dtype_nbytes(dtype)
    index_bytes = _dtype_nbytes(torch.long)
    persistent_bytes = (
        2 * edge_count * index_bytes
        + edge_count * value_bytes
        + 2 * node_count * value_bytes
    )
    if not adaptive:
        return int(persistent_bytes)
    work_dtype = _adaptive_weight_work_dtype(dtype)
    work_value_bytes = _dtype_nbytes(work_dtype)
    chunk_edges = min(
        edge_count,
        _adaptive_weight_chunk_size(num_regions=region_count, dtype=work_dtype),
    )
    adaptive_peak_bytes = (
        chunk_edges * region_count * work_value_bytes
        + 3 * chunk_edges * work_value_bytes
    )
    return int(persistent_bytes + adaptive_peak_bytes)


def estimate_dense_complete_solver_peak_bytes(
    num_nodes: int,
    *,
    num_regions: int,
    dtype: torch.dtype,
    include_dual: bool = True,
    include_split: bool = True,
    include_graph: bool = True,
) -> int:
    """Conservative peak-memory estimate for complete-graph dense ADMM.

    Streaming bounds temporary edge chunks, but it still retains the scaled
    multiplier and split variable. Terminal certificate refinement can overlap
    an incoming multiplier with one newly allocated output multiplier, so an
    additional edge tensor is included when ``include_dual`` is true.
    """

    node_count = max(int(num_nodes), 0)
    region_count = max(int(num_regions), 1)
    edge_count = _complete_graph_edge_count(node_count)
    value_bytes = _dtype_nbytes(dtype)
    edge_value_bytes = edge_count * region_count * value_bytes
    estimate = node_count * region_count * value_bytes * 8
    if include_dual or include_split:
        if edge_value_bytes <= COMPLETE_ADMM_EDGE_WORK_BYTES:
            # The historical dense ALM loop can overlap the multiplier, prior
            # split, forward differences, shrinkage argument/output, residual,
            # and next multiplier. Count the actual peak, not just state retained
            # between iterations.
            dense_edge_states = 11
            estimate += dense_edge_states * edge_value_bytes
        else:
            persistent_edge_states = int(bool(include_dual)) * 2 + int(
                bool(include_split)
            )
            estimate += persistent_edge_states * edge_value_bytes
            # The streaming ALM bounds each temporary, but several chunk values
            # and norms can coexist during shrinkage and residual updates.
            estimate += 6 * COMPLETE_ADMM_EDGE_WORK_BYTES
    if include_graph:
        estimate += estimate_complete_tensor_graph_bytes(
            node_count,
            num_regions=region_count,
            dtype=dtype,
            adaptive=False,
        )
    return int(estimate)


def _parse_memory_limit_bytes(value: str | None) -> int | None:
    if value is None or not str(value).strip():
        return None
    limit = int(float(str(value).strip()))
    if limit <= 0:
        raise ValueError(
            f"{COMPLETE_GRAPH_MEMORY_LIMIT_ENV} must be positive when set."
        )
    return limit


def _cuda_memory_limit_bytes(runtime: TorchRuntime) -> int | None:
    if runtime.device.type != "cuda":
        return None
    try:
        free_bytes, _ = torch.cuda.mem_get_info(runtime.device)
    except Exception:
        return None
    return int(COMPLETE_GRAPH_MEMORY_SAFETY_FRACTION * int(free_bytes))


def _cpu_memory_limit_bytes(runtime: TorchRuntime) -> int | None:
    if runtime.device.type != "cpu":
        return None
    try:
        page_count = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    if int(page_count) <= 0 or int(page_size) <= 0:
        return None
    return int(COMPLETE_GRAPH_MEMORY_SAFETY_FRACTION * int(page_count) * int(page_size))


def _complete_graph_memory_limit_bytes(
    runtime: TorchRuntime,
    *,
    memory_limit_bytes: int | None,
) -> int | None:
    if memory_limit_bytes is not None:
        return int(memory_limit_bytes)
    env_limit = _parse_memory_limit_bytes(
        os.environ.get(COMPLETE_GRAPH_MEMORY_LIMIT_ENV)
    )
    if env_limit is not None:
        return env_limit
    cuda_limit = _cuda_memory_limit_bytes(runtime)
    if cuda_limit is not None:
        return cuda_limit
    return _cpu_memory_limit_bytes(runtime)


def dense_complete_solver_memory_preflight(
    *,
    num_nodes: int,
    num_regions: int,
    runtime: TorchRuntime,
    memory_limit_bytes: int | None = None,
) -> tuple[bool, int, int | None]:
    estimate = estimate_dense_complete_solver_peak_bytes(
        num_nodes,
        num_regions=num_regions,
        dtype=runtime.dtype,
    )
    limit = _complete_graph_memory_limit_bytes(
        runtime,
        memory_limit_bytes=memory_limit_bytes,
    )
    return limit is None or estimate <= limit, estimate, limit


def _check_complete_tensor_graph_memory(
    *,
    num_nodes: int,
    num_regions: int,
    runtime: TorchRuntime,
    adaptive: bool,
    memory_limit_bytes: int | None,
) -> None:
    estimate = estimate_complete_tensor_graph_bytes(
        num_nodes,
        num_regions=num_regions,
        dtype=runtime.dtype,
        adaptive=adaptive,
    )
    limit = _complete_graph_memory_limit_bytes(
        runtime, memory_limit_bytes=memory_limit_bytes
    )
    if limit is None or estimate <= limit:
        return
    graph_kind = "adaptive" if adaptive else "uniform"
    edge_count = _complete_graph_edge_count(num_nodes)
    raise MemoryError(
        f"Estimated complete {graph_kind} tensor graph allocation for "
        f"{int(num_nodes)} nodes ({edge_count} edges) is {estimate} bytes, "
        f"exceeding the configured limit of {int(limit)} bytes. "
        "Use a smaller graph, provide a sparse graph, or raise "
        f"{COMPLETE_GRAPH_MEMORY_LIMIT_ENV}."
    )


def _is_canonical_complete_edge_index(
    edge_index: torch.Tensor, *, num_nodes: int
) -> bool:
    edge_count = _complete_graph_edge_count(num_nodes)
    if int(edge_index.shape[0]) != 2 or int(edge_index.shape[1]) != edge_count:
        return False
    if edge_count == 0:
        return True
    expected = torch.triu_indices(
        int(num_nodes),
        int(num_nodes),
        offset=1,
        dtype=edge_index.dtype,
        device=edge_index.device,
    )
    return bool(torch.equal(edge_index, expected))


def _complete_adaptive_raw_weights(
    pilot: torch.Tensor,
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    gamma: float,
    tau: float,
) -> torch.Tensor:
    num_edges = int(edge_u.numel())
    raw_weight = torch.empty((num_edges,), dtype=pilot.dtype, device=pilot.device)
    chunk_size = max(
        1,
        min(
            num_edges,
            _adaptive_weight_chunk_size(
                num_regions=int(pilot.shape[1]), dtype=pilot.dtype
            ),
        ),
    )
    for start in range(0, num_edges, chunk_size):
        stop = min(start + chunk_size, num_edges)
        diff = graph_forward_edges(
            pilot,
            edge_u=edge_u[start:stop],
            edge_v=edge_v[start:stop],
        )
        distance = torch.linalg.vector_norm(diff, dim=1)
        raw_weight[start:stop] = distance.clamp_min(float(tau)).pow(-float(gamma))
    return raw_weight


def _tensor_graph_from_edges(
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    weight: torch.Tensor,
    num_nodes: int,
    name: str,
    known_complete: bool | None = None,
) -> TensorFusionGraph:
    edge_index = torch.stack(
        [edge_u.to(dtype=torch.long), edge_v.to(dtype=torch.long)], dim=0
    )
    degree = torch.zeros((int(num_nodes),), dtype=weight.dtype, device=weight.device)
    if edge_u.numel():
        one = torch.ones_like(weight)
        degree.index_add_(0, edge_index[0], one)
        degree.index_add_(0, edge_index[1], one)
    pdhg_tau_node = (PDHG_PRECONDITIONER_ETA / degree.clamp_min(1.0))[:, None]
    is_complete = (
        _is_canonical_complete_edge_index(edge_index, num_nodes=int(num_nodes))
        if known_complete is None
        else bool(known_complete)
    )
    return TensorFusionGraph(
        edge_index=edge_index,
        weight=weight,
        degree=degree,
        pdhg_tau_node=pdhg_tau_node,
        num_nodes=int(num_nodes),
        is_complete=is_complete,
        name=str(name),
    )


def tensorize_graph(
    graph: PairwiseFusionGraph,
    runtime: TorchRuntime,
    *,
    num_nodes: int,
) -> TensorFusionGraph:
    edge_u = torch.as_tensor(graph.edge_u, dtype=torch.long, device=runtime.device)
    edge_v = torch.as_tensor(graph.edge_v, dtype=torch.long, device=runtime.device)
    weight = torch.as_tensor(graph.edge_w, dtype=runtime.dtype, device=runtime.device)
    return _tensor_graph_from_edges(
        edge_u=edge_u,
        edge_v=edge_v,
        weight=weight,
        num_nodes=int(num_nodes),
        name=str(graph.name),
    )


def build_complete_uniform_tensor_graph(
    num_nodes: int,
    runtime: TorchRuntime,
    *,
    memory_limit_bytes: int | None = None,
) -> TensorFusionGraph:
    _check_complete_tensor_graph_memory(
        num_nodes=int(num_nodes),
        num_regions=1,
        runtime=runtime,
        adaptive=False,
        memory_limit_bytes=memory_limit_bytes,
    )
    edge_index = torch.triu_indices(
        int(num_nodes),
        int(num_nodes),
        offset=1,
        dtype=torch.long,
        device=runtime.device,
    )
    weight = torch.full(
        (int(edge_index.shape[1]),),
        _complete_graph_weight(num_nodes),
        dtype=runtime.dtype,
        device=runtime.device,
    )
    return _tensor_graph_from_edges(
        edge_u=edge_index[0],
        edge_v=edge_index[1],
        weight=weight,
        num_nodes=int(num_nodes),
        name="complete_uniform",
        known_complete=True,
    )


def build_complete_adaptive_tensor_graph(
    pilot_phi: torch.Tensor,
    runtime: TorchRuntime,
    *,
    gamma: float = 1.0,
    tau: float = 1e-6,
    baseline: float = 1.0,
    memory_limit_bytes: int | None = None,
) -> TensorFusionGraph:
    if pilot_phi.ndim != 2:
        raise ValueError(
            "pilot_phi must be a two-dimensional mutation-by-region matrix."
        )
    if gamma <= 0.0:
        raise ValueError("Adaptive pairwise weight exponent gamma must be positive.")
    if tau <= 0.0:
        raise ValueError("Adaptive pairwise weight floor tau must be positive.")
    if baseline <= 0.0 or not np.isfinite(baseline):
        raise ValueError(
            "Adaptive pairwise weight baseline must be finite and positive."
        )

    weight_work_dtype = _adaptive_weight_work_dtype(runtime.dtype)
    pilot = pilot_phi.to(dtype=weight_work_dtype, device=runtime.device)
    num_nodes = int(pilot.shape[0])
    _check_complete_tensor_graph_memory(
        num_nodes=num_nodes,
        num_regions=int(pilot.shape[1]),
        runtime=runtime,
        adaptive=True,
        memory_limit_bytes=memory_limit_bytes,
    )
    edge_index = torch.triu_indices(
        num_nodes,
        num_nodes,
        offset=1,
        dtype=torch.long,
        device=runtime.device,
    )
    if edge_index.shape[1] == 0:
        weight = torch.zeros((0,), dtype=runtime.dtype, device=runtime.device)
        return _tensor_graph_from_edges(
            edge_u=edge_index[0],
            edge_v=edge_index[1],
            weight=weight,
            num_nodes=num_nodes,
            name=f"complete_adaptive_gamma{gamma:g}",
            known_complete=True,
        )

    raw_weight = _complete_adaptive_raw_weights(
        pilot,
        edge_u=edge_index[0],
        edge_v=edge_index[1],
        gamma=float(gamma),
        tau=float(tau),
    )
    mean_raw_weight = torch.mean(raw_weight)
    if (
        not bool(torch.isfinite(mean_raw_weight).item())
        or float(mean_raw_weight.item()) <= 0.0
    ):
        raise ValueError("Adaptive pairwise weights must have a positive finite mean.")
    target_mean_weight = float(baseline) * _complete_graph_weight(num_nodes)
    weight = raw_weight.mul_(target_mean_weight / mean_raw_weight).to(
        dtype=runtime.dtype
    )
    return _tensor_graph_from_edges(
        edge_u=edge_index[0],
        edge_v=edge_index[1],
        weight=weight,
        num_nodes=num_nodes,
        name=f"complete_adaptive_gamma{gamma:g}_mean_normalized",
        known_complete=True,
    )


def likelihood_noise_distance_floor_torch(
    curvature: torch.Tensor,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    minimum: float = 1e-6,
) -> torch.Tensor:
    """Torch equivalent of the likelihood-derived adaptive distance floor.

    The returned scalar stays on the input device. The explicit sorted median
    matches NumPy's midpoint rule for an even number of mutations; PyTorch's
    ``median`` instead selects the lower middle value.
    """

    if (
        curvature.ndim != 2
        or lower.shape != curvature.shape
        or upper.shape != curvature.shape
    ):
        raise ValueError("curvature, lower, and upper must have the same 2D shape.")
    if not np.isfinite(float(minimum)) or float(minimum) <= 0.0:
        raise ValueError("minimum likelihood-noise floor must be finite and positive.")
    if lower.device != curvature.device or upper.device != curvature.device:
        raise ValueError("curvature, lower, and upper must be on the same device.")

    work_dtype = _adaptive_weight_work_dtype(curvature.dtype)
    h = curvature.to(dtype=work_dtype)
    lo = lower.to(dtype=work_dtype)
    hi = upper.to(dtype=work_dtype)
    if not bool(torch.all(torch.isfinite(h) & (h > 0.0)).item()):
        raise ValueError("curvature must contain only finite positive values.")
    valid_bounds = torch.isfinite(lo) & torch.isfinite(hi) & (hi >= lo)
    if not bool(torch.all(valid_bounds).item()):
        raise ValueError("likelihood-noise bounds must be finite with upper >= lower.")

    width_sq = torch.square(hi - lo)
    local_variance = torch.minimum(torch.reciprocal(h), width_sq)
    mutation_scale = torch.sqrt(2.0 * torch.sum(local_variance, dim=1))
    finite_positive = mutation_scale[
        torch.isfinite(mutation_scale) & (mutation_scale > 0.0)
    ]
    minimum_tensor = torch.as_tensor(
        float(minimum), dtype=work_dtype, device=curvature.device
    )
    if int(finite_positive.numel()) == 0:
        return minimum_tensor

    ordered = torch.sort(finite_positive).values
    middle = int(ordered.numel()) // 2
    if int(ordered.numel()) % 2:
        median = ordered[middle]
    else:
        median = 0.5 * (ordered[middle - 1] + ordered[middle])
    return torch.maximum(median, minimum_tensor)


def build_likelihood_noise_regularized_adaptive_tensor_graph(
    pilot_phi: torch.Tensor,
    curvature: torch.Tensor,
    runtime: TorchRuntime,
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    gamma: float = 1.0,
    minimum_tau: float = 1e-6,
    baseline: float = 1.0,
    noise_divisor: float = 1.0,
    memory_limit_bytes: int | None = None,
) -> tuple[TensorFusionGraph, float]:
    """Build the likelihood-noise adaptive graph entirely on ``runtime``.

    A single scalar synchronization is required to encode ``tau`` in the
    stable graph name and selection diagnostics. Edge construction and weight
    normalization remain on the runtime device.
    """

    if not np.isfinite(float(noise_divisor)) or float(noise_divisor) <= 0.0:
        raise ValueError("noise_divisor must be finite and positive.")
    runtime_device = runtime.device
    for name, value in (
        ("pilot_phi", pilot_phi),
        ("curvature", curvature),
        ("lower", lower),
        ("upper", upper),
    ):
        if not _matches_runtime_device(value.device, runtime_device):
            raise ValueError(f"{name} must be on the Torch runtime device.")

    node_noise_scale = likelihood_noise_distance_floor_torch(
        curvature,
        lower=lower,
        upper=upper,
        minimum=float(minimum_tau),
    )
    minimum_tensor = torch.as_tensor(
        float(minimum_tau),
        dtype=node_noise_scale.dtype,
        device=runtime_device,
    )
    tau_tensor = torch.maximum(node_noise_scale / float(noise_divisor), minimum_tensor)
    tau = float(tau_tensor.item())
    graph = build_complete_adaptive_tensor_graph(
        pilot_phi,
        runtime,
        gamma=float(gamma),
        tau=tau,
        baseline=float(baseline),
        memory_limit_bytes=memory_limit_bytes,
    )
    return (
        replace(
            graph,
            name=(
                f"complete_adaptive_likelihood_noise_gamma{float(gamma):g}_"
                f"tau{tau:.6g}_div{float(noise_divisor):.6g}_mean_normalized"
            ),
        ),
        tau,
    )


def tensor_graph_to_pairwise_graph(graph: TensorFusionGraph) -> PairwiseFusionGraph:
    edge_u = graph.edge_u.detach().cpu().numpy().astype(np.int32, copy=False)
    edge_v = graph.edge_v.detach().cpu().numpy().astype(np.int32, copy=False)
    edge_w = graph.weight.detach().cpu().numpy().astype(np.float64, copy=False)
    degree_bound = (
        int(torch.max(graph.degree).detach().cpu().item())
        if graph.degree.numel()
        else 1
    )
    return PairwiseFusionGraph(
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        name=str(graph.name),
        degree_bound=max(degree_bound, 1),
    )


def graph_forward(phi: torch.Tensor, graph: TensorFusionGraph) -> torch.Tensor:
    return graph_forward_edges(phi, edge_u=graph.edge_u, edge_v=graph.edge_v)


def graph_forward_edges(
    phi: torch.Tensor,
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
) -> torch.Tensor:
    return phi.index_select(0, edge_u) - phi.index_select(0, edge_v)


def graph_adjoint(dual: torch.Tensor, graph: TensorFusionGraph) -> torch.Tensor:
    return graph_adjoint_edges(
        dual,
        edge_u=graph.edge_u,
        edge_v=graph.edge_v,
        num_nodes=graph.num_nodes,
    )


def graph_adjoint_edges(
    dual: torch.Tensor,
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    result = dual.new_zeros((int(num_nodes), int(dual.shape[1])))
    if edge_u.numel():
        result.index_add_(0, edge_u, dual)
        # ``alpha=-1`` avoids allocating a complete negated edge tensor.  This
        # matters for complete graphs, where E=M(M-1)/2 can be several million.
        result.index_add_(0, edge_v, dual, alpha=-1.0)
    return result


def project_dual_ball(dual: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    if dual.numel() == 0:
        return dual
    norm = torch.linalg.vector_norm(dual, dim=1, keepdim=True)
    scale = torch.maximum(
        torch.ones_like(norm),
        norm / radius[:, None].clamp_min(torch.finfo(dual.dtype).tiny),
    )
    return dual / scale
