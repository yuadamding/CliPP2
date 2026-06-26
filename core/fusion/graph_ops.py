from __future__ import annotations

import os

import numpy as np
import torch

from .types import PairwiseFusionGraph, TensorFusionGraph, TorchRuntime

PDHG_PRECONDITIONER_ETA = 0.99
COMPLETE_GRAPH_MEMORY_SAFETY_FRACTION = 0.80
COMPLETE_GRAPH_MEMORY_LIMIT_ENV = "CLIPP2_MAX_COMPLETE_GRAPH_BYTES"


def _complete_graph_weight(num_nodes: int) -> float:
    return 1.0 / float(max(int(num_nodes) - 1, 1))


def _dtype_nbytes(dtype: torch.dtype) -> int:
    return int(torch.empty((), dtype=dtype).element_size())


def estimate_complete_tensor_graph_bytes(
    num_nodes: int,
    *,
    num_samples: int,
    dtype: torch.dtype,
    adaptive: bool,
) -> int:
    node_count = max(int(num_nodes), 0)
    sample_count = max(int(num_samples), 1)
    edge_count = node_count * max(node_count - 1, 0) // 2
    value_bytes = _dtype_nbytes(dtype)
    index_bytes = _dtype_nbytes(torch.long)
    persistent_bytes = (
        2 * edge_count * index_bytes
        + edge_count * value_bytes
        + 2 * node_count * value_bytes
    )
    if not adaptive:
        return int(persistent_bytes)
    adaptive_peak_bytes = edge_count * sample_count * value_bytes + 3 * edge_count * value_bytes
    return int(persistent_bytes + adaptive_peak_bytes)


def _parse_memory_limit_bytes(value: str | None) -> int | None:
    if value is None or not str(value).strip():
        return None
    limit = int(float(str(value).strip()))
    if limit <= 0:
        raise ValueError(f"{COMPLETE_GRAPH_MEMORY_LIMIT_ENV} must be positive when set.")
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
    env_limit = _parse_memory_limit_bytes(os.environ.get(COMPLETE_GRAPH_MEMORY_LIMIT_ENV))
    if env_limit is not None:
        return env_limit
    cuda_limit = _cuda_memory_limit_bytes(runtime)
    if cuda_limit is not None:
        return cuda_limit
    return _cpu_memory_limit_bytes(runtime)


def _check_complete_tensor_graph_memory(
    *,
    num_nodes: int,
    num_samples: int,
    runtime: TorchRuntime,
    adaptive: bool,
    memory_limit_bytes: int | None,
) -> None:
    estimate = estimate_complete_tensor_graph_bytes(
        num_nodes,
        num_samples=num_samples,
        dtype=runtime.dtype,
        adaptive=adaptive,
    )
    limit = _complete_graph_memory_limit_bytes(runtime, memory_limit_bytes=memory_limit_bytes)
    if limit is None or estimate <= limit:
        return
    graph_kind = "adaptive" if adaptive else "uniform"
    edge_count = int(num_nodes) * max(int(num_nodes) - 1, 0) // 2
    raise MemoryError(
        f"Estimated complete {graph_kind} tensor graph allocation for "
        f"{int(num_nodes)} nodes ({edge_count} edges) is {estimate} bytes, "
        f"exceeding the configured limit of {int(limit)} bytes. "
        "Use a smaller graph, provide a sparse graph, or raise "
        f"{COMPLETE_GRAPH_MEMORY_LIMIT_ENV}."
    )


def _tensor_graph_from_edges(
    *,
    edge_u: torch.Tensor,
    edge_v: torch.Tensor,
    weight: torch.Tensor,
    num_nodes: int,
    name: str,
) -> TensorFusionGraph:
    edge_index = torch.stack([edge_u.to(dtype=torch.long), edge_v.to(dtype=torch.long)], dim=0)
    degree = torch.zeros((int(num_nodes),), dtype=weight.dtype, device=weight.device)
    if edge_u.numel():
        one = torch.ones_like(weight)
        degree.index_add_(0, edge_index[0], one)
        degree.index_add_(0, edge_index[1], one)
    pdhg_tau_node = (PDHG_PRECONDITIONER_ETA / degree.clamp_min(1.0))[:, None]
    complete_edge_count = int(num_nodes) * max(int(num_nodes) - 1, 0) // 2
    is_complete = bool(edge_u.numel() == complete_edge_count)
    is_uniform = bool(weight.numel() == 0 or torch.allclose(weight, weight[:1].expand_as(weight)))
    return TensorFusionGraph(
        edge_index=edge_index,
        weight=weight,
        degree=degree,
        pdhg_tau_node=pdhg_tau_node,
        num_nodes=int(num_nodes),
        is_complete=is_complete,
        is_uniform=is_uniform,
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
        num_samples=1,
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
        raise ValueError("pilot_phi must be a two-dimensional mutation-by-region matrix.")
    if gamma <= 0.0:
        raise ValueError("Adaptive pairwise weight exponent gamma must be positive.")
    if tau <= 0.0:
        raise ValueError("Adaptive pairwise weight floor tau must be positive.")
    if baseline <= 0.0 or not np.isfinite(baseline):
        raise ValueError("Adaptive pairwise weight baseline must be finite and positive.")

    pilot = pilot_phi.to(dtype=runtime.dtype, device=runtime.device)
    num_nodes = int(pilot.shape[0])
    _check_complete_tensor_graph_memory(
        num_nodes=num_nodes,
        num_samples=int(pilot.shape[1]),
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
        )

    diff = graph_forward_edges(pilot, edge_u=edge_index[0], edge_v=edge_index[1])
    distance = torch.linalg.vector_norm(diff, dim=1)
    raw_weight = distance.clamp_min(float(tau)).pow(-float(gamma))
    mean_raw_weight = torch.mean(raw_weight)
    if not bool(torch.isfinite(mean_raw_weight).item()) or float(mean_raw_weight.item()) <= 0.0:
        raise ValueError("Adaptive pairwise weights must have a positive finite mean.")
    target_mean_weight = float(baseline) * _complete_graph_weight(num_nodes)
    weight = raw_weight * (target_mean_weight / mean_raw_weight)
    return _tensor_graph_from_edges(
        edge_u=edge_index[0],
        edge_v=edge_index[1],
        weight=weight,
        num_nodes=num_nodes,
        name=f"complete_adaptive_gamma{gamma:g}_mean_normalized",
    )


def tensor_graph_to_pairwise_graph(graph: TensorFusionGraph) -> PairwiseFusionGraph:
    edge_u = graph.edge_u.detach().cpu().numpy().astype(np.int32, copy=False)
    edge_v = graph.edge_v.detach().cpu().numpy().astype(np.int32, copy=False)
    edge_w = graph.weight.detach().cpu().numpy().astype(np.float64, copy=False)
    degree_bound = int(torch.max(graph.degree).detach().cpu().item()) if graph.degree.numel() else 1
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
        result.index_add_(0, edge_v, -dual)
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
