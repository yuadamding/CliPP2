from __future__ import annotations

import numpy as np
import torch

from .types import PairwiseFusionGraph, TensorFusionGraph, TorchRuntime


def _complete_graph_weight(num_nodes: int) -> float:
    return 1.0 / float(max(int(num_nodes) - 1, 1))


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
    complete_edge_count = int(num_nodes) * max(int(num_nodes) - 1, 0) // 2
    is_complete = bool(edge_u.numel() == complete_edge_count)
    is_uniform = bool(weight.numel() == 0 or torch.allclose(weight, weight[:1].expand_as(weight)))
    return TensorFusionGraph(
        edge_index=edge_index,
        weight=weight,
        degree=degree,
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
) -> TensorFusionGraph:
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
