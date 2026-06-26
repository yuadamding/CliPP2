from __future__ import annotations

import torch

from .types import PairwiseFusionGraph, TensorFusionGraph, TorchRuntime


def tensorize_graph(
    graph: PairwiseFusionGraph,
    runtime: TorchRuntime,
    *,
    num_nodes: int,
) -> TensorFusionGraph:
    edge_u = torch.as_tensor(graph.edge_u, dtype=torch.long, device=runtime.device)
    edge_v = torch.as_tensor(graph.edge_v, dtype=torch.long, device=runtime.device)
    weight = torch.as_tensor(graph.edge_w, dtype=runtime.dtype, device=runtime.device)
    edge_index = torch.stack([edge_u, edge_v], dim=0)
    degree = torch.zeros((int(num_nodes),), dtype=runtime.dtype, device=runtime.device)
    if edge_u.numel():
        one = torch.ones_like(weight)
        degree.index_add_(0, edge_u, one)
        degree.index_add_(0, edge_v, one)
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
        name=str(graph.name),
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
