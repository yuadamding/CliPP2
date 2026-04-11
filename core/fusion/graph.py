from __future__ import annotations

import numpy as np

from .types import PairwiseFusionGraph


_EDGE_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _complete_graph_edges(num_mutations: int) -> tuple[np.ndarray, np.ndarray]:
    cached = _EDGE_CACHE.get(int(num_mutations))
    if cached is not None:
        return cached
    edges = np.triu_indices(int(num_mutations), k=1)
    cached = (edges[0].astype(np.int32), edges[1].astype(np.int32))
    _EDGE_CACHE[int(num_mutations)] = cached
    return cached


def _complete_graph_weight(num_mutations: int) -> float:
    return 1.0 / float(max(int(num_mutations) - 1, 1))


def build_complete_uniform_graph(num_mutations: int) -> PairwiseFusionGraph:
    edge_u, edge_v = _complete_graph_edges(num_mutations)
    edge_w = np.full(edge_u.shape[0], _complete_graph_weight(num_mutations), dtype=np.float64)
    return PairwiseFusionGraph(edge_u=edge_u, edge_v=edge_v, edge_w=edge_w, name="complete_uniform")


def coerce_graph(num_mutations: int, graph: PairwiseFusionGraph | None) -> PairwiseFusionGraph:
    if graph is None:
        return build_complete_uniform_graph(num_mutations)
    edge_u = np.asarray(graph.edge_u, dtype=np.int32)
    edge_v = np.asarray(graph.edge_v, dtype=np.int32)
    edge_w = np.asarray(graph.edge_w, dtype=np.float64)
    if edge_u.shape != edge_v.shape or edge_u.shape != edge_w.shape:
        raise ValueError("PairwiseFusionGraph edge arrays must have identical shapes.")
    return PairwiseFusionGraph(edge_u=edge_u, edge_v=edge_v, edge_w=edge_w, name=str(graph.name))


def edge_degree_bound(num_mutations: int, edge_u: np.ndarray, edge_v: np.ndarray) -> int:
    if edge_u.size == 0:
        return 1
    degree = np.bincount(
        np.concatenate([edge_u.astype(np.int64), edge_v.astype(np.int64)]),
        minlength=int(num_mutations),
    )
    return max(int(np.max(degree)), 1)

