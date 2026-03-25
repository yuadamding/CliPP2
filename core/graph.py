from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class GraphData:
    src: np.ndarray
    dst: np.ndarray
    weight: np.ndarray

    @property
    def num_edges(self) -> int:
        return int(self.src.shape[0])


def build_knn_graph(phi_matrix: np.ndarray, k: int = 10, min_weight: float = 1e-4) -> GraphData:
    phi_matrix = np.asarray(phi_matrix, dtype=np.float32)
    num_mutations = phi_matrix.shape[0]
    if num_mutations <= 1:
        empty_int = np.zeros(0, dtype=np.int64)
        empty_float = np.zeros(0, dtype=np.float32)
        return GraphData(src=empty_int, dst=empty_int, weight=empty_float)

    k = max(1, min(int(k), num_mutations - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(phi_matrix)
    distances, neighbors = nn.kneighbors(phi_matrix, return_distance=True)

    local_scale = distances[:, -1].copy()
    nonzero = local_scale > 0
    scale_fallback = float(np.median(local_scale[nonzero])) if np.any(nonzero) else 1.0
    local_scale = np.where(local_scale > 0, local_scale, scale_fallback)
    local_scale = np.clip(local_scale, 1e-6, None)

    edge_map: dict[tuple[int, int], float] = {}
    for i in range(num_mutations):
        for dist, j in zip(distances[i, 1:], neighbors[i, 1:]):
            a, b = (i, int(j)) if i < j else (int(j), i)
            if a == b:
                continue
            denom = local_scale[a] * local_scale[b]
            weight = float(np.exp(-(float(dist) ** 2) / max(denom, 1e-6)))
            if weight < min_weight:
                continue
            current = edge_map.get((a, b))
            if current is None or weight > current:
                edge_map[(a, b)] = weight

    if not edge_map:
        empty_int = np.zeros(0, dtype=np.int64)
        empty_float = np.zeros(0, dtype=np.float32)
        return GraphData(src=empty_int, dst=empty_int, weight=empty_float)

    edges = sorted(edge_map.items())
    src = np.fromiter((edge[0][0] for edge in edges), dtype=np.int64)
    dst = np.fromiter((edge[0][1] for edge in edges), dtype=np.int64)
    weight = np.fromiter((edge[1] for edge in edges), dtype=np.float32)
    return GraphData(src=src, dst=dst, weight=weight)
