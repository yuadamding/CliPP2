from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
    return PairwiseFusionGraph(
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        name="complete_uniform",
        degree_bound=max(int(num_mutations) - 1, 1),
    )


def build_complete_adaptive_graph(
    pilot_phi: np.ndarray,
    *,
    gamma: float = 1.0,
    tau: float = 1e-6,
    baseline: float = 1.0,
) -> PairwiseFusionGraph:
    pilot_phi = np.asarray(pilot_phi, dtype=np.float64)
    if pilot_phi.ndim != 2:
        raise ValueError("pilot_phi must be a two-dimensional mutation-by-region matrix.")
    if gamma <= 0.0:
        raise ValueError("Adaptive pairwise weight exponent gamma must be positive.")
    if tau <= 0.0:
        raise ValueError("Adaptive pairwise weight floor tau must be positive.")
    if baseline <= 0.0 or not np.isfinite(baseline):
        raise ValueError("Adaptive pairwise weight baseline must be finite and positive.")

    num_mutations = int(pilot_phi.shape[0])
    edge_u, edge_v = _complete_graph_edges(num_mutations)
    if edge_u.size == 0:
        return PairwiseFusionGraph(
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=np.zeros((0,), dtype=np.float64),
            name=f"complete_adaptive_gamma{gamma:g}",
            degree_bound=max(num_mutations - 1, 1),
        )

    pairwise_norm = np.linalg.norm(pilot_phi[edge_u] - pilot_phi[edge_v], axis=1)
    edge_w = float(baseline) / np.power(np.maximum(pairwise_norm, float(tau)), float(gamma))
    return PairwiseFusionGraph(
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w.astype(np.float64, copy=False),
        name=f"complete_adaptive_gamma{gamma:g}",
        degree_bound=max(num_mutations - 1, 1),
    )


def resolve_pairwise_fusion_graph(
    num_mutations: int,
    *,
    graph: PairwiseFusionGraph | None,
    pilot_phi: np.ndarray | None = None,
    gamma: float = 1.0,
    tau: float = 1e-6,
    baseline: float = 1.0,
) -> PairwiseFusionGraph:
    if graph is not None:
        return coerce_graph(num_mutations, graph)
    if pilot_phi is not None:
        return build_complete_adaptive_graph(
            pilot_phi,
            gamma=gamma,
            tau=tau,
            baseline=baseline,
        )
    return build_complete_uniform_graph(num_mutations)


def coerce_graph(num_mutations: int, graph: PairwiseFusionGraph | None) -> PairwiseFusionGraph:
    if graph is None:
        return build_complete_uniform_graph(num_mutations)
    edge_u = np.asarray(graph.edge_u, dtype=np.int32)
    edge_v = np.asarray(graph.edge_v, dtype=np.int32)
    edge_w = np.asarray(graph.edge_w, dtype=np.float64)
    if edge_u.ndim != 1 or edge_v.ndim != 1 or edge_w.ndim != 1:
        raise ValueError("PairwiseFusionGraph edge arrays must be one-dimensional.")
    if edge_u.shape != edge_v.shape or edge_u.shape != edge_w.shape:
        raise ValueError("PairwiseFusionGraph edge arrays must have identical shapes.")
    if edge_u.size == 0:
        return PairwiseFusionGraph(
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            name=str(graph.name),
            degree_bound=1,
        )
    if np.any(edge_u < 0) or np.any(edge_v < 0) or np.any(edge_u >= int(num_mutations)) or np.any(edge_v >= int(num_mutations)):
        raise ValueError("PairwiseFusionGraph edge indices must lie in [0, num_mutations).")
    if np.any(edge_u == edge_v):
        raise ValueError("PairwiseFusionGraph may not contain self-loops.")
    if not np.all(np.isfinite(edge_w)):
        raise ValueError("PairwiseFusionGraph weights must be finite.")
    if np.any(edge_w < 0.0):
        raise ValueError("PairwiseFusionGraph weights must be nonnegative.")

    left = np.minimum(edge_u, edge_v)
    right = np.maximum(edge_u, edge_v)
    canonical_edges = np.stack([left, right], axis=1)
    _, unique_index = np.unique(canonical_edges, axis=0, return_index=True)
    if unique_index.size != canonical_edges.shape[0]:
        raise ValueError("PairwiseFusionGraph may not contain duplicate undirected edges.")

    order = np.lexsort((right, left))
    return PairwiseFusionGraph(
        edge_u=left[order],
        edge_v=right[order],
        edge_w=edge_w[order],
        name=str(graph.name),
        degree_bound=edge_degree_bound(num_mutations, edge_u=left[order], edge_v=right[order]),
    )


def edge_degree_bound(num_mutations: int, edge_u: np.ndarray, edge_v: np.ndarray) -> int:
    if edge_u.size == 0:
        return 1
    degree = np.bincount(
        np.concatenate([edge_u.astype(np.int64), edge_v.astype(np.int64)]),
        minlength=int(num_mutations),
    )
    return max(int(np.max(degree)), 1)


def load_pairwise_fusion_graph_tsv(
    file_path: str | Path,
    *,
    num_mutations: int,
    mutation_ids: list[str] | None = None,
    default_weight: float = 1.0,
) -> PairwiseFusionGraph:
    file_path = Path(file_path)
    df = pd.read_csv(file_path, sep="\t").copy()
    if df.empty:
        raise ValueError(f"Graph file {file_path} is empty.")

    if {"edge_u", "edge_v"}.issubset(df.columns):
        edge_u = df["edge_u"].to_numpy(dtype=np.int32, copy=True)
        edge_v = df["edge_v"].to_numpy(dtype=np.int32, copy=True)
    else:
        left_col = "mutation_u" if "mutation_u" in df.columns else "mutation_id_u" if "mutation_id_u" in df.columns else None
        right_col = "mutation_v" if "mutation_v" in df.columns else "mutation_id_v" if "mutation_id_v" in df.columns else None
        if left_col is None or right_col is None:
            raise ValueError(
                f"Graph file {file_path} must contain either ('edge_u', 'edge_v') or "
                f"('mutation_u'/'mutation_id_u', 'mutation_v'/'mutation_id_v')."
            )
        if mutation_ids is None:
            raise ValueError(f"Graph file {file_path} uses mutation IDs, but no mutation_ids mapping was provided.")
        mutation_index = {str(mutation_id): idx for idx, mutation_id in enumerate(mutation_ids)}
        try:
            edge_u = df[left_col].astype(str).map(mutation_index.__getitem__).to_numpy(dtype=np.int32, copy=True)
            edge_v = df[right_col].astype(str).map(mutation_index.__getitem__).to_numpy(dtype=np.int32, copy=True)
        except KeyError as exc:
            raise ValueError(f"Graph file {file_path} references unknown mutation ID {exc.args[0]!r}.") from exc

    if "edge_w" in df.columns:
        edge_w = df["edge_w"].to_numpy(dtype=np.float64, copy=True)
    else:
        edge_w = np.full(edge_u.shape[0], float(default_weight), dtype=np.float64)

    return coerce_graph(
        num_mutations,
        PairwiseFusionGraph(
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            name=file_path.stem,
        ),
    )
