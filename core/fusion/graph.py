from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .types import PairwiseFusionGraph


def _complete_graph_edges(num_mutations: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.triu_indices(int(num_mutations), k=1)
    return edges[0].astype(np.int32), edges[1].astype(np.int32)


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
    raw_edge_w = 1.0 / np.power(np.maximum(pairwise_norm, float(tau)), float(gamma))
    mean_raw_weight = float(np.mean(raw_edge_w)) if raw_edge_w.size else 1.0
    if not np.isfinite(mean_raw_weight) or mean_raw_weight <= 0.0:
        raise ValueError("Adaptive pairwise weights must have a positive finite mean.")
    target_mean_weight = float(baseline) * _complete_graph_weight(num_mutations)
    edge_w = raw_edge_w * (target_mean_weight / mean_raw_weight)
    return PairwiseFusionGraph(
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w.astype(np.float64, copy=False),
        name=f"complete_adaptive_gamma{gamma:g}_mean_normalized",
        degree_bound=max(num_mutations - 1, 1),
    )


def likelihood_noise_distance_floor(
    curvature: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    minimum: float = 1e-6,
) -> float:
    """Estimate a pilot-distance floor from local likelihood information.

    ``1 / curvature`` is the local variance approximation for each mutation-
    region CCF.  It is capped by the squared feasible box width, accumulated
    across regions, and multiplied by ``sqrt(2)`` to represent the noise of a
    difference between two independently estimated mutation vectors.  The
    median mutation scale is robust to a minority of flat/boundary likelihoods.

    This quantity depends only on the observed likelihood and feasible domain;
    it does not use a clustering guide or a lambda path.
    """

    h = np.asarray(curvature, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    if h.ndim != 2 or lo.shape != h.shape or hi.shape != h.shape:
        raise ValueError("curvature, lower, and upper must have the same 2D shape.")
    if not np.isfinite(float(minimum)) or float(minimum) <= 0.0:
        raise ValueError("minimum likelihood-noise floor must be finite and positive.")
    if np.any(~np.isfinite(h)) or np.any(h <= 0.0):
        raise ValueError("curvature must contain only finite positive values.")
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(hi < lo):
        raise ValueError("likelihood-noise bounds must be finite with upper >= lower.")

    width_sq = np.square(hi - lo)
    local_variance = np.minimum(1.0 / h, width_sq)
    mutation_scale = np.sqrt(2.0 * np.sum(local_variance, axis=1))
    finite_positive = mutation_scale[
        np.isfinite(mutation_scale) & (mutation_scale > 0.0)
    ]
    if finite_positive.size == 0:
        return float(minimum)
    return float(max(float(np.median(finite_positive)), float(minimum)))


def build_likelihood_noise_regularized_adaptive_graph(
    pilot_phi: np.ndarray,
    curvature: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    gamma: float = 1.0,
    minimum_tau: float = 1e-6,
    baseline: float = 1.0,
    noise_divisor: float = 1.0,
) -> tuple[PairwiseFusionGraph, float]:
    """Build adaptive complete-graph weights with a likelihood-noise floor.

    ``noise_divisor`` distributes a node-level uncertainty scale across the
    incident pairwise terms. Guided complete-graph mode uses its degree
    ``M - 1``; the resulting floor remains positive and data-derived while
    avoiding the near-infinite contrast caused by a fixed ``1e-6`` floor.
    """

    if not np.isfinite(float(noise_divisor)) or float(noise_divisor) <= 0.0:
        raise ValueError("noise_divisor must be finite and positive.")

    node_noise_scale = likelihood_noise_distance_floor(
        curvature,
        lower=lower,
        upper=upper,
        minimum=float(minimum_tau),
    )
    tau = max(
        float(node_noise_scale) / float(noise_divisor),
        float(minimum_tau),
    )
    graph = build_complete_adaptive_graph(
        pilot_phi,
        gamma=float(gamma),
        tau=float(tau),
        baseline=float(baseline),
    )
    graph = PairwiseFusionGraph(
        edge_u=graph.edge_u,
        edge_v=graph.edge_v,
        edge_w=graph.edge_w,
        name=(
            f"complete_adaptive_likelihood_noise_gamma{float(gamma):g}_"
            f"tau{float(tau):.6g}_div{float(noise_divisor):.6g}_mean_normalized"
        ),
        degree_bound=graph.degree_bound,
    )
    return graph, float(tau)


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
        raise ValueError("PairwiseFusionGraph weights must be nonnegative; omit zero-weight edges explicitly.")
    positive_weight = edge_w > 0.0
    edge_u = edge_u[positive_weight]
    edge_v = edge_v[positive_weight]
    edge_w = edge_w[positive_weight]
    if edge_u.size == 0:
        return PairwiseFusionGraph(
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            name=str(graph.name),
            degree_bound=1,
        )

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
