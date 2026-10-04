"""One immutable mutation-similarity tree; never a tumor phylogeny.

The exact streamed Prim construction uses O(N²R) distance work and O(NR)
storage. Edge IDs, rather than independently chosen regional edges, define
the shared cluster support. No imputation creates missing-overlap edges.
"""

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class FrozenTree:
    edges: np.ndarray
    n: int
    mutation_ids: tuple = ()
    kind: str = "tree"

    def __post_init__(self):
        if isinstance(self.n, bool) or not isinstance(self.n, (int, np.integer)) or self.n < 1:
            raise ValueError("A tree requires a positive integer node count")
        edges = np.asarray(self.edges)
        if edges.shape != (self.n - 1, 2) or not np.issubdtype(edges.dtype, np.integer):
            raise ValueError("Tree edges must be an (N-1, 2) integer array")
        edges = edges.astype(np.int64, copy=True)
        if np.any(edges < 0) or np.any(edges >= self.n):
            raise ValueError("Tree edge endpoint outside node range")
        ids = tuple(str(x) for x in self.mutation_ids) if len(self.mutation_ids) else tuple(map(str, range(self.n)))
        if len(ids) != self.n or len(set(ids)) != self.n:
            raise ValueError("Tree mutation IDs must be unique and match the node count")
        parent = np.arange(self.n)

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in edges:
            a, b = root(i), root(j)
            if a == b:
                raise ValueError("Tree has a cycle, duplicate edge, or self edge")
            parent[b] = a
        # N-1 acyclic edges imply connectivity. Bytes-backed arrays cannot be
        # made writeable again with setflags, unlike a write-protected owner.
        object.__setattr__(self, "edges", np.frombuffer(edges.tobytes(), dtype=np.int64).reshape(-1, 2))
        object.__setattr__(self, "mutation_ids", ids)

    @property
    def n_nodes(self):
        return self.n

    @property
    def identity(self):
        digest = hashlib.sha256()
        digest.update(self.kind.encode())
        for value in self.mutation_ids:
            encoded = value.encode()
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        digest.update(self.edges.astype("<i8", copy=False).tobytes())
        return digest.hexdigest()


def build_tree(pilot, observed=None, mutation_ids=None, chain_order=None):
    """Freeze a deterministic exact MST, or the native ordered chain for R=1.

    Complete vectors use squared Euclidean CCF distance. With missingness,
    distance is the mean squared difference over jointly observed coordinates;
    an empty overlap has infinite distance and is never silently connected.
    ``chain_order`` supplies the coordinate-aware scalar specialization order.
    Without genomic coordinates, scalar ties use the canonical mutation ID.
    """
    x = np.asarray(pilot, dtype=float)
    if x.ndim != 2 or not all(x.shape):
        raise ValueError("Pilot must be a nonempty N by R CCF matrix")
    n, r = x.shape
    mask = np.ones_like(x, dtype=bool) if observed is None else np.asarray(observed, dtype=bool)
    if mask.shape != x.shape or np.any(~np.isfinite(x[mask])):
        raise ValueError("Observed pilot coordinates must be finite")
    if np.any(~mask.any(axis=1)):
        raise ValueError("Disconnected observation overlap: a mutation has no observed region")
    ids = tuple(map(str, range(n))) if mutation_ids is None else tuple(map(str, mutation_ids))
    if len(ids) != n or len(set(ids)) != n:
        raise ValueError("Mutation IDs must be unique and match pilot rows")
    canonical = np.argsort(np.asarray(ids), kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[canonical] = np.arange(n)
    if r == 1:
        if chain_order is None:
            order = np.lexsort((ranks, x[:, 0]))
        else:
            raw_order = np.asarray(chain_order)
            if not np.issubdtype(raw_order.dtype, np.integer):
                raise ValueError("Chain order must contain integer node indices")
            order = raw_order.astype(np.int64, copy=True)
            if order.shape != (n,) or not np.array_equal(np.sort(order), np.arange(n)):
                raise ValueError("Chain order must be a permutation of mutation indices")
            if np.any(np.diff(x[order, 0]) < 0):
                raise ValueError("Chain order must respect the pilot ordering")
        return FrozenTree(np.column_stack((order[:-1], order[1:])), n, ids, "chain")
    if chain_order is not None:
        raise ValueError("A scalar chain order is invalid for regional vectors")
    selected = np.zeros(n, dtype=bool)
    best = np.full(n, np.inf)
    best_key = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
    parent = np.full(n, -1, dtype=np.int64)
    current = int(canonical[0])
    edges = []
    complete = bool(mask.all())
    safe = np.where(mask, x, 0.0)
    for step in range(n):
        selected[current] = True
        if step:
            a, b = int(parent[current]), current
            edges.append((a, b) if ranks[a] < ranks[b] else (b, a))
        if step == n - 1:
            break
        overlap = mask & mask[current]
        count = overlap.sum(axis=1)
        distance = np.sum(np.where(overlap, (safe - safe[current]) ** 2, 0.0), axis=1)
        if not complete:
            np.divide(distance, count, out=distance, where=count > 0)
        distance[count == 0] = np.inf
        keys = np.minimum(ranks, ranks[current]) * n + np.maximum(ranks, ranks[current])
        improve = ~selected & ((distance < best) | ((distance == best) & (keys < best_key)))
        best[improve], parent[improve], best_key[improve] = distance[improve], current, keys[improve]
        eligible = np.where(selected, np.inf, best)
        smallest = eligible.min()
        if not np.isfinite(smallest):
            raise ValueError("Disconnected observation-overlap graph: no finite spanning tree")
        tied = np.flatnonzero(eligible == smallest)
        current = int(tied[np.argmin(best_key[tied])])
    edges.sort(key=lambda edge: (ranks[edge[0]], ranks[edge[1]]))
    return FrozenTree(np.asarray(edges, dtype=np.int64).reshape(-1, 2), n, ids, "mst")


def _cut_mask(tree, cuts):
    raw = np.asarray(cuts)
    if raw.ndim != 1 or (raw.size and not np.issubdtype(raw.dtype, np.integer)):
        raise ValueError("Cuts must be a one-dimensional integer edge-ID array")
    cuts = raw.astype(np.int64)
    if np.any(cuts < 0) or np.any(cuts >= len(tree.edges)) or len(np.unique(cuts)) != len(cuts):
        raise ValueError("Invalid or duplicate tree cut IDs")
    mask = np.zeros(len(tree.edges), dtype=bool)
    mask[cuts] = True
    return mask


def partition_from_cuts(tree, cuts):
    """Canonical connected components in the original mutation row order."""
    cut = _cut_mask(tree, cuts)
    parent = np.arange(tree.n)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in tree.edges[~cut]:
        a, b = root(i), root(j)
        parent[b] = a
    roots = np.array([root(i) for i in range(tree.n)])
    mapping = {}
    for i in sorted(range(tree.n), key=lambda i: tree.mutation_ids[i]):
        mapping.setdefault(int(roots[i]), len(mapping))
    return np.array([mapping[int(root)] for root in roots], dtype=np.int64)


def project_cuts(tree, x, k):
    """Euclidean projection support onto at most K-1 nonzero vector rows."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[0] != tree.n or not np.isfinite(x).all():
        raise ValueError("Projection requires finite N by R CCF vectors")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= min(10, tree.n):
        raise ValueError("Capacity must be an integer in 1..min(10,N)")
    # Extended accumulation preserves genuinely nonzero subnormal-sized jumps
    # instead of treating a squared-float64 underflow as an exactly fused row.
    jumps = x[tree.edges[:, 0]].astype(np.longdouble) - x[tree.edges[:, 1]]
    scores = np.einsum("ij,ij->i", jumps, jumps)
    nonzero = np.flatnonzero(scores > 0)
    chosen = nonzero[np.argsort(-scores[nonzero], kind="stable")[: k - 1]]
    return np.sort(chosen)


def topology_diagnostics(tree, truth_labels):
    """Offline truth-only fragmentation audit; never used to fit the tree."""
    labels = np.asarray(truth_labels)
    if labels.shape != (tree.n,):
        raise ValueError("Truth labels must match the frozen mutation population")
    q = len(np.unique(labels))
    q_tree = 1 + int(np.count_nonzero(labels[tree.edges[:, 0]] != labels[tree.edges[:, 1]]))
    return {"true_clusters": q, "tree_components": q_tree, "fragmentation": q_tree - q,
            "representable": q_tree == q, "tree_identity": tree.identity}
