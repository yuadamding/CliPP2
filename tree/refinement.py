"""Connected tree boundary moves, exact only for a pair at fixed centers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundaryMove:
    edge: int
    labels: np.ndarray
    centers: np.ndarray
    conditional_log_likelihood: float
    changed: bool


def _finite_terms(values):
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("Likelihoods must be finite or negative infinity")
    return np.where(np.isfinite(values), values, 0.0), np.isneginf(values).astype(np.int64)


def best_pair_boundary(tree, labels, centers, log_columns, edge, *, adjacency=None, components=None):
    """Best replacement of one boundary, including both center orientations.

    Finite likelihood sums and impossible-assignment counts are accumulated
    separately. No subtraction involving infinite log likelihoods is used.
    Ties retain the current partition, then use canonical edge/orientation order.
    Returned labels keep the input cluster IDs so returned centers are aligned.
    """
    labels = np.asarray(labels, dtype=np.int64)
    centers = np.asarray(centers, dtype=float)
    columns = np.asarray(log_columns, dtype=float)
    edges = np.asarray(tree.edges, dtype=np.int64)
    if labels.shape != (tree.n,) or columns.shape != (tree.n, len(centers)):
        raise ValueError("Boundary likelihoods and memberships must match the tree")
    edge = int(edge)
    if not 0 <= edge < len(edges):
        raise ValueError("Invalid boundary edge")
    a, b = map(int, labels[edges[edge]])
    if a == b:
        raise ValueError("The requested edge is not a cluster boundary")
    members = (np.flatnonzero((labels == a) | (labels == b)) if components is None
               else np.sort(np.r_[components[a], components[b]]))
    positions = {int(node): index for index, node in enumerate(members)}
    if adjacency is None:
        adjacency = [[] for _ in range(tree.n)]
        for eid, (u, v) in enumerate(edges):
            adjacency[u].append((int(v), eid))
            adjacency[v].append((int(u), eid))
    local = {int(node): [(neighbor, eid) for neighbor, eid in adjacency[node]
                        if neighbor in positions] for node in members}
    internal = sorted({eid for neighbors in local.values() for _, eid in neighbors})
    # Two connected components joined by one tree edge form a subtree.
    if (len(internal) != len(members) - 1 or sum(
            labels[edges[eid, 0]] != labels[edges[eid, 1]] for eid in internal) != 1):
        raise ValueError("Boundary refinement requires connected input components")
    root = int(members[0])
    parent = {root: (-1, -1)}
    order = [root]
    for node in order:
        for neighbor, eid in local[node]:
            if neighbor != parent[node][0]:
                parent[neighbor] = (node, eid)
                order.append(neighbor)
    if len(order) != len(members):
        raise ValueError("Adjacent component union is disconnected")

    finite, impossible = _finite_terms(columns[np.ix_(members, [a, b])])
    sums, counts = finite.copy(), impossible.copy()
    child_for_edge = {}
    for node in reversed(order[1:]):
        ancestor, eid = parent[node]
        sums[positions[ancestor]] += sums[positions[node]]
        counts[positions[ancestor]] += counts[positions[node]]
        child_for_edge[eid] = node
    current = columns[members, labels[members]]
    best = float(current.sum())
    selected_edge, selected_orientation = edge, None
    for eid in sorted(internal):
        child = positions[child_for_edge[eid]]
        for side in (0, 1):
            other = 1 - side
            bad = counts[child, side] + counts[0, other] - counts[child, other]
            score = (-np.inf if bad else float(
                sums[child, side] + sums[0, other] - sums[child, other]))
            if score > best:
                best, selected_edge, selected_orientation = score, eid, side
    if selected_orientation is None:
        return BoundaryMove(edge, labels.copy(), centers.copy(), best, False)

    child = child_for_edge[selected_edge]
    nodes = [child]
    for node in nodes:
        nodes.extend(neighbor for neighbor, _ in local[node]
                     if neighbor != parent[node][0])
    proposed = labels.copy()
    cluster_ids = (a, b)
    proposed[members] = cluster_ids[1 - selected_orientation]
    proposed[nodes] = cluster_ids[selected_orientation]
    return BoundaryMove(selected_edge, proposed, centers.copy(), best,
                        not np.array_equal(proposed, labels))


def canonical_partition(labels, centers=None):
    """Canonicalize by the first node in each block, preserving center alignment."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("A partition must be a nonempty label vector")
    seen = {}
    result = np.empty(len(labels), dtype=np.int64)
    for index, value in enumerate(labels.tolist()):
        if value not in seen:
            seen[value] = len(seen)
        result[index] = seen[value]
    if centers is None:
        return result
    return result, np.asarray(centers)[list(seen)].copy()


def refine_partition(model, tree, candidate, consider):
    """Yield every accepted local state; scoring is independent of acceptance.

    ``consider`` refits and records candidates, preserving supplied-center
    fallbacks. It must return a candidate with a ``refit`` result and ``cuts``.
    The path improves conditional likelihood, while the caller separately keeps
    the best observed-mixture score over the complete candidate bank.
    """
    from ..kernel.topology import partition_from_cuts

    current = candidate
    visited = set()
    adjacency = [[] for _ in range(tree.n)]
    for eid, (u, v) in enumerate(tree.edges):
        adjacency[u].append((int(v), eid))
        adjacency[v].append((int(u), eid))
    while True:
        changed = False
        fit = current.refit
        labels = fit.labels
        components = {cluster: np.flatnonzero(labels == cluster) for cluster in range(len(fit.centers))}
        # The model columns already enforce the complete regional assignment box.
        columns = model.log_likelihood_columns(fit.centers)
        for edge in current.cuts:
            move = best_pair_boundary(tree, labels, fit.centers, columns, edge,
                                      adjacency=adjacency, components=components)
            if not move.changed:
                continue
            proposed_labels, centers = canonical_partition(move.labels, move.centers)
            cuts = tuple(np.flatnonzero(proposed_labels[tree.edges[:, 0]] != proposed_labels[tree.edges[:, 1]]))
            proposal_key = (cuts, centers.tobytes())
            if proposal_key in visited:
                continue
            if not np.array_equal(proposed_labels, partition_from_cuts(tree, cuts)):
                raise ValueError("Boundary move violated connected partition identity")
            visited.add(proposal_key)
            proposed = consider(cuts, proposed_labels, centers, {
                "route": "boundary_refinement", "parent_cuts": current.cuts,
                "replaced_edge": int(edge), "replacement_edge": move.edge,
            })
            if proposed is None:
                continue
            old = float(fit.conditional_log_likelihood)
            new = float(proposed.refit.conditional_log_likelihood)
            # Numerical ties do not cycle; no arbitrary full-fit time ceiling.
            tolerance = 32 * np.finfo(float).eps * max(1.0, abs(old), abs(new))
            if new > old + tolerance:
                current, changed = proposed, True
                yield proposed
                break
        if not changed:
            return
