from __future__ import annotations

import hashlib

import numpy as np

from ..core.fusion.types import RawFit
from ..core.fusion.partition_starts import PartitionCandidate
from ..core.scalar import canonical_partition_labels as _canonical_partition_labels
from ..core.fusion.types import (
    CompressedEdgeCertificate,
    DenseEdgeCertificate,
    PairwiseFusionGraph,
)
from .config import (
    LIKELIHOOD_PARTITION_K_MAX,
)
from .types import FusionPartition


def _partition_candidate_requested_k(candidate: PartitionCandidate) -> int:
    return int(candidate.K if candidate.requested_k is None else candidate.requested_k)


def _best_partition_candidate(
    candidates: list[PartitionCandidate],
) -> PartitionCandidate | None:
    finite_candidates = [
        candidate
        for candidate in candidates
        if bool(candidate.finite_candidate_found)
        and np.isfinite(float(candidate.bic))
        and np.isfinite(float(candidate.fit_loss))
    ]
    if not finite_candidates:
        return None
    return min(
        finite_candidates,
        key=lambda candidate: (
            float(candidate.bic),
            float(candidate.fit_loss),
            int(candidate.K),
            str(candidate.source),
        ),
    )


def _likelihood_partition_refinement_k_grid(
    candidates: list[PartitionCandidate],
    sparse_grid: list[int],
    *,
    num_mutations: int,
) -> tuple[list[int], str]:
    if not candidates or not sparse_grid:
        return [], "none"
    best = _best_partition_candidate(candidates)
    if best is None:
        return [], "none"

    k_cap = min(int(LIKELIHOOD_PARTITION_K_MAX), int(num_mutations))
    grid = sorted({int(k) for k in sparse_grid if 1 <= int(k) <= k_cap})
    if not grid:
        return [], "none"

    requested_k = int(np.clip(_partition_candidate_requested_k(best), 1, k_cap))
    effective_k = int(np.clip(int(best.K), 1, k_cap))
    focus_k = requested_k if requested_k in grid else effective_k
    if focus_k not in grid:
        focus_k = min(grid, key=lambda value: abs(int(value) - int(effective_k)))
    focus_idx = grid.index(int(focus_k))

    left_anchor = grid[focus_idx - 1] if focus_idx > 0 else int(focus_k)
    right_anchor = grid[focus_idx + 1] if focus_idx + 1 < len(grid) else int(focus_k)
    left_gap = int(focus_k) - int(left_anchor)
    right_gap = int(right_anchor) - int(focus_k)
    hits_k_cap = bool(k_cap == int(LIKELIHOOD_PARTITION_K_MAX) and effective_k >= k_cap)
    in_sparse_interval = bool(left_gap > 1 or right_gap > 1)

    if hits_k_cap and focus_idx > 0:
        left_anchor = grid[focus_idx - 1]
        right_anchor = int(focus_k)
        reason = "k_cap"
    elif in_sparse_interval:
        reason = "coarse_interval"
    else:
        return [], "none"

    refine_grid = [
        int(k)
        for k in range(int(left_anchor) + 1, int(right_anchor))
        if int(k) not in grid and 1 <= int(k) <= k_cap
    ]
    if not refine_grid:
        return [], "none"
    return refine_grid, reason


def _deduplicate_partition_candidates(
    candidates: list[PartitionCandidate],
) -> list[PartitionCandidate]:
    best_by_signature: dict[str, PartitionCandidate] = {}
    for candidate in candidates:
        signature = _partition_signature(candidate.labels)
        current = best_by_signature.get(signature)
        if current is None or (
            float(candidate.bic),
            float(candidate.fit_loss),
            int(candidate.K),
            str(candidate.source),
        ) < (
            float(current.bic),
            float(current.fit_loss),
            int(current.K),
            str(current.source),
        ):
            best_by_signature[signature] = candidate
    return sorted(
        best_by_signature.values(),
        key=lambda candidate: (
            float(candidate.bic),
            int(candidate.K),
            str(candidate.source),
        ),
    )


def _partition_blocks(labels: np.ndarray) -> tuple[tuple[int, ...], ...]:
    canonical = _canonical_partition_labels(labels)
    if canonical.size == 0:
        return ()
    blocks = [
        tuple(int(idx) for idx in np.flatnonzero(canonical == int(label)).tolist())
        for label in np.unique(canonical)
    ]
    return tuple(sorted(blocks))


def _partition_signature(
    labels: np.ndarray,
    mutation_ids: tuple[str, ...] | list[str] | None = None,
) -> str:
    blocks = _partition_blocks(labels)
    if not blocks:
        return "empty"
    hasher = hashlib.blake2b(digest_size=12)
    if mutation_ids is None:
        identity_blocks: list[tuple[str, ...] | tuple[int, ...]] = list(blocks)
    else:
        identities = tuple(str(value) for value in mutation_ids)
        if len(identities) != np.asarray(labels).size:
            raise ValueError("mutation_ids must match the partition label count.")
        if len(set(identities)) != len(identities):
            raise ValueError("mutation_ids must be unique for partition identity.")
        identity_blocks = sorted(
            tuple(sorted(identities[index] for index in block)) for block in blocks
        )
    for block in identity_blocks:
        hasher.update(np.asarray([len(block)], dtype=np.int64).tobytes())
        for value in block:
            encoded = str(value).encode("utf-8")
            hasher.update(len(encoded).to_bytes(8, "little"))
            hasher.update(encoded)
    return f"{len(blocks)}:{hasher.hexdigest()}"


def _exact_partition_diameters(phi: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.asarray(phi, dtype=np.float64)
    canonical = _canonical_partition_labels(labels)
    n_clusters = int(canonical.max()) + 1 if canonical.size else 0
    diameters = np.zeros(n_clusters, dtype=np.float64)
    for cluster in range(n_clusters):
        rows = values[canonical == cluster]
        if rows.shape[0] <= 1:
            continue
        if rows.shape[1] == 1:
            diameters[cluster] = float(np.max(rows[:, 0]) - np.min(rows[:, 0]))
            continue
        maximum = 0.0
        for start in range(0, rows.shape[0], 512):
            block = rows[start : start + 512]
            distances = np.linalg.norm(block[:, None, :] - rows[None, :, :], axis=-1)
            if distances.size:
                maximum = max(maximum, float(np.max(distances)))
        diameters[cluster] = maximum
    return diameters


def _maximum_cross_distance(
    phi: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    stop_above: float,
) -> float:
    """Return the exact maximum cross-block distance with bounded workspace."""

    if left.size == 0 or right.size == 0:
        return 0.0
    values = np.asarray(phi, dtype=np.float64)
    # Limit the temporary ``left x right x region`` tensor.  This matters when a
    # nearly fused tumor contains one very large block.
    regions = max(int(values.shape[1]), 1)
    rows_per_chunk = max(1, (8 * 1024 * 1024) // max(right.size * regions * 8, 1))
    maximum = 0.0
    for start in range(0, int(left.size), rows_per_chunk):
        chunk = left[start : start + rows_per_chunk]
        distances = np.linalg.norm(
            values[chunk, None, :] - values[None, right, :],
            axis=2,
        )
        if distances.size:
            maximum = max(maximum, float(np.max(distances)))
            if maximum > float(stop_above):
                break
    return maximum


def _diameter_constrained_labels(
    phi: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """Build a deterministic, merge-maximal complete-link partition.

    A block is admissible exactly when its Euclidean diameter is no larger than
    ``tolerance``.  Mutations are visited in lexicographic raw-CCF order and put
    in the first block for which *every* pair remains in tolerance.  A block
    created later has a seed that failed every earlier block and block growth
    cannot reduce that distance, so the result is pairwise merge-maximal.  The
    Each mutation follows the identical complete-link admission rule.
    """

    values = np.asarray(phi, dtype=np.float64)
    num_mutations = int(values.shape[0])
    if num_mutations == 0:
        return np.zeros(0, dtype=np.int64)

    blocks: list[list[int]] = []
    remaining = np.arange(num_mutations, dtype=np.int64)
    if remaining.size:
        # ``np.lexsort`` uses the last key as primary.  Mutation index is the
        # final tie-breaker, so equal raw CCF rows remain reproducible.
        sort_keys: list[np.ndarray] = [remaining]
        sort_keys.extend(
            values[remaining, region]
            for region in range(int(values.shape[1]) - 1, -1, -1)
        )
        remaining = remaining[np.lexsort(tuple(sort_keys))]

    for index in remaining:
        assigned = False
        for block in blocks:
            block_indices = np.asarray(block, dtype=np.int64)
            distances = np.linalg.norm(
                values[block_indices] - values[int(index)],
                axis=1,
            )
            if distances.size and bool(
                np.all(np.isfinite(distances))
                and float(np.max(distances)) <= float(tolerance)
            ):
                block.append(int(index))
                assigned = True
                break
        if not assigned:
            blocks.append([int(index)])

    labels = np.empty(num_mutations, dtype=np.int64)
    for block_label, block in enumerate(blocks):
        labels[np.asarray(block, dtype=np.int64)] = int(block_label)
    return _canonical_partition_labels(labels)


def _mergeable_cross_block_pair_found(
    phi: np.ndarray,
    labels: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    """Return whether two reported blocks can be merged without excess diameter."""

    canonical = _canonical_partition_labels(labels)
    blocks = [
        np.flatnonzero(canonical == cluster).astype(np.int64, copy=False)
        for cluster in range(int(canonical.max()) + 1 if canonical.size else 0)
    ]
    for left_index, left in enumerate(blocks):
        for right in blocks[left_index + 1 :]:
            if _maximum_cross_distance(
                phi,
                left,
                right,
                stop_above=float(tolerance),
            ) <= float(tolerance):
                return True
    return False


def extract_certified_fusion_partition(
    fit: RawFit,
    *,
    graph: PairwiseFusionGraph,
    tolerance: float,
    mutation_ids: tuple[str, ...] | list[str] | None = None,
) -> FusionPartition:
    """Extract one deterministic, diameter-constrained raw-fusion partition."""

    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("Partition tolerance must be positive and finite.")
    phi = np.asarray(fit.phi, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError("fit.phi must be a mutation-by-region matrix.")

    labels = _diameter_constrained_labels(phi, tolerance=tol)
    if labels.shape != (phi.shape[0],):
        raise ValueError("Partition labels do not match the raw fit mutation count.")
    diameters = _exact_partition_diameters(phi, labels)
    max_diameter = float(np.max(diameters)) if diameters.size else 0.0
    within_ok = bool(np.all(np.isfinite(diameters)) and max_diameter <= tol)
    # ``cross_close`` retains its schema name but now has the precise complete-
    # linkage meaning: two whole blocks, rather than merely one chaining edge,
    # can be combined while preserving the diameter contract.
    cross_close = _mergeable_cross_block_pair_found(
        phi,
        labels,
        tolerance=tol,
    )
    state = fit.state
    certificate = getattr(state, "certificate", None)
    certificate_graph_hash_matches = True
    if isinstance(certificate, (CompressedEdgeCertificate, DenseEdgeCertificate)):
        expected_graph_hash = str(fit.provenance.original_graph_hash)
        certificate_graph_hash_matches = bool(
            expected_graph_hash and str(certificate.graph_hash) == expected_graph_hash
        )
    certified = bool(
        within_ok
        and not cross_close
        and certificate_graph_hash_matches
    )
    if not certificate_graph_hash_matches:
        failure_reason = (
            "compressed_certificate_graph_hash_mismatch"
            if isinstance(certificate, CompressedEdgeCertificate)
            else "dense_certificate_graph_hash_mismatch"
        )
    elif not within_ok:
        failure_reason = "raw_partition_chaining_or_solver_tolerance"
    elif cross_close:
        failure_reason = "mergeable_cross_block_pair_within_partition_tolerance"
    else:
        failure_reason = "none"
    return FusionPartition(
        labels=labels.astype(np.int64, copy=False),
        signature=_partition_signature(labels, mutation_ids),
        certified=certified,
        source="tolerance_defined_primal",
        certification_failure_reason=str(failure_reason),
        mutation_ids=() if mutation_ids is None else tuple(mutation_ids),
    )


def extract_connected_component_partition(
    fit: RawFit,
    *,
    graph: PairwiseFusionGraph,
    tolerance: float,
    mutation_ids: tuple[str, ...] | list[str] | None = None,
) -> FusionPartition:
    """Extract the declared legacy threshold-connectivity raw summary.

    This is a partition summary of the certified raw matrix, not a replacement
    raw optimizer. It is available only through the explicit legacy contract
    because transitive chains may have diameters larger than ``tolerance``.
    """

    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("Partition tolerance must be positive and finite.")
    phi = np.asarray(fit.phi, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError("fit.phi must be a mutation-by-region matrix.")
    num_mutations = int(phi.shape[0])
    expected_edges = num_mutations * max(num_mutations - 1, 0) // 2
    edge_u = np.asarray(graph.edge_u, dtype=np.int64)
    edge_v = np.asarray(graph.edge_v, dtype=np.int64)
    complete_graph = bool(
        edge_u.size == expected_edges
        and edge_v.size == expected_edges
        and int(graph.degree_bound) == max(num_mutations - 1, 0)
    )
    parent = np.arange(num_mutations, dtype=np.int64)
    rank = np.zeros(num_mutations, dtype=np.int8)

    def find(value: int) -> int:
        root = int(value)
        while int(parent[root]) != root:
            parent[root] = parent[int(parent[root])]
            root = int(parent[root])
        return root

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    for start in range(0, int(edge_u.size), 1_000_000):
        stop = min(start + 1_000_000, int(edge_u.size))
        left = edge_u[start:stop]
        right = edge_v[start:stop]
        distances = np.linalg.norm(phi[left] - phi[right], axis=1)
        for left_value, right_value in zip(
            left[distances <= tol], right[distances <= tol], strict=True
        ):
            union(int(left_value), int(right_value))
    labels = _canonical_partition_labels(
        np.asarray([find(index) for index in range(num_mutations)], dtype=np.int64)
    )
    state = fit.state
    certificate = None if state is None else state.certificate
    certificate_graph_hash_matches = True
    if isinstance(certificate, (CompressedEdgeCertificate, DenseEdgeCertificate)):
        expected_graph_hash = str(fit.provenance.original_graph_hash)
        certificate_graph_hash_matches = bool(
            expected_graph_hash and str(certificate.graph_hash) == expected_graph_hash
        )
    certified = bool(complete_graph and certificate_graph_hash_matches)
    failure_reason = (
        "none"
        if certified
        else (
            "legacy_connected_component_requires_complete_graph"
            if not complete_graph
            else "raw_certificate_graph_hash_mismatch"
        )
    )
    return FusionPartition(
        labels=labels,
        signature=_partition_signature(labels, mutation_ids),
        certified=certified,
        source="legacy_connected_components",
        certification_failure_reason=str(failure_reason),
        mutation_ids=() if mutation_ids is None else tuple(mutation_ids),
    )
