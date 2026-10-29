"""Separate connected-boundary and bounded fixed-K membership proposal policies."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..kernel.topology import canonical_partition, partition_from_cuts


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


def best_pair_boundary(tree, labels, centers, log_columns, edge, *, adjacency=None, components=None,
                       column_rows=None):
    """Best replacement of one boundary, including both center orientations.

    Finite likelihood sums and impossible-assignment counts are accumulated
    separately. No subtraction involving infinite log likelihoods is used.
    Ties retain the current partition, then use canonical edge/orientation order.
    Returned labels keep the input cluster IDs so returned centers are aligned.
    """
    labels = np.asarray(labels, dtype=np.int64)
    centers = np.asarray(centers, dtype=float)
    columns = None if column_rows is not None else np.asarray(log_columns, dtype=float)
    edges = np.asarray(tree.edges, dtype=np.int64)
    if (labels.shape != (tree.n,) or (columns is not None and columns.shape != (tree.n, len(centers)))
            or (column_rows is not None and not callable(column_rows))):
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
        def adjacent(node):
            start, stop = tree.adjacency_offsets[node:node+2]
            return zip(tree.adjacent_nodes[start:stop], tree.adjacent_edge_ids[start:stop])
    else:
        adjacent = adjacency.__getitem__
    local = {int(node): [(neighbor, eid) for neighbor, eid in adjacent(node)
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

    pair_columns = (columns[np.ix_(members, [a, b])] if column_rows is None
                    else np.asarray(column_rows(members, [a, b]), dtype=float))
    if pair_columns.shape != (len(members), 2):
        raise ValueError("Boundary gather must return component-union rows and two columns")
    finite, impossible = _finite_terms(pair_columns)
    sums, counts = finite.copy(), impossible.copy()
    child_for_edge = {}
    for node in reversed(order[1:]):
        ancestor, eid = parent[node]
        sums[positions[ancestor]] += sums[positions[node]]
        counts[positions[ancestor]] += counts[positions[node]]
        child_for_edge[eid] = node
    current = pair_columns[np.arange(len(members)), (labels[members] == b).astype(int)]
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


def refine_partition(model, tree, candidate, consider, *, max_proposals=None, prepare=None,
                     device_proposals=False):
    """Yield every accepted local state; scoring is independent of acceptance.

    ``consider`` refits and records candidates, preserving supplied-center
    fallbacks. It must return a candidate with a ``fit`` result and ``cuts``.
    The path improves conditional likelihood, while the caller separately keeps
    the best observed-mixture score over the complete candidate bank.
    ``max_proposals`` optionally bounds exact proposal evaluations, including
    rejected moves. Existing native refinement uses the unlimited default.
    """
    if max_proposals is not None and (type(max_proposals) is not int or max_proposals < 0):
        raise ValueError("max_proposals must be a nonnegative integer or None")
    current = candidate
    visited = set()
    attempted = 0
    while True:
        changed = False
        fit = current.fit
        labels = fit.labels
        handles = (model.joint_column_set(fit.centers)
                   if prepare is not None and hasattr(model, "joint_column_set") else None)
        prepared = (None if prepare is None else
                    prepare(current.cuts, labels, fitted=fit, columns=handles))
        components = ({cluster: group.members for cluster, group in enumerate(prepared.groups)}
                      if prepared is not None and prepared.groups else
                      {cluster: np.flatnonzero(labels == cluster) for cluster in range(len(fit.centers))})
        # The model columns already enforce the complete regional assignment box.
        gather = (handles.subset_values if device_proposals and handles is not None
                  and getattr(getattr(model, "backend", None), "kind", None) == "cuda" else None)
        columns = (None if gather is not None else model.log_likelihood_columns(fit.centers)
                   if handles is None else handles.materialize())
        for edge in current.cuts:
            move = best_pair_boundary(tree, labels, fit.centers, columns, edge,
                                      components=components, column_rows=gather)
            if not move.changed:
                continue
            proposed_labels, centers = canonical_partition(move.labels, move.centers)
            cuts = tuple(np.flatnonzero(proposed_labels[tree.edges[:, 0]] != proposed_labels[tree.edges[:, 1]]))
            proposal_key = (cuts, centers.tobytes())
            if proposal_key in visited:
                continue
            if not np.array_equal(proposed_labels, partition_from_cuts(tree, cuts)):
                raise ValueError("Boundary move violated connected partition identity")
            if max_proposals is not None and attempted >= max_proposals:
                return
            visited.add(proposal_key)
            attempted += 1
            options = {} if prepared is None else {"parent": prepared}
            proposed = consider(cuts, proposed_labels, centers, {
                "route": "boundary_refinement", "parent_cuts": current.cuts,
                "replaced_edge": int(edge), "replacement_edge": move.edge,
            }, **options)
            if proposed is None:
                continue
            old = float(fit.conditional_log_likelihood)
            new = float(proposed.fit.conditional_log_likelihood)
            # Numerical ties do not cycle; no arbitrary full-fit time ceiling.
            tolerance = 32 * np.finfo(float).eps * max(1.0, abs(old), abs(new))
            if new > old + tolerance:
                current, changed = proposed, True
                yield proposed
                break
        if not changed:
            return


START_LIMIT = 4
ROUND_LIMIT = 2
PROPOSALS_PER_ROUND = 2
SCORE_ATOL = 1e-7
SCORE_RTOL = 1e-10


def _partition_key(candidate):
    """Avoid materializing labels for the original compact tree bank."""
    if getattr(candidate, "membership_kind", "tree") == "tree":
        return "tree", tuple(candidate.cuts)
    labels = candidate.fit.labels
    groups = {}
    return "general", tuple(groups.setdefault(int(k), len(groups)) for k in labels)


def _starting_states(original, winner):
    # Preserve the tree bank's ordering for exact score ties. No fitted state
    # is removed: distinct-membership selection only limits proposal starts.
    ordered = sorted(original, key=lambda c: (
        c.score, len(c.fit.centers), tuple(c.cuts)))
    representatives = {}
    for candidate in ordered:
        representatives.setdefault(len(candidate.fit.centers), candidate)
    chosen, seen = [], set()
    for candidate in (winner, *representatives.values(), *ordered):
        key = _partition_key(candidate)
        if key not in seen:
            chosen.append(candidate)
            seen.add(key)
        if len(chosen) == START_LIMIT:
            break
    return chosen


def _proposals(labels, centers, columns, observed, keys, *, positive=None):
    """One ranked scan per proposal, with dynamic support in the greedy batch.

    Destination keys are scientific group multisets and center coordinates,
    not arbitrary label numbers. Mutation indices break only identical-row
    ties. A legal positive-gain transfer leaves occupied K and all regional
    source support intact. Missing-read CN boxes are already in ``columns``.
    """
    n, q = (len(labels), len(centers)) if columns is None else columns.shape
    if (labels.shape != (n,) or observed.shape != (n, centers.shape[1])
            or not np.array_equal(np.unique(labels), np.arange(q))):
        raise ValueError("Reassignment requires occupied aligned canonical labels")
    if positive is None:
        if np.isnan(columns).any() or np.isposinf(columns).any():
            raise ValueError("Reassignment likelihood columns must be finite or -inf")
        assigned = columns[np.arange(n), labels]
        if not np.isfinite(assigned).all():
            raise ValueError("A reassignment parent has inadmissible assigned centers")
    sizes = np.bincount(labels, minlength=q)
    counts = np.zeros((q, observed.shape[1]), dtype=np.int64)
    np.add.at(counts, labels, observed)
    if np.any(counts == 0):
        raise ValueError("A reassignment parent lacks regional observation support")
    group_keys = [(tuple(sorted(keys[i] for i in np.flatnonzero(labels == k))),
                   tuple(map(float, centers[k]))) for k in range(q)]
    # Compare the <=K full descriptors once, not during every transfer tie.
    # Equal descriptors share a rank so stable identical-row ties are unchanged.
    group_order = sorted(range(q), key=group_keys.__getitem__)
    group_ranks = [0] * q
    for previous, current in zip(group_order, group_order[1:]):
        group_ranks[current] = (group_ranks[previous]
                                + (group_keys[current] != group_keys[previous]))
    if positive is None:
        gains = columns - assigned[:, None]
        rows, destinations = np.nonzero(np.isfinite(gains) & (gains > 0))
        positive_gains = gains[rows, destinations]
    else:
        rows, destinations, positive_gains = map(np.asarray, positive)
        if (rows.ndim != 1 or destinations.shape != rows.shape or positive_gains.shape != rows.shape
                or rows.dtype.kind not in "iu" or destinations.dtype.kind not in "iu"
                or np.any(rows < 0) or np.any(rows >= n) or np.any(destinations < 0)
                or np.any(destinations >= q) or not np.isfinite(positive_gains).all()
                or np.any(positive_gains <= 0) or np.any(rows[1:] < rows[:-1])
                or np.any((rows[1:] == rows[:-1]) & (destinations[1:] <= destinations[:-1]))):
            raise ValueError("Positive transfer vectors must be valid, unique and row-major")
    order = sorted(range(len(rows)), key=lambda index: (
        -float(positive_gains[index]), keys[rows[index]], group_ranks[destinations[index]], rows[index]))
    ranked = [(int(rows[index]), int(destinations[index])) for index in order]

    def legal(row, source, current_sizes, current_counts):
        return (current_sizes[source] > 1
                and np.all(current_counts[source] - observed[row] > 0))

    single, single_checks = None, 0
    for row, destination in ranked:
        single_checks += 1
        if legal(row, labels[row], sizes, counts):
            single = labels.copy()
            single[row] = destination
            break
    batch, moved = labels.copy(), np.zeros(n, dtype=bool)
    batch_checks = 0
    for row, destination in ranked:
        if moved[row]:
            continue
        batch_checks += 1
        source = labels[row]
        if not legal(row, source, sizes, counts):
            continue
        batch[row], moved[row] = destination, True
        sizes[source] -= 1
        sizes[destination] += 1
        counts[source] -= observed[row]
        counts[destination] += observed[row]
    return (single, batch if moved.any() else None), {
        "scanned_transfer_pairs": n * (q - 1),
        "positive_gain_pairs": len(ranked),
        "single_legal_checks": single_checks,
        "batch_legal_checks": batch_checks,
        "single_moves": int(single is not None),
        "batch_moves": int(moved.sum()),
    }


def _counter_snapshot(model, bank):
    return {
        "model": dict(getattr(model, "telemetry", {})),
        "bank": {key: getattr(bank, key, 0) for key in (
            "refit_calls", "attempt_cache_hits", "structural_cache_hits", "refit_seconds")},
    }


def _retained_bytes(candidates, bank):
    """Logical array/key payload only, not Python overhead, cache RAM, or RSS."""
    return {
        "new_candidate_numeric_array_bytes": sum(
            np.asarray(getattr(candidate.fit, field)).nbytes
            for candidate in candidates for field in ("labels", "centers", "weights")),
        "membership_attempt_key_bytes": sum(
            len(labels) + (0 if centers is None else len(centers))
            for labels, centers in getattr(bank, "membership_attempts", {})),
        "structural_key_bytes": sum(len(key) for kind, key in getattr(bank, "_structural", {})
                                    if kind == "general"),
        "scope": "logical_array_and_key_payload_not_RSS_or_Python_overhead",
    }


def expand_memberships(model, bank):
    """Append at most 16 general-membership proposal callbacks, in-place.

    The original bank is frozen for selecting <=4 starting states, including
    its exact winner. Each <=2-round path spends exactly two proposal slots
    per proposal-ranking joint-matrix evaluation (<=8); absent, duplicate,
    rejected, and non-improving proposals do not earn replacement slots. K=1 has no moves.
    Eligible states are retained even when their path does not advance.
    """
    started = perf_counter()
    original = tuple(bank.candidates)
    if not original:
        raise ValueError("Reassignment requires an eligible original candidate bank")
    winner = getattr(bank, "original_selected", None)
    if winner is None:
        winner = min(original, key=lambda c: (c.score, len(c.fit.centers), tuple(c.cuts)))
    if not any(candidate is winner for candidate in original):
        raise ValueError("The original winner must belong to the original bank")
    starts = _starting_states(original, winner)
    keys = tuple(model.observation_keys())
    observed = np.asarray(model.observed, dtype=bool)
    before = _counter_snapshot(model, bank)
    bytes_before = _retained_bytes((), bank)
    paths, slots, callbacks, evaluations, accepted = [], 0, 0, 0, 0
    for state in starts:
        q = len(state.fit.centers)
        path = {"start_bank_index": next(i for i, c in enumerate(original) if c is state),
                "occupied_q": q, "start_score": state.score, "rounds": [],
                "final_score": state.score,
                "skipped": "one_occupied_group" if q == 1 else None}
        paths.append(path)
        if q == 1:
            continue
        for round_index in range(ROUND_LIMIT):
            labels = np.asarray(state.fit.labels, dtype=np.int64)
            centers = np.asarray(state.fit.centers, dtype=float)
            handles = model.joint_column_set(centers) if hasattr(model, "joint_column_set") else None
            prepared = (bank.prepare(state.cuts, labels, fitted=state.fit, columns=handles)
                        if hasattr(bank, "prepare") else None)
            use_device = (getattr(bank, "device_proposals", False) and handles is not None
                          and getattr(getattr(model, "backend", None), "kind", None) == "cuda")
            positive = handles.positive_transfers(labels) if use_device else None
            columns = (None if use_device else np.asarray(model.log_likelihood_columns(centers), dtype=float)
                       if handles is None else handles.materialize())
            if (columns is not None and columns.shape != (model.n, q)) or len(keys) != model.n:
                raise ValueError("Joint likelihood columns and scientific keys must match N by K")
            evaluations += 1
            proposals, scan = (_proposals(labels, centers, columns, observed, keys) if positive is None
                               else _proposals(labels, centers, columns, observed, keys, positive=positive))
            round_record = {"round": round_index, "parent_score": state.score,
                            **scan, "proposals": [], "advanced": False}
            path["rounds"].append(round_record)
            returned = []
            independent = [] if getattr(bank, "weight_batch_size", 1) > 1 else None

            def record_result(record, candidate):
                record["outcome"] = "ineligible" if candidate is None else "eligible"
                if candidate is not None:
                    if len(candidate.fit.centers) != q or not np.isfinite(candidate.score):
                        raise ValueError("A reassignment callback changed K or returned a nonfinite score")
                    record["score"] = candidate.score
                    returned.append(candidate)

            for kind, proposed in zip(("single", "greedy_batch"), proposals):
                slots += 1
                record = {"kind": kind, "callback": proposed is not None}
                round_record["proposals"].append(record)
                if proposed is None:
                    record["outcome"] = "no_legal_positive_gain_transfer"
                    continue
                callbacks += 1
                options = {} if prepared is None else {"parent": prepared}
                provenance = {
                    "route": "fixed_k_joint_reassignment", "proposal_kind": kind,
                    "requested_k": q, "occupied_q": q,
                    "starting_bank_index": path["start_bank_index"], "round": round_index,
                    "parent_score": state.score,
                    "moved_mutations": int(np.count_nonzero(proposed != labels)),
                }
                if independent is not None:
                    independent.append((record, (None, proposed, centers, provenance, prepared)))
                else:
                    record_result(record, bank.consider_membership(proposed, centers, provenance, **options))
            if independent:
                fitted = bank.consider_independent([request for _, request in independent])
                for (record, _), candidate in zip(independent, fitted):
                    record_result(record, candidate)
            # Ties retain the first (single) proposal; both complete fitted
            # states remain in the bank even when improvement is sub-tolerance.
            best = min(returned, key=lambda c: c.score, default=None)
            if best is None:
                break
            tolerance = SCORE_ATOL + SCORE_RTOL * max(abs(best.score), abs(state.score))
            round_record["advance_tolerance"] = tolerance
            if state.score - best.score <= tolerance:
                break
            state = best
            round_record["advanced"] = True
            accepted += 1
        path["final_score"] = state.score
    after = _counter_snapshot(model, bank)
    delta = {group: {key: value - before[group].get(key, 0)
                     for key, value in values.items()} for group, values in after.items()}
    retained = _retained_bytes(bank.candidates[len(original):], bank)
    for key in ("membership_attempt_key_bytes", "structural_key_bytes"):
        retained[key] -= bytes_before[key]
    scans = {key: sum(record[key] for path in paths for record in path["rounds"])
             for key in ("scanned_transfer_pairs", "positive_gain_pairs",
                         "single_legal_checks", "batch_legal_checks")}
    return {"applied": True, "method": "bounded_fixed_k_joint_unweighted_reassignment_v1",
            "starting_state_limit": START_LIMIT, "round_limit": ROUND_LIMIT,
            "proposal_slots_per_round": PROPOSALS_PER_ROUND,
            "proposal_callback_limit": START_LIMIT * ROUND_LIMIT * PROPOSALS_PER_ROUND,
            "proposal_joint_column_evaluation_limit": START_LIMIT * ROUND_LIMIT,
            "score_atol": SCORE_ATOL, "score_rtol": SCORE_RTOL,
            "tie_rule": "gain_then_ID_free_row_and_destination_content_then_identical_row_index",
            "original_candidate_count": len(original), "original_winner_score": winner.score,
            "starting_states": len(starts), "proposal_slots": slots,
            "proposal_callbacks": callbacks, "proposal_joint_column_evaluations": evaluations,
            "accepted_steps": accepted, "added_candidates": len(bank.candidates) - len(original),
            "scan_counts": scans,
            "paths": paths, "counter_deltas": delta, "seconds": perf_counter() - started,
            "additional_retained_bytes": retained,
            "exhaustive_membership_search": False}
