"""Bounded fixed-K membership proposals aligned with the selected objective.

Classification ranks exact fixed-center read-plus-count gains with dynamic
counts/support after each greedy transfer. Legacy mixture and assigned-reads
comparators retain read-only ranking. Conditional refits and authoritative
scores decide path advancement; this is not EM or exhaustive optimization.
"""

from time import perf_counter
from collections import deque

import numpy as np


START_LIMIT = 4
ROUND_LIMIT = 2
PROPOSALS_PER_ROUND = 2
SCORE_ATOL = 1e-7
SCORE_RTOL = 1e-10


def _partition_key(candidate):
    """Avoid materializing labels for the original compact tree bank."""
    if getattr(candidate, "membership_kind", "tree") == "tree":
        return "tree", tuple(candidate.cuts), candidate.refit.clonal_cluster
    labels = candidate.refit.labels
    groups = {}
    canonical = tuple(groups.setdefault(int(k), len(groups)) for k in labels)
    return "general", canonical, groups[candidate.refit.clonal_cluster]


def _starting_states(original, winner, start_limit=START_LIMIT):
    # Preserve the tree bank's ordering for exact score ties. No fitted state
    # is removed: distinct-membership selection only limits proposal starts.
    ordered = sorted(original, key=lambda c: (
        c.score, len(c.refit.centers), _partition_key(c)))
    representatives = {}
    for candidate in ordered:
        representatives.setdefault(len(candidate.refit.centers), candidate)
    chosen, seen = [], set()
    for candidate in (winner, *representatives.values(), *ordered):
        key = _partition_key(candidate)
        if key not in seen:
            chosen.append(candidate)
            seen.add(key)
        if len(chosen) == start_limit:
            break
    return chosen


def _proposals(labels, centers, columns, observed, keys, *, support=None,
               objective="legacy_mixture"):
    """One ranked scan per proposal, with dynamic support in the greedy batch.

    Destination keys are scientific group multisets and center coordinates,
    not arbitrary label numbers. Mutation indices break only identical-row
    ties. A legal positive-gain transfer leaves occupied K and all regional
    source support intact. Missing-read CN boxes are already in ``columns``.
    """
    n, q = columns.shape
    if (labels.shape != (n,) or observed.shape != (n, centers.shape[1])
            or not np.array_equal(np.unique(labels), np.arange(q))):
        raise ValueError("Reassignment requires occupied aligned canonical labels")
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
    positive = zero = None
    if support is not None:
        summary = support.summarize_partition(labels)
        if support.check_partition(summary) is not None:
            raise ValueError("A reassignment parent has incompatible regional support")
        positive = summary.positive_counts.copy()
        zero = summary.hard_zero_counts.copy()
    group_keys = [(tuple(sorted(keys[i] for i in np.flatnonzero(labels == k))),
                   tuple(map(float, centers[k]))) for k in range(q)]
    # Compare the <=K full descriptors once, not during every transfer tie.
    # Equal descriptors share a rank so stable identical-row ties are unchanged.
    group_order = sorted(range(q), key=group_keys.__getitem__)
    group_ranks = [0] * q
    for previous, current in zip(group_order, group_order[1:]):
        group_ranks[current] = (group_ranks[previous]
                                + (group_keys[current] != group_keys[previous]))
    from ..kernel.objectives import transfer_count_gain, validate_selection_objective
    validate_selection_objective(objective)
    read_gains = np.asarray(columns, dtype=float)-assigned[:, None]

    gains = read_gains.copy()
    if objective == "classification":
        gains += transfer_count_gain(sizes, np.arange(q)[:, None], np.arange(q)[None, :])[labels]
    gains[np.arange(n), labels] = -np.inf
    rows, destinations = np.nonzero(np.isfinite(gains) & (gains > 0))
    ranked = sorted(zip(rows.tolist(), destinations.tolist()), key=lambda move: (
        -float(gains[move]), keys[move[0]], group_ranks[move[1]], move[0]))
    initial_pairs = len(ranked)

    def legal(row, source, destination):
        return (sizes[source] > 1 and np.all(counts[source] - observed[row] > 0)
                and (support is None or not np.any(
                    ((positive[destination]+support.positive[row]) > 0)
                    & ((zero[destination]+support.hard_zero[row]) > 0))))

    single, single_checks = None, 0
    for row, destination in ranked:
        single_checks += 1
        if legal(row, labels[row], destination):
            single = labels.copy()
            single[row] = destination
            break
    batch, moved = labels.copy(), np.zeros(n, dtype=bool)
    batch_checks, batch_gain, gain_updates = 0, 0., 0
    pair_rows, cursors = {}, {}
    if objective == "classification":
        # All transfers sharing a source/destination pair receive the SAME
        # count correction. Sort read gains once per pair; only its current
        # legal head competes after each dynamic count/support update. This
        # avoids re-sorting the full N-by-q matrix once per moved mutation.
        for source in range(q):
            for destination in range(q):
                if source == destination:
                    continue
                rows = np.flatnonzero((labels == source) & np.isfinite(read_gains[:, destination]))
                pair = source, destination
                pair_rows[pair] = sorted(rows.tolist(), key=lambda row: (
                    -float(read_gains[row, destination]), keys[row], row))
                cursors[pair] = 0
    else:
        ranked = deque(ranked)
    while ranked:
        chosen = None
        chosen_gain = None
        if objective == "classification":
            options = []
            count_gains = transfer_count_gain(sizes, np.arange(q)[:, None], np.arange(q)[None, :])
            for (source, destination), rows in pair_rows.items():
                if sizes[source] <= 1:
                    continue
                cursor = cursors[source, destination]
                while cursor < len(rows):
                    row = rows[cursor]
                    batch_checks += 1
                    if not moved[row] and legal(row, source, destination):
                        gain = float(read_gains[row, destination]+count_gains[source, destination])
                        if gain > 0:
                            options.append((-gain, keys[row], group_ranks[destination], row, destination))
                        break
                    cursor += 1
                cursors[source, destination] = cursor
            if options:
                negative_gain, _, _, row, destination = min(options)
                chosen, chosen_gain = (row, destination), -negative_gain
        else:
            while ranked:
                row, destination = ranked.popleft()
                if moved[row]:
                    continue
                batch_checks += 1
                if legal(row, labels[row], destination):
                    chosen, chosen_gain = (row, destination), float(gains[row, destination])
                    break
        if chosen is None:
            break
        row, destination = chosen
        source = labels[row]
        batch_gain += chosen_gain
        batch[row], moved[row] = destination, True
        sizes[source] -= 1
        sizes[destination] += 1
        counts[source] -= observed[row]
        counts[destination] += observed[row]
        if support is not None:
            positive[source] -= support.positive[row]
            positive[destination] += support.positive[row]
            zero[source] -= support.hard_zero[row]
            zero[destination] += support.hard_zero[row]
        # The count term changes every source/destination size preference.
        # Re-rank all remaining pairs after EACH accepted classification move;
        # static read-only ranks are valid only for the legacy comparator.
        if objective == "classification":
            # Previously inadmissible rows can become admissible when either
            # endpoint changes support. Revisit only those affected queues.
            for pair in cursors:
                if source in pair or destination in pair:
                    cursors[pair] = 0
            gain_updates += 1
    return (single, batch if moved.any() else None), {
        "scanned_transfer_pairs": n * (q - 1),
        "positive_gain_pairs": initial_pairs,
        "single_legal_checks": single_checks,
        "batch_legal_checks": batch_checks,
        "single_moves": int(single is not None),
        "batch_moves": int(moved.sum()),
        "ranking_objective": objective,
        "dynamic_count_updates": gain_updates,
        "batch_fixed_center_objective_gain": batch_gain,
        "stop_reason": "no_legal_positive_gain_transfer" if single is None else None,
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
            np.asarray(getattr(candidate.refit, field)).nbytes
            for candidate in candidates for field in ("labels", "centers", "weights")),
        "membership_attempt_key_bytes": sum(
            len(labels) + (0 if centers is None else len(centers))
            for labels, _, centers in getattr(bank, "membership_attempts", {})),
        "structural_key_bytes": sum(map(len, getattr(bank, "structural_rejections", {}))),
        "scope": "logical_array_and_key_payload_not_RSS_or_Python_overhead",
    }


def expand_memberships(model, bank, search=None):
    """Append configured starts x rounds x two general-membership callbacks.

    Every start includes the original bank's immutable winner. Absent,
    duplicate, rejected and non-improving proposals consume their slots;
    K=1 has no moves. Eligible states survive even without path advancement.
    """
    started = perf_counter()
    original = tuple(bank.candidates)
    if not original:
        raise ValueError("Reassignment requires an eligible original candidate bank")
    winner = getattr(bank, "original_selected", None)
    if winner is None:
        winner = min(original, key=lambda c: (c.score, len(c.refit.centers), _partition_key(c)))
    if not any(candidate is winner for candidate in original):
        raise ValueError("The original winner must belong to the original bank")
    start_limit = getattr(search, "reassignment_starts", START_LIMIT)
    round_limit = getattr(search, "reassignment_rounds", ROUND_LIMIT)
    starts = _starting_states(original, winner, start_limit)
    keys = tuple(model.observation_keys())
    observed = np.asarray(model.observed, dtype=bool)
    before = _counter_snapshot(model, bank)
    bytes_before = _retained_bytes((), bank)
    paths, slots, callbacks, evaluations, accepted = [], 0, 0, 0, 0
    for state in starts:
        q = len(state.refit.centers)
        path = {"start_bank_index": next(i for i, c in enumerate(original) if c is state),
                "occupied_q": q, "start_score": state.score, "rounds": [],
                "final_score": state.score,
                "skipped": "one_occupied_group" if q == 1 else None}
        path.update(stop_reason="one_occupied_group" if q == 1 else "round_budget_exhausted",
                    last_accepted_improvement=None, last_accepted_moved_mutations=0)
        paths.append(path)
        if q == 1:
            continue
        for round_index in range(round_limit):
            labels = np.asarray(state.refit.labels, dtype=np.int64)
            centers = np.asarray(state.refit.centers, dtype=float)
            clonal = state.refit.clonal_cluster
            if (not 0 <= clonal < q or not np.any(labels == clonal)
                    or not np.array_equal(centers[clonal], np.ones(model.r))):
                raise ValueError("Reassignment parent lost its occupied clonal center")
            columns = np.asarray(getattr(model, "proposal_log_likelihood_columns",
                                         model.log_likelihood_columns)(centers), dtype=float)
            if columns.shape != (model.n, q) or len(keys) != model.n:
                raise ValueError("Joint likelihood columns and scientific keys must match N by K")
            evaluations += 1
            support = getattr(model, "regional_support", None)
            constraints = {"support": support} if support is not None and support.hard_zero.any() else {}
            proposals, scan = _proposals(labels, centers, columns, observed, keys,
                objective=getattr(model, "selection_objective", "legacy_mixture"), **constraints)
            round_record = {"round": round_index, "parent_score": state.score,
                            **scan, "proposals": [], "advanced": False}
            path["rounds"].append(round_record)
            returned = []
            for kind, proposed in zip(("single", "greedy_batch"), proposals):
                slots += 1
                record = {"kind": kind, "callback": proposed is not None}
                round_record["proposals"].append(record)
                if proposed is None:
                    record["outcome"] = "no_legal_positive_gain_transfer"
                    continue
                callbacks += 1
                candidate = bank.consider_membership(proposed, centers, {
                    "route": "fixed_k_joint_reassignment", "proposal_kind": kind,
                    "requested_k": q, "occupied_q": q,
                    "starting_bank_index": path["start_bank_index"], "round": round_index,
                    "parent_score": state.score,
                    "parent_clonal_cluster": clonal,
                    "moved_mutations": int(np.count_nonzero(proposed != labels)),
                })
                record["outcome"] = "ineligible" if candidate is None else "eligible"
                if candidate is not None:
                    if len(candidate.refit.centers) != q or not np.isfinite(candidate.score):
                        raise ValueError("A reassignment callback changed K or returned a nonfinite score")
                    fixed = candidate.refit.clonal_cluster
                    if (not 0 <= fixed < q or not np.any(candidate.refit.labels == fixed)
                            or not np.array_equal(candidate.refit.centers[fixed], np.ones(model.r))):
                        raise ValueError("Reassignment callback lost its occupied clonal center")
                    record["score"] = candidate.score
                    returned.append(candidate)
            # Ties retain the first (single) proposal; both complete fitted
            # states remain in the bank even when improvement is sub-tolerance.
            best = min(returned, key=lambda c: c.score, default=None)
            if best is None:
                path["stop_reason"] = ("no_legal_positive_gain_transfer" if all(
                    proposal is None for proposal in proposals) else "no_generated_improving_move")
                break
            tolerance = SCORE_ATOL + SCORE_RTOL * max(abs(best.score), abs(state.score))
            round_record["advance_tolerance"] = tolerance
            if state.score - best.score <= tolerance:
                path["stop_reason"] = "no_generated_improving_move"
                break
            path["last_accepted_improvement"] = float(state.score-best.score)
            path["last_accepted_moved_mutations"] = int(np.count_nonzero(
                state.refit.labels != best.refit.labels))
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
    return {"applied": True, "method": "bounded_fixed_k_objective_aligned_reassignment_v2",
            "ranking_objective": getattr(model, "selection_objective", "legacy_mixture"),
            "starting_state_limit": start_limit, "round_limit": round_limit,
            "proposal_slots_per_round": PROPOSALS_PER_ROUND,
            "proposal_callback_limit": start_limit * round_limit * PROPOSALS_PER_ROUND,
            "proposal_joint_column_evaluation_limit": start_limit * round_limit,
            "score_atol": SCORE_ATOL, "score_rtol": SCORE_RTOL,
            "tie_rule": "gain_then_ID_free_row_and_destination_content_then_identical_row_index",
            "original_candidate_count": len(original), "original_winner_score": winner.score,
            "starting_states": len(starts), "proposal_slots": slots,
            "proposal_callbacks": callbacks, "proposal_joint_column_evaluations": evaluations,
            "membership_proposal_callbacks": callbacks,
            **{name: delta["model"].get(name, 0) for name in (
                "anchored_variant_evaluations", "free_block_refits", "mixture_weight_fits")},
            "accepted_steps": accepted, "added_candidates": len(bank.candidates) - len(original),
            "scan_counts": scans,
            "paths": paths, "counter_deltas": delta, "seconds": perf_counter() - started,
            "additional_retained_bytes": retained,
            "exhaustive_membership_search": False}
