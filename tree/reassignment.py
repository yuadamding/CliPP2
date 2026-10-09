"""Bounded fixed-K joint-membership proposals after the complete tree bank.

Unweighted, fixed-center likelihood gains rank transfers; only the unchanged
conditional refitter and mixture score decide path advancement. This is not
EM, a monotone likelihood iteration, or an exhaustive partition search.
"""

from time import perf_counter

import numpy as np


START_LIMIT = 4
ROUND_LIMIT = 2
PROPOSALS_PER_ROUND = 2
SCORE_ATOL = 1e-7
SCORE_RTOL = 1e-10


def _partition_key(candidate):
    """Avoid materializing labels for the original compact tree bank."""
    if getattr(candidate, "membership_kind", "tree") == "tree":
        return "tree", tuple(candidate.cuts)
    labels = candidate.refit.labels
    groups = {}
    return "general", tuple(groups.setdefault(int(k), len(groups)) for k in labels)


def _starting_states(original, winner):
    # Preserve the tree bank's ordering for exact score ties. No fitted state
    # is removed: distinct-membership selection only limits proposal starts.
    ordered = sorted(original, key=lambda c: (
        c.score, len(c.refit.centers), tuple(c.cuts)))
    representatives = {}
    for candidate in ordered:
        representatives.setdefault(len(candidate.refit.centers), candidate)
    chosen, seen = [], set()
    for candidate in (winner, *representatives.values(), *ordered):
        key = _partition_key(candidate)
        if key not in seen:
            chosen.append(candidate)
            seen.add(key)
        if len(chosen) == START_LIMIT:
            break
    return chosen


def _proposals(labels, centers, columns, observed, keys):
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
    group_keys = [(tuple(sorted(keys[i] for i in np.flatnonzero(labels == k))),
                   tuple(map(float, centers[k]))) for k in range(q)]
    # Compare the <=K full descriptors once, not during every transfer tie.
    # Equal descriptors share a rank so stable identical-row ties are unchanged.
    group_order = sorted(range(q), key=group_keys.__getitem__)
    group_ranks = [0] * q
    for previous, current in zip(group_order, group_order[1:]):
        group_ranks[current] = (group_ranks[previous]
                                + (group_keys[current] != group_keys[previous]))
    gains = columns - assigned[:, None]
    rows, destinations = np.nonzero(np.isfinite(gains) & (gains > 0))
    ranked = sorted(zip(rows.tolist(), destinations.tolist()), key=lambda move: (
        -float(gains[move]), keys[move[0]], group_ranks[move[1]], move[0]))

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
            np.asarray(getattr(candidate.refit, field)).nbytes
            for candidate in candidates for field in ("labels", "centers", "weights")),
        "membership_attempt_key_bytes": sum(
            len(labels) + (0 if centers is None else len(centers))
            for labels, centers in getattr(bank, "membership_attempts", {})),
        "structural_key_bytes": sum(map(len, getattr(bank, "structural_rejections", {}))),
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
        winner = min(original, key=lambda c: (c.score, len(c.refit.centers), tuple(c.cuts)))
    if not any(candidate is winner for candidate in original):
        raise ValueError("The original winner must belong to the original bank")
    starts = _starting_states(original, winner)
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
        paths.append(path)
        if q == 1:
            continue
        for round_index in range(ROUND_LIMIT):
            labels = np.asarray(state.refit.labels, dtype=np.int64)
            centers = np.asarray(state.refit.centers, dtype=float)
            columns = np.asarray(model.log_likelihood_columns(centers), dtype=float)
            if columns.shape != (model.n, q) or len(keys) != model.n:
                raise ValueError("Joint likelihood columns and scientific keys must match N by K")
            evaluations += 1
            proposals, scan = _proposals(labels, centers, columns, observed, keys)
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
                    "moved_mutations": int(np.count_nonzero(proposed != labels)),
                })
                record["outcome"] = "ineligible" if candidate is None else "eligible"
                if candidate is not None:
                    if len(candidate.refit.centers) != q or not np.isfinite(candidate.score):
                        raise ValueError("A reassignment callback changed K or returned a nonfinite score")
                    record["score"] = candidate.score
                    returned.append(candidate)
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
