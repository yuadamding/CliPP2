"""Bounded support-compatible starts and splits; no likelihood or score changes.

Missing cells are unknown. Aggregate positive/zero masks, not pairwise chains,
define compatibility. Construction failure is never an infeasibility proof.
"""

import numpy as np

from .refinement import canonical_partition


SEED_ATTEMPT_LIMIT = 2


def active_support(model):
    support = getattr(model, "regional_support", None)
    return support if support is not None and support.hard_zero.any() else None


def support_preflight(model, maximum):
    """Certify only a lower bound and the necessary collective clonal support."""
    support = active_support(model)
    result = {"support_lower_bound": 1, "support_lower_bound_certified": True,
              "support_lower_bound_method": "no_active_constraints"}
    if support is None:
        return result
    patterns = np.unique(np.where(support.positive, 1,
                                  np.where(support.hard_zero, -1, 0)), axis=0)
    if support.observed.all():
        lower = len(patterns)
        method = "distinct_complete_detection_patterns"
    else:
        # Each accepted representative conflicts with EVERY earlier one. A
        # greedy clique is a certified bound, not the chromatic number; unknown
        # bridges never make distinct ternary-pattern count a valid bound.
        representatives = []
        for row in sorted(patterns.tolist(), key=lambda row: (-np.count_nonzero(row), tuple(row))):
            row = np.asarray(row)
            if all(np.any(row * old == -1) for old in representatives):
                representatives.append(row)
                if len(representatives) > maximum:
                    break
        lower, method = len(representatives), "greedy_mutually_conflicting_pattern_clique"
    result.update(support_lower_bound=lower, support_lower_bound_method=method)
    if lower > maximum:
        raise ValueError(f"support_capacity_infeasible: certified lower bound {lower} exceeds max_clusters {maximum}")
    eligible = np.asarray(model.clonal_eligible) & ~support.hard_zero.any(axis=1)
    if not eligible.any() or not support.observed[eligible].any(axis=0).all():
        raise ValueError("no_zero_compatible_clonal_block: eligible mutations lack collective regional support")
    return result


def _compatible(positive, zero):
    return not np.any((positive > 0) & (zero > 0))


def support_seed_proposals(model, maximum, *, limit=SEED_ATTEMPT_LIMIT):
    """At most two deterministic constructions, with bounded K-group placement.

    Complete cases group by detection pattern. Partial cases reserve a supported
    clonal group then greedily place remaining rows using evolving counts. The
    two row orders are a heuristic, not a completeness guarantee. Failed starts
    are returned as diagnostics so their construction work remains visible.
    """
    support = active_support(model)
    if support is None or limit <= 0:
        return
    eligible = np.asarray(model.clonal_eligible) & ~support.hard_zero.any(axis=1)
    pattern = np.where(support.positive, 1, np.where(support.hard_zero, -1, 0))
    if support.observed.all():
        _, labels = np.unique(pattern, axis=0, return_inverse=True)
        # The all-positive pattern can contain infinite clonal emissions; split
        # it only when needed to isolate the finite supported designation.
        if eligible.any() and any(not eligible[labels == k].all() for k in np.unique(labels[eligible])):
            labels[eligible] = labels.max()+1
        labels = canonical_partition(labels)
        reason = None if labels.max()+1 <= maximum else "support_seed_search_exhausted"
        yield (labels if reason is None else None), {
            "construction": "complete_detection_patterns", "construction_index": 0,
            "rows_screened": model.n, "placement_checks": 0, "reason": reason}
        return
    for attempt in range(min(limit, SEED_ATTEMPT_LIMIT)):
        labels = np.full(model.n, -1, dtype=np.int64)
        labels[eligible] = 0
        positive = [support.positive[eligible].sum(axis=0)]
        zero = [support.hard_zero[eligible].sum(axis=0)]
        observed = [support.observed[eligible].sum(axis=0)]
        remaining = np.flatnonzero(~eligible)
        order = sorted(remaining, key=lambda i: (
            -int(support.observed[i].sum()), tuple((1 if attempt == 0 else -1)*pattern[i]), int(i)))
        checks = 0
        reason = None
        for i in order:
            choices = []
            for k in range(1, len(positive)):
                checks += 1
                if _compatible(positive[k]+support.positive[i], zero[k]+support.hard_zero[i]):
                    choices.append((-int(np.count_nonzero((observed[k] == 0) & support.observed[i])), k))
            if choices:
                k = min(choices)[1]
                positive[k] += support.positive[i]
                zero[k] += support.hard_zero[i]
                observed[k] += support.observed[i]
            else:
                if len(positive) == maximum:
                    reason = "support_seed_search_exhausted"
                    break
                k = len(positive)
                positive.append(support.positive[i].astype(np.int64))
                zero.append(support.hard_zero[i].astype(np.int64))
                observed.append(support.observed[i].astype(np.int64))
            labels[i] = k
        repairs = 0
        if reason is None:
            # Unknown-only coordinates can be supplied by eligible rows now in
            # the clonal block, provided its regional evidence stays occupied.
            # One pass per missing block-coordinate is deliberately bounded.
            for k in range(1, len(positive)):
                for region in np.flatnonzero(observed[k] == 0):
                    if observed[k][region] > 0:
                        continue
                    for i in np.flatnonzero((labels == 0) & support.observed[:, region]):
                        checks += 1
                        if (np.all(observed[0]-support.observed[i] > 0)
                                and _compatible(positive[k]+support.positive[i],
                                                zero[k]+support.hard_zero[i])):
                            labels[i] = k
                            for counts, values in ((positive, support.positive), (zero, support.hard_zero),
                                                   (observed, support.observed)):
                                counts[0] -= values[i]
                                counts[k] += values[i]
                            repairs += 1
                            break
            labels = canonical_partition(labels)
            if support.check_partition(support.summarize_partition(labels)) is not None:
                reason = "support_seed_search_exhausted"
        yield (labels if reason is None else None), {
            "construction": "partial_aggregate_greedy", "construction_index": attempt,
            "rows_screened": len(order), "placement_checks": checks,
            "placement_check_limit": model.n*maximum*(model.r+1),
            "evidence_repair_moves": repairs, "reason": reason}


def _overlap_connected(observed):
    """Connectivity using only genuine shared regional evidence, O(NR)."""
    parent = np.arange(len(observed))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for region in range(observed.shape[1]):
        rows = np.flatnonzero(observed[:, region])
        if len(rows):
            target = root(int(rows[0]))
            for row in rows[1:]:
                parent[root(int(row))] = target
    return len({root(i) for i in range(len(parent))}) == 1


def _profile_modes(model, region, grid, profiles, cache_limit):
    """Stream coarse columns; reuse only bounded O(N) modal coordinates."""
    if region in profiles:
        return profiles[region]
    best, modes = np.full(model.n, -np.inf), np.zeros(model.n)
    width = max(1, min(8, (8*1024*1024)//(16*model.n)))
    for first in range(0, len(grid), width):
        block = grid[first:first+width]
        values = model.regional_profile_columns(region, block)
        if np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError("Proposal profiles must be finite or negative infinity")
        selected = np.argmax(values, axis=1)
        likelihood = values[np.arange(model.n), selected]
        improve = likelihood > best
        best[improve], modes[improve] = likelihood[improve], block[selected[improve]]
    if modes.nbytes <= cache_limit:
        while profiles and sum(value.nbytes for value in profiles.values())+modes.nbytes > cache_limit:
            profiles.pop(next(iter(profiles)))
        profiles[region] = modes
    return modes


def support_split_proposals(model, labels, block, budget, *, profiles=None):
    """Rank bounded observed-profile cuts within a general-membership block.

    No new model or artificial missing-cell distance is constructed. Unknown
    rows are placed wholly on either side in two explicit alternatives. Blocks
    with disconnected regional overlap have no distance-based split route.
    Every tested child must retain all-region evidence and compatibility.
    """
    support = active_support(model)
    if support is None or budget <= 0:
        return [], {"screened": 0, "reason": "no_active_support"}
    rows = np.flatnonzero(labels == block)
    if len(rows) < 2 or not _overlap_connected(support.observed[rows]):
        return [], {"screened": 0, "reason": "disconnected_or_singleton_block"}
    profiles = {} if profiles is None else profiles
    cache_limit = min(8*1024*1024, getattr(getattr(model, "memory", None), "workspace_bytes", 0)//4)
    grid = np.linspace(0., 1., 17)
    orders = []
    screen_limit = 4*budget
    for region in range(model.r):
        observed = rows[support.observed[rows, region]]
        if len(observed) < 2 or support.hard_zero[observed, region].any():
            continue
        modes = _profile_modes(model, region, grid, profiles, cache_limit)[observed]
        ordering = np.lexsort((observed, modes))
        ordered, modes = observed[ordering], modes[ordering]
        gaps = np.diff(modes)
        # Only observed gaps rank a split; no value is assigned to missing rows.
        positions = sorted(np.flatnonzero(gaps > 0), key=lambda i: (-gaps[i], int(i)))[:budget]
        for position in positions:
            orders.append((-float(gaps[position]), region, ordered, int(position)+1))
        # Retain only the prefix this call can inspect. In particular, never
        # keep an N-by-grid-by-regions emission tensor or all regional orders.
        orders.sort(key=lambda item: (item[0], item[1], item[3]))
        del orders[screen_limit:]
    proposals, seen, screened = [], set(), 0
    for gap, region, ordered, position in orders:
        missing = rows[~support.observed[rows, region]]
        for missing_left in range(2 if len(missing) else 1):
            if screened >= screen_limit or len(proposals) >= budget:
                break
            screened += 1
            left = np.r_[ordered[:position], missing if missing_left else np.empty(0, dtype=int)]
            proposed = labels.copy()
            proposed[left] = int(labels.max())+1
            proposed = canonical_partition(proposed)
            key = proposed.tobytes()
            if key in seen:
                continue
            seen.add(key)
            if support.check_partition(support.summarize_partition(proposed)) is None:
                proposals.append((proposed, {"split_block": int(block), "ordering_region": region,
                    "profile_gap": -gap, "missing_rows_side": "left" if missing_left else "right"}))
        if screened >= screen_limit or len(proposals) >= budget:
            break
    return proposals, {"screened": screened, "screen_limit": screen_limit,
                       "regional_orderings": len(orders), "available": len(proposals),
                       "profile_column_block": max(1, min(8, (8*1024*1024)//(16*model.n))),
                       "profile_mode_cache_limit_bytes": cache_limit,
                       "retained_ordering_limit": screen_limit,
                       "ranking": "observed_regional_coarse_profile_mode_gap"}
