"""Bounded support-compatible starts and splits; no likelihood or score changes.

Missing cells are unknown. Aggregate positive/zero masks, not pairwise chains,
define compatibility. Construction failure is never an infeasibility proof.
"""

import numpy as np

from ..io.data import readonly_array
from .refinement import canonical_partition


SEED_ATTEMPT_LIMIT = 2
PROFILE_POLICY = "observed-coarse-modes-v1"


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


def support_seed_proposals(model, maximum, *, limit=SEED_ATTEMPT_LIMIT, support=None):
    """At most two deterministic constructions, with bounded K-group placement.

    Complete cases group by detection pattern. Partial cases reserve a supported
    clonal group then greedily place remaining rows using evolving counts. The
    two row orders are a heuristic, not a completeness guarantee. Failed starts
    are returned as diagnostics so their construction work remains visible.
    """
    support = active_support(model) if support is None else support
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


def _profile_modes(model, region, grid, profiles=None, cache_limit=0):
    """Stream columns; immutable fit-scope modes share the column-cache budget."""
    grid = np.asarray(grid, dtype=np.float64)
    key = (model.identity, int(region), grid.dtype.str, grid.shape,
           grid.tobytes(), PROFILE_POLICY)
    store = getattr(model, "_proposal_profiles", None) if profiles is None else None
    cached = store.get(key) if store is not None else (profiles or {}).get(key)
    if cached is not None:
        return cached
    best, modes = np.full(model.n, -np.inf), np.zeros(model.n)
    width = max(1, min(8, (8*1024*1024)//(16*model.n)))
    for first in range(0, len(grid), width):
        block = grid[first:first+width]
        values = model.regional_profile_columns(region, block)
        model.telemetry["proposal_profile_columns"] += len(block)
        if np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError("Proposal profiles must be finite or negative infinity")
        selected = np.argmax(values, axis=1)
        likelihood = values[np.arange(model.n), selected]
        improve = likelihood > best
        best[improve], modes[improve] = likelihood[improve], block[selected[improve]]
    modes = readonly_array(modes)
    if store is not None:
        store.put(key, modes, modes.nbytes+grid.nbytes+512)
    elif profiles is not None and modes.nbytes <= cache_limit:
        while profiles and sum(value.nbytes for value in profiles.values())+modes.nbytes > cache_limit:
            profiles.pop(next(iter(profiles)))
        profiles[key] = modes
    return modes


def _within_bin_coordinates(model, rows, region, modes, pilot):
    """Observed continuous ordering, never a fitted center or scoring change.

    Single-multiplicity modes are exact. Otherwise use the observed pilot,
    falling back to the component mode nearest the coarse profile mode. This
    removes the coarse-grid gate without substituting a denser fixed grid.
    """
    ratio = model.alt[rows, region]/(model.depth[rows, region]*model.scaling[rows, region])
    multiplicity = np.ones(len(rows))
    np.divide(ratio, modes, out=multiplicity, where=modes > 0)
    multiplicity = np.clip(np.rint(multiplicity), 1, model.support[rows, region])
    component = np.clip(ratio/multiplicity, model.lower[rows, region], model.upper[rows, region])
    values = component.copy()
    if pilot is not None:
        proposed = np.asarray(pilot)[rows, region]
        use = np.isfinite(proposed) & (model.support[rows, region] > 1)
        values[use] = proposed[use]
    # Pilots may themselves use a coarse grid. Keep the continuous component
    # coordinate as a secondary ordering so pilot ties cannot erase variation.
    return values, component


def _observation_components(rows, observed):
    """Connected observation-overlap components, without cross-component edges."""
    if observed[rows].all():
        return (rows,)
    parent = np.arange(len(rows))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for region in range(observed.shape[1]):
        local = np.flatnonzero(observed[rows, region])
        if len(local):
            target = root(int(local[0]))
            for row in local[1:]:
                parent[root(int(row))] = target
    groups = {}
    for local, row in enumerate(rows):
        groups.setdefault(root(local), []).append(row)
    return tuple(np.asarray(group) for group in groups.values())


def _component_split_sides(components, orderings, support, limit):
    """Bounded relative-rank alignment; not cross-component identification.

    One construction per attempt, never a Cartesian product of component cuts.
    Whole-side placement handles unknown cells within a component; one relative
    flip varies the unconstrained alignment between components.
    """
    for attempt in range(limit):
        rank, variant = divmod(attempt, 4)
        flip = 1+rank % (len(components)-1) if variant % 2 else -1
        pieces, regions, gaps = [], [], []
        for index, (rows, choices) in enumerate(zip(components, orderings)):
            _, gap, region, ordered, position = choices[rank % len(choices)]
            missing = rows[~support.observed[rows, region]]
            left = np.r_[ordered[:position], missing if variant >= 2 else np.empty(0, dtype=int)]
            if index == flip:
                left = rows[~np.isin(rows, left)]
            pieces.append(left)
            regions.append(int(region))
            gaps.append(-gap)
        yield np.concatenate(pieces), {
            "profile_gap": min(gaps), "profile_ordering": "componentwise",
            "ordering_region": -1, "component_ordering_regions": regions,
            "component_rank": rank, "component_relative_flip": flip,
            "missing_rows_side": "left" if variant >= 2 else "right",
            "component_alignment": "bounded_relative_rank_heuristic_not_identified"}


def support_split_proposals(model, labels, block, budget, *, profiles=None, pilot=None,
                            support=None):
    """Rank bounded observed-profile cuts within a general-membership block.

    No new model or artificial missing-cell distance is constructed. Unknown
    rows are placed wholly on either side in two explicit alternatives. No
    cross-component distance is inferred. Every tested child must retain
    all-region evidence and compatibility, including in disconnected forests.
    """
    support = active_support(model) if support is None else support
    if support is None or budget <= 0:
        return [], {"screened": 0, "reason": "no_active_support"}
    rows = np.flatnonzero(labels == block)
    if len(rows) < 2:
        return [], {"screened": 0, "reason": "singleton_block"}
    components = _observation_components(rows, support.observed)
    disconnected = len(components) > 1
    if disconnected and any(len(component) < 2 for component in components):
        return [], {"screened": 0, "reason": "singleton_observation_component",
                    "observation_components": len(components)}
    component_of = ({int(row): index for index, group in enumerate(components) for row in group}
                    if disconnected else {})
    component_orders = [[] for _ in components]
    cache_limit = (min(8*1024*1024, getattr(getattr(model, "memory", None), "workspace_bytes", 0)//4)
                   if profiles is not None else getattr(getattr(model, "_proposal_profiles", None),
                                                         "budget_bytes", 0))
    grid = np.linspace(0., 1., 17)
    orders = []
    screen_limit = 4*budget
    component_constructions = 0
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
        regional = []
        for position in positions:
            regional.append((0, -float(gaps[position]), region, ordered, int(position)+1))
        # A positive coarse gap is NOT necessary: continuous ordering inside a
        # tied bin supplies bounded alternatives to the same refitter/scorer.
        coordinates, secondary = _within_bin_coordinates(model, ordered, region, modes, pilot)
        ordering = np.lexsort((ordered, secondary, coordinates, modes))
        fine_rows, fine_modes, fine = ordered[ordering], modes[ordering], coordinates[ordering]
        fine_gaps = np.diff(fine)
        fine_gaps = np.where(fine_gaps == 0, np.diff(secondary[ordering]), fine_gaps)
        positions = np.flatnonzero((np.diff(fine_modes) == 0) & (fine_gaps > 0))
        for position in sorted(positions, key=lambda i: (-fine_gaps[i], int(i)))[:budget]:
            regional.append((1, -float(fine_gaps[position]), region, fine_rows, int(position)+1))
        # Retain only the prefix this call can inspect. In particular, never
        # keep an N-by-grid-by-regions emission tensor or all regional orders.
        target = component_orders[component_of[int(observed[0])]] if disconnected else orders
        component_constructions += len(regional) if disconnected else 0
        target.extend(regional)
        target.sort(key=lambda item: (item[0], item[1], item[2], item[4]))
        del target[budget if disconnected else screen_limit:]
    if disconnected:
        for group, choices in zip(components, component_orders):
            if choices:
                continue
            # A tied/zero-only component still needs one occupied side in each
            # child when other components split. Canonical rank is explicit
            # allocation ambiguity, not an invented measured distance.
            counts = support.observed[group].sum(axis=0)
            region = int(np.argmax(counts))
            ordered = group[support.observed[group, region]]
            positions = dict.fromkeys([len(ordered)//2, *np.linspace(
                1, len(ordered)-1, min(budget, len(ordered)-1), dtype=int)])
            choices.extend((2, 0., region, ordered, int(position))
                           for position in list(positions)[:budget])
            component_constructions += len(choices)
    proposals, seen, screened = [], set(), 0
    def regional_sides():
        for tier, gap, region, ordered, position in orders:
            missing = rows[~support.observed[rows, region]]
            for missing_left in range(2 if len(missing) else 1):
                yield np.r_[ordered[:position], missing if missing_left else np.empty(0, dtype=int)], {
                    "ordering_region": region, "profile_gap": -gap,
                    "profile_ordering": "within_bin" if tier else "coarse",
                    "missing_rows_side": "left" if missing_left else "right"}
    # A one-region ordering plus whole-block unknown placement cannot support
    # BOTH children across disconnected components. Skip those provably invalid
    # constructions so they cannot consume the component fallback's allowance.
    sides = (_component_split_sides(components, component_orders, support, screen_limit)
             if disconnected else regional_sides())
    for left, details in sides:
        if screened >= screen_limit or len(proposals) >= budget:
            break
        screened += 1
        proposed = labels.copy()
        proposed[left] = int(labels.max())+1
        proposed = canonical_partition(proposed)
        key = proposed.tobytes()
        if key in seen:
            continue
        seen.add(key)
        if support.check_partition(support.summarize_partition(proposed)) is None:
            proposals.append((proposed, {"split_block": int(block), **details}))
    return proposals, {"screened": screened, "screen_limit": screen_limit,
                       "regional_orderings": len(orders), "available": len(proposals),
                       "profile_column_block": max(1, min(8, (8*1024*1024)//(16*model.n))),
                       "profile_mode_cache_limit_bytes": cache_limit,
                       "retained_ordering_limit": screen_limit,
                       "within_bin_orderings": sum(item[0] == 1 for group in [orders, *component_orders]
                                                    for item in group),
                       "observation_components": len(components),
                       "component_local_orderings": sum(map(len, component_orders)),
                       "component_local_ordering_limit": len(components)*budget,
                       "component_local_constructions": component_constructions,
                       "component_local_construction_limit": 2*model.r*budget+len(components)*budget,
                       "component_alignment": "bounded_relative_rank_heuristic_not_identified" if disconnected else None,
                       "ranking": ("componentwise_observed_regional_rank" if disconnected
                                   else "observed_regional_coarse_then_within_bin_gap")}
