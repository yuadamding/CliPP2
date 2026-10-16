"""Deterministic, explicitly nonexhaustive search for large cluster capacities.

Only the proposal policy changes. Every admitted partition enumerates its
eligible clonal designations with the ordinary conditional refit and score.
The complete evaluated bank survives beam eviction and determines publication.
"""

from dataclasses import dataclass
from itertools import islice
from time import perf_counter

import numpy as np

from .refinement import best_pair_boundary, canonical_partition


@dataclass
class _Proposal:
    labels: np.ndarray
    centers: object
    details: dict


def _membership(candidate):
    fitted = candidate.refit
    return (candidate.membership_kind, fitted.cuts, fitted.groups)


def _population(candidates, width, indices):
    """Interleave score, conditional likelihood, partition and anchor diversity."""
    unique = {id(candidate): candidate for candidate in candidates}
    score = sorted(unique.values(), key=lambda c: (c.score, indices[id(c)]))
    conditional = sorted(score, key=lambda c: (-c.refit.conditional_log_likelihood,
                                              c.score, indices[id(c)]))
    partition, anchor = {}, {}
    for candidate in score:
        membership = _membership(candidate)
        partition.setdefault(membership, candidate)
        # A designation is meaningful within its own partition; do not equate
        # numeric cluster IDs across unrelated memberships or decode N rows.
        anchor.setdefault((membership, candidate.refit.clonal_cluster), candidate)
    streams = [iter(score), iter(conditional), iter(partition.values()), iter(anchor.values())]
    selected, seen = [], set()
    while len(selected) < min(width, len(score)):
        for stream in streams:
            candidate = next((c for c in stream if id(c) not in seen), None)
            if candidate is not None:
                selected.append(candidate)
                seen.add(id(candidate))
            if len(selected) == width:
                break
    return selected


class _Moves:
    """Lazy, bounded-neighborhood ranking; matrices shared within one parent."""

    def __init__(self, model, tree, bank, candidate, limit, split_ranks, keys):
        self.model, self.tree, self.bank = model, tree, bank
        self.candidate, self.limit = candidate, limit
        self.labels, self.centers = candidate.refit.labels, candidate.refit.centers
        self.q = len(self.centers)
        self.split_ranks, self.keys = split_ranks, keys
        self._columns = None
        self.stats = {name: {"scanned": False, "available": None, "generated": 0}
                      for name in ("merge", "split", "boundary", "reassignment")}

    def columns(self):
        if self._columns is None:
            self._columns = self.model.log_likelihood_columns(self.centers)
        return self._columns

    def merge(self):
        pairs = sorted({tuple(sorted(map(int, self.labels[edge]))) for edge in self.tree.edges
                        if self.labels[edge[0]] != self.labels[edge[1]]})
        pairs.sort(key=lambda pair: (float(np.sum((self.centers[pair[0]]-self.centers[pair[1]])**2)), pair))
        self.stats["merge"].update(scanned=True, available=len(pairs),
                                   ranking="adjacent_center_squared_distance")
        for a, b in pairs:
            self.stats["merge"]["generated"] += 1
            yield _Proposal(canonical_partition(np.where(self.labels == b, a, self.labels)),
                            None, {"merged_clusters": (a, b)})

    def split(self):
        from .selection import _likelihood_ranked_edges, _supported_split_edges
        stats = self.stats["split"]
        if self.candidate.membership_kind != "tree":
            stats.update(scanned=True, available=0, reason="disconnected_parent_not_tree_splittable")
            return
        cuts = self.candidate.cuts
        if cuts not in self.split_ranks:
            supported = _supported_split_edges(self.model, self.tree, cuts)
            ranked, diagnostic = _likelihood_ranked_edges(self.model, self.tree, supported, cuts)
            self.split_ranks[cuts] = (tuple(ranked[:self.limit]), len(ranked), diagnostic)
        edges, count, diagnostic = self.split_ranks[cuts]
        stats.update(scanned=True, available=count, ranking=diagnostic,
                     ranked_prefix_limit=self.limit)
        for edge in edges:
            proposed = tuple(sorted((*cuts, edge)))
            stats["generated"] += 1
            yield _Proposal(self.bank.labels(proposed), None, {"added_edge": int(edge)})

    def boundary(self):
        stats = self.stats["boundary"]
        if self.candidate.membership_kind != "tree":
            stats.update(scanned=True, available=0, reason="disconnected_parent_not_tree_refinable")
            return
        edges = sorted(self.candidate.cuts, key=lambda edge: (
            float(np.sum(np.diff(self.centers[self.labels[self.tree.edges[edge]]], axis=0)**2)), edge))
        stats.update(scanned=True, available=0, structural_edges=len(edges), profile_examined=0,
                     prescreen="adjacent_center_squared_distance", profile_limit=self.limit)
        if not edges:
            return
        adjacency = [[] for _ in range(self.tree.n)]
        for edge, (a, b) in enumerate(self.tree.edges):
            adjacency[a].append((int(b), edge))
            adjacency[b].append((int(a), edge))
        components = {k: np.flatnonzero(self.labels == k) for k in range(self.q)}
        proposals = []
        for edge in edges[:self.limit]:
            move = best_pair_boundary(self.tree, self.labels, self.centers, self.columns(), edge,
                                      adjacency=adjacency, components=components)
            stats["profile_examined"] += 1
            if move.changed:
                proposals.append(move)
        proposals.sort(key=lambda move: (-move.conditional_log_likelihood, move.edge))
        stats.update(changed_profiles=len(proposals), available=len(proposals),
                     unprofiled_edges=len(edges)-stats["profile_examined"])
        for move in proposals:
            labels, centers = canonical_partition(move.labels, move.centers)
            stats["generated"] += 1
            yield _Proposal(labels, centers, {"replacement_edge": int(move.edge),
                "ranking": "fixed_center_conditional_likelihood", "heuristic_not_pruning_bound": True})

    def reassignment(self):
        from .reassignment import _proposals
        stats = self.stats["reassignment"]
        if self.model.r == 1 or self.q == 1:
            stats.update(scanned=True, available=0, reason="joint_multi_group_move_only")
            return
        proposals, scans = _proposals(self.labels, self.centers, self.columns(),
                                     self.model.observed, self.keys)
        stats.update(scanned=True, available=sum(p is not None for p in proposals), **scans)
        for kind, labels in zip(("single", "greedy_batch"), proposals):
            if labels is not None:
                labels, centers = canonical_partition(labels, self.centers)
                stats["generated"] += 1
                yield _Proposal(labels, centers, {"proposal_kind": kind})


def fit_bounded(model, tree, pilot, seeds, capacities, search, *, pilot_diagnostics=None,
                initialization_seconds=0., tree_seconds=0., continuation_seconds=0.):
    """Consume precomputed scalar-chain or joint-tree seeds without enumeration.

    Native proposals do not consume the extra structural budget. Every other
    family, including anchor rescue, does. Sweeps alternate downward/upward;
    each visited q freezes at most W anchored starts and each start has d slots.
    Incoming proposals additionally obey the separate per-target-q allowance.
    """
    from .selection import _CandidateBank, _score_key, TreeFit
    from ..kernel import ALGORITHM, MAX_CLUSTERS

    started = perf_counter()
    capacities = tuple(capacities)
    if (not capacities or any(isinstance(k, (bool, np.bool_))
                             or not isinstance(k, (int, np.integer))
                             or not 1 <= k <= min(MAX_CLUSTERS, model.n) for k in capacities)
            or len(set(capacities)) != len(capacities)):
        raise ValueError("Bounded search requires unique valid integer capacities")
    capacities = tuple(map(int, capacities))
    maximum = max(capacities)
    search = search.resolve(maximum)
    if search.mode != "bounded":
        raise ValueError("fit_bounded requires the bounded search policy")
    # Native proposals are outside the additional-move budget, so enforce their
    # separate linear bound before any fit. Never consume an unbounded iterable
    # or allow worker completion order to determine the active population.
    seeds = tuple(islice(iter(seeds), len(capacities)+1))
    requested = tuple(seed.get("requested_k") for seed in seeds)
    if (len(seeds) != len(capacities)
            or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) for k in requested)
            or len(set(requested)) != len(requested) or set(requested) != set(capacities)):
        raise ValueError("Native seeds must cover each requested capacity exactly once")
    by_capacity = dict(zip(map(int, requested), seeds))
    seeds = tuple(by_capacity[k] for k in capacities)
    bank = _CandidateBank(model, tree, compact_general=True)
    beams = {q: [] for q in range(1, maximum + 1)}
    indices, expanded = {}, set()
    spent = {q: 0 for q in beams}
    anchor_reserved = {q: 0 for q in beams}
    actual_anchors = {q: 0 for q in beams}
    ledger, seeds_diagnostics, paths = [], [], []
    evaluated_partitions = set()

    def update(candidates):
        by_q = {}
        for candidate in candidates:
            if id(candidate) not in indices:
                indices[id(candidate)] = len(indices)
            by_q.setdefault(len(candidate.refit.centers), []).append(candidate)
        for q, additions in by_q.items():
            beams[q] = _population([*beams[q], *additions], search.width, indices)

    def consider(labels, centers, provenance, *, native=False, declared_cuts=None):
        labels, centers = canonical_partition(labels, centers) if centers is not None else (canonical_partition(labels), None)
        q = int(labels.max()) + 1
        record = {"family": provenance["route"], "occupied_q": q,
                  "native": native, "provenance": provenance}
        ledger.append(record)
        if not 1 <= q <= maximum:
            record["outcome"] = "capacity_rejected"
            return []
        if not native:
            if spent[q] >= search.partition_limit_per_q:
                record["outcome"] = "target_partition_budget_exhausted"
                return []
            spent[q] += 1
            anchor_reserved[q] += q
        cuts = tuple(map(int, np.flatnonzero(labels[tree.edges[:, 0]] != labels[tree.edges[:, 1]])))
        tree_member = len(cuts) + 1 == q
        if native and (not tree_member or cuts != tuple(declared_cuts)):
            raise ValueError("Native candidate labels disagree with its frozen-tree support")
        key = ("tree", cuts) if tree_member else ("general", bank.membership_key(labels))
        evaluated_partitions.add(key)
        before, count = bank.refit_calls, len(bank.candidates)
        variants = bank.consider_variants(cuts if tree_member else None, labels, centers, provenance)
        evaluated = bank.refit_calls-before
        if evaluated > q:
            raise AssertionError("One partition exceeded its clonal-hypothesis allowance")
        if not native:
            actual_anchors[q] += evaluated
        record.update(outcome="eligible" if variants else "ineligible", anchored_evaluations=evaluated,
                      retained_states=len(bank.candidates)-count,
                      returned_state_indices=[])
        update(variants)
        record["returned_state_indices"] = [indices[id(c)] for c in variants]
        return variants

    for index, seed in enumerate(seeds):
        cuts = tuple(sorted(map(int, seed["cuts"])))
        requested = int(seed["requested_k"])
        if requested not in capacities or len(cuts) >= requested:
            raise ValueError("Generated seed exceeds its declared capacity")
        provenance = {"route": "native", "seed_index": index, "requested_k": requested,
                      "occupied_q": len(cuts)+1, "native_collapsed": len(cuts)+1 < requested}
        seeds_diagnostics.append({**provenance, "cuts": cuts, "solver": seed.get("diagnostics", {})})
        consider(seed["labels"], seed.get("centers"), provenance, native=True, declared_cuts=cuts)
    native_count = len(bank.candidates)
    native_anchor_evaluations = bank.refit_calls
    if not bank.candidates and maximum >= 2:
        consider(np.where(model.clonal_eligible, 0, 1), None,
                 {"route": "eligible_set_anchor_rescue", "requested_k": maximum})
    split_ranks = {}
    keys = tuple(model.observation_keys()) if model.r > 1 else ()
    for sweep in range(search.sweeps):
        downward = sweep % 2 == 0
        levels = range(maximum, 0, -1) if downward else range(1, maximum+1)
        families = ("merge", "boundary", "reassignment", "split") if downward else (
                    "split", "boundary", "reassignment", "merge")
        for q in levels:
            for candidate in tuple(beams[q]):
                parent = indices[id(candidate)]
                expanded.add(parent)
                moves = _Moves(model, tree, bank, candidate, search.proposals_per_state, split_ranks, keys)
                generators = {family: iter(getattr(moves, family)()) for family in families}
                path = {"sweep": sweep, "direction": "downward" if downward else "upward",
                        "occupied_q": q, "parent_bank_index": parent, "slots": []}
                paths.append(path)
                for slot in range(search.proposals_per_state):
                    family = families[slot % len(families)]
                    target = q + (1 if family == "split" else -1 if family == "merge" else 0)
                    entry = {"slot": slot, "family": family, "target_q": target}
                    path["slots"].append(entry)
                    if target not in spent:
                        entry["outcome"] = "outside_capacity"
                    elif spent[target] >= search.partition_limit_per_q:
                        entry["outcome"] = "target_partition_budget_exhausted"
                    else:
                        proposal = next(generators[family], None)
                        if proposal is None:
                            entry["outcome"] = "no_ranked_proposal"
                        else:
                            before = len(ledger)
                            consider(proposal.labels, proposal.centers, {
                                "route": f"bounded_{family}", "requested_k": target,
                                "sweep": sweep, "parent_bank_index": parent, **proposal.details})
                            entry.update(outcome=ledger[-1]["outcome"], attempt_index=before)
                path["neighborhood"] = moves.stats
                for stats in moves.stats.values():
                    stats["unexpanded_known"] = (None if stats["available"] is None else
                                                   max(0, stats["available"]-stats["generated"]))
    if not bank.candidates:
        raise ValueError("no_eligible_anchored_candidate: bounded proposal search exhausted")
    selected = min(bank.candidates, key=_score_key)
    if any(actual_anchors[q] > q*search.partition_limit_per_q for q in beams):
        raise AssertionError("Anchored search budget exceeded")
    elapsed = perf_counter()-started
    cache = bank.partition_cache.diagnostics()
    return TreeFit(selected, bank.candidates, tree, np.asarray(pilot), {
        "pipeline": ALGORITHM, "tree_identity": tree.identity,
        "tree_kind": "mutation_similarity_not_phylogeny", "pilots": pilot_diagnostics or {},
        "seeds": seeds_diagnostics, "capacities": capacities, "enumerated_cut_subsets": 0,
        "eligible_scored_candidates": len(bank.candidates),
        "unique_eligible_partitions": len(bank.by_cuts)+len(bank.by_membership),
        "evaluated_unique_partitions": len(evaluated_partitions),
        "ineligible_candidates": list(bank.ineligible.values()),
        "general_ineligible_candidates": bank.general_rejections,
        "anchor_ineligible_candidates": bank.anchor_rejections,
        "selected_membership_kind": selected.membership_kind,
        "partition_cache": cache,
        "candidate_membership_storage": "compact_tree_cuts_and_general_component_maps_with_bounded_decoded_cache",
        "selection": "minimum_over_complete_eligible_evaluated_bank",
        "global_optimality_certified": False,
        "search_policy": {
            "mode": "bounded", "exhaustive": False, "width": search.width,
            "proposals_per_state": search.proposals_per_state, "sweeps": search.sweeps,
            "partition_limit_per_q": search.partition_limit_per_q,
            "partition_limit_total": maximum*search.partition_limit_per_q,
            "anchored_limit_total": search.partition_limit_per_q*maximum*(maximum+1)//2,
            "initial_native_candidates": native_count,
            "initial_native_partition_limit": len(capacities),
            "initial_native_anchored_limit": sum(capacities),
            "initial_native_anchored_evaluations": native_anchor_evaluations,
            "additional_partition_attempts": spent, "anchored_slots_reserved": anchor_reserved,
            "additional_anchored_evaluations": actual_anchors,
            "active_bank_indices_by_q": {q: [indices[id(c)] for c in active] for q, active in beams.items()},
            "unexpanded_evaluated_bank_indices": sorted(set(indices.values())-expanded),
            "population_rule": "round_robin_score_conditional_partition_anchor_diversity",
            "final_score_tie_rule": "score_q_tree_before_general_compact_component_encoding_clonal_index",
            "width_is_anchored_fitted_states": True,
            "population_growth_order": "synchronous_deterministic_sweep_q_state_slot",
            "wider_search_is_guaranteed_superset": False,
            "unknown_unexpanded_neighborhoods_recorded_as_null": True,
            "attempts": ledger, "paths": paths,
        },
        "timings": {"initialization_seconds": initialization_seconds, "tree_seconds": tree_seconds,
                    "continuation_seconds": continuation_seconds, "bounded_search_seconds": elapsed,
                    "refit_and_score_seconds": bank.refit_seconds,
                    "fit_total_seconds": initialization_seconds+tree_seconds+continuation_seconds+elapsed},
    })
