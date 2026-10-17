"""Deterministic, explicitly nonexhaustive search for large cluster capacities.

Only the proposal policy changes. Every admitted partition enumerates its
eligible clonal designations with the ordinary conditional refit and score.
The complete evaluated bank survives beam eviction and determines publication.
"""

from dataclasses import dataclass
from itertools import combinations, islice
from time import perf_counter

import numpy as np

from .refinement import best_pair_boundary, canonical_partition, tree_adjacency
from .support_proposals import (SEED_ATTEMPT_LIMIT, active_support, support_preflight,
                                support_seed_proposals, support_split_proposals,
                                supported_native_repairs)


FAMILIES = ("merge", "split", "boundary", "single_reassignment", "batch_reassignment")


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

    def __init__(self, model, tree, bank, candidate, limit, split_ranks, keys, pilot=None):
        self.model, self.tree, self.bank = model, tree, bank
        self.candidate, self.limit = candidate, limit
        self.labels, self.centers = candidate.refit.labels, candidate.refit.centers
        self.q = len(self.centers)
        self.split_ranks, self.keys = split_ranks, keys
        self.support = (model.regional_support if getattr(tree, "component_count", 1) > 1
                        else active_support(model))
        self.pilot = pilot
        self._columns = None
        self._reassignment = None
        self.stats = {name: {"scanned": False, "available": None, "generated": 0}
                      for name in FAMILIES}

    def columns(self):
        if self._columns is None:
            self._columns = getattr(self.model, "proposal_log_likelihood_columns",
                                    self.model.log_likelihood_columns)(self.centers)
        return self._columns

    def merge(self):
        general = self.candidate.membership_kind == "general"
        pairs = (list(combinations(range(self.q), 2)) if general else
                 sorted({tuple(sorted(map(int, self.labels[edge]))) for edge in self.tree.edges
                         if self.labels[edge[0]] != self.labels[edge[1]]}))
        screened = len(pairs)
        if self.support is not None:
            state = self.support.summarize_partition(self.labels)
            pairs = [(a, b) for a, b in pairs if not np.any(
                ((state.positive_counts[a]+state.positive_counts[b]) > 0)
                & ((state.hard_zero_counts[a]+state.hard_zero_counts[b]) > 0))]
        pairs.sort(key=lambda pair: (float(np.sum((self.centers[pair[0]]-self.centers[pair[1]])**2)), pair))
        self.stats["merge"].update(scanned=True, available=len(pairs),
            support_pairs_screened=screened,
            ranking="all_compatible_pairs_center_squared_distance" if general else "adjacent_center_squared_distance")
        for a, b in pairs:
            self.stats["merge"]["generated"] += 1
            yield _Proposal(canonical_partition(np.where(self.labels == b, a, self.labels)),
                            None, {"merged_clusters": (a, b)})

    def split(self):
        from .selection import _likelihood_ranked_edges, _supported_split_edges
        stats = self.stats["split"]
        if self.candidate.membership_kind != "tree":
            # Bounded block screening uses only observed pilot variation. The
            # actual proposals are ordered by observed regional profile gaps.
            def variation(k):
                rows = self.labels == k
                if self.pilot is None:
                    return -int(rows.sum())
                return -sum(float(np.ptp(self.pilot[rows & self.model.observed[:, r], r]))
                            for r in range(self.model.r) if np.any(rows & self.model.observed[:, r]))
            blocks = sorted(range(self.q), key=lambda k: (variation(k), k))[:self.limit]
            proposals, diagnostics = [], []
            for block in blocks:
                found, diagnostic = support_split_proposals(
                    self.model, self.labels, block, self.limit, pilot=self.pilot, support=self.support)
                diagnostics.append({"block": block, **diagnostic})
                proposals.extend(found)
            proposals.sort(key=lambda item: ({"coarse": 0, "within_bin": 1, "componentwise": 2}[
                                                 item[1]["profile_ordering"]],
                                             -item[1]["profile_gap"], item[1]["split_block"],
                                             item[1]["ordering_region"], tuple(item[0])))
            stats.update(scanned=True, available=len(proposals),
                         ranking="observed_regional_profile_general_split", blocks=diagnostics,
                         block_screen_limit=self.limit, structural_screen_limit=4*self.limit**2)
            for labels, details in proposals[:self.limit]:
                stats["generated"] += 1
                yield _Proposal(labels, None, details)
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
        adjacency = tree_adjacency(self.tree)
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

    def _scan_reassignment(self):
        from .reassignment import _proposals
        if self._reassignment is None:
            if self.q == 1:
                self._reassignment = ((None, None), {"reason": "one_occupied_group"})
            else:
                self._reassignment = _proposals(self.labels, self.centers, self.columns(),
                    self.model.observed, self.keys,
                    objective=getattr(self.model, "selection_objective", "legacy_mixture"),
                    **({"support": self.support} if self.support is not None else {}))
        return self._reassignment

    def _reassignment_family(self, index, family, kind):
        proposals, scan = self._scan_reassignment()
        labels = proposals[index]
        stats = self.stats[family]
        stats.update(scanned=True, available=int(labels is not None), shared_scan=scan)
        if labels is not None:
            labels, centers = canonical_partition(labels, self.centers)
            stats["generated"] += 1
            yield _Proposal(labels, centers, {"proposal_kind": kind,
                "moved_mutations": scan["single_moves" if index == 0 else "batch_moves"]})

    def single_reassignment(self):
        yield from self._reassignment_family(0, "single_reassignment", "single")

    def batch_reassignment(self):
        yield from self._reassignment_family(1, "batch_reassignment", "greedy_batch")


def fit_bounded(model, tree, pilot, seeds, capacities, search, *, pilot_diagnostics=None,
                initialization_seconds=0., tree_seconds=0., continuation_seconds=0.,
                checkpoint_path=None, resume=False, checkpoint_stop_after=None,
                checkpoint_context=None):
    """Consume precomputed scalar-chain or joint-tree seeds without enumeration.

    Native proposals do not consume the extra structural budget. Every other
    family, including anchor rescue, does. Sweeps alternate downward/upward;
    each visited q freezes at most W anchored starts and each start has d slots.
    Incoming proposals additionally obey the separate per-target-q allowance.
    """
    from .selection import _CandidateBank, _score_key, TreeFit
    from ..kernel import ALGORITHM, MAX_CLUSTERS
    from .checkpoint import (SearchPaused, identities, read_checkpoint, restore_bank,
                             snapshot_bank, write_checkpoint)
    from pathlib import Path

    if resume and checkpoint_path is None:
        raise ValueError("resume requires a checkpoint_path")
    if checkpoint_stop_after is not None and (checkpoint_path is None
            or type(checkpoint_stop_after) is not int or checkpoint_stop_after < 1):
        raise ValueError("checkpoint_stop_after requires a path and positive integer state budget")
    if checkpoint_path is not None and not resume and Path(checkpoint_path).exists():
        raise FileExistsError("Search checkpoint exists; explicitly resume or use a fresh path")

    started = perf_counter()
    capacities = tuple(capacities)
    if (not capacities or any(isinstance(k, (bool, np.bool_))
                             or not isinstance(k, (int, np.integer))
                             or not 1 <= k <= min(MAX_CLUSTERS, model.n) for k in capacities)
            or len(set(capacities)) != len(capacities)):
        raise ValueError("Bounded search requires unique valid integer capacities")
    capacities = tuple(map(int, capacities))
    maximum = max(capacities)
    support_diagnostics = support_preflight(model, maximum)
    support = active_support(model)
    components = tree.component_count
    if support is None and components > 1:
        support = model.regional_support
    requested_mode = search.mode
    search = search.resolve(maximum, active_hard_zero=support is not None)
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
    restored = read_checkpoint(checkpoint_path, model, tree, pilot) if resume else None
    if restored is not None:
        old_policy = restored["policy"]
        if (not set(restored["capacities"]).issubset(capacities) or old_policy["width"] != search.width
                or old_policy["proposals_per_state"] != search.proposals_per_state
                or old_policy["sweeps"] > search.sweeps):
            raise ValueError("Checkpoint schedule changed: retain earlier capacities, width, slots and sweep budget")
        if (restored["capacities"] != capacities
                and restored["state"]["schedule"]["sweep"] < old_policy["sweeps"]):
            raise ValueError("Finish the bound schedule before extending its cluster capacity")
        if restored["capacities"] != capacities and search.sweeps <= old_policy["sweeps"]:
            raise ValueError("Capacity extension requires additional cumulative search sweeps")
        checkpoint_context = restored.get("checkpoint_context")
    bank = (restore_bank(model, tree, restored["bank"]) if restored is not None
            else _CandidateBank(model, tree, compact_general=True))
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

    def consider(labels, centers, provenance, *, native=False, declared_cuts=None,
                 force_general=False):
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
        tree_member = len(cuts) + components == q and not force_general
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

    def consider_native(seed, index):
        """Use the same validated contract for initial and resumed capacities."""
        if "unavailable_reason" in seed:
            if (components <= 1 or seed["unavailable_reason"] != "disconnected_observation_overlap"
                    or set(seed) != {"requested_k", "unavailable_reason"}):
                raise ValueError("Invalid unavailable native-capacity record")
            seeds_diagnostics.append({"route": "native_unavailable", "seed_index": index, **seed,
                "topology_component_count": components, "support_infeasibility_claimed": False})
            return None
        cuts = tuple(sorted(map(int, seed["cuts"])))
        requested, occupied = int(seed["requested_k"]), len(cuts)+components
        if occupied > requested:
            raise ValueError("Generated seed exceeds its declared capacity")
        provenance = {"route": "native", "seed_index": index, "requested_k": requested,
                      "occupied_q": occupied, "native_collapsed": occupied < requested}
        seeds_diagnostics.append({**provenance, "cuts": cuts, "solver": seed.get("diagnostics", {})})
        consider(seed["labels"], seed.get("centers"), provenance, native=True, declared_cuts=cuts)
        return cuts

    def repair_native(seed, index):
        for labels, details in supported_native_repairs(model, tree, seed["labels"],
                seed["cuts"], search.proposals_per_state):
            consider(labels, None, {"route": "native_support_transformation",
                "seed_index": index, "requested_k": seed["requested_k"], **details})

    for index, seed in enumerate(seeds if restored is None else ()):
        consider_native(seed, index)
    native_count = len(bank.candidates)
    native_anchor_evaluations = bank.refit_calls
    native_reasons = {}
    failed_anchor_cuts = {record["cuts"] for record in bank.anchor_rejections}
    for rejection in [*(record for cuts, record in bank.ineligible.items()
                        if cuts not in failed_anchor_cuts), *bank.anchor_rejections]:
        reason = str(rejection["reason"])
        native_reasons[reason] = native_reasons.get(reason, 0)+1
    if restored is None:
        for index, seed in enumerate(seeds):
            if "cuts" in seed:
                repair_native(seed, index)
    # Seed work has a fixed construction bound and reserves ordinary target-q
    # slots. Reservations are released before sweeps; no new fit allowance is
    # introduced. The seeds are explicitly general, never native tree seeds.
    reserved_seed_slots = min(SEED_ATTEMPT_LIMIT, search.partition_limit_per_q) if support is not None else 0
    support_seed_records, seed_spent = [], {q: 0 for q in beams}
    if support is not None and restored is None:
        for labels, diagnostic in support_seed_proposals(model, maximum, limit=reserved_seed_slots,
                                                         support=support):
            record = dict(diagnostic)
            support_seed_records.append(record)
            if labels is None:
                record["outcome"] = "support_seed_search_exhausted"
                continue
            q = int(labels.max())+1
            if seed_spent[q] >= reserved_seed_slots:
                record["outcome"] = "reserved_seed_budget_exhausted"
                continue
            seed_spent[q] += 1
            consider(labels, None, {"route": "regional_support_seed", "requested_k": q,
                                   **diagnostic}, force_general=True)
            record.update(occupied_q=q, outcome=ledger[-1]["outcome"], attempt_index=len(ledger)-1)
    if restored is None and not bank.candidates and maximum >= 2 and support is None:
        consider(np.where(model.clonal_eligible, 0, 1), None,
                 {"route": "eligible_set_anchor_rescue", "requested_k": maximum})
    split_ranks = {}
    keys = tuple(model.observation_keys())
    family_cursors, family_consumed = {}, {}
    schedule = {"sweep": 0, "level": 0, "queue": None, "position": 0}
    if restored is not None:
        saved = restored["state"]
        if "model_telemetry" in restored and hasattr(model, "telemetry"):
            model.telemetry.update(restored["model_telemetry"])
        beams.update({q: [bank.candidates[i] for i in values] for q, values in saved["beams"].items()})
        indices = {id(candidate): i for i, candidate in enumerate(bank.candidates)}
        expanded = set(saved["expanded"])
        spent, anchor_reserved, actual_anchors = (saved[key] for key in (
            "spent", "anchor_reserved", "actual_anchors"))
        ledger, seeds_diagnostics, paths = (saved[key] for key in ("ledger", "seeds_diagnostics", "paths"))
        evaluated_partitions = set(saved["evaluated_partitions"])
        native_count, native_anchor_evaluations, native_reasons = (saved[key] for key in (
            "native_count", "native_anchor_evaluations", "native_reasons"))
        reserved_seed_slots, support_seed_records, seed_spent = (saved[key] for key in (
            "reserved_seed_slots", "support_seed_records", "seed_spent"))
        family_cursors, family_consumed, schedule = (saved[key] for key in (
            "family_cursors", "family_consumed", "schedule"))
        for q in beams:
            spent.setdefault(q, 0)
            anchor_reserved.setdefault(q, 0)
            actual_anchors.setdefault(q, 0)
            seed_spent.setdefault(q, 0)
        native_cuts_seen = {tuple(seed["cuts"]) for seed in restored["seeds"] if "cuts" in seed}
        for index, seed in enumerate(seeds):
            requested = seed["requested_k"]
            if requested in restored["capacities"]:
                continue
            before = len(bank.candidates)
            calls = bank.refit_calls
            rejection_count = len(bank.anchor_rejections)
            cuts = consider_native(seed, index)
            if cuts is None:
                continue
            native_count += len(bank.candidates)-before
            native_anchor_evaluations += bank.refit_calls-calls
            new_rejections = bank.anchor_rejections[rejection_count:]
            if not new_rejections and cuts not in native_cuts_seen and cuts in bank.ineligible:
                new_rejections = [bank.ineligible[cuts]]
            for rejection in new_rejections:
                reason = str(rejection["reason"])
                native_reasons[reason] = native_reasons.get(reason, 0)+1
            native_cuts_seen.add(cuts)
            repair_native(seed, index)

    checkpoint_identity = identities(model, tree, pilot) if checkpoint_path is not None else None
    checkpoint_work = 0

    def persist():
        if checkpoint_path is None:
            return
        state = dict(spent=spent, anchor_reserved=anchor_reserved, actual_anchors=actual_anchors,
            ledger=ledger, seeds_diagnostics=seeds_diagnostics, paths=paths,
            evaluated_partitions=sorted(evaluated_partitions, key=repr),
            native_count=native_count, native_anchor_evaluations=native_anchor_evaluations,
            native_reasons=native_reasons, reserved_seed_slots=reserved_seed_slots,
            support_seed_records=support_seed_records, seed_spent=seed_spent,
            family_cursors=family_cursors, family_consumed=family_consumed, schedule=schedule,
            expanded=sorted(expanded), beams={q: [indices[id(c)] for c in values]
                                             for q, values in beams.items()})
        write_checkpoint(checkpoint_path, dict(identities=checkpoint_identity,
            policy=dict(width=search.width, proposals_per_state=search.proposals_per_state,
                        sweeps=search.sweeps), capacities=capacities, bank=snapshot_bank(bank),
            tree=dict(edges=tree.edges, n=tree.n, mutation_ids=tree.mutation_ids, kind=tree.kind),
            pilot=np.asarray(pilot), pilot_diagnostics=pilot_diagnostics, seeds=seeds,
            checkpoint_context=checkpoint_context,
            model_telemetry=dict(getattr(model, "telemetry", {})),
            timings=dict(initialization_seconds=initialization_seconds, tree_seconds=tree_seconds,
                         continuation_seconds=continuation_seconds), state=state))

    persist()
    while schedule["sweep"] < search.sweeps:
        sweep = schedule["sweep"]
        downward = sweep % 2 == 0
        levels = list(range(maximum, 0, -1) if downward else range(1, maximum+1))
        while schedule["level"] < len(levels):
            q = levels[schedule["level"]]
            if schedule["queue"] is None:
                schedule["queue"] = [indices[id(candidate)] for candidate in beams[q]]
                schedule["position"] = 0
            while schedule["position"] < len(schedule["queue"]):
                candidate = bank.candidates[schedule["queue"][schedule["position"]]]
                parent = indices[id(candidate)]
                expanded.add(parent)
                moves = _Moves(model, tree, bank, candidate,
                    search.proposals_per_state*search.sweeps, split_ranks, keys, pilot)
                generators = {family: iter(getattr(moves, family)()) for family in FAMILIES}
                consumed = family_consumed.setdefault(parent, dict.fromkeys(FAMILIES, 0))
                for family, generator in generators.items():
                    for _ in range(consumed[family]):
                        next(generator, None)
                path = {"sweep": sweep, "direction": "downward" if downward else "upward",
                        "occupied_q": q, "parent_bank_index": parent, "slots": [],
                        "family_cursor_start": family_cursors.get(parent, 0),
                        "last_accepted_improvement": None, "last_accepted_moved_mutations": 0}
                paths.append(path)
                for slot in range(search.proposals_per_state):
                    family = FAMILIES[(family_cursors.get(parent, 0)+slot) % len(FAMILIES)]
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
                            consumed[family] += 1
                            variants = consider(proposal.labels, proposal.centers, {
                                "route": f"bounded_{family}", "requested_k": target,
                                "sweep": sweep, "parent_bank_index": parent, **proposal.details})
                            entry.update(outcome=ledger[-1]["outcome"], attempt_index=before)
                            improvement = candidate.score-min((c.score for c in variants), default=np.inf)
                            entry["improves_parent"] = bool(improvement > 1e-7+1e-10*abs(candidate.score))
                            if entry["improves_parent"]:
                                entry["improvement"] = float(improvement)
                                path["last_accepted_improvement"] = float(improvement)
                                path["last_accepted_moved_mutations"] = proposal.details.get("moved_mutations")
                family_cursors[parent] = (family_cursors.get(parent, 0)+search.proposals_per_state) % len(FAMILIES)
                path["family_cursor_next"] = family_cursors[parent]
                path["stop_reason"] = ("target_q_budget_exhausted" if any(
                    e["outcome"] == "target_partition_budget_exhausted" for e in path["slots"])
                    else "proposal_budget_exhausted")
                path["neighborhood"] = moves.stats
                for stats in moves.stats.values():
                    stats["unexpanded_known"] = (None if stats["available"] is None else
                                                   max(0, stats["available"]-stats["generated"]))
                for family, stats in moves.stats.items():
                    entries = [entry for entry in path["slots"] if entry["family"] == family]
                    stats.update(admitted=sum("attempt_index" in entry for entry in entries),
                        rejected=sum(entry["outcome"] == "ineligible" for entry in entries),
                        accepted=sum(entry.get("improves_parent", False) for entry in entries),
                        unconsumed=stats["unexpanded_known"])
                schedule["position"] += 1
                checkpoint_work += 1
                persist()
                if checkpoint_stop_after is not None and checkpoint_work >= checkpoint_stop_after:
                    raise SearchPaused("Requested state budget reached; resume the committed search checkpoint")
            schedule.update(level=schedule["level"]+1, queue=None, position=0)
        schedule.update(sweep=sweep+1, level=0, queue=None, position=0)
    persist()
    if not bank.candidates:
        if support is not None:
            raise ValueError("support_seed_search_exhausted: bounded support-compatible search found no eligible anchored fit; global infeasibility is not established")
        raise ValueError("no_eligible_anchored_candidate: bounded proposal search exhausted")
    selected = min(bank.candidates, key=_score_key)
    if any(actual_anchors[q] > q*search.partition_limit_per_q for q in beams):
        raise AssertionError("Anchored search budget exceeded")
    elapsed = perf_counter()-started
    cache = bank.partition_cache.diagnostics()
    return TreeFit(selected, bank.candidates, tree, np.asarray(pilot), {
        "pipeline": ALGORITHM, "tree_identity": tree.identity,
        "tree_kind": "mutation_similarity_not_phylogeny", "pilots": pilot_diagnostics or {},
        "topology_kind": tree.kind,
        "observation_overlap_components": components,
        "cross_component_distances_imputed": False,
        "cross_component_membership_identified": False if components > 1 else None,
        "topology_uncertainty": ("relative memberships across disconnected observation components are not identified by overlap"
                                 if components > 1 else None),
        "seeds": seeds_diagnostics, "capacities": capacities, "enumerated_cut_subsets": 0,
        "eligible_scored_candidates": len(bank.candidates),
        "unique_eligible_partitions": len(bank.by_cuts)+len(bank.by_membership),
        "evaluated_unique_partitions": len(evaluated_partitions),
        "ineligible_candidates": list(bank.ineligible.values()),
        "general_ineligible_candidates": bank.general_rejections,
        "anchor_ineligible_candidates": bank.anchor_rejections,
        **support_diagnostics,
        "support_conflicts_rejected": bank.support_conflicts_rejected,
        "support_rejection_witnesses": bank.support_rejection_witnesses,
        "support_seed_attempts": len(support_seed_records),
        "support_seeds": support_seed_records,
        "native_support_outcome": "eligible_native_bank" if native_count else "no_eligible_native_seed",
        "native_rejection_reasons": native_reasons,
        "native_rejection_counting": "structurally_rejected_partitions_and_failed_anchored_variants",
        "native_unavailable_capacities": [record["requested_k"] for record in seeds_diagnostics
                                          if record["route"] == "native_unavailable"],
        "native_support_scope": "evaluated_native_seed_family_only_not_all_tree_partitions",
        "structural_candidates_screened": len(ledger),
        "structural_screen_scope": "admission_callbacks; bounded neighborhood prescreens recorded in paths",
        "support_seed_construction_attempts": len(support_seed_records),
        "support_seed_construction_failures": sum(record.get("reason") is not None for record in support_seed_records),
        "full_partition_attempts": bank.full_partition_attempts,
        "anchored_variants_evaluated": bank.refit_calls,
        "selected_membership_kind": selected.membership_kind,
        "partition_cache": cache,
        "candidate_membership_storage": "compact_tree_cuts_and_general_component_maps_with_bounded_decoded_cache",
        "selection": "minimum_over_complete_eligible_evaluated_bank",
        "global_optimality_certified": False,
        "search_policy": {
            "mode": "bounded", "exhaustive": False, "width": search.width,
            "requested_mode": requested_mode, "resolved_mode": "bounded",
            "proposals_per_state": search.proposals_per_state, "sweeps": search.sweeps,
            "reassignment_starts_rounds": "not_applicable_five_family_sweep_schedule",
            "partition_limit_per_q": search.partition_limit_per_q,
            "partition_limit_total": maximum*search.partition_limit_per_q,
            "anchored_limit_total": search.partition_limit_per_q*maximum*(maximum+1)//2,
            "initial_native_candidates": native_count,
            "initial_native_partition_limit": len(capacities),
            "initial_native_anchored_limit": sum(capacities),
            "initial_native_anchored_evaluations": native_anchor_evaluations,
            "additional_partition_attempts": spent, "anchored_slots_reserved": anchor_reserved,
            "additional_anchored_evaluations": actual_anchors,
            "support_seed_reserved_slots_per_q": reserved_seed_slots,
            "support_seed_slots_used_by_q": seed_spent,
            "support_seed_unused_reservations_released_before_sweeps": True,
            "active_bank_indices_by_q": {q: [indices[id(c)] for c in active] for q, active in beams.items()},
            "unexpanded_evaluated_bank_indices": sorted(set(indices.values())-expanded),
            "population_rule": "round_robin_score_conditional_partition_anchor_diversity",
            "final_score_tie_rule": "score_q_tree_before_general_compact_component_encoding_clonal_index",
            "width_is_anchored_fitted_states": True,
            "population_growth_order": "synchronous_deterministic_sweep_q_state_slot",
            "wider_search_is_guaranteed_superset": False,
            "resumed_archive_retained": restored is not None,
            "checkpoint_complete": checkpoint_path is not None,
            "termination_reason": "round_budget_exhausted",
            "stabilization_certified": False,
            "last_accepted_improvement": next((p["last_accepted_improvement"] for p in reversed(paths)
                                               if p["last_accepted_improvement"] is not None), None),
            "family_cursors": family_cursors,
            "unknown_unexpanded_neighborhoods_recorded_as_null": True,
            "attempts": ledger, "paths": paths,
        },
        "timings": {"initialization_seconds": initialization_seconds, "tree_seconds": tree_seconds,
                    "continuation_seconds": continuation_seconds, "bounded_search_seconds": elapsed,
                    "refit_and_score_seconds": bank.refit_seconds,
                    "fit_total_seconds": initialization_seconds+tree_seconds+continuation_seconds+elapsed},
    })
