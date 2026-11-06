"""Bounded tree exploration and exact-state terminal membership certification."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .proposals import refine_partition
from ..config import DEFAULT_AUDIT_SCOPE, DEFAULT_MAX_CLUSTERS, validate_audit_scope
from ..kernel import ALGORITHM
from .candidates import Candidate as Candidate, CandidateBank as _CandidateBank, score_key as _score_key
from .policy import POLICY, SearchWork
from ..kernel.topology import build_tree, canonical_partition, partition_from_cuts


@dataclass(eq=False)
class TreeFit:
    selected: Candidate
    candidate_bank: list
    tree: object
    pilot: np.ndarray
    diagnostics: dict
    attempt_links: tuple | None = None
    attempt_outcomes: tuple | None = None
    membership_audits: tuple | None = None
    # Legacy explicitly constructed evidence covered every q. Public fitting
    # passes its requested scope explicitly, defaulting to selected.
    audit_scope: str = "all"

    @property
    def labels(self):
        return self.selected.fit.labels

    @property
    def centers(self):
        return self.selected.fit.centers

    @property
    def weights(self):
        return self.selected.fit.weights

    @property
    def score(self):
        return self.selected.score


def _forest_structure(tree, cuts):
    removed = set(cuts)
    adjacency = [[] for _ in range(tree.n)]
    for edge, (a, b) in enumerate(tree.edges):
        if edge not in removed:
            adjacency[a].append((int(b), edge))
            adjacency[b].append((int(a), edge))
    parent = np.full(tree.n, -1, dtype=np.int64)
    parent_edge = np.full(tree.n, -1, dtype=np.int64)
    component = np.empty(tree.n, dtype=np.int64)
    order, roots = [], []
    for root in range(tree.n):
        if parent[root] >= 0:
            continue
        roots.append(root)
        parent[root] = root
        stack = [root]
        while stack:
            node = stack.pop()
            order.append(node)
            component[node] = root
            for neighbor, edge in adjacency[node]:
                if parent[neighbor] < 0:
                    parent[neighbor], parent_edge[neighbor] = node, edge
                    stack.append(neighbor)
    return parent, parent_edge, component, order, roots


def _supported_split_edges(model, tree, cuts=()):
    """Exact observation-supported forest splits, initially support-ranked.

    One forest traversal costs O(NR) work/storage. Support is necessary, not
    sufficient: exact refits still check original boxes and mixture weights.
    Large-tree shortlists are separately ranked by regional likelihood gain.
    """
    parent, parent_edge, component, order, roots = _forest_structure(tree, cuts)
    counts = np.asarray(model.observed, dtype=np.int64).copy()
    sizes = np.ones(tree.n, dtype=np.int64)
    for node in reversed(order):
        if parent[node] != node:
            counts[parent[node]] += counts[node]
            sizes[parent[node]] += sizes[node]
    if np.any(counts[roots] == 0):
        return []  # Splitting cannot repair a region missing from a whole block.
    ranked = []
    for node in order:
        if parent[node] == node:
            continue
        root = component[node]
        evidence = int(np.minimum(counts[node], counts[root] - counts[node]).min())
        if evidence > 0:
            balance = int(min(sizes[node], sizes[root] - sizes[node]))
            ranked.append((-evidence, -balance, int(parent_edge[node])))
    return [edge for _, _, edge in sorted(ranked)]


def _likelihood_ranked_edges(model, tree, edges, cuts=()):
    """Coarse regional profile gains, with ID-free content ties, not final scores.

    Stream fixed-grid columns through postorder sums and impossible counts:
    O(N R G) work and O(N) storage for fixed G, without per-edge member gathers.
    The mixed lower endpoint ensures even a narrow original box has a grid
    point. Resolution can still miss modes; no likelihood bound is claimed.
    """
    parent, parent_edge, component, order, _ = _forest_structure(tree, cuts)
    children = {int(parent_edge[node]): node for node in order if parent[node] != node}
    nodes = np.asarray([children[edge] for edge in edges], dtype=int)
    if not len(nodes):
        return [], {"method": "regional_grid_profile_gain_v1", "ranked_edges": 0}
    grid = np.unique(np.r_[np.linspace(0., 1., 33), 1e-6])
    width = 8
    gains = np.zeros(len(nodes))
    roots = component[nodes]
    for region in range(model.r):
        left, right, whole = [np.full(len(nodes), -np.inf) for _ in range(3)]
        for first in range(0, len(grid), width):
            values = model.regional_profile_columns(region, grid[first:first+width])
            if np.isnan(values).any() or np.isposinf(values).any():
                raise ValueError("Proposal profiles must be finite or negative infinity")
            invalid = np.isneginf(values).astype(np.int64)
            values = np.where(invalid, 0., values)
            for node in reversed(order):
                if parent[node] != node:
                    values[parent[node]] += values[node]
                    invalid[parent[node]] += invalid[node]
            left = np.maximum(left, np.where(invalid[nodes] == 0, values[nodes], -np.inf).max(axis=1))
            right = np.maximum(right, np.where(invalid[roots]-invalid[nodes] == 0,
                values[roots]-values[nodes], -np.inf).max(axis=1))
            whole = np.maximum(whole, np.where(invalid[roots] == 0, values[roots], -np.inf).max(axis=1))
        finite = np.isfinite(left) & np.isfinite(right) & np.isfinite(whole)
        # A failed approximation never removes a supported cut from eligibility.
        # Rank unresolved profiles first for exact evaluation, not as impossible.
        gains += np.subtract(left+right, whole, out=np.full(len(nodes), np.inf), where=finite)
    keys, sizes = list(model.observation_keys()), np.ones(tree.n, dtype=int)
    modulus = 1 << 256
    for node in reversed(order):
        if parent[node] != node:
            keys[parent[node]] = (keys[parent[node]]+keys[node]) % modulus
            sizes[parent[node]] += sizes[node]
    # Quantize only the heuristic rank to suppress reduction-roundoff tie breaks.
    # The exact final refit/score remains untouched.
    scale = max(1., float(np.max(np.abs(gains[np.isfinite(gains)]), initial=0.)))
    quantum = 2. ** (int(np.floor(np.log2(scale)))-32)
    ranks = [None if not np.isfinite(gain) else int(np.rint(gain/quantum)) for gain in gains]
    def key(index):
        node, root = nodes[index], roots[index]
        content = tuple(sorted(((int(sizes[node]), keys[node]),
                                (int(sizes[root]-sizes[node]), (keys[root]-keys[node]) % modulus))))
        rank = ranks[index]
        return (rank is not None, 0 if rank is None else -rank, content, edges[index])
    ordered = sorted(range(len(edges)), key=key)
    return [edges[i] for i in ordered], {
        "method": "regional_grid_profile_gain_v1", "ranked_edges": len(edges),
        "grid_points": len(grid), "grid_column_block": width,
        "proxy_rank_quantum": quantum, "proxy_tied_edges": len(ranks)-len(set(ranks)),
        "unresolved_profiles": sum(rank is None for rank in ranks),
        "tie_rule": "quantized_gain_then_ID_free_observation_multisets_then_edge",
        "maximum_profile_gain": float(np.max(gains)) if np.isfinite(gains).all() else None,
    }


def _add_supported_proposals(model, tree, bank, capacities, seeds, work):
    """One bounded split/exchange phase; its winners join the ordinary frontier."""
    unsupported = {}
    for seed in seeds:
        cuts = tuple(sorted(map(int, seed["cuts"])))
        reason = str(bank.ineligible.get(cuts, {}).get("reason", ""))
        if reason.startswith(("unsupported_cluster_region", "empty_cluster_region_box")):
            requested_k = int(seed["requested_k"])
            unsupported[cuts] = min(unsupported.get(cuts, requested_k), requested_k)
    repair_capacity = min((k for k in unsupported.values() if k >= 2), default=None)
    requested = 2 if 2 in capacities else repair_capacity
    edges = _supported_split_edges(model, tree) if requested is not None else []
    rank_diagnostic = {"method": "all_supported_splits_within_budget"}
    if len(edges) > POLICY.split_slots:
        edges, rank_diagnostic = _likelihood_ranked_edges(model, tree, edges)
    chosen = sorted(edges) if len(edges) <= POLICY.split_slots else edges[:POLICY.split_slots]
    work.spend("split", len(chosen))
    attempted = set(getattr(bank, "by_cuts", {})) | set(bank.ineligible)

    def consider(cuts, labels, centers, provenance, *, parent=None):
        attempted.add(tuple(cuts))
        return bank.consider(cuts, labels, centers, provenance, parent=parent)

    if getattr(bank, "weight_batch_size", 1) > 1:
        proposals = (((edge,), partition_from_cuts(tree, (edge,)), None, {
            "route": "supported_one_edge_split", "requested_k": requested,
            "occupied_q": 2, "zero_jumps_allowed": True}, None) for edge in chosen)
        candidates = bank.consider_independent(proposals)
        for edge, candidate in zip(chosen, candidates):
            attempted.add((edge,))
    else:
        for edge in chosen:
            cuts = (edge,)
            consider(cuts, partition_from_cuts(tree, cuts), None, {
                "route": "supported_one_edge_split", "requested_k": requested,
                "occupied_q": 2, "zero_jumps_allowed": True,
            })
    exchanged, scanned, exchange_slots = set(), 0, 0
    for original in sorted(unsupported)[:POLICY.exchange_parents]:
        if len(original) < 2:
            continue  # One-cut alternatives are already covered above.
        for removed in original:
            base = tuple(edge for edge in original if edge != removed)
            scanned += 1
            for edge in _supported_split_edges(model, tree, base):
                cuts = tuple(sorted((*base, edge)))
                work.spend("exchange")
                exchange_slots += 1
                exchanged.add(cuts)
                consider(cuts, partition_from_cuts(tree, cuts), None, {
                    "route": "supported_cut_exchange", "requested_k": unsupported[original],
                    "occupied_q": len(cuts) + 1, "parent_cuts": original,
                    "removed_edge": removed, "replacement_edge": edge,
                })
                if exchange_slots == POLICY.exchange_slots:
                    break
            if exchange_slots == POLICY.exchange_slots:
                break
        if exchange_slots == POLICY.exchange_slots:
            break
    evaluated = sum((edge,) in attempted for edge in edges)
    return {"split_limit": POLICY.split_slots,
            "supported_one_edge_splits": len(edges), "one_edge_proposals": len(chosen),
            "ranking": rank_diagnostic,
            "evaluated_supported_one_edge_splits": evaluated,
            "supported_one_edge_fraction": evaluated/len(edges) if edges else None,
            "all_supported_one_edge_splits_considered": requested is not None and evaluated == len(edges),
            "exchange_parent_limit": POLICY.exchange_parents,
            "exchange_proposal_limit": POLICY.exchange_slots,
            "exchange_proposals": exchange_slots, "distinct_exchanges": len(exchanged),
            "exchange_forests_scanned": scanned,
            "exhaustive_tree_partition_search": False}


def _repair_seed(model, tree, bank, seed, pilot, work):
    """Width-two structural beam; unsupported intermediates can be expanded.

    Ranking work is not a numerical slot. Empty intersections cannot be fixed
    by adding members. Losing a beam alternative is never an impossibility
    proof, and numerical failures never become structural exclusions.
    """
    original = tuple(sorted(map(int, seed["cuts"])))
    reason = str(bank.ineligible.get(original, {}).get("reason", ""))
    report = {"seed_requested_k": int(seed["requested_k"]), "seed_cuts": original,
              "width": POLICY.repair_width, "depths": [], "outcome": "not_needed"}
    if reason.startswith("empty_cluster_region_box:"):
        return dict(report, outcome="impossible_by_coarsening", reason=reason)
    outcome_index = bank._outcomes.get((("tree", original), None, bank.numerical_policy))
    outcome = None if outcome_index is None else bank._outcome_records[outcome_index].outcome
    if outcome == "unresolved_numerical":
        return dict(report, outcome="numerical_unresolved", reason=reason)
    if not (reason.startswith("unsupported_cluster_region:") or
            (reason == "zero_mixture_weight" and outcome == "completed_ineligible")):
        return report
    report["seed_rejection"] = reason

    def rank(cuts):
        labels = partition_from_cuts(tree, cuts)
        deficits, distance = 0, 0.
        for group in range(len(cuts)+1):
            members = labels == group
            if np.any(model.lower[members].max(axis=0) > model.upper[members].min(axis=0)):
                return None
            observed = model.observed[members]
            deficits += int(np.count_nonzero(~observed.any(axis=0)))
            values = pilot[members]
            count = observed.sum(axis=0)
            mean = np.divide(np.where(observed, values, 0.).sum(axis=0), count,
                             out=np.zeros(model.r), where=count > 0)
            distance += float(np.where(observed, (values-mean)**2, 0.).sum())
        return deficits, distance, cuts

    frontier, unresolved, completed_ineligible = [original], False, False
    while frontier and frontier[0]:
        alternatives = {tuple(edge for edge in cuts if edge != removed)
                        for cuts in frontier for removed in cuts}
        ranked = sorted(value for cuts in alternatives if (value := rank(cuts)) is not None)
        chosen = ranked[:POLICY.repair_width]
        next_frontier = []
        report["depths"].append({"occupied_q": len(frontier[0]), "alternatives": len(alternatives),
                                "admissible_boxes": len(ranked), "selected": len(chosen)})
        success = False
        for deficits, _, cuts in chosen:
            work.spend("repair")
            candidate = bank.consider_cuts(cuts, None, {"route": "supported_coarsening_repair",
                "requested_k": seed["requested_k"], "occupied_q": len(cuts)+1,
                "parent_cuts": original, "support_deficits": deficits})
            success |= candidate is not None
            if candidate is None:
                next_frontier.append(cuts)
            if not deficits and candidate is None:
                key = (("tree", cuts), None, bank.numerical_policy)
                outcome = bank._outcome_records[bank._outcomes[key]].outcome
                unresolved |= outcome == "unresolved_numerical"
                completed_ineligible |= outcome == "completed_ineligible"
        if success:
            return dict(report, outcome="repaired")
        frontier = next_frontier
    return dict(report, outcome=("numerical_unresolved" if unresolved else "repair_budget_exhausted"),
                completed_ineligible=completed_ineligible, impossibility_proven=False)


def _tree_frontier(candidates):
    """Keep the exact score winner, then favor uncovered cut edges.

    Score-only beams can discard a useful native edge at every depth. This
    small deterministic coverage beam changes proposals, never eligibility or
    final scoring. The number of expanded memberships remains explicitly bound.
    """
    remaining = sorted(candidates, key=_score_key)
    chosen, covered = [], set()
    while remaining and len(chosen) < POLICY.tree_width:
        candidate = (remaining[0] if not chosen else min(remaining,
            key=lambda c: (-len(set(c.cuts)-covered), _score_key(c))))
        remaining.remove(candidate)
        chosen.append(candidate)
        covered.update(candidate.cuts)
    return chosen


def fit_regions(model, max_clusters=DEFAULT_MAX_CLUSTERS, capacities=None, *, progress_callback=None,
                audit_scope=DEFAULT_AUDIT_SCOPE):
    """Bound tree exploration, then certify the requested exact-state winners.

    Truth is deliberately absent. Every R>=1 uses this same forest continuation,
    conditional refitter, candidate registry and complete single-move final audit.
    """
    from ..kernel.solver import generate_candidates
    from .progress import ProgressReporter

    device_proposals = getattr(getattr(model, "backend", None), "kind", None) == "cuda"
    validate_audit_scope(audit_scope)
    weight_batch_size = 32 if device_proposals else 1
    continuation_batch_size = 4 if device_proposals else 1
    progress = None if progress_callback is None else ProgressReporter(model, progress_callback)

    total_started = perf_counter()
    telemetry_started = dict(getattr(model, "telemetry", {}))
    if isinstance(max_clusters, bool) or not isinstance(max_clusters, (int, np.integer)) or max_clusters < 1:
        raise ValueError("max_clusters must be a positive integer")
    max_clusters = int(max_clusters)
    work = SearchWork(min(max_clusters, model.n))
    if capacities is None:
        capacities = tuple(range(1, min(max_clusters, model.n) + 1))
    else:
        capacities = tuple(capacities)
        if not capacities or any(isinstance(k, bool) or int(k) != k or
                                 not 1 <= k <= min(max_clusters, model.n) for k in capacities):
            raise ValueError("capacities must contain valid requested cluster counts")
        capacities = tuple(sorted(set(map(int, capacities))))
    times = {}
    if progress is not None:
        progress.start_phase("initialization")
    started = perf_counter()
    pilot, pilot_diagnostics = model.pilots()
    times["initialization_seconds"] = perf_counter() - started
    if progress is not None:
        progress.emit(complete=True)
        progress.start_phase("tree")
    started = perf_counter()
    tree = build_tree(pilot, model.observed, model.mutation_ids)
    times["tree_seconds"] = perf_counter() - started
    if progress is not None:
        progress.emit(complete=True)
        progress.start_phase("continuation")
    started = perf_counter()
    continuation = {} if continuation_batch_size == 1 else {"native_batch_size": continuation_batch_size}
    seeds = (generate_candidates(model, tree, pilot, capacities, keep_raw_phi=False, **continuation)
             if progress is None else generate_candidates(
                 model, tree, pilot, capacities, keep_raw_phi=False,
                 progress_callback=lambda operation: progress.completed_operation(None, None, operation),
                 **continuation))
    times["continuation_seconds"] = perf_counter() - started
    if progress is not None:
        progress.emit(complete=True)
        progress.start_phase("candidate_search")
    bank_options = {} if weight_batch_size == 1 and not device_proposals else {
        "weight_batch_size": weight_batch_size, "device_proposals": device_proposals}
    bank = (_CandidateBank(model, tree, **bank_options) if progress is None else
            _CandidateBank(model, tree, progress=progress.completed_candidate,
                           request_progress=progress.completed_operation,
                           audit_progress=progress.membership_audit, **bank_options))
    seeds_diagnostics, requested_seeds = [], set()
    started = perf_counter()
    for index, seed in enumerate(seeds):
        cuts = tuple(sorted(map(int, seed["cuts"])))
        requested = int(seed["requested_k"])
        if progress is not None:
            progress.seed_index, progress.requested_k, progress.occupied_q = index, requested, len(cuts)+1
        if len(cuts) >= requested or requested not in capacities or requested in requested_seeds:
            raise ValueError("Generated seed exceeds its declared capacity")
        requested_seeds.add(requested)
        labels = canonical_partition(seed["labels"])
        if not np.array_equal(labels, partition_from_cuts(tree, cuts)):
            raise ValueError("Native candidate labels disagree with its support")
        native = {
            "route": "native", "seed_index": index, "requested_k": requested,
            "occupied_q": len(cuts) + 1, "native_collapsed": len(cuts) + 1 < requested,
        }
        seeds_diagnostics.append({**native, "cuts": cuts, "solver": seed["diagnostics"]})
        work.spend("native")
        bank.consider(cuts, labels, None, native)
    if requested_seeds != set(capacities):
        raise ValueError("Continuation did not return every requested capacity")
    times["native_refits_seconds"] = perf_counter()-started
    started = perf_counter()
    repairs = [_repair_seed(model, tree, bank, seed, pilot, work) for seed in seeds]
    times["support_repair_seconds"] = perf_counter()-started
    started = perf_counter()
    if progress is not None:
        progress.emit(bank, complete=True)
        progress.start_phase("supplemental_search", bank)
    supported_proposals = _add_supported_proposals(model, tree, bank, capacities, seeds, work)
    times["supplemental_search_inclusive_seconds"] = perf_counter()-started
    if progress is not None:
        progress.emit(bank, complete=True)
        progress.start_phase("coarsening", bank)
    started = perf_counter()
    frontier_counts, boundary_seconds = {}, 0.
    for q in range(min(max_clusters, model.n), 1, -1):
        # Freeze once at this q. General memberships never enter the tree beam.
        frontier = _tree_frontier(bank.tree_by_q.get(q, {}).values())
        frontier_counts[q] = len(frontier)
        for parent in frontier:
            boundary_started = perf_counter()
            advanced = refine_partition(model, tree, parent, bank.consider, prepare=bank.prepare,
                max_proposals=POLICY.boundary_slots, device_proposals=device_proposals,
                consume_slot=lambda: work.spend("boundary"))
            for candidate in advanced:
                parent = candidate
            boundary_seconds += perf_counter()-boundary_started
            children = []
            for removed in parent.cuts:
                work.spend("coarsening")
                cuts = tuple(edge for edge in parent.cuts if edge != removed)
                children.append((cuts, partition_from_cuts(tree, cuts), None,
                    {"route": "bounded_coarsening", "parent_state_index": parent.state_index,
                     "parent_cuts": parent.cuts, "occupied_q": q-1, "removed_edge": removed}, None))
            bank.consider_independent(children)
    times["coarsening_inclusive_seconds"] = perf_counter()-started
    times["boundary_inclusive_seconds"] = boundary_seconds
    if progress is not None:
        progress.emit(bank, complete=True)
        progress.start_phase("reassignment", bank)
    if not bank.candidates:
        reasons = sorted({entry["reason"] for entry in bank.ineligible.values()})
        raise ValueError(f"No supported finite-score tree candidate: {reasons}")
    # Finish the entire original bank before broadening the membership family.
    bank.original_selected = bank.best_overall
    original_count = len(bank.candidates)
    from .proposals import expand_memberships
    reassignment = expand_memberships(model, bank, work=work)
    bank.compact_payloads()
    times["reassignment_inclusive_seconds"] = reassignment.get("seconds", 0.0)
    if progress is not None:
        progress.emit(bank, complete=True)
        progress.start_phase("membership_audit", bank)
    from .membership_audit import refine_memberships
    started = perf_counter()
    terminal_audits, audit_passes = refine_memberships(model, bank, batch_size=weight_batch_size,
                                                      audit_scope=audit_scope)
    times["membership_audit_inclusive_seconds"] = perf_counter()-started
    if progress is not None:
        progress.emit(bank, complete=True)
        progress.start_phase("selection", bank)
    selected = bank.best_overall
    if selected.score > bank.original_selected.score:
        raise AssertionError("Reassignment lost the original selected score")
    unique_partitions = len(bank.by_cuts)
    for candidate in bank.by_membership.values():
        labels = candidate.fit.labels
        cuts = tuple(np.flatnonzero(labels[tree.edges[:, 0]] != labels[tree.edges[:, 1]]))
        if cuts not in bank.by_cuts or not np.array_equal(partition_from_cuts(tree, cuts), labels):
            unique_partitions += 1
    times["refit_and_score_seconds"] = bank.refit_seconds
    times["move_preparation_seconds"] = bank.preparation_seconds
    times["bank_registration_seconds"] = bank.registration_seconds
    times["fit_total_seconds"] = perf_counter() - total_started
    for name in ("scalar_refit_seconds", "weight_seconds"):
        if name in getattr(model, "telemetry", {}):
            times[name] = model.telemetry[name] - telemetry_started.get(name, 0.0)
    if progress is not None:
        progress.requested_k, progress.occupied_q = max_clusters, len(selected.fit.centers)
        progress.emit(bank, complete=True)
    # Count shared parent arrays once, not once per edit; this remains logical
    # retained payload, not Python containers, numerical caches, or process RSS.
    label_keys = {id(c.fit.membership.label_key): c.fit.membership.label_key
                  for c in bank.candidates if c.membership_kind == "general"}
    label_bases = {id(key.base): key.base for key in label_keys.values()}
    label_bytes = sum(base.nbytes for base in label_bases.values()) + sum(
        8*len(key.order)+16 for key in label_keys.values() if key.row is not None)
    diagnostics = {
        "diagnostics_schema_version": 3,
        "pipeline": ALGORITHM,
        "tree_identity": tree.identity, "tree_kind": "mutation_similarity_not_phylogeny",
        "pilots": pilot_diagnostics, "seeds": seeds_diagnostics,
        "capacities": capacities, "search_policy": POLICY.name, "search_work": work.summary(),
        "support_repairs": repairs, "frontier_counts": frontier_counts,
        "frontier_policy": "score_winner_then_uncovered_cuts_then_exact_score_key",
        "audit_scope": audit_scope,
        "selected_count_certified": True,
        "all_represented_counts_certified": len(terminal_audits) == len(bank.best_by_q),
        "supported_proposals": supported_proposals,
        "reassignment": reassignment,
        "membership_audit_passes": audit_passes,
        "original_tree_candidate_count": original_count,
        "original_tree_selected_score": bank.original_selected.score,
        "selected_membership_kind": selected.membership_kind,
        "eligible_scored_candidates": len(bank.candidates),
        "unique_eligible_partitions": unique_partitions,
        "ineligible_candidate_count": len(bank.ineligible),
        "ineligible_candidates": list(bank.ineligible.values()),
        "general_ineligible_candidate_count": len(bank.general_rejections),
        "unsupported_candidate_count": sum(str(entry["reason"]).startswith(
            "unsupported_cluster_region") for entry in bank.ineligible.values()),
        "unsupported_candidates": [entry for entry in bank.ineligible.values()
                                   if str(entry["reason"]).startswith("unsupported_cluster_region")],
        "duplicate_routes": bank.duplicate_routes, "timings": times,
        "refit_calls": bank.refit_calls, "attempt_cache_hits": bank.attempt_cache_hits,
        "structural_cache_hits": bank.structural_cache_hits,
        "prepared_single_moves": bank.prepared_moves,
        "candidate_numeric_payload_bytes": label_bytes + sum(candidate.fit.centers.nbytes
            + candidate.fit.weights.nbytes
            + (0 if candidate.fit.coordinate_fits is None else candidate.fit.coordinate_fits.nbytes)
            + (8 * len(candidate.cuts) if candidate.membership_kind == "tree" else 0)
            for candidate in bank.candidates),
        "candidate_membership_storage": "tree_cuts_or_shared_parent_single_edits_or_explicit_labels",
        "selection": "minimum_over_complete_eligible_scored_bank",
        "global_optimality_certified": False,
    }
    return TreeFit(selected, bank.candidates, tree, np.asarray(pilot), diagnostics,
                   bank.compact_attempt_links(), tuple(bank._outcome_records), terminal_audits, audit_scope)
