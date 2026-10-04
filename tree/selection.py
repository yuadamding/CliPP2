"""Tree proposals, bounded general memberships and joint-mixture selection.

The winner is reconciled against every eligible scored candidate, including
intermediate refinement states. Exhaustive enumeration covers only connected
coarsenings of generated seeds and supported one-edge splits of small trees,
not all possible tree partitions or continuous optima.
"""

from dataclasses import dataclass, field
from itertools import combinations
from time import perf_counter

import numpy as np

from .refinement import canonical_partition, refine_partition
from ..kernel import ALGORITHM
from ..kernel.model import RefittedPartition
from ..kernel.topology import build_tree, partition_from_cuts


@dataclass(frozen=True, eq=False)
class _TreeRefit:
    """Compact candidate record; reconstruct N labels only when requested."""

    tree: object
    cuts: tuple
    centers: np.ndarray
    weights: np.ndarray
    conditional_log_likelihood: float
    mixture_log_likelihood: float
    complexity_penalty: float
    score: float
    eligible: bool
    reason: object
    diagnostics: dict

    @property
    def labels(self):
        return partition_from_cuts(self.tree, self.cuts)

    @property
    def penalty(self):
        return self.complexity_penalty

    @classmethod
    def from_fit(cls, tree, cuts, fit):
        return cls(tree, cuts, fit.centers, fit.weights,
                   float(fit.conditional_log_likelihood), float(fit.mixture_log_likelihood),
                   float(fit.complexity_penalty), float(fit.score), bool(fit.eligible),
                   fit.reason, fit.diagnostics)


@dataclass(eq=False)
class Candidate:
    cuts: tuple | None
    refit: object
    provenance: list = field(default_factory=list)
    membership_kind: str = "tree"

    @property
    def score(self):
        return float(self.refit.score)

    @property
    def fit(self):
        return self.refit

    @property
    def requested_k(self):
        return self.provenance[0].get("requested_k", len(self.refit.centers))

    @property
    def kind(self):
        return self.provenance[0].get("route", "unspecified")

    @property
    def diagnostics(self):
        return {"provenance": self.provenance, "refit": self.refit.diagnostics}


@dataclass(eq=False)
class TreeFit:
    selected: Candidate
    candidate_bank: list
    tree: object
    pilot: np.ndarray
    diagnostics: dict

    @property
    def labels(self):
        return self.selected.refit.labels

    @property
    def centers(self):
        return self.selected.refit.centers

    @property
    def weights(self):
        return self.selected.refit.weights

    @property
    def score(self):
        return self.selected.score


def cut_subsets(cuts):
    """Stream all connected coarsenings; public K<=10 means <=512 per seed."""
    cuts = tuple(sorted(set(map(int, cuts))))
    if len(cuts) > 9:
        raise ValueError("Exhaustive coarsening supports at most ten clusters")
    for size in range(len(cuts) + 1):
        yield from combinations(cuts, size)


def _score_key(candidate):
    # Preserve the exact old tree ordering, preferring old records on an exact
    # cross-family tie. K comes from occupied centers, never crossing edges.
    tree = candidate.membership_kind == "tree"
    return (candidate.score, len(candidate.refit.centers), 0 if tree else 1,
            candidate.cuts if tree else tuple(candidate.refit.labels))


def _canonical_inputs(labels, centers, n, r):
    """Validate occupied IDs and move center rows with first-occurrence labels."""
    labels = np.asarray(labels)
    if (labels.shape != (n,) or labels.dtype.kind not in "iuf"
            or not np.isfinite(labels).all() or np.any(labels != np.rint(labels))):
        raise ValueError("Candidate labels must be one finite integer per mutation")
    occupied = np.unique(labels)
    if not np.array_equal(occupied, np.arange(len(occupied))):
        raise ValueError("Candidate labels must be occupied consecutive IDs starting at zero")
    labels = labels.astype(np.int64)
    if centers is None:
        return canonical_partition(labels), None
    centers = np.asarray(centers, dtype=float)
    if centers.shape != (len(occupied), r) or not np.isfinite(centers).all():
        raise ValueError("Supplied centers must be finite occupied q by R rows aligned with labels")
    return canonical_partition(labels, centers)


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
    # Sums, invalid counts and complement work arrays share this soft budget;
    # one column is the minimum when N alone exceeds it.
    width = max(1, min(8, (8*1024*1024)//(48*tree.n)))
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


def _add_supported_proposals(model, tree, bank, capacities, seeds):
    """Supplement, never replace, the original continuation/refinement bank.

    For N<=64, all observation-supported one-edge splits are scored when q=2
    is requested (or repairs an unsupported higher-capacity seed). Larger
    trees score at most 32 likelihood-ranked splits. Up to 16 same-q exchanges
    from at most eight unsupported native cut sets repair missed support
    without changing the frozen tree. Each new proposal gets the same refit
    and score. Four strongest supplemental fits share a 16-proposal boundary
    refinement budget; every intermediate is retained in the ordinary bank.
    These fixed proposal counts are not elapsed-time limits or exhaustive
    search claims. Higher-order splits and unexamined large-tree edges remain
    possible search deficits even when every stored candidate is reconciled.
    """
    unsupported = {}
    for seed in seeds:
        cuts = tuple(sorted(map(int, seed["cuts"])))
        reason = str(bank.ineligible.get(cuts, {}).get("reason", ""))
        if reason.startswith(("unsupported_cluster_region", "empty_cluster_region_box")):
            unsupported[cuts] = min(unsupported.get(cuts, 10), int(seed["requested_k"]))
    repair_capacity = min((k for k in unsupported.values() if k >= 2), default=None)
    requested = 2 if 2 in capacities else repair_capacity
    small = tree.n <= 64
    edges = _supported_split_edges(model, tree) if requested is not None else []
    rank_diagnostic = {"method": "all_supported_small_tree_splits"}
    if not small and edges:
        edges, rank_diagnostic = _likelihood_ranked_edges(model, tree, edges)
    chosen = sorted(edges) if small else edges[:32]
    supplemental = {}
    attempted = set(getattr(bank, "by_cuts", {})) | set(bank.ineligible)

    def consider(cuts, labels, centers, provenance):
        attempted.add(tuple(cuts))
        candidate = bank.consider(cuts, labels, centers, provenance)
        if candidate is not None:
            supplemental[candidate.cuts] = candidate
        return candidate

    for edge in chosen:
        cuts = (edge,)
        consider(cuts, partition_from_cuts(tree, cuts), None, {
            "route": "supported_one_edge_split", "requested_k": requested,
            "occupied_q": 2, "zero_jumps_allowed": True,
        })
    exchanged, scanned = set(), 0
    for original in sorted(unsupported)[:8]:
        if len(original) < 2:
            continue  # One-cut alternatives are already covered above.
        for removed in original:
            base = tuple(edge for edge in original if edge != removed)
            scanned += 1
            for edge in _supported_split_edges(model, tree, base):
                cuts = tuple(sorted((*base, edge)))
                if cuts == original or cuts in exchanged:
                    continue
                exchanged.add(cuts)
                consider(cuts, partition_from_cuts(tree, cuts), None, {
                    "route": "supported_cut_exchange", "requested_k": unsupported[original],
                    "occupied_q": len(cuts) + 1, "parent_cuts": original,
                    "removed_edge": removed, "replacement_edge": edge,
                })
                if len(exchanged) == 16:
                    break
            if len(exchanged) == 16:
                break
        if len(exchanged) == 16:
            break
    # Use exact fitted scores, with proposal order retaining content-based ties.
    # No elapsed-time deadline or extension of native refinement is introduced.
    strongest = sorted(supplemental.values(), key=lambda c: c.score)[:4]
    refinement_proposals = refinement_candidates = 0

    def refine_consider(*args):
        nonlocal refinement_proposals
        refinement_proposals += 1
        return consider(*args)

    for candidate in strongest:
        if refinement_proposals == 16:
            break
        refinement_candidates += 1
        for _ in refine_partition(model, tree, candidate, refine_consider,
                                  max_proposals=16-refinement_proposals):
            pass
    evaluated = sum((edge,) in attempted for edge in edges)
    return {"small_tree_node_limit": 64, "large_tree_split_limit": 32,
            "supported_one_edge_splits": len(edges), "one_edge_proposals": len(chosen),
            "ranking": rank_diagnostic,
            "evaluated_supported_one_edge_splits": evaluated,
            "supported_one_edge_fraction": evaluated/len(edges) if edges else None,
            "all_supported_one_edge_splits_considered": requested is not None and evaluated == len(edges),
            "exchange_parent_limit": 8, "exchange_proposal_limit": 16,
            "exchange_proposals": len(exchanged), "exchange_forests_scanned": scanned,
            "refinement_candidate_limit": 4, "refinement_proposal_limit": 16,
            "refinement_candidates": refinement_candidates, "refinement_proposals": refinement_proposals,
            "exhaustive_tree_partition_search": False}


class _CandidateBank:
    def __init__(self, model, tree):
        self.model, self.tree = model, tree
        self.candidates = []
        self.by_cuts = {}
        self.ineligible = {}
        self.duplicate_routes = 0
        self.refit_seconds = 0.0
        self.refit_calls = 0
        # Only the bounded general stage stores explicit membership keys. Old
        # tree candidates stay compact; never materialize a dense label bank.
        self.by_membership = {}
        self.membership_attempts = {}
        self.structural_rejections = {}
        self.general_rejections = []
        self.attempt_cache_hits = self.structural_cache_hits = 0

    def consider(self, cuts, labels, centers, provenance):
        cuts = tuple(sorted(map(int, cuts)))
        labels, centers = _canonical_inputs(labels, centers, self.model.n, self.model.r)
        if not np.array_equal(labels, partition_from_cuts(self.tree, cuts)):
            raise ValueError("Candidate labels and frozen-tree cuts disagree")
        previous = self.by_cuts.get(cuts)
        if cuts in self.ineligible and str(self.ineligible[cuts]["reason"]).startswith(
                ("unsupported_cluster_region", "empty_cluster_region_box")):
            self.ineligible[cuts]["provenance"].append(provenance)
            return None
        if previous is not None and (centers is None or np.array_equal(
                np.asarray(centers), previous.refit.centers)):
            previous.provenance.append(provenance)
            self.duplicate_routes += 1
            return previous
        started = perf_counter()
        self.refit_calls += 1
        fit = self.model.refit(labels, initial_centers=centers)
        self.refit_seconds += perf_counter() - started
        if not fit.eligible:
            self.ineligible[cuts] = {
                "cuts": cuts, "reason": fit.reason, "provenance": [provenance],
            }
            return None
        if not np.isfinite(fit.score):
            raise ValueError("An eligible candidate has a nonfinite score")
        candidate = Candidate(cuts, _TreeRefit.from_fit(self.tree, cuts, fit), [provenance])
        self.candidates.append(candidate)
        if previous is None or _score_key(candidate) < _score_key(previous):
            self.by_cuts[cuts] = candidate
        return candidate

    def consider_membership(self, labels, initial_centers, provenance):
        """Refit a general partition without collapsing distinct fitted states.

        Partition identity, exact refit attempt and retained numerical state are
        separate. Only structural failures persist across supplied-center states.
        No tree candidates or their provenance are mutated by this new route.
        """
        labels, centers = _canonical_inputs(labels, initial_centers, self.model.n, self.model.r)
        key = labels.tobytes()
        attempt = (key, None if centers is None else centers.tobytes())
        if key in self.structural_rejections:
            self.structural_cache_hits += 1
            return None
        if attempt in self.membership_attempts:
            self.attempt_cache_hits += 1
            candidate = self.membership_attempts[attempt]
            if candidate is not None:
                candidate.provenance.append(dict(provenance))
            return candidate
        self.refit_calls += 1
        started = perf_counter()
        try:
            fitted = self.model.refit(labels, initial_centers=centers)
        finally:
            self.refit_seconds += perf_counter()-started
        self.membership_attempts[attempt] = None
        if not fitted.eligible:
            rejection = {"labels": labels.tolist(), "reason": fitted.reason,
                         "provenance": [dict(provenance)]}
            self.general_rejections.append(rejection)
            if str(fitted.reason).startswith(("unsupported_cluster_region", "empty_cluster_region_box")):
                self.structural_rejections[key] = rejection
            return None
        if not np.isfinite(fitted.score):
            raise ValueError("An eligible general candidate has a nonfinite score")
        # The fitter labels index its center/weight rows. Reorder all three
        # together, then verify that it did not change the proposed partition.
        canonical, aligned = _canonical_inputs(fitted.labels, fitted.centers, self.model.n, self.model.r)
        if not np.array_equal(canonical, labels):
            raise ValueError("Conditional refit changed the proposed membership")
        order = list(dict.fromkeys(np.asarray(fitted.labels, dtype=int).tolist()))
        weights = np.asarray(fitted.weights)
        if (weights.shape != (len(order),) or not np.isfinite(weights).all()
                or np.any(weights <= 0) or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
            raise ValueError("Eligible fitted weights must be positive normalized occupied rows")
        if not np.isfinite([fitted.conditional_log_likelihood, fitted.mixture_log_likelihood,
                            fitted.complexity_penalty]).all():
            raise ValueError("Eligible fitted likelihoods and penalty must be finite")
        fit = RefittedPartition(canonical, aligned, weights[order],
            float(fitted.conditional_log_likelihood), float(fitted.mixture_log_likelihood),
            float(fitted.complexity_penalty), float(fitted.score), True,
            fitted.reason, dict(fitted.diagnostics))
        candidate = Candidate(None, fit, [dict(provenance)], "general")
        self.candidates.append(candidate)
        self.membership_attempts[attempt] = candidate
        previous = self.by_membership.get(key)
        if previous is None or _score_key(candidate) < _score_key(previous):
            self.by_membership[key] = candidate
        return candidate


def fit_tree(model, max_clusters=10, capacities=None):
    """Fit tree proposals followed by bounded free-center joint reassignment.

    Truth is deliberately absent from this interface. The single-region
    kernel specialization lives in the public dispatcher, not this tree search.
    """
    from ..kernel.solver import generate_candidates

    total_started = perf_counter()
    telemetry_started = dict(getattr(model, "telemetry", {}))
    if isinstance(max_clusters, bool) or int(max_clusters) != max_clusters or not 1 <= max_clusters <= 10:
        raise ValueError("max_clusters must be an integer between one and ten")
    max_clusters = int(max_clusters)
    if capacities is None:
        capacities = tuple(range(1, min(max_clusters, model.n) + 1))
    else:
        capacities = tuple(capacities)
        if not capacities or any(isinstance(k, bool) or int(k) != k or
                                 not 1 <= k <= min(max_clusters, model.n) for k in capacities):
            raise ValueError("capacities must contain valid requested cluster counts")
        capacities = tuple(sorted(set(map(int, capacities))))
    times = {}
    started = perf_counter()
    pilot, pilot_diagnostics = model.pilots()
    times["initialization_seconds"] = perf_counter() - started
    started = perf_counter()
    tree = build_tree(pilot, model.observed, model.mutation_ids)
    times["tree_seconds"] = perf_counter() - started
    started = perf_counter()
    seeds = generate_candidates(model, tree, pilot, capacities)
    times["continuation_seconds"] = perf_counter() - started
    bank = _CandidateBank(model, tree)
    seeds_diagnostics, refined = [], set()
    refinement_seconds, subset_count = 0.0, 0

    def refine_once(proposal):
        nonlocal refinement_seconds
        if proposal is None or proposal.cuts in refined:
            return
        refined.add(proposal.cuts)
        started = perf_counter()
        for intermediate in refine_partition(model, tree, proposal, bank.consider):
            refined.add(intermediate.cuts)
        refinement_seconds += perf_counter() - started

    def repair_unsupported(cuts, requested, parent):
        # A selected capacity is an at-most-K budget. Ordinary coarsening
        # routes target requested q, but support repairs retain their parent's
        # budget even when that lower q was not separately requested. When
        # capacities are the default 1..K, ordinary enumeration covers every
        # reduction and this adds no fits or changes their order.
        for reduced in cut_subsets(cuts):
            if reduced == cuts or len(reduced) + 1 in capacities:
                continue
            repaired = bank.consider(reduced, partition_from_cuts(tree, reduced), None, {
                "route": "supported_coarsening_repair", "requested_k": requested,
                "occupied_q": len(reduced) + 1, "parent_cuts": cuts,
                "parent_route": parent["route"],
                "reason": bank.ineligible[cuts]["reason"],
            })
            refine_once(repaired)

    for index, seed in enumerate(seeds):
        cuts = tuple(sorted(map(int, seed["cuts"])))
        requested = int(seed["requested_k"])
        if len(cuts) >= requested or requested not in capacities:
            raise ValueError("Generated seed exceeds its declared capacity")
        labels = canonical_partition(seed["labels"])
        if not np.array_equal(labels, partition_from_cuts(tree, cuts)):
            raise ValueError("Native candidate labels disagree with its support")
        native = {
            "route": "native", "seed_index": index, "requested_k": requested,
            "occupied_q": len(cuts) + 1, "native_collapsed": len(cuts) + 1 < requested,
        }
        seeds_diagnostics.append({**native, "cuts": cuts, "solver": seed["diagnostics"]})
        # Refit and record the native route before its connected coarsenings.
        native_candidate = bank.consider(cuts, labels, None, native)
        if native_candidate is None:
            repair_unsupported(cuts, requested, native)
        for subset in cut_subsets(cuts):
            subset_count += 1
            if len(subset) + 1 not in capacities:
                continue
            proposal = bank.consider(subset, partition_from_cuts(tree, subset), None, {
                "route": "connected_coarsening", "seed_index": index,
                "seed_requested_k": requested, "requested_k": len(subset) + 1,
                "occupied_q": len(subset) + 1, "seed_cuts": cuts,
            })
            if proposal is None:
                repair_unsupported(subset, len(subset) + 1, {"route": "connected_coarsening"})
            refine_once(proposal)
    started = perf_counter()
    supported_proposals = _add_supported_proposals(model, tree, bank, capacities, seeds)
    times["supplemental_search_inclusive_seconds"] = perf_counter()-started
    if not bank.candidates:
        reasons = sorted({entry["reason"] for entry in bank.ineligible.values()})
        raise ValueError(f"No supported finite-score tree candidate: {reasons}")
    # Finish the entire original bank before broadening the membership family.
    bank.original_selected = min(bank.candidates, key=_score_key)
    original_count = len(bank.candidates)
    from .reassignment import expand_memberships
    reassignment = (expand_memberships(model, bank) if model.r > 1 else
                    {"applied": False, "reason": "single_region_path_unchanged"})
    times["reassignment_inclusive_seconds"] = reassignment.get("seconds", 0.0)
    selected = min(bank.candidates, key=_score_key)
    if selected.score > bank.original_selected.score:
        raise AssertionError("Reassignment lost the original selected score")
    unique_partitions = len(bank.by_cuts)
    for candidate in bank.by_membership.values():
        labels = candidate.refit.labels
        cuts = tuple(np.flatnonzero(labels[tree.edges[:, 0]] != labels[tree.edges[:, 1]]))
        if cuts not in bank.by_cuts or not np.array_equal(partition_from_cuts(tree, cuts), labels):
            unique_partitions += 1
    times["refit_and_score_seconds"] = bank.refit_seconds
    # Refinement wall time includes its refits; do not add overlapping timers.
    times["refinement_inclusive_seconds"] = refinement_seconds
    times["fit_total_seconds"] = perf_counter() - total_started
    for name in ("scalar_refit_seconds", "weight_seconds"):
        if name in getattr(model, "telemetry", {}):
            times[name] = model.telemetry[name] - telemetry_started.get(name, 0.0)
    return TreeFit(selected, bank.candidates, tree, np.asarray(pilot), {
        "pipeline": ALGORITHM,
        "tree_identity": tree.identity, "tree_kind": "mutation_similarity_not_phylogeny",
        "pilots": pilot_diagnostics, "seeds": seeds_diagnostics,
        "capacities": capacities, "enumerated_cut_subsets": subset_count,
        "supported_proposals": supported_proposals,
        "reassignment": reassignment,
        "original_tree_candidate_count": original_count,
        "original_tree_selected_score": bank.original_selected.score,
        "selected_membership_kind": selected.membership_kind,
        "eligible_scored_candidates": len(bank.candidates),
        "unique_eligible_partitions": unique_partitions,
        "ineligible_candidate_count": len(bank.ineligible),
        "ineligible_candidates": list(bank.ineligible.values()),
        "general_ineligible_candidates": bank.general_rejections,
        "unsupported_candidate_count": sum(str(entry["reason"]).startswith(
            "unsupported_cluster_region") for entry in bank.ineligible.values()),
        "unsupported_candidates": [entry for entry in bank.ineligible.values()
                                   if str(entry["reason"]).startswith("unsupported_cluster_region")],
        "duplicate_routes": bank.duplicate_routes, "timings": times,
        "candidate_numeric_payload_bytes": sum(candidate.refit.centers.nbytes
            + candidate.refit.weights.nbytes + (8 * len(candidate.cuts) if candidate.membership_kind == "tree"
                                               else candidate.refit.labels.nbytes)
            for candidate in bank.candidates),
        "candidate_membership_storage": "compact_tree_cut_sets_plus_bounded_explicit_general_labels",
        "selection": "minimum_over_complete_eligible_scored_bank",
        "global_optimality_certified": False,
    })


def exhaustive_tree_reference(model, tree, *, max_nodes=8):
    """Offline tiny-N reference; continuous scalar refits remain numerical."""
    if model.n > min(int(max_nodes), 8):
        raise ValueError("Exhaustive tree reference is restricted to N<=8")
    bank = _CandidateBank(model, tree)
    for cuts in cut_subsets(range(len(tree.edges))):
        bank.consider(cuts, partition_from_cuts(tree, cuts), None, {"route": "offline_exhaustive"})
    if not bank.candidates:
        raise ValueError("No supported exhaustive tree candidate")
    return {
        "selected": min(bank.candidates, key=_score_key), "candidate_bank": bank.candidates,
        "unsupported_candidates": list(bank.ineligible.values()),
        "continuous_global_optimality_certified": False,
    }


def exhaustive_unrestricted_reference(model, *, max_nodes=8):
    """Offline Bell-family comparator for topology restriction, N<=8 only.

    Restricted-growth strings enumerate every unlabeled set partition once.
    Keep only the current winner and counters, not an exponential label bank.
    This is not called by fitting; exhaustive discrete labels cannot certify
    global optimality of the numerical continuous conditional refits.
    """
    if not 1 <= model.n <= min(int(max_nodes), 8):
        raise ValueError("Exhaustive unrestricted reference is restricted to N<=8")
    labels = np.zeros(model.n, dtype=np.int64)

    def partitions(index, maximum):
        if index == model.n:
            yield labels.copy()
            return
        for value in range(maximum + 2):
            labels[index] = value
            yield from partitions(index + 1, max(maximum, value))

    best, best_key = None, None
    total, eligible = 0, 0
    for proposal in partitions(1, 0):
        total += 1
        fitted = model.refit(proposal)
        if not fitted.eligible:
            continue
        if not np.isfinite(fitted.score):
            raise ValueError("An eligible unrestricted reference candidate has a nonfinite score")
        eligible += 1
        key = float(fitted.score), len(fitted.centers), tuple(proposal)
        if best is None or key < best_key:
            best, best_key = fitted, key
    if best is None:
        raise ValueError("No supported unrestricted candidate")
    return {"selected": best, "enumerated_partitions": total,
            "eligible_partitions": eligible, "ineligible_partitions": total - eligible,
            "continuous_global_optimality_certified": False}


def offline_truth_replay(model, tree, truth_labels, selected_labels):
    """Truth-only diagnostic, never called by inference or topology construction."""
    truth, selected = map(canonical_partition, (truth_labels, selected_labels))
    if truth.shape != (model.n,) or selected.shape != (model.n,):
        raise ValueError("Replay labels must align with retained mutation identities")
    q = len(np.unique(truth))
    q_tree = 1 + int(np.count_nonzero(truth[tree.edges[:, 0]] != truth[tree.edges[:, 1]]))
    result = {"truth_clusters": q, "tree_truth_components": q_tree,
              "fragmentation": q_tree - q, "truth_representable": q_tree == q,
              "tree_identity": tree.identity, "partitions": {}}
    for name, labels in (("truth", truth), ("selected", selected)):
        fit = model.refit(labels)
        result["partitions"][name] = {
            "eligible": fit.eligible, "reason": fit.reason,
            "conditional_log_likelihood": float(fit.conditional_log_likelihood),
            "mixture_log_likelihood": float(fit.mixture_log_likelihood),
            "complexity_penalty": float(fit.complexity_penalty), "score": float(fit.score),
        }
    if all(value["eligible"] for value in result["partitions"].values()):
        result["selected_minus_truth_score"] = (
            result["partitions"]["selected"]["score"] - result["partitions"]["truth"]["score"])
    return result
