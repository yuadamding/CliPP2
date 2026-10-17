"""Tree proposals, bounded general memberships and objective-bound selection.

The winner is reconciled against every eligible scored candidate, including
intermediate refinement states. Exhaustive enumeration covers only connected
coarsenings of generated seeds and supported one-edge splits of small trees,
not all possible tree partitions or continuous optima.
"""

from dataclasses import dataclass, field, replace
from itertools import combinations
from time import perf_counter

import numpy as np

from .refinement import canonical_partition, refine_partition, tree_adjacency
from ..config import DEFAULT_MAX_CLUSTERS, EXHAUSTIVE_MAX_CLUSTERS, SearchPolicy
from ..kernel import ALGORITHM, MAX_CLUSTERS
from ..kernel.model import RefittedPartition, RegionalModel
from ..kernel.partitions import TreePartitionCache
from ..kernel.topology import build_tree


@dataclass(frozen=True, eq=False)
class _TreeRefit:
    """Compact record; decoded arrays belong only to the shared bounded LRU."""

    tree: object
    cuts: tuple
    centers: np.ndarray
    weights: np.ndarray
    clonal_cluster: int
    conditional_log_likelihood: float
    mixture_log_likelihood: float
    complexity_penalty: float
    score: float
    eligible: bool
    reason: object
    diagnostics: dict
    partition_cache: object
    groups: tuple | None = None
    selection_log_likelihood: float = np.nan
    classification_count_term: float = np.nan
    selection_objective: str = "classification"

    @property
    def labels(self):
        return (self.partition_cache.labels(self.cuts) if self.groups is None else
                self.partition_cache.get_membership(self.cuts, self.groups).labels)

    @classmethod
    def from_fit(cls, tree, cuts, fit, partition_cache, groups=None):
        return cls(tree=tree, cuts=cuts, centers=fit.centers, weights=fit.weights,
                   clonal_cluster=fit.clonal_cluster,
                   conditional_log_likelihood=float(fit.conditional_log_likelihood),
                   mixture_log_likelihood=float(fit.mixture_log_likelihood),
                   complexity_penalty=float(fit.complexity_penalty), score=float(fit.score),
                   eligible=bool(fit.eligible), reason=fit.reason, diagnostics=fit.diagnostics,
                   partition_cache=partition_cache, groups=groups,
                   selection_log_likelihood=getattr(fit, "selection_log_likelihood", np.nan),
                   classification_count_term=getattr(fit, "classification_count_term", np.nan),
                   selection_objective=getattr(fit, "selection_objective", "legacy_mixture"))


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

    @property
    def clonal_cluster(self):
        return self.selected.refit.clonal_cluster


def cut_subsets(cuts):
    """Small-K reference family only; large capacities use bounded search."""
    cuts = tuple(sorted(set(map(int, cuts))))
    if len(cuts) >= EXHAUSTIVE_MAX_CLUSTERS:
        raise ValueError(f"Exhaustive coarsening supports at most {EXHAUSTIVE_MAX_CLUSTERS} clusters")
    for size in range(len(cuts) + 1):
        yield from combinations(cuts, size)


def _score_key(candidate):
    # Preserve the exact old tree ordering, preferring old records on an exact
    # cross-family tie. K comes from occupied centers, never crossing edges.
    tree = candidate.membership_kind == "tree"
    if tree:
        membership = candidate.cuts
    elif isinstance(candidate.refit, _TreeRefit) and candidate.refit.groups is not None:
        # Bounded-search ties use the compact component encoding; selecting the
        # winner must not decode every historical N-vector. The small-K route
        # retains its original dense lexicographic-label tie order.
        membership = (1, candidate.refit.cuts, candidate.refit.groups)
    else:
        membership = (0, tuple(candidate.refit.labels))
    return (candidate.score, len(candidate.refit.centers), 0 if tree else 1,
            membership, candidate.refit.clonal_cluster)


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
    adjacency = tree_adjacency(tree, cuts)
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
    ineligible = (~model.clonal_eligible).astype(np.int64)
    for node in reversed(order):
        if parent[node] != node:
            counts[parent[node]] += counts[node]
            sizes[parent[node]] += sizes[node]
            ineligible[parent[node]] += ineligible[node]
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
            isolates_anchor = ineligible[node] == 0 or ineligible[root]-ineligible[node] == 0
            ranked.append((not isolates_anchor, -evidence, -balance, int(parent_edge[node])))
    return [item[-1] for item in sorted(ranked)]


def _anchor_isolating_splits(model, tree, cuts=()):
    """All edges with an all-eligible side, in one linear forest traversal."""
    parent, parent_edge, component, order, _ = _forest_structure(tree, cuts)
    bad = (~model.clonal_eligible).astype(np.int64)
    for node in reversed(order):
        if parent[node] != node:
            bad[parent[node]] += bad[node]
    return {int(parent_edge[node]) for node in order if parent[node] != node
            and (bad[node] == 0 or bad[component[node]]-bad[node] == 0)}


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
        if reason.startswith(("unsupported_cluster_region", "empty_cluster_region_box",
                              "no_eligible_clonal_block", "clonal_block_inadmissible")):
            unsupported[cuts] = min(unsupported.get(cuts, MAX_CLUSTERS), int(seed["requested_k"]))
    repair_capacity = min((k for k in unsupported.values() if k >= 2), default=None)
    requested = 2 if 2 in capacities else repair_capacity
    small = tree.n <= 64
    edges = _supported_split_edges(model, tree) if requested is not None else []
    rank_diagnostic = {"method": "all_supported_small_tree_splits"}
    if not small and edges:
        edges, rank_diagnostic = _likelihood_ranked_edges(model, tree, edges)
        # Preserve profile ranks within the anchor-isolating priority class.
        isolating = _anchor_isolating_splits(model, tree)
        edges.sort(key=lambda edge: edge not in isolating)
    chosen = sorted(edges) if small else edges[:32]
    supplemental = {}
    attempted = set(getattr(bank, "by_cuts", {})) | set(bank.ineligible)

    def consider(cuts, labels, centers, provenance):
        attempted.add(tuple(cuts))
        candidate = bank.consider(cuts, labels, centers, provenance)
        if candidate is not None:
            supplemental[candidate.cuts] = candidate
        return candidate

    def decode(cuts):
        if hasattr(bank, "labels"):
            return bank.labels(cuts)
        from ..kernel.topology import partition_from_cuts
        return partition_from_cuts(tree, cuts)

    # The same cheap support-preserving route is used by bounded search.
    # Raw native proposals and solver status remain in their original bank.
    from .support_proposals import supported_native_repairs
    shared_repairs = 0
    for seed in seeds:
        if shared_repairs >= 16:
            break
        seed_labels = seed.get("labels")
        if seed_labels is None:
            seed_labels = decode(tuple(seed["cuts"]))
        for labels, details in supported_native_repairs(model, tree, seed_labels, seed["cuts"],
                                                         16-shared_repairs):
            shared_repairs += 1
            bank.consider_membership(labels, None, {
                "route": "native_support_transformation", "requested_k": seed["requested_k"], **details})

    for edge in chosen:
        cuts = (edge,)
        consider(cuts, decode(cuts), None, {
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
                consider(cuts, decode(cuts), None, {
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
            "shared_support_repair_proposals": shared_repairs,
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
    def __init__(self, model, tree, *, compact_general=False):
        self.model, self.tree = model, tree
        self.compact_general = compact_general
        self.partition_cache = TreePartitionCache(
            model, tree, getattr(getattr(model, "memory", None), "partition_bytes", 0))
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
        self.tree_attempts = {}
        self.fitted_states = {}
        self.anchor_rejections = []
        self.attempt_cache_hits = self.structural_cache_hits = 0
        self.support_conflicts_rejected = self.full_partition_attempts = 0
        self.support_rejection_witnesses = []

    def labels(self, cuts):
        return self.partition_cache.labels(cuts)

    def membership_key(self, labels):
        if not self.compact_general:
            return labels.tobytes()
        cuts = tuple(map(int, np.flatnonzero(labels[self.tree.edges[:, 0]] != labels[self.tree.edges[:, 1]])))
        pieces = self.labels(cuts)
        _, first = np.unique(pieces, return_index=True)
        return cuts, tuple(map(int, labels[first]))

    def consider_variants(self, cuts, labels, centers, provenance, *, clonal_cluster=None):
        """Retain all anchored fits; only common structural errors reject a partition."""
        original_labels = labels
        labels, centers = _canonical_inputs(labels, centers, self.model.n, self.model.r)
        if clonal_cluster is not None:
            old_order = list(dict.fromkeys(np.asarray(original_labels).tolist()))
            clonal_cluster = old_order.index(clonal_cluster)
        tree = cuts is not None
        cuts = tuple(sorted(map(int, cuts))) if tree else None
        state = self.partition_cache.get(cuts) if tree else None
        if tree:
            if not np.array_equal(labels, state.labels):
                raise ValueError("Candidate labels and frozen-tree cuts disagree")
            labels = state.labels
        key = cuts if tree else self.membership_key(labels)
        if not tree:
            state = (self.partition_cache.get_membership(*key) if self.compact_general
                     else self.partition_cache.from_labels(labels))
            labels = state.labels
        partition = ("tree" if tree else "general", key)
        if not tree and key in self.structural_rejections:
            self.structural_cache_hits += 1
            return []
        attempts = self.tree_attempts if tree else self.membership_attempts
        center_key = None if centers is None else centers.tobytes()
        q = int(labels.max())+1
        support = getattr(self.model, "regional_support", None)
        strict = support is not None and support.hard_zero.any()
        structural = None
        for k in range(q):
            if not (state.observed_counts[k] > 0).all():
                structural = "unsupported_cluster_region"
            elif np.any(state.lower[k] > state.upper[k]):
                structural = "empty_cluster_region_box"
        if strict and np.any(state.observed_counts == 0):
            structural = "unsupported_cluster_region"
        if structural is None and state.support_conflict.any():
            structural = "regional_zero_positive_conflict"
            self.support_conflicts_rejected += 1
            if len(self.support_rejection_witnesses) < 8:
                k, region = map(int, np.argwhere(state.support_conflict)[0])
                rows = state.members[k]
                self.support_rejection_witnesses.append({
                    "cluster": k, "region_index": region,
                    "positive_mutation_indices": rows[support.positive[rows, region]][:3].tolist(),
                    "zero_mutation_indices": rows[support.hard_zero[rows, region]][:3].tolist()})
        designations = ([clonal_cluster] if clonal_cluster is not None else
                        [k for k in range(q) if state.clonal_eligible[k]])
        if strict:
            designations = [k for k in designations if state.clonal_eligible[k]]
        rejection = {"cuts": cuts, "reason": structural or (
            "no_zero_compatible_clonal_block" if strict else "no_eligible_clonal_block"),
                     "provenance": [dict(provenance)]}
        if structural or not designations:
            if tree:
                self.ineligible[cuts] = rejection
            else:
                self.general_rejections.append(rejection)
                if structural:
                    self.structural_rejections[key] = rejection
            return []
        returned = []
        pending = []
        for clonal in designations:
            attempt = (key, clonal, center_key)
            if attempt in attempts:
                self.attempt_cache_hits += 1
                candidate = attempts[attempt]
                if candidate is not None:
                    candidate.provenance.append(dict(provenance))
                    self.duplicate_routes += 1
                    returned.append(candidate)
            else:
                pending.append(clonal)
        if not pending:
            return returned
        started = perf_counter()
        self.full_partition_attempts += 1
        # The private enumerator allows a cached subset without repeating the
        # other weight fits; all free blocks are shared within this context.
        refit = self.model._refit_variants
        supports_state = getattr(refit, "__func__", None) is RegionalModel._refit_variants
        fits = refit(labels, pending, centers, **({"partition_state": state} if supports_state else {}))
        try:
            for fitted in fits:
                self.refit_calls += 1
                if fitted.eligible:
                    canonical, aligned = _canonical_inputs(fitted.labels, fitted.centers,
                                                           self.model.n, self.model.r)
                    order = list(dict.fromkeys(np.asarray(fitted.labels).tolist()))
                    if (canonical.shape != labels.shape or not np.array_equal(canonical, labels)
                            or np.asarray(fitted.weights).shape != (q,)
                            or fitted.clonal_cluster not in order):
                        raise ValueError("Conditional refit changed membership or clonal identity")
                    fit = RefittedPartition(labels=canonical, centers=aligned,
                        weights=np.asarray(fitted.weights)[order],
                        clonal_cluster=order.index(fitted.clonal_cluster),
                        conditional_log_likelihood=fitted.conditional_log_likelihood,
                        mixture_log_likelihood=fitted.mixture_log_likelihood,
                        complexity_penalty=fitted.complexity_penalty, score=fitted.score,
                        eligible=True, reason=fitted.reason, diagnostics=dict(fitted.diagnostics),
                        selection_log_likelihood=getattr(fitted, "selection_log_likelihood", np.nan),
                        classification_count_term=getattr(fitted, "classification_count_term", np.nan),
                        selection_objective=getattr(fitted, "selection_objective", "legacy_mixture"))
                else:
                    fit = fitted
                clonal = fit.clonal_cluster
                attempt = (key, clonal, center_key)
                attempts[attempt] = None
                if not fit.eligible:
                    rejection = {"cuts": cuts, "clonal_cluster": clonal,
                                 "reason": fit.reason, "provenance": [dict(provenance)]}
                    self.anchor_rejections.append(rejection)
                    if not tree:
                        self.general_rejections.append(rejection)
                    continue
                authoritative_ll = (fit.mixture_log_likelihood if getattr(
                    self.model, "selection_objective", "legacy_mixture") == "legacy_mixture"
                    else fit.selection_log_likelihood)
                if (not np.isfinite([fit.score, fit.conditional_log_likelihood,
                                    fit.complexity_penalty, authoritative_ll]).all()
                        or not np.array_equal(fit.labels, labels)
                        or not np.array_equal(fit.centers[clonal], np.ones(self.model.r))
                        or np.any(fit.weights <= 0)
                        or not np.isclose(fit.weights.sum(), 1., atol=1e-12, rtol=0)):
                    raise ValueError("Invalid occupied anchored fitted state")
                if strict and not support.assigned_mask(labels, fit.centers).all():
                    raise ValueError("regional_zero_positive_conflict: fitted state violates hard membership")
                fit.diagnostics.update(
                    support_identity=state.support_identity,
                    support_validated_at_admission=True,
                    forced_zero_cluster_coordinates=int(state.forced_zero.sum()))
                identity = (partition, clonal, fit.centers.tobytes(), fit.weights.tobytes())
                candidate = self.fitted_states.get(identity)
                if candidate is None:
                    if tree or self.compact_general:
                        compact = _TreeRefit.from_fit(self.tree, cuts if tree else key[0], fit,
                            self.partition_cache, groups=None if tree else key[1])
                    else:
                        compact = fit
                    candidate = Candidate(cuts=cuts, refit=compact, provenance=[dict(provenance)],
                                          membership_kind="tree" if tree else "general")
                    self.candidates.append(candidate)
                    self.fitted_states[identity] = candidate
                else:
                    candidate.provenance.append(dict(provenance))
                attempts[attempt] = candidate
                returned.append(candidate)
                lookup = self.by_cuts if tree else self.by_membership
                previous = lookup.get(key)
                if previous is None or _score_key(candidate) < _score_key(previous):
                    lookup[key] = candidate
        finally:
            self.refit_seconds += perf_counter()-started
        if tree and not returned and cuts not in self.by_cuts:
            self.ineligible[cuts] = rejection
        return returned

    def consider(self, cuts, labels, centers, provenance, *, clonal_cluster=None):
        variants = self.consider_variants(cuts, labels, centers, provenance,
                                          clonal_cluster=clonal_cluster)
        # Every variant is stored; only boundary-path advancement uses the
        # conditional objective rather than the final selection score.
        key = (lambda c: (-c.refit.conditional_log_likelihood, _score_key(c))) if (
            provenance.get("route") == "boundary_refinement") else _score_key
        return min(variants, key=key, default=None)

    def consider_membership(self, labels, initial_centers, provenance, *, clonal_cluster=None):
        return self.consider(None, labels, initial_centers, provenance,
                             clonal_cluster=clonal_cluster)


def fit_tree(model, max_clusters=DEFAULT_MAX_CLUSTERS, capacities=None, *, search=None,
             checkpoint_path=None, resume=False, checkpoint_stop_after=None):
    """Fit tree proposals followed by bounded free-center joint reassignment.

    Truth is deliberately absent from this interface. The single-region
    kernel specialization lives in the public dispatcher, not this tree search.
    """
    from ..kernel.solver import generate_candidates

    total_started = perf_counter()
    telemetry_started = dict(getattr(model, "telemetry", {}))
    if isinstance(max_clusters, bool) or int(max_clusters) != max_clusters or not 1 <= max_clusters <= MAX_CLUSTERS:
        raise ValueError(f"max_clusters must be an integer in 1..{MAX_CLUSTERS}")
    max_clusters = int(max_clusters)
    from .support_proposals import active_support, support_preflight
    requested_search = SearchPolicy() if search is None else search
    requested_mode = requested_search.mode
    strict = active_support(model) is not None
    search = requested_search.resolve(max_clusters, active_hard_zero=strict)
    if (checkpoint_path is not None or resume or checkpoint_stop_after is not None) and search.mode != "bounded":
        raise ValueError("Search checkpoints require explicit bounded search; no route is changed silently")
    if capacities is None:
        capacities = tuple(range(1, min(max_clusters, model.n) + 1))
    else:
        capacities = tuple(capacities)
        if not capacities or any(isinstance(k, bool) or int(k) != k or
                                 not 1 <= k <= min(max_clusters, model.n) for k in capacities):
            raise ValueError("capacities must contain valid requested cluster counts")
        capacities = tuple(sorted(set(map(int, capacities))))
    support_preflight(model, max(capacities))
    model.check_clonal_feasibility(max(capacities))
    times = {}
    saved = None
    if resume:
        if checkpoint_path is None:
            raise ValueError("resume requires a checkpoint_path")
        from .checkpoint import read_checkpoint
        saved = read_checkpoint(checkpoint_path, model)
        from ..kernel.topology import FrozenTree
        pilot, pilot_diagnostics = saved["pilot"], saved["pilot_diagnostics"]
        tree = FrozenTree(**saved["tree"])
        times.update(saved["timings"])
    started = perf_counter()
    if saved is None:
        pilot, pilot_diagnostics = model.pilots()
        times["initialization_seconds"] = perf_counter() - started
    started = perf_counter()
    if saved is None:
        tree = build_tree(pilot, model.observed, model.mutation_ids)
        times["tree_seconds"] = perf_counter() - started
    started = perf_counter()
    old_seeds = list(saved["seeds"]) if saved is not None else []
    pending_capacities = tuple(k for k in capacities if k not in {seed["requested_k"] for seed in old_seeds})
    if tree.component_count > 1:
        if search.mode != "bounded":
            if requested_mode != "auto":
                raise ValueError("disconnected_topology_requires_bounded_search: use search_policy auto or bounded")
            search = replace(search, mode="bounded")
        # Observation-overlap components have disjoint regional coverage. No
        # native component or cut can acquire evidence in every region. Keep
        # each capacity explicit without running unusable continuation work;
        # general memberships can instead unite pieces across components.
        seeds = old_seeds+[{"requested_k": k, "unavailable_reason": "disconnected_observation_overlap"}
                           for k in pending_capacities]
    else:
        seeds = old_seeds+(list(generate_candidates(model, tree, pilot, pending_capacities))
                           if pending_capacities else [])
    times["continuation_seconds"] = times.get("continuation_seconds", 0.)+perf_counter()-started
    if search.mode == "bounded":
        from .search import fit_bounded
        result = fit_bounded(model, tree, pilot, seeds, capacities, search,
                             pilot_diagnostics=pilot_diagnostics, checkpoint_path=checkpoint_path,
                             resume=resume, checkpoint_stop_after=checkpoint_stop_after, **times)
        result.diagnostics["search_policy"].update(requested_mode=requested_mode, resolved_mode=search.mode)
        return result
    bank = _CandidateBank(model, tree)
    seeds_diagnostics, refined = [], set()
    refinement_seconds, subset_count = 0.0, 0

    def refine_once(proposal):
        nonlocal refinement_seconds
        if proposal is None or (proposal.cuts, proposal.refit.clonal_cluster) in refined:
            return
        refined.add((proposal.cuts, proposal.refit.clonal_cluster))
        started = perf_counter()
        for intermediate in refine_partition(model, tree, proposal, bank.consider):
            refined.add((intermediate.cuts, intermediate.refit.clonal_cluster))
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
            repaired = bank.consider(reduced, bank.labels(reduced), None, {
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
        if not np.array_equal(labels, bank.labels(cuts)):
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
            proposal = bank.consider(subset, bank.labels(subset), None, {
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
    rescue_proposals = 0
    if not bank.candidates:
        for capacity in capacities:
            if capacity < 2:
                continue
            seed = min(seeds, key=lambda s: abs(len(s["cuts"])+1-(capacity-1)))
            base = bank.labels(tuple(seed["cuts"])[:capacity-2])
            labels = canonical_partition(np.where(model.clonal_eligible, 0, base+1))
            rescue_proposals += 1
            bank.consider_membership(labels, None, {
                "route": "eligible_set_anchor_rescue", "requested_k": int(capacity),
                "occupied_q": len(np.unique(labels)), "witness_membership_fixed": False})
    if not bank.candidates:
        reasons = sorted({entry["reason"] for entry in bank.ineligible.values()})
        raise ValueError(f"no_eligible_anchored_candidate: searched proposals failed: {reasons}")
    # Finish the entire original bank before broadening the membership family.
    bank.original_selected = min(bank.candidates, key=_score_key)
    original_count = len(bank.candidates)
    from .reassignment import expand_memberships
    reassignment = (expand_memberships(model, bank, search) if model.r > 1 else
                    {"applied": False, "reason": "single_region_path_unchanged"})
    times["reassignment_inclusive_seconds"] = reassignment.get("seconds", 0.0)
    selected = min(bank.candidates, key=_score_key)
    if selected.score > bank.original_selected.score:
        raise AssertionError("Reassignment lost the original selected score")
    unique_partitions = len(bank.by_cuts)
    for candidate in bank.by_membership.values():
        labels = candidate.refit.labels
        cuts = tuple(np.flatnonzero(labels[tree.edges[:, 0]] != labels[tree.edges[:, 1]]))
        if cuts not in bank.by_cuts or not np.array_equal(bank.labels(cuts), labels):
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
        "search_policy": {"mode": "exhaustive", "exhaustive_capacity_limit": EXHAUSTIVE_MAX_CLUSTERS,
                          "global_optimality_certified": False},
        "tree_identity": tree.identity, "tree_kind": "mutation_similarity_not_phylogeny",
        "pilots": pilot_diagnostics, "seeds": seeds_diagnostics,
        "capacities": capacities, "enumerated_cut_subsets": subset_count,
        "supported_proposals": supported_proposals,
        "anchor_rescue_proposals": rescue_proposals,
        "anchor_ineligible_candidates": bank.anchor_rejections,
        "anchored_work": {name: model.telemetry[name]-telemetry_started.get(name, 0)
                          for name in ("anchored_variant_evaluations", "free_block_refits", "mixture_weight_fits")},
        "reassignment": reassignment,
        "original_tree_candidate_count": original_count,
        "original_tree_selected_score": bank.original_selected.score,
        "original_candidate_count": original_count,
        "original_selected_score": bank.original_selected.score,
        "original_bank_membership_counts": {kind: sum(c.membership_kind == kind
            for c in bank.candidates[:original_count]) for kind in ("tree", "general")},
        "legacy_original_tree_fields_scope": "deprecated aliases for the full pre-reassignment bank, including support transformations",
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
        "partition_cache": bank.partition_cache.diagnostics(),
        "selection": "minimum_over_complete_eligible_scored_bank",
        "global_optimality_certified": False,
    })
