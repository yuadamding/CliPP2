"""Frozen-tree candidate lifecycle and joint observed-mixture selection.

The winner is reconciled against every eligible scored candidate, including
intermediate refinement states. Exhaustive enumeration covers only connected
coarsenings of generated seeds, not all possible trees or continuous optima.
"""

from dataclasses import dataclass, field
from itertools import combinations
from time import perf_counter

import numpy as np

from .refinement import canonical_partition, refine_partition
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
    cuts: tuple
    refit: object
    provenance: list = field(default_factory=list)

    @property
    def score(self):
        return float(self.refit.score)

    @property
    def fit(self):
        return self.refit

    @property
    def requested_k(self):
        return self.provenance[0].get("requested_k", len(self.cuts) + 1)

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
    # Stable structural tie rule; no special clonal preference.
    return candidate.score, len(candidate.cuts), candidate.cuts


class _CandidateBank:
    def __init__(self, model, tree):
        self.model, self.tree = model, tree
        self.candidates = []
        self.by_cuts = {}
        self.ineligible = {}
        self.duplicate_routes = 0
        self.refit_seconds = 0.0

    def consider(self, cuts, labels, centers, provenance):
        cuts = tuple(sorted(map(int, cuts)))
        labels = canonical_partition(labels)
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


def fit_tree(model, max_clusters=10, capacities=None):
    """Fit a free-center joint regional model with one frozen similarity tree.

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
    if not bank.candidates:
        reasons = sorted({entry["reason"] for entry in bank.ineligible.values()})
        raise ValueError(f"No supported finite-score tree candidate: {reasons}")
    selected = min(bank.candidates, key=_score_key)
    times["refit_and_score_seconds"] = bank.refit_seconds
    # Refinement wall time includes its refits; do not add overlapping timers.
    times["refinement_inclusive_seconds"] = refinement_seconds
    times["fit_total_seconds"] = perf_counter() - total_started
    for name in ("scalar_refit_seconds", "weight_seconds"):
        if name in getattr(model, "telemetry", {}):
            times[name] = model.telemetry[name] - telemetry_started.get(name, 0.0)
    return TreeFit(selected, bank.candidates, tree, np.asarray(pilot), {
        "pipeline": "frozen_tree_conditional_centers_joint_mixture_v1",
        "tree_identity": tree.identity, "tree_kind": "mutation_similarity_not_phylogeny",
        "pilots": pilot_diagnostics, "seeds": seeds_diagnostics,
        "capacities": capacities, "enumerated_cut_subsets": subset_count,
        "eligible_scored_candidates": len(bank.candidates),
        "unique_eligible_partitions": len(bank.by_cuts),
        "ineligible_candidate_count": len(bank.ineligible),
        "ineligible_candidates": list(bank.ineligible.values()),
        "unsupported_candidate_count": sum(str(entry["reason"]).startswith(
            "unsupported_cluster_region") for entry in bank.ineligible.values()),
        "unsupported_candidates": [entry for entry in bank.ineligible.values()
                                   if str(entry["reason"]).startswith("unsupported_cluster_region")],
        "duplicate_routes": bank.duplicate_routes, "timings": times,
        "candidate_numeric_payload_bytes": sum(candidate.refit.centers.nbytes
            + candidate.refit.weights.nbytes + 8 * len(candidate.cuts) for candidate in bank.candidates),
        "candidate_membership_storage": "tree_cut_sets; labels reconstructed on demand",
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
