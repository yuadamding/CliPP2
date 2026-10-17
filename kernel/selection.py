"""Refit distance-to-set chain partitions under an explicit selection objective.

Multiplicity is marginalized over 1..major_cn. Each occupied partition block
has one fitted cellular prevalence (CP), with CCF = CP / purity. The scalar
refit searches a dense grid and every component's binomial mode, then refines
all bracketed maxima; this is a deterministic multimode numerical search, not
a certificate of global optimality. Classification likelihood includes the
published labels and profiles proportions to counts/N. The explicit legacy
route alone fits observed-mixture weights. The nominal parameter convention
counts q-1 free centers and q-1 proportions; one occupied center is fixed at purity.
Adjacent coarsenings and conditional boundary polishing improve the proposal
bank on the same frozen chain. Centers use conditional block refitting;
All proposals are ultimately selected by the named authoritative score.
"""

from collections import OrderedDict
import hashlib
import json
import math
import sys
import time

import numpy as np
import pandas as pd

from .proposals import iter_chain_proposals, _integers as _integer_vector
from . import LEGACY_SCORING_VERSION, CANDIDATE_SEARCH_VERSION
from ..config import EXHAUSTIVE_MAX_CLUSTERS
from .refinement import polish_chain_partition
from .refitting import refit_center
from .scoring import fit_cluster_weights
from .memory import ByteLRU
from .objectives import assigned_objective, score_definition_for, validate_selection_objective


_SCORE_FIELDS = (
    "num_clusters", "bic", "log_likelihood", "conditional_log_likelihood",
    "weight_optimality_gap", "weight_active_score_gap", "selection_objective",
    "score_definition", "selection_log_likelihood", "classification_count_term",
    "mixture_log_likelihood", "anchoring_loss", "anchoring_loss_available",
    "anchoring_loss_unavailable_reason", "proportions_definition",
    "mixture_weight_optimization", "bic_definition",
)


def _selection_objective(model):
    return validate_selection_objective(getattr(model, "selection_objective", "classification"))


def _parameters(result, order):
    """Lossless O(blocks + q) evidence; visits need not follow center order."""
    ordered = result["labels"][order]
    cuts = np.flatnonzero(np.diff(ordered)) + 1
    return json.dumps(
        {
            "cuts": cuts.tolist(),
            "block_labels": ordered[np.r_[0, cuts]].tolist(),
            "centers": result["centers"].tolist(),
            "weights": result["cluster_weights"].tolist(),
            "clonal_cluster": result["clonal_cluster"],
            "selection_objective": result["selection_objective"],
            "score_definition": result["score_definition"],
        },
        separators=(",", ":"),
        allow_nan=False,
    )


class _ChainRefitCache(OrderedDict):
    """One model/order's interval refits and a bounded LRU of likelihood columns."""

    def __init__(self, model, chain_order, max_column_bytes=32 * 1024 * 1024, *, clonal_eligible=None,
                 memory=None, refit_store=None):
        super().__init__()
        self.model = model
        self.selection_objective = _selection_objective(model)
        self.order = np.asarray(chain_order, dtype=np.int64).copy()
        self.ranks = np.empty(len(model), dtype=np.int64)
        self.ranks[self.order] = np.arange(len(model))
        self.columns = OrderedDict()
        self.column_bytes = 0
        self.max_column_bytes = max_column_bytes if memory is None else memory.column_bytes
        self.max_intervals = 16384 if memory is None else memory.refit_bytes//512
        self.refit_store = refit_store
        self.weight_results = ByteLRU(8 * 1024 * 1024 if memory is None else memory.weight_bytes)
        self.clonal_column = np.asarray(model.log_likelihood(model.purity)).copy()
        self.clonal_eligible = np.isfinite(self.clonal_column)
        if clonal_eligible is not None:
            self.clonal_eligible &= np.asarray(clonal_eligible, dtype=bool)
        self.clonal_column[~self.clonal_eligible] = -np.inf
        self.clonal_column = np.frombuffer(self.clonal_column.tobytes(), dtype=float)
        self.clonal_eligible = np.frombuffer(self.clonal_eligible.tobytes(), dtype=bool)
        self.telemetry = dict(
            interval_cache_hits=0,
            interval_refits=0,
            conditional_refit_seconds=0.0,
            likelihood_cache_hits=0,
            likelihood_evaluations=0,
            weight_cache_hits=0,
            weight_solver_calls=0,
            weight_solver_failures=0,
            weight_solver_seconds=0.0,
            free_block_refits=0, mixture_weight_fits=0,
            anchored_variant_evaluations=0, anchored_refit_failures=0, clonal_column_hits=0,
        )

    def get(self, key, default=None):
        value = super().get(key, default)
        if key in self:
            self.move_to_end(key)
            self.telemetry["interval_cache_hits"] += 1
        return value

    def __setitem__(self, key, value):
        if not self.max_intervals:
            return
        if key not in self and len(self) >= self.max_intervals:
            self.popitem(last=False)
        super().__setitem__(key, value)

    def fit_weights(self, centers, counts):
        # Model/order and numerical implementation are fixed by this cache.
        # Ordered float64 centers AND original count initialization are required;
        # duplicate columns can otherwise change exact-zero support.
        key = (fit_cluster_weights, centers.dtype.str, centers.tobytes(), counts.dtype.str, counts.tobytes())
        cached = self.weight_results.get(key)
        if cached is not None:
            self.telemetry["weight_cache_hits"] += 1
            return {**cached, "cluster_weights": cached["cluster_weights"].copy()}
        kernel = self.log_kernel(centers)
        start = time.perf_counter()
        self.telemetry["weight_solver_calls"] += 1
        self.telemetry["mixture_weight_fits"] += 1
        try:
            value = fit_cluster_weights(
                self.model, centers, counts, log_kernel=kernel, telemetry=self.telemetry
            )
        except RuntimeError:
            self.telemetry["weight_solver_failures"] += 1
            raise
        finally:
            self.telemetry["weight_solver_seconds"] += time.perf_counter() - start
        size = (
            sys.getsizeof(key)
            + sum(sys.getsizeof(part) for part in key)
            + sys.getsizeof(value)
            + sys.getsizeof(value["cluster_weights"])
            + 256
        )
        self.weight_results.put(key, {**value, "cluster_weights": value["cluster_weights"].copy()}, size)
        return value

    def block_key(self, rows):
        ranks = self.ranks[rows]
        left, right = int(ranks.min()), int(ranks.max()) + 1
        if right - left != len(rows):
            raise ValueError("Cached refit requires a contiguous chain interval")
        return left, right

    def log_kernel(self, centers):
        columns = []
        for cp in centers:
            if cp == self.model.purity:
                self.telemetry["clonal_column_hits"] += 1
                columns.append(self.clonal_column)
                continue
            key = float(cp)
            column = self.columns.pop(key, None)
            if column is None:
                self.telemetry["likelihood_evaluations"] += 1
                column = self.model.log_likelihood(cp)
                if column.nbytes <= self.max_column_bytes:
                    while self.column_bytes + column.nbytes > self.max_column_bytes:
                        _, removed = self.columns.popitem(last=False)
                        self.column_bytes -= removed.nbytes
                    self.columns[key] = column
                    self.column_bytes += column.nbytes
            else:
                self.telemetry["likelihood_cache_hits"] += 1
                self.columns[key] = column
            columns.append(column)
        return np.column_stack(columns)


def refit_partition_variants(model, labels, cache=None, chain_order=None, *,
                             designations=None, initial_centers=None):
    """Refit blocks; merge exact adjacent equal centers on a supplied chain.

    Without an explicit chain permutation no adjacency is inferred from labels.
    Equal centers separated by another center always remain separate blocks.
    """
    labels = np.atleast_1d(np.asarray(labels))
    if (labels.shape != (len(model),) or labels.dtype.kind not in "iuf"
            or not np.all(np.isfinite(labels)) or np.any(labels != np.rint(labels))):
        raise ValueError("Partition labels must be one integer per mutation")
    if isinstance(cache, _ChainRefitCache) and (
        cache.model is not model or chain_order is None or not np.array_equal(cache.order, chain_order)
        or cache.selection_objective != _selection_objective(model)
    ):
        raise ValueError("Chain refit cache belongs to a different model or order")
    objective = _selection_objective(model)
    clusters, inverse = np.unique(labels, return_inverse=True)
    if initial_centers is not None:
        initial_centers = np.asarray(initial_centers, dtype=float)
        if initial_centers.shape != (len(clusters),):
            raise ValueError("Initial centers must follow sorted occupied label order")
    clonal_column = cache.clonal_column if isinstance(cache, _ChainRefitCache) else model.log_likelihood(model.purity)
    eligible = np.isfinite(clonal_column)
    if designations is None:
        designations = [k for k in clusters if eligible[labels == k].all()]
    elif any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
             or k not in clusters for k in designations):
        raise ValueError("Clonal designation must be an occupied label")
    clonal_indices = [int(np.flatnonzero(clusters == k)[0]) for k in designations]
    free, errors = {}, {}
    for index in range(len(clusters)):
        if not any(clonal != index for clonal in clonal_indices):
            continue
        rows = np.flatnonzero(inverse == index)
        key = cache.block_key(rows) if isinstance(cache, _ChainRefitCache) else rows.tobytes()
        fitted = cache.get(key) if cache is not None else None
        if fitted is None:
            start = time.perf_counter()
            try:
                scalar = model.subset(rows)
                store = cache.refit_store if isinstance(cache, _ChainRefitCache) else None
                fitted = refit_center(scalar, **({'grid_evaluator': lambda grid:
                    store.grid_log_likelihood(rows, 0, grid, scalar=scalar)} if store is not None else {}))
            except (ValueError, RuntimeError) as error:
                errors[index] = str(error)
                continue
            if isinstance(cache, _ChainRefitCache):
                cache.telemetry["interval_refits"] += 1
                cache.telemetry["free_block_refits"] += 1
                cache.telemetry["conditional_refit_seconds"] += time.perf_counter() - start
            if cache is not None:
                cache[key] = fitted
        if initial_centers is not None:
            previous = float(initial_centers[index])
            if 0 <= previous <= model.purity:
                old_ll = float(model.subset(rows).log_likelihood(previous).sum())
                if old_ll > fitted[1]:
                    fitted = (previous, old_ll)
        free[index] = fitted
    if chain_order is not None:
        chain_order = _integer_vector(chain_order, "Frozen chain order", len(model))
        if not np.array_equal(np.sort(chain_order), np.arange(len(model))):
            raise ValueError("Frozen chain order must be a permutation of original row indices")
        validate_chain_labels(inverse[chain_order], len(model), len(clusters))
    for clonal in clonal_indices:
        if isinstance(cache, _ChainRefitCache):
            cache.telemetry["anchored_variant_evaluations"] += 1
        if any(k != clonal for k in errors) or not eligible[inverse == clonal].all():
            if isinstance(cache, _ChainRefitCache):
                cache.telemetry["anchored_refit_failures"] += 1
            continue
        centers = np.asarray([model.purity if k == clonal else free[k][0] for k in range(len(clusters))])
        conditional = math.fsum(float(clonal_column[inverse == k].sum()) if k == clonal
                                else free[k][1] for k in range(len(clusters)))
        fitted_labels = inverse.copy()
        clonal_member = int(np.flatnonzero(inverse == clonal)[0])
        if chain_order is not None:
            ordered_cp = centers[inverse[chain_order]]
            starts = np.r_[True, ordered_cp[1:] != ordered_cp[:-1]]
            fitted_labels[chain_order] = np.cumsum(starts)-1
            centers = ordered_cp[starts]
        order = np.argsort(-centers, kind="stable")
        relabel = np.empty(len(order), dtype=int)
        relabel[order] = np.arange(len(order))
        fitted_labels = relabel[fitted_labels]
        fixed = int(fitted_labels[clonal_member])
        centers = centers[order]
        counts = np.bincount(fitted_labels, minlength=len(centers))
        if objective == "legacy_mixture":
            try:
                scored = (cache.fit_weights(centers, counts) if isinstance(cache, _ChainRefitCache)
                          else fit_cluster_weights(model, centers, counts))
            except (ValueError, RuntimeError):
                if isinstance(cache, _ChainRefitCache):
                    cache.telemetry["anchored_refit_failures"] += 1
                continue
            scored.update(selection_log_likelihood=scored["log_likelihood"],
                          mixture_log_likelihood=scored["log_likelihood"],
                          classification_count_term=np.nan,
                          mixture_weight_optimization="converged_at_fixed_centers",
                          proportions_definition="legacy_observed_mixture_optimal_weights")
        else:
            scored = assigned_objective(conditional, counts, 1, objective)
            scored.update(log_likelihood=scored["selection_log_likelihood"],
                          mixture_log_likelihood=np.nan,
                          weight_optimality_gap=None, weight_active_score_gap=None,
                          mixture_weight_optimization="not_applicable",
                          proportions_definition="occupied_cluster_counts_over_N")
        parameters = 2*(len(centers)-1)
        same_anchor_members = np.array_equal(fitted_labels == fixed, inverse == clonal)
        loss_available = clonal in free and same_anchor_members
        yield {"labels": fitted_labels, "centers": centers, "clonal_cluster": fixed,
               **scored, "conditional_log_likelihood": conditional,
               "selection_objective": objective, "score_definition": score_definition_for(objective),
               "anchoring_loss": (float(free[clonal][1]-clonal_column[inverse == clonal].sum())
                                  if loss_available else None),
               "anchoring_loss_available": loss_available,
               "anchoring_loss_unavailable_reason": (None if loss_available else
                    "anchor_membership_changed_by_exact_center_merge" if not same_anchor_members else
                    "optional_free_anchor_fit_unavailable"),
               "num_clusters": len(centers), "num_parameters": parameters,
               "num_mutations": len(model),
               "bic": -2*scored["selection_log_likelihood"]+parameters*math.log(len(model)),
               "bic_definition": (LEGACY_SCORING_VERSION if objective == "legacy_mixture"
                                  else score_definition_for(objective))}


def refit_partition(model, labels, cache=None, chain_order=None, *, clonal_cluster,
                    initial_centers=None):
    fitted = next(refit_partition_variants(model, labels, cache, chain_order,
                  designations=[clonal_cluster], initial_centers=initial_centers), None)
    if fitted is None:
        raise RuntimeError("anchored_partition_refit_failed")
    return fitted


def validate_chain_labels(labels, length, requested_k):
    """Allow label permutations, but require each occupied label to be one run."""
    labels = _integer_vector(labels, "Chain labels", length)
    starts = np.r_[True, labels[1:] != labels[:-1]]
    block_labels = labels[starts]
    if len(np.unique(block_labels)) != len(block_labels):
        raise ValueError("Each cluster must be contiguous along the frozen chain")
    if len(block_labels) > requested_k:
        raise ValueError("Partition exceeds its requested K block budget")
    return labels


def search_chain_coarsenings(
    model, seeds, chain_order, budgets, cache=None, *, store, replicate=1
):
    """Score every anchored variant of native proposals and their coarsenings."""
    n = len(model)
    chain_order = _integer_vector(chain_order, "Frozen chain order", n)
    if not np.array_equal(np.sort(chain_order), np.arange(n)):
        raise ValueError("Frozen chain order must be a permutation")
    budgets = _integer_vector(budgets, "Cluster budgets")
    if (len(np.unique(budgets)) != len(budgets) or np.any(budgets < 1)
            or np.any(budgets > min(EXHAUSTIVE_MAX_CLUSTERS, n))):
        raise ValueError(f"Expected unique exhaustive K in 1..min({EXHAUSTIVE_MAX_CLUSTERS},N)")
    cache = _ChainRefitCache(model, chain_order) if cache is None else cache
    winners, ranks = {}, {}
    partitions = 0
    chain_digest = hashlib.sha256(np.asarray(chain_order, dtype="<i8").tobytes())

    def fingerprint(cuts):
        digest = chain_digest.copy()
        digest.update(np.asarray(cuts, dtype="<i8").tobytes())
        return digest.hexdigest()

    for cuts, routes in iter_chain_proposals(seeds, chain_order, budgets):
        partitions += 1
        labels = np.empty(n, dtype=int)
        labels[chain_order] = np.searchsorted(cuts, np.arange(n), side="right")
        variants = refit_partition_variants(model, labels, cache, chain_order)
        count = 0
        for result in variants:
            count += 1
            final_cuts = tuple(np.flatnonzero(np.diff(result["labels"][chain_order]))+1)
            for budget, provenance in sorted(routes.items()):
                record = _scored_record(result, chain_order, provenance, replicate)
                record.update(candidate_id=store.next_id(replicate),
                    proposal_partition_sha256=fingerprint(cuts),
                    partition_sha256=fingerprint(final_cuts),
                    chain_cuts=",".join(map(str, cuts)), num_input_blocks=len(cuts)+1)
                record = store.append(record)
                rank = (result["bic"], result["num_clusters"],
                        provenance["candidate_kind"] != "native",
                        provenance["parent_requested_k"], provenance["parent_replicate"],
                        cuts, result["clonal_cluster"])
                if budget not in ranks or rank < ranks[budget]:
                    ranks[budget] = rank
                    winners[budget] = {**record, "cuts": final_cuts, "result": result}
        if not count:
            for provenance in routes.values():
                store.append({**provenance, "replicate": replicate,
                    "candidate_id": store.next_id(replicate), "status": "refit_failed",
                    "error": "no_eligible_anchored_candidate", "publication_eligible": False,
                    "proposal_partition_sha256": fingerprint(cuts),
                    "chain_cuts": ",".join(map(str, cuts)),
                    "candidate_search_version": CANDIDATE_SEARCH_VERSION,
                    "selected_for_k": False})
    for winner in winners.values():
        store.update(replicate, winner["candidate_id"], {"selected_for_k": True})
    return {"replicate": replicate, "winners": winners, "candidates": store,
            "stats": {"unique_partitions": partitions,
                      "scored_budget_candidates": store.count(replicate=replicate, status="scored"),
                      "cached_refit_blocks": len(cache),
                      "likelihood_cache_bytes": cache.column_bytes}}


def _scored_record(result, order, parent, replicate):
    fields = ("requested_k", "parent_requested_k", "parent_replicate", "parent_partition_sha256")
    record = {key: parent[key] for key in fields}
    record.update({key: result[key] for key in (*_SCORE_FIELDS, "clonal_cluster")})
    record.update(replicate=replicate, candidate_kind=parent["candidate_kind"], status="scored",
        partition_parameters=_parameters(result, order),
        publication_eligible=bool(np.all(result["cluster_weights"] > 0)),
        active_mixture_components=int(np.count_nonzero(result["cluster_weights"] > 0)),
        minimum_mixture_weight=float(np.min(result["cluster_weights"])),
        distinct_centers=int(len(np.unique(result["centers"]))),
        joint_mixture_center_mle=False, candidate_search_version=CANDIDATE_SEARCH_VERSION,
        selected_for_k=False)
    return record


def _restore_result(record, order):
    """Restore exact recorded anchored state; never silently rerun a refit."""
    parameters = json.loads(record["partition_parameters"])
    labels = np.empty(len(order), dtype=int)
    labels[order] = np.asarray(parameters["block_labels"], dtype=int)[
        np.searchsorted(parameters["cuts"], np.arange(len(order)), side="right")]
    result = {key: record[key] for key in _SCORE_FIELDS}
    result.update(labels=labels, centers=np.asarray(parameters["centers"], dtype=float),
        cluster_weights=np.asarray(parameters["weights"], dtype=float),
        clonal_cluster=int(parameters["clonal_cluster"]),
        num_parameters=2*(record["num_clusters"]-1), num_mutations=len(order))
    return result


def refine_chain_search(model, search, chain_order, cache):
    """Anchored conditional boundary paths; retain all scored intermediate variants.

    Partition, designation, and fitted state are separate identities. Exact-zero
    weights are repaired by actual adjacent coarsening, never by a weight floor.
    CandidateStore remains the bounded SQLite evidence owner.
    """
    n, records, replicate = len(model), search["candidates"], search["replicate"]
    scope = f"anchored-refine:{replicate}"
    chain_digest = hashlib.sha256(np.asarray(chain_order, dtype="<i8").tobytes())
    merge_attempts = set()

    def cuts_of(result):
        return tuple(int(i) for i in np.flatnonzero(np.diff(result["labels"][chain_order]))+1)

    def fingerprint(cuts):
        digest = chain_digest.copy()
        digest.update(np.asarray(cuts, dtype="<i8").tobytes())
        return digest.hexdigest()

    def consider(result, parent, kind, diagnostics=None, existing=None):
        budget = int(parent["requested_k"])
        cuts = cuts_of(result)
        validate_chain_labels(result["labels"][chain_order], n, budget)
        clonal = result["clonal_cluster"]
        if not np.any(result["labels"] == clonal) or result["centers"][clonal] != model.purity:
            raise ValueError("Scalar candidate lost its occupied exact clonal center")
        state_key = json.dumps([budget, json.loads(_parameters(result, chain_order))],
                               separators=(",", ":"))
        if not records.visited_add(scope, state_key):
            return
        if existing is None:
            record = _scored_record(result, chain_order, {**parent, "candidate_kind": kind}, replicate)
            record.update(candidate_id=records.next_id(replicate),
                proposal_partition_sha256=fingerprint(cuts), partition_sha256=fingerprint(cuts),
                chain_cuts=",".join(map(str, cuts)), num_input_blocks=len(cuts)+1,
                refinement_parent_partition_sha256=parent["partition_sha256"],
                refinement_parent_clonal_cluster=parent["clonal_cluster"])
            if diagnostics:
                record.update({"boundary_"+key: value for key, value in diagnostics.items()})
            records.append(record)
        if np.all(result["cluster_weights"] > 0):
            return
        ordered = result["labels"][chain_order]
        visits = ordered[np.r_[0, np.asarray(cuts, dtype=int)]]
        unsupported = np.flatnonzero(result["cluster_weights"][visits] == 0)
        removals = {edge for block in unsupported for edge in (block-1, block) if 0 <= edge < len(cuts)}
        for edge in sorted(removals):
            reduced = cuts[:edge]+cuts[edge+1:]
            if (budget, reduced) in merge_attempts:
                continue
            merge_attempts.add((budget, reduced))
            labels = np.empty(n, dtype=int)
            labels[chain_order] = np.searchsorted(reduced, np.arange(n), side="right")
            for repaired in refit_partition_variants(model, labels, cache, chain_order):
                consider(repaired, parent, "unsupported_component_coarsening")

    starts = []
    start_keys = set()
    def start(record):
        if record is None or record.get("status") != "scored":
            return
        result = _restore_result(record, chain_order)
        key = (record["requested_k"], record["partition_parameters"])
        consider(result, record, record["candidate_kind"], existing=record)
        if key not in start_keys:
            starts.append((record, result))
            start_keys.add(key)

    for budget, candidate in sorted(search["winners"].items()):
        records.update(replicate, candidate["candidate_id"], {"selected_for_k": False})
        start(records.get(replicate, candidate["candidate_id"]))
        start(records.best(replicate, budget, eligible=True))
    # All native anchored variants can seed conditional paths, not just the
    # mixture-score winner of their shared partition.
    native = list(records.iter_rows(replicate=replicate, candidate_kind="native", status="scored"))
    for record in native:
        start(record)
    for parent, fitted in starts:
        def refit_all(labels, initial_centers):
            variants = list(refit_partition_variants(model, labels, cache, chain_order,
                                                     initial_centers=initial_centers))
            for variant in variants:
                consider(variant, parent, "chain_boundary_polish")
            if not variants:
                raise RuntimeError("no_eligible_anchored_candidate")
            # Zero-weight variants stay as evidence/repair seeds, not path states.
            eligible = [v for v in variants if np.all(v["cluster_weights"] > 0)]
            if not eligible:
                raise RuntimeError("no_positive_weight_anchored_variant")
            if _selection_objective(model) == "legacy_mixture":
                return max(eligible, key=lambda v: (v["conditional_log_likelihood"], -v["clonal_cluster"]))
            return min(eligible, key=lambda v: (v["bic"], v["clonal_cluster"]))
        refinement = polish_chain_partition(model, fitted, chain_order, refit_all,
                                             column_evaluator=cache.log_kernel)
        consider(refinement["result"], parent, "chain_boundary_polish", refinement["diagnostics"])
        records.update(replicate, parent["candidate_id"],
                       {"polish_"+key: value for key, value in refinement["diagnostics"].items()})
    winners = {}
    for budget, parent in search["winners"].items():
        record = records.best(replicate, budget, eligible=True)
        if record is None:
            for result in refit_partition_variants(model, np.zeros(n, dtype=int), cache, chain_order):
                consider(result, parent, "supported_anchored_single_block_fallback")
            record = records.best(replicate, budget, eligible=True)
        if record is None:
            continue
        result = _restore_result(record, chain_order)
        records.update(replicate, record["candidate_id"],
                       {"selected_for_k": True, "selected_for_replicate_k": True})
        winners[budget] = {**record, "result": result, "cuts": cuts_of(result)}
    return {"winners": winners, "candidates": records, "original_bank_stats": search["stats"],
            "stats": {"unique_partitions": records.distinct_proposals(replicate),
                      "scored_budget_candidates": records.count(replicate=replicate, status="scored"),
                      "cached_refit_blocks": len(cache), "likelihood_cache_bytes": cache.column_bytes}}


def select_chain(model, chain_order, proposals, cluster_list, *, candidate_store, clonal_eligible=None,
                 memory=None, refit_store=None):
    """Select one full-data chain fit from in-memory native proposals.

    Proposals contain chain-ordered labels and unprefixed raw diagnostics.
    There are no file outputs, subsampling, independent replicates or progress
    callbacks. The supplied scratch store remains owned by the caller. Original
    replicate=1 evidence keys, routing, candidate order and scoring are retained.
    """
    chain_order = _integer_vector(chain_order, "Frozen chain order", len(model))
    if not np.array_equal(np.sort(chain_order), np.arange(len(model))):
        raise ValueError("Frozen chain order must be a permutation of original row indices")
    budgets = _integer_vector(cluster_list, "Cluster budgets")
    if (
        len(np.unique(budgets)) != len(budgets)
        or np.any(budgets < 1)
        or np.any(budgets > min(EXHAUSTIVE_MAX_CLUSTERS, len(model)))
    ):
        raise ValueError(f"Expected unique exhaustive K in 1..min({EXHAUSTIVE_MAX_CLUSTERS},N)")
    if candidate_store is None:
        raise ValueError("The caller must own an explicit candidate store")
    supplied = {}
    for proposal in proposals:
        requested_k = proposal["requested_k"]
        if isinstance(requested_k, bool) or int(requested_k) != requested_k:
            raise ValueError("Native proposal capacity must be an integer")
        requested_k = int(requested_k)
        if requested_k not in budgets or requested_k in supplied:
            raise ValueError("Duplicate or unrequested native proposal capacity")
        supplied[requested_k] = proposal
    if set(supplied) != set(budgets):
        raise ValueError("Every requested capacity needs one native proposal")
    rep = 1
    records = [
        {
            "requested_k": k,
            "replicate": rep,
            "status": "missing_partition",
            "num_mutations": len(model),
            "selected_for_k": False,
            "selected": False,
            "raw_status": np.nan,
        }
        for k in budgets
    ]
    by_attempt = {(int(row["requested_k"]), row["replicate"]): row for row in records}
    cache = _ChainRefitCache(model, chain_order, clonal_eligible=clonal_eligible,
                            memory=memory, refit_store=refit_store)
    search_records = candidate_store
    seeds, raw_diagnostics = [], {}
    for requested_k in budgets:
        proposal = supplied[int(requested_k)]
        ordered_labels = validate_chain_labels(proposal["labels"], len(model), requested_k)
        labels = np.empty(len(model), dtype=int)
        labels[chain_order] = ordered_labels
        seeds.append({"requested_k": int(requested_k), "replicate": rep, "labels": labels})
        raw_diagnostics[int(requested_k)] = {
            "raw_" + str(key): value for key, value in proposal["diagnostics"].items()
        }
    search = search_chain_coarsenings(
        model, seeds, chain_order, budgets, cache, store=search_records, replicate=rep
    )
    search = refine_chain_search(model, search, chain_order, cache)
    for candidate in search["winners"].values():
        search_records.update(rep, candidate["candidate_id"], {"selected_for_k": False})
    native_records = {}
    for row in search_records.iter_rows(replicate=rep, candidate_kind="native"):
        if not row.get("publication_eligible"):
            continue
        key = row["requested_k"]
        previous = native_records.get(key)
        if previous is None or (row["bic"], row["candidate_id"]) < (previous["bic"], previous["candidate_id"]):
            native_records[key] = row
    best_by_k = {}
    for requested_k, candidate in search["winners"].items():
        record = by_attempt[(requested_k, rep)]
        result = candidate["result"]
        record["center_refit"] = "conditional"
        record.update({key: result[key] for key in (*_SCORE_FIELDS, "num_parameters")})
        record.update(
            {
                key: candidate[key]
                for key in (
                    "candidate_kind",
                    "parent_requested_k",
                    "parent_replicate",
                    "parent_partition_sha256",
                    "proposal_partition_sha256",
                    "partition_sha256",
                    "candidate_search_version",
                    "active_mixture_components",
                    "candidate_id",
                    "publication_eligible",
                    "minimum_mixture_weight",
                    "distinct_centers",
                    "joint_mixture_center_mle",
                )
            }
        )
        record["native_bic"] = native_records.get(requested_k, {}).get("bic")
        record["native_num_clusters"] = native_records.get(requested_k, {}).get("num_clusters")
        record["candidate_count"] = search_records.count(replicate=rep, requested_k=requested_k)
        parent_raw = raw_diagnostics[candidate["parent_requested_k"]]
        record.update({"parent_" + key: value for key, value in parent_raw.items()})
        if candidate["candidate_kind"] == "native":
            record.update(parent_raw)
        record["status"] = "numerical_multimode_refit"
        best_by_k[requested_k] = (result["bic"], result["num_clusters"], rep, result, record)
    winners = []
    for requested_k, best in sorted(best_by_k.items()):
        best[4]["selected_for_k"] = True
        winners.append((best[0], best[1], requested_k, best[2], best[3], best[4]))
    if not winners:
        raise RuntimeError("no_eligible_anchored_candidate: no requested K produced a supported fit")
    winner = min(winners, key=lambda item: item[:4])
    winner[5]["selected"] = True
    for requested_k, best in best_by_k.items():
        row = best[4]
        search_records.update(
            row["replicate"],
            row["candidate_id"],
            {"selected_for_k": True, "selected": requested_k == winner[2]},
        )
    fields = (
        "requested_k",
        "replicate",
        "candidate_id",
        "candidate_kind",
        "parent_requested_k",
        "parent_replicate",
        "parent_partition_sha256",
        "proposal_partition_sha256",
        "partition_sha256",
        "num_clusters",
        "bic",
        "log_likelihood",
        "conditional_log_likelihood",
        "weight_optimality_gap",
        "weight_active_score_gap",
    )
    fits = []
    for record in records:
        if "candidate_id" not in record:
            continue
        candidate = search_records.get(record["replicate"], record["candidate_id"])
        fits.append(
            {
                **{
                    name: record[name].item() if isinstance(record[name], np.generic) else record[name]
                    for name in fields
                },
                "proposal_cuts": [int(cut) for cut in candidate["chain_cuts"].split(",") if cut],
                "partition_parameters": json.loads(candidate["partition_parameters"]),
            }
        )
    return {
        "selected_k": int(winner[2]),
        "fits": fits,
        "selection": pd.DataFrame(records),
        "candidates": search_records,
        "telemetry": cache.telemetry,
    }
