"""Refit distance-to-set chain partitions and select observed-likelihood BIC.

Multiplicity is marginalized over 1..major_cn. Each occupied partition block
has one fitted cellular prevalence (CP), with CCF = CP / purity. The scalar
refit searches a dense grid and every component's binomial mode, then refines
all bracketed maxima; this is a deterministic multimode numerical search, not
a certificate of global optimality. BIC marginalizes both unknown cluster and
multiplicity, fitting cluster weights at the candidate's fixed CP centers. It
counts q center parameters and q-1 independent weights for q occupied blocks.
Adjacent coarsenings and conditional boundary polishing improve the proposal
bank on the same frozen chain. Centers use conditional block refitting;
observed-mixture weights are fitted only for BIC. No fusion/allocation penalty enters BIC.
"""

from collections import OrderedDict
import hashlib
import json
import math
import sys
import time

import numpy as np
import pandas as pd

from .proposals import iter_chain_proposals
from . import MAX_CLUSTERS, SCORING_VERSION, CANDIDATE_SEARCH_VERSION
from .refinement import polish_chain_partition
from .refitting import refit_center
from .scoring import fit_cluster_weights


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
        },
        separators=(",", ":"),
        allow_nan=False,
    )


class _ChainRefitCache(OrderedDict):
    """One model/order's interval refits and a bounded LRU of likelihood columns."""

    def __init__(self, model, chain_order, max_column_bytes=32 * 1024 * 1024):
        super().__init__()
        self.model = model
        self.order = np.asarray(chain_order, dtype=np.int64).copy()
        self.ranks = np.empty(len(model), dtype=np.int64)
        self.ranks[self.order] = np.arange(len(model))
        self.columns = OrderedDict()
        self.column_bytes = 0
        self.max_column_bytes = max_column_bytes
        self.max_intervals = 16384
        self.weight_results = OrderedDict()
        self.weight_bytes = 0
        self.max_weight_bytes = 8 * 1024 * 1024
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
        )

    def get(self, key, default=None):
        value = super().get(key, default)
        if key in self:
            self.move_to_end(key)
            self.telemetry["interval_cache_hits"] += 1
        return value

    def __setitem__(self, key, value):
        if key not in self and len(self) >= self.max_intervals:
            self.popitem(last=False)
        super().__setitem__(key, value)

    def fit_weights(self, centers, counts):
        # Model/order and numerical implementation are fixed by this cache.
        # Ordered float64 centers AND original count initialization are required;
        # duplicate columns can otherwise change exact-zero support.
        key = (fit_cluster_weights, centers.dtype.str, centers.tobytes(), counts.dtype.str, counts.tobytes())
        cached = self.weight_results.pop(key, None)
        if cached is not None:
            self.weight_results[key] = cached
            self.telemetry["weight_cache_hits"] += 1
            value = cached[0]
            return {**value, "cluster_weights": value["cluster_weights"].copy()}
        kernel = self.log_kernel(centers)
        start = time.perf_counter()
        self.telemetry["weight_solver_calls"] += 1
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
        if size <= self.max_weight_bytes:
            while self.weight_bytes + size > self.max_weight_bytes:
                _, (_, removed_size) = self.weight_results.popitem(last=False)
                self.weight_bytes -= removed_size
            self.weight_results[key] = ({**value, "cluster_weights": value["cluster_weights"].copy()}, size)
            self.weight_bytes += size
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


def refit_partition(model, labels, cache=None, chain_order=None):
    """Refit blocks; merge exact adjacent equal centers on a supplied chain.

    Without an explicit chain permutation no adjacency is inferred from labels.
    Equal centers separated by another center always remain separate blocks.
    """
    labels = np.atleast_1d(np.asarray(labels))
    if labels.shape != (len(model),) or not np.all(np.isfinite(labels)) or np.any(labels != np.rint(labels)):
        raise ValueError("Partition labels must be one integer per mutation")
    if isinstance(cache, _ChainRefitCache) and (
        cache.model is not model or chain_order is None or not np.array_equal(cache.order, chain_order)
    ):
        raise ValueError("Chain refit cache belongs to a different model or order")
    clusters, inverse = np.unique(labels, return_inverse=True)
    centers, log_likelihoods = [], []
    for index in range(len(clusters)):
        rows = np.flatnonzero(inverse == index)
        key = cache.block_key(rows) if isinstance(cache, _ChainRefitCache) else rows.tobytes()
        fitted = cache.get(key) if cache is not None else None
        if fitted is None:
            start = time.perf_counter()
            fitted = refit_center(model.subset(rows))
            if isinstance(cache, _ChainRefitCache):
                cache.telemetry["interval_refits"] += 1
                cache.telemetry["conditional_refit_seconds"] += time.perf_counter() - start
            if cache is not None:
                cache[key] = fitted
        centers.append(fitted[0])
        log_likelihoods.append(fitted[1])
    centers = np.asarray(centers)
    if chain_order is not None:
        chain_order = _integer_vector(chain_order, "Frozen chain order", len(model))
        if not np.array_equal(np.sort(chain_order), np.arange(len(model))):
            raise ValueError("Frozen chain order must be a permutation of original row indices")
        ordered = validate_chain_labels(inverse[chain_order], len(model), len(centers))
        ordered_cp = centers[ordered]
        starts = np.r_[True, ordered_cp[1:] != ordered_cp[:-1]]
        inverse[chain_order] = np.cumsum(starts) - 1
        centers = ordered_cp[starts]
    order = np.argsort(-centers, kind="stable")
    relabel = np.empty(len(order), dtype=int)
    relabel[order] = np.arange(len(order))
    labels = relabel[inverse]
    centers = centers[order]
    conditional_log_likelihood = math.fsum(log_likelihoods)
    k = len(centers)
    counts = np.bincount(labels, minlength=k)
    mixture = (
        cache.fit_weights(centers, counts)
        if isinstance(cache, _ChainRefitCache)
        else fit_cluster_weights(model, centers, counts)
    )
    parameters = 2 * k - 1
    bic = -2 * mixture["log_likelihood"] + parameters * math.log(len(model))
    return {
        "labels": labels,
        "centers": centers,
        **mixture,
        "conditional_log_likelihood": conditional_log_likelihood,
        "num_clusters": k,
        "num_parameters": parameters,
        "num_mutations": len(model),
        "bic": bic,
        "bic_definition": SCORING_VERSION,
    }


def _integer_vector(values, name, length=None):
    values = np.atleast_1d(np.asarray(values, dtype=float))
    if (
        values.ndim != 1
        or not values.size
        or (length is not None and values.size != length)
        or np.any(~np.isfinite(values))
        or np.any(values != np.rint(values))
    ):
        raise ValueError("%s must be a nonempty integer vector of the expected length" % name)
    return values.astype(np.int64)


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
    """Compare native proposals and all their adjacent-block coarsenings.

    K remains a requested hyperparameter. A derived q-block partition is routed
    only to requested K=q, not used to introduce unrequested lower-K searches.
    Native proposals may themselves collapse below their capacity. No new cuts,
    ordering, likelihood, penalty, or unconstrained label reassignment is used.
    This searches a finite proposal bank, not all feasible chain partitions.
    """
    n = len(model)
    chain_order = _integer_vector(chain_order, "Frozen chain order", n)
    if not np.array_equal(np.sort(chain_order), np.arange(n)):
        raise ValueError("Frozen chain order must be a permutation of original row indices")
    budgets = _integer_vector(budgets, "Cluster budgets")
    if (
        len(np.unique(budgets)) != len(budgets)
        or np.any(budgets < 1)
        or np.any(budgets > min(MAX_CLUSTERS, n))
    ):
        raise ValueError(f"Expected unique K in 1..min({MAX_CLUSTERS},N)")
    cache = _ChainRefitCache(model, chain_order) if cache is None else cache
    chain_digest = hashlib.sha256(np.asarray(chain_order, dtype="<i8").tobytes())

    def fingerprint(cuts):
        digest = chain_digest.copy()
        digest.update(np.asarray(cuts, dtype="<i8").tobytes())
        return digest.hexdigest()

    records = store
    winners, winner_keys = {}, {}
    unique_partitions = 0

    def append(record):
        record["replicate"] = replicate
        record["publication_eligible"] = (
            record["status"] == "scored" and record["active_mixture_components"] == record["num_clusters"]
        )
        return records.append(record)

    for cuts, by_budget in iter_chain_proposals(seeds, chain_order, budgets):
        unique_partitions += 1
        labels = np.empty(n, dtype=int)
        labels[chain_order] = np.searchsorted(cuts, np.arange(n), side="right")
        proposal_digest = fingerprint(cuts)
        try:
            result = refit_partition(model, labels, cache, chain_order=chain_order)
        except RuntimeError as error:
            if any(p["candidate_kind"] == "native" for p in by_budget.values()):
                raise
            for provenance in by_budget.values():
                append(
                    {
                        **provenance,
                        "status": "refit_failed",
                        "error": str(error),
                        "proposal_partition_sha256": proposal_digest,
                        "chain_cuts": ",".join(map(str, cuts)),
                        "candidate_search_version": CANDIDATE_SEARCH_VERSION,
                        "selected_for_k": False,
                    }
                )
            continue
        ordered_result = result["labels"][chain_order]
        final_cuts = tuple(int(i) for i in np.flatnonzero(ordered_result[1:] != ordered_result[:-1]) + 1)
        digest = proposal_digest if final_cuts == cuts else fingerprint(final_cuts)
        parameters = _parameters(result, chain_order)
        for budget, provenance in sorted(by_budget.items()):
            validate_chain_labels(result["labels"][chain_order], n, budget)
            record = {
                **provenance,
                "status": "scored",
                "partition_sha256": digest,
                "proposal_partition_sha256": proposal_digest,
                "chain_cuts": ",".join(map(str, cuts)),
                "num_input_blocks": len(cuts) + 1,
                "num_clusters": result["num_clusters"],
                "partition_parameters": parameters,
                "bic": result["bic"],
                "log_likelihood": result["log_likelihood"],
                "conditional_log_likelihood": result["conditional_log_likelihood"],
                "weight_optimality_gap": result["weight_optimality_gap"],
                "weight_active_score_gap": result["weight_active_score_gap"],
                "active_mixture_components": int(np.count_nonzero(result["cluster_weights"] > 0)),
                "candidate_search_version": CANDIDATE_SEARCH_VERSION,
                "selected_for_k": False,
            }
            record = append(record)
            rank = (
                result["bic"],
                result["num_clusters"],
                provenance["candidate_kind"] != "native",
                provenance["parent_requested_k"],
                provenance["parent_replicate"],
                cuts,
            )
            if budget not in winners or rank < winner_keys[budget]:
                winners[budget] = {**record, "cuts": cuts, "result": result}
                winner_keys[budget] = rank
    for winner in winners.values():
        records.update(replicate, winner["candidate_id"], {"selected_for_k": True})
    return {
        "replicate": replicate,
        "winners": winners,
        "candidates": records,
        "stats": {
            "unique_partitions": unique_partitions,
            "scored_budget_candidates": records.count(replicate=replicate),
            "cached_refit_blocks": len(cache),
            "likelihood_cache_bytes": getattr(cache, "column_bytes", 0),
        },
    }


def refine_chain_search(model, search, seeds, chain_order, cache):
    """Improve the same constrained objective and reconcile boundary weights.

    The native and coarsening bank remains evidence, including degenerate
    scores. Each budget's bank winner and native partition seed an alternating
    exact-boundary/conditional-center polish. This never optimizes unrestricted
    mixture memberships or changes the frozen order. An exactly zero mixture
    weight cannot describe a published occupied component: repair it by actual
    adjacent merges and refits, not by relabeling or a positive weight floor.
    """
    n = len(model)
    chain_digest = hashlib.sha256(np.asarray(chain_order, dtype="<i8").tobytes())
    records = search["candidates"]
    winners = {}
    replicate = search["replicate"]
    visited_scope = f"refine:{replicate}"

    def cuts_of(result):
        ordered = result["labels"][chain_order]
        return tuple(int(i) for i in np.flatnonzero(np.diff(ordered)) + 1)

    def fingerprint(cuts):
        digest = chain_digest.copy()
        digest.update(np.asarray(cuts, dtype="<i8").tobytes())
        return digest.hexdigest()

    def admissible(result):
        return bool(np.all(result["cluster_weights"] > 0))

    for candidate in search["winners"].values():
        records.update(replicate, candidate["candidate_id"], {"selected_for_k": False})

    def visit_key(budget, cuts):
        return json.dumps([int(budget), list(cuts)], separators=(",", ":"))

    def consider(result, parent, kind, diagnostics=None, existing=None):
        budget = int(parent["requested_k"])
        cuts = cuts_of(result)
        validate_chain_labels(result["labels"][chain_order], n, budget)
        if not records.visited_add(visited_scope, visit_key(budget, cuts)):
            return
        if existing is not None:
            record = existing
        else:
            record = {
                key: parent[key]
                for key in (
                    "requested_k",
                    "parent_requested_k",
                    "parent_replicate",
                    "parent_partition_sha256",
                )
            }
            record.update(
                {
                    "candidate_id": records.next_id(replicate),
                    "candidate_kind": kind,
                    "status": "scored",
                    "proposal_partition_sha256": fingerprint(cuts),
                    "partition_sha256": fingerprint(cuts),
                    "chain_cuts": ",".join(map(str, cuts)),
                    "num_input_blocks": len(cuts) + 1,
                    "selected_for_k": False,
                    "candidate_search_version": CANDIDATE_SEARCH_VERSION,
                    "refinement_parent_partition_sha256": parent["partition_sha256"],
                }
            )
            record.update(
                {
                    key: result[key]
                    for key in (
                        "num_clusters",
                        "bic",
                        "log_likelihood",
                        "conditional_log_likelihood",
                        "weight_optimality_gap",
                        "weight_active_score_gap",
                    )
                }
            )
        record.update(
            {
                "partition_parameters": _parameters(result, chain_order),
                "publication_eligible": admissible(result),
                "active_mixture_components": int(np.count_nonzero(result["cluster_weights"] > 0)),
                "minimum_mixture_weight": float(np.min(result["cluster_weights"])),
                "distinct_centers": int(len(np.unique(result["centers"]))),
                "joint_mixture_center_mle": False,
            }
        )
        if diagnostics:
            record.update({"boundary_" + key: value for key, value in diagnostics.items()})
        record["replicate"] = replicate
        if existing is None:
            record = records.append(record)
        else:
            record = records.update(
                replicate,
                record["candidate_id"],
                {k: v for k, v in record.items() if k not in {"replicate", "candidate_id"}},
            )
        candidate = {**record, "result": result, "cuts": cuts}
        if admissible(result):
            rank = (result["bic"], result["num_clusters"], record["candidate_id"])
            old = winners.get(budget)
            if old is None or rank < (old["bic"], old["num_clusters"], old["candidate_id"]):
                winners[budget] = candidate
            return
        # Explore only merges touching unsupported blocks. Every accepted
        # reduction is a newly fitted chain partition under the same K budget.
        ordered = result["labels"][chain_order]
        starts = np.r_[0, np.asarray(cuts)]
        unsupported = np.flatnonzero(result["cluster_weights"][ordered[starts]] == 0)
        remove = set()
        for block in unsupported:
            if block > 0:
                remove.add(block - 1)
            if block < len(cuts):
                remove.add(block)
        for edge in sorted(remove):
            reduced_cuts = cuts[:edge] + cuts[edge + 1 :]
            if records.visited_contains(visited_scope, visit_key(budget, reduced_cuts)):
                continue
            labels = np.empty(n, dtype=int)
            labels[chain_order] = np.searchsorted(reduced_cuts, np.arange(n), side="right")
            try:
                repaired = refit_partition(model, labels, cache, chain_order)
            except RuntimeError as error:
                records.visited_add(visited_scope, visit_key(budget, reduced_cuts))
                records.append(
                    {
                        "replicate": replicate,
                        "candidate_id": records.next_id(replicate),
                        "requested_k": budget,
                        "candidate_kind": "unsupported_component_coarsening",
                        "status": "refit_failed",
                        "publication_eligible": False,
                        "selected_for_k": False,
                        "error": str(error),
                        "parent_requested_k": parent["parent_requested_k"],
                        "parent_replicate": parent["parent_replicate"],
                        "parent_partition_sha256": parent["parent_partition_sha256"],
                        "proposal_partition_sha256": fingerprint(reduced_cuts),
                        "chain_cuts": ",".join(map(str, reduced_cuts)),
                        "candidate_search_version": CANDIDATE_SEARCH_VERSION,
                    }
                )
                continue
            consider(repaired, candidate, "unsupported_component_coarsening")

    # Preserve every admissible old winner before adding improvements, so a
    # conditional-likelihood improvement cannot silently worsen the BIC winner.
    # Duplicate starts were already ignored by polishing; avoid refitting them.
    starts, start_partitions = [], set()
    for budget, candidate in sorted(search["winners"].items()):
        original = records.find_first(
            replicate=replicate,
            requested_k=budget,
            proposal_partition_sha256=candidate["proposal_partition_sha256"],
        )
        consider(candidate["result"], candidate, candidate["candidate_kind"], existing=original)
        starts.append(candidate)
        start_partitions.add((budget, candidate["partition_sha256"]))
        baseline = records.best(replicate, budget, eligible=True)
        if baseline is not None:
            if (budget, baseline["partition_sha256"]) in start_partitions:
                continue
            cuts = tuple(int(x) for x in baseline["chain_cuts"].split(",") if x)
            labels = np.empty(n, dtype=int)
            labels[chain_order] = np.searchsorted(cuts, np.arange(n), side="right")
            fitted = refit_partition(model, labels, cache, chain_order)
            consider(fitted, baseline, baseline["candidate_kind"], existing=baseline)
            starts.append({**baseline, "result": fitted})
            start_partitions.add((budget, baseline["partition_sha256"]))
    for seed in seeds:
        budget = seed["requested_k"]
        record = records.find_first(replicate=replicate, requested_k=budget, candidate_kind="native")
        if (budget, record["partition_sha256"]) in start_partitions:
            continue
        fitted = refit_partition(model, seed["labels"], cache, chain_order)
        consider(fitted, record, "native", existing=record)
        starts.append({**record, "result": fitted})
        start_partitions.add((budget, record["partition_sha256"]))
    polished = set()
    for parent in starts:
        key = (parent["requested_k"], cuts_of(parent["result"]))
        if key in polished:
            continue
        polished.add(key)
        refinement = polish_chain_partition(
            model,
            parent["result"],
            chain_order,
            lambda labels: refit_partition(model, labels, cache, chain_order),
        )
        consider(refinement["result"], parent, "chain_boundary_polish", refinement["diagnostics"])
        # Preserve convergence evidence even when polish leaves the same cuts.
        original = records.find_first(
            replicate=replicate,
            requested_k=parent["requested_k"],
            proposal_partition_sha256=parent["proposal_partition_sha256"],
        )
        records.update(
            replicate,
            original["candidate_id"],
            {"polish_" + key: value for key, value in refinement["diagnostics"].items()},
        )
    for budget, parent in search["winners"].items():
        if budget not in winners:
            fallback = refit_partition(model, np.zeros(n, dtype=int), cache, chain_order)
            # A failed optional route must not hide a finite single-block
            # candidate. This is an explicit refit within the at-most-K set.
            records.visited_discard(visited_scope, visit_key(budget, ()))
            consider(fallback, parent, "supported_single_block_fallback")
    # Visited final partitions suppress repeated exploration, not candidate
    # ranking. A repair can visit an older eligible record's partition first,
    # leaving its smaller ID unconsidered. Reconcile only after the search so
    # its starts, refits and proposal order stay unchanged. Restore the recorded
    # fit exactly; refitting here could change a tied score or zero-weight support.
    for budget, winner in winners.items():
        record = records.best(replicate, budget, eligible=True)
        if record["candidate_id"] == winner["candidate_id"]:
            continue
        parameters = json.loads(record["partition_parameters"])
        labels = np.empty(n, dtype=int)
        labels[chain_order] = np.asarray(parameters["block_labels"], dtype=int)[
            np.searchsorted(parameters["cuts"], np.arange(n), side="right")
        ]
        centers = np.asarray(parameters["centers"], dtype=float)
        weights = np.asarray(parameters["weights"], dtype=float)
        result = {
            key: record[key]
            for key in (
                "num_clusters",
                "bic",
                "log_likelihood",
                "conditional_log_likelihood",
                "weight_optimality_gap",
                "weight_active_score_gap",
            )
        }
        result.update(
            labels=labels,
            centers=centers,
            cluster_weights=weights,
            num_parameters=2 * record["num_clusters"] - 1,
            num_mutations=n,
            bic_definition=SCORING_VERSION,
        )
        record = records.update(
            replicate,
            record["candidate_id"],
            {
                "minimum_mixture_weight": float(weights.min()),
                "distinct_centers": int(len(np.unique(centers))),
                "joint_mixture_center_mle": False,
            },
        )
        winners[budget] = {**record, "result": result, "cuts": tuple(parameters["cuts"])}
    for winner in winners.values():
        records.update(
            replicate, winner["candidate_id"], {"selected_for_k": True, "selected_for_replicate_k": True}
        )
    return {
        "winners": winners,
        "candidates": records,
        "original_bank_stats": search["stats"],
        "stats": {
            "unique_partitions": records.distinct_proposals(replicate),
            "scored_budget_candidates": records.count(replicate=replicate, status="scored"),
            "cached_refit_blocks": len(cache),
            "likelihood_cache_bytes": getattr(cache, "column_bytes", 0),
        },
    }


def select_chain(model, chain_order, proposals, cluster_list, *, candidate_store):
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
        or np.any(budgets > min(MAX_CLUSTERS, len(model)))
    ):
        raise ValueError(f"Expected unique K in 1..min({MAX_CLUSTERS},N)")
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
    cache = _ChainRefitCache(model, chain_order)
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
    search = refine_chain_search(model, search, seeds, chain_order, cache)
    for candidate in search["winners"].values():
        search_records.update(rep, candidate["candidate_id"], {"selected_for_k": False})
    native_records = {
        row["requested_k"]: row
        for row in search_records.iter_rows(replicate=rep, candidate_kind="native")
    }
    best_by_k = {}
    for requested_k, candidate in search["winners"].items():
        record = by_attempt[(requested_k, rep)]
        result = candidate["result"]
        record["center_refit"] = "conditional"
        record.update(
            {
                key: result[key]
                for key in (
                    "num_clusters",
                    "num_parameters",
                    "log_likelihood",
                    "conditional_log_likelihood",
                    "weight_optimality_gap",
                    "weight_active_score_gap",
                    "bic",
                    "bic_definition",
                )
            }
        )
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
        record["native_bic"] = native_records[requested_k]["bic"]
        record["native_num_clusters"] = native_records[requested_k]["num_clusters"]
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
        raise RuntimeError("No requested K produced a partition for BIC selection")
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
