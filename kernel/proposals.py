"""Stream the legacy adjacent-coarsening bank in its exact canonical order.

The generator keeps source parents and one proposal's routes in memory. It
does not retain a set of previously emitted partitions. Equal owner identities
are merged lazily, which also preserves ordering for duplicate source parents.
"""

from dataclasses import dataclass
import hashlib
import heapq
import itertools

import numpy as np

from . import MAX_CLUSTERS


@dataclass(frozen=True, slots=True)
class _Parent:
    requested_k: int
    replicate: int
    cuts: tuple
    mask: int
    digest: str

    @property
    def owner(self):
        return self.requested_k, self.replicate


def _integers(values, name, length=None):
    # Match selection._integer_vector, including its accepted input types.
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


def _route_rank(provenance):
    return (
        provenance["candidate_kind"] != "native",
        provenance["parent_requested_k"],
        provenance["parent_replicate"],
        provenance["candidate_kind"],
    )


def _resolve_routes(cuts, mask, parents, allowed):
    """Preserve source-scan insertion order, including replacements in place."""
    routes = {}
    target = len(cuts) + 1

    def add(parent, budget, kind):
        provenance = {
            "requested_k": budget,
            "parent_requested_k": parent.requested_k,
            "parent_replicate": parent.replicate,
            "candidate_kind": kind,
            "parent_partition_sha256": parent.digest,
        }
        if budget not in routes or _route_rank(provenance) < _route_rank(routes[budget]):
            routes[budget] = provenance

    for parent in parents:
        if mask & parent.mask != mask:
            continue
        same = cuts == parent.cuts
        if same and parent.requested_k in allowed:
            add(parent, parent.requested_k, "native")
        if target in allowed:
            add(parent, target, "capacity_reuse" if same else "adjacent_coarsening")
    return routes


def iter_chain_proposals(seeds, chain_order, budgets):
    """Yield ``(cuts, insertion_ordered_routes)`` exactly as the legacy bank.

    ``seeds`` have original-row ``labels``, ``requested_k``, and optional
    ``replicate`` (default 1). The frozen order must be a permutation of those
    rows. Native collapsed partitions remain routed to their requested K;
    coarsenings are routed only to requested K equal to their input block count.

    Ordering is (smallest resolved parent (K, replicate), cut count, cuts).
    No numerical fitting, pruning, or candidate limit is performed. Exhaustive
    enumeration is still exponential, but auxiliary memory depends on the
    parents and their boundary union rather than the emitted candidate count.
    """
    chain_order = _integers(chain_order, "Frozen chain order")
    n = len(chain_order)
    if not np.array_equal(np.sort(chain_order), np.arange(n)):
        raise ValueError("Frozen chain order must be a permutation of original row indices")
    budgets = _integers(budgets, "Cluster budgets")
    if (
        len(np.unique(budgets)) != len(budgets)
        or np.any(budgets < 1)
        or np.any(budgets > min(MAX_CLUSTERS, n))
    ):
        raise ValueError(f"Expected unique K in 1..min({MAX_CLUSTERS},N)")
    allowed = set(int(k) for k in budgets)
    chain_digest = hashlib.sha256(np.asarray(chain_order, dtype="<i8").tobytes())
    raw_parents = []
    for seed in seeds:
        source_k = int(seed["requested_k"])
        rep = int(seed.get("replicate", 1))
        if (
            source_k != seed["requested_k"]
            or not 1 <= source_k <= min(MAX_CLUSTERS, n)
            or rep != seed.get("replicate", 1)
            or rep < 1
        ):
            raise ValueError("Invalid native candidate K or replicate")
        labels = _integers(seed["labels"], "Native chain labels", n)[chain_order]
        starts = np.r_[True, labels[1:] != labels[:-1]]
        if len(np.unique(labels[starts])) != np.count_nonzero(starts):
            raise ValueError("Each cluster must be contiguous along the frozen chain")
        if np.count_nonzero(starts) > source_k:
            raise ValueError("Partition exceeds its requested K block budget")
        cuts = tuple(int(i) for i in np.flatnonzero(starts)[1:])
        digest = chain_digest.copy()
        digest.update(np.asarray(cuts, dtype="<i8").tobytes())
        raw_parents.append((source_k, rep, cuts, digest.hexdigest()))

    boundaries = sorted({cut for _, _, cuts, _ in raw_parents for cut in cuts})
    bits = {cut: 1 << index for index, cut in enumerate(boundaries)}
    parents = [
        _Parent(k, rep, cuts, sum(bits[cut] for cut in cuts), digest) for k, rep, cuts, digest in raw_parents
    ]
    groups = {}
    for parent in parents:
        groups.setdefault(parent.owner, []).append(parent)
    for owner, group in sorted(groups.items()):
        cardinalities = set()
        for parent in group:
            cardinalities.update(k - 1 for k in allowed if k <= len(parent.cuts) + 1)
            if parent.requested_k in allowed:
                cardinalities.add(len(parent.cuts))
        for size in sorted(cardinalities):
            streams = [
                itertools.combinations(parent.cuts, size)
                for parent in group
                if size <= len(parent.cuts)
                and (size + 1 in allowed or (size == len(parent.cuts) and parent.requested_k in allowed))
            ]
            previous = None
            for cuts in heapq.merge(*streams):
                if cuts == previous:
                    continue
                previous = cuts
                mask = sum(bits[cut] for cut in cuts)
                routes = _resolve_routes(cuts, mask, parents, allowed)
                resolved_owner = min(
                    (p["parent_requested_k"], p["parent_replicate"]) for p in routes.values()
                )
                if resolved_owner == owner:
                    yield cuts, routes
