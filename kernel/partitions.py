"""Bounded decoded tree memberships, shared by every anchored variant.

Candidates retain a cache handle and cut tuple, never a decoded state. Eviction
therefore releases cache-owned arrays even while the complete candidate bank is
kept for final reconciliation. A zero budget preserves reconstruct-on-demand.
"""

from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .topology import partition_from_cuts
from .regional_support import RegionalSupport
from ..io.data import readonly_array


@dataclass(frozen=True, slots=True)
class PartitionState:
    labels: np.ndarray
    members: tuple
    counts: np.ndarray
    observed_counts: np.ndarray
    positive_counts: np.ndarray
    hard_zero_counts: np.ndarray
    forced_zero: np.ndarray
    support_conflict: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    clonal_eligible: np.ndarray
    support_identity: str

    @classmethod
    def from_labels(cls, labels, model):
        labels = readonly_array(labels, dtype=np.int64)
        q = int(labels.max()) + 1
        members = tuple(readonly_array(np.flatnonzero(labels == k)) for k in range(q))
        support = getattr(model, "regional_support", None)
        if support is None:
            support = RegionalSupport(model.observed, model.observed,
                                      np.zeros_like(model.observed), "likelihood")
        summary = support.summarize_partition(labels)
        return cls(labels, members, readonly_array([len(rows) for rows in members], dtype=np.int64),
                   summary.observed_counts, summary.positive_counts, summary.hard_zero_counts,
                   summary.forced_zero, summary.support_conflict,
                   readonly_array([model.lower[rows].max(axis=0) for rows in members]),
                   readonly_array([model.upper[rows].min(axis=0) for rows in members]),
                   readonly_array([model.clonal_eligible[rows].all() for rows in members]),
                   support.identity)

    @property
    def nbytes(self):
        return sum(value.nbytes for value in (
            self.labels, *self.members, self.counts, self.observed_counts,
            self.positive_counts, self.hard_zero_counts, self.forced_zero, self.support_conflict,
            self.lower, self.upper, self.clonal_eligible))


class TreePartitionCache:
    """One model/tree-local LRU; bounded payload plus conservative entry overhead."""

    def __init__(self, model, tree, budget_bytes=0):
        if (isinstance(budget_bytes, (bool, np.bool_))
                or not isinstance(budget_bytes, (int, np.integer)) or budget_bytes < 0):
            raise ValueError("Partition cache budget must be a nonnegative integer")
        self.tree = tree
        # Retain only immutable metadata needed for reconstruction, not the
        # RegionalModel's device tensors, likelihood caches or fitting state.
        self._metadata = SimpleNamespace(**{name: getattr(model, name) for name in (
            "observed", "lower", "upper", "clonal_eligible")})
        self._metadata.regional_support = getattr(model, "regional_support", None)
        self.budget_bytes = int(budget_bytes)
        self._entries = OrderedDict()
        self.retained_bytes = self.peak_bytes = 0
        self.hits = self.misses = self.decodes = self.evictions = self.bypasses = 0

    @staticmethod
    def entry_bytes(cuts, state):
        # Charge array headers/backing bytes, tuple/key integers, dataclass and
        # ordered-dict links as well as numeric buffers. This is conservative
        # accounting, not a measurement of process RSS or allocator high water.
        return state.nbytes + 1152 + 256 * (11 + len(state.members)) + 36 * len(cuts)

    def get(self, cuts):
        cuts = tuple(cuts)
        return self._get(cuts, lambda: partition_from_cuts(self.tree, cuts), len(cuts))

    def get_membership(self, cuts, groups):
        """Decode a general partition as connected pieces plus block identities."""
        cuts, groups = tuple(cuts), tuple(groups)
        return self._get(("general", cuts, groups),
                         lambda: np.asarray(groups)[partition_from_cuts(self.tree, cuts)],
                         len(cuts) + len(groups))

    def _get(self, key, decode, key_items):
        entry = self._entries.get(key)
        if entry is not None:
            self.hits += 1
            self._entries.move_to_end(key)
            return entry[0]
        self.misses += 1
        self.decodes += 1
        state = PartitionState.from_labels(decode(), self._metadata)
        size = self.entry_bytes(range(key_items), state)
        if size > self.budget_bytes:
            self.bypasses += 1
            return state
        while self.retained_bytes + size > self.budget_bytes:
            _, (_, removed) = self._entries.popitem(last=False)
            self.retained_bytes -= removed
            self.evictions += 1
        self._entries[key] = state, size
        self.retained_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.retained_bytes)
        return state

    def labels(self, cuts):
        if not self.budget_bytes:
            # Disabled caching keeps the old lightweight label-only access;
            # do not build all block metadata just to read a compact candidate.
            self.misses += 1
            self.decodes += 1
            self.bypasses += 1
            return readonly_array(partition_from_cuts(self.tree, cuts))
        return self.get(cuts).labels

    def diagnostics(self):
        return {"budget_bytes": self.budget_bytes, "retained_bytes": self.retained_bytes,
                "peak_bytes": self.peak_bytes, "entries": len(self._entries),
                "hits": self.hits, "misses": self.misses, "decodes": self.decodes,
                "evictions": self.evictions, "bypasses": self.bypasses,
                "accounting": "numeric_buffers_plus_conservative_entry_overhead"}
