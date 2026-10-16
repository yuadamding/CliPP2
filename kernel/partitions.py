"""Bounded decoded tree memberships, shared by every anchored variant.

Candidates retain a cache handle and cut tuple, never a decoded state. Eviction
therefore releases cache-owned arrays even while the complete candidate bank is
kept for final reconciliation. A zero budget preserves reconstruct-on-demand.
"""

from collections import OrderedDict
from dataclasses import dataclass, fields
from types import SimpleNamespace
import weakref

import numpy as np

from .topology import partition_from_cuts
from .regional_support import RegionalSupport
from ..io.data import ImmutableArrayRecord, readonly_array


class _WeakRecord(ImmutableArrayRecord):
    __slots__ = ("__weakref__",)


def _array_signature(array):
    # Bytes-backed buffers prevent value writes, not ndarray descriptor edits.
    return (id(array), array.shape, array.strides, array.dtype.str,
            array.__array_interface__["data"], id(array.base))


def _immutable_buffer(array):
    base = array
    while isinstance(base, np.ndarray):
        if base.flags.writeable:
            return False
        base = base.base
    return isinstance(base, bytes)


@dataclass(frozen=True, slots=True)
class PartitionState(_WeakRecord):
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

    def __post_init__(self):
        for item in fields(self):
            if item.name not in ("members", "support_identity"):
                object.__setattr__(self, item.name, readonly_array(getattr(self, item.name)))
        object.__setattr__(self, "members", tuple(readonly_array(rows) for rows in self.members))

    @classmethod
    def from_labels(cls, labels, model):
        labels = np.asarray(labels)
        if (labels.shape != (len(model.observed),) or labels.dtype.kind not in "iu"
                or not len(labels) or np.any(labels < 0) or labels.max() >= len(labels)
                or not np.array_equal(np.unique(labels),
                                      np.arange(int(labels.max())+1))):
            raise ValueError("Partition labels must be occupied consecutive integers")
        labels = np.asarray(labels, dtype=np.int64)
        q = int(labels.max()) + 1
        members = tuple(np.flatnonzero(labels == k) for k in range(q))
        support = getattr(model, "regional_support", None)
        if support is None:
            support = RegionalSupport(model.observed, model.observed,
                                      np.zeros_like(model.observed), "likelihood")
        summary = support.summarize_partition(labels)
        return cls(labels, members, np.array([len(rows) for rows in members], dtype=np.int64),
                   summary.observed_counts, summary.positive_counts, summary.hard_zero_counts,
                   summary.forced_zero, summary.support_conflict,
                   np.array([model.lower[rows].max(axis=0) for rows in members]),
                   np.array([model.upper[rows].min(axis=0) for rows in members]),
                   np.array([model.clonal_eligible[rows].all() for rows in members]),
                   support.identity)

    @property
    def nbytes(self):
        return sum(value.nbytes for value in self._arrays())

    def _arrays(self):
        return (
            self.labels, *self.members, self.counts, self.observed_counts,
            self.positive_counts, self.hard_zero_counts, self.forced_zero, self.support_conflict,
            self.lower, self.upper, self.clonal_eligible)


class _PartitionAuthority:
    """Issue exact-object capabilities without retaining evicted decoded states.

    Digests or dataclass fields supplied by a caller are not capabilities.
    Issuance also binds immutable metadata references, so replaced arrays and
    copied/reconstructed records return to the fully checked path.
    """

    def __init__(self, model):
        self._owner = weakref.ref(model)
        self._metadata = SimpleNamespace(**{name: getattr(model, name) for name in (
            "observed", "lower", "upper", "clonal_eligible", "regional_support")})
        self._support_masks = tuple(getattr(model.regional_support, name)
                                    for name in ("observed", "positive", "hard_zero"))
        self._metadata_signatures = tuple(_array_signature(value) for value in (
            self._metadata.observed, self._metadata.lower, self._metadata.upper,
            self._metadata.clonal_eligible, *self._support_masks))
        self._issued = {}

    def matches(self, model):
        arrays = (model.observed, model.lower, model.upper, model.clonal_eligible,
                  model.regional_support.observed, model.regional_support.positive,
                  model.regional_support.hard_zero)
        return (self._owner() is model
                and model.regional_support is self._metadata.regional_support
                and all(map(_immutable_buffer, arrays))
                and tuple(map(_array_signature, arrays)) == self._metadata_signatures)

    def construct(self, labels):
        state = PartitionState.from_labels(labels, self._metadata)
        key, authority = id(state), weakref.ref(self)

        def release(reference):
            owner = authority()
            if owner is not None:
                owner._issued.pop(key, None)

        self._issued[key] = (weakref.ref(state, release),
                            tuple(getattr(state, item.name) for item in fields(state)),
                            tuple(map(_array_signature, state._arrays())))
        return state

    def trusts(self, state, model):
        record = self._issued.get(id(state))
        return (type(state) is PartitionState and record is not None
                and record[0]() is state and self.matches(model)
                and state.support_identity == model.regional_support.identity
                and all(getattr(state, item.name) is value
                        for item, value in zip(fields(state), record[1]))
                and tuple(map(_array_signature, state._arrays())) == record[2])


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
        self._authority = getattr(model, "_partition_authority", None)
        if self._authority is not None and not self._authority.matches(model):
            # A new bank binds current model bounds, not a constructor-time
            # snapshot. Old issued records cannot acquire the fresh authority.
            self._authority = model._partition_authority = _PartitionAuthority(model)
        self.budget_bytes = int(budget_bytes)
        self._entries = OrderedDict()
        self.retained_bytes = self.peak_bytes = 0
        self.hits = self.misses = self.decodes = self.evictions = self.bypasses = 0

    @staticmethod
    def entry_bytes(cuts, state):
        # Charge array headers/backing bytes, tuple/key integers, dataclass and
        # ordered-dict links and weak issuance/array-descriptor metadata as well
        # as numeric buffers. This is conservative
        # accounting, not a measurement of process RSS or allocator high water.
        return state.nbytes + 2176 + 768 * (11 + len(state.members)) + 36 * len(cuts)

    def from_labels(self, labels):
        if self._authority is not None:
            return self._authority.construct(labels)
        return PartitionState.from_labels(labels, self._metadata)

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
        state = self.from_labels(decode())
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
