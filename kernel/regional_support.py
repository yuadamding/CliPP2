"""Immutable read-support constraints, separate from the emission domain."""

from dataclasses import dataclass, field
import hashlib

import numpy as np

from ..io.data import ImmutableArrayRecord, readonly_array


@dataclass(frozen=True, slots=True)
class SupportSummary(ImmutableArrayRecord):
    observed_counts: np.ndarray
    positive_counts: np.ndarray
    hard_zero_counts: np.ndarray
    forced_zero: np.ndarray
    support_conflict: np.ndarray

    def __post_init__(self):
        for name in ("observed_counts", "positive_counts", "hard_zero_counts",
                     "forced_zero", "support_conflict"):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class RegionalSupport(ImmutableArrayRecord):
    observed: np.ndarray
    positive: np.ndarray
    hard_zero: np.ndarray
    policy: str = "strict"
    identity: str = field(init=False)

    def __post_init__(self):
        if self.policy not in ("strict", "likelihood"):
            raise ValueError("regional_zero_policy must be strict or likelihood")
        for name in ("observed", "positive", "hard_zero"):
            object.__setattr__(self, name, readonly_array(getattr(self, name), dtype=bool))
        if (self.observed.ndim != 2 or self.positive.shape != self.observed.shape
                or self.hard_zero.shape != self.observed.shape
                or np.any((self.positive | self.hard_zero) & ~self.observed)
                or np.any(self.positive & self.hard_zero)
                or (self.policy == "likelihood" and self.hard_zero.any())
                or (self.policy == "strict" and not np.array_equal(
                    self.positive | self.hard_zero, self.observed))):
            raise ValueError("Invalid regional observation support masks")
        digest = hashlib.sha256(("regional_support_v1:"+self.policy).encode())
        digest.update(repr(self.observed.shape).encode())
        for values in (self.observed, self.positive, self.hard_zero):
            digest.update(values.tobytes())
        object.__setattr__(self, "identity", digest.hexdigest())

    @classmethod
    def from_counts(cls, alt, depth, count_observed=None, *, policy="strict"):
        alt, depth = np.asarray(alt), np.asarray(depth)
        if alt.shape != depth.shape or alt.ndim != 2:
            raise ValueError("Read counts must have matching N by R shapes")
        observed = (np.ones_like(depth, dtype=bool) if count_observed is None
                    else np.asarray(count_observed, dtype=bool))
        if observed.shape != depth.shape:
            raise ValueError("Observation mask must match read counts")
        observed = observed & np.isfinite(depth) & (depth > 0)
        if np.any(~np.isfinite(alt[observed])) or np.any(alt[observed] < 0):
            raise ValueError("Observed alternate counts must be finite and nonnegative")
        return cls(observed, observed & (alt > 0),
                   observed & (alt == 0) if policy == "strict" else np.zeros_like(observed),
                   policy)

    def summarize_partition(self, labels):
        labels = np.asarray(labels)
        if (labels.shape != (len(self.observed),) or labels.dtype.kind not in "iu"
                or not len(labels) or not np.array_equal(np.unique(labels),
                                                        np.arange(int(labels.max())+1))):
            raise ValueError("Support labels must be occupied consecutive integers")
        shape = (int(labels.max())+1, self.observed.shape[1])
        counts = []
        for mask in (self.observed, self.positive, self.hard_zero):
            count = np.zeros(shape, dtype=np.int64)
            np.add.at(count, labels, mask)
            counts.append(count)
        return SupportSummary(*counts, counts[2] > 0, (counts[1] > 0) & (counts[2] > 0))

    @staticmethod
    def check_partition(summary):
        if np.any(summary.observed_counts == 0):
            return "unsupported_cluster_region"
        if np.any(summary.support_conflict):
            return "regional_zero_positive_conflict"
        return None

    def assignment_mask(self, centers):
        centers = np.asarray(centers, dtype=float)
        if (centers.ndim != 2 or not len(centers)
                or centers.shape[1] != self.observed.shape[1]
                or not np.isfinite(centers).all()):
            raise ValueError("Centers must be finite nonempty q by R CCFs")
        if self.policy == "likelihood":
            return np.ones((len(self.observed), len(centers)), dtype=bool)
        return ((~self.hard_zero[:, None, :] | (centers[None, :, :] == 0.))
                & (~self.positive[:, None, :] | (centers[None, :, :] > 0.))).all(axis=2)

    def assigned_mask(self, labels, centers):
        """Check only chosen memberships in O(N R), without an N-by-q table."""
        labels, centers = np.asarray(labels), np.asarray(centers, dtype=float)
        if (labels.shape != (len(self.observed),) or labels.dtype.kind not in "iu"
                or centers.ndim != 2 or not len(centers)
                or centers.shape[1] != self.observed.shape[1]
                or not np.isfinite(centers).all()
                or np.any(labels < 0) or np.any(labels >= len(centers))):
            raise ValueError("Assigned centers must match regional support")
        if self.policy == "likelihood":
            return np.ones(len(labels), dtype=bool)
        assigned = centers[labels]
        return ((~self.hard_zero | (assigned == 0.))
                & (~self.positive | (assigned > 0.))).all(axis=1)

    def continuation_bounds(self, physical_lower, physical_upper):
        lower, upper = np.array(physical_lower, dtype=float), np.array(physical_upper, dtype=float)
        if (lower.shape != self.observed.shape or upper.shape != lower.shape
                or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or np.any(lower > upper)):
            raise ValueError("Invalid physical continuation bounds")
        interior = self.positive if self.policy == "strict" else np.ones_like(self.observed)
        lower[interior] = np.maximum(lower[interior], 1e-8)
        upper[interior] = np.minimum(upper[interior], 1.-1e-8)
        if np.any(self.hard_zero & ((lower > 0.) | (upper < 0.))):
            raise ValueError("Hard zero lies outside the physical domain")
        lower[self.hard_zero] = upper[self.hard_zero] = 0.
        if np.any(lower > upper):
            raise ValueError("Empty continuation bounds")
        return lower, upper
