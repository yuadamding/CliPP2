"""Immutable retained biological inputs and their CN exclusion provenance."""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from ..config import validate_max_major_cn


def readonly_array(value: object, *, dtype=None) -> np.ndarray:
    """Copy into an immutable bytes-backed array, not a reversible write flag."""
    array = np.asarray(value, dtype=dtype, order="C")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def restore_immutable_record(cls, inputs):
    """Re-run validation/freezing rather than restoring NumPy's writable state."""
    return cls(**inputs)


class ImmutableArrayRecord:
    """Copy/pickle immutable dataclasses through their authoritative constructors."""
    __slots__ = ()

    def __reduce__(self):
        return restore_immutable_record, (type(self), {
            item.name: getattr(self, item.name) for item in fields(self) if item.init
        })


@dataclass(frozen=True, slots=True)
class CNFilterRecord:
    mutation_id: str
    sample_id: str
    segment_id: str
    reason: str
    n_distinct_cn_states: int
    max_major_cn: int


@dataclass(frozen=True, slots=True)
class CNFilterReport:
    policy_id: str
    max_major_cn: int
    input_mutation_count: int
    retained_mutation_count: int
    excluded_mutation_ids: tuple[str, ...]
    records: tuple[CNFilterRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_major_cn", validate_max_major_cn(self.max_major_cn))


@dataclass(frozen=True)
class TumorData(ImmutableArrayRecord):
    """CN-compiled inputs, with no CN-population state in the optimizer.

    major_cn/minor_cn are per-allele maxima; mean_total_cn is the actual
    fraction-weighted total. Only clonal entries have a single CN pair.
    """

    tumor_id: str
    mutation_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    alt_counts: np.ndarray
    total_counts: np.ndarray
    purity: np.ndarray
    major_cn: np.ndarray
    minor_cn: np.ndarray
    normal_cn: np.ndarray
    scaling: np.ndarray
    phi_upper: np.ndarray
    phi_init: np.ndarray
    count_observed: np.ndarray | None = None
    cn_filter_report: CNFilterReport | None = None
    mean_total_cn: np.ndarray | None = None
    cn_state_count: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("mutation_ids", "region_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.mean_total_cn is None:
            object.__setattr__(self, "mean_total_cn", np.asarray(self.major_cn) + np.asarray(self.minor_cn))
        if self.cn_state_count is None:
            object.__setattr__(self, "cn_state_count", np.ones_like(self.major_cn, dtype=np.int64))
        for name in (
            "alt_counts", "total_counts", "purity", "major_cn", "minor_cn",
            "normal_cn", "scaling", "phi_upper", "phi_init", "count_observed",
            "mean_total_cn", "cn_state_count",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, readonly_array(value))

    @property
    def num_mutations(self) -> int:
        return int(self.alt_counts.shape[0])

    @property
    def num_regions(self) -> int:
        return int(self.alt_counts.shape[1])


__all__ = ["CNFilterRecord", "CNFilterReport", "TumorData"]
