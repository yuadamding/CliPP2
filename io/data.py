"""Immutable retained biological inputs and their CN exclusion provenance."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields

import numpy as np

from ..config import (
    CLONAL_INTEGER_GENERATOR_VERSION, CLONAL_INTEGER_MODEL_ID,
    CLONAL_INTEGER_PRIOR_MODE,
    validate_max_major_cn,
)


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
    _compiled_models: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("mutation_ids", "region_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in (
            "alt_counts", "total_counts", "purity", "major_cn", "minor_cn",
            "normal_cn", "scaling", "phi_upper", "phi_init", "count_observed",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, readonly_array(value))
        object.__setattr__(self, "_fingerprint", _retained_data_fingerprint(self))

    @property
    def num_mutations(self) -> int:
        return int(self.alt_counts.shape[0])

    @property
    def num_regions(self) -> int:
        return int(self.alt_counts.shape[1])


def _hash_text(digest, value: str) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _hash_array(digest, name: str, values: np.ndarray) -> None:
    _hash_text(digest, name)
    array = np.ascontiguousarray(values)
    _hash_text(digest, str(array.dtype))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _retained_data_fingerprint(data: TumorData) -> str:
    """Identify the retained numerical source, excluding eligibility-only audit."""
    digest = hashlib.sha256(b"clipp2.retained-integer-input.v3")
    _hash_text(digest, data.tumor_id)
    for ids in (data.mutation_ids, data.region_ids):
        digest.update(len(ids).to_bytes(8, "little"))
        for value in ids:
            _hash_text(digest, value)
    for name in (
        "alt_counts", "total_counts", "purity", "major_cn", "minor_cn",
        "normal_cn", "scaling", "phi_upper", "phi_init",
    ):
        _hash_array(digest, name, getattr(data, name))
    _hash_array(digest, "count_observed", np.ones_like(data.alt_counts, dtype=bool)
                if data.count_observed is None else data.count_observed)
    for value in (CLONAL_INTEGER_MODEL_ID, "1", CLONAL_INTEGER_GENERATOR_VERSION,
                  CLONAL_INTEGER_PRIOR_MODE):
        _hash_text(digest, value)
    return digest.hexdigest()


def tumor_data_fingerprint(data: TumorData) -> str:
    """Return the once-computed identity of this immutable retained input."""
    return data._fingerprint


__all__ = ["CNFilterRecord", "CNFilterReport", "TumorData", "tumor_data_fingerprint"]
