"""Validated mutation–region evidence and its distinct read masks."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

PURITY_MEAN_ATOL = 1e-10


def readonly_array(value, *, dtype=None):
    """Detach arrays into immutable bytes-backed storage."""
    array = np.asarray(value, dtype=dtype, order="C")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def freeze_record(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_record(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(map(freeze_record, value))
    return readonly_array(value) if isinstance(value, np.ndarray) else value


@dataclass(frozen=True, slots=True)
class TumorData:
    """Full input population; original CN and distinct measured/imputed masks."""

    tumor_id: str
    mutation_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    alt_counts: np.ndarray
    total_counts: np.ndarray
    purity: np.ndarray
    major_cn: np.ndarray
    minor_cn: np.ndarray
    normal_cn: np.ndarray | None = None
    count_observed: np.ndarray | None = None
    input_excluded: np.ndarray | None = None
    mapping_genuine: np.ndarray | None = None
    count_imputed: np.ndarray | None = None
    read_count_report: Mapping = field(default_factory=dict)
    fixed_multiplicity: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        from .tumor_txt import _identifier, _safe_tumor_id

        object.__setattr__(self, "tumor_id", _safe_tumor_id(self.tumor_id))
        for name in ("mutation_ids", "region_ids"):
            ids = tuple(getattr(self, name))
            if any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids):
                raise ValueError(f"{name} must contain unique nonempty strings")
            for value in ids:
                _identifier(value, name=name)
            object.__setattr__(self, name, ids)
        shape = len(self.mutation_ids), len(self.region_ids)
        for name in ("alt_counts", "total_counts", "purity", "major_cn", "minor_cn"):
            values = np.asarray(getattr(self, name))
            if values.shape != shape or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite aligned numerical arrays")
            values = readonly_array(values, dtype=float)
            if name != "purity" and (np.any(values < 0) or np.any(values != np.rint(values))):
                raise ValueError(f"{name} must contain nonnegative integers")
            object.__setattr__(self, name, values)
        if np.any(self.alt_counts > self.total_counts):
            raise ValueError("Alternate counts cannot exceed total counts")
        if np.any(self.major_cn < 1) or np.any(self.major_cn >= 2**63) or np.any(self.minor_cn > self.major_cn):
            raise ValueError("Single-state tumor CN requires integer 1 <= major and 0 <= minor <= major")
        if np.any(self.purity <= 0) or np.any(self.purity > 1) or not np.all(self.purity == self.purity[:1]):
            raise ValueError("Purity must be canonical within regions and lie in (0,1]")
        if self.normal_cn is not None:
            normal = np.asarray(self.normal_cn)
            if normal.shape not in ((), shape) or normal.dtype.kind not in "iuf" or np.any(normal != 2):
                raise ValueError("normal_cn is fixed at 2")
        object.__setattr__(self, "normal_cn", np.broadcast_to(readonly_array(2.), shape))
        for name, default in (("count_observed", True), ("count_imputed", False),
                              ("input_excluded", False), ("mapping_genuine", True)):
            value = getattr(self, name)
            value = np.full(shape, default) if value is None else np.asarray(value)
            if value.shape != shape or value.dtype.kind != "b":
                raise ValueError(f"{name} must contain aligned boolean flags")
            object.__setattr__(self, name, readonly_array(value))
        if np.any(self.count_imputed & (self.count_observed | self.input_excluded |
                                        (self.alt_counts != 0) | (self.total_counts == 0))):
            raise ValueError("Imputation requires an eligible missing observation with zero ALT and positive depth")
        if np.any(self.informative & ~self.mapping_genuine):
            raise ValueError("Scientific observations require genuine, unambiguous single-state CN mapping")
        object.__setattr__(self, "read_count_report", freeze_record(self.read_count_report))
        # Preprocessing availability is independent of eligibility and imputation.
        available = self.count_observed & (self.total_counts > 0)
        calls = np.full(shape, np.nan)
        calls[available] = 1.
        positive = available & (self.alt_counts > 0)
        rho = self.purity[positive]
        denominator = rho * self.total_cn[positive] + 2. * (1. - rho)
        fraction = self.alt_counts[positive] / self.total_counts[positive]
        with np.errstate(over="ignore"):
            copy_scale = denominator / rho
            saturated = np.isinf(copy_scale)
            copy_scale[saturated] = denominator[saturated]
            raw = fraction * copy_scale
            # At tiny valid purities, scale after multiplication to avoid premature overflow.
            raw[saturated] /= rho[saturated]
        calls[positive] = np.clip(np.rint(raw), 1., self.major_cn[positive])
        object.__setattr__(self, "fixed_multiplicity", readonly_array(calls))

    @property
    def total_cn(self):
        return self.major_cn + self.minor_cn

    @property
    def informative(self):
        return ~self.input_excluded & (self.count_observed | self.count_imputed) & (self.total_counts > 0)

    @property
    def measured(self):
        return ~self.input_excluded & self.count_observed & (self.total_counts > 0)
