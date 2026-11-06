"""Immutable prediction preparation sharing the fitting emission contract.

Prediction deliberately retains uninformative rows and regions. Native contexts
consume these fixed arrays without changing support in an existing context.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib

import numpy as np
from scipy.special import gammaln

from ..io.data import (ImmutableArrayRecord, OriginalCatalog, MultiplicityBounds, TumorData,
                       freeze_record, readonly_array)


def observed_mask(data):
    """The shared positive-depth observation mask; population rules stay outside."""
    if not isinstance(data, TumorData):
        raise TypeError("Numerical preparation requires validated TumorData")
    depth = np.asarray(data.total_counts)
    shape = len(data.mutation_ids), len(data.region_ids)
    observed = np.ones(shape, dtype=bool) if data.count_observed is None else np.asarray(data.count_observed, dtype=bool)
    if depth.shape != shape or observed.shape != shape:
        raise ValueError("Read counts and masks must align with mutation-region keys")
    excluded = getattr(data, "input_excluded", None)
    if excluded is not None:
        excluded = np.asarray(excluded, dtype=bool)
        if excluded.shape != shape:
            raise ValueError("Input exclusions must align with mutation-region keys")
        observed = observed & ~excluded
    return observed & np.isfinite(depth) & (depth > 0)


def prepare_observations(data, multiplicity_support=None, *, rows=None, regions=None):
    """One production preparation of biological arrays, domains and emissions.

    Callers choose the population: fitting excludes uninformative rows and requires
    regional coverage; prediction preserves requested rows, including empty inputs.
    """
    rows = np.arange(len(data.mutation_ids)) if rows is None else rows
    regions = np.arange(len(data.region_ids)) if regions is None else regions
    ix = np.ix_(rows, regions)
    observed = observed_mask(data)[ix]
    arrays = {name: np.asarray(source, dtype=float)[ix] for name, source in (
        ("alt", data.alt_counts), ("depth", data.total_counts), ("major", data.major_cn),
        ("normal", data.normal_cn), ("total", data.total_cn), ("purity", data.purity))}
    alt, depth, major, normal, total, purity = (arrays[name] for name in arrays)
    minor = np.asarray(data.minor_cn, dtype=float)[ix]
    if any(not np.isfinite(value).all() for value in (major, normal, total, purity)):
        raise ValueError("CN and purity must be finite in every mutation-region")
    if np.any(major < 1) or np.any(major != np.rint(major)):
        raise ValueError("Major CN must be a positive integer")
    if np.any(normal < 0) or np.any(normal != np.rint(normal)):
        raise ValueError("Normal CN must be a nonnegative integer")
    if (not np.isfinite(minor).all() or np.any(minor < 0) or np.any(minor > major)
            or np.any(minor != np.rint(minor)) or not np.array_equal(total, major + minor)):
        raise ValueError("Single-state CN needs integer major >= minor >= 0 and total = major + minor")
    if np.any(purity <= 0) or np.any(purity > 1) or not np.all(purity == purity[:1]):
        raise ValueError("Purity must be constant within each region and in (0,1]")
    if (not np.isfinite(alt[observed]).all() or np.any(alt[observed] < 0)
            or np.any(alt[observed] > depth[observed]) or np.any(alt[observed] != np.rint(alt[observed]))
            or np.any(depth[observed] != np.rint(depth[observed]))):
        raise ValueError("Informative read counts must be finite nonnegative integers")
    alt, depth = np.where(observed, alt, 0.), np.where(observed, depth, 0.)
    denominator = (1-purity)*normal + purity*total
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise ValueError("Normal-plus-tumor denominator must be finite and positive")
    default_maximum = major
    if multiplicity_support is None:
        minimum, maximum = np.ones_like(default_maximum), default_maximum
    else:
        if not isinstance(multiplicity_support, MultiplicityBounds):
            raise TypeError("multiplicity_support must be immutable MultiplicityBounds or None")
        minimum, maximum = (values[ix] for values in multiplicity_support.for_data(data))
    if (np.any(minimum < 1) or np.any(minimum > maximum) or np.any(maximum > default_maximum)
            or np.any(maximum >= 2**63)):
        raise ValueError("Multiplicity intervals must satisfy 1 <= lower <= upper <= original support")
    changed = (minimum != 1) | (maximum != default_maximum)
    minimum, maximum = minimum.astype(np.int64), maximum.astype(np.int64)
    support = maximum-minimum+1
    lower, upper = np.zeros_like(major), np.ones_like(major)
    arrays.update(alt=alt, depth=depth, observed=observed, multiplicity_min=minimum,
        multiplicity_max=maximum, support=support, multiplicity_restricted=bool(changed.any()),
        lower=lower, upper=upper, scaling=purity/denominator,
        constants=gammaln(depth+1)-gammaln(alt+1)-gammaln(depth-alt+1)-np.log(support))
    return freeze_record(arrays)


@dataclass(frozen=True, slots=True)
class PreparedObservations(ImmutableArrayRecord):
    data: TumorData
    mutation_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    device: str
    alt: np.ndarray
    depth: np.ndarray
    observed: np.ndarray
    eligible: np.ndarray
    major: np.ndarray
    normal: np.ndarray
    total: np.ndarray
    purity: np.ndarray
    scaling: np.ndarray
    constants: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    multiplicity_min: np.ndarray
    support: np.ndarray
    identity: str
    bounds: MultiplicityBounds

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, item.name, readonly_array(value))

    @property
    def n(self):
        return len(self.mutation_ids)

    @property
    def r(self):
        return len(self.region_ids)

    @property
    def multiplicity_max(self):
        return readonly_array(self.multiplicity_min + self.support - 1)


def prepare_prediction(catalog: OriginalCatalog, targets, bounds: MultiplicityBounds | None = None,
                       *, device="cpu", cpu_threads=None, normal_cn_policy=None) -> PreparedObservations:
    """Prepare every target in requested order, including excluded/missing rows.

    ``cpu_threads`` is accepted for a uniform caller interface; execution owners
    pass it to ``NativeBackend`` when constructing their separate context.
    """
    if not isinstance(catalog, OriginalCatalog):
        raise TypeError("Prediction requires an OriginalCatalog")
    if device not in ("cpu", "cuda"):
        raise ValueError("Prediction device must be cpu or cuda")
    if cpu_threads is not None and (type(cpu_threads) is not int or not 1 <= cpu_threads <= 2**31-1):
        raise ValueError("cpu_threads must be a positive int32 or None")
    data = catalog.prediction_data(tuple(targets), normal_cn_policy=normal_cn_policy)
    bounds = catalog.default_bounds() if bounds is None else bounds
    if not isinstance(bounds, MultiplicityBounds):
        raise TypeError("Prediction bounds must be MultiplicityBounds")
    selected_bounds = bounds.subset(data.mutation_ids, data.region_ids)
    numerical = prepare_observations(data, selected_bounds)
    row_lookup = {value: index for index, value in enumerate(catalog.mutation_ids)}
    rows = np.asarray([row_lookup[value] for value in data.mutation_ids], dtype=int)
    eligible = catalog.eligible[rows]
    digest = hashlib.sha256(b"clipp2.prediction-observations.v1\0")
    digest.update(catalog.content_hash.encode())
    digest.update(selected_bounds.content_hash.encode())
    return PreparedObservations(data, data.mutation_ids, data.region_ids, device,
        eligible=eligible, identity=digest.hexdigest(), bounds=selected_bounds,
        **{name: value for name, value in numerical.items()
           if name not in ("multiplicity_max", "multiplicity_restricted")})


__all__ = ["PreparedObservations", "prepare_prediction", "prepare_observations", "observed_mask"]
