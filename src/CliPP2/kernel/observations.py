"""Immutable prediction preparation sharing the fitting emission contract.

Prediction deliberately retains uninformative rows and regions. Native contexts
consume these fixed arrays without changing support in an existing context.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib

import numpy as np
from scipy.special import gammaln

from ..io.data import ImmutableArrayRecord, OriginalCatalog, MultiplicityBounds, TumorData, readonly_array


@dataclass(frozen=True, slots=True)
class PreparedObservations(ImmutableArrayRecord):
    data: TumorData
    mutation_ids: tuple[str, ...]
    region_ids: tuple[str, ...]
    device: str
    alt: np.ndarray
    depth: np.ndarray
    observed: np.ndarray
    mixed: np.ndarray
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
                       *, device="cpu", cpu_threads=None) -> PreparedObservations:
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
    data = catalog.prediction_data(tuple(targets))
    bounds = catalog.default_bounds() if bounds is None else bounds
    if not isinstance(bounds, MultiplicityBounds):
        raise TypeError("Prediction bounds must be MultiplicityBounds")
    lower_m, upper_m = bounds.for_data(data)
    selected_bounds = bounds.subset(data.mutation_ids, data.region_ids)
    width = upper_m-lower_m+1
    observed = np.asarray(data.count_observed, dtype=bool) & (data.total_counts > 0)
    alt = np.where(observed, data.alt_counts, 0.)
    depth = np.where(observed, data.total_counts, 0.)
    if (not np.isfinite(alt).all() or not np.isfinite(depth).all()
            or np.any(alt < 0) or np.any(alt > depth)
            or np.any(alt != np.rint(alt)) or np.any(depth != np.rint(depth))):
        raise ValueError("Informative prediction counts must be finite nonnegative integers")
    mixed = data.cn_state_count > 1
    denominator = (1-data.purity)*data.normal_cn + data.purity*data.mean_total_cn
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise ValueError("Prediction requires positive biological denominators")
    scaling = data.purity/denominator
    constants = gammaln(depth+1)-gammaln(alt+1)-gammaln(depth-alt+1)-np.log(width)
    row_lookup = {value: index for index, value in enumerate(catalog.mutation_ids)}
    rows = np.asarray([row_lookup[value] for value in data.mutation_ids], dtype=int)
    eligible = catalog.eligible[rows]
    digest = hashlib.sha256(b"clipp2.prediction-observations.v1\0")
    digest.update(catalog.content_hash.encode())
    digest.update(selected_bounds.content_hash.encode())
    return PreparedObservations(data, data.mutation_ids, data.region_ids, device,
        alt, depth, observed, mixed, eligible, data.major_cn, data.normal_cn, data.mean_total_cn,
        data.purity, scaling, constants, np.where(mixed, 1e-6, 0.),
        np.where(mixed, data.phi_upper, 1.), lower_m, width, digest.hexdigest(), selected_bounds)


__all__ = ["PreparedObservations", "prepare_prediction"]
