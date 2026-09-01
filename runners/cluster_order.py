"""Deterministic presentation ordering for selected CCF clusters."""

from __future__ import annotations

import numpy as np


CCF_CLUSTER_ORDERING_METHOD = "identified_region_rms_distance_to_one_v1"


def ccf_cluster_order(
    centers: np.ndarray,
    *,
    statistically_identified: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return zero-based display labels and CCF distance for canonical clusters.

    Distances are root-mean-square distances from CCF 1 over statistically
    identified region coordinates. Clusters without an identified coordinate
    have an undefined distance and sort last. Exact ties retain canonical
    cluster order.
    """

    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("cluster centers must be a finite two-dimensional array.")
    identified = np.asarray(statistically_identified, dtype=bool)
    if identified.shape != values.shape:
        raise ValueError("cluster-center identification mask has the wrong shape.")

    counts = np.sum(identified, axis=1)
    squared = np.sum(np.where(identified, np.square(values - 1.0), 0.0), axis=1)
    distances = np.full(values.shape[0], np.nan, dtype=np.float64)
    np.sqrt(
        squared,
        out=distances,
        where=counts > 0,
    )
    distances[counts > 0] /= np.sqrt(counts[counts > 0])

    canonical = np.arange(values.shape[0], dtype=np.int64)
    order = np.lexsort((canonical, np.where(np.isnan(distances), np.inf, distances)))
    ordered_labels = np.empty(values.shape[0], dtype=np.int64)
    ordered_labels[order] = canonical
    return ordered_labels, distances, counts.astype(np.int64, copy=False)


__all__ = ["CCF_CLUSTER_ORDERING_METHOD", "ccf_cluster_order"]
