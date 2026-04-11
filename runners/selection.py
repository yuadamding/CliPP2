from __future__ import annotations

import numpy as np

from ..io.data import TumorData


def _default_lambda_ratios(mode: str) -> np.ndarray:
    if mode == "standard":
        return np.array([0.5, 2.0, 8.0, 32.0, 128.0], dtype=float)
    if mode == "dense":
        return np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0], dtype=float)
    if mode == "dense_no_zero":
        return np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0], dtype=float)
    if mode == "ultra_dense_no_zero":
        return np.array(
            [0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0, 256.0, 384.0, 512.0, 768.0, 1024.0],
            dtype=float,
        )
    if mode == "coarse_no_zero":
        return np.array([1.0, 8.0, 64.0, 512.0], dtype=float)
    raise ValueError(f"Unknown lambda grid mode: {mode}")


def default_lambda_grid(
    data: TumorData,
    mode: str = "dense_no_zero",
) -> list[float]:
    del data
    grid = _default_lambda_ratios(mode=mode)
    return [float(value) for value in np.unique(np.round(grid, 6))]


def compute_classic_bic(loglik: float, num_clusters: int, data: TumorData) -> float:
    num_observations = max(data.num_mutations * data.num_samples, 1)
    degrees_of_freedom = max(int(num_clusters), 1) * data.num_samples
    return float(-2.0 * loglik + degrees_of_freedom * np.log(num_observations))


def compute_extended_bic(
    loglik: float,
    num_clusters: int,
    data: TumorData,
    bic_df_scale: float,
    bic_cluster_penalty: float,
) -> float:
    num_observations = max(data.num_mutations * data.num_samples, 1)
    cluster_count = max(int(num_clusters), 1)
    cp_degrees_of_freedom = cluster_count * data.num_samples
    cluster_complexity = cluster_count
    return float(
        -2.0 * loglik
        + bic_df_scale * cp_degrees_of_freedom * np.log(num_observations)
        + bic_cluster_penalty * cluster_complexity * np.log(max(data.num_mutations, 2))
    )


__all__ = [
    "compute_classic_bic",
    "compute_extended_bic",
    "default_lambda_grid",
]
