from __future__ import annotations

import numpy as np

from ..io.data import PatientData


def default_lambda_grid(data: PatientData, mode: str = "dense_no_zero") -> list[float]:
    sample_scale = float(np.sqrt(max(data.num_samples, 1)))
    base = max(data.depth_scale, 1.0) * sample_scale
    if mode == "standard":
        grid = base * np.array([0.0, 0.005, 0.015, 0.05, 0.15], dtype=float)
    elif mode == "dense":
        grid = base * np.array([0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16], dtype=float)
    elif mode == "dense_no_zero":
        grid = base * np.array([0.005, 0.01, 0.02, 0.04, 0.08, 0.16], dtype=float)
    elif mode == "coarse_no_zero":
        grid = base * np.array([0.005, 0.02, 0.08, 0.16], dtype=float)
    else:
        raise ValueError(f"Unknown lambda grid mode: {mode}")
    return [float(value) for value in np.unique(np.round(grid, 6))]


def compute_classic_bic(loglik: float, num_clusters: int, data: PatientData) -> float:
    num_observations = max(data.num_mutations * data.num_samples, 1)
    degrees_of_freedom = max(int(num_clusters), 1) * data.num_samples
    return float(-2.0 * loglik + degrees_of_freedom * np.log(num_observations))


def compute_extended_bic(
    loglik: float,
    num_clusters: int,
    data: PatientData,
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
