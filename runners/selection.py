from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.data import TumorData


FIXED_LAMBDA_GRID_MODES = (
    "standard",
    "dense",
    "dense_no_zero",
    "coarse_no_zero",
    "ultra_dense_no_zero",
)
CV_STABILITY_LAMBDA_GRID_MODES = (
    "adaptive_cv",
    "adaptive_cv_stability",
    "adaptive_cv_stability_one_se",
)
BIC_ADAPTIVE_LAMBDA_GRID_MODES = (
    "adaptive_bic",
    "adaptive_ebic_path",
)
ADAPTIVE_LAMBDA_GRID_MODES = BIC_ADAPTIVE_LAMBDA_GRID_MODES + CV_STABILITY_LAMBDA_GRID_MODES
LAMBDA_GRID_MODES = FIXED_LAMBDA_GRID_MODES + ADAPTIVE_LAMBDA_GRID_MODES


@dataclass(frozen=True)
class LambdaBracket:
    lambda_min: float
    lambda_eq: float
    lambda_full: float
    anchors: list[float]
    diagnostics: dict[str, float]


def is_adaptive_lambda_grid_mode(mode: str) -> bool:
    return str(mode).strip().lower() in ADAPTIVE_LAMBDA_GRID_MODES


def is_cv_stability_lambda_grid_mode(mode: str) -> bool:
    return str(mode).strip().lower() in CV_STABILITY_LAMBDA_GRID_MODES


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
    if is_adaptive_lambda_grid_mode(mode):
        raise ValueError(f"{mode} requires data-adaptive lambda anchors, not a fixed default grid.")
    grid = _default_lambda_ratios(mode=mode)
    return [float(value) for value in np.unique(np.round(grid, 6))]


def compute_classic_bic(loglik: float, num_clusters: int, data: TumorData) -> float:
    num_observations = effective_bic_cell_count(data)
    degrees_of_freedom = bic_degrees_of_freedom(num_clusters, data)
    return float(-2.0 * loglik + degrees_of_freedom * np.log(num_observations))


def effective_bic_cell_count(data: TumorData) -> int:
    return max(int(np.sum(np.asarray(data.total_counts, dtype=np.float64) > 0.0)), 1)


def effective_bic_depth_count(data: TumorData) -> float:
    positive_depth = np.asarray(data.total_counts, dtype=np.float64)
    return float(max(float(np.sum(positive_depth[positive_depth > 0.0])), 1.0))


def bic_degrees_of_freedom(num_clusters: int, data: TumorData) -> int:
    return max(int(num_clusters), 1) * int(data.num_samples)


def compute_classic_bic_depth_n(loglik: float, num_clusters: int, data: TumorData) -> float:
    num_observations = effective_bic_depth_count(data)
    degrees_of_freedom = bic_degrees_of_freedom(num_clusters, data)
    return float(-2.0 * loglik + degrees_of_freedom * np.log(num_observations))


def compute_extended_bic(
    loglik: float,
    num_clusters: int,
    data: TumorData,
    bic_df_scale: float,
    bic_cluster_penalty: float,
) -> float:
    num_observations = effective_bic_cell_count(data)
    cluster_count = max(int(num_clusters), 1)
    cp_degrees_of_freedom = bic_degrees_of_freedom(cluster_count, data)
    cluster_complexity = cluster_count
    return float(
        -2.0 * loglik
        + bic_df_scale * cp_degrees_of_freedom * np.log(num_observations)
        + bic_cluster_penalty * cluster_complexity * np.log(max(data.num_mutations, 2))
    )


__all__ = [
    "ADAPTIVE_LAMBDA_GRID_MODES",
    "BIC_ADAPTIVE_LAMBDA_GRID_MODES",
    "CV_STABILITY_LAMBDA_GRID_MODES",
    "FIXED_LAMBDA_GRID_MODES",
    "LAMBDA_GRID_MODES",
    "LambdaBracket",
    "bic_degrees_of_freedom",
    "compute_classic_bic",
    "compute_classic_bic_depth_n",
    "compute_extended_bic",
    "default_lambda_grid",
    "effective_bic_cell_count",
    "effective_bic_depth_count",
    "is_adaptive_lambda_grid_mode",
    "is_cv_stability_lambda_grid_mode",
]
