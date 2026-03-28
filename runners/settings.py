from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.data import PatientData


@dataclass(frozen=True)
class PatientRegime:
    num_samples: int
    num_mutations: int
    depth_scale: float
    mean_purity: float
    non_diploid_rate: float


@dataclass(frozen=True)
class RecommendedSettings:
    profile_name: str
    graph_k: int
    lambda_grid_mode: str
    bic_df_scale: float
    bic_cluster_penalty: float
    center_merge_tol: float


def summarize_patient_regime(data: PatientData) -> PatientRegime:
    non_diploid_mask = data.has_cna & ((data.major_cn != 1.0) | (data.minor_cn != 1.0))
    return PatientRegime(
        num_samples=int(data.num_samples),
        num_mutations=int(data.num_mutations),
        depth_scale=float(data.depth_scale),
        mean_purity=float(np.mean(data.purity)),
        non_diploid_rate=float(np.mean(non_diploid_mask)),
    )


def _refit_ebic_default_settings() -> RecommendedSettings:
    return RecommendedSettings(
        profile_name="refit_ebic_default",
        graph_k=8,
        lambda_grid_mode="dense_no_zero",
        bic_df_scale=8.0,
        bic_cluster_penalty=4.0,
        center_merge_tol=0.08,
    )


def recommend_settings_from_regime(regime: PatientRegime, *, selection_score: str = "refit_ebic") -> RecommendedSettings:
    if selection_score == "refit_ebic":
        return _refit_ebic_default_settings()

    if regime.depth_scale <= 100.0:
        return RecommendedSettings(
            profile_name="strong_low_depth",
            graph_k=8,
            lambda_grid_mode="dense_no_zero",
            bic_df_scale=10.0,
            bic_cluster_penalty=6.0,
            center_merge_tol=0.10,
        )

    if regime.num_samples <= 1:
        return RecommendedSettings(
            profile_name="moderate_sparse_graph",
            graph_k=6,
            lambda_grid_mode="dense_no_zero",
            bic_df_scale=8.0,
            bic_cluster_penalty=4.0,
            center_merge_tol=0.08,
        )

    if regime.mean_purity <= 0.35 and regime.num_samples <= 5:
        return RecommendedSettings(
            profile_name="fast_low_purity",
            graph_k=8,
            lambda_grid_mode="coarse_no_zero",
            bic_df_scale=10.0,
            bic_cluster_penalty=6.0,
            center_merge_tol=0.10,
        )

    if regime.num_samples >= 10:
        return RecommendedSettings(
            profile_name="balanced_high_dimension",
            graph_k=8,
            lambda_grid_mode="dense_no_zero",
            bic_df_scale=8.0,
            bic_cluster_penalty=4.0,
            center_merge_tol=0.08,
        )

    return RecommendedSettings(
        profile_name="strong_default",
        graph_k=8,
        lambda_grid_mode="dense_no_zero",
        bic_df_scale=10.0,
        bic_cluster_penalty=6.0,
        center_merge_tol=0.10,
    )


def recommend_settings_from_data(data: PatientData, *, selection_score: str = "refit_ebic") -> RecommendedSettings:
    return recommend_settings_from_regime(summarize_patient_regime(data), selection_score=selection_score)


__all__ = [
    "PatientRegime",
    "RecommendedSettings",
    "recommend_settings_from_data",
    "recommend_settings_from_regime",
    "summarize_patient_regime",
]
