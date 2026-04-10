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

    @property
    def num_regions(self) -> int:
        return self.num_samples


@dataclass(frozen=True)
class RecommendedSettings:
    profile_name: str
    lambda_grid_mode: str
    bic_df_scale: float
    bic_cluster_penalty: float


TumorRegime = PatientRegime


def summarize_patient_regime(data: PatientData) -> PatientRegime:
    non_diploid_mask = data.has_cna & ((data.major_cn != 1.0) | (data.minor_cn != 1.0))
    return PatientRegime(
        num_samples=int(data.num_samples),
        num_mutations=int(data.num_mutations),
        depth_scale=float(data.depth_scale),
        mean_purity=float(np.mean(data.purity)),
        non_diploid_rate=float(np.mean(non_diploid_mask)),
    )


def summarize_tumor_regime(data: PatientData) -> TumorRegime:
    return summarize_patient_regime(data)


def recommend_settings_from_regime(
    regime: PatientRegime,
    *,
    selection_score: str = "refit_ebic",
) -> RecommendedSettings:
    del selection_score

    if regime.num_samples <= 2 or regime.depth_scale <= 300.0 or regime.num_mutations <= 800:
        return RecommendedSettings(
            profile_name="direct_partition_ultra",
            lambda_grid_mode="ultra_dense_no_zero",
            bic_df_scale=8.0,
            bic_cluster_penalty=4.0,
        )

    if regime.num_samples >= 10 or regime.num_mutations >= 2500:
        return RecommendedSettings(
            profile_name="direct_partition_dense",
            lambda_grid_mode="dense_no_zero",
            bic_df_scale=8.0,
            bic_cluster_penalty=4.0,
        )

    return RecommendedSettings(
        profile_name="direct_partition_default",
        lambda_grid_mode="dense_no_zero",
        bic_df_scale=8.0,
        bic_cluster_penalty=4.0,
    )


def recommend_settings_from_data(
    data: PatientData,
    *,
    selection_score: str = "refit_ebic",
) -> RecommendedSettings:
    return recommend_settings_from_regime(
        summarize_patient_regime(data),
        selection_score=selection_score,
    )


__all__ = [
    "PatientRegime",
    "TumorRegime",
    "RecommendedSettings",
    "recommend_settings_from_data",
    "recommend_settings_from_regime",
    "summarize_patient_regime",
    "summarize_tumor_regime",
]
