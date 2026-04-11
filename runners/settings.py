from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..io.data import TumorData


@dataclass(frozen=True)
class TumorRegime:
    num_regions: int
    num_mutations: int
    depth_scale: float
    mean_purity: float
    non_diploid_rate: float

    @property
    def num_samples(self) -> int:
        return self.num_regions


@dataclass(frozen=True)
class RecommendedSettings:
    profile_name: str
    lambda_grid_mode: str
    bic_df_scale: float
    bic_cluster_penalty: float


PatientRegime = TumorRegime


def summarize_tumor_regime(data: TumorData) -> TumorRegime:
    non_diploid_mask = data.has_cna & ((data.major_cn != 1.0) | (data.minor_cn != 1.0))
    return TumorRegime(
        num_regions=int(data.num_regions),
        num_mutations=int(data.num_mutations),
        depth_scale=float(data.depth_scale),
        mean_purity=float(np.mean(data.purity)),
        non_diploid_rate=float(np.mean(non_diploid_mask)),
    )


def summarize_patient_regime(data: TumorData) -> PatientRegime:
    return summarize_tumor_regime(data)


def recommend_settings_from_regime(
    regime: TumorRegime,
    *,
    selection_score: str = "ebic",
) -> RecommendedSettings:
    normalized_score = str(selection_score).strip().lower()

    if normalized_score == "oracle_ari":
        return RecommendedSettings(
            profile_name="pairwise_fusion_oracle_dense",
            lambda_grid_mode="ultra_dense_no_zero",
            bic_df_scale=8.0,
            bic_cluster_penalty=4.0,
        )

    if regime.num_regions <= 2 or regime.depth_scale <= 300.0 or regime.num_mutations <= 800:
        return RecommendedSettings(
            profile_name="pairwise_fusion_ultra",
            lambda_grid_mode="ultra_dense_no_zero",
            bic_df_scale=8.0,
            bic_cluster_penalty=4.0,
        )

    if regime.num_regions >= 10 or regime.num_mutations >= 2500:
        return RecommendedSettings(
            profile_name="pairwise_fusion_dense",
            lambda_grid_mode="dense_no_zero",
            bic_df_scale=8.0,
            bic_cluster_penalty=4.0,
        )

    return RecommendedSettings(
        profile_name="pairwise_fusion_default",
        lambda_grid_mode="dense_no_zero",
        bic_df_scale=8.0,
        bic_cluster_penalty=4.0,
    )


def recommend_settings_from_data(
    data: TumorData,
    *,
    selection_score: str = "ebic",
) -> RecommendedSettings:
    return recommend_settings_from_regime(
        summarize_tumor_regime(data),
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
