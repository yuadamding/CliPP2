"""Configuration and defaults for CliPP2 evolutionary simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_CNA_EVENT_RATE = 1.5
DEFAULT_CNA_EVENT_RATE_GRID = (1.0, 1.5, 2.0)
DEFAULT_MAX_LOCAL_CN_STATES_PER_MUTATION = 2
DEFAULT_MIN_TWO_STATE_SNV_FRACTION = 0.60
GENERATOR_VERSION = "evolution_gain_only_benchmark_v5"
OUTPUT_SCHEMA_VERSION = "5.0"


@dataclass(frozen=True)
class CopyNumberEvolutionConfig:
    """Settings for the inference-matched gain-only evolutionary simulator."""

    mode: str = "gain_only_retained_mutation_v1"
    n_segments: int = 100
    segment_size_bp: int = 1_000_000
    cna_event_rate: float = DEFAULT_CNA_EVENT_RATE
    mean_cna_span_segments: float = 1.0
    max_allele_cn: int = 6
    trunk_cna_rate_multiplier: float = 1.0
    mutation_time_mode: str = "uniform_on_origin_branch"
    ensure_positive_descendant_dosage: bool = True
    require_unique_cn_profiles: bool = False
    min_cn_clone_count: int = 1
    max_local_cn_states_per_mutation: int | None = (
        DEFAULT_MAX_LOCAL_CN_STATES_PER_MUTATION
    )
    min_two_state_snv_fraction: float | None = DEFAULT_MIN_TWO_STATE_SNV_FRACTION
    allow_losses: bool = False
    allow_cnloh: bool = False
    allow_wgd: bool = False


@dataclass(frozen=True)
class SimulationGridConfig:
    out_dir: str | Path = "CliPP2Sim"
    purity_list: tuple[float, ...] = (0.3, 0.6, 0.9)
    amp_rate_list: tuple[float, ...] = DEFAULT_CNA_EVENT_RATE_GRID
    N_list: tuple[int, ...] = (50, 75, 100, 200, 300, 400, 500, 1000)
    n_samples_list: tuple[int, ...] = (2, 5, 10, 15)
    reps: int = 20
    seed: int | None = None
    K_min: int = 2
    K_max: int = 10
    lambda_mut: int = 2000
    lambda_mut_list: tuple[int, ...] | None = (300, 600, 1000, 2000, 4000)
    alpha_mut: float = 10.0
    alpha_split: float = 1.0
    alpha_lambda: float = 5.0
    tau_lineage_min: float = 1.0
    tau_lineage_max: float = 50.0
    purity_conc: float = 50.0
    lineage_zero_prob: float = 0.0
    min_clone_ccf: float = 0.02
    min_clone_ccf_l2_norm: float = 0.05
    min_mutations_per_clone: int = 15
    min_clone_ccf_distance: float = 0.10
    max_rejection_tries: int = 1024
    copy_number: CopyNumberEvolutionConfig = field(
        default_factory=CopyNumberEvolutionConfig
    )


@dataclass(frozen=True)
class TumorSimulationConfig:
    """One reproducible tumor written in CliPP2's canonical input format."""

    out_dir: str | Path = "simulations"
    tumor_id: str = "simulatedTumor1"
    seed: int = 1
    mutation_count: int = 300
    mean_depth: int = 100
    purity: float = 0.6
    region_count: int = 2
    clone_count: int = 3
    min_mutations_per_clone: int = 15
    max_rejection_tries: int = 1024
    copy_number: CopyNumberEvolutionConfig = field(
        default_factory=lambda: CopyNumberEvolutionConfig(n_segments=10)
    )


def _validate_copy_number_config(config: CopyNumberEvolutionConfig) -> None:
    if config.mode != "gain_only_retained_mutation_v1":
        raise ValueError(
            "Only mode='gain_only_retained_mutation_v1' is implemented; "
            "loss, cnLOH, WGD, and mutation-loss models are reserved for labeled stress simulations."
        )
    if config.n_segments < 1:
        raise ValueError("n_segments must be at least 1.")
    if config.segment_size_bp < 1:
        raise ValueError("segment_size_bp must be at least 1.")
    if config.cna_event_rate < 0.0:
        raise ValueError("cna_event_rate must be nonnegative.")
    if config.mean_cna_span_segments < 1.0:
        raise ValueError("mean_cna_span_segments must be at least 1.")
    if config.max_allele_cn < 1:
        raise ValueError("max_allele_cn must be at least 1.")
    if config.trunk_cna_rate_multiplier < 0.0:
        raise ValueError("trunk_cna_rate_multiplier must be nonnegative.")
    if config.mutation_time_mode != "uniform_on_origin_branch":
        raise ValueError(
            "Only mutation_time_mode='uniform_on_origin_branch' is implemented."
        )
    if config.min_cn_clone_count < 1:
        raise ValueError("min_cn_clone_count must be at least 1.")
    max_local_states = config.max_local_cn_states_per_mutation
    if max_local_states is not None and (
        isinstance(max_local_states, bool)
        or int(max_local_states) != max_local_states
        or int(max_local_states) < 1
    ):
        raise ValueError(
            "max_local_cn_states_per_mutation must be a positive integer or None."
        )
    min_two_state_fraction = config.min_two_state_snv_fraction
    if min_two_state_fraction is not None and (
        not np.isfinite(float(min_two_state_fraction))
        or float(min_two_state_fraction) <= 0.5
        or float(min_two_state_fraction) > 1.0
    ):
        raise ValueError(
            "min_two_state_snv_fraction must lie in (0.5, 1.0] or be None."
        )
    if (
        min_two_state_fraction is not None
        and max_local_states is not None
        and int(max_local_states) < 2
    ):
        raise ValueError(
            "min_two_state_snv_fraction requires "
            "max_local_cn_states_per_mutation >= 2 or None."
        )
    if config.allow_losses or config.allow_cnloh or config.allow_wgd:
        raise ValueError(
            "Loss, cnLOH, and WGD are out-of-model stress settings and are not enabled in the matched gain-only simulator."
        )


__all__ = [
    "CopyNumberEvolutionConfig",
    "SimulationGridConfig",
    "TumorSimulationConfig",
]
