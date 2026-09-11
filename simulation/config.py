"""Configuration and defaults for CliPP2 evolutionary simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAX_SIMULATION_ALLELE_CN = 6
DEFAULT_CNA_EVENT_RATE = 1.5
GENERATOR_VERSION = "tree_clonal_cn_uniform_multiplicity_v7"
OUTPUT_SCHEMA_VERSION = "7.0"


@dataclass(frozen=True)
class CopyNumberEvolutionConfig:
    """Clonal trunk gains; every descendant inherits the same bounded CN."""

    n_segments: int = 100
    segment_size_bp: int = 1_000_000
    cna_event_rate: float = DEFAULT_CNA_EVENT_RATE
    mean_cna_span_segments: float = 1.0
    max_allele_cn: int = MAX_SIMULATION_ALLELE_CN


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


def _positive_integer(value: object, name: str, *, minimum: int = 1) -> None:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < minimum
        or int(value) != value
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _validate_copy_number_config(config: CopyNumberEvolutionConfig) -> None:
    for name in ("n_segments", "segment_size_bp", "max_allele_cn"):
        _positive_integer(getattr(config, name), name)
    if config.max_allele_cn > MAX_SIMULATION_ALLELE_CN:
        raise ValueError(f"max_allele_cn must not exceed {MAX_SIMULATION_ALLELE_CN}.")
    for name, minimum in (("cna_event_rate", 0.0), ("mean_cna_span_segments", 1.0)):
        value = getattr(config, name)
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or value < minimum
        ):
            raise ValueError(f"{name} must be finite and >= {minimum}.")


__all__ = [
    "CopyNumberEvolutionConfig",
    "TumorSimulationConfig",
]
