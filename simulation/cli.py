"""Command-line interface for CliPP2 simulation generation."""

from __future__ import annotations

import argparse

from .config import (
    CopyNumberEvolutionConfig,
    TumorSimulationConfig,
)
from .generator import simulate_tumor


def add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the compact, canonical single-tumor simulation interface."""

    defaults = TumorSimulationConfig()
    arguments = (
        ("out-dir", None, str(defaults.out_dir), "Parent directory for the generated tumor."),
        ("tumor-id", None, defaults.tumor_id, "Name of the generated tumor directory."),
        ("seed", int, defaults.seed, "Deterministic master seed."),
        ("mutation-count", int, defaults.mutation_count, "Exact number of SNVs."),
        ("mean-depth", int, defaults.mean_depth, "Mean Poisson sequencing depth."),
        ("purity", float, defaults.purity, "Target mean tumor purity."),
        ("cna-event-rate", float, defaults.copy_number.cna_event_rate,
         "Gain rate per genomic segment in each independent region; CN is locally clonal."),
        ("region-count", int, defaults.region_count,
         "Number of regions, written as region1 through regionN."),
        ("clone-count", int, defaults.clone_count, "Exact evolutionary clone count."),
        ("n-segments", int, defaults.copy_number.n_segments,
         "Number of genomic copy-number intervals."),
        ("segment-size-bp", int, defaults.copy_number.segment_size_bp,
         "Length of each simulated genomic interval."),
        ("max-rejection-tries", int, defaults.max_rejection_tries,
         "Maximum attempts used to satisfy the simulation invariants."),
    )
    for name, value_type, default, help_text in arguments:
        parser.add_argument(f"--{name}", type=value_type, default=default, help=help_text)


def tumor_simulation_config_from_args(
    args: argparse.Namespace,
) -> TumorSimulationConfig:
    return TumorSimulationConfig(
        out_dir=args.out_dir,
        tumor_id=args.tumor_id,
        seed=args.seed,
        mutation_count=args.mutation_count,
        mean_depth=args.mean_depth,
        purity=args.purity,
        region_count=args.region_count,
        clone_count=args.clone_count,
        max_rejection_tries=args.max_rejection_tries,
        copy_number=CopyNumberEvolutionConfig(
            n_segments=args.n_segments,
            segment_size_bp=args.segment_size_bp,
            cna_event_rate=args.cna_event_rate,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one canonical CliPP2 simulation tumor."
    )
    add_simulation_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    written_dir = simulate_tumor(tumor_simulation_config_from_args(args))
    print(f"Generated {written_dir}")


__all__ = [
    "add_simulation_arguments",
    "build_parser",
    "main",
    "tumor_simulation_config_from_args",
]
