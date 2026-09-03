"""Command-line interface for CliPP2 simulation generation."""

from __future__ import annotations

import argparse

from .config import (
    DEFAULT_MAX_LOCAL_CN_STATES_PER_MUTATION,
    CopyNumberEvolutionConfig,
    TumorSimulationConfig,
)


def add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the compact, canonical single-tumor simulation interface."""

    defaults = TumorSimulationConfig()
    parser.add_argument(
        "--out-dir",
        default=str(defaults.out_dir),
        help="Parent directory for the generated tumor.",
    )
    parser.add_argument(
        "--tumor-id",
        default=defaults.tumor_id,
        help="Name of the generated tumor directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Deterministic master seed.",
    )
    parser.add_argument(
        "--mutation-count",
        type=int,
        default=defaults.mutation_count,
        help="Exact number of SNVs.",
    )
    parser.add_argument(
        "--mean-depth",
        type=int,
        default=defaults.mean_depth,
        help="Mean Poisson sequencing depth.",
    )
    parser.add_argument(
        "--purity",
        type=float,
        default=defaults.purity,
        help="Target mean tumor purity.",
    )
    parser.add_argument(
        "--cna-event-rate",
        type=float,
        default=defaults.copy_number.cna_event_rate,
        help="Evolutionary branch gain rate per genomic segment.",
    )
    parser.add_argument(
        "--region-count",
        type=int,
        default=defaults.region_count,
        help="Number of regions, written as region1 through regionN.",
    )
    parser.add_argument(
        "--clone-count",
        type=int,
        default=defaults.clone_count,
        help="Exact evolutionary clone count.",
    )
    parser.add_argument(
        "--n-segments",
        type=int,
        default=defaults.copy_number.n_segments,
        help="Number of genomic copy-number intervals.",
    )
    parser.add_argument(
        "--segment-size-bp",
        type=int,
        default=defaults.copy_number.segment_size_bp,
        help="Length of each simulated genomic interval.",
    )
    parser.add_argument(
        "--min-two-state-snv-fraction",
        type=float,
        default=defaults.copy_number.min_two_state_snv_fraction,
        help="Required SNV fraction on exactly two local copy-number states.",
    )
    parser.add_argument(
        "--max-rejection-tries",
        type=int,
        default=defaults.max_rejection_tries,
        help="Maximum attempts used to satisfy the simulation invariants.",
    )


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
            max_local_cn_states_per_mutation=(DEFAULT_MAX_LOCAL_CN_STATES_PER_MUTATION),
            min_two_state_snv_fraction=args.min_two_state_snv_fraction,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one canonical CliPP2 simulation tumor."
    )
    add_simulation_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    from .generator import simulate_tumor

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
