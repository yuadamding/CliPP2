"""Command-line interface for CliPP2 simulation generation."""

from __future__ import annotations

import argparse

from .config import (
    DEFAULT_CNA_EVENT_RATE_GRID,
    DEFAULT_MAX_LOCAL_CN_STATES_PER_MUTATION,
    DEFAULT_MIN_TWO_STATE_SNV_FRACTION,
    CopyNumberEvolutionConfig,
    SimulationGridConfig,
    TumorSimulationConfig,
)
from .generator import simulate_tumor


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def build_grid_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CliPP2 simulation datasets.")
    parser.add_argument(
        "--out-dir",
        default="CliPP2Sim",
        help="Directory to write simulated tumor folders.",
    )
    parser.add_argument(
        "--purity-list", default="0.3,0.6,0.9", help="Comma-separated purity values."
    )
    parser.add_argument(
        "--cna-event-rate-list",
        "--amp-rate-list",
        dest="amp_rate_list",
        default=",".join(str(value) for value in DEFAULT_CNA_EVENT_RATE_GRID),
        help="Comma-separated evolutionary branch gain rates per genomic segment.",
    )
    parser.add_argument(
        "--N-list",
        default="50,75,100,200,300,400,500,1000",
        help="Comma-separated mean depth values.",
    )
    parser.add_argument(
        "--n-samples-list",
        default="2,5,10,15",
        help="Comma-separated region-count values.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=20,
        help="Number of simulation replicates per scenario.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional master RNG seed."
    )
    parser.add_argument(
        "--K-min", type=int, default=2, help="Minimum number of clones."
    )
    parser.add_argument(
        "--K-max", type=int, default=10, help="Maximum number of clones."
    )
    parser.add_argument(
        "--lambda-mut",
        type=int,
        default=2000,
        help="Legacy single Poisson mean for mutation count.",
    )
    parser.add_argument(
        "--lambda-mut-list",
        default="300,600,1000,2000,4000",
        help="Comma-separated Poisson means for mutation counts; used by default to cover a wider mutation range.",
    )
    parser.add_argument(
        "--alpha-mut",
        type=float,
        default=10.0,
        help="Dirichlet concentration for mutation allocation.",
    )
    parser.add_argument(
        "--alpha-split",
        type=float,
        default=1.0,
        help="Tree-branching preference; 1 preserves uniform parent selection.",
    )
    parser.add_argument(
        "--alpha-lambda",
        type=float,
        default=5.0,
        help="Dirichlet concentration for lineage residual masses.",
    )
    parser.add_argument(
        "--tau-lineage-min",
        type=float,
        default=1.0,
        help="Minimum lineage concentration per region.",
    )
    parser.add_argument(
        "--tau-lineage-max",
        type=float,
        default=50.0,
        help="Maximum lineage concentration per region.",
    )
    parser.add_argument(
        "--purity-conc",
        type=float,
        default=50.0,
        help="Beta concentration for region purities.",
    )
    parser.add_argument(
        "--lineage-zero-prob",
        type=float,
        default=0.0,
        help="Probability of zeroing a lineage in a region. Default is 0.0 because clone CCF is now constrained to stay positive in every region.",
    )
    parser.add_argument(
        "--min-clone-ccf",
        type=float,
        default=0.02,
        help="Minimum allowed clone CCF in every region.",
    )
    parser.add_argument(
        "--min-clone-ccf-l2-norm",
        type=float,
        default=0.05,
        help="Minimum L2 norm of each clone's multiregion CCF vector.",
    )
    parser.add_argument(
        "--min-mutations-per-clone",
        type=int,
        default=15,
        help="Minimum number of mutations assigned to each clone.",
    )
    parser.add_argument(
        "--min-clone-ccf-distance",
        type=float,
        default=0.10,
        help="Minimum L2 distance between any two clones' multiregion CCF profiles.",
    )
    parser.add_argument(
        "--max-rejection-tries",
        type=int,
        default=1024,
        help="Maximum rejection-sampling attempts when enforcing clone CCF constraints.",
    )
    parser.add_argument(
        "--copy-number-mode",
        default="gain_only_retained_mutation_v1",
        help="Copy-number evolution model (currently gain_only_retained_mutation_v1).",
    )
    parser.add_argument(
        "--n-segments",
        type=int,
        default=100,
        help="Number of patient-level genomic segments.",
    )
    parser.add_argument(
        "--segment-size-bp",
        type=int,
        default=1_000_000,
        help="Size of each simulated genomic segment.",
    )
    parser.add_argument(
        "--mean-cna-span-segments",
        type=float,
        default=1.0,
        help="Mean contiguous span of one gain event.",
    )
    parser.add_argument(
        "--max-allele-cn",
        type=int,
        default=6,
        help="Maximum physical copies per parental homolog.",
    )
    parser.add_argument(
        "--trunk-cna-rate-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to the gain rate on the founding-clone branch.",
    )
    parser.add_argument(
        "--require-unique-cn-profiles",
        action="store_true",
        help="Require every evolutionary clone to have a distinct whole-genome CN profile.",
    )
    parser.add_argument(
        "--min-cn-clone-count",
        type=int,
        default=1,
        help="Minimum number of distinct whole-genome CN profiles.",
    )
    parser.add_argument(
        "--min-two-state-snv-fraction",
        type=float,
        default=DEFAULT_MIN_TWO_STATE_SNV_FRACTION,
        help=(
            "Minimum fraction of SNVs, weighted by mutation IDs, that must lie "
            "on segments with exactly two local allele-specific CN states "
            f"(default: {DEFAULT_MIN_TWO_STATE_SNV_FRACTION:.2f}; must exceed 0.5)."
        ),
    )
    return parser


def simulation_config_from_args(args: argparse.Namespace) -> SimulationGridConfig:
    return SimulationGridConfig(
        out_dir=args.out_dir,
        purity_list=tuple(parse_float_list(args.purity_list)),
        amp_rate_list=tuple(parse_float_list(args.amp_rate_list)),
        N_list=tuple(parse_int_list(args.N_list)),
        n_samples_list=tuple(parse_int_list(args.n_samples_list)),
        reps=args.reps,
        seed=args.seed,
        K_min=args.K_min,
        K_max=args.K_max,
        lambda_mut=args.lambda_mut,
        lambda_mut_list=tuple(parse_int_list(args.lambda_mut_list))
        if args.lambda_mut_list
        else None,
        alpha_mut=args.alpha_mut,
        alpha_split=args.alpha_split,
        alpha_lambda=args.alpha_lambda,
        tau_lineage_min=args.tau_lineage_min,
        tau_lineage_max=args.tau_lineage_max,
        purity_conc=args.purity_conc,
        lineage_zero_prob=args.lineage_zero_prob,
        min_clone_ccf=args.min_clone_ccf,
        min_clone_ccf_l2_norm=args.min_clone_ccf_l2_norm,
        min_mutations_per_clone=args.min_mutations_per_clone,
        min_clone_ccf_distance=args.min_clone_ccf_distance,
        max_rejection_tries=args.max_rejection_tries,
        copy_number=CopyNumberEvolutionConfig(
            mode=args.copy_number_mode,
            n_segments=args.n_segments,
            segment_size_bp=args.segment_size_bp,
            mean_cna_span_segments=args.mean_cna_span_segments,
            max_allele_cn=args.max_allele_cn,
            trunk_cna_rate_multiplier=args.trunk_cna_rate_multiplier,
            require_unique_cn_profiles=args.require_unique_cn_profiles,
            min_cn_clone_count=args.min_cn_clone_count,
            max_local_cn_states_per_mutation=(DEFAULT_MAX_LOCAL_CN_STATES_PER_MUTATION),
            min_two_state_snv_fraction=args.min_two_state_snv_fraction,
        ),
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
    parser = build_parser()
    args = parser.parse_args(argv)
    written_dir = simulate_tumor(tumor_simulation_config_from_args(args))
    print(f"Generated {written_dir}")


__all__ = [
    "add_simulation_arguments",
    "build_grid_parser",
    "build_parser",
    "main",
    "simulation_config_from_args",
    "tumor_simulation_config_from_args",
]
