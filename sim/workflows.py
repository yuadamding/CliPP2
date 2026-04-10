from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from ..io.conversion import ConversionConfig, convert_simulation_root_from_config
from .generation import (
    SimulationGridConfig,
    build_parser as build_simulation_parser,
    run_simulation_grid_from_config,
    simulation_config_from_args,
)


@dataclass(frozen=True)
class SimulationPackageConfig:
    simulation: SimulationGridConfig
    conversion: ConversionConfig


def generate_and_convert_simulation(config: SimulationPackageConfig) -> tuple[list[Path], list[Path]]:
    written_dirs = run_simulation_grid_from_config(config.simulation)
    written_files = convert_simulation_root_from_config(config.conversion)
    return written_dirs, written_files


def build_parser() -> argparse.ArgumentParser:
    parser = build_simulation_parser()
    parser.description = "Generate simulation folders and convert them into per-tumor TSV files."
    parser.add_argument(
        "--merged-out-dir",
        default="CliPP2Sim_TSV",
        help="Directory to write the merged per-tumor TSV files.",
    )
    return parser


def simulation_package_config_from_args(args: argparse.Namespace) -> SimulationPackageConfig:
    simulation_config = simulation_config_from_args(args)
    conversion_config = ConversionConfig(input_root=simulation_config.out_dir, output_root=args.merged_out_dir)
    return SimulationPackageConfig(simulation=simulation_config, conversion=conversion_config)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = simulation_package_config_from_args(args)
    written_dirs, written_files = generate_and_convert_simulation(config)
    print(
        f"Generated {len(written_dirs)} simulation folders in {config.simulation.out_dir} "
        f"and wrote {len(written_files)} TSV files in {config.conversion.output_root}"
    )


__all__ = [
    "SimulationPackageConfig",
    "build_parser",
    "generate_and_convert_simulation",
    "main",
    "simulation_package_config_from_args",
]
