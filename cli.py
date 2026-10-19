from __future__ import annotations

import argparse
from dataclasses import fields
import math
from pathlib import Path

import numpy as np

from .config import (DEFAULT_DEVICE, DEFAULT_MAX_CLUSTERS, DEFAULT_MAX_MAJOR_CN,
                     MAX_CLUSTERS, FitConfig, resolve_fit_config)
from ._version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipp2", allow_abbrev=False,
        description="Fit the occupied-clonal joint-region penalized classification estimator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ('fit', 'import-and-polish'):
        fit = subparsers.add_parser(command, allow_abbrev=False,
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        _add_fit_arguments(fit)
        if command == 'import-and-polish':
            fit.set_defaults(search_policy='bounded')
            fit.add_argument('--source-checkpoint', required=True,
                             help='Immutable historical candidate checkpoint; never rewritten.')
            fit.add_argument('--import-authority', required=True,
                             help='Versioned checksum/source/scientific-input import authority JSON.')
    return parser


def _add_fit_arguments(fit):
    fit.add_argument("--input-file", required=True)
    fit.add_argument("--outdir", default="clipp2_results")
    fit.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULT_DEVICE)
    fit.add_argument('--regional-zero-policy', choices=['strict', 'likelihood'], default='strict',
                     help='Strict: observed zero-alt positive-depth cells require exact-zero assigned CCF; '
                          'likelihood: use zero-ALT observations in the read likelihood without this constraint.')
    fit.add_argument('--selection-objective',
                     choices=['classification', 'assigned_reads', 'legacy_mixture'], default='classification',
                     help='Classification: assigned reads plus occupied-count proportions and nominal penalty; '
                          'assigned_reads omits the count term; legacy_mixture explicitly restores mixture scoring.')
    fit.add_argument(
        "--max-major-cn", type=int, default=DEFAULT_MAX_MAJOR_CN,
        help="Exclude a mutation from all regions if any CN state's major CN in any region exceeds this limit.",
    )
    fit.add_argument("--verbose", action="store_true")
    fit.add_argument('--ram-budget-gib', type=float, default=1.0,
                     help='Additional host acceleration workspace ceiling; 0 disables resident caches.')
    fit.add_argument('--vram-budget-gib', type=float, default=0.0,
                     help='Additional CUDA cache/batch workspace ceiling; opt-in, not total device memory.')
    fit.add_argument('--search-policy', choices=['auto', 'exhaustive', 'bounded'], default='auto',
                     help='Auto uses bounded search for active strict zeros, disconnected observation overlap '
                          'or K>10; otherwise reference search.')
    fit.add_argument('--search-width', type=int, default=16,
                     help='Active anchored states per occupied cluster count in bounded search.')
    fit.add_argument('--search-proposals', type=int, default=5,
                     help='Move-family slots per active state per sweep in bounded search; '
                          'five covers merge, split, boundary, single and greedy batch.')
    fit.add_argument('--search-sweeps', type=int, default=2,
                     help='Bounded exploration sweeps (downward then upward, alternating).')
    fit.add_argument('--reassignment-starts', type=int, default=4,
                     help='Starting states for exhaustive-route fixed-K polishing; bounded mode uses its sweep schedule.')
    fit.add_argument('--reassignment-rounds', type=int, default=2,
                     help='Maximum exhaustive-route polishing rounds per start; bounded mode uses its sweep schedule. '
                          'Hitting a budget is not convergence.')
    fit.add_argument('--checkpoint-path', help='JSON search checkpoint path for bounded search.')
    fit.add_argument('--resume', action='store_true', help='Resume a source/model/input-bound bounded-search checkpoint.')
    fit.add_argument('--numerical-diagnostics-dir',
                     help='Opt-in directory for bounded frozen first/worst QP failure snapshots.')
    fit.add_argument('--numerical-diagnostics-max-bytes', type=int, default=16_777_216,
                     help='Maximum total failure-snapshot payload bytes.')
    fit.add_argument('--polish-max-visits', type=int, default=256,
                     help='Dedicated cumulative finalization visits, separate from discovery. '
                          'Unfinished checks prevent successful publication.')
    fit.add_argument('--polish-diverse-starts', type=int, default=8,
                     help='Additional distinct finalists; overall and best-per-K states are mandatory.')
    fit.add_argument('--polish-joint-limit', type=int, default=12,
                     help='Bounded atomic anchor/swap proposal admissions per finalization visit.')
    fit.add_argument('--polish-joint-moves', action=argparse.BooleanOptionalAction, default=True,
                     help='Enable atomic anchor neighborhoods; disable for explicit Stage A transfer/refit comparison.')
    fit.add_argument('--qualify-device-weights', action='store_true',
                     help='Compare experimental batched CUDA weights with authoritative CPU fits (extra work).')
    fit.add_argument("--max-clusters", type=int, choices=range(1, MAX_CLUSTERS + 1),
                     default=DEFAULT_MAX_CLUSTERS,
                     help=f"Largest occupied cluster capacity (at most {MAX_CLUSTERS}).")
    fit.add_argument("--version", action="version", version=__version__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _fit_config_from_args(args)
        if args.resume and not args.checkpoint_path:
            raise ValueError('--resume requires --checkpoint-path')
        if args.checkpoint_path and args.search_policy != 'bounded':
            raise ValueError('--checkpoint-path requires --search-policy bounded')
        if args.command == 'import-and-polish' and not args.checkpoint_path:
            raise ValueError('import-and-polish requires a fresh --checkpoint-path (or explicit --resume)')
    except ValueError as error:
        parser.error(str(error))
    return args


def _fit_config_from_args(args: argparse.Namespace) -> FitConfig:
    return resolve_fit_config(**{field.name: getattr(args, field.name) for field in fields(FitConfig)})


def _printable_summary(value: object) -> object:
    """Return the summary with non-finite floats replaced by None.

    The printed representation is consumed by launch wrappers as a Python
    literal; bare nan/inf tokens are name nodes, not literals, so a
    non-finite value must never reach stdout.
    """

    if isinstance(value, np.ndarray):
        return _printable_summary(value.tolist())
    if isinstance(value, np.generic):
        return _printable_summary(value.item())
    if isinstance(value, dict):
        return {key: _printable_summary(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_printable_summary(entry) for entry in value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from .api import process_tumor

    summary = process_tumor(
        tumor_file=Path(args.input_file),
        outdir=Path(args.outdir),
        fit_config=_fit_config_from_args(args),
        checkpoint_path=args.checkpoint_path, resume=args.resume,
        import_checkpoint=getattr(args, 'source_checkpoint', None),
        import_authority=getattr(args, 'import_authority', None),
    )
    print(_printable_summary(summary))


__all__ = ["build_parser", "main", "parse_args"]


if __name__ == "__main__":
    main()
