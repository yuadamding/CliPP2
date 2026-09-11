from __future__ import annotations

import argparse
import math
from pathlib import Path

from .config import DEFAULT_DEVICE, DEFAULT_MAX_MAJOR_CN, FitConfig, resolve_fit_config, validate_max_major_cn
from ._version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipp2", allow_abbrev=False,
        description="Fit the independent integer-mixture hybrid fusion estimator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit", allow_abbrev=False,
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    fit.add_argument("--input-file", required=True)
    fit.add_argument("--outdir", default="clipp2_results")
    fit.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULT_DEVICE)
    fit.add_argument(
        "--max-major-cn", type=int, default=DEFAULT_MAX_MAJOR_CN,
        help="Exclude a mutation from all regions if any region's major CN exceeds this limit.",
    )
    fit.add_argument("--verbose", action="store_true")
    fit.add_argument("--version", action="version", version=__version__)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_max_major_cn(args.max_major_cn)
    except ValueError as error:
        parser.error(str(error))
    return args


def _fit_config_from_args(args: argparse.Namespace) -> FitConfig:
    return resolve_fit_config(device=args.device, verbose=args.verbose, max_major_cn=args.max_major_cn)


def _printable_summary(value: object) -> object:
    """Return the summary with non-finite floats replaced by None.

    The printed representation is consumed by launch wrappers as a Python
    literal; bare nan/inf tokens are name nodes, not literals, so a
    non-finite value must never reach stdout.
    """

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
    )
    print(_printable_summary(summary))


__all__ = ["build_parser", "main", "parse_args"]


if __name__ == "__main__":
    main()
