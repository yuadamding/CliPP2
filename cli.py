from __future__ import annotations

import argparse
import math
from pathlib import Path

from .config import (
    MULTIPLICITY_POLICIES,
    DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
    DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    DEFAULT_DENSE_FALLBACK_POLICY,
    DEFAULT_DEVICE,
    DEFAULT_WORKSET_ADD_BATCH,
    DEFAULT_WORKSET_MAX_BYTES,
    DEFAULT_WORKSET_MAX_EXPANSIONS,
    DENSE_FALLBACK_POLICIES,
    normalize_dense_fallback_policy,
)

from .config import (
    COMPUTATION_PROFILE_NAMES,
    DEFAULT_COMPUTATION_PROFILE,
)
from .config import FitConfig, resolve_fit_config


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--outdir", default="clipp2_results")
    parser.add_argument(
        "--multiplicity-policy", choices=MULTIPLICITY_POLICIES,
        default="independent_broad",
        help="Shared policies require region-invariant allele-specific CN; balanced_single is simulator-specific.",
    )
    parser.add_argument(
        "--profile",
        choices=COMPUTATION_PROFILE_NAMES,
        default=DEFAULT_COMPUTATION_PROFILE,
        help=(
            "Single-tumor computation contract. Strict strengthens "
            "per-candidate KKT checks and fixed-partition refit certification; "
            "all profiles use a bounded lambda search."
        ),
    )
    parser.add_argument("--outer-max-iter", type=int, default=None)
    parser.add_argument("--inner-max-iter", type=int, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--selection-partition-tol", type=float, default=None)
    parser.add_argument("--selection-refit-tol", type=float, default=None)
    parser.add_argument("--selection-refit-max-iter", type=int, default=None)
    parser.add_argument("--disable-warm-start", action="store_true")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_DEVICE
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float64"],
        default=None,
    )
    parser.add_argument(
        "--workset-max-bytes", type=int, default=DEFAULT_WORKSET_MAX_BYTES
    )
    parser.add_argument(
        "--compressed-cache-max-bytes",
        type=int,
        default=DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    )
    parser.add_argument(
        "--dense-fallback-policy",
        choices=[value.replace("_", "-") for value in DENSE_FALLBACK_POLICIES],
        default=DEFAULT_DENSE_FALLBACK_POLICY.replace("_", "-"),
    )
    parser.add_argument(
        "--workset-add-batch", type=int, default=DEFAULT_WORKSET_ADD_BATCH
    )
    parser.add_argument(
        "--workset-max-expansions", type=int, default=DEFAULT_WORKSET_MAX_EXPANSIONS
    )
    parser.add_argument("--certificate-max-iter", type=int, default=None)
    parser.add_argument(
        "--certificate-refinement-rounds",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--certificate-column-tol-scale",
        type=float,
        default=DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-outputs", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipp2",
        description=(
            "Fit raw pairwise-fusion candidates and select an immutable "
            "partition by a reconstructible fixed-partition score under an "
            "explicit computation profile."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser(
        "fit", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_fit_args(fit_parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fit":
        for option_name in (
            "selection_partition_tol",
            "selection_refit_tol",
        ):
            raw_value = getattr(args, option_name)
            if raw_value is not None and (
                not math.isfinite(float(raw_value)) or float(raw_value) <= 0.0
            ):
                parser.error(
                    f"--{option_name.replace('_', '-')} must be positive and finite"
                )
        if (
            args.selection_refit_max_iter is not None
            and int(args.selection_refit_max_iter) < 1
        ):
            parser.error("--selection-refit-max-iter must be positive")
    return args


def _fit_config_from_args(args: argparse.Namespace) -> FitConfig:
    return resolve_fit_config(
        lambda_value=0.0,
        multiplicity_policy=args.multiplicity_policy,
        outer_max_iter=args.outer_max_iter,
        inner_max_iter=args.inner_max_iter,
        tol=args.tol,
        selection_partition_tol=args.selection_partition_tol,
        selection_refit_tol=args.selection_refit_tol,
        selection_refit_max_iter=args.selection_refit_max_iter,
        device=args.device,
        dtype=args.dtype,
        workset_max_bytes=args.workset_max_bytes,
        compressed_cache_max_bytes=args.compressed_cache_max_bytes,
        dense_fallback_policy=normalize_dense_fallback_policy(
            args.dense_fallback_policy
        ),
        workset_add_batch=args.workset_add_batch,
        workset_max_expansions=args.workset_max_expansions,
        certificate_max_iter=args.certificate_max_iter,
        certificate_refinement_rounds=args.certificate_refinement_rounds,
        certificate_column_tol_scale=args.certificate_column_tol_scale,
        verbose=args.verbose,
        computation_profile=args.profile,
    )


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
        use_warm_starts=not args.disable_warm_start,
        write_outputs=not args.skip_outputs,
    )
    print(_printable_summary(summary))


__all__ = ["build_parser", "main", "parse_args"]


if __name__ == "__main__":
    main()
