from __future__ import annotations

import argparse
import math
from pathlib import Path

from .core.fusion.defaults import (
    DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
    DEFAULT_CERTIFICATE_MAX_ITER,
    DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS,
    DEFAULT_COMPRESSED_CACHE_MAX_BYTES,
    DEFAULT_DENSE_FALLBACK_POLICY,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_INNER_BACKEND,
    DEFAULT_OPTIMIZATION_TOLERANCE,
    DEFAULT_WORKSET_ADD_BATCH,
    DEFAULT_WORKSET_MAX_BYTES,
    DEFAULT_WORKSET_MAX_EXPANSIONS,
    DENSE_FALLBACK_POLICIES,
    INNER_BACKENDS,
    normalize_dense_fallback_policy,
    normalize_inner_backend,
)
from .core.model import FitOptions
from .io.tumor_txt import DEFAULT_DOSAGE_PRIOR_PENALTY
from .model_selection.config import FINAL_PHI_WARD_LADDER_KMAX
from .runners.pipeline import process_tumor
from .simulation import simulate_tumor
from .simulation.cli import (
    add_simulation_arguments,
    tumor_simulation_config_from_args,
)


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--outdir", default="clipp2_results")
    parser.add_argument(
        "--unsupported-policy", choices=["error", "mask"], default="error"
    )
    parser.add_argument(
        "--dosage-prior-penalty",
        type=float,
        default=DEFAULT_DOSAGE_PRIOR_PENALTY,
        help=(
            "Fixed phi-independent penalty alpha on endpoint mutant-copy mass. "
            "This applies only when at least one locus contains subclonal copy "
            "number."
        ),
    )
    parser.add_argument("--outer-max-iter", type=int, default=8)
    parser.add_argument("--inner-max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=DEFAULT_OPTIMIZATION_TOLERANCE)
    parser.add_argument("--summary-tol", type=float, default=1e-4)
    parser.add_argument("--selection-partition-tol", type=float, default=1e-4)
    parser.add_argument("--selection-refit-tol", type=float, default=1e-7)
    parser.add_argument("--selection-refit-max-iter", type=int, default=128)
    parser.add_argument("--reporting-partition-tol", type=float, default=1e-4)
    parser.add_argument(
        "--selection-score",
        choices=["clonal-fixed-partition-bic", "fixed-partition-bic"],
        default="clonal-fixed-partition-bic",
    )
    parser.add_argument(
        "--selection-anchor",
        choices=["clonal-required", "none"],
        default="clonal-required",
        help=(
            "Clonal-required fixes one deterministically selected mutation at "
            "its feasible clonal CCF in the raw fusion objective; none leaves "
            "all raw CCF coordinates free."
        ),
    )
    parser.add_argument(
        "--raw-clonal-anchor-mode",
        choices=["none", "specified-seed", "enumerated-seed", "screened-seed"],
        default="screened-seed",
        help=(
            "Hard CCF=1 raw-anchor search: specified seed, complete enumeration, "
            "or an explicitly incomplete deviance-screened candidate set."
        ),
    )
    parser.add_argument(
        "--raw-clonal-anchor-mutation",
        action="append",
        default=[],
        help="Retained mutation ID for specified-seed mode (exactly one).",
    )
    parser.add_argument("--raw-clonal-anchor-target", type=float, default=1.0)
    parser.add_argument(
        "--raw-clonal-anchor-feasibility-tol", type=float, default=1e-8
    )
    parser.add_argument("--raw-clonal-anchor-candidate-max", type=int, default=8)
    parser.add_argument("--disable-warm-start", action="store_true")
    parser.add_argument("--major-prior", type=float, default=0.5)
    parser.add_argument(
        "--kmax",
        type=int,
        default=FINAL_PHI_WARD_LADDER_KMAX,
        help="Deprecated compatibility option; production requires 0",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_DEVICE
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "float32", "float64"],
        default=DEFAULT_DTYPE,
    )
    parser.add_argument(
        "--inner-backend",
        choices=[value.replace("_", "-") for value in INNER_BACKENDS],
        default=DEFAULT_INNER_BACKEND.replace("_", "-"),
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
    parser.add_argument(
        "--certificate-max-iter", type=int, default=DEFAULT_CERTIFICATE_MAX_ITER
    )
    parser.add_argument(
        "--certificate-refinement-rounds",
        type=int,
        default=DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS,
    )
    parser.add_argument(
        "--certificate-column-tol-scale",
        type=float,
        default=DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE,
    )
    parser.add_argument(
        "--allow-heuristic-structure-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--materialize-full-dual", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-outputs", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipp2",
        description=(
            "Fit certified raw pairwise-fusion candidates and select an "
            "immutable partition by fixed-partition BIC."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser(
        "fit", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    _add_fit_args(fit_parser)
    simulate_parser = subparsers.add_parser(
        "simulate", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_simulation_arguments(simulate_parser)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fit":
        penalty = float(args.dosage_prior_penalty)
        if not math.isfinite(penalty) or penalty < 0.0:
            parser.error("--dosage-prior-penalty must be finite and nonnegative")
        for option_name in (
            "selection_partition_tol",
            "selection_refit_tol",
            "reporting_partition_tol",
        ):
            value = float(getattr(args, option_name))
            if not math.isfinite(value) or value <= 0.0:
                parser.error(
                    f"--{option_name.replace('_', '-')} must be positive and finite"
                )
        if int(args.selection_refit_max_iter) < 1:
            parser.error("--selection-refit-max-iter must be positive")
        if int(args.kmax) != 0:
            parser.error("--kmax is deprecated and must be 0 in production mode")
        expected_anchor = (
            "clonal-required"
            if args.selection_score == "clonal-fixed-partition-bic"
            else "none"
        )
        if args.selection_anchor != expected_anchor:
            parser.error(
                f"--selection-score {args.selection_score} requires "
                f"--selection-anchor {expected_anchor}"
            )
        raw_anchor_mode = str(args.raw_clonal_anchor_mode)
        expected_raw_anchor = expected_anchor == "clonal-required"
        if expected_raw_anchor == (raw_anchor_mode == "none"):
            parser.error(
                "clonal-fixed-partition-bic requires a non-none raw clonal "
                "anchor mode; fixed-partition-bic requires mode none"
            )
        if raw_anchor_mode == "specified-seed" and len(
            args.raw_clonal_anchor_mutation
        ) != 1:
            parser.error("specified-seed requires exactly one anchor mutation")
        if raw_anchor_mode != "specified-seed" and args.raw_clonal_anchor_mutation:
            parser.error("anchor mutation IDs apply only to specified-seed mode")
        if float(args.raw_clonal_anchor_target) != 1.0:
            parser.error("production raw clonal-anchor target must equal 1")
        if (
            not math.isfinite(float(args.raw_clonal_anchor_feasibility_tol))
            or float(args.raw_clonal_anchor_feasibility_tol) < 0.0
        ):
            parser.error("raw clonal-anchor feasibility tolerance must be nonnegative")
        if raw_anchor_mode == "screened-seed" and int(
            args.raw_clonal_anchor_candidate_max
        ) < 1:
            parser.error("screened-seed candidate maximum must be positive")
    return args


def _fit_options_from_args(args: argparse.Namespace) -> FitOptions:
    return FitOptions(
        lambda_value=0.0,
        outer_max_iter=args.outer_max_iter,
        inner_max_iter=args.inner_max_iter,
        tol=args.tol,
        summary_tol=args.summary_tol,
        selection_score=args.selection_score.replace("-", "_"),
        selection_anchor=args.selection_anchor.replace("-", "_"),
        raw_clonal_anchor_mode=args.raw_clonal_anchor_mode.replace("-", "_"),
        raw_clonal_anchor_mutation_ids=tuple(args.raw_clonal_anchor_mutation),
        raw_clonal_anchor_target=args.raw_clonal_anchor_target,
        raw_clonal_anchor_feasibility_tol=args.raw_clonal_anchor_feasibility_tol,
        raw_clonal_anchor_candidate_max=args.raw_clonal_anchor_candidate_max,
        selection_partition_tol=args.selection_partition_tol,
        selection_refit_tol=args.selection_refit_tol,
        selection_refit_max_iter=args.selection_refit_max_iter,
        reporting_partition_tol=args.reporting_partition_tol,
        major_prior=args.major_prior,
        device=args.device,
        dtype=args.dtype,
        inner_backend=normalize_inner_backend(args.inner_backend),
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
        allow_heuristic_structure_splits=args.allow_heuristic_structure_splits,
        materialize_full_dual=args.materialize_full_dual,
        verbose=args.verbose,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "simulate":
        print(simulate_tumor(tumor_simulation_config_from_args(args)))
        return
    summary = process_tumor(
        tumor_file=Path(args.input_file),
        outdir=Path(args.outdir),
        fit_options=_fit_options_from_args(args),
        use_warm_starts=not args.disable_warm_start,
        write_outputs=not args.skip_outputs,
        unsupported_policy=args.unsupported_policy,
        dosage_prior_penalty=args.dosage_prior_penalty,
        ward_ladder_kmax=max(int(args.kmax), 0),
    )
    print(summary)


__all__ = ["build_parser", "main", "parse_args"]


if __name__ == "__main__":
    main()
