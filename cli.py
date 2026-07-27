from __future__ import annotations

import argparse
import math
from pathlib import Path

from .core.bic import LAMBDA_GRID_MODES
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
from .io.tumor_input import DEFAULT_DOSAGE_PRIOR_PENALTY
from .model_selection.config import (
    DEFAULT_LAMBDA_GRID_MODE,
    DEFAULT_SELECTION_SCORE,
    SELECTION_SCORE_NAMES,
)
from .runners.pipeline import process_tumor, run_cohort
from .simulation import simulate_tumor
from .simulation.cli import (
    add_simulation_arguments,
    tumor_simulation_config_from_args,
)


def _parse_lambda_grid(value: str | None) -> list[float] | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "auto":
        return None
    return [float(piece) for piece in cleaned.split(",") if piece.strip()]


def _add_common_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lambda-grid",
        default=None,
        help=(
            "Comma-separated legacy lambda grid or 'auto'. Prespecified values "
            "require --lambda-grid-mode adaptive_bic."
        ),
    )
    parser.add_argument(
        "--lambda-grid-mode",
        choices=list(LAMBDA_GRID_MODES),
        default=DEFAULT_LAMBDA_GRID_MODE,
        help=(
            "Automatic lambda strategy. partition_guided_admm discovers lambda "
            "online from the likelihood-partition initializer and ADMM fits and "
            "rejects an explicit --lambda-grid."
        ),
    )
    parser.add_argument(
        "--graph-file",
        default=None,
        help="Optional TSV defining a custom pairwise-fusion graph using either edge_u/edge_v or mutation_u/mutation_v columns, with optional edge_w.",
    )
    parser.add_argument(
        "--outer-max-iter",
        type=int,
        default=8,
        help="Maximum outer majorization iterations.",
    )
    parser.add_argument(
        "--inner-max-iter",
        type=int,
        default=30,
        help="Maximum inner convex-solver iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=DEFAULT_OPTIMIZATION_TOLERANCE,
        help="Optimization tolerance.",
    )
    parser.add_argument(
        "--summary-tol",
        type=float,
        default=1e-4,
        help="Explicit post-hoc fusion tolerance used for display summary clustering.",
    )
    parser.add_argument(
        "--bic-partition-tol",
        type=float,
        default=1e-4,
        help="Explicit fusion tolerance used to extract partitions for partition-refit BIC.",
    )
    parser.add_argument(
        "--disable-warm-start",
        action="store_true",
        help="Disable lambda-path warm starts.",
    )
    parser.add_argument(
        "--major-prior",
        type=float,
        default=0.5,
        help="Prior probability assigned to major-copy multiplicity.",
    )
    parser.add_argument(
        "--selection-score",
        choices=list(SELECTION_SCORE_NAMES),
        default=DEFAULT_SELECTION_SCORE,
        help=(
            "Model-selection criterion. partition_icl is assignment-aware and "
            "recommended; bic retains the legacy center-only criterion."
        ),
    )
    parser.add_argument(
        "--bic-df-scale",
        type=float,
        default=1.0,
        help="Continuous-parameter scale used by extended_bic.",
    )
    parser.add_argument(
        "--bic-cluster-penalty",
        type=float,
        default=0.0,
        help="Additional K*log(M) coefficient used by extended_bic.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=DEFAULT_DEVICE,
        help=(
            "Torch execution device. CUDA is required by default; 'auto' allows "
            "the runtime to select CPU when CUDA is unavailable."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "float32", "float64"],
        default=DEFAULT_DTYPE,
        help=(
            "Torch numeric dtype. Float64 is the production default; float16 "
            "requires CUDA."
        ),
    )
    parser.add_argument(
        "--inner-backend",
        choices=[value.replace("_", "-") for value in INNER_BACKENDS],
        default=DEFAULT_INNER_BACKEND.replace("_", "-"),
        help=(
            "Inner fusion backend. Dense is the production default, quotient-"
            "workset is opt-in, and auto is a compatibility alias for dense."
        ),
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
        help=(
            "Exact fallback policy. device-only never moves solver work to CPU; "
            "cpu-allowed permits CPU fallback; error disables dense fallback."
        ),
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
    parser.add_argument(
        "--materialize-full-dual",
        action="store_true",
        help=(
            "Quotient-workset debugging option: also materialize the guided "
            "E-by-S dual after a memory preflight. Dense already materializes "
            "the guide dual it requires."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print optimizer progress."
    )


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--cohort-dir",
        help="Directory whose immediate child directories are CliPP2 tumors.",
    )
    inputs.add_argument(
        "--tumor-dir",
        help="One directory in CliPP2's required tumor input format.",
    )
    parser.add_argument(
        "--unsupported-policy",
        choices=["error", "mask"],
        default="error",
        help=(
            "Fail on local copy-number mixtures outside the one/two-state model, "
            "or mask those mutation-sample count likelihoods with reason codes."
        ),
    )
    parser.add_argument(
        "--dosage-prior-penalty",
        type=float,
        default=DEFAULT_DOSAGE_PRIOR_PENALTY,
        help=(
            "Fixed phi-independent penalty alpha on endpoint mutant-copy mass: "
            "path weight is exp(-alpha * max(M_path(1)-1, 0)); the calibrated "
            f"default is {DEFAULT_DOSAGE_PRIOR_PENALTY:g}."
        ),
    )
    parser.add_argument("--outdir", default="clipp2_results", help="Output directory.")
    parser.add_argument(
        "--simulation-root",
        default=None,
        help="Optional simulation truth root used only to report benchmark metrics.",
    )
    _add_common_selection_args(parser)
    parser.add_argument(
        "--skip-outputs",
        "--skip-tumor-outputs",
        action="store_true",
        help="Skip per-tumor mutation/cluster/lambda files.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process-level parallelism for directory runs.",
    )
    parser.add_argument(
        "--max-tumors",
        type=int,
        default=None,
        help="Optional cap on the number of tumors processed.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipp2",
        description=(
            "CliPP2 canonical simulation and objective-faithful observed-data "
            "pairwise fusion. Production fitting defaults use CUDA, float64, "
            "dense device-only fusion, partition-guided ADMM, and partition ICL."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser(
        "fit",
        help="Fit tumor directories with certified partition-ICL model selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_fit_args(fit_parser)
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Generate one canonical tumor with hidden benchmark truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_simulation_arguments(simulate_parser)
    return parser


def _validate_fit_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    try:
        lambda_grid = _parse_lambda_grid(args.lambda_grid)
    except ValueError as exc:
        parser.error(f"invalid --lambda-grid: {exc}")

    prior_penalty = float(args.dosage_prior_penalty)
    if not math.isfinite(prior_penalty) or prior_penalty < 0.0:
        parser.error("--dosage-prior-penalty must be finite and nonnegative")
    if args.tumor_dir and args.max_tumors is not None:
        parser.error("--max-tumors is valid only with --cohort-dir")

    if args.lambda_grid_mode != DEFAULT_LAMBDA_GRID_MODE:
        return
    if lambda_grid is not None:
        parser.error(
            "--lambda-grid is incompatible with partition_guided_admm; use "
            "--lambda-grid-mode adaptive_bic for prespecified values"
        )
    if args.selection_score != DEFAULT_SELECTION_SCORE:
        parser.error(
            "partition_guided_admm requires --selection-score partition_icl; "
            "use --lambda-grid-mode adaptive_bic for bic or extended_bic"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fit":
        _validate_fit_args(parser, args)
    return args


def _fit_options_from_args(args: argparse.Namespace) -> FitOptions:
    return FitOptions(
        lambda_value=0.0,
        outer_max_iter=args.outer_max_iter,
        inner_max_iter=args.inner_max_iter,
        tol=args.tol,
        summary_tol=args.summary_tol,
        bic_partition_tol=args.bic_partition_tol,
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


def _run_fit(args: argparse.Namespace) -> None:
    fit_options = _fit_options_from_args(args)
    lambda_grid = _parse_lambda_grid(args.lambda_grid)

    if args.tumor_dir:
        summary = process_tumor(
            tumor_dir=Path(args.tumor_dir),
            outdir=Path(args.outdir),
            simulation_root=Path(args.simulation_root)
            if args.simulation_root
            else None,
            lambda_grid=lambda_grid,
            lambda_grid_mode=args.lambda_grid_mode,
            fit_options=fit_options,
            bic_df_scale=args.bic_df_scale,
            bic_cluster_penalty=args.bic_cluster_penalty,
            selection_score=args.selection_score,
            use_warm_starts=not args.disable_warm_start,
            write_outputs=not args.skip_outputs,
            graph_file=Path(args.graph_file) if args.graph_file else None,
            unsupported_policy=args.unsupported_policy,
            dosage_prior_penalty=args.dosage_prior_penalty,
        )
        print(summary)
        return

    summary_df = run_cohort(
        cohort_dir=Path(args.cohort_dir),
        outdir=Path(args.outdir),
        simulation_root=Path(args.simulation_root) if args.simulation_root else None,
        lambda_grid=lambda_grid,
        lambda_grid_mode=args.lambda_grid_mode,
        fit_options=fit_options,
        max_tumors=args.max_tumors,
        bic_df_scale=args.bic_df_scale,
        bic_cluster_penalty=args.bic_cluster_penalty,
        selection_score=args.selection_score,
        use_warm_starts=not args.disable_warm_start,
        write_outputs=not args.skip_outputs,
        graph_file=Path(args.graph_file) if args.graph_file else None,
        unsupported_policy=args.unsupported_policy,
        dosage_prior_penalty=args.dosage_prior_penalty,
        workers=args.workers,
    )
    print(summary_df.head().to_string(index=False))


def _run_simulate(args: argparse.Namespace) -> None:
    output_dir = simulate_tumor(tumor_simulation_config_from_args(args))
    print(output_dir)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "fit":
        _run_fit(args)
        return
    _run_simulate(args)


__all__ = ["build_parser", "main", "parse_args"]


if __name__ == "__main__":
    main()
