from __future__ import annotations

import argparse
from pathlib import Path

from .core.model import FitOptions
from .runners.pipeline import process_one_file, run_directory


def _resolve_effective_device(device: str | None) -> str:
    requested = "auto" if device is None else str(device).strip().lower()
    if requested in {"cpu", "cuda"}:
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _parse_lambda_grid(value: str | None) -> list[float] | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "auto":
        return None
    return [float(piece) for piece in cleaned.split(",") if piece.strip()]


def _add_common_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lambda-grid", default=None, help="Comma-separated lambda grid or 'auto'.")
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
    parser.add_argument("--tol", type=float, default=1e-4, help="Optimization tolerance.")
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
    parser.add_argument("--disable-warm-start", action="store_true", help="Disable lambda-path warm starts.")
    parser.add_argument("--major-prior", type=float, default=0.5, help="Prior probability assigned to major-copy multiplicity.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
        help="Execution device for the Torch fusion backend. CUDA is the default; use 'auto' for CPU fallback.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "float32", "float64"],
        default="float64",
        help="Numeric dtype for Torch execution. Float64 is the default for BIC model selection; float16 requires CUDA.",
    )
    parser.add_argument(
        "--missing-cna-policy",
        choices=["error", "all_true"],
        default="error",
        help="Behavior when neither has_cna nor cna_observed is present in an input TSV.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print optimizer progress.")


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", default="CliPP2Sim_TSV", help="Directory with per-tumor TSV files.")
    parser.add_argument("--input-file", default=None, help="Optional single tumor TSV file.")
    parser.add_argument("--outdir", default="multi_region_clipp_results", help="Output directory.")
    _add_common_selection_args(parser)
    parser.add_argument(
        "--skip-outputs",
        "--skip-patient-outputs",
        "--skip-tumor-outputs",
        action="store_true",
        help="Skip per-tumor mutation/cluster/lambda files.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Process-level parallelism for directory runs.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on the number of files processed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipp2",
        description=(
            "CliPP2 BIC model selection for objective-faithful observed-data pairwise fusion."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit", help="Fit TSV files with certified BIC model selection.")
    _add_fit_args(fit_parser)
    return parser


def _fit_options_from_args(args: argparse.Namespace) -> FitOptions:
    return FitOptions(
        lambda_value=0.0,
        outer_max_iter=args.outer_max_iter,
        inner_max_iter=args.inner_max_iter,
        tol=args.tol,
        summary_tol=args.summary_tol,
        bic_partition_tol=args.bic_partition_tol,
        major_prior=args.major_prior,
        device=_resolve_effective_device(args.device),
        dtype=args.dtype,
        verbose=args.verbose,
    )


def _run_fit(args: argparse.Namespace) -> None:
    fit_options = _fit_options_from_args(args)
    lambda_grid = _parse_lambda_grid(args.lambda_grid)

    if args.input_file:
        summary = process_one_file(
            file_path=Path(args.input_file),
            outdir=Path(args.outdir),
            simulation_root=None,
            lambda_grid=lambda_grid,
            lambda_grid_mode="adaptive_bic",
            fit_options=fit_options,
            bic_df_scale=1.0,
            bic_cluster_penalty=0.0,
            selection_score="bic",
            use_warm_starts=not args.disable_warm_start,
            write_outputs=not args.skip_outputs,
            graph_file=Path(args.graph_file) if args.graph_file else None,
            missing_cna_policy=args.missing_cna_policy,
        )
        print(summary)
        return

    summary_df = run_directory(
        input_dir=Path(args.input_dir),
        outdir=Path(args.outdir),
        simulation_root=None,
        lambda_grid=lambda_grid,
        lambda_grid_mode="adaptive_bic",
        fit_options=fit_options,
        max_files=args.max_files,
        bic_df_scale=1.0,
        bic_cluster_penalty=0.0,
        selection_score="bic",
        use_warm_starts=not args.disable_warm_start,
        write_outputs=not args.skip_outputs,
        graph_file=Path(args.graph_file) if args.graph_file else None,
        missing_cna_policy=args.missing_cna_policy,
        workers=args.workers,
    )
    print(summary_df.head().to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _run_fit(args)


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
