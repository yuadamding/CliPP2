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
    DEFAULT_WORKSET_ADD_BATCH,
    DEFAULT_WORKSET_MAX_BYTES,
    DEFAULT_WORKSET_MAX_EXPANSIONS,
    DENSE_FALLBACK_POLICIES,
    normalize_dense_fallback_policy,
)

from .core.fusion.profiles import (
    COMPUTATION_PROFILE_NAMES,
    DEFAULT_COMPUTATION_PROFILE,
    get_computation_profile,
)
from .core.model import FitOptions
from .model_selection.contracts import (
    DEFAULT_SELECTION_CONTRACT,
    SELECTION_CONTRACT_IDS,
)
from .io.tumor_txt import DEFAULT_DOSAGE_PRIOR_PENALTY
from .runners.pipeline import process_tumor
from .simulation import simulate_tumor
from .simulation.cli import (
    add_simulation_arguments,
    tumor_simulation_config_from_args,
)


def _selection_score_argument(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized.startswith("clonal-"):
        raise argparse.ArgumentTypeError(
            "clonal-anchor selection scores were removed; use "
            "fixed-partition-dirichlet-score or fixed-partition-bic"
        )
    allowed = {
        "fixed-partition-dirichlet-score",
        "fixed-partition-bic",
    }
    if normalized not in allowed:
        raise argparse.ArgumentTypeError(
            "selection score must be fixed-partition-dirichlet-score, "
            "or fixed-partition-bic"
        )
    return normalized


def _add_fit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--outdir", default="clipp2_results")
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
    parser.add_argument("--outer-max-iter", type=int, default=None)
    parser.add_argument("--inner-max-iter", type=int, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--summary-tol", type=float, default=None)
    parser.add_argument("--selection-partition-tol", type=float, default=None)
    parser.add_argument("--selection-refit-tol", type=float, default=None)
    parser.add_argument("--selection-refit-max-iter", type=int, default=None)
    parser.add_argument(
        "--selection-contract",
        choices=SELECTION_CONTRACT_IDS,
        default=DEFAULT_SELECTION_CONTRACT,
        help=(
            "Immutable partition-selection contract. Raw-only preserves the "
            "0.3 estimator; hybrid adds selectable Ward/CEM partitions; the "
            "legacy compatibility contract uses its declared float64, graph, "
            "Dirichlet-weight, component-death, and raw-partition settings."
        ),
    )
    parser.add_argument(
        "--selection-score",
        type=_selection_score_argument,
        default="fixed-partition-dirichlet-score",
        help=(
            "Fixed-label selection criterion. The Dirichlet score is BIC plus "
            "the active selection contract's declared weight times the "
            "deviance of one exact allocation under its integrated symmetric "
            "Dirichlet prior. This is not posterior-entropy ICL."
        ),
    )
    parser.add_argument("--disable-warm-start", action="store_true")
    parser.add_argument("--major-prior", type=float, default=0.5)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default=DEFAULT_DEVICE
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "float32", "float64"],
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
    parser.add_argument(
        "--allow-heuristic-structure-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    simulate_parser = subparsers.add_parser(
        "simulate", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_simulation_arguments(simulate_parser)
    return parser


def _resolve_profile_defaults(args: argparse.Namespace) -> None:
    """Resolve profile-controlled defaults while preserving CLI overrides."""

    profile = get_computation_profile(args.profile)
    defaults = {
        "outer_max_iter": int(profile.outer_max_iter),
        "inner_max_iter": int(profile.inner_max_iter),
        "tol": float(profile.solver_tolerance),
        "dtype": str(profile.raw_dtype),
        "summary_tol": (
            1e-4
            if profile.is_strict
            else (2e-4 if profile.name == "balanced" else 1e-3)
        ),
        "selection_partition_tol": (
            1e-4
            if profile.is_strict
            else (2e-4 if profile.name == "balanced" else 1e-3)
        ),
        "selection_refit_tol": (
            1e-7
            if profile.is_strict
            else (1e-5 if profile.name == "balanced" else 1e-4)
        ),
        "selection_refit_max_iter": (
            128 if profile.is_strict else (64 if profile.name == "balanced" else 32)
        ),
        "certificate_max_iter": (
            int(DEFAULT_CERTIFICATE_MAX_ITER)
            if profile.is_strict
            else (128 if profile.name == "balanced" else 64)
        ),
        "certificate_refinement_rounds": (
            int(DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS)
            if profile.is_strict
            else (1 if profile.name == "balanced" else 0)
        ),
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fit":
        _resolve_profile_defaults(args)
        penalty = float(args.dosage_prior_penalty)
        if not math.isfinite(penalty) or penalty < 0.0:
            parser.error("--dosage-prior-penalty must be finite and nonnegative")
        for option_name in (
            "selection_partition_tol",
            "selection_refit_tol",
        ):
            value = float(getattr(args, option_name))
            if not math.isfinite(value) or value <= 0.0:
                parser.error(
                    f"--{option_name.replace('_', '-')} must be positive and finite"
                )
        if int(args.selection_refit_max_iter) < 1:
            parser.error("--selection-refit-max-iter must be positive")
    return args


def _fit_options_from_args(args: argparse.Namespace) -> FitOptions:
    return FitOptions(
        lambda_value=0.0,
        outer_max_iter=args.outer_max_iter,
        inner_max_iter=args.inner_max_iter,
        tol=args.tol,
        summary_tol=args.summary_tol,
        selection_score=args.selection_score.replace("-", "_"),
        selection_partition_tol=args.selection_partition_tol,
        selection_refit_tol=args.selection_refit_tol,
        selection_refit_max_iter=args.selection_refit_max_iter,
        selection_contract=args.selection_contract,
        major_prior=args.major_prior,
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
        allow_heuristic_structure_splits=args.allow_heuristic_structure_splits,
        verbose=args.verbose,
        computation_profile=args.profile,
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
    )
    print(summary)


__all__ = ["build_parser", "main", "parse_args"]


if __name__ == "__main__":
    main()
