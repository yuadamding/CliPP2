from __future__ import annotations

import argparse
import math
from pathlib import Path

from .core.fusion.defaults import (
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

from .core.fusion.profiles import (
    COMPUTATION_PROFILE_NAMES,
    DEFAULT_COMPUTATION_PROFILE,
)
from .config import FitConfig, resolve_fit_config
from .model_selection.contracts import (
    DEFAULT_SELECTION_CONTRACT,
    SELECTION_CONTRACT_IDS,
)
from .io.tumor_txt import DEFAULT_DOSAGE_PRIOR_PENALTY
from .runners.pipeline import (
    DEFAULT_FAILURE_POLICY,
    FAILURE_POLICIES,
    process_tumor,
)
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
        "--failure-policy",
        choices=FAILURE_POLICIES,
        default=DEFAULT_FAILURE_POLICY,
        help=(
            "Behavior when no primary certified estimator is available. "
            "Best-effort saves the highest valid typed tier; save-diagnostics "
            "suppresses conditional point estimates; error exits. No policy "
            "relaxes a certificate gate."
        ),
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
    parser.add_argument(
        "--max-tumor-edge-pass-equivalents",
        type=int,
        default=None,
        help=(
            "Deterministic cumulative raw-solver work cap. The current lambda "
            "observation finishes atomically before the search stops unresolved."
        ),
    )
    parser.add_argument(
        "--max-partition-refit-objective-evaluations",
        type=int,
        default=None,
        help=(
            "Cumulative scalar-objective budget for fixed-partition refits; "
            "the current candidate finishes atomically before stopping."
        ),
    )
    parser.add_argument(
        "--max-direct-partition-candidates",
        type=int,
        default=None,
        help="Maximum number of direct Ward/CEM candidates evaluated.",
    )
    parser.add_argument(
        "--recovery-policy",
        choices=["staged", "legacy"],
        default="staged",
        help=(
            "Staged recovery detects objective/KKT stagnation and probes terminal "
            "certificate refinement before deepening."
        ),
    )
    parser.add_argument(
        "--stagnation-audit-patience",
        type=int,
        default=4,
        help="Number of recovery KKT audits in the stagnation window.",
    )
    parser.add_argument(
        "--lambda-no-progress-patience",
        type=int,
        default=3,
        help=(
            "Certified refinement proposals tolerated without a new scored "
            "partition, improved score/KKT, or narrower partition event."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-lambda",
        action="store_true",
        help=(
            "Atomically checkpoint controller, candidates, and CPU-offloaded "
            "solver states after every completed lambda observation."
        ),
    )
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help=(
            "Checkpoint directory; defaults to "
            "OUTDIR/.clipp2-checkpoints/TUMOR.npz."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help=(
            "Resume the exact identity-matched online search from this checkpoint "
            "directory and keep updating it."
        ),
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
        if (
            args.max_tumor_edge_pass_equivalents is not None
            and int(args.max_tumor_edge_pass_equivalents) < 1
        ):
            parser.error("--max-tumor-edge-pass-equivalents must be positive")
        if (
            args.max_partition_refit_objective_evaluations is not None
            and int(args.max_partition_refit_objective_evaluations) < 1
        ):
            parser.error(
                "--max-partition-refit-objective-evaluations must be positive"
            )
        if (
            args.max_direct_partition_candidates is not None
            and int(args.max_direct_partition_candidates) < 1
        ):
            parser.error("--max-direct-partition-candidates must be positive")
        if int(args.stagnation_audit_patience) < 1:
            parser.error("--stagnation-audit-patience must be positive")
        if int(args.lambda_no_progress_patience) < 1:
            parser.error("--lambda-no-progress-patience must be positive")
        if args.checkpoint_file and not args.checkpoint_every_lambda:
            parser.error("--checkpoint-file requires --checkpoint-every-lambda")
    return args


def _fit_config_from_args(args: argparse.Namespace) -> FitConfig:
    return resolve_fit_config(
        lambda_value=0.0,
        outer_max_iter=args.outer_max_iter,
        inner_max_iter=args.inner_max_iter,
        tol=args.tol,
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
        max_tumor_edge_pass_equivalents=args.max_tumor_edge_pass_equivalents,
        max_partition_refit_objective_evaluations=(
            args.max_partition_refit_objective_evaluations
        ),
        max_direct_partition_candidates=args.max_direct_partition_candidates,
        recovery_policy=args.recovery_policy,
        stagnation_audit_patience=args.stagnation_audit_patience,
        lambda_no_progress_patience=args.lambda_no_progress_patience,
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
    if args.command == "simulate":
        print(simulate_tumor(tumor_simulation_config_from_args(args)))
        return
    summary = process_tumor(
        tumor_file=Path(args.input_file),
        outdir=Path(args.outdir),
        fit_config=_fit_config_from_args(args),
        use_warm_starts=not args.disable_warm_start,
        write_outputs=not args.skip_outputs,
        unsupported_policy=args.unsupported_policy,
        dosage_prior_penalty=args.dosage_prior_penalty,
        failure_policy=args.failure_policy,
        checkpoint_every_lambda=bool(args.checkpoint_every_lambda),
        checkpoint_file=args.checkpoint_file,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(_printable_summary(summary))


__all__ = ["build_parser", "main", "parse_args"]


if __name__ == "__main__":
    main()
