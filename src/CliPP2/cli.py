from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (DEFAULT_DEVICE, DEFAULT_MAX_CLUSTERS,
                     DEFAULT_MAX_MAJOR_CN, DEFAULT_MISSING_COUNTS, PROTOCOL_VERSION, FitConfig, resolve_fit_config)
from ._version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipp2", allow_abbrev=False,
        description="Fit joint regional CCFs with tree-free marginal-mixture search (new fits: CPU/marginal only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit", allow_abbrev=False,
                               help="Fit mixture centers and weights directly from observations; CPU/marginal only.",
                               description="Tree-free joint marginal-mixture fitting; CUDA/profile requests fail explicitly.",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    inputs = fit.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-file", help="Canonical tumor TSV (optionally gzip-compressed).")
    inputs.add_argument("--input-dir", dest="input_file", metavar="INPUT_DIR",
                        help="Tumor directory with regional snv.txt/cna.txt/purity.txt bundles.")
    fit.add_argument("--output-dir", default="clipp2_results")
    fit.add_argument("--multiplicity-support", help="Complete keyed TSV of multiplicity_lower/multiplicity_upper.")
    fit.add_argument("--normal-cn-policy", choices=["diploid"], default=None,
                     help="Explicitly assume diploid normal CN only where it was not supplied.")
    fit.add_argument("--missing-counts", choices=("zero_alt", "unobserved"), default=DEFAULT_MISSING_COUNTS,
                     help="Missing pairs use ALT=0 and REF=regional median observed total depth; unobserved opts out.")
    fit.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULT_DEVICE,
                     help="New mixture fits require cpu; cuda is unsupported and never falls back.")
    fit.add_argument("--fit-multiplicity", choices=("marginal", "profile"), default="marginal",
                     help="New fits require marginal: jointly optimize centers and weights. Profile fitting is unsupported; Stage 3 profile assignment is separate.")
    fit.add_argument("--max-multiplicity-difference", choices=("1", "none"), default="1",
                     help="Maximum regional multiplicity difference; none uses independent regional supports.")
    fit.add_argument(
        "--max-major-cn", type=int, default=DEFAULT_MAX_MAJOR_CN,
        help="Explicit restricted-population CN ceiling; default retains all valid integer CN.",
    )
    fit.add_argument("--max-clusters", type=int, default=DEFAULT_MAX_CLUSTERS,
                     help="Maximum positive mixture components, including components without MAP members (positive integer).")
    fit.add_argument("--cpu-threads", type=int, default=None,
                     help="CPU threads; fallback CLIPP2_CPU_THREADS, OMP_NUM_THREADS, then affinity/quota.")
    fit.add_argument("--run-manifest-id", help="External execution record identifier; no source scan is performed.")
    fit.add_argument("--version", action="version", version=__version__)
    workflow = subparsers.add_parser("workflow", allow_abbrev=False,
                                    help="Run Protocol 2.5 from a v7 plan; new fits require CPU/marginal, historical accepted fits may be replayed.")
    workflow.add_argument("--plan", required=True)
    workflow.add_argument("--output-dir", required=True)
    assign = subparsers.add_parser("assign", allow_abbrev=False,
                                  help="Reassess frozen targets once under full original support.")
    assign.add_argument("--run-manifest-id", help="External identifier for this assignment execution.")
    assign.add_argument("--bundle", required=True)
    assign.add_argument("--assignment-method", choices=("marginal", "profile"), default=None,
                        help="Use the frozen method unless explicitly deriving a new request.")
    assign.add_argument("--output-dir", required=True)
    audit = subparsers.add_parser("audit-input", allow_abbrev=False,
                                 help="Report input readiness and read imputation without fitting or changing source files.")
    inputs = audit.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-file", dest="input_path")
    inputs.add_argument("--input-dir", dest="input_path")
    audit.add_argument("--output", required=True)
    audit.add_argument("--normal-cn-policy", choices=["diploid"])
    audit.add_argument("--max-major-cn", type=int, default=DEFAULT_MAX_MAJOR_CN)
    audit.add_argument("--missing-counts", choices=("zero_alt", "unobserved"), default=DEFAULT_MISSING_COUNTS)
    experiment = subparsers.add_parser("experiment-mixture", allow_abbrev=False,
        help="CPU saved-population mixture comparison; output is not an accepted production fit.")
    experiment.add_argument("--catalog", required=True, help="Saved original_catalog.json.")
    experiment.add_argument("--accepted-fit", required=True,
                            help="Validated saved marginal Stage 1/2 fit supplying the population, support and initializer.")
    experiment.add_argument("--output", required=True, help="New experimental JSON; not a workflow fit.")
    experiment.add_argument("--max-clusters", type=int, default=None,
                            help="Mixture component cap; defaults to the saved fit's search cap.")
    experiment.add_argument("--proposal-budget", type=int, default=32,
                            help="Bounded proposal slots; no global-optimality certificate.")
    experiment.add_argument("--cpu-threads", type=int, default=1)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "fit":
            args._fit_config = _fit_config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def _fit_config_from_args(args: argparse.Namespace) -> FitConfig:
    return resolve_fit_config(device=args.device,
                              max_major_cn=args.max_major_cn, max_clusters=args.max_clusters,
                              cpu_threads=args.cpu_threads,
                              fit_multiplicity=args.fit_multiplicity,
                              max_multiplicity_difference=(None if args.max_multiplicity_difference == "none" else 1))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "experiment-mixture":
        from .experimental import run_mixture_experiment
        run_mixture_experiment(args.catalog, args.accepted_fit, args.output,
            max_clusters=args.max_clusters, proposal_budget=args.proposal_budget,
            cpu_threads=args.cpu_threads)
        return
    if args.command == "audit-input":
        from .io.tumor_txt import audit_input
        from .io.publication import publish_json
        result = audit_input(args.input_path, max_major_cn=args.max_major_cn, normal_cn_policy=args.normal_cn_policy,
                             missing_counts=args.missing_counts)
        publish_json(args.output, json.dumps(result, indent=2, allow_nan=False) + "\n")
        if result["status"] != "ready":
            raise SystemExit(1)
        return
    if args.command == "assign":
        from .assignment import assign_mutations, derive_assignment_bundle
        from .io.assignment import load_assignment_bundle
        request = load_assignment_bundle(args.bundle)
        request = derive_assignment_bundle(request, assignment_method=args.assignment_method)
        assign_mutations(request, outdir=args.output_dir, run_manifest_id=args.run_manifest_id)
        return
    if args.command == "workflow":
        from .config import WorkflowConfig
        from .workflow import run_workflow
        plan_path = Path(args.plan).resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if (plan.get("schema") != "clipp2.workflow.plan.v7"
                or plan.get("protocol_version") != PROTOCOL_VERSION):
            raise ValueError("Workflow requires plan.v7 with protocol_version='2.5'")
        extra = set(plan) - {"schema", "protocol_version", "input_path", "workflow_config", "accepted_stage_one", "accepted_stage_two", "run_build_manifest", "run_manifest_id"}
        if extra:
            raise ValueError(f"Unknown workflow plan fields: {sorted(extra)}")
        def relative(value):
            path = Path(value)
            return path if path.is_absolute() else plan_path.parent / path
        receipt = run_workflow(relative(plan["input_path"]), args.output_dir,
            WorkflowConfig.from_dict(plan["workflow_config"]),
            accepted_stage_one=relative(plan["accepted_stage_one"]) if plan.get("accepted_stage_one") else None,
            accepted_stage_two=relative(plan["accepted_stage_two"]) if plan.get("accepted_stage_two") else None,
            run_build_manifest=relative(plan["run_build_manifest"]) if plan.get("run_build_manifest") else None,
            run_manifest_id=plan.get("run_manifest_id"))
        if receipt["status"] == "blocked":
            raise SystemExit("Workflow blocked: " + str(receipt.get("unsupported_regions", [])))
        return
    from .api import process_tumor

    process_tumor(
        tumor_file=Path(args.input_file),
        outdir=Path(args.output_dir),
        fit_config=args._fit_config,
        multiplicity_support=args.multiplicity_support,
        normal_cn_policy=args.normal_cn_policy,
        missing_counts=args.missing_counts,
        run_manifest_id=args.run_manifest_id,
    )


__all__ = ["build_parser", "main", "parse_args"]


if __name__ == "__main__":
    main()
