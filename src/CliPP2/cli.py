from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (DEFAULT_AUDIT_SCOPE, DEFAULT_DEVICE, DEFAULT_MAX_CLUSTERS,
                     DEFAULT_MAX_MAJOR_CN, PROTOCOL_VERSION, FitConfig, resolve_fit_config)
from ._version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clipp2", allow_abbrev=False,
        description="Fit the free-center tree-proposed joint-region multiplicity-mixture estimator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit", allow_abbrev=False,
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    inputs = fit.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-file", help="Canonical tumor TSV (optionally gzip-compressed).")
    inputs.add_argument("--input-dir", dest="input_file", metavar="INPUT_DIR",
                        help="Tumor directory with regional snv.txt/cna.txt/purity.txt bundles.")
    fit.add_argument("--output-dir", default="clipp2_results")
    fit.add_argument("--multiplicity-support", help="Complete keyed TSV of multiplicity_lower/multiplicity_upper.")
    fit.add_argument("--normal-cn-policy", choices=["diploid"], default=None,
                     help="Explicitly assume diploid normal CN only where it was not supplied.")
    fit.add_argument("--device", choices=["cpu", "cuda"], default=DEFAULT_DEVICE)
    fit.add_argument(
        "--max-major-cn", type=int, default=DEFAULT_MAX_MAJOR_CN,
        help="Exclude a mutation only from regions where major CN exceeds this limit.",
    )
    fit.add_argument("--max-clusters", type=int, default=DEFAULT_MAX_CLUSTERS,
                     help="Largest occupied cluster capacity (positive integer; no fixed upper limit).")
    fit.add_argument("--audit-scope", choices=("selected", "all"), default=DEFAULT_AUDIT_SCOPE,
                     help="Certify the selected count, or every represented count, by complete single-mutation passes.")
    fit.add_argument("--cpu-threads", type=int, default=None,
                     help="CPU threads; fallback CLIPP2_CPU_THREADS, OMP_NUM_THREADS, then affinity/quota.")
    fit.add_argument("--run-manifest-id", help="External execution record identifier; no source scan is performed.")
    fit.add_argument("--version", action="version", version=__version__)
    workflow = subparsers.add_parser("workflow", allow_abbrev=False,
                                    help="Run Protocol 2.3 from an explicit v4 JSON plan.")
    workflow.add_argument("--plan", required=True)
    workflow.add_argument("--output-dir", required=True)
    assign = subparsers.add_parser("assign", allow_abbrev=False,
                                  help="Reassess frozen targets once under full original support.")
    assign.add_argument("--run-manifest-id", help="External identifier for this assignment execution.")
    assign.add_argument("--bundle", required=True)
    assign.add_argument("--output-dir", required=True)
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
                              cpu_threads=args.cpu_threads, audit_scope=args.audit_scope)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "assign":
        from .assignment import assign_mutations
        from .io.assignment import load_assignment_bundle
        assign_mutations(load_assignment_bundle(args.bundle), outdir=args.output_dir, run_manifest_id=args.run_manifest_id)
        return
    if args.command == "workflow":
        from .config import WorkflowConfig
        from .workflow import run_workflow
        plan_path = Path(args.plan).resolve()
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if (plan.get("schema") != "clipp2.workflow.plan.v4"
                or plan.get("protocol_version") != PROTOCOL_VERSION):
            raise ValueError("Workflow requires plan.v4 with protocol_version='2.3'; earlier contracts require a new run")
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
        run_manifest_id=args.run_manifest_id,
    )


__all__ = ["build_parser", "main", "parse_args"]


if __name__ == "__main__":
    main()
