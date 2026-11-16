"""One command for the complete A-initialized B workflow and unchanged Stage 3."""
import argparse
import json
from pathlib import Path

from .config import PROTOCOL_VERSION, WorkflowConfig
from ._version import __version__


def build_parser():
    parser = argparse.ArgumentParser(prog="clipp2", allow_abbrev=False,
        description="Fresh A initialization → B Stage 1 → new caps → fresh A → B Stage 2 → Stage 3.")
    parser.add_argument("--version", action="version", version=__version__)
    workflow = parser.add_subparsers(dest="command", required=True).add_parser(
        "workflow", allow_abbrev=False, help="Run the complete CPU/marginal workflow from a v7 plan.")
    workflow.add_argument("--plan", required=True)
    workflow.add_argument("--output-dir", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (not isinstance(plan, dict) or plan.get("schema") != "clipp2.workflow.plan.v7"
            or plan.get("protocol_version") != PROTOCOL_VERSION):
        raise ValueError("Workflow requires plan.v7 with protocol_version='2.5'")
    extra = set(plan) - {"schema", "protocol_version", "input_path", "workflow_config", "run_manifest_id"}
    if extra:
        raise ValueError(f"Unknown workflow plan fields: {sorted(extra)}")
    input_path = Path(plan["input_path"])
    from .workflow import run_workflow
    receipt = run_workflow(input_path if input_path.is_absolute() else plan_path.parent / input_path,
        args.output_dir, WorkflowConfig.from_dict(plan["workflow_config"]),
        run_manifest_id=plan.get("run_manifest_id"))
    if receipt["status"] == "blocked":
        raise SystemExit("Workflow blocked: " + str(receipt.get("unsupported_regions", [])))


if __name__ == "__main__":
    main()
