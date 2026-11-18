"""One command for the complete Protocol 3.0 CPU workflow."""
import argparse
import json
from pathlib import Path

from .config import PLAN_SCHEMA, PROTOCOL_VERSION, WorkflowConfig
from ._version import __version__


def build_parser():
    parser = argparse.ArgumentParser(prog="clipp2", allow_abbrev=False,
        description="Discover, refine, and publish one clonal/single-copy-free mixture.")
    parser.add_argument("--version", action="version", version=__version__)
    workflow = parser.add_subparsers(dest="command", required=True).add_parser(
        "workflow", allow_abbrev=False, help="Run the Protocol 3.0 CPU workflow from a v8 plan.")
    workflow.add_argument("--plan", required=True)
    workflow.add_argument("--output-dir", required=True)
    return parser


def load_plan(path):
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA
            or plan.get("protocol_version") != PROTOCOL_VERSION):
        raise ValueError("Protocol 3.0 requires clipp2.workflow.plan.v8. Migrate v7/2.5 plans; "
                         "old fit records, assignment modes, read floors and cap settings are not replayable.")
    extra = set(plan)-{"schema", "protocol_version", "input_path", "workflow_config", "run_manifest_id"}
    if extra:
        raise ValueError(f"Unknown workflow plan fields: {sorted(extra)}")
    if not isinstance(plan.get("input_path"), str) or not plan["input_path"].strip():
        raise ValueError("A nonempty input_path is required.")
    return path.parent/plan["input_path"], WorkflowConfig.from_dict(plan.get("workflow_config", {})), plan.get("run_manifest_id")


def main(argv=None):
    args = build_parser().parse_args(argv)
    path, config, run_manifest_id = load_plan(Path(args.plan).resolve())
    from .workflow import run_workflow
    run_workflow(path, args.output_dir, config, run_manifest_id=run_manifest_id)


if __name__ == "__main__":
    main()
