"""One command for Protocol 3.0 CPU discovery and publication."""
import argparse
from pathlib import Path

from ._version import __version__
from .config import DEFAULT_MAX_CLUSTERS, DEFAULT_MISSING_COUNTS, DEFAULT_PROPOSAL_BUDGET, FitConfig, WorkflowConfig


def build_parser():
    parser = argparse.ArgumentParser(prog="clipp2", allow_abbrev=False,
        description="Fit and publish one clonal/single-copy-free mixture.")
    parser.add_argument("--version", action="version", version=__version__)
    fit = parser.add_subparsers(dest="command", required=True).add_parser(
        "fit", allow_abbrev=False, help="Run one Stage 1 CPU fit.")
    inputs = fit.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-dir", type=Path, help="Tumor directory with regional input files.")
    inputs.add_argument("--input-file", type=Path, help="Canonical tumor TSV, optionally gzip-compressed.")
    fit.add_argument("--device", choices=("cpu",), default="cpu", help="Execution device (CPU only).")
    fit.add_argument("--output-dir", type=Path, required=True, help="New directory for results.")
    fit.add_argument("--max-clusters", type=int, default=DEFAULT_MAX_CLUSTERS,
                     help=f"Maximum biological clusters (default: {DEFAULT_MAX_CLUSTERS}).")
    fit.add_argument("--max-major-cn", type=int, help="Optional major-CN cutoff.")
    fit.add_argument("--cpu-threads", type=int, help="Optional BLAS thread limit.")
    fit.add_argument("--proposal-budget", type=int, default=DEFAULT_PROPOSAL_BUDGET,
                     help=f"Optional search proposals (default: {DEFAULT_PROPOSAL_BUDGET}).")
    fit.add_argument("--normal-cn-policy", choices=("diploid",),
                     help="Assume diploid normal CN where it was not supplied.")
    fit.add_argument("--missing-counts", choices=("zero_alt", "unobserved"), default=DEFAULT_MISSING_COUNTS,
                     help=f"Missing read-count policy (default: {DEFAULT_MISSING_COUNTS}).")
    fit.add_argument("--run-manifest-id", help="Optional provenance identifier.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input_dir is not None and not args.input_dir.is_dir():
        parser.error(f"--input-dir must name an existing directory: {args.input_dir}")
    if args.input_file is not None and not args.input_file.is_file():
        parser.error(f"--input-file must name an existing file: {args.input_file}")
    try:
        config = WorkflowConfig(
            fit=FitConfig(device=args.device, max_clusters=args.max_clusters, max_major_cn=args.max_major_cn,
                          cpu_threads=args.cpu_threads, proposal_budget=args.proposal_budget),
            normal_cn_policy=args.normal_cn_policy, missing_counts=args.missing_counts,
        )
    except ValueError as error:
        parser.error(str(error))
    from .workflow import run_workflow
    run_workflow(args.input_dir or args.input_file, args.output_dir, config, run_manifest_id=args.run_manifest_id)


if __name__ == "__main__":
    main()
