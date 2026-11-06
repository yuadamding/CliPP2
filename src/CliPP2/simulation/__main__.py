"""Run with clipp2-simulate or python -m CliPP2.simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .design import make_contract
from .study import execute, read_json, report


def main(argv=None):
    parser = argparse.ArgumentParser(description="CliPP2 physical-copy biological simulation study")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Qualify, time, develop thresholds, freeze and execute a balanced cohort")
    run.add_argument("--outdir", type=Path, required=True)
    run.add_argument("--resume", action="store_true", help="Use the immutable contract already in outdir")
    run.add_argument("--retry-failed", action="store_true", help="Retry worker failures as new retained attempts; frozen development settings remain frozen")
    run.add_argument("--profile", choices=("balanced", "full"), default="balanced")
    run.add_argument("--workers", type=int, default=10)
    run.add_argument("--budget-minutes", type=float, default=60.)
    run.add_argument("--disk-budget-gib", type=float, default=8.)
    run.add_argument("--cohort-settings", type=Path, default=Path("CliPP2_phylogeny_cohort_settings.json"))
    run.add_argument("--source-record", type=Path, help="Recorded source identity for an isolated installed build")
    run.add_argument("--run-manifest-id", help="External execution record link; defaults to CLIPP2_RUN_MANIFEST_ID")
    run.add_argument("--main-replicates", type=int, help="Replicates per each of the60 subcases; full design is10")
    run.add_argument("--development-replicates", type=int, help="Replicates per subcase; full design is2")
    qualify = sub.add_parser("qualify", help="Validate the complete supplied cohort and run bounded native pilots")
    qualify.add_argument("--cohort-settings", type=Path, default=Path("CliPP2_phylogeny_cohort_settings.json"))
    qualify.add_argument("--outdir", type=Path, required=True)
    qualify.add_argument("--workers", type=int, default=3)
    qualify.add_argument("--native-timeout", type=float, default=180.)
    qualify.add_argument("--skip-native", action="store_true")
    qualify.add_argument("--reference-match-atol", type=float, default=None,
        help="Qualification only: optional per-region center tolerance for one-to-one reference recall; default unassessed")
    materialize = sub.add_parser("materialize", help="Generate all physical truth and prescribed input catalogs without fitting")
    materialize.add_argument("--cohort-settings", type=Path, default=Path("CliPP2_phylogeny_cohort_settings.json"))
    materialize.add_argument("--outdir", type=Path, required=True)
    render = sub.add_parser("report", help="Rebuild reports from immutable completed cells")
    render.add_argument("outdir", type=Path)
    progress = sub.add_parser("status", help="Inspect completed cells and in-flight native operations")
    progress.add_argument("outdir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "status":
        from .runtime import status
        result = status(args.outdir)
    elif args.command == "qualify":
        from .qualification import qualify_cohort
        result = qualify_cohort(args.cohort_settings, args.outdir, native_pilot=not args.skip_native,
                                timeout_seconds=args.native_timeout, workers=args.workers,
                                reference_match_atol=args.reference_match_atol)
    elif args.command == "materialize":
        from .qualification import materialize_cohort
        result = materialize_cohort(args.cohort_settings, args.outdir)
    elif args.command == "report":
        result = report(args.outdir)
    else:
        contract = (read_json(args.outdir / "study_contract.json") if args.resume else
            make_contract(workers=args.workers, budget_minutes=args.budget_minutes,
                disk_budget_gib=args.disk_budget_gib, profile=args.profile,
                main_histories=args.main_replicates, development_histories=args.development_replicates,
                cohort_settings=args.cohort_settings, run_manifest_id=args.run_manifest_id))
        if args.source_record and not args.resume:
            contract["source_identity"] = read_json(args.source_record)
        result = execute(args.outdir, contract, resume=args.resume, retry_failed=args.retry_failed)
    if args.command in ("qualify", "materialize"):
        output = {key: result[key] for key in ("status", "validation", "counts", "native_pilot_counts") if key in result}
        output["output_dir"] = str(args.outdir.resolve())
    else:
        output = {key: value for key, value in result.items() if key != "summary"}
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
