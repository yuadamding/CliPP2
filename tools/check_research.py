#!/usr/bin/env python3
"""Deterministic research regressions against an installed CliPP2 package.

Small native fixtures are included; no full cohort or calibration study is run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile


GROUPS = ("workflow", "biology", "evaluation", "study", "publication")


def run_group(group, root, cohort_settings):
    if group == "workflow":
        from check_workflow import run_checks
        return run_checks(root)
    if group == "biology":
        from check_simulation_biology import run_biology_checks, run_cohort_identity_checks
        result = run_biology_checks(root)
        result["cohort_identity"] = run_cohort_identity_checks(root / "identity", cohort_settings)
        return result
    if group == "evaluation":
        from check_simulation_evaluation import run_checks
        return run_checks()
    if group == "study":
        from check_simulation_study import run_checks
        return run_checks(root, cohort_settings)
    from check_publication import run_checks
    return run_checks(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=list(GROUPS))
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--cohort-settings", type=Path, default=Path("CliPP2_phylogeny_cohort_settings.json"))
    args = parser.parse_args(argv)
    if len(set(args.groups)) != len(args.groups):
        parser.error("Each check group may be requested once")
    root = args.outdir or Path(tempfile.mkdtemp(prefix="clipp2-research-checks-"))
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    for group in args.groups:
        results[group] = run_group(group, root / group, args.cohort_settings)
        print(f"Passed {group}", flush=True)
    path = root / "checks.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(status="passed", groups=results), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Results: {path}")


if __name__ == "__main__":
    main()
