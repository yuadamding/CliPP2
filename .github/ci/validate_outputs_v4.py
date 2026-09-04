#!/usr/bin/env python3
"""Fail-closed, analysis-tier-aware validation for one CliPP2 v0.4 result."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


SUFFIXES = ("analysis.json", "clusters.tsv", "mutations.tsv", "attempts.tsv")


def _count_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise RuntimeError(f"missing TSV header: {path}")
        return sum(1 for _ in reader)


def _require_hex64(analysis: dict[str, object], name: str) -> None:
    value = analysis.get(name)
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"analysis identity is missing for {name}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError(f"analysis identity is not hexadecimal for {name}") from exc


def _require_empty(analysis: dict[str, object], name: str) -> None:
    if analysis.get(name) != "":
        raise RuntimeError(f"conditional analysis fabricates {name}")


def validate_outputs(
    *,
    outdir: Path,
    tumor_id: str,
    expected_mutations: int,
) -> dict[str, object]:
    """Validate four artifacts and return distinct operational/scientific status."""

    expected = {f"{tumor_id}_{suffix}" for suffix in SUFFIXES}
    actual = {
        path.name
        for path in outdir.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual != expected:
        raise RuntimeError(f"unexpected v0.4 result inventory: {sorted(actual)}")
    paths = {suffix: outdir / f"{tumor_id}_{suffix}" for suffix in SUFFIXES}
    if any(
        not path.is_file() or path.is_symlink() or path.stat().st_size == 0
        for path in paths.values()
    ):
        raise RuntimeError("a required v0.4 result is missing, linked, or empty")

    with paths["analysis.json"].open(encoding="utf-8") as handle:
        analysis = json.load(handle)
    required_equal = {
        "summary_schema_version": 11,
        "output_schema_version": 2,
        "tumor_id": tumor_id,
        "computation_profile": "balanced",
        "selection_policy_id": "hybrid-ward-cem-bic-v1",
    }
    for key, value in required_equal.items():
        if analysis.get(key) != value:
            raise RuntimeError(
                f"analysis contract mismatch for {key}: {analysis.get(key)!r}"
            )
    if set(analysis.get("output_files", [])) != expected:
        raise RuntimeError("analysis output-file authority mismatch")
    if (
        not isinstance(analysis.get("selected_n_clusters"), int)
        or int(analysis["selected_n_clusters"]) < 1
        or not analysis.get("selected_partition_signature")
    ):
        raise RuntimeError("selected partition identity is missing")

    primary = analysis.get("primary_estimator_available")
    if not isinstance(primary, bool):
        raise RuntimeError("primary_estimator_available must be Boolean")
    selected_hashes = (
        "selected_raw_reference_objective_spec_hash",
        "selected_raw_reference_original_graph_hash",
    )
    if primary:
        if analysis.get("analysis_tier") != "joint_certified":
            raise RuntimeError("primary result is not joint_certified")
        if analysis.get("raw_reference_objective_certified") is not True:
            raise RuntimeError("primary result lacks a certified raw reference")
        if analysis.get("selected_refit_numerically_resolved") is not True:
            raise RuntimeError("primary result lacks a resolved partition refit")
        for name in selected_hashes:
            _require_hex64(analysis, name)
        scientific_status = "primary_estimator_available"
    else:
        if analysis.get("analysis_tier") != "conditional_partition_refit":
            raise RuntimeError("non-primary panel result is not a conditional refit")
        for name in selected_hashes:
            _require_empty(analysis, name)
        if int(analysis.get("num_raw_candidates", 0)) > 0:
            _require_hex64(analysis, "attempted_objective_spec_hash")
            _require_hex64(analysis, "attempted_original_graph_hash")
        reason = str(analysis.get("failure_reason", ""))
        if not reason.startswith("NoCertifiedRawReferenceError:"):
            raise RuntimeError("conditional result lacks raw-reference failure provenance")
        scientific_status = "no_certified_raw_reference"

    cluster_rows = _count_rows(paths["clusters.tsv"])
    mutation_rows = _count_rows(paths["mutations.tsv"])
    attempt_rows = _count_rows(paths["attempts.tsv"])
    if cluster_rows < 1 or mutation_rows != int(expected_mutations):
        raise RuntimeError(
            f"unexpected table rows: clusters={cluster_rows}, mutations={mutation_rows}, "
            f"attempts={attempt_rows}"
        )
    if int(analysis.get("num_raw_solver_attempts", 0)) > 0 and attempt_rows < 1:
        raise RuntimeError("raw solver attempts are absent from attempts.tsv")

    return {
        "tumor_id": tumor_id,
        "execution_status": "completed",
        "artifact_status": "valid",
        "scientific_status": scientific_status,
        "analysis_tier": analysis["analysis_tier"],
        "selection_status": analysis.get("selection_status"),
        "selected_n_clusters": analysis.get("selected_n_clusters"),
        "raw_reference_objective_certified": analysis.get(
            "raw_reference_objective_certified"
        ),
        "mutation_rows": mutation_rows,
        "cluster_rows": cluster_rows,
        "attempt_rows": attempt_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--tumor-id", required=True)
    parser.add_argument("--expected-mutations", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate_outputs(
        outdir=args.outdir,
        tumor_id=args.tumor_id,
        expected_mutations=args.expected_mutations,
    )
    descriptor = os.open(
        args.output,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o440,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
