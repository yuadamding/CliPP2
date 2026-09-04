"""End-to-end release contract for the compact CLI and four-file output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys


SCHEMA_COLUMNS = (
    "mutation_id",
    "sample_id",
    "alt_count",
    "ref_count",
    "count_observed",
    "purity",
    "normal_cn",
    "segment_id",
    "cn_state_id",
    "cn_state_fraction",
    "allele_a_cn",
    "allele_b_cn",
)


def _write_smoke_input(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA_COLUMNS, delimiter="\t")
        writer.writeheader()
        for index, alt_count in enumerate((10, 11, 30, 31), start=1):
            writer.writerow(
                {
                    "mutation_id": f"m{index}",
                    "sample_id": "r1",
                    "alt_count": alt_count,
                    "ref_count": 100 - alt_count,
                    "count_observed": 1,
                    "purity": 0.6,
                    "normal_cn": 2,
                    "segment_id": f"s{index}",
                    "cn_state_id": "state1",
                    "cn_state_fraction": 1.0,
                    "allele_a_cn": 1,
                    "allele_b_cn": 1,
                }
            )


def _run_fit(
    input_file: Path,
    outdir: Path,
    config_file: Path,
    *options: str,
    failure_policy: str = "error",
) -> dict[str, object]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "CliPP2",
            "fit",
            "--input-file",
            str(input_file),
            "--outdir",
            str(outdir),
            "--profile",
            "fast",
            "--device",
            "cpu",
            "--failure-policy",
            failure_policy,
            "--config",
            str(config_file),
            *options,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads((outdir / "smoke_analysis.json").read_text())


def test_cli_checkpoint_resume_and_four_file_output(tmp_path: Path) -> None:
    input_file = tmp_path / "smoke.tsv"
    config_file = tmp_path / "config.json"
    checkpoint = tmp_path / "checkpoint"
    _write_smoke_input(input_file)
    config_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fit": {
                    "dtype": "float64",
                    "max_direct_partition_candidates": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    first_dir = tmp_path / "first"
    resumed_dir = tmp_path / "resumed"
    first = _run_fit(
        input_file, first_dir, config_file, "--checkpoint", str(checkpoint)
    )
    resumed = _run_fit(
        input_file, resumed_dir, config_file, "--resume", str(checkpoint)
    )

    assert first["summary_schema_version"] == 11
    assert first["analysis_tier"] == "joint_certified"
    assert first["primary_estimator_available"] is True
    assert first["selection_score_name"] == "fixed_partition_bic"
    assert first["selection_policy_id"] == "hybrid-ward-cem-bic-v1"
    assert first["computation_profile"] == "fast"
    assert first["device"] == "cpu"
    assert first["dtype"] == "float64"
    assert resumed["resumed_from_checkpoint"] is True
    for field in (
        "selected_partition_signature",
        "selected_n_clusters",
        "selection_score",
        "search_work_edge_pass_equivalents",
        "selection_pool_stop_reason",
    ):
        assert resumed[field] == first[field]

    assert (checkpoint / "manifest.json").is_file()
    assert any((checkpoint / "arrays").iterdir())
    assert {path.name for path in first_dir.iterdir()} == {
        "smoke_analysis.json",
        "smoke_attempts.tsv",
        "smoke_clusters.tsv",
        "smoke_mutations.tsv",
    }


def test_resource_stop_remains_explicitly_unresolved(tmp_path: Path) -> None:
    input_file = tmp_path / "smoke.tsv"
    config_file = tmp_path / "config.json"
    _write_smoke_input(input_file)
    config_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fit": {
                    "dtype": "float64",
                    "max_tumor_edge_pass_equivalents": 1,
                    "max_direct_partition_candidates": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    analysis = _run_fit(
        input_file,
        tmp_path / "resource-stop",
        config_file,
        failure_policy="save-diagnostics",
    )
    assert analysis["search_stop_reason"] == "tumor_work_budget_reached"
    assert analysis["selection_optimum_resolved"] is False
    assert analysis["global_hybrid_optimum_certified"] is False
