"""Standard-library writers for a prepared analysis view."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .view import AnalysisView, TableView


_MODERN_SUFFIXES = ("analysis.json", "clusters.tsv", "mutations.tsv", "attempts.tsv")
_RETIRED_SUFFIXES = (
    "analysis_status.json",
    "region_status.tsv",
    "raw_attempts.tsv",
    "cluster_region_estimates.tsv",
    "mutation_region_estimates.tsv",
    "mutation_clusters.tsv",
    "cluster_centers.tsv",
    "mutation_region_multiplicity.tsv",
    "secondary_cluster_centers.tsv",
    "secondary_mutation_region_estimates.tsv",
    "run_summary.tsv",
    "lambda_search.tsv",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(entry) for entry in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(float(value)) else None
    return value


def _tsv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return ""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_table(path: Path, table: TableView) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(table.columns),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(
            {column: _tsv_value(row.get(column)) for column in table.columns}
            for row in table.rows
        )


def write_analysis_outputs(
    analysis: AnalysisView,
    *,
    outdir: Path,
    summary: dict[str, object],
) -> None:
    """Write exactly four modern files without changing analysis state."""

    destination = Path(outdir)
    destination.mkdir(parents=True, exist_ok=True)
    tumor_id = str(analysis.data.tumor_id)
    for suffix in _RETIRED_SUFFIXES:
        retired = destination / f"{tumor_id}_{suffix}"
        if retired.is_file() or retired.is_symlink():
            retired.unlink()
    payload = dict(summary)
    payload["output_files"] = [f"{tumor_id}_{suffix}" for suffix in _MODERN_SUFFIXES]
    with (destination / f"{tumor_id}_analysis.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(
            _json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False
        )
        handle.write("\n")
    _write_table(destination / f"{tumor_id}_clusters.tsv", analysis.clusters)
    _write_table(destination / f"{tumor_id}_mutations.tsv", analysis.mutations)
    _write_table(destination / f"{tumor_id}_attempts.tsv", analysis.attempts)


__all__ = ["write_analysis_outputs"]
