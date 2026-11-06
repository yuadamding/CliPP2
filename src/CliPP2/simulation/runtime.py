"""Read-only progress summaries for long-running, resumable simulation cohorts."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
import zipfile


def recorded_costs(root):
    """Count started work once, including attempts interrupted before completion."""
    counts = Counter()
    root = Path(root)
    for attempt in (root / "cells").glob("*/attempt-*"):
        completed_attempts = {json.loads(path.read_text())["attempt"] for path in attempt.parent.glob("completion*.json")}
        if str(attempt.relative_to(root)) not in completed_attempts:
            counts["unfinished_attempts"] += 1
        configurations = {}
        for path in attempt.glob("configuration-*"):
            if path.is_dir() or path.suffix == ".zip":
                key = path.stem if path.suffix == ".zip" else path.name
                if path.is_dir() or key not in configurations:
                    configurations[key] = path
        # A process may stop after linking the verified archive but before
        # removing its source directory; that is one operation, not two.
        for config in configurations.values():
            try:
                if config.is_dir():
                    journal = config / "execution.jsonl"
                    if not journal.exists():
                        continue
                    lines = journal.read_text().splitlines()
                elif config.suffix == ".zip":
                    with zipfile.ZipFile(config) as archive:
                        lines = archive.read("execution.jsonl").decode().splitlines()
                else:
                    continue
            except (OSError, KeyError, zipfile.BadZipFile):
                continue
            latest = {}
            for line in lines:
                try:
                    event = json.loads(line)
                    latest[event["operation_index"]] = event
                except (ValueError, KeyError):
                    continue
            for event in latest.values():
                kind = event["operation"]
                if kind not in ("fit", "assignment", "reuse"):
                    continue
                counts[kind+"_attempts"] += 1
                counts[kind+"_"+event["status"]] += 1
    return dict(counts)


def status(outdir):
    from .study import _latest_completion, read_json
    root = Path(outdir).resolve()
    contract = read_json(root / "study_contract.json")
    allocation = read_json(root / "allocation.json") if (root / "allocation.json").exists() else {}
    cells = [value for directory in (root / "cells").glob("*") if directory.is_dir()
             for value in [_latest_completion(directory)] if value is not None]
    counts = Counter((cell["job"]["phase"], cell["status"]) for cell in cells)
    active = []
    for path in (root / "cells").glob("*/attempt-*/configuration-*/execution.jsonl"):
        directory = path.parent.parent.parent
        completed = _latest_completion(directory)
        if completed is not None and Path(completed["attempt"]).name >= path.parent.parent.name:
            continue
        try:
            lines = path.read_text().splitlines()
            last = json.loads(lines[-1]) if lines else None
        except (OSError, ValueError):
            continue  # Concurrent partial append: try again on the next status call.
        if last:
            progress_path = path.parent.parent / "progress.json"
            try:
                progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
                age = time.time()-(progress_path.stat().st_mtime if progress else path.stat().st_mtime)
            except (OSError, ValueError):
                progress, age = None, None
            active.append(dict(cell_id=directory.name, configuration=path.parent.name, **last,
                               progress=progress, seconds_since_update=age))
    reports = sorted((root / "reports").glob("report-*/comparison_report.html"))
    costs = recorded_costs(root)
    return dict(study_directory=str(root), study_version=contract["study_version"],
        protocol_version=contract.get("protocol_version"),
        implementation_version=contract.get("implementation_version"), run_manifest_id=contract.get("run_manifest_id"),
        source_cohort_protocol_version=contract.get("source_cohort_protocol_version"),
        profile=contract["profile"], workers=contract["resource_limits"]["workers"],
        planned_development_cells=allocation.get("planned_development_datasets"),
        planned_main_cells=allocation.get("planned_main_datasets"),
        processed_cells=[dict(phase=phase, status=state, count=count) for (phase, state), count in sorted(counts.items())],
        operation_costs=costs,
        active=active, thresholds_frozen=(root / "study_plan.json").exists(),
        report=str(reports[-1]) if reports else None,
        cost_note="Durable operation journals include completed, failed and unfinished attempts; scientific applicability is in the report")
