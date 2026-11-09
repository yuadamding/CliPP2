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
        configurations = {path.stem if path.suffix == ".zip" else path.name
                          for path in attempt.glob("configuration-*")
                          if path.is_dir() or path.suffix == ".zip"}
        # A process may stop after linking the verified archive but before
        # removing its source directory; that is one operation, not two.
        for name in configurations:
            try:
                lines = (attempt / name / "execution.jsonl").read_text().splitlines()
            except OSError:
                # Directory cleanup can remove the journal before removing its
                # parent. The complete archive remains authoritative evidence.
                try:
                    with zipfile.ZipFile(attempt / (name + ".zip")) as archive:
                        lines = archive.read("execution.jsonl").decode().splitlines()
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


def qualification_costs(root):
    """Keep interrupted control costs unknown rather than silently counting zero."""
    attempts, known = [], Counter()
    for attempt in sorted((Path(root) / "qualification_steps").glob("*/attempt-*")):
        completion = attempt / "completion.json"
        if completion.exists():
            receipt = json.loads(completion.read_text())
            ledger = receipt.get("cost_ledger")
            state = receipt["status"]
        else:
            ledger, state = None, "interrupted_or_running"
        if ledger is not None:
            for key in ("fit_count", "assignment_pass_count"):
                known[key] += ledger.get(key, 0)
        attempts.append(dict(step=attempt.parent.name, attempt=attempt.name, status=state,
                             cost_status="known" if ledger is not None else "unknown"))
    return dict(attempts=attempts, known_costs=dict(known),
                unknown_cost_attempts=sum(row["cost_status"] == "unknown" for row in attempts),
                total_cost_status="unknown" if any(row["cost_status"] == "unknown" for row in attempts) else "known")


def status(outdir):
    from .study import _latest_completion, read_json
    root = Path(outdir).resolve()
    contract = read_json(root / "study_contract.json")
    from .design import roster_counts
    allocation = (roster_counts(contract) if "replicate_roster" in contract else
                  read_json(root / "allocation.json") if (root / "allocation.json").exists() else {})
    development = read_json(root / "threshold_development.json") if (root / "threshold_development.json").exists() else None
    assessment_paths = list(root.glob("development_assessment*.json"))
    cells = [value for directory in (root / "cells").glob("*") if directory.is_dir()
             for value in [_latest_completion(directory)] if value is not None]
    from .reporting import planned_roster
    roster = planned_roster(root, contract, allocation, cells)
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
        population_policy="explicit_replicates" if "replicate_roster" in contract else contract.get("profile"),
        workers=contract["resource_limits"]["workers"],
        planned_development_cells=allocation.get("planned_development_datasets"),
        planned_main_cells=len(roster["jobs"]) if roster["complete"] else allocation.get("planned_main_datasets"),
        conditional_development_extension_cells=allocation.get("conditional_development_extension_datasets", 0),
        processed_cells=[dict(phase=phase, status=state, count=count) for (phase, state), count in sorted(counts.items())],
        operation_costs=costs,
        qualification_costs=qualification_costs(root),
        recorded_execution_failures=sum(len(cell.get("execution_failures", [])) for cell in cells),
        active=active, development_assessed=bool(assessment_paths),
        historical_thresholds_frozen=development is not None and development.get("selected") is not None,
        report=str(reports[-1]) if reports else None,
        cost_note="Durable operation journals include completed, failed and unfinished attempts; scientific applicability is in the report")
