"""Offline evaluation of preserved study evidence, with no inference execution."""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import zipfile

import numpy as np

from CliPP2._version import __version__
from CliPP2.io.data import content_hash
from CliPP2.io.tumor_txt import load_original_catalog
from CliPP2.provenance import validate_run_manifest


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _within(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Preserved evidence path is outside its study")
    return path


def preserved_jobs(root):
    """Read submitted jobs without reopening or regenerating an old cohort."""
    result = {}
    for path in sorted((Path(root) / "job_manifests").glob("batch-*.json")):
        for job in _read(path):
            previous = result.setdefault(job["cell_id"], job)
            if previous != job:
                raise ValueError("Preserved job manifests disagree on a cell's identity")
    return list(result.values())


def planned_roster(root, contract, allocation, cells):
    """Use a frozen roster; submitted legacy jobs are only a known subset."""
    path = Path(root) / "main_roster.json"
    if path.exists():
        record = _read(path)
        payload = {key: record.get(key) for key in ("schema", "families", "jobs")}
        jobs = record.get("jobs", [])
        if (record.get("schema") != "clipp2.simulation-roster.v1"
                or record.get("roster_id") != content_hash(payload)
                or record.get("families") != contract["families"]
                or len(jobs) != allocation.get("planned_main_datasets")
                or len({job["cell_id"] for job in jobs}) != len(jobs)
                or any(job["phase"] != "main" for job in jobs)):
            raise ValueError("Frozen main roster differs from its identity or allocation")
        lookup = {job["cell_id"]: job for job in jobs}
        for cell in cells:
            if cell["job"]["phase"] == "main" and lookup.get(cell["job"]["cell_id"]) != cell["job"]:
                raise ValueError("Completed main cell differs from the frozen roster")
        return dict(jobs=jobs, complete=True, status="frozen_roster", roster_id=record["roster_id"])
    known = {job["cell_id"]: job for job in preserved_jobs(root) if job["phase"] == "main"}
    for cell in cells:
        job = cell["job"]
        if job["phase"] == "main" and known.setdefault(job["cell_id"], job) != job:
            raise ValueError("Completion differs from retained submitted job")
    return dict(jobs=list(known.values()), complete=False, status="unavailable_complete_frozen_roster", roster_id=None)


def _configuration(attempt, index=1):
    """Read a retained paired result, including interrupted archive publication."""
    directory = attempt / f"configuration-{index}"
    payloads = []
    if (directory / "paired_result.json").is_file():
        payloads.append(_read(directory / "paired_result.json"))
    archive_path = directory.with_suffix(".zip")
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            payloads.append(json.loads(archive.read("paired_result.json")))
    if not payloads:
        raise FileNotFoundError("No preserved paired result for the selected attempt")
    if any(content_hash(value) != content_hash(payloads[0]) for value in payloads[1:]):
        raise ValueError("Directory and archived paired evidence disagree")
    return payloads[0]


def _load_evidence(root, cell):
    from .biology import History
    from .acquisition import validate_history_acquisition_link
    job = cell["job"]
    attempt = _within(root, cell["attempt"])
    if attempt.parent != root.resolve() / "cells" / job["cell_id"]:
        raise ValueError("Selected attempt belongs to a different cell")
    if _read(attempt / "job.json") != job:
        raise ValueError("Attempt job differs from the selected completion")
    directory = _within(root, "histories/" + job["history_id"])
    history_manifest = _read(directory / "history_manifest.json")
    if not isinstance(history_manifest, Mapping):
        raise TypeError("Preserved history manifest must be a JSON object")
    required = ("history_id", "family", "history_seed", "dimensions")
    if any(history_manifest.get(key) != job[key] for key in required):
        raise ValueError("Preserved physical history manifest differs from the selected job")
    if job.get("cohort_content_id") is not None and history_manifest.get("cohort_content_id") != job["cohort_content_id"]:
        raise ValueError("Preserved physical history belongs to a different cohort content")
    history = History.from_dict(_read(directory / "history.json"))
    if job.get("cohort_content_id") is not None and history.metadata.get("cohort_content_id") != job["cohort_content_id"]:
        raise ValueError("Preserved truth belongs to a different cohort content")
    if history.family != job["family"] or history.metadata.get("seed") != job["history_seed"]:
        raise ValueError("Preserved truth does not match the job's family and physical history seed")
    if job.get("cohort_settings") and history.history_id != job.get("source_history_id", job["history_id"]):
        raise ValueError("Preserved authoritative history has a different identity")
    dims = job["dimensions"]
    if (len(history.mutation_ids), len(history.region_ids), len(history.segment_keys)) != (
            dims["mutations"], dims["regions"], dims["segments"]):
        raise ValueError("Preserved history dimensions differ from the job")
    catalog = load_original_catalog(attempt / "input_catalog.tsv")
    if job.get("cohort_content_id") is not None and dict(catalog.metadata).get("cohort_content_id") != job["cohort_content_id"]:
        raise ValueError("Preserved input catalog belongs to a different cohort content")
    if catalog.tumor_id != job["cell_id"]:
        raise ValueError("Preserved input catalog belongs to another cell")
    acquisition = _read(attempt / "acquisition.json")
    if not isinstance(acquisition, Mapping):
        raise TypeError("Preserved acquisition must be a JSON object")
    if acquisition.get("id", acquisition.get("acquisition_id")) != job["acquisition"]["id"]:
        raise ValueError("Preserved acquisition differs from the selected job")
    if "history_id" in acquisition and acquisition["history_id"] != history.history_id:
        raise ValueError("Preserved acquisition belongs to another physical history")
    if "seed" in acquisition and acquisition["seed"] != job["acquisition_seed"]:
        raise ValueError("Preserved acquisition seed differs from the selected job")
    if job.get("cohort_content_id") is not None and acquisition.get("cohort_content_id") != job["cohort_content_id"]:
        raise ValueError("Preserved acquisition belongs to another cohort content")
    for key, attribute in (("alt_counts", "alt_counts"), ("total_counts", "total_counts"),
                           ("count_observed", "count_observed"), ("supplied_major_cn", "major_cn"),
                           ("supplied_minor_cn", "minor_cn"), ("supplied_purity", "purity"),
                           ("supplied_segment_ids", "segment_ids"), ("input_excluded", "input_excluded")):
        if key in acquisition and not np.array_equal(np.asarray(acquisition[key]), getattr(catalog, attribute)):
            raise ValueError("Preserved acquisition and input catalog disagree on " + key)
    paired = _configuration(attempt)
    if not isinstance(paired, Mapping):
        raise TypeError("Preserved paired inference must be a JSON object")
    if tuple(paired.get("original_mutation_ids", ())) != catalog.mutation_ids or tuple(paired.get("region_ids", ())) != catalog.region_ids:
        raise ValueError("Preserved paired inference has different original axes")
    manifest = paired.get("run_manifest")
    validate_run_manifest(manifest, tumor_id=catalog.tumor_id, biological_identity=catalog.biological_identity)
    if manifest["input_identity"].get("original_catalog_identity") != content_hash(catalog.to_dict()):
        raise ValueError("Preserved paired inference belongs to a different input catalog")
    truth_link = validate_history_acquisition_link(history, acquisition, catalog, history_manifest=history_manifest)
    return history, catalog, paired, acquisition, truth_link


def unavailable_evaluation(job, status, *, evaluation_config, error=None,
                           execution_state=None, evidence_state="missing_truth_evidence",
                           arm_applicability=None, original_evaluable_count=None):
    """Unavailable derived evidence receives no invented scientific decisions."""
    from .evaluation import evaluation_definition, normalize_evaluation_config
    definition = evaluation_definition(evaluation_config)
    arms = ({row["arm"]: row for row in arm_applicability} if isinstance(arm_applicability, (list, tuple))
            else dict(arm_applicability or {}))
    if original_evaluable_count is not None and (type(original_evaluable_count) is not int or original_evaluable_count < 0):
        raise ValueError("An established original evaluable count must be a nonnegative integer")
    metrics, applicability = [], []
    for arm in ("A", "B", "C", "B3", "C3"):
        state = arms.get(arm, {})
        arm_status = state.get("status", None if execution_state == "execution_failed" else execution_state)
        failed = arm_status in ("failed", "execution_failed")
        executed = None if arm_status is None else arm_status != "not_executed"
        scientifically_applicable = state.get("scientifically_applicable")
        available = state.get("available")
        if available is None and (failed or arm_status == "not_executed"):
            available = False
        for name, value in (("execution_available", executed), ("execution_failure", None if arm_status is None else failed),
                            ("analysis_available", available), ("analysis_failure", None if arm_status is None else failed),
                            ("evaluation_available", False), ("scientifically_applicable", scientifically_applicable)):
            metrics.append(dict(name=name, arm=arm, view="all_planned_cells", stratum="all",
                value=None if value is None else float(value), numerator=None if value is None else int(value),
                denominator=1, status="available" if value is not None else "unknown_execution_evidence",
                population_scope="planned_population", truth_required=False))
        zero = bool(state.get("no_decisions") and original_evaluable_count and (
            failed or state.get("scientific_rejection")))
        if arm != "A":
            for name in ("correct_per_original_evaluable", "binary_correct_per_original_evaluable"):
                metrics.append(dict(name=name, arm=arm, view="", stratum="all", value=0. if zero else None,
                    numerator=0 if zero else None, denominator=original_evaluable_count,
                    status="known_no_decisions" if zero else "unknown_scientific_evidence",
                    population_scope="planned_population", truth_required=not zero))
        applicability.append(dict(arm=arm, available=available, status=arm_status or status, planned=True,
                                  scientifically_applicable=scientifically_applicable))
    return dict(schema="clipp2.simulation.evaluation.v3", evaluation_definition=definition,
        evaluation_config=normalize_evaluation_config(evaluation_config), truth_qualification=evidence_state,
        evaluation_definition_id=content_hash(definition), cell_id=job["cell_id"], history_id=job["history_id"],
        family=job["family"], acquisition_id=job["acquisition"]["id"], acquisition=job["acquisition"],
        phase=job["phase"], job=job, status=status, execution_state=execution_state,
        original_evaluable_count=original_evaluable_count, error=error, metrics=metrics, mutations=[],
        calibration=[], cap_audit=[], stage_three_containment=[], information=[],
        applicability=applicability)


def placeholder_for_cell(root, cell, evaluation_config, *, evidence_state="missing_truth_evidence", error=None):
    """Recover operational evidence without assigning unknown scientific outcomes."""
    root, job = Path(root), cell["job"]
    count = cell.get("original_evaluable_count")
    arms = {}
    # A worker exception may follow successful inference. Only explicit phases
    # preceding accepted decisions establish arm-specific no-decision credit.
    for failure in cell.get("execution_failures", ()):
        arm, stage = failure.get("arm"), failure.get("failed_stage")
        if arm is None and (stage == "stage_one" or
                stage == "worker" and cell.get("execution_phase") == "before_inference"):
            affected = ("A", "B", "C", "B3", "C3")
        elif arm in ("B", "C") and stage == "stage_two":
            affected = (arm, arm+"3")
        elif arm in ("B", "C") and stage == "assignment":
            affected = (arm+"3",)
        else:
            affected = ()
        arms.update({name: dict(status="failed", available=False, no_decisions=True) for name in affected})
    try:
        attempt = _within(root, cell["attempt"])
        if _read(attempt / "job.json") != job:
            raise ValueError("Selected attempt job differs")
        catalog = load_original_catalog(attempt / "input_catalog.tsv")
        if catalog.tumor_id != job["cell_id"]:
            raise ValueError("Original population belongs to another cell")
        acquisition = _read(attempt / "acquisition.json")
        if not isinstance(acquisition, Mapping):
            raise TypeError("Preserved acquisition must be a JSON object")
        known = np.asarray(acquisition.get("biological_truth_available"), bool)
        if known.shape == catalog.alt_counts.shape:
            count = int((known.all(axis=1) & (catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)).any(axis=1)).sum())
        paired = _configuration(attempt)
        if not isinstance(paired, Mapping):
            raise TypeError("Preserved paired inference must be a JSON object")
        validate_run_manifest(paired.get("run_manifest"), tumor_id=catalog.tumor_id,
                              biological_identity=catalog.biological_identity)
        if paired["run_manifest"]["input_identity"].get("original_catalog_identity") != content_hash(catalog.to_dict()):
            raise ValueError("Paired evidence belongs to a different catalog")
        root_failed = paired.get("status") == "failed" and paired.get("stage_one") is None
        rejection = paired.get("status") in ("no_eligible_clonal_reference", "blocked")
        arms["A"] = dict(status="failed" if root_failed else "complete", available=paired.get("stage_one") is not None,
                         scientifically_applicable=True, no_decisions=root_failed)
        for name in ("B", "C"):
            arm = paired.get("arms", {}).get(name, {})
            state = "failed" if root_failed else arm.get("status", paired.get("status"))
            for label, decisions in ((name, arm.get("combined_labels")), (name+"3", arm.get("assignment"))):
                arms[label] = dict(status=state, available=bool(decisions) and state in ("complete", "empty_remainder"),
                    no_decisions=not decisions, scientific_rejection=rejection or state in ("blocked", "no_eligible_clonal_reference"),
                    scientifically_applicable=False if rejection else None)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile):
        pass  # A cell-level exception alone does not prove inference failed before decisions.
    return unavailable_evaluation(job, "unavailable_derived_evidence", evaluation_config=evaluation_config,
        execution_state=cell.get("status"), evidence_state=evidence_state, arm_applicability=arms,
        original_evaluable_count=count, error=error)


def reevaluate_cell(root, cell, evaluation_config):
    """Derive a new record solely from the selected attempt's saved evidence."""
    from .evaluation import evaluate
    root = Path(root).resolve()
    job = cell["job"]
    truth_link = None
    try:
        history, catalog, paired, acquisition, truth_link = _load_evidence(root, cell)
        qualification = truth_link["status"]
        if qualification in ("conflicting_truth", "missing_truth_evidence"):
            value = placeholder_for_cell(root, cell, evaluation_config, evidence_state=qualification,
                error=dict(type="TruthEvidenceError", message=truth_link["reason"]))
        else:
            value = evaluate(history, catalog, paired, acquisition=acquisition, evaluation_config=evaluation_config,
                             truth_qualification=qualification)
            value.update(cell_id=job["cell_id"], history_id=job["history_id"], family=job["family"],
                acquisition_id=job["acquisition"]["id"], acquisition=job["acquisition"], phase=job["phase"], job=job,
                status="derived_from_preserved_evidence")
    except (OSError, ValueError, KeyError, TypeError, AssertionError, zipfile.BadZipFile) as error:
        qualification = ("conflicting_truth" if isinstance(error, (ValueError, AssertionError))
                         else "missing_truth_evidence")
        value = placeholder_for_cell(root, cell, evaluation_config, evidence_state=qualification,
                                     error=dict(type=type(error).__name__, message=str(error)))
    value["truth_qualification"] = qualification
    if truth_link is not None:
        value["truth_link"] = truth_link
    value["derivation"] = dict(implementation_version=__version__, source_study=str(root),
        selected_attempt=cell.get("attempt"), source_implementation_version=cell.get("implementation_version"),
        source_protocol_version=cell.get("protocol_version"), inference_decisions_preserved=True,
        fit_count=0, assignment_pass_count=0, threshold_selection_count=0)
    return value
