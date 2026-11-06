"""Offline evaluation of preserved study evidence, with no inference execution."""
from __future__ import annotations

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
    job = cell["job"]
    attempt = _within(root, cell["attempt"])
    if attempt.parent != root.resolve() / "cells" / job["cell_id"]:
        raise ValueError("Selected attempt belongs to a different cell")
    if _read(attempt / "job.json") != job:
        raise ValueError("Attempt job differs from the selected completion")
    directory = _within(root, "histories/" + job["history_id"])
    manifest = _read(directory / "history_manifest.json")
    required = ("history_id", "family", "history_seed", "dimensions")
    if any(manifest.get(key) != job[key] for key in required):
        raise ValueError("Preserved physical history manifest differs from the selected job")
    if job.get("cohort_content_id") is not None and manifest.get("cohort_content_id") != job["cohort_content_id"]:
        raise ValueError("Preserved physical history belongs to a different cohort content")
    history = History.from_dict(_read(directory / "history.json"))
    if history.family != job["family"] or history.metadata.get("seed") != job["history_seed"]:
        raise ValueError("Preserved truth does not match the job's family and physical history seed")
    if job.get("cohort_settings") and history.history_id != job.get("source_history_id", job["history_id"]):
        raise ValueError("Preserved authoritative history has a different identity")
    dims = job["dimensions"]
    if (len(history.mutation_ids), len(history.region_ids), len(history.segment_keys)) != (
            dims["mutations"], dims["regions"], dims["segments"]):
        raise ValueError("Preserved history dimensions differ from the job")
    catalog = load_original_catalog(attempt / "input_catalog.tsv")
    if catalog.tumor_id != job["cell_id"]:
        raise ValueError("Preserved input catalog belongs to another cell")
    acquisition = _read(attempt / "acquisition.json")
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
    if tuple(paired.get("original_mutation_ids", ())) != catalog.mutation_ids or tuple(paired.get("region_ids", ())) != catalog.region_ids:
        raise ValueError("Preserved paired inference has different original axes")
    manifest = paired.get("run_manifest")
    validate_run_manifest(manifest, tumor_id=catalog.tumor_id, biological_identity=catalog.biological_identity)
    if manifest["input_identity"].get("original_catalog_identity") != content_hash(catalog.to_dict()):
        raise ValueError("Preserved paired inference belongs to a different input catalog")
    return history, catalog, paired, acquisition


def unavailable_evaluation(job, status, *, evaluation_config, error=None):
    """Unavailable derived evidence receives no invented scientific decisions."""
    from .evaluation import evaluation_definition
    definition = evaluation_definition(evaluation_config)
    # The definition helper is also used by the live evaluator; unavailable rows
    # must participate in the same explicit aggregation definition.
    return dict(schema="clipp2.simulation.evaluation.v2", evaluation_definition=definition,
        evaluation_definition_id=content_hash(definition), cell_id=job["cell_id"], history_id=job["history_id"],
        family=job["family"], acquisition_id=job["acquisition"]["id"], acquisition=job["acquisition"],
        phase=job["phase"], job=job, status=status, error=error, metrics=[], mutations=[],
        calibration=[], cap_audit=[], stage_three_containment=[], information=[],
        applicability=[dict(arm=arm, available=False, status=status, planned=True)
                       for arm in ("A", "B", "C", "B3", "C3")])


def reevaluate_cell(root, cell, evaluation_config):
    """Derive a new record solely from the selected attempt's saved evidence."""
    from .evaluation import evaluate
    root = Path(root).resolve()
    job = cell["job"]
    try:
        history, catalog, paired, acquisition = _load_evidence(root, cell)
        value = evaluate(history, catalog, paired, acquisition=acquisition, evaluation_config=evaluation_config)
        value.update(cell_id=job["cell_id"], history_id=job["history_id"], family=job["family"],
            acquisition_id=job["acquisition"]["id"], acquisition=job["acquisition"], phase=job["phase"], job=job,
            status="derived_from_preserved_evidence")
    except (OSError, ValueError, KeyError, TypeError, AssertionError, zipfile.BadZipFile) as error:
        value = unavailable_evaluation(job, "unavailable_derived_evidence", evaluation_config=evaluation_config,
                                       error=dict(type=type(error).__name__, message=str(error)))
    value["derivation"] = dict(implementation_version=__version__, source_study=str(root),
        selected_attempt=cell.get("attempt"), source_implementation_version=cell.get("implementation_version"),
        source_protocol_version=cell.get("protocol_version"), inference_decisions_preserved=True,
        fit_count=0, assignment_pass_count=0, threshold_selection_count=0)
    return value
