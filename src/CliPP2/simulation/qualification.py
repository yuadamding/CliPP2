"""Bounded validation of the exact cohort, distinct from a fitted main study."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from html import escape
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from CliPP2.io.data import OriginalCatalog
from CliPP2.config import PROTOCOL_VERSION
from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt

from .cohort import acquire_cohort, cohort_history, load_cohort
from .design import jobs, make_contract, workflow_config
from .study import read_json, save_json

PILOT_VARIANTS = ("S01.01", "S03.02", "S05.02")


def _new_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Qualification output must be empty: {path}")
    return path


def _assert_equal(actual, expected, name):
    if not np.array_equal(actual, expected, equal_nan=False):
        raise AssertionError(f"Canonical catalog differs from prescribed {name}")


def _roundtrip(history, frame, directory):
    """Compare raw catalog values, not the masked numerical placeholders."""
    directory.mkdir(parents=True, exist_ok=True)
    path = write_tumor_txt(directory / "input_catalog.tsv", frame, frame.attrs["metadata"])
    catalog = load_original_catalog(path)
    acquisition = frame.attrs["acquisition"]
    if catalog.mutation_ids != history.mutation_ids or catalog.region_ids != history.region_ids:
        raise AssertionError("Canonical catalog axes do not align with physical truth")
    fields = dict(alt_counts="alt_counts", total_counts="depth", count_observed="count_observed",
        input_excluded="input_excluded", eligible="original_eligible", major_cn="supplied_major_cn",
        minor_cn="supplied_minor_cn", purity="supplied_purity", normal_cn="normal_cn",
        segment_ids="supplied_segment_ids", mapping_genuine="mapping_genuine")
    for name, expected in fields.items():
        _assert_equal(getattr(catalog, name), acquisition[expected], name)
    _assert_equal(catalog.normal_cn_supplied, np.ones(history.phi.shape, bool), "normal CN provenance")
    _assert_equal(catalog.major_cn, history.major_cn, "true major CN")
    _assert_equal(catalog.minor_cn, history.minor_cn, "true minor CN")
    bounds = catalog.default_bounds()
    _assert_equal(bounds.lower, np.ones(history.phi.shape), "full support lower endpoint")
    _assert_equal(bounds.upper, history.major_cn, "full support upper endpoint")
    positive = history.phi > 0
    if np.any(positive & ((history.multiplicity < 1) | (history.multiplicity > bounds.upper))):
        raise AssertionError("Unperturbed original support excludes positive-carrier truth")
    # Exercise the canonical writer a second time using the loader's retained
    # rows, and independently exercise the immutable JSON-payload constructor.
    reloaded = OriginalCatalog.from_dict(catalog.to_dict())
    table = pd.DataFrame([dict(row) for row in catalog.source_rows])
    repeated = load_original_catalog(write_tumor_txt(directory / "roundtrip.tsv", table, dict(catalog.metadata)))
    for name in (*fields, "normal_cn_supplied"):
        _assert_equal(getattr(reloaded, name), getattr(catalog, name), "serialized " + name)
        _assert_equal(getattr(repeated, name), getattr(catalog, name), "rewritten " + name)
    (directory / "roundtrip.tsv").unlink()
    observed = catalog.count_observed
    return dict(pairs=int(observed.size), observed_pairs=int(observed.sum()),
        missing_pairs=int((~observed).sum()), input_excluded_pairs=int(catalog.input_excluded.sum()),
        observed_zero_depth_pairs=int((observed & (catalog.total_counts == 0)).sum()),
        informative_pairs=int((observed & catalog.eligible & (catalog.total_counts > 0)).sum()),
        original_mutations=len(catalog.mutation_ids), positive_carrier_pairs=int(positive.sum()),
        inherited_support_exclusions=0, status="passed")


def _validate_cohort(cohort, root, *, materialize):
    """Replay every realized mixture; retain full catalogs only on request."""
    histories, acquisitions = [], []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".roundtrip-", dir=root) as scratch:
        for index, (history_id, setting) in enumerate(sorted(cohort.histories.items()), 1):
            record = dict(history_id=history_id, variant_id=setting["variant_id"],
                          cohort_content_id=cohort.content_id,
                          split=setting["split"], replicate=setting["replicate"])
            try:
                # Construction calls History.validate(), including chronological
                # copy inheritance replay and every mutation-region reduction.
                history = cohort_history(cohort, history_id)
                if history.phi.shape != (1500, 4) or len(history.segment_keys) != 50:
                    raise AssertionError("Qualification must retain the prescribed 1500/50/4 dimensions")
                record.update(status="passed", pairs=int(history.phi.size), mutations=len(history.mutation_ids),
                    segments=len(history.segment_keys), physical_events=len(history.events),
                    physical_copies=len(history.copies), positive_carrier_pairs=int((history.phi > 0).sum()))
                if materialize:
                    history.write_truth(root / "histories" / history_id)
                selected = (setting["acquisition_ids"] if materialize else
                            setting["acquisition_ids"] if setting["split"] == "development"
                            and setting["replicate"] == 1 else ())
                for acquisition_id in selected:
                    cell = dict(history_id=history_id, variant_id=setting["variant_id"],
                                cohort_content_id=cohort.content_id,
                                split=setting["split"], acquisition_id=acquisition_id)
                    try:
                        frame = acquire_cohort(history, cohort, acquisition_id)
                        directory = (root / "catalogs" / f"{history_id}.{acquisition_id}"
                                     if materialize else Path(scratch))
                        cell.update(_roundtrip(history, frame, directory))
                        if materialize:
                            cell["input_path"] = str((directory / "input_catalog.tsv").relative_to(root))
                    except Exception as error:
                        cell.update(status="failed", error_type=type(error).__name__, error=str(error))
                    acquisitions.append(cell)
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error))
            histories.append(record)
            if index % 60 == 0 or index == len(cohort.histories):
                print(json.dumps(dict(event="cohort_validation", completed_histories=index,
                    planned_histories=len(cohort.histories), failed_histories=sum(r["status"] != "passed" for r in histories),
                    elapsed_seconds=round(time.perf_counter()-started, 2))), flush=True)
    return histories, acquisitions


def _worker(task_path):
    """Observe real operations durably without changing their inputs/results."""
    from . import experiment
    from .study import _run_cell
    task = read_json(task_path)
    log = Path(task_path).with_name("operation_log.jsonl")
    originals = {name: getattr(experiment, name) for name in ("_fit_accepted", "assign_mutations")}
    sequence = 0

    def append(record):
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def observed(name):
        def call(*args, **kwargs):
            nonlocal sequence
            sequence += 1
            operation_id = sequence
            kind = "fit" if name == "_fit_accepted" else "assignment"
            arm = str(args[4]) if kind == "fit" else str(args[0].provenance.get("arm", "unknown"))+"3"
            began = time.perf_counter()
            append(dict(operation_id=operation_id, operation=kind, arm=arm, status="started",
                        started_monotonic_seconds=began))
            try:
                value = originals[name](*args, **kwargs)
            except BaseException as error:
                append(dict(operation_id=operation_id, operation=kind, arm=arm, status="failed",
                    elapsed_seconds=time.perf_counter()-began, error_type=type(error).__name__, error=str(error)))
                raise
            append(dict(operation_id=operation_id, operation=kind, arm=arm, status="complete",
                        elapsed_seconds=time.perf_counter()-began))
            return value
        return call

    try:
        for name in originals:
            setattr(experiment, name, observed(name))
        value = _run_cell(task)
        save_json(Path(task_path).with_name("worker_result.json"), value)
    finally:
        for name, original in originals.items():
            setattr(experiment, name, original)


def _pilot(task, timeout_seconds):
    """A new interpreter and process group give each real cell a hard limit."""
    root, job = Path(task["root"]), task["job"]
    directory = root / "workers" / job["cell_id"]
    directory.mkdir(parents=True)
    task_path = directory / "task.json"
    save_json(task_path, task)
    environment = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[key] = "1"
    began = time.perf_counter()
    timed_out = False
    process, launch_error = None, None
    with (directory / "stdout.log").open("w") as stdout, (directory / "stderr.log").open("w") as stderr:
        try:
            process = subprocess.Popen([sys.executable, "-c",
                "from CliPP2.simulation.qualification import _worker; import sys; _worker(sys.argv[1])",
                str(task_path)], stdout=stdout, stderr=stderr, env=environment, start_new_session=True)
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Kill the whole worker group, including any descendants, before
            # inspecting artifacts. Native CPU threads die with the process.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        except OSError as error:
            launch_error = dict(error_type=type(error).__name__, error=str(error))
    returncode = process.returncode if process is not None else None
    result_path = directory / "worker_result.json"
    result = read_json(result_path) if result_path.exists() else None
    records = []
    log = directory / "operation_log.jsonl"
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A killed final write is incomplete evidence, never a success.
                break
    operations = {}
    for record in records:
        operations[record["operation_id"]] = record
    for record in operations.values():
        if record["status"] == "started":
            record["elapsed_seconds"] = time.perf_counter()-record["started_monotonic_seconds"]
            record["uncompleted_reason"] = "resource_timeout" if timed_out else "worker_terminated"
    costs = {f"{kind}_{state}": sum(r["operation"] == kind and r["status"] == state for r in operations.values())
             for kind in ("fit", "assignment") for state in ("started", "complete", "failed")}
    costs.update(fit_attempts=sum(r["operation"] == "fit" for r in operations.values()),
                 assignment_attempts=sum(r["operation"] == "assignment" for r in operations.values()))
    cell = root / "cells" / job["cell_id"]
    paired = [(path, read_json(path)) for path in sorted(cell.glob("attempt-*/configuration-*/paired_result.json"))]
    accepted = sorted(str(path.relative_to(root)) for path in cell.glob("attempt-*/configuration-*/arm_outputs/*/accepted_fit.json"))
    status = ("resource_timeout" if timed_out else "execution_failed" if launch_error or returncode or result is None
              or result["status"] != "complete" else "complete")
    return dict(cell_id=job["cell_id"], history_id=job["history_id"], variant_id=job["variant_id"],
        acquisition=job["acquisition"], dimensions=job["dimensions"], status=status,
        analysis_status=("unavailable_resource_limit" if timed_out else
                         "unavailable_execution_failure" if status != "complete" else
                         paired[-1][1]["status"] if paired else "unavailable_missing_paired_result"),
        timeout_seconds=timeout_seconds, elapsed_seconds=time.perf_counter()-began, returncode=returncode,
        launch_error=launch_error,
        operation_counts=costs, operations=list(operations.values()), accepted_fit_count=len(accepted),
        accepted_fit_paths=accepted, published_pair_results=[str(p.relative_to(root)) for p, _ in paired],
        native_cost_ledgers=[p["cost_ledger"] for _, p in paired],
        artifact_root="native_pilots",
        worker_result=(str(result_path.relative_to(root)) if result is not None else None),
        partial_output_directory=str(cell.relative_to(root)),
        stdout=str((directory / "stdout.log").relative_to(root)), stderr=str((directory / "stderr.log").relative_to(root)),
        output_bytes=sum(p.stat().st_size for p in cell.rglob("*") if p.is_file()),
        scientific_abstention=False if timed_out else None)


def _html(summary):
    counts, validation = summary["counts"], summary["validation"]
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Cohort qualification</title>",
        "<style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:0 1em}"
        "table{border-collapse:collapse;width:100%}td,th{padding:.4em;border:1px solid #ddd;text-align:left}</style>",
        "<h1>Authoritative cohort qualification</h1><p><strong>This is not a completed fitted cohort study.</strong> "
        "Generator validation and canonical input round trips do not establish scientific performance or calibration. "
        "The main cohort has not been fitted by this command. CUDA was not exercised.</p>",
        f"<p>Status: {escape(summary['status'])}. NumPy {escape(summary['numpy_version'])}; "
        "exact prescribed semantic seeds, recorded mixtures, 1,500 mutations, 50 segments and four regions.</p>",
        f"<p>Physical histories: {validation['histories_passed']}/{counts['planned_histories']}; "
        f"validated mutation-region pairs: {validation['physical_pairs_validated']:,}; "
        f"catalog round trips: {validation['acquisitions_passed']}/{counts['planned_roundtrips']}. "
        f"Materialized truth histories: {counts['materialized_histories']}; "
        f"materialized input catalogs: {counts['materialized_catalogs']}.</p>",
        "<h2>Native pilots</h2><p>Only the cells below were requested for native inference. "
        "Each has a fresh worker and a fixed wall-clock limit. Timeouts are resource-unavailable analyses, "
        "not scientific abstentions. Started operations and published accepted fits are counted separately.</p>",
        "<table><tr><th>Variant</th><th>Status</th><th>Analysis</th><th>Accepted fits</th><th>Fit attempts</th><th>Seconds</th></tr>"]
    for row in summary["native_pilots"]:
        parts.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in
            (row["variant_id"], row["status"], row["analysis_status"], row["accepted_fit_count"],
             row["operation_counts"]["fit_attempts"], round(row["elapsed_seconds"], 2))) + "</tr>")
    parts += ["</table>", f"<p>Native pilot disposition: {escape(summary['native_pilot_status'])}.</p>",
        "<h2>Case matrix</h2><table><tr><th>Variant</th><th>Histories passed/planned</th>"
        "<th>Round trips passed/planned</th><th>Pilot</th></tr>"]
    for row in summary["case_matrix"]:
        parts.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in
            (row["variant_id"], f"{row['histories_passed']}/{row['histories_planned']}",
             f"{row['roundtrips_passed']}/{row['roundtrips_planned']}", row["native_pilot_status"])) + "</tr>")
    parts.append("</table><p>Detailed records and actual worker artifacts are linked by relative paths in "
                 "<a href='qualification_summary.json'>qualification_summary.json</a>.</p></html>")
    return "\n".join(parts)


def qualify_cohort(settings_path, outdir, *, native_pilot=True, timeout_seconds=180, workers=3, materialize=False,
                   reference_match_atol=None):
    """Validate 720 histories and 120 catalogs; optionally run three paired cells.

    ``materialize=True`` additionally retains all 720 full physical truths and
    generates/round-trips all 2,640 prescribed input catalogs. It never changes
    the inference population or expands the three-cell native pilot.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or type(workers) is not int or workers < 1:
        raise ValueError("Timeout must be positive and workers a positive integer")
    if type(native_pilot) is not bool or type(materialize) is not bool:
        raise TypeError("native_pilot and materialize must be booleans")
    cohort = load_cohort(settings_path)
    if np.__version__ != cohort.settings["numpy_version"]:
        raise RuntimeError(f"Exact cohort qualification requires NumPy {cohort.settings['numpy_version']}; got {np.__version__}")
    if sys.flags.optimize:
        raise RuntimeError("Qualification requires Python assertions enabled for physical replay validation")
    root, began = _new_directory(outdir), time.perf_counter()
    contract = make_contract(cohort_settings=cohort.path, workers=workers,
                             reference_match_atol=reference_match_atol)
    if contract["cohort"]["content_id"] != cohort.content_id:
        raise ValueError("Authoritative cohort changed while preparing qualification")
    save_json(root / "qualification_contract.json", dict(contract=contract,
        native_pilot_variants=list(PILOT_VARIANTS) if native_pilot else [],
        native_timeout_seconds=timeout_seconds, materialize=materialize,
        qualification_scope="physical_and_input_validation_plus_bounded_development_pilot_not_main_study"))
    histories, acquisitions = _validate_cohort(cohort, root, materialize=materialize)
    planned_roundtrips = sum(len(h["acquisition_ids"]) for h in cohort.histories.values()
        if materialize or h["split"] == "development" and h["replicate"] == 1)
    validation = dict(histories_passed=sum(row["status"] == "passed" for row in histories),
        histories_failed=sum(row["status"] != "passed" for row in histories),
        physical_pairs_validated=sum(row.get("pairs", 0) for row in histories if row["status"] == "passed"),
        acquisitions_passed=sum(row["status"] == "passed" for row in acquisitions),
        acquisitions_failed=sum(row["status"] != "passed" for row in acquisitions),
        acquisitions_unavailable=planned_roundtrips-len(acquisitions))
    valid = (validation["histories_passed"] == len(cohort.histories)
        and validation["acquisitions_passed"] == planned_roundtrips)
    pilots = []
    if native_pilot and valid:
        selected = [job for job in jobs(contract, "development")
                    if job["history_index"] == 0 and job["variant_id"] in PILOT_VARIANTS and job["acquisition"]["id"] == "A4"]
        if len(selected) != len(PILOT_VARIANTS):
            raise ValueError("Exact prescribed native pilot cells are missing or duplicated")
        tasks = [dict(root=str(root / "native_pilots"), job=job,
            execution_configs=[asdict(workflow_config())],
            reference_match_atol=reference_match_atol) for job in selected]
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            pilots = list(executor.map(lambda task: _pilot(task, timeout_seconds), tasks))
    pilot_status = ("not_requested" if not native_pilot else "skipped_invalid_generator_or_catalog" if not valid
                    else "finished" if all(row["status"] == "complete" for row in pilots) else "partial_or_unavailable")
    matrix = []
    for variant in sorted(cohort.variants):
        h = [row for row in histories if row["variant_id"] == variant]
        a = [row for row in acquisitions if row["variant_id"] == variant]
        p = next((row for row in pilots if row["variant_id"] == variant), None)
        matrix.append(dict(variant_id=variant, histories_planned=len(h),
            histories_passed=sum(row["status"] == "passed" for row in h),
            roundtrips_planned=sum(len(row["acquisition_ids"]) for row in cohort.histories.values()
                if row["variant_id"] == variant and (materialize or row["split"] == "development" and row["replicate"] == 1)),
            roundtrips_passed=sum(row["status"] == "passed" for row in a),
            native_pilot_status=p["status"] if p else "not_run"))
    summary = dict(schema="clipp2.cohort-qualification.v3", study_version="1.2", protocol_version=PROTOCOL_VERSION,
        cohort_content_id=cohort.content_id,
        source_cohort_protocol_version=cohort.settings["workflow_protocol"],
        status="validation_passed" if valid else "validation_failed", settings_path=cohort.path,
        numpy_version=np.__version__, required_numpy_version=cohort.settings["numpy_version"],
        exact_rng_version=True, dimensions=contract["dimensions"], validation=validation,
        counts=dict(planned_histories=len(cohort.histories), planned_datasets=sum(len(h["acquisition_ids"]) for h in cohort.histories.values()),
            planned_roundtrips=planned_roundtrips, event_recipes=len(cohort.variants),
            materialized_histories=validation["histories_passed"] if materialize else 0,
            materialized_catalogs=validation["acquisitions_passed"] if materialize else 0,
            main_native_cells_fitted=0, native_pilot_cells=len(pilots),
            native_fit_attempts=sum(row["operation_counts"]["fit_attempts"] for row in pilots),
            native_assignment_attempts=sum(row["operation_counts"]["assignment_attempts"] for row in pilots),
            native_accepted_fits=sum(row["accepted_fit_count"] for row in pilots)),
        history_validation=histories, acquisition_validation=acquisitions, case_matrix=matrix,
        native_pilot_status=pilot_status, native_pilots=pilots,
        native_pilot_counts=dict(Counter(row["status"] for row in pilots)),
        full_study_completed=False, scientific_calibration_approved=False, cuda_executed=False,
        materialization_requested=materialize, elapsed_seconds=time.perf_counter()-began,
        limitations=["Generator qualification does not establish fitted biological accuracy.",
            "Native pilot timeouts are resource limitations, not scientific abstentions.",
            "The 600 main histories and their 2400 acquisitions have not been fitted by this command.",
            "The pilot uses one declared Protocol 2.3 workflow; scientific performance is descriptive."])
    save_json(root / "qualification_summary.json", summary)
    (root / "qualification.html").write_text(_html(summary), encoding="utf-8")
    if materialize:
        save_json(root / "materialization_manifest.json", dict(schema="clipp2.cohort-materialization.v1",
            status="complete" if valid else "incomplete", counts=summary["counts"],
            full_study_completed=False, inference_requested=native_pilot,
            catalogs=[row for row in acquisitions if row.get("input_path")],
            qualification_summary="qualification_summary.json"))
    return summary


def materialize_cohort(settings_path, outdir):
    """Generate all prescribed truth/catalog files, without any native fitting."""
    return qualify_cohort(settings_path, outdir, native_pilot=False, materialize=True)
