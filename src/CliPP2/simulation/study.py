"""Resumable study execution with history-level pairing and fixed configuration."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd

from CliPP2.config import PROTOCOL_VERSION, STAGE_THREE_REFERENCE_POLICY
from CliPP2._version import __version__
from CliPP2.io.data import content_hash, json_value
from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt

from .design import jobs, roster_counts, resolve_config, seed_for, workflow_config


def save_json(path, payload):
    """New-file publication: interrupted work never masquerades as completion."""
    from CliPP2.io.publication import publish_json
    return publish_json(path, json.dumps(json_value(payload), allow_nan=False, separators=(",", ":"))+"\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _latest_completion(directory):
    paths = list(Path(directory).glob("completion*.json"))
    values = [read_json(path) for path in paths]
    return max(values, key=lambda value: value["attempt"]) if values else None


def _archive_completed(directory):
    """Compress only verified, generated outputs; preserve their entire contents."""
    import shutil
    for source in sorted(directory.glob("configuration-*")):
        if not source.is_dir():
            continue
        destination = source.with_suffix(".zip")
        temporary = destination.with_suffix(".zip.partial")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    archive.write(path, str(path.relative_to(source)))
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise IOError("Generated output archive failed its integrity check")
        os.link(temporary, destination)
        temporary.unlink()
        shutil.rmtree(source)


def _paired_result(attempt, index=1):
    from .reporting import _configuration
    return _configuration(Path(attempt), index)


@lru_cache(maxsize=2)
def _cached_cohort(path, expected_content_id, process_id):
    from .cohort import load_cohort
    return load_cohort(path, expected_content_id=expected_content_id)


def _cohort(path, expected_content_id):
    path = str(Path(path).resolve())
    if not expected_content_id:
        raise ValueError("Authoritative cohort requires an expected content identity")
    return _cached_cohort(path, expected_content_id, os.getpid())


def _verify_cohort_snapshot(path, expected_content_id):
    """A coordinator execution boundary rereads content; worker use stays immutable."""
    _cached_cohort.cache_clear()
    return _cohort(path, expected_content_id)


def _history(job):
    if job.get("cohort_settings"):
        from .cohort import cohort_history
        return cohort_history(_cohort(job["cohort_settings"], job["cohort_content_id"]), job.get("source_history_id", job["history_id"]))
    from .biology import generate_history
    dims = job["dimensions"]
    history = generate_history(job["family"], job["history_seed"],
        n_mutations=dims["mutations"], n_segments=dims["segments"], n_regions=dims["regions"])
    return history


def _acquire(job, history):
    if job.get("cohort_settings"):
        from .cohort import acquire_cohort
        return acquire_cohort(history, _cohort(job["cohort_settings"], job["cohort_content_id"]), job["acquisition"]["id"])
    from .acquisition import acquire
    return acquire(history, job["acquisition_seed"], job["acquisition"]["purity"], job["acquisition"]["depth"],
                   perturbation=job.get("perturbation"))


def _peak_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value*1024)


def _experiment_execution_failures(experiment):
    """Distinguish caught execution errors from completed scientific rejection."""
    failures = []
    for arm, record in ((None, experiment), *experiment.get("arms", {}).items()):
        if record.get("status") not in ("failed", "execution_failed"):
            continue
        failures.append(dict(configuration_index=1, arm=arm,
            failed_stage=record.get("failed_stage"), failed_phase=record.get("failed_phase"),
            error_type=record.get("error_type"), error=record.get("error")))
    return failures


def _run_cell(task):
    from .evaluation import evaluate
    from .experiment import run_paired_experiment
    root, job = Path(task["root"]), task["job"]
    evaluation_config = dict(reference_match_atol=task.get("reference_match_atol"),
                             reference_distance=task.get("reference_distance", "l2"))
    directory = root / "cells" / job["cell_id"]
    if job.get("cohort_settings"):
        _cohort(job["cohort_settings"], job["cohort_content_id"])
    prior = _latest_completion(directory)
    if prior is not None:
        value = prior
        if (value.get("protocol_version") != PROTOCOL_VERSION
                or value.get("implementation_version") != __version__):
            raise ValueError("Cell replay requires the current execution protocol and implementation; earlier results remain historical")
        if (value["job"] != job or value.get("execution_configs") != task.get("execution_configs")
                or value.get("evaluation_config") != evaluation_config):
            raise ValueError("Resume cell identity/configuration differs from the frozen job")
        if value["status"] != "execution_failed" or not task.get("retry_failed", False):
            return value
    directory.mkdir(parents=True, exist_ok=True)
    attempts = sorted(directory.glob("attempt-*"))
    attempt = directory / f"attempt-{len(attempts)+1:03d}"
    completion = directory / ("completion.json" if not attempts else f"completion.{len(attempts)+1:03d}.json")
    attempt.mkdir()
    started = time.perf_counter()
    save_json(attempt / "job.json", job)
    results, costs, execution_failures = [], [], []
    original_evaluable_count, execution_phase = None, "history"
    try:
        from CliPP2.config import WorkflowConfig
        configurations = task.get("execution_configs") or [asdict(workflow_config())]
        if len(configurations) != 1:
            raise ValueError("Protocol 2.3 executes exactly one declared workflow configuration")
        config = WorkflowConfig.from_dict(configurations[0])
        from .biology import history_content_id
        from .acquisition import validate_history_acquisition_link
        history = _history(job)
        history_manifest = read_json(root / "histories" / job["history_id"] / "history_manifest.json")
        if history_manifest.get("history_content_id") != history_content_id(history):
            raise ValueError("Generated history differs from its prepared history manifest")
        execution_phase = "acquisition"
        acquisition = job["acquisition"]
        frame = _acquire(job, history)
        frame.attrs["acquisition"].update(id=acquisition["id"], panel=job["phase"])
        metadata = dict(frame.attrs.get("metadata", {}), tumor_id=job["cell_id"])
        input_path = attempt / "input_catalog.tsv"
        write_tumor_txt(input_path, frame, metadata)
        save_json(attempt / "acquisition.json", frame.attrs.get("acquisition", {}))
        catalog = load_original_catalog(input_path, max_major_cn=config.fit.max_major_cn, missing_counts=config.missing_counts)
        truth_link = validate_history_acquisition_link(history, frame.attrs["acquisition"], catalog,
                                                       history_manifest=history_manifest)
        if truth_link["status"] != "verified_history_binding":
            raise ValueError("Live history/acquisition linkage is not verified: " + str(truth_link))
        original_evaluable_count = int((np.isfinite(history.phi).all(axis=1)
            & (catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)).any(axis=1)).sum())
        execution_phase = "before_inference"
        def progress(event):
            fields = ("phase", "stage", "elapsed_seconds", "phase_elapsed_seconds", "occupied_q", "membership_audit", "operation")
            payload = dict(configuration=1, **{key: event[key] for key in fields if key in event})
            temporary = attempt / ".progress.json.tmp"
            temporary.write_text(json.dumps(json_value(payload), allow_nan=False))
            os.replace(temporary, attempt / "progress.json")
        execution_phase = "paired_inference"
        experiment = run_paired_experiment(input_path, attempt / "configuration-1", config,
            progress_callback=progress,
            run_manifest_id=task.get("run_manifest_id"))
        costs.append(experiment["cost_ledger"])
        execution_failures.extend(_experiment_execution_failures(experiment))
        save_json(attempt / "cost-ledger-1.json", experiment["cost_ledger"])
        execution_phase = "evaluation"
        evaluation = evaluate(history, catalog, experiment, acquisition=frame.attrs.get("acquisition"),
                              evaluation_config=evaluation_config, truth_qualification=truth_link["status"])
        evaluation.update(cell_id=job["cell_id"], history_id=job["history_id"], family=job["family"],
                          acquisition_id=acquisition["id"], acquisition=acquisition, phase=job["phase"], job=job)
        save_json(attempt / "evaluation-1.json", evaluation)
        results.append(str((attempt / "evaluation-1.json").relative_to(root)))
        if task.get("archive_completed", False):
            _archive_completed(attempt)
        value = dict(status="execution_failed" if execution_failures else "complete")
    except Exception as exc:
        value = dict(status="execution_failed", error_type=type(exc).__name__, error=str(exc),
                     traceback=traceback.format_exc())
        execution_failures.append(dict(configuration_index=None, arm=None, failed_stage="worker",
            error_type=type(exc).__name__, error=str(exc)))
    value.update(job=job, evaluations=results, cost_ledgers=costs, attempt=str(attempt.relative_to(root)),
                 elapsed_seconds=time.perf_counter()-started, peak_resident_bytes=_peak_bytes())
    value["execution_failures"] = execution_failures
    value["original_evaluable_count"] = original_evaluable_count
    value["execution_phase"] = execution_phase
    value["execution_configs"] = task.get("execution_configs")
    value["evaluation_config"] = evaluation_config
    value["evaluation_schema"] = "clipp2.simulation.evaluation.v5"
    value["reference_policy"] = STAGE_THREE_REFERENCE_POLICY
    value["protocol_version"] = PROTOCOL_VERSION
    value["implementation_version"] = __version__
    value["prior_cost_ledgers"] = ((prior.get("prior_cost_ledgers", [])+prior.get("cost_ledgers", []))
                                  if prior else [])
    value["output_bytes"] = sum(p.stat().st_size for p in attempt.rglob("*") if p.is_file())
    if value["status"] == "execution_failed":
        save_json(attempt / "failure.json", value)
    save_json(completion, value)
    return value


def _prepare_histories(root, job_list):
    from .biology import History, history_content_id
    planned = {}
    for job in job_list:
        expected = {k: job[k] for k in ("history_id", "family", "history_seed", "dimensions")}
        if job.get("cohort_settings"):
            expected["cohort_content_id"] = job["cohort_content_id"]
        source_id = job.get("source_history_id", job["history_id"])
        if source_id != job["history_id"]:
            expected["source_history_id"] = source_id
        previous = planned.get(job["history_id"])
        if previous is not None:
            if previous[1] != expected:
                raise ValueError("Repeated history ID has incompatible biological layout or source identity")
            continue
        planned[job["history_id"]] = job, expected
    for job, expected in planned.values():
        directory = root / "histories" / job["history_id"]
        if job.get("cohort_settings"):
            _cohort(job["cohort_settings"], job["cohort_content_id"])
        if (directory / "history_manifest.json").exists():
            existing = read_json(directory / "history_manifest.json")
            preserved = History.from_dict(read_json(directory / "history.json"))
            expected["history_content_id"] = history_content_id(preserved)
            if existing != expected:
                raise ValueError("Resume physical history differs")
            continue
        history = _history(job)
        expected["history_content_id"] = history_content_id(history)
        directory.mkdir(parents=True, exist_ok=True)
        history.write_truth(directory)
        save_json(directory / "history_manifest.json", expected)


def run_jobs(root, job_list, *, workers, contract=None, retry_failed=False):
    job_list = list(job_list)
    for path, identity in {(job["cohort_settings"], job["cohort_content_id"])
                           for job in job_list if job.get("cohort_settings")}:
        _verify_cohort_snapshot(path, identity)
    manifests = root / "job_manifests"
    manifests.mkdir(exist_ok=True)
    manifest = manifests / f"batch-{len(list(manifests.glob('batch-*.json')))+1:03d}.json"
    save_json(manifest, job_list)
    _prepare_histories(root, job_list)
    tasks = []
    for job in job_list:
        config = resolve_config(contract) if contract else workflow_config()
        tasks.append(dict(root=str(root), job=job, execution_configs=[asdict(config)],
                          retry_failed=retry_failed,
                          archive_completed=contract["resource_limits"].get("archive_completed_outputs", False) if contract else False,
                          reference_match_atol=contract.get("evaluation", {}).get("reference_match_atol") if contract else None,
                          reference_distance=contract.get("evaluation", {}).get("reference_distance", "l2") if contract else "l2",
                          run_manifest_id=contract.get("run_manifest_id") if contract else None))
    values = []
    # Prevent BLAS oversubscription; each native fit has an explicit single thread.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_cell, task): task["job"] for task in tasks}
        for future in as_completed(futures):
            value = future.result()
            values.append(value)
            print(json.dumps(dict(event="cell_complete", cell=value["job"]["cell_id"],
                status=value["status"], completed=len(values), planned=len(tasks),
                elapsed_seconds=round(value["elapsed_seconds"], 2))), flush=True)
    return sorted(values, key=lambda row: row["job"]["cell_id"])


def assess_development(root, cells, contract, main_roster):
    """Describe every planned development cell under the fixed configuration.

    Numerical failures remain visible. This receipt never chooses a policy or
    prevents main execution because of scientific performance.
    """
    planned = list(jobs(contract, "development"))
    completed = {cell["job"]["cell_id"]: cell for cell in cells}
    if len(completed) != len(cells) or set(completed) - {job["cell_id"] for job in planned}:
        raise ValueError("Development completions differ from the declared population")
    records = []
    for job in planned:
        cell = completed.get(job["cell_id"])
        if cell is not None and cell["job"] != job:
            raise ValueError("Development completion job differs from the declared population")
        records.append(dict(cell_id=job["cell_id"], history_id=job["history_id"], family=job["family"],
            acquisition=job["acquisition"]["id"], status=cell["status"] if cell else "not_executed",
            selected_attempt=cell.get("attempt") if cell else None,
            completion_identity=content_hash(cell) if cell else None,
            evaluations=list(cell.get("evaluations", ())) if cell else [],
            execution_failures=list(cell.get("execution_failures", ())) if cell else []))
    return dict(schema="clipp2.simulation-development-assessment.v1", protocol_version=PROTOCOL_VERSION,
        study_contract_id=content_hash(contract), main_roster_id=main_roster["roster_id"],
        workflow_config=asdict(resolve_config(contract)), policy="descriptive_fixed_configuration",
        selection_performed=False, performance_controls_main_execution=False,
        planned_cells=len(planned), completed_cells=sum(row["status"] == "complete" for row in records),
        execution_failed_cells=sum(row["status"] == "execution_failed" for row in records),
        not_executed_cells=sum(row["status"] == "not_executed" for row in records), cells=records)


def _publish_development_assessment(root, assessment):
    """Append revised attempt evidence on retry; preserve all prior assessments."""
    paths = list(root.glob("development_assessment*.json"))
    for path in paths:
        previous = read_json(path)
        if any(previous.get(key) != assessment[key] for key in ("schema", "protocol_version",
                "study_contract_id", "main_roster_id", "workflow_config")):
            raise ValueError("Development assessment differs from the frozen study configuration")
        if previous == assessment:
            return path
    name = "development_assessment.json" if not paths else f"development_assessment.{len(paths)+1:03d}.json"
    path = root / name
    save_json(path, assessment)
    return path


def _qualification_summary(output, summary_name):
    """Validate a passed control summary and its published output inventory."""
    output = Path(output)
    summary = read_json(output / summary_name)
    ledger = summary.get("cost_ledger", {})
    passes = ledger.get("assignment_pass_count")
    if (summary.get("status") != "passed" or ledger.get("fit_count") != 0
            or type(passes) is not int or passes < 0
            or any(event.get("status") != "complete" for event in ledger.get("events", []))):
        raise ValueError("Qualification summary is not a completed zero-fit control result")
    receipts = list(output.rglob("*assignment_summary.json"))
    if len(receipts) != passes:
        raise ValueError("Qualification assignment publications do not match its cost summary")
    for receipt in receipts:
        record = read_json(receipt)
        if record.get("manifest_schema") != "clipp2.file-list.v1" or not record.get("files"):
            raise ValueError("Qualification publication lacks an output inventory")
        for name in record["files"]:
            path = (receipt.parent / name).resolve()
            if not path.is_relative_to(output.resolve()) or not path.is_file():
                raise ValueError("Qualification publication output is missing or outside its step")
    for name in summary.get("paths", []):
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file():
            raise ValueError("Qualification summary output is missing or outside its step")
    return summary


def _qualification_step(root, name, runner, *, summary_name, identity, retry_failed=False):
    """Retain numbered control attempts and recover published success without rerun."""
    directory = Path(root) / "qualification_steps" / name
    directory.mkdir(parents=True, exist_ok=True)
    binding = dict(step=name, protocol_version=PROTOCOL_VERSION, implementation_version=__version__, **identity)
    attempts = sorted(directory.glob("attempt-*"))

    def completed(attempt, summary, recovered=False):
        receipt = dict(schema="clipp2.qualification-step.v1", status="passed", identity=binding,
            attempt=attempt.name, summary_name=summary_name, summary_content_id=content_hash(summary),
            recovered_from_summary=recovered, cost_status="known", cost_ledger=summary["cost_ledger"])
        save_json(attempt / "completion.json", receipt)
        return receipt

    if attempts and (attempts[-1] / "started.json").exists():
        prior = attempts[-1]
        started = read_json(prior / "started.json")
        if started.get("identity") != binding:
            raise ValueError("Qualification step identity differs from its original attempt")
        completion = prior / "completion.json"
        if completion.exists():
            receipt = read_json(completion)
            if receipt.get("identity") != binding:
                raise ValueError("Qualification completion identity differs")
            if receipt["status"] == "passed":
                summary = _qualification_summary(prior / "output", summary_name)
                if receipt["summary_content_id"] != content_hash(summary):
                    raise ValueError("Completed qualification summary content changed")
                return receipt
            if not retry_failed:
                raise RuntimeError(f"Qualification step {name} failed; use --retry-failed for a retained new attempt")
        elif (prior / "output" / summary_name).exists():
            try:
                summary = _qualification_summary(prior / "output", summary_name)
            except (ValueError, OSError):
                pass  # Incomplete outputs are retained; a new attempt gets a fresh directory.
            else:
                return completed(prior, summary, recovered=True)
        # An incomplete attempt without a cost summary remains unknown-cost evidence.
    attempt = directory / f"attempt-{len(attempts)+1:03d}"
    attempt.mkdir()
    save_json(attempt / "started.json", dict(schema="clipp2.qualification-step.v1", status="started",
        identity=binding, attempt=attempt.name, summary_name=summary_name, cost_status="unknown"))
    try:
        runner(attempt / "output")
        summary = _qualification_summary(attempt / "output", summary_name)
        return completed(attempt, summary)
    except Exception as error:
        save_json(attempt / "completion.json", dict(schema="clipp2.qualification-step.v1", status="failed",
            identity=binding, attempt=attempt.name, summary_name=summary_name,
            error_type=type(error).__name__, error=str(error), cost_status="unknown", cost_ledger=None))
        raise


def execute(root, contract, *, resume=False, retry_failed=False):
    import fcntl
    root = Path(root).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / ("."+root.name+".lock")).open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("This study is already running; use the status command") from exc
        return _execute(root, contract, resume=resume, retry_failed=retry_failed)


def _freeze_main_roster(root, contract):
    """Freeze exact main-cell weights before development or main submission."""
    from .reporting import roster_identity
    payload = dict(schema="clipp2.simulation-roster.v1", families=contract["families"],
                   identity_policy="semantic_jobs_without_cohort_path", jobs=list(jobs(contract, "main")))
    record = dict(payload, roster_id=roster_identity(payload))
    path = root / "main_roster.json"
    if path.exists():
        if read_json(path) != record:
            raise ValueError("Main roster differs from the frozen replicate declaration")
    else:
        save_json(path, record)
    return record


def _execute(root, contract, *, resume=False, retry_failed=False):
    root = Path(root).resolve()
    if (contract.get("protocol_version") != PROTOCOL_VERSION or contract.get("schema") != "clipp2.simulation-contract.v7"
            or contract.get("implementation_version") != __version__
            or contract.get("evaluation", {}).get("reference_policy") != STAGE_THREE_REFERENCE_POLICY):
        raise ValueError(f"Execution requires a Protocol {PROTOCOL_VERSION} simulation contract; earlier studies require their original installed environment")
    if resume:
        frozen = read_json(root / "study_contract.json")
        if frozen != contract:
            raise ValueError("Resume requires the exact frozen study contract")
        if contract.get("cohort"):
            _verify_cohort_snapshot(root / "cohort_settings.json", contract["cohort"]["content_id"])
            if Path(contract["cohort"]["path"]).resolve() != (root / "cohort_settings.json").resolve():
                raise ValueError("Resume must use the verified frozen cohort snapshot")
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError("Study outputs must be new or empty; use --resume for this study")
        root.mkdir(parents=True, exist_ok=True)
        if contract.get("cohort"):
            from .cohort import load_cohort
            validated = load_cohort(contract["cohort"]["path"], expected_content_id=contract["cohort"]["content_id"])
            frozen_cohort = root / "cohort_settings.json"
            save_json(frozen_cohort, validated.settings)
            contract["cohort"]["path"] = str(frozen_cohort)
            _verify_cohort_snapshot(frozen_cohort, contract["cohort"]["content_id"])
        save_json(root / "study_contract.json", contract)
    main_roster = _freeze_main_roster(root, contract)
    qualification_path = root / "qualification.json"
    if qualification_path.exists():
        existing = read_json(qualification_path)
        cohort_id = contract["cohort"]["content_id"] if contract.get("cohort") else None
        expected_steps = {"mechanisms", "cohort"} if contract.get("cohort") else {"mechanisms"}
        if (existing.get("cohort_content_id") != cohort_id
                or existing.get("protocol_version") != PROTOCOL_VERSION
                or set(existing.get("controls", {})) != expected_steps):
            raise ValueError("Qualification aggregate differs from the frozen study identity")
        for name, receipt in existing["controls"].items():
            expected = dict(step=name, protocol_version=PROTOCOL_VERSION,
                            implementation_version=__version__, cohort_content_id=cohort_id)
            attempt = root / "qualification_steps" / name / receipt["attempt"]
            if (receipt.get("identity") != expected or receipt.get("status") != "passed"
                    or read_json(attempt / "completion.json") != receipt
                    or content_hash(_qualification_summary(attempt / "output", receipt["summary_name"]))
                    != receipt["summary_content_id"]):
                raise ValueError("Qualification aggregate is not backed by matching completed steps")
    if not qualification_path.exists():
        from .biology import generate_history
        from .controls import run_controls
        for family in contract["families"]:
            for variant in range(3):
                generate_history(family, variant+100, n_mutations=60, n_segments=6, n_regions=4).validate()
        binding = dict(cohort_content_id=contract["cohort"]["content_id"] if contract.get("cohort") else None)
        controls = dict(mechanisms=_qualification_step(root, "mechanisms", run_controls,
            summary_name="controls_summary.json", identity=binding, retry_failed=retry_failed))
        if contract.get("cohort"):
            from .controls import run_cohort_controls
            controls["cohort"] = _qualification_step(root, "cohort", lambda output: run_cohort_controls(
                _cohort(contract["cohort"]["path"], contract["cohort"]["content_id"]), output),
                summary_name="cohort_controls_summary.json", identity=binding, retry_failed=retry_failed)
        save_json(qualification_path, dict(physical_histories_validated=36, controls=controls,
            cohort_content_id=binding["cohort_content_id"],
            protocol_version=PROTOCOL_VERSION, source_cohort_protocol_version=contract.get("source_cohort_protocol_version"),
            numerical_policy=f"Every native fit and assignment retains the independent Protocol {PROTOCOL_VERSION} verification receipt"))
    workers = contract["resource_limits"]["workers"]
    development_cells = run_jobs(root, jobs(contract, "development"), workers=workers, contract=contract,
                                 retry_failed=retry_failed)
    assessment = assess_development(root, development_cells, contract, main_roster)
    _publish_development_assessment(root, assessment)
    run_jobs(root, main_roster["jobs"], workers=workers, contract=contract, retry_failed=retry_failed)
    run_optional_panels(root, contract, workers, retry_failed=retry_failed)
    return report(root)


def _reference_audit_destination(output, family, source):
    """Reuse only an audit bound to this source attempt; preserve older audits."""
    root = output / "reference_audits" / family
    for directory in [root, *sorted(root.glob("attempt-*"))]:
        binding = directory / "source_attempt.json"
        if binding.exists() and read_json(binding) == source and (directory / "reference_audit.json").exists():
            return directory, True
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"attempt-{len(list(root.glob('attempt-*')))+1:03d}"
    directory.mkdir()
    save_json(directory / "source_attempt.json", source)
    return directory, False


def run_optional_panels(root, contract, workers, *, retry_failed=False):
    from .acquisition import acquire
    from .controls import run_reference_audit
    from .panels import contributor_density_diagnostics, run_scaling, sensitivity_jobs
    options = contract["optional_panels"]
    if not any(options.get(key) for key in ("external_reference", "technical_sensitivity", "assignment_scaling", "contributor_density")):
        return
    config = resolve_config(contract)
    output = root / "optional_panels"
    output.mkdir(exist_ok=True)
    if options.get("technical_sensitivity"):
        run_jobs(root, sensitivity_jobs(contract, extended=options.get("extended_sensitivity", False)),
                 workers=workers, contract=contract, retry_failed=retry_failed)
    if options.get("external_reference"):
        for family in ("S03", "S04", "S05", "S11", "S12"):
            if contract.get("cohort"):
                candidate = next(job for job in jobs(contract, "main")
                                 if job["family"] == family and job["acquisition"]["id"] == "A4")
                cell_id = candidate["cell_id"]
            else:
                cell_id = f"main-{family}-H001-A4"
            completion = _latest_completion(root / "cells" / cell_id)
            if completion is None:
                continue
            source = dict(cell_id=cell_id, selected_attempt=completion["attempt"],
                          completion_identity=content_hash(completion))
            directory, reused = _reference_audit_destination(output, family, source)
            if reused:
                continue
            job = completion["job"]
            history = _history(job)
            attempt = root / completion["attempt"]
            catalog = load_original_catalog(attempt / "input_catalog.tsv", max_major_cn=config.fit.max_major_cn,
                                              missing_counts=config.missing_counts)
            paired = _paired_result(attempt) if completion.get("cost_ledgers") else {}
            second_catalog = None
            if options.get("independent_counts"):
                if contract.get("cohort"):
                    from .cohort import acquire_cohort
                    second = acquire_cohort(history, _cohort(contract["cohort"]["path"], contract["cohort"]["content_id"]), "A4", read_replicate_id=2)
                else:
                    second = acquire(history, seed_for(contract["master_seed"], "main", family, 0, 104), .8, 500)
                second_path = output / f"{family}_independent_counts.tsv"
                if not second_path.exists():
                    write_tumor_txt(second_path, second, dict(second.attrs["metadata"], tumor_id=catalog.tumor_id))
                second_catalog = load_original_catalog(second_path, max_major_cn=config.fit.max_major_cn,
                                                         missing_counts=config.missing_counts)
            run_reference_audit(history, catalog, paired, config, directory / "output", second_catalog=second_catalog)
            save_json(directory / "reference_audit.json", dict(source_attempt=source,
                result=read_json(directory / "output" / "reference_audit.json")))
    density_path = output / "contributor_density.json"
    if options.get("contributor_density") and not density_path.exists():
        from .biology import generate_history
        history = generate_history("S09", seed_for(contract["master_seed"], "controls", "S09", 0),
                                   n_mutations=1500, n_segments=3, n_regions=4)
        frame = acquire(history, seed_for(contract["master_seed"], "controls", "S09", 0, 4), .8, 500)
        save_json(density_path, contributor_density_diagnostics(history, frame, config))
    if options.get("assignment_scaling") and not (output / "scaling" / "scaling_summary.json").exists():
        run_scaling(output / "scaling", config)


def report(root, *, evaluation_config=None, report_dir=None):
    """Render cached metrics or explicitly reevaluate preserved inference evidence."""
    from .evaluation import (aggregate, normalize_evaluation_config, evaluation_definition,
                             legacy_evaluation_definition, HISTORICAL_V5_REFERENCE_POLICY)
    from .figures import make_figures
    from .runtime import recorded_costs, qualification_costs
    from .reporting import planned_roster, reevaluate_cell, placeholder_for_cell, validate_cached_evaluation
    root = Path(root).resolve()
    contract = read_json(root / "study_contract.json")
    reference_policy = contract.get("evaluation", {}).get("reference_policy", HISTORICAL_V5_REFERENCE_POLICY)
    derive = evaluation_config is not None
    if derive:
        if isinstance(evaluation_config, (str, Path)):
            evaluation_config = read_json(evaluation_config)
        evaluation_config = normalize_evaluation_config(evaluation_config)
        schema = ("clipp2.simulation.evaluation.v5" if contract.get("schema") == "clipp2.simulation-contract.v7" and contract.get("protocol_version") == "2.3" else
                  "clipp2.simulation.evaluation.v4" if contract.get("protocol_version") == "2.3" else "clipp2.simulation.evaluation.v3")
    else:
        frozen_evaluation = contract.get("evaluation", {})
        schema = frozen_evaluation.get("record_schema")
        if schema is None:
            version = tuple(int(part) for part in contract.get("implementation_version", "0.0.0").split(".")[:3])
            schema = "clipp2.simulation.evaluation." + (
                "v3" if version >= (1, 2, 3) else "v2" if version >= (1, 2, 2) else "v1")
        if schema not in ("clipp2.simulation.evaluation.v1", "clipp2.simulation.evaluation.v2",
                          "clipp2.simulation.evaluation.v3", "clipp2.simulation.evaluation.v4", "clipp2.simulation.evaluation.v5"):
            raise ValueError("Frozen study declares an unsupported evaluation record schema")
        evaluation_config = {key: frozen_evaluation[key]
            for key in ("reference_match_atol", "reference_distance") if key in frozen_evaluation}
        if schema.endswith(".v1"):
            evaluation_config["reference_distance"] = "linf"
        evaluation_config = normalize_evaluation_config(evaluation_config)
    definition = (evaluation_definition(evaluation_config, reference_policy=reference_policy) if schema.endswith(".v5") else
                  legacy_evaluation_definition(schema, evaluation_config))
    definition_id = content_hash(definition)
    cells = [value for directory in sorted((root / "cells").glob("*")) if directory.is_dir()
             for value in [_latest_completion(directory)] if value is not None]
    main = [cell for cell in cells if cell["job"]["phase"] == "main"]
    allocation_path = root / "allocation.json"
    allocation = (roster_counts(contract) if "replicate_roster" in contract else
                  read_json(allocation_path) if allocation_path.exists() else {})
    roster = planned_roster(root, contract, allocation, cells)
    completed = {cell["job"]["cell_id"]: cell for cell in main}
    main = [completed.get(job["cell_id"], dict(job=job, status="not_executed", evaluations=[], cost_ledgers=[],
                elapsed_seconds=0., peak_resident_bytes=0, output_bytes=0)) for job in roster["jobs"]]
    evaluations = []
    for cell in main:
        if derive:
            value = reevaluate_cell(root, cell, evaluation_config, evaluation_schema=schema,
                                    reference_policy=reference_policy)
        else:
            try:
                cached_path = root / cell["evaluations"][0]
                value = read_json(cached_path)
            except (OSError, ValueError, IndexError, KeyError):
                value = placeholder_for_cell(root, cell, evaluation_config, evaluation_schema=(
                    schema if schema.endswith((".v4", ".v5")) else "clipp2.simulation.evaluation.v3"))
                # Missing cached records retain the frozen study's definition.
                # Historical files retain their original inference interpretation.
                value.update(schema=schema, evaluation_config=evaluation_config,
                    evaluation_definition=definition, evaluation_definition_id=definition_id,
                    reference_match_atol=definition["reference_match_atol"],
                    reference_distance=definition["reference_distance"])
            else:
                validate_cached_evaluation(value, cell["job"], cached_path)
        evaluations.append(value)
    summary = aggregate(evaluations, planned_roster=roster["jobs"], roster_complete=roster["complete"],
                        expected_definition=definition, families=contract["families"])
    summary.update(roster_status=roster["status"], roster_id=roster["roster_id"],
                   retained_planned_cells=len(roster["jobs"]))
    costs = recorded_costs(root)
    planned = len(roster["jobs"]) if roster["complete"] else allocation.get("planned_main_datasets", len(main))
    summary.update(planned_main_cells=planned, worker_failures=sum(cell["status"] == "execution_failed" for cell in main),
                   unexecuted_main_cells=max(0, planned-sum(cell["status"] != "not_executed" for cell in main)),
                   all_cell_count=len(cells), peak_resident_bytes=max((c["peak_resident_bytes"] for c in cells), default=0),
                   fit_count=costs.get("fit_attempts", 0), assignment_pass_count=costs.get("assignment_attempts", 0),
                   operation_costs=costs, qualification_costs=qualification_costs(root),
                   report_mode="offline_reevaluation" if derive else "cached_metrics",
                   derived_evidence_unavailable=sum(e.get("status") == "unavailable_derived_evidence" for e in evaluations))
    for counts in summary.get("population_counts", []):
        counts["planned"] = planned
    report_dir = (Path(report_dir).resolve() if report_dir is not None else
                  root / "reports" / f"report-{len(list((root / 'reports').glob('report-*')))+1:03d}")
    if report_dir == root or (report_dir.is_relative_to(root) and report_dir.relative_to(root).parts[0] != "reports"):
        raise ValueError("A report inside the source study must use a new directory under reports")
    report_dir.mkdir(parents=True)
    if derive:
        save_json(report_dir / "evaluation_config.json", evaluation_config)
        selection = root / "threshold_development.json"
        save_json(report_dir / "derivation_manifest.json", dict(schema="clipp2.simulation.report-derivation.v1",
            source_study=str(root), source_contract_schema=contract.get("schema"),
            source_implementation_version=contract.get("implementation_version"),
            source_cohort_content_id=contract.get("cohort", {}).get("content_id") if contract.get("cohort") else None,
            historical_cohort_identity_backfilled=False,
            frozen_threshold_development=read_json(selection) if selection.exists() else None,
            development_assessments=[read_json(path) for path in sorted(root.glob("development_assessment*.json"))],
            evaluation_config=evaluation_config, implementation_version=__version__,
            fit_count=0, assignment_pass_count=0, threshold_selection_count=0))
        for index, evaluation in enumerate(evaluations, 1):
            save_json(report_dir / "evaluations" / f"evaluation-{index:05d}.json", evaluation)
    save_json(report_dir / "aggregate.json", summary)
    figures = make_figures(evaluations, summary, report_dir)
    metrics = [dict(cell_id=e["cell_id"], history_id=e["history_id"], family=e["family"],
                    acquisition=e["acquisition_id"], truth_qualification=e.get("truth_qualification"), **row)
               for e in evaluations for row in e["metrics"]]
    pd.DataFrame(metrics, columns=None if metrics else ["cell_id", "history_id", "family", "acquisition",
        "truth_qualification", "name", "arm", "view", "stratum", "value"]).to_csv(
            report_dir / "tumor_metrics.tsv", sep="\t", index=False, na_rep="NA")
    audits = [dict(cell_id=e["cell_id"], **row) for e in evaluations for row in e.get("cap_audit", ())]
    pd.DataFrame(audits, columns=None if audits else ["cell_id", "mutation_id", "region_id"]).to_csv(
        report_dir / "bound_audit.tsv", sep="\t", index=False, na_rep="NA")
    transitions = [dict(cell_id=e["cell_id"], truth_qualification=e.get("truth_qualification"),
                        arm=arm, origin=origin, **row) for e in evaluations
        for arm, strata in e.get("detailed_membership_transitions", {}).items()
        for origin, rows in strata.items() for row in rows]
    pd.DataFrame(transitions, columns=None if transitions else ["cell_id", "truth_qualification", "arm", "origin",
        "baseline_state", "final_state", "count"]).to_csv(
            report_dir / "detailed_transitions.tsv", sep="\t", index=False, na_rep="NA")
    pd.DataFrame([dict(cell_id=c["job"]["cell_id"], phase=c["job"]["phase"], family=c["job"]["family"],
                       status=c["status"], elapsed_seconds=c["elapsed_seconds"],
                       peak_resident_bytes=c["peak_resident_bytes"], output_bytes=c["output_bytes"])
                  for c in cells], columns=["cell_id", "phase", "family", "status", "elapsed_seconds", "peak_resident_bytes", "output_bytes"]).to_csv(report_dir / "execution_ledger.tsv", sep="\t", index=False)
    _write_report(root, report_dir, cells, evaluations, summary, figures)
    print(json.dumps(dict(event="report_complete", path=str(report_dir / "comparison_report.html"))), flush=True)
    return dict(output_dir=str(root), report=str(report_dir / "comparison_report.html"), summary=summary)


def _write_report(root, directory, cells, evaluations, summary, figures):
    import html
    contract = read_json(root / "study_contract.json")
    allocation = (roster_counts(contract) if "replicate_roster" in contract else
                  read_json(root / "allocation.json") if (root / "allocation.json").exists() else {})
    development = read_json(root / "threshold_development.json") if (root / "threshold_development.json").exists() else {}
    selected = development.get("selected")
    assessments = [read_json(path) for path in sorted(root.glob("development_assessment*.json"))]
    if "replicate_roster" in contract:
        allocation_description = "The population was declared by explicit frozen replicate blocks; timing and disk measurements did not select its membership."
    elif allocation.get("resource_reduction_applied") is False or contract.get("profile") == "full":
        allocation_description = "The explicit full-cohort allocation did not use timing or disk measurements to select its population."
    elif allocation.get("measured_seconds"):
        allocation_description = "The balanced allocation was frozen from timing and disk measurements before main generation."
    else:
        allocation_description = "The retained evidence does not establish a timing-based allocation procedure."
    text = ["<!doctype html><meta charset='utf-8'><title>CliPP2 biological simulation 1.2</title>",
        "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;line-height:1.5}img{max-width:100%}pre{white-space:pre-wrap}td,th{padding:6px;border:1px solid #ddd}table{border-collapse:collapse}</style>",
        "<h1>CliPP2 biological simulation 1.2</h1>",
        f"<p>Execution Protocol {html.escape(contract['protocol_version'])}; source cohort Protocol {html.escape(str(contract.get('source_cohort_protocol_version') or 'not specified'))}; source revision {html.escape(contract['source_identity'].get('git_commit') or 'unavailable')}. Source evidence is supplied externally; no source-directory scan is performed. CPU execution; CUDA is not qualified by this study.</p>",
        f"<p>Completed main datasets: {sum(c['status']=='complete' for c in cells if c['job']['phase']=='main')}. Planned main datasets: {allocation.get('planned_main_datasets', 0)}. "
        f"Native fits executed across all phases: {summary['fit_count']}; full-support assignment passes: {summary['assignment_pass_count']}.</p>",
        f"<p>Executed scale: {contract['dimensions']['mutations']} mutations, {contract['dimensions']['segments']} segments, {contract['dimensions']['regions']} regions; "
        f"{allocation.get('main_histories_per_family', 0)} independent main histories per family. The proposed full scale is 1,500 mutations, 50 segments, 50 main and 10 development histories per family.</p>",
        f"<p>{allocation_description} Repeated acquisitions share physical histories; uncertainty resamples histories and carries all paired arms together. A reduced cohort does not establish rare-event safety or noninferiority.</p>",
        "<h2>Frozen population and settings</h2><pre>"+html.escape(json.dumps(dict(replicate_roster=contract.get("replicate_roster"), counts=allocation, historical_selected=selected,
            workflow_config=contract.get("base_workflow_config"), development_assessments=assessments), indent=2))+"</pre>",
        "<h2>Applicability and scientific results</h2><p>Unknown and unavailable estimates remain explicit. Exact aliases and absent candidate explanations cannot be resolved by restoring integer support. Diagnostics are measurements; Protocol 2.3 returns observed MAP estimates without statistical acceptance screens.</p>"]
    text.append("<h2>Evaluation definition and population</h2><pre>"+html.escape(json.dumps(dict(
        definition=summary.get("evaluation_definition"), report_mode=summary["report_mode"],
        roster_status=summary["roster_status"]), indent=2))+"</pre>")
    text.append(pd.DataFrame(summary["population_counts"]).to_html(index=False, na_rep="NA"))
    selected_metrics = {"binary_correct_per_original_evaluable", "binary_correct_resolution", "binary_selective_error",
        "mapped_correct_resolution", "mapped_selective_error", "unmapped_acceptance_fraction",
        "geometric_correct_reference_resolution", "analysis_available", "analysis_failure", "evaluation_available",
        "data_supported_map_coverage", "binary_map_error", "detailed_map_error", "map_ccf_rmse",
        "binary_map_correct_per_original_evaluable", "prior_only_fraction", "model_incompatible_fraction", "computation_failure_count"}
    rows = [dict(endpoint=row["name"], arm=row["arm"], view=row["view"], stratum=row["stratum"],
                 scope=row.get("population_scope"), planned=row.get("planned_population_estimate"),
                 planned_status=row.get("planned_population_status"),
                 available_case=row.get("available_case_estimate"),
                 planned_lower_95=row.get("planned_lower_95"), planned_upper_95=row.get("planned_upper_95"),
                 available_case_lower_95=row.get("available_case_lower_95"), available_case_upper_95=row.get("available_case_upper_95"),
                 available_cells=row["available_cells"], status=row["status"])
            for row in summary.get("metrics", ()) if row["name"] in selected_metrics]
    text.append(pd.DataFrame(rows).to_html(index=False, na_rep="NA", float_format=lambda x: f"{x:.4g}"))
    transitions = [dict(arm=arm, origin=origin, **row) for e in evaluations
        if e.get("truth_qualification") == "verified_history_binding"
        for arm, origins in e.get("detailed_membership_transitions", {}).items()
        for origin, rows in origins.items() for row in rows if row["count"]]
    text.append("<h2>Detailed membership transitions by target origin</h2>"
        "<p>Mc: mapped correct; Mw: mapped wrong; U: selected but unmapped; unresolved: no data-supported final label. "
        "Counts below use verified history bindings only; they do not fill missing planned cells. "
        "Raw TSV exports retain historical diagnostics with their truth qualification.</p>")
    if transitions:
        grouped = pd.DataFrame(transitions).groupby(["arm", "origin", "baseline_state", "final_state"])["count"].sum().reset_index()
        text.append(grouped.to_html(index=False))
    else:
        text.append("<p>Detailed transitions are unavailable.</p>")
    for figure in figures:
        filename = Path(figure).name
        if Path(filename).suffix.lower() == ".png":
            text.append(f"<figure><img src='{html.escape(filename)}'><figcaption>{html.escape(filename)}</figcaption></figure>")
    fixed_total = []
    for cell in cells:
        job = cell.get("job", {})
        condition = job.get("acquisition", {}).get("id", "")
        if (job.get("phase") == "sensitivity" and condition.startswith("regions")
                and condition.endswith("_fixed_total")
                and condition[len("regions"):-len("_fixed_total")].isdigit()):
            fixed_total.append(dict(cell_id=job["cell_id"], family=job["family"], condition=condition,
                outcome={"complete": "completed", "execution_failed": "failed"}.get(cell.get("status"), "unknown")))
    text.append("<h2>Optional execution evidence</h2>")
    if fixed_total:
        counts = {state: sum(row["outcome"] == state for row in fixed_total)
                  for state in ("completed", "failed", "unknown")}
        text.append(f"<p>Optional fixed-total-depth condition outcomes: {counts['completed']} completed, "
            f"{counts['failed']} failed, {counts['unknown']} unknown retained cell records. "
            "Cells without retained completion evidence have unknown execution state. "
            "These descriptions do not aggregate optional-panel scientific results.</p>")
        text.append(pd.DataFrame(fixed_total).to_html(index=False))
    else:
        text.append("<p>No retained completion evidence for optional fixed-total-depth conditions; "
                    "their execution is unknown. Configuration flags alone do not establish execution.</p>")
    text.extend(["<h2>Reproducibility and limits</h2>",
        "<p>Per-cell outputs preserve native A/B/C fits, frozen combined references, B3/C3 assignments, exact cap and contributor records, and every original mutation's status. Physical histories and truth are stored separately from inference inputs. Source snapshots and native numerical receipts are retained; file-byte checksums are intentionally absent.</p>",
        "<p>Core results do not imply a global clustering optimum or patient-population frequencies. Corrected-cap refitting, BAM simulation, and unavailable historical generators were not executed.</p>",
        "<p>Machine-readable results: <a href='aggregate.json'>aggregate.json</a>; <a href='tumor_metrics.tsv'>tumor metrics</a>; <a href='detailed_transitions.tsv'>detailed transitions</a>; <a href='bound_audit.tsv'>cap audit</a>; <a href='execution_ledger.tsv'>execution ledger</a>.</p>"])
    (directory / "comparison_report.html").write_text("\n".join(text), encoding="utf-8")
