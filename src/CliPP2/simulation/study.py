"""Resumable study execution with history-level pairing and frozen development."""
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

from CliPP2.assignment import posterior_passes
from CliPP2.config import PROTOCOL_VERSION
from CliPP2._version import __version__
from CliPP2.io.data import content_hash, json_value
from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt

from .design import (CONTRIBUTOR_GRID, POSTERIOR_GRID, allocate, jobs, resolve_config,
                     seed_for, workflow_config)


def save_json(path, payload):
    """New-file publication: interrupted work never masquerades as completion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(json_value(payload), stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _decision(row, posterior):
    if row is None or row.get("best_label") is None or not row.get("informative_regions"):
        return None
    if row.get("cluster_tie"):
        return None
    if not posterior_passes(float(row["log_competitor_ratio"]), posterior,
                             row["has_finite_alternative"]):
        return None
    if not row.get("reference_quality_pass", False) or row.get("fit_screen_status") != "pass":
        return None
    return int(row["best_label"])


def development_scores(history, catalog, experiment, evaluation, job, contributor, *, posterior_grid=POSTERIOR_GRID,
                       fixed_families=("S01", "S02", "S07")):
    """Fixed original population; failed selection/fits receive zero credit."""
    from .evaluation import information_diagnostics
    observed = catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)
    # Biological truth, not perturbed supplied inputs, defines the tuning stratum.
    acquisition = job["acquisition"]
    info = information_diagnostics(history.phi, history.multiplicity,
        history.major_cn, history.minor_cn, np.broadcast_to(acquisition["purity"], history.phi.shape),
        np.full(history.phi.shape, 2.), catalog.total_counts, observed,
        history.observable_truth_ids)
    fixed = observed.any(axis=1) & np.array([row["information_stratum"] in
        ("separated", "not_applicable_no_competing_truth") for row in info])
    fixed &= job["family"] in fixed_families
    truth_clonal = np.all(np.isclose(history.phi, 1., atol=1e-12, rtol=0), axis=1)
    arm = experiment.get("arms", {}).get("C", {})
    if arm.get("status") not in ("complete", "empty_remainder"):
        arm = {}  # An unavailable analysis receives zero credit on the fixed population.
    membership = arm.get("combined_labels", {})
    targets, unchanged = set(arm.get("targets", ())), set(arm.get("non_targets", ()))
    records = {row["mutation_id"]: row for row in (arm.get("assignment") or {}).get("assignments", ())}
    rows = []
    for posterior in posterior_grid:
        correct_f = correct_t = wrong_t = accepted = evaluable_targets = 0
        for i, mutation in enumerate(history.mutation_ids):
            if not fixed[i]:
                continue
            if mutation in unchanged and mutation in membership:
                correct_f += int((membership[mutation] == 0) == truth_clonal[i])
            if mutation in targets:
                evaluable_targets += 1
                selected = _decision(records.get(mutation), posterior)
                if selected is not None:
                    accepted += 1
                    correct = (selected == 0) == truth_clonal[i]
                    correct_t += int(correct)
                    wrong_t += int(not correct)
        rows.append(dict(cell_id=job["cell_id"], history_id=job["history_id"], family=job["family"],
            acquisition=acquisition["id"], contributor=contributor, posterior=posterior,
            configuration_id=f"c{contributor:.2f}-p{posterior:.2f}", fixed_evaluable=int(fixed.sum()),
            correct_unchanged=correct_f, correct_targets=correct_t, accepted_targets=accepted,
            wrong_targets=wrong_t, evaluable_targets=evaluable_targets,
            experiment_status=experiment.get("status"), arm_status=arm.get("status")))
    return rows


def _experiment_execution_failures(experiment, configuration_index):
    """Distinguish caught execution errors from completed scientific rejection."""
    failures = []
    for arm, record in ((None, experiment), *experiment.get("arms", {}).items()):
        if record.get("status") not in ("failed", "execution_failed"):
            continue
        failures.append(dict(configuration_index=configuration_index, arm=arm,
            failed_stage=record.get("failed_stage"), failed_phase=record.get("failed_phase"),
            error_type=record.get("error_type"), error=record.get("error")))
    return failures


def _run_cell(task):
    from .evaluation import evaluate
    from .experiment import run_paired_experiment
    root, job, selected = Path(task["root"]), task["job"], task.get("selected")
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
    objectives, results, costs, execution_failures = [], [], [], []
    original_evaluable_count, execution_phase = None, "history"
    try:
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
        catalog = load_original_catalog(input_path)
        truth_link = validate_history_acquisition_link(history, frame.attrs["acquisition"], catalog,
                                                       history_manifest=history_manifest)
        if truth_link["status"] != "verified_history_binding":
            raise ValueError("Live history/acquisition linkage is not verified: " + str(truth_link))
        original_evaluable_count = int((np.isfinite(history.phi).all(axis=1)
            & (catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)).any(axis=1)).sum())
        execution_phase = "before_inference"
        from CliPP2.config import WorkflowConfig
        configurations = task.get("execution_configs") or [asdict(workflow_config(c, p)) for c, p in
            ([(c, .80) for c in CONTRIBUTOR_GRID] if job["phase"] == "development"
             else [(selected["contributor"], selected["posterior"])])]
        first = second = None
        for index, values in enumerate(configurations):
            config = WorkflowConfig.from_dict(values)
            contributor = config.contributor_probability_min
            def progress(event):
                fields = ("phase", "stage", "elapsed_seconds", "phase_elapsed_seconds", "occupied_q", "membership_audit", "operation")
                payload = dict(configuration=index+1, **{key: event[key] for key in fields if key in event})
                temporary = attempt / ".progress.json.tmp"
                temporary.write_text(json.dumps(json_value(payload), allow_nan=False))
                os.replace(temporary, attempt / "progress.json")
            execution_phase = "paired_inference"
            experiment = run_paired_experiment(input_path, attempt / f"configuration-{index+1}", config,
                accepted_stage_one=first, accepted_uncapped_stage_two=second, progress_callback=progress,
                run_manifest_id=task.get("run_manifest_id"))
            costs.append(experiment["cost_ledger"])
            execution_failures.extend(_experiment_execution_failures(experiment, index+1))
            save_json(attempt / f"cost-ledger-{index+1}.json", experiment["cost_ledger"])
            first = first or experiment.get("stage_one")
            second = second or experiment.get("arms", {}).get("B", {}).get("stage_two")
            execution_phase = "evaluation"
            evaluation = evaluate(history, catalog, experiment, acquisition=frame.attrs.get("acquisition"),
                                  evaluation_config=evaluation_config, truth_qualification=truth_link["status"])
            evaluation.update(cell_id=job["cell_id"], history_id=job["history_id"], family=job["family"],
                              acquisition_id=acquisition["id"], acquisition=acquisition, phase=job["phase"], job=job)
            save_json(attempt / f"evaluation-{index+1}.json", evaluation)
            results.append(str((attempt / f"evaluation-{index+1}.json").relative_to(root)))
            if job["phase"] == "development":
                objectives.extend(development_scores(history, catalog, experiment, evaluation, job, contributor,
                    posterior_grid=task.get("posterior_grid", POSTERIOR_GRID),
                    fixed_families=task.get("fixed_families", ("S01", "S02", "S07"))))
                save_json(attempt / f"threshold-objectives-{index+1}.json", objectives)
        if objectives:
            save_json(attempt / "threshold_objectives.json", objectives)
        if task.get("archive_completed", False):
            _archive_completed(attempt)
        value = dict(job=job, status="execution_failed" if execution_failures else "complete", evaluations=results, cost_ledgers=costs,
            objective_path=str((attempt / "threshold_objectives.json").relative_to(root)) if objectives else None,
            attempt=str(attempt.relative_to(root)), elapsed_seconds=time.perf_counter()-started,
            peak_resident_bytes=_peak_bytes())
    except Exception as exc:
        if objectives and not (attempt / "threshold_objectives.json").exists():
            save_json(attempt / "threshold_objectives.json", objectives)
        value = dict(job=job, status="execution_failed", error_type=type(exc).__name__, error=str(exc),
            traceback=traceback.format_exc(), evaluations=results, cost_ledgers=costs,
            objective_path=str((attempt / "threshold_objectives.json").relative_to(root)) if objectives else None,
            attempt=str(attempt.relative_to(root)), elapsed_seconds=time.perf_counter()-started,
            peak_resident_bytes=_peak_bytes())
        execution_failures.append(dict(configuration_index=None, arm=None, failed_stage="worker",
            error_type=type(exc).__name__, error=str(exc)))
    value["execution_failures"] = execution_failures
    value["original_evaluable_count"] = original_evaluable_count
    value["execution_phase"] = execution_phase
    value["execution_configs"] = task.get("execution_configs")
    value["evaluation_config"] = evaluation_config
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


def run_jobs(root, job_list, *, workers, selected=None, contract=None, retry_failed=False):
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
        settings = contract["threshold_development"] if contract else {}
        grid = settings.get("contributor_grid", CONTRIBUTOR_GRID)
        posterior_grid = settings.get("posterior_grid", POSTERIOR_GRID)
        values = [(c, min(posterior_grid)) for c in grid] if job["phase"] == "development" else [(selected["contributor"], selected["posterior"])]
        resolved = []
        for contributor, posterior in values:
            resolved.append(asdict(resolve_config(contract, contributor, posterior)) if contract else
                            asdict(workflow_config(contributor, posterior)))
        tasks.append(dict(root=str(root), job=job, selected=selected, execution_configs=resolved,
                          posterior_grid=list(posterior_grid),
                          retry_failed=retry_failed,
                          archive_completed=contract["resource_limits"].get("archive_completed_outputs", False) if contract else False,
                          reference_match_atol=contract.get("evaluation", {}).get("reference_match_atol") if contract else None,
                          reference_distance=contract.get("evaluation", {}).get("reference_distance", "l2") if contract else "l2",
                          run_manifest_id=contract.get("run_manifest_id") if contract else None,
                          fixed_families=settings.get("fixed_population_families", ["S01", "S02", "S07"])))
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


def select_thresholds(root, cells, contract):
    settings = contract["threshold_development"]
    rows = [row for cell in cells if cell.get("objective_path")
            for row in read_json(root / cell["objective_path"])]
    # A failed worker still belongs to its original prespecified population.
    # Recover that denominator from the untouched biological input and give zero
    # credit for all thresholds, rather than dropping its history or family.
    expected = {(c, p) for c in settings["contributor_grid"] for p in settings["posterior_grid"]}
    for cell in cells:
        available = {(row["contributor"], row["posterior"]) for row in rows if row["cell_id"] == cell["job"]["cell_id"]}
        missing = expected-available
        if not missing:
            continue
        job = cell["job"]
        h = _history(job)
        frame = _acquire(job, h)
        temporary = root / cell["attempt"] / "failed_population.tsv"
        if not temporary.exists():
            write_tumor_txt(temporary, frame, frame.attrs.get("metadata"))
        catalog = load_original_catalog(temporary)
        for contributor in settings["contributor_grid"]:
            rows.extend(row for row in development_scores(h, catalog, {}, {}, job, contributor,
                posterior_grid=settings["posterior_grid"], fixed_families=settings["fixed_population_families"])
                if (row["contributor"], row["posterior"]) in missing)
    frame = pd.DataFrame(rows)
    candidates = []
    for configuration, group in frame.groupby("configuration_id"):
        usable = group[group.fixed_evaluable > 0].copy()
        for field in ("correct_unchanged", "correct_targets", "accepted_targets", "wrong_targets"):
            usable[field + "_rate"] = usable[field] / usable.fixed_evaluable
        rates = [key + "_rate" for key in ("correct_unchanged", "correct_targets", "accepted_targets", "wrong_targets")]
        per_history = usable.groupby(["family", "history_id"])[rates].mean()
        per_family = per_history.groupby("family").mean()
        means = per_family.mean()
        acceptance = float(means.get("accepted_targets_rate", 0.))
        error = float(means.get("wrong_targets_rate", 0.)) / acceptance if acceptance else None
        first = group.iloc[0]
        candidates.append(dict(configuration_id=configuration, contributor=float(first.contributor),
            posterior=float(first.posterior), objective=float(means.get("correct_unchanged_rate", 0.)+
                                                            means.get("correct_targets_rate", 0.)),
            correct_unchanged_rate=float(means.get("correct_unchanged_rate", 0.)),
            correct_target_rate=float(means.get("correct_targets_rate", 0.)),
            target_wrong_acceptance=float(means.get("wrong_targets_rate", 0.)),
            target_acceptance=acceptance, selective_error=error,
            qualifying=bool(acceptance > 0 and error <= settings["selective_error_limit"]
                            and set(per_family.index) == set(settings["fixed_population_families"])),
            represented_families=list(per_family.index), fixed_evaluable=int(group.fixed_evaluable.sum())))
    qualified = [row for row in candidates if row["qualifying"]]
    selected = min(qualified, key=lambda row: (-row["objective"], row["target_wrong_acceptance"],
        -row["posterior"], -row["contributor"], row["configuration_id"])) if qualified else None
    return dict(schema="clipp2.simulation-development.v1", selected=selected,
                endpoint="binary_clonality", constraint_scope="empirical_development_not_population_error_guarantee",
                selective_error_definition="wrong_accepted_binary_clonality_over_accepted_targets",
                configurations=candidates, cell_objectives=rows, tuning_target="C3", B3_separately_optimized=False)


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


def _freeze_main_roster(root, contract, allocation):
    """Freeze exact main-cell weights before development or main submission."""
    count = allocation.get("main_replicates_per_variant") or allocation["main_histories_per_family"]
    payload = dict(schema="clipp2.simulation-roster.v1", families=contract["families"],
                   jobs=list(jobs(contract, "main", count)))
    if len(payload["jobs"]) != allocation["planned_main_datasets"]:
        raise ValueError("Main roster size differs from frozen allocation")
    record = dict(payload, roster_id=content_hash(payload))
    path = root / "main_roster.json"
    if path.exists():
        if read_json(path) != record:
            raise ValueError("Main roster differs from frozen allocation")
    else:
        save_json(path, record)
    return record["jobs"]


def _execute(root, contract, *, resume=False, retry_failed=False):
    root = Path(root).resolve()
    if (contract.get("protocol_version") != PROTOCOL_VERSION or contract.get("schema") != "clipp2.simulation-contract.v4"
            or contract.get("implementation_version") != __version__):
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
    allocation_path = root / "allocation.json"
    if allocation_path.exists():
        allocation = read_json(allocation_path)
    elif contract["profile"] == "full":
        allocation = allocate(contract, [1.], [1])
        allocation.update(timing_seconds_per_dataset=None, measured_seconds=[], measured_bytes=[],
            dataset_capacity=None, resource_reduction_applied=False,
            reason="Entire supplied cohort explicitly requested; fixed allocation, no timing-based reduction",
            pilot_policy="Bounded resource qualification is separate; no extra optimizer fits for allocation")
        save_json(allocation_path, allocation)
    else:
        pilot = []
        if contract.get("cohort"):
            wanted = {"S01.01", "S02.05", "S04.01", "S06.05", "S09.05"}
            for job in jobs(contract, "development", 1):
                if job["variant_id"] in wanted and job["acquisition"]["id"] == "A4":
                    pilot.append(dict(job, phase="pilot", cell_id="pilot."+job["cell_id"],
                                      source_history_id=job["history_id"]))
        else:
            for family in ("S01", "S02", "S04", "S06", "S09"):
                pilot.append(dict(cell_id=f"pilot-{family}", phase="pilot", family=family,
                    history_id=f"pilot-{family}", history_index=0,
                    history_seed=seed_for(contract["master_seed"], "pilot", family, 0),
                    acquisition_seed=seed_for(contract["master_seed"], "pilot", family, 0, 4),
                    acquisition=contract["acquisitions"][-1], dimensions=contract["dimensions"]))
        cells = run_jobs(root, pilot, workers=workers, selected=dict(contributor=.95, posterior=.95),
                         contract=contract, retry_failed=retry_failed)
        if any(cell["status"] != "complete" for cell in cells):
            raise RuntimeError("Timing pilot execution failed; inspect immutable cell failure records")
        projected = [cell["elapsed_seconds"]*3/max(1, cell["cost_ledgers"][0]["fit_count"]) for cell in cells]
        allocation = allocate(contract, projected,
                              [cell["output_bytes"] for cell in cells])
        allocation["pilot_raw_elapsed_seconds"] = [cell["elapsed_seconds"] for cell in cells]
        allocation["pilot_cost_ledgers"] = [cell["cost_ledgers"] for cell in cells]
        allocation["timing_projection"] = "Scale each pilot to three fits when a valid branch skipped fits; then apply the frozen safety factor"
        save_json(allocation_path, allocation)
        print(json.dumps(dict(event="allocation_frozen", allocation=allocation)), flush=True)
    main_jobs = _freeze_main_roster(root, contract, allocation)
    development_path = root / "threshold_development.json"
    if development_path.exists():
        development = read_json(development_path)
    else:
        count = allocation.get("development_replicates_per_variant") or allocation["development_histories_per_family"]
        development_cells = run_jobs(root, jobs(contract, "development", count), workers=workers, contract=contract,
                                     retry_failed=retry_failed)
        development = select_thresholds(root, development_cells, contract)
        if development["selected"] is None and (not contract.get("cohort") or count < 2):
            print(json.dumps(dict(event="development_extension", reason="No threshold met the frozen criterion")), flush=True)
            development_cells += run_jobs(root, jobs(contract, "development", count, start=count), workers=workers, contract=contract,
                                          retry_failed=retry_failed)
            development = select_thresholds(root, development_cells, contract)
        save_json(development_path, development)
    if development["selected"] is None:
        report(root)
        raise RuntimeError("No development setting qualified after the prespecified extension; final cohort not generated")
    plan_path = root / "study_plan.json"
    if not plan_path.exists():
        save_json(plan_path, dict(contract=contract, allocation=allocation,
            selected_thresholds=development["selected"],
            workflow_config=asdict(resolve_config(contract, development["selected"]["contributor"],
                                                 development["selected"]["posterior"])),
            frozen_before_main_generation=True))
    run_jobs(root, main_jobs,
             workers=workers, selected=development["selected"], contract=contract, retry_failed=retry_failed)
    run_optional_panels(root, contract, development["selected"], workers, retry_failed=retry_failed)
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


def run_optional_panels(root, contract, selected, workers, *, retry_failed=False):
    from .acquisition import acquire
    from .controls import run_reference_audit
    from .panels import contributor_density_diagnostics, run_scaling, sensitivity_jobs
    options = contract["optional_panels"]
    if not any(options.get(key) for key in ("external_reference", "technical_sensitivity", "assignment_scaling", "contributor_density")):
        return
    config = resolve_config(contract, selected["contributor"], selected["posterior"])
    output = root / "optional_panels"
    output.mkdir(exist_ok=True)
    if options.get("technical_sensitivity"):
        run_jobs(root, sensitivity_jobs(contract, extended=options.get("extended_sensitivity", False)),
                 workers=workers, selected=selected, contract=contract, retry_failed=retry_failed)
    if options.get("external_reference"):
        for family in ("S03", "S04", "S05", "S11", "S12"):
            if contract.get("cohort"):
                candidate = next(job for job in jobs(contract, "main", 1)
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
            catalog = load_original_catalog(attempt / "input_catalog.tsv")
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
                second_catalog = load_original_catalog(second_path)
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
    from .evaluation import aggregate, normalize_evaluation_config, evaluation_definition, legacy_evaluation_definition
    from .figures import make_figures
    from .runtime import recorded_costs, qualification_costs
    from .reporting import planned_roster, reevaluate_cell, placeholder_for_cell
    root = Path(root).resolve()
    contract = read_json(root / "study_contract.json")
    derive = evaluation_config is not None
    if derive:
        if isinstance(evaluation_config, (str, Path)):
            evaluation_config = read_json(evaluation_config)
        evaluation_config = normalize_evaluation_config(evaluation_config)
        schema = "clipp2.simulation.evaluation.v3"
    else:
        frozen_evaluation = contract.get("evaluation", {})
        schema = frozen_evaluation.get("record_schema")
        if schema is None:
            version = tuple(int(part) for part in contract.get("implementation_version", "0.0.0").split(".")[:3])
            schema = "clipp2.simulation.evaluation." + (
                "v3" if version >= (1, 2, 3) else "v2" if version >= (1, 2, 2) else "v1")
        if schema not in ("clipp2.simulation.evaluation.v1", "clipp2.simulation.evaluation.v2",
                          "clipp2.simulation.evaluation.v3"):
            raise ValueError("Frozen study declares an unsupported evaluation record schema")
        evaluation_config = {key: frozen_evaluation[key]
            for key in ("reference_match_atol", "reference_distance") if key in frozen_evaluation}
        if schema.endswith(".v1"):
            evaluation_config["reference_distance"] = "linf"
        evaluation_config = normalize_evaluation_config(evaluation_config)
    definition = (evaluation_definition(evaluation_config) if schema.endswith(".v3") else
                  legacy_evaluation_definition(schema, evaluation_config))
    definition_id = content_hash(definition)
    cells = [value for directory in sorted((root / "cells").glob("*")) if directory.is_dir()
             for value in [_latest_completion(directory)] if value is not None]
    main = [cell for cell in cells if cell["job"]["phase"] == "main"]
    allocation_path = root / "allocation.json"
    allocation = read_json(allocation_path) if allocation_path.exists() else {}
    roster = planned_roster(root, contract, allocation, cells)
    completed = {cell["job"]["cell_id"]: cell for cell in main}
    main = [completed.get(job["cell_id"], dict(job=job, status="not_executed", evaluations=[], cost_ledgers=[],
                elapsed_seconds=0., peak_resident_bytes=0, output_bytes=0)) for job in roster["jobs"]]
    evaluations = []
    for cell in main:
        if derive:
            value = reevaluate_cell(root, cell, evaluation_config)
        else:
            try:
                value = read_json(root / cell["evaluations"][0])
            except (OSError, ValueError, IndexError, KeyError):
                value = placeholder_for_cell(root, cell, evaluation_config)
                # Missing cached records retain the frozen study's definition.
                # Historical files are never relabeled as newly evaluated v3.
                value.update(schema=schema, evaluation_config=evaluation_config,
                    evaluation_definition=definition, evaluation_definition_id=definition_id,
                    reference_match_atol=definition["reference_match_atol"],
                    reference_distance=definition["reference_distance"])
        evaluations.append(value)
    summary = aggregate(evaluations, planned_roster=roster["jobs"], roster_complete=roster["complete"],
                        expected_definition=definition, families=contract["families"])
    summary.update(roster_status=roster["status"], roster_id=roster["roster_id"],
                   retained_planned_cells=len(roster["jobs"]))
    costs = recorded_costs(root)
    planned = allocation.get("planned_main_datasets", len(main))
    summary.update(planned_main_cells=planned, worker_failures=sum(cell["status"] == "execution_failed" for cell in main),
                   unexecuted_main_cells=max(0, planned-sum(cell["status"] != "not_executed" for cell in main)),
                   all_cell_count=len(cells), peak_resident_bytes=max((c["peak_resident_bytes"] for c in cells), default=0),
                   fit_count=costs.get("fit_attempts", 0), assignment_pass_count=costs.get("assignment_attempts", 0),
                   operation_costs=costs, qualification_costs=qualification_costs(root),
                   report_mode="offline_reevaluation" if derive else "cached_metrics",
                   derived_evidence_unavailable=sum(e.get("status") == "unavailable_derived_evidence" for e in evaluations))
    summary["population_counts"] = [dict(arm=arm, planned=planned,
        executed=sum(row.get("execution_available") == 1
            for row in summary.get("availability", []) if row["arm"] == arm),
        scientifically_applicable=sum(row.get("scientifically_applicable") == 1
            for row in summary.get("availability", []) if row["arm"] == arm),
        evaluated=sum(row.get("evaluation_available") == 1
            for row in summary.get("availability", []) if row["arm"] == arm),
        unavailable_derived=sum(e.get("status") == "unavailable_derived_evidence" for e in evaluations))
        for arm in ("A", "B", "C", "B3", "C3")]
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
    allocation = read_json(root / "allocation.json") if (root / "allocation.json").exists() else {}
    development = read_json(root / "threshold_development.json") if (root / "threshold_development.json").exists() else {}
    selected = development.get("selected")
    text = ["<!doctype html><meta charset='utf-8'><title>CliPP2 biological simulation 1.2</title>",
        "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;line-height:1.5}img{max-width:100%}pre{white-space:pre-wrap}td,th{padding:6px;border:1px solid #ddd}table{border-collapse:collapse}</style>",
        "<h1>CliPP2 biological simulation 1.2</h1>",
        f"<p>Execution Protocol {html.escape(contract['protocol_version'])}; source cohort Protocol {html.escape(str(contract.get('source_cohort_protocol_version') or 'not specified'))}; source revision {html.escape(contract['source_identity'].get('git_commit') or 'unavailable')}. Source evidence is supplied externally; no source-directory scan is performed. CPU execution; CUDA is not qualified by this study.</p>",
        f"<p>Completed main datasets: {sum(c['status']=='complete' for c in cells if c['job']['phase']=='main')}. Planned main datasets: {allocation.get('planned_main_datasets', 0)}. "
        f"Native fits executed across all phases: {summary['fit_count']}; full-support assignment passes: {summary['assignment_pass_count']}.</p>",
        f"<p>Executed scale: {contract['dimensions']['mutations']} mutations, {contract['dimensions']['segments']} segments, {contract['dimensions']['regions']} regions; "
        f"{allocation.get('main_histories_per_family', 0)} independent main histories per family. The proposed full scale is 1,500 mutations, 50 segments, 50 main and 10 development histories per family.</p>",
        "<p>The allocation below was frozen from the timing and disk pilot before final generation. Repeated acquisitions share physical histories; uncertainty resamples histories and carries all paired arms together. A reduced cohort does not establish rare-event safety or noninferiority.</p>",
        "<h2>Frozen allocation and settings</h2><pre>"+html.escape(json.dumps(dict(allocation=allocation, selected=selected), indent=2))+"</pre>",
        "<h2>Applicability and scientific results</h2><p>Unknown and unavailable estimates remain explicit. Exact aliases and absent candidate explanations cannot be resolved by restoring integer support. Descriptive screens are not calibrated tests.</p>"]
    text.append("<h2>Evaluation definition and population</h2><pre>"+html.escape(json.dumps(dict(
        definition=summary.get("evaluation_definition"), report_mode=summary["report_mode"],
        roster_status=summary["roster_status"]), indent=2))+"</pre>")
    text.append(pd.DataFrame(summary["population_counts"]).to_html(index=False, na_rep="NA"))
    selected_metrics = {"binary_correct_per_original_evaluable", "binary_correct_resolution", "binary_selective_error",
        "mapped_correct_resolution", "mapped_selective_error", "unmapped_acceptance_fraction",
        "geometric_correct_reference_resolution", "analysis_available", "analysis_failure", "evaluation_available"}
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
        "<p>Mc: mapped correct; Mw: mapped wrong; U: accepted but unmapped; unresolved: no accepted label. "
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
