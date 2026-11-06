"""Fast study regressions with an eight-row native run and archived retry."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import shutil
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt
from CliPP2.simulation.biology import generate_history
from CliPP2.simulation.design import allocate, jobs, make_contract, workflow_config
from CliPP2.simulation.runtime import qualification_costs, recorded_costs, status
from CliPP2.simulation.study import (_decision, _execute, _paired_result, _prepare_histories, _qualification_step,
                                    _run_cell, development_scores, read_json, report,
                                    save_json, select_thresholds)


def _fixture():
    history = generate_history("S01", 300, n_mutations=8, n_segments=2, n_regions=2)
    values = [[50, 50]]*4 + [[20, 15]]*4
    frame = pd.DataFrame([dict(mutation_id=mutation, sample_id=region,
        alt_count=values[i][j], ref_count=100-values[i][j], count_observed=1,
        purity=1., normal_cn=2, normal_cn_supplied=1, segment_id="s0", mapping_genuine=1,
        cn_state_id="0", cn_state_fraction=1., allele_a_cn=1, allele_b_cn=1)
        for i, mutation in enumerate(history.mutation_ids) for j, region in enumerate(history.region_ids)])
    frame.attrs["metadata"] = dict(tumor_id="orchestration_regression")
    frame.attrs["acquisition"] = dict(count_observed=np.ones((8, 2), bool))
    job = dict(cell_id="main-S01-orchestration", phase="main", family="S01",
        history_id="main-S01-orchestration", history_seed=300, acquisition_seed=400,
        acquisition=dict(id="A4", purity=1., depth=100), dimensions=dict(mutations=8, segments=2, regions=2))
    return history, frame, job


def _fixed_information(*args, **kwargs):
    return [dict(information_stratum=value) for value in
            ["not_applicable_no_competing_truth", *(["separated"]*5), "exact_alias", "exact_alias"]]


def _design_check(root, settings):
    snapshot = root / "cohort_settings.json"
    shutil.copyfile(settings, snapshot)
    contract = make_contract(profile="full", cohort_settings=snapshot)
    assert contract["protocol_version"] == "2.2" and contract["source_cohort_protocol_version"] == "2.1"
    assert contract["schema"] == "clipp2.simulation-contract.v4"
    assert contract["evaluation"]["reference_match_atol"] is None
    assert json.loads(snapshot.read_text())["workflow_protocol"] == "2.1"
    main, development = list(jobs(contract, "main", 10)), list(jobs(contract, "development", 2))
    assert len(main) == 2400 and len(development) == 240
    assert len({row["cell_id"] for row in main + development}) == 2640
    main_histories = {row["history_id"] for row in main}
    dev_histories = {row["history_id"] for row in development}
    assert len(main_histories) == 600 and len(dev_histories) == 120 and main_histories.isdisjoint(dev_histories)
    assert len({row["variant_id"] for row in main}) == len({row["variant_id"] for row in development}) == 60
    assert all(row["cohort_content_id"] == contract["cohort"]["content_id"] for row in main+development)
    for population, expected in ((main, {"A1", "A2", "A3", "A4"}), (development, {"A1", "A4"})):
        for history_id in {row["history_id"] for row in population}:
            entries = [row for row in population if row["history_id"] == history_id]
            assert {row["acquisition"]["id"] for row in entries} == expected
            assert len({row["history_seed"] for row in entries}) == 1
            assert all(row["acquisition"] == dict(id="A4", purity=.8, depth=500)
                       for row in entries if row["acquisition"]["id"] == "A4")
    allocation = allocate(contract, [1.], [1])
    assert allocation["planned_main_datasets"] + allocation["planned_development_datasets"] == 2640
    assert allocation["planned_core_max_fits"] == 7920 and allocation["planned_core_max_assignment_passes"] == 5280
    config = contract["base_workflow_config"]
    assert (config["fit"]["max_clusters"], config["stage_two_max_clusters"], config["clonal_distance_max"]) == (10, 20, .05)
    return contract


def _failed_credit_check(root, history, frame, job):
    path = root / "fixture.tsv"
    write_tumor_txt(path, frame, frame.attrs["metadata"])
    catalog = load_original_catalog(path)
    non_targets = [m for m, phi in zip(history.mutation_ids, history.phi) if not np.all(phi == 1)]
    frozen = dict(combined_labels={m: 1 for m in non_targets},
                  targets=[], non_targets=non_targets, assignment=None)
    with patch("CliPP2.simulation.evaluation.information_diagnostics", side_effect=_fixed_information):
        success = development_scores(history, catalog, dict(arms={"C": dict(frozen, status="complete")}),
                                     {}, job, .95, posterior_grid=[.8, .99])
        failure = development_scores(history, catalog, dict(status="partial_failure",
            arms={"C": dict(frozen, status="failed")}), {}, job, .95, posterior_grid=[.8, .99])
        unavailable = development_scores(history, catalog, {}, {}, job, .95, posterior_grid=[.8, .99])
    assert all(row["fixed_evaluable"] == 6 for row in success + failure + unavailable)
    expected_correct = sum(m in non_targets for m in history.mutation_ids[:6])
    assert expected_correct > 0 and all(row["correct_unchanged"] == expected_correct for row in success)
    assert all(row["correct_unchanged"] == row["correct_targets"] == row["accepted_targets"] == 0
               for row in failure + unavailable)


def _selection_check(root, contract, history, frame, job):
    cells = []
    policies = [(.90, .80, "pooled_favorite"), (.95, .90, "balanced"), (.95, .99, "balanced"),
                (.99, .99, "balanced"), (.90, .99, "too_many_errors"), (.95, .95, "no_acceptance")]
    for family in ("S01", "S02", "S07"):
        population = 1000 if family == "S01" else 10
        rows = []
        configured = {(c, p): policy for c, p, policy in policies}
        grid = contract["threshold_development"]
        for contributor, posterior in ((c, p) for c in grid["contributor_grid"] for p in grid["posterior_grid"]):
            policy = configured.get((contributor, posterior), "unavailable")
            if policy == "pooled_favorite":
                correct, wrong, accepted, unchanged = (population if family == "S01" else 0), 0, (population if family == "S01" else 0), 0
            elif policy == "balanced":
                correct, wrong, accepted, unchanged = int(.6*population), 0, int(.6*population), 0
            elif policy == "too_many_errors":
                correct, wrong, accepted, unchanged = int(.9*population), int(.1*population), population, 0
            elif policy == "no_acceptance":
                correct, wrong, accepted, unchanged = 0, 0, 0, int(.9*population)
            else:
                correct = wrong = accepted = unchanged = 0
            rows.append(dict(cell_id=family+"-good", history_id=family+"-good", family=family,
                acquisition="A4", contributor=contributor, posterior=posterior,
                configuration_id=f"c{contributor:.2f}-p{posterior:.2f}", fixed_evaluable=population,
                correct_unchanged=unchanged, correct_targets=correct, accepted_targets=accepted,
                wrong_targets=wrong, evaluable_targets=population, experiment_status="complete", arm_status="complete"))
        path = root / (family + "_objectives.json")
        save_json(path, rows)
        cells.append(dict(objective_path=path.name, job=dict(job, phase="development", family=family,
                          cell_id=family+"-good", history_id=family+"-good"), attempt=family+"_attempt"))
    failure_job = dict(job, phase="development", cell_id="S01-failed", history_id="S01-failed")
    (root / "failed_attempt").mkdir()
    cells.append(dict(job=failure_job, objective_path=None, attempt="failed_attempt", status="execution_failed"))
    with patch("CliPP2.simulation.study._history", return_value=history), \
            patch("CliPP2.simulation.study._acquire", return_value=frame), \
            patch("CliPP2.simulation.evaluation.information_diagnostics", side_effect=_fixed_information):
        selection = select_thresholds(root, cells, contract)
    assert selection["selected"]["configuration_id"] == "c0.99-p0.99"
    assert np.isclose(selection["selected"]["objective"], .5)
    candidates = {row["configuration_id"]: row for row in selection["configurations"]}
    assert np.isclose(candidates["c0.90-p0.80"]["objective"], 1/6)
    assert not candidates["c0.90-p0.99"]["qualifying"] and not candidates["c0.95-p0.95"]["qualifying"]
    failures = [row for row in selection["cell_objectives"] if row["cell_id"] == "S01-failed"]
    assert len(failures) == 12 and all(row["fixed_evaluable"] == 6 for row in failures)
    save_json(root / "selection_check.json", selection)


def _native_failure_check(root, history, frame, job, contract):
    save_json(root / "study_contract.json", contract)
    config = asdict(workflow_config(.99, .90))
    config["assignment"]["screen"]["pair_deviance_max"] = 11.
    task = dict(root=str(root), job=job, selected=dict(contributor=.99, posterior=.90), execution_configs=[config])
    with patch("CliPP2.simulation.study._history", return_value=history), \
            patch("CliPP2.simulation.study._acquire", return_value=frame), \
            patch("CliPP2.simulation.evaluation.evaluate", side_effect=RuntimeError("regression_evaluator_failure_after_native_work")):
        failure = _run_cell(task)
    assert failure["status"] == "execution_failed"
    assert len(failure["cost_ledgers"]) == 1
    ledger = failure["cost_ledgers"][0]
    assert (ledger["fit_count"], ledger["assignment_pass_count"]) == (3, 2)
    attempt = root / failure["attempt"]
    paired = read_json(attempt / "configuration-1/paired_result.json")
    assert paired["status"] == "complete" and paired["config"] == config
    assert read_json(attempt / "failure.json")["cost_ledgers"] == failure["cost_ledgers"]
    assert read_json(attempt / "cost-ledger-1.json") == ledger
    saved = (attempt / "failure.json").read_bytes()
    with patch("CliPP2.simulation.experiment.run_paired_experiment", side_effect=AssertionError("default resume refitted terminal failure")):
        resumed = _run_cell(task)
    assert resumed == failure and (attempt / "failure.json").read_bytes() == saved
    assert len(list((root / "cells" / job["cell_id"]).glob("attempt-*"))) == 1
    changed = deepcopy(task)
    changed["execution_configs"][0]["assignment"]["posterior_min"] = .99
    try:
        _run_cell(changed)
    except ValueError as error:
        assert "configuration" in str(error)
    else:
        raise AssertionError("Resume silently reused results with changed frozen thresholds")
    try:
        _run_cell(dict(task, reference_match_atol=.05))
    except ValueError as error:
        assert "configuration" in str(error)
    else:
        raise AssertionError("Resume reused center qualification under a changed tolerance")
    original_completion = (root / "cells" / job["cell_id"] / "completion.json").read_bytes()
    with patch("CliPP2.simulation.study._history", side_effect=RuntimeError("explicit_retry_prefit_failure")):
        retried = _run_cell(dict(task, retry_failed=True))
    assert retried["status"] == "execution_failed" and retried["attempt"].endswith("attempt-002")
    assert retried["prior_cost_ledgers"] == failure["cost_ledgers"]
    assert (root / "cells" / job["cell_id"] / "completion.002.json").exists()
    assert (root / "cells" / job["cell_id"] / "completion.json").read_bytes() == original_completion
    assert (attempt / "failure.json").read_bytes() == saved
    costs = recorded_costs(root)
    assert costs["fit_attempts"] == costs["fit_complete"] == 3
    assert costs["assignment_attempts"] == costs["assignment_complete"] == 2
    captured = []
    from CliPP2.simulation.evaluation import evaluate as real_evaluator

    def evaluator(_history, _catalog, experiment, **kwargs):
        captured.append(deepcopy(experiment))
        evaluated = real_evaluator(_history, _catalog, experiment, **kwargs)
        assert evaluated["protocol_version"] == "2.2" and evaluated["metrics"]
        return evaluated

    with patch("CliPP2.simulation.study._history", return_value=history), \
            patch("CliPP2.simulation.study._acquire", return_value=frame), \
            patch("CliPP2.simulation.evaluation.evaluate", side_effect=evaluator):
        archived = _run_cell(dict(task, retry_failed=True, archive_completed=True))
    assert archived["status"] == "complete" and archived["attempt"].endswith("attempt-003")
    retry = root / archived["attempt"]
    zipped = retry / "configuration-1.zip"
    assert zipped.is_file() and not (retry / "configuration-1").exists()
    restored = _paired_result(retry)
    assert restored == captured[0]
    with zipfile.ZipFile(zipped) as archive:
        assert archive.testzip() is None and "execution.jsonl" in archive.namelist()
        for arm, expected in (("A", restored["stage_one"]), ("B", restored["arms"]["B"]["stage_two"]),
                              ("C", restored["arms"]["C"]["stage_two"])):
            accepted = json.loads(archive.read(f"arm_outputs/{arm}/accepted_fit.json"))["payload"]
            assert accepted == expected
            assert accepted["acceptance_evidence"]["independently_verified"]
            assert accepted["acceptance_evidence"]["membership_audit"]["complete"]
            assert any(name.startswith(f"arm_outputs/{arm}/native/") and name.endswith(".tsv")
                       for name in archive.namelist())
    assert archived["prior_cost_ledgers"] == failure["cost_ledgers"]
    costs = recorded_costs(root)
    assert costs["fit_attempts"] == costs["fit_complete"] == 6
    assert costs["assignment_attempts"] == costs["assignment_complete"] == 4
    assert costs.get("unfinished_attempts", 0) == 0
    assert (attempt / "failure.json").read_bytes() == saved
    # An archive can coexist with its source directory if interruption occurs
    # after the archive link but before directory removal. Count it only once.
    overlap = retry / "configuration-1"
    with zipfile.ZipFile(zipped) as archive:
        archive.extractall(overlap)
    assert recorded_costs(root) == costs
    assert _paired_result(retry) == restored
    # Partial directory cleanup must fall back to the already verified archive.
    (overlap / "execution.jsonl").unlink()
    (overlap / "paired_result.json").unlink()
    assert recorded_costs(root) == costs
    assert _paired_result(retry) == restored
    (overlap / "paired_result.json").write_text(json.dumps(dict(restored, status="conflicting")))
    try:
        _paired_result(retry)
    except ValueError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("Conflicting directory and archive evidence was reused")
    shutil.rmtree(overlap)
    before = {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
              for p in root.rglob("*") if p.is_file()}
    summary = status(root)
    assert summary["operation_costs"] == costs and not summary["active"]
    after = {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
             for p in root.rglob("*") if p.is_file()}
    assert before == after
    # A partial final append must not erase earlier durable started-work markers.
    interrupted = root / "cells" / "interrupted" / "attempt-001" / "configuration-1"
    interrupted.mkdir(parents=True)
    journal = interrupted / "execution.jsonl"
    events = [dict(operation_index=1, operation="fit", arm="A", status="running"),
              dict(operation_index=1, operation="fit", arm="A", status="complete"),
              dict(operation_index=1, operation="fit", arm="A", status="complete"),
              dict(operation_index=2, operation="assignment", arm="B3", status="running")]
    journal.write_text("\n".join(json.dumps(event) for event in events)+"\n", encoding="utf-8")
    interrupted_costs = recorded_costs(root)
    assert interrupted_costs["fit_attempts"] == 7 and interrupted_costs["assignment_attempts"] == 5
    assert interrupted_costs["assignment_running"] == 1 and interrupted_costs["unfinished_attempts"] == 1
    assert len(status(root)["active"]) == 1
    with journal.open("a", encoding="utf-8") as stream:
        stream.write('{"operation_index":')
    assert recorded_costs(root) == interrupted_costs


def _partial_objective_check(root, contract, history, frame, job):
    job = dict(job, phase="development", cell_id="partial-development", history_id="partial-development")
    values = [asdict(workflow_config(c, .8)) for c in (.9, .95, .99)]
    objective = [dict(cell_id=job["cell_id"], history_id=job["history_id"], family="S01", acquisition="A4",
        contributor=.9, posterior=p, configuration_id=f"c0.90-p{p:.2f}", fixed_evaluable=6,
        correct_unchanged=0, correct_targets=3, accepted_targets=3, wrong_targets=0,
        evaluable_targets=6, experiment_status="complete", arm_status="complete")
        for p in contract["threshold_development"]["posterior_grid"]]
    mock_experiment = dict(stage_one=None, arms={}, cost_ledger=dict(fit_count=0, assignment_pass_count=0))
    task = dict(root=str(root), job=job, execution_configs=values)
    with patch("CliPP2.simulation.study._history", return_value=history), \
            patch("CliPP2.simulation.study._acquire", return_value=frame), \
            patch("CliPP2.simulation.experiment.run_paired_experiment", return_value=mock_experiment), \
            patch("CliPP2.simulation.evaluation.evaluate", side_effect=[dict(metrics=[]), RuntimeError("second_evaluation_failed")]), \
            patch("CliPP2.simulation.study.development_scores", return_value=objective):
        failure = _run_cell(task)
    assert failure["status"] == "execution_failed" and len(failure["evaluations"]) == 1
    assert read_json(root / failure["objective_path"]) == objective
    with patch("CliPP2.simulation.study._history", return_value=history), \
            patch("CliPP2.simulation.study._acquire", return_value=frame), \
            patch("CliPP2.simulation.evaluation.information_diagnostics", side_effect=_fixed_information):
        selection = select_thresholds(root, [failure], contract)
    rows = selection["cell_objectives"]
    assert len(rows) == 12 and len({row["configuration_id"] for row in rows}) == 12
    assert all(row["fixed_evaluable"] == 6 for row in rows)
    assert all(row["correct_targets"] == (3 if row["contributor"] == .9 else 0) for row in rows)


def _planned_report_check(root, contract):
    contract = deepcopy(contract)
    contract["profile"] = "balanced"
    contract["requested_histories_per_family"] = dict(main=1, development=1)
    save_json(root / "study_contract.json", contract)
    save_json(root / "allocation.json", allocate(contract, [1.], [1]))
    captured = []

    def aggregate(values):
        captured.extend(values)
        return dict(metrics=[], planned_datasets=len(values))

    with patch("CliPP2.simulation.evaluation.aggregate", side_effect=aggregate), \
            patch("CliPP2.simulation.figures.make_figures", return_value=[]), \
            patch("CliPP2.simulation.study._write_report"):
        value = report(root)
    assert value["summary"]["planned_main_cells"] == value["summary"]["unexecuted_main_cells"] == 240
    assert value["summary"]["worker_failures"] == 0 and len(captured) == 240
    assert len({row["cell_id"] for row in captured}) == 240
    assert len({".".join(row["history_id"].split(".")[1:3]) for row in captured}) == 60
    assert all(all(not state["available"] and state["status"] == "not_executed"
                   for state in row["applicability"]) for row in captured)
    assert all(metric["value"] == 0 for row in captured for metric in row["metrics"])


def _offline_report_check(root, history, job):
    """A legacy archived result can acquire new metrics without inference IO."""
    from contextlib import ExitStack
    from CliPP2.simulation.reporting import reevaluate_cell
    from CliPP2.simulation.study import _latest_completion
    history_dir = root / "histories" / job["history_id"]
    history.write_truth(history_dir)
    save_json(history_dir / "history_manifest.json", {key: job[key]
        for key in ("history_id", "family", "history_seed", "dimensions")})
    save_json(root / "job_manifests/batch-001.json", [job])
    frozen = dict(selected=dict(contributor=.99, posterior=.90), frozen_before_main_generation=True)
    save_json(root / "threshold_development.json", frozen)
    # Model a preserved 1.2.1 contract whose original cohort source is gone. The
    # derived report must depend only on the retained selected attempt and truth.
    contract = read_json(root / "study_contract.json")
    contract.update(schema="clipp2.simulation-contract.v3", implementation_version="1.2.1")
    contract["cohort"].pop("content_id", None)
    contract["cohort"]["path"] = str(root / "not_present_original_cohort.json")
    (root / "study_contract.json").write_text(json.dumps(contract))
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    destination = root.parent / "offline_report"
    config = dict(reference_match_atol=.025, reference_distance="l2")
    forbidden = ("CliPP2.fit_execution.fit_prepared", "CliPP2.assignment.assign_mutations",
        "CliPP2.kernel.model.RegionalModel", "CliPP2.simulation.study._history", "CliPP2.simulation.study._acquire",
        "CliPP2.simulation.study.select_thresholds", "CliPP2.simulation.cohort.load_cohort")
    with ExitStack() as stack:
        for name in forbidden:
            stack.enter_context(patch(name, side_effect=AssertionError("Offline report executed " + name)))
        stack.enter_context(patch("CliPP2.simulation.figures.make_figures", return_value=[]))
        result = report(root, evaluation_config=config, report_dir=destination)
    assert result["summary"]["report_mode"] == "offline_reevaluation"
    assert result["summary"]["derived_evidence_unavailable"] == 0
    derived = read_json(destination / "evaluations/evaluation-00001.json")
    assert derived["status"] == "derived_from_preserved_evidence"
    assert derived["evaluation_definition"]["reference_distance"] == "l2"
    assert derived["derivation"]["fit_count"] == derived["derivation"]["assignment_pass_count"] == 0
    manifest = read_json(destination / "derivation_manifest.json")
    assert manifest["frozen_threshold_development"] == frozen and not manifest["historical_cohort_identity_backfilled"]
    assert before == {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    cell = _latest_completion(root / "cells" / job["cell_id"])
    fixture = root.parent / "offline_invalid_evidence"
    shutil.copytree(root, fixture)
    attempt = fixture / cell["attempt"]
    (attempt / "configuration-1").mkdir()
    assert reevaluate_cell(fixture, cell, config)["status"] == "derived_from_preserved_evidence"
    # Directory and archive copies must agree if both survived publication.
    with zipfile.ZipFile(attempt / "configuration-1.zip") as archive:
        archive.extractall(attempt / "configuration-1")
    assert reevaluate_cell(fixture, cell, config)["status"] == "derived_from_preserved_evidence"
    paired_path = attempt / "configuration-1/paired_result.json"
    changed = read_json(paired_path)
    changed["status"] = "different_retained_evidence"
    paired_path.write_text(json.dumps(changed))
    assert reevaluate_cell(fixture, cell, config)["status"] == "unavailable_derived_evidence"
    shutil.rmtree(attempt / "configuration-1")
    altered_job = dict(job, acquisition=dict(job["acquisition"], id="A1"))
    (attempt / "job.json").write_text(json.dumps(altered_job))
    assert reevaluate_cell(fixture, cell, config)["status"] == "unavailable_derived_evidence"
    (attempt / "job.json").write_text(json.dumps(job))
    (fixture / "histories" / job["history_id"] / "history.json").unlink()
    with patch("CliPP2.simulation.study._history", side_effect=AssertionError("missing truth regenerated")):
        assert reevaluate_cell(fixture, cell, config)["status"] == "unavailable_derived_evidence"
    history.write_truth(fixture / "histories" / job["history_id"])
    bound_cell = deepcopy(cell)
    bound_cell["job"]["cohort_content_id"] = "expected_cohort"
    (attempt / "job.json").write_text(json.dumps(bound_cell["job"]))
    manifest_path = fixture / "histories" / job["history_id"] / "history_manifest.json"
    bound_manifest = dict(read_json(manifest_path), cohort_content_id="expected_cohort")
    manifest_path.write_text(json.dumps(bound_manifest))
    invalid = reevaluate_cell(fixture, bound_cell, config)
    assert invalid["status"] == "unavailable_derived_evidence" and "truth belongs" in invalid["error"]["message"]


def _structured_failure_checks(root):
    history, frame, job = _fixture()
    for case in ("stage_one", "capped_stage_two", "assignment", "no_reference", "blocked", "empty"):
        directory = root / case
        current = dict(status="complete", stage_one=None,
            arms={"B": dict(status="complete", retained_source_evidence="B"), "C": dict(status="complete")},
            cost_ledger=dict(fit_count=1, assignment_pass_count=0))
        if case == "stage_one":
            current.update(status="failed", failed_stage="stage_one", error_type="RuntimeError", error="injected")
        elif case in ("capped_stage_two", "assignment"):
            current["status"] = "partial_failure"
            current["arms"]["C"].update(status="failed", failed_stage=case, error_type="RuntimeError", error="injected")
        elif case in ("no_reference", "blocked"):
            current["status"] = "no_eligible_clonal_reference" if case == "no_reference" else "blocked"
            for arm in current["arms"].values():
                arm["status"] = current["status"]
        task = dict(root=str(directory), job=dict(job, phase="development"),
                    execution_configs=[asdict(workflow_config(.95, .8))])
        def infer(_input, output, config, **kwargs):
            save_json(Path(output) / "paired_result.json", current)
            return deepcopy(current)
        with patch("CliPP2.simulation.study._history", return_value=history), \
                patch("CliPP2.simulation.study._acquire", return_value=frame), \
                patch("CliPP2.simulation.experiment.run_paired_experiment", side_effect=infer) as runner, \
                patch("CliPP2.simulation.evaluation.evaluate", side_effect=lambda h, c, e, **k: dict(metrics=[], arms=e["arms"])), \
                patch("CliPP2.simulation.study.development_scores", return_value=[dict(fixed_evaluable=8, correct_targets=0)]):
            first = _run_cell(task)
            failed = case in ("stage_one", "capped_stage_two", "assignment")
            assert (first["status"] == "execution_failed") == failed
            assert first["cost_ledgers"] == [current["cost_ledger"]]
            assert read_json(directory / first["objective_path"])[0]["fixed_evaluable"] == 8
            assert read_json(directory / first["evaluations"][0])["arms"]["B"] == current["arms"]["B"]
            completion = directory / "cells" / job["cell_id"] / "completion.json"
            prior = completion.read_bytes()
            assert _run_cell(task) == first and runner.call_count == 1
            if failed:
                current.update(status="complete")
                current["arms"]["C"] = dict(status="complete")
                second = _run_cell(dict(task, retry_failed=True))
                assert second["attempt"].endswith("attempt-002") and second["status"] == "complete"
                assert second["prior_cost_ledgers"] == first["cost_ledgers"] and runner.call_count == 2
            else:
                assert _run_cell(dict(task, retry_failed=True)) == first and runner.call_count == 1
            assert completion.read_bytes() == prior


def _qualification_lifecycle_checks(root, contract):
    def passed(output, summary="controls_summary.json"):
        save_json(output / "evidence.json", dict(checked=True))
        save_json(output / summary, dict(status="passed", paths=["evidence.json"],
            cost_ledger=dict(fit_count=0, assignment_pass_count=0, events=[])))
    def interrupted(output):
        save_json(output / "partial.json", dict(started=True))
        raise KeyboardInterrupt("fixture interruption")
    for name in ("mechanisms", "cohort"):
        directory = root / (name + "_interruption")
        kwargs = dict(summary_name="controls_summary.json", identity=dict(cohort_content_id="fixture"))
        try:
            _qualification_step(directory, name, interrupted, **kwargs)
        except KeyboardInterrupt:
            pass
        completed = _qualification_step(directory, name, passed, **kwargs)
        assert completed["attempt"] == "attempt-002"
        assert qualification_costs(directory)["unknown_cost_attempts"] == 1
        recovered_root = root / (name + "_published")
        def published_then_interrupted(output):
            passed(output)
            raise KeyboardInterrupt("summary published")
        try:
            _qualification_step(recovered_root, name, published_then_interrupted, **kwargs)
        except KeyboardInterrupt:
            pass
        with patch("builtins.print"):
            recovered = _qualification_step(recovered_root, name,
                lambda p: (_ for _ in ()).throw(AssertionError("completed controls reran")), **kwargs)
        assert recovered["recovered_from_summary"] and recovered["attempt"] == "attempt-001"
        assert qualification_costs(recovered_root)["unknown_cost_attempts"] == 0
        (recovered_root / "qualification_steps" / name / "attempt-001" / "output" / "evidence.json").unlink()
        try:
            _qualification_step(recovered_root, name, passed, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("Completed qualification reused missing outputs")
    failed_root = root / "recorded_failure"
    kwargs = dict(summary_name="controls_summary.json", identity=dict(cohort_content_id="fixture"))
    try:
        _qualification_step(failed_root, "mechanisms", lambda p: (_ for _ in ()).throw(ValueError("failed")), **kwargs)
    except ValueError:
        pass
    try:
        _qualification_step(failed_root, "mechanisms", passed, **kwargs)
    except RuntimeError as error:
        assert "--retry-failed" in str(error)
    else:
        raise AssertionError("Recorded qualification failure retried implicitly")
    assert _qualification_step(failed_root, "mechanisms", passed, retry_failed=True, **kwargs)["attempt"] == "attempt-002"
    aggregate_root = root / "before_aggregate"
    frozen = deepcopy(contract)
    original_save = save_json
    def interrupt_aggregate(path, payload):
        if Path(path).name == "qualification.json":
            raise KeyboardInterrupt("before aggregate publication")
        return original_save(path, payload)
    with patch("CliPP2.simulation.biology.generate_history", return_value=SimpleNamespace(validate=lambda: None)), \
            patch("CliPP2.simulation.controls.run_controls", side_effect=passed), \
            patch("CliPP2.simulation.controls.run_cohort_controls", side_effect=lambda c, p: passed(p, "cohort_controls_summary.json")), \
            patch("CliPP2.simulation.study.save_json", side_effect=interrupt_aggregate):
        try:
            _execute(aggregate_root, frozen)
        except KeyboardInterrupt:
            pass
    with patch("CliPP2.simulation.biology.generate_history", return_value=SimpleNamespace(validate=lambda: None)), \
            patch("CliPP2.simulation.controls.run_controls", side_effect=AssertionError("mechanisms reran")), \
            patch("CliPP2.simulation.controls.run_cohort_controls", side_effect=AssertionError("cohort reran")), \
            patch("CliPP2.simulation.study.allocate", side_effect=RuntimeError("stop after qualification")):
        try:
            _execute(aggregate_root, frozen, resume=True)
        except RuntimeError as error:
            assert str(error) == "stop after qualification"
    receipt = read_json(aggregate_root / "qualification.json")
    assert receipt["cohort_content_id"] == contract["cohort"]["content_id"]
    assert all(value["status"] == "passed" for value in receipt["controls"].values())


def run_lifecycle_checks(root, settings):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    contract = _design_check(root, settings)
    job = next(jobs(contract, "development", 1))
    histories = root / "history_reuse"
    writer = SimpleNamespace(write_truth=lambda p: save_json(p / "history.json", dict(fixture=True)))
    with patch("CliPP2.simulation.study._history", return_value=writer) as create:
        _prepare_histories(histories, [job])
        _prepare_histories(histories, [job])
        assert create.call_count == 1
    manifest = histories / "histories" / job["history_id"] / "history_manifest.json"
    value = read_json(manifest)
    assert value["cohort_content_id"] == contract["cohort"]["content_id"]
    value["cohort_content_id"] = "different"
    manifest.write_text(json.dumps(value))
    try:
        _prepare_histories(histories, [job])
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched historical content identity reused")
    _structured_failure_checks(root / "structured")
    _qualification_lifecycle_checks(root / "qualification", contract)
    frozen_root = root / "qualification" / "before_aggregate"
    frozen = read_json(frozen_root / "study_contract.json")
    content = read_json(frozen_root / "cohort_settings.json")
    content["acquisitions"]["A1"]["mean_depth"] += 1
    (frozen_root / "cohort_settings.json").write_text(json.dumps(content))
    with patch("CliPP2.simulation.study._qualification_step", side_effect=AssertionError("changed source reached qualification")):
        try:
            _execute(frozen_root, frozen, resume=True)
        except ValueError as error:
            assert "content" in str(error)
        else:
            raise AssertionError("Changed frozen cohort resumed")
    return dict(status="passed", native_fit_count=0, assignment_pass_count=0,
        checks=["cohort_job_and_history_identity", "matching_history_reuse_and_changed_manifest_rejection",
                "frozen_content_verified_before_resume_reuse", "caught_stage1_stage2_assignment_failures_retryable",
                "partial_arm_objective_cost_evidence_retained", "scientific_rejections_not_retried",
                "qualification_interruptions_and_published_summary_recovery", "recorded_qualification_failure_requires_explicit_retry",
                "before_aggregate_interruption_reuses_both_steps", "interrupted_qualification_cost_unknown"])


def run_checks(directory, cohort_settings):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    contract = _design_check(root, cohort_settings)
    lifecycle = run_lifecycle_checks(root / "lifecycle", cohort_settings)
    for protocol, schema in (("2.1", "clipp2.simulation-contract.v1"), ("2.2", "clipp2.simulation-contract.v2"), ("2.2", "clipp2.simulation-contract.v3")):
        old = dict(contract, protocol_version=protocol, schema=schema)
        with patch("CliPP2.simulation.study.read_json", side_effect=AssertionError("old contract reached resume IO")):
            try:
                _execute(root / "old_protocol", old, resume=True)
            except ValueError as error:
                assert "original installed environment" in str(error)
            else:
                raise AssertionError("An older contract resumed under the revised implementation")
    record = dict(best_label=1, informative_regions=2, cluster_tie=False, fit_screen_status="pass",
                  log_competitor_ratio=-1000., has_finite_alternative=True, reference_quality_pass=False)
    assert _decision(record, .8) is None and _decision(record, .99) is None
    record["reference_quality_pass"] = True
    assert _decision(record, .99) == 1 and _decision(record, 1.) is None
    history, frame, job = _fixture()
    _failed_credit_check(root, history, frame, job)
    selection_root = root / "selection"
    selection_root.mkdir()
    _selection_check(selection_root, contract, history, frame, job)
    _native_failure_check(root / "native_failure", history, frame, job, contract)
    _offline_report_check(root / "native_failure", history, job)
    _partial_objective_check(root / "partial_objective", contract, history, frame, job)
    _planned_report_check(root / "unexecuted_report", contract)
    result = dict(status="passed", directory=str(root), native_fit_count=6, assignment_pass_count=4, lifecycle=lifecycle,
        checks=["exact_60_variant_720_history_2640_cell_design", "full_10_20_strict_point05_contract",
                "source_protocol21_execution22_and_cross_protocol_resume_rejection",
                "threshold_reuse_preserves_reference_quality_and_finite_rival_certainty",
                "failed_C_preserves_fixed_denominator_zero_credit", "equal_family_selection_includes_failed_histories",
                "zero_acceptance_and_excess_error_rejected_threshold_ties_frozen", "post_evaluator_failure_retains_native_costs",
                "default_resume_terminal_failure_immutable", "changed_execution_config_rejected",
                "changed_reference_evaluation_tolerance_rejected",
                "explicit_retry_appends_attempt_and_preserves_prior_costs", "partial_objectives_preserved_missing_configurations_zero_filled",
                "successful_retry_archives_complete_numerical_receipts", "archive_roundtrip_preserves_paired_state_and_costs",
                "full_truth_evaluator_accepts_protocol22_assignment_and_final_membership",
                "runtime_counts_archived_failed_and_interrupted_work_once", "runtime_status_does_not_modify_outputs",
                "frozen_configuration_reaches_real_runner", "unexecuted_main_cells_remain_in_planned_denominator",
                "offline_archived_legacy_evaluation_preserves_sources_and_frozen_thresholds",
                "offline_missing_or_ambiguous_evidence_unavailable_without_regeneration"])
    save_json(root / "study_checks.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--cohort-settings", type=Path, default=Path("CliPP2_phylogeny_cohort_settings.json"))
    args = parser.parse_args()
    print(json.dumps(run_checks(args.outdir or tempfile.mkdtemp(prefix="clipp2-study-check-"), args.cohort_settings), indent=2))
