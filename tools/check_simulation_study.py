"""Fast study regressions with an eight-row native run and archived retry."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from CliPP2.simulation.biology import generate_history
# Import before history-generator mocks: controls binds biology helpers at import time.
from CliPP2.simulation import controls
from CliPP2.simulation.design import jobs, roster_counts, make_contract, workflow_config
from CliPP2.simulation.runtime import recorded_costs, status
from CliPP2.simulation.study import (_execute, _paired_result, _prepare_histories,
                                    _run_cell, read_json, report, save_json)


def _fixture():
    from CliPP2.simulation.acquisition import acquire
    history = generate_history("S01", 300, n_mutations=8, n_segments=2, n_regions=2)
    values = [[50, 50]]*4 + [[20, 15]]*4
    frame = pd.DataFrame([dict(mutation_id=mutation, sample_id=region,
        alt_count=values[i][j], ref_count=100-values[i][j], count_observed=1,
        purity=1., normal_cn=2, normal_cn_supplied=1, segment_id="s0", mapping_genuine=1,
        cn_state_id="0", cn_state_fraction=1., allele_a_cn=1, allele_b_cn=1)
        for i, mutation in enumerate(history.mutation_ids) for j, region in enumerate(history.region_ids)])
    generated = acquire(history, 400, purity=1., depth=100)
    frame.attrs = deepcopy(generated.attrs)
    frame.attrs["acquisition"].update(alt_counts=np.asarray(values), depth=np.full((8, 2), 100),
        total_counts=np.full((8, 2), 100), count_observed=np.ones((8, 2), bool),
        supplied_segment_ids=np.full((8, 2), "s0"))
    job = dict(cell_id="main-S01-orchestration", phase="main", family="S01",
        history_id="main-S01-orchestration", history_seed=300, acquisition_seed=400,
        acquisition=dict(id="A4", purity=1., depth=100), dimensions=dict(mutations=8, segments=2, regions=2))
    return history, frame, job


def _prepare_fixture(root, history, job):
    from CliPP2.simulation.biology import history_content_id
    directory = Path(root) / "histories" / job["history_id"]
    if not (directory / "history_manifest.json").exists():
        history.write_truth(directory)
        save_json(directory / "history_manifest.json", dict(
            {key: job[key] for key in ("history_id", "family", "history_seed", "dimensions")},
            history_content_id=history_content_id(history)))


def _design_check(root, settings):
    from CliPP2.simulation.cohort import load_cohort
    snapshot = root / "cohort_settings.json"
    save_json(snapshot, load_cohort(settings).settings)
    contract = make_contract(cohort_settings=snapshot)
    assert contract["protocol_version"] == "2.5" and contract["source_cohort_protocol_version"] == "2.1"
    assert contract["schema"] == "clipp2.simulation-contract.v9"
    assert contract["evaluation"]["reference_match_atol"] is None
    assert json.loads(snapshot.read_text())["workflow_protocol"] == "2.1"
    main, development = list(jobs(contract, "main")), list(jobs(contract, "development"))
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
    other_workers = deepcopy(contract)
    other_workers["resource_limits"]["workers"] = 1
    assert list(jobs(other_workers, "main")) == main
    assert list(jobs(other_workers, "development")) == development
    assert "development_extension" not in contract["replicate_roster"]
    assert "threshold_development" not in contract
    reduced = make_contract(cohort_settings=snapshot, main_replicates=1, development_replicates=1)
    assert reduced["replicate_roster"]["development"] == [1]
    assert "development_extension" not in reduced["replicate_roster"]
    assert len(list(jobs(reduced, "development"))) == 120
    from CliPP2.simulation.reporting import roster_identity
    payload = dict(schema="clipp2.simulation-roster.v1", families=contract["families"],
                   identity_policy="semantic_jobs_without_cohort_path", jobs=main)
    relocated = deepcopy(payload)
    for job in relocated["jobs"]:
        job["cohort_settings"] = "/relocated/cohort_settings.json"
    assert roster_identity(payload) == roster_identity(relocated)
    relocated["jobs"][0]["history_seed"] += 1
    assert roster_identity(payload) != roster_identity(relocated)
    worker_payload = dict(payload, jobs=list(jobs(other_workers, "main")))
    assert roster_identity(worker_payload) == roster_identity(payload)
    allocation = roster_counts(contract)
    assert allocation["planned_main_datasets"] + allocation["planned_development_datasets"] == 2640
    assert allocation["planned_core_max_fits"] == 7920 and allocation["planned_core_max_assignment_passes"] == 5280
    config = contract["base_workflow_config"]
    assert (config["fit"]["max_clusters"], config["stage_two_max_clusters"], config["fit"]["max_major_cn"]) == (10, 20, None)
    assert "stage_two_near_clonal_distance_max" not in config
    return contract


def _native_failure_check(root, history, frame, job, contract):
    _prepare_fixture(root, history, job)
    save_json(root / "study_contract.json", contract)
    config = asdict(workflow_config())
    task = dict(root=str(root), job=job, execution_configs=[config])
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
    changed["execution_configs"][0]["stage_two_max_clusters"] = 19
    try:
        _run_cell(changed)
    except ValueError as error:
        assert "configuration" in str(error)
    else:
        raise AssertionError("Resume silently reused results with changed frozen configuration")
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
        assert evaluated["protocol_version"] == "2.5" and evaluated["metrics"]
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


def _planned_report_check(root, contract):
    contract = deepcopy(contract)
    contract["replicate_roster"].update(main=[1], development=[1])
    save_json(root / "study_contract.json", contract)
    from CliPP2.simulation.study import _freeze_main_roster
    _freeze_main_roster(root, contract)
    captured = []

    def aggregate(values, **kwargs):
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
    assert all(row["schema"] == "clipp2.simulation.evaluation.v7" for row in captured)
    assert all(metric["value"] is None for row in captured for metric in row["metrics"]
               if metric["name"] == "binary_correct_per_original_evaluable")


def _optional_execution_report_check(root):
    """Describe retained optional outcomes without treating flags as execution."""
    from contextlib import ExitStack
    from CliPP2.simulation.study import _write_report

    def cell(name, status, *, phase="sensitivity"):
        return dict(status=status, job=dict(cell_id=f"{phase}-S02-{name}", family="S02",
            phase=phase, acquisition=dict(id=name)))

    cases = (
        ("completed", False, [cell("regions2_fixed_total", "complete")], "1 completed, 0 failed, 0 unknown"),
        ("failed", False, [cell("regions4_fixed_total", "execution_failed")], "0 completed, 1 failed, 0 unknown"),
        ("unknown", True, [cell("regions8_fixed_total", "interrupted")], "0 completed, 0 failed, 1 unknown"),
        ("no_evidence_enabled", True, [], "No retained completion evidence"),
        ("no_evidence_disabled", False, [], "No retained completion evidence"),
        ("unrelated_conditions", True, [cell("regions4_per_region", "complete"),
            cell("regions4_fixed_total", "complete", phase="main")], "No retained completion evidence"),
        ("mixed", False, [cell("regions1_fixed_total", "complete"),
            cell("regions2_fixed_total", "execution_failed"), cell("regions8_fixed_total", None)],
            "1 completed, 1 failed, 1 unknown"),
    )
    with ExitStack() as stack:
        for target in ("CliPP2.fit_execution.fit_prepared", "CliPP2.assignment.assign_mutations",
                       "CliPP2.kernel.model.RegionalModel", "CliPP2.simulation.study._history",
                       "CliPP2.simulation.study._acquire"):
            stack.enter_context(patch(target, side_effect=AssertionError("Report attempted inference")))
        for name, enabled, cells, expected in cases:
            directory = root / name
            contract = make_contract(mutations=12, segments=2, regions=4, workers=1)
            contract["optional_panels"].update(technical_sensitivity=enabled, extended_sensitivity=enabled)
            save_json(directory / "study_contract.json", contract)
            summary = dict(fit_count=0, assignment_pass_count=0, report_mode="cached_metrics",
                           roster_status="incomplete", population_counts=[], metrics=[])
            _write_report(directory, directory, cells, [], summary, [])
            text = (directory / "comparison_report.html").read_text()
            section = text.split("<h2>Optional execution evidence</h2>")[1].split("<h2>")[0]
            assert expected in section
            assert "Optional fixed-total-depth refitting" not in text
            if "No retained" in expected:
                assert "their execution is unknown" in section
            else:
                assert "Cells without retained completion evidence have unknown execution state" in section
                assert "do not aggregate optional-panel scientific results" in section
                for value in cells:
                    assert value["job"]["cell_id"] in section
    return dict(status="passed", cases=len(cases), retained_outcomes_only=True,
                configuration_flags_do_not_establish_execution=True,
                fit_count=0, assignment_pass_count=0)




def _report_integrity_check(root):
    """Per-arm ternary counts and contextual cache errors require no inference."""
    from contextlib import ExitStack
    from CliPP2.io.data import content_hash
    from CliPP2.simulation.evaluation import evaluation_definition
    definition = evaluation_definition()
    contract = make_contract(mutations=12, segments=2, regions=4, workers=1)
    contract.update(families=["S01"], acquisitions=["A1"])
    jobs = [dict(cell_id=f"integrity-{i}", history_id=f"integrity-h{i}", family="S01",
                 phase="main", acquisition=dict(id="A1", purity=.8, depth=100)) for i in range(4)]
    records = []
    for job, states in zip(jobs, ((0, 0, None), (0, None, None), (None, None, None), (1, 1, 1))):
        records.append(dict(schema="clipp2.simulation.evaluation.v7", max_multiplicity_difference=1, **job, acquisition_id="A1",
            status="execution_failed" if states == (0, 0, None) else "complete",
            evaluation_definition=definition, evaluation_definition_id=content_hash(definition),
            metrics=[dict(name=name, arm=arm, view="all_planned_cells", stratum="all", value=value,
                truth_required=False) for arm in ("A", "B", "C", "B3", "C3")
                for name, value in zip(("execution_available", "evaluation_available", "scientifically_applicable"), states)]))

    def prepare(directory, values):
        contract["replicate_roster"].update(main=list(range(1, len(values)+1)), development=[1])
        save_json(directory / "study_contract.json", contract)
        payload = dict(schema="clipp2.simulation-roster.v1", families=["S01"],
            identity_policy="semantic_jobs_without_cohort_path", jobs=jobs[:len(values)])
        save_json(directory / "main_roster.json", dict(payload, roster_id=content_hash(payload)))
        paths = []
        for job, value in zip(jobs, values):
            path = Path("cells") / job["cell_id"] / "attempt-001/evaluation-1.json"
            save_json(directory / path, value)
            save_json(directory / "cells" / job["cell_id"] / "completion.json", dict(job=job,
                status="complete", attempt=str(path.parent), evaluations=[str(path)], cost_ledgers=[],
                elapsed_seconds=0., peak_resident_bytes=0, output_bytes=0))
            paths.append(directory / path)
        return paths

    with ExitStack() as stack:
        for target in ("CliPP2.fit_execution.fit_prepared", "CliPP2.assignment.assign_mutations",
                       "CliPP2.simulation.study._history", "CliPP2.simulation.study._acquire"):
            stack.enter_context(patch(target, side_effect=AssertionError("Report attempted inference")))
        stack.enter_context(patch("CliPP2.simulation.figures.make_figures", return_value=[]))
        directory = root / "accounting"
        prepare(directory, records)
        output = report(directory)
        for row in output["summary"]["population_counts"]:
            assert (row["executed"], row["not_executed"], row["execution_unknown"]) == (1, 2, 1)
            assert (row["evaluated"], row["unavailable_derived"], row["evaluation_unknown"]) == (1, 1, 2)
            assert row["retained_roster"] == 4
            metric = next(x for x in output["summary"]["metrics"] if x["arm"] == row["arm"]
                          and x["name"] == "evaluation_available")
            assert metric["unavailable_derived_cells"] == row["unavailable_derived"]
            assert metric["unknown_evaluation_cells"] == row["evaluation_unknown"]
        assert "timing and disk measurements did not select its membership" in Path(output["report"]).read_text()
        bad = [None, [], dict(records[0], metrics=None), dict(records[0], metrics=[None]),
               dict(records[0], cell_id="other"), dict(records[0], evaluation_definition_id="invalid")]
        for index, value in enumerate(bad):
            directory = root / f"invalid-{index}"
            path = prepare(directory, [value])[0]
            before = path.read_bytes()
            try:
                report(directory)
            except ValueError as error:
                assert str(path) in str(error) and jobs[0]["cell_id"] in str(error)
            else:
                raise AssertionError("Structurally invalid cache was accepted")
            assert path.read_bytes() == before and not (directory / "reports").exists()
        directory = root / "truncated"
        path = prepare(directory, [records[0]])[0]
        path.write_text("{truncated")
        assert report(directory)["summary"]["derived_evidence_unavailable"] == 1
    return dict(status="passed", per_arm_ternary_accounting=True, contextual_cache_errors=len(bad),
                truncated_cache_remains_unavailable=True, fit_count=0, assignment_pass_count=0)


def _partial_evidence_report_checks(root):
    """Exercise real public report publication without fitting or assigning."""
    from contextlib import ExitStack
    from CliPP2.io.data import content_hash
    from CliPP2.simulation import figures
    from CliPP2.simulation.evaluation import evaluation_definition, _operational_states
    from CliPP2.simulation.reporting import unavailable_evaluation
    definition = evaluation_definition()

    def job(index):
        return dict(cell_id=f"report-h{index}-A1", history_id=f"report-h{index}", family="S01",
                    phase="main", acquisition=dict(id="A1", purity=.8, depth=100))

    def successful(index, *, truth="verified_history_binding", transitions=3):
        record = job(index)
        metrics = [dict(name=name, arm="C3", view="", stratum="all", value=1.,
            numerator=10, denominator=10, status="available", truth_required=True)
            for name in ("binary_map_correct_per_original_evaluable",)]
        metrics += [dict(name=name, arm="C3", view="all_planned_cells", stratum="all", value=value,
            numerator=value, denominator=1, status="available", truth_required=False)
            for name, value in (("analysis_available", 1), ("analysis_failure", 0),
                               ("execution_available", 1), ("execution_failure", 0),
                               ("evaluation_available", 1), ("scientifically_applicable", 1))]
        return dict(schema="clipp2.simulation.evaluation.v7", max_multiplicity_difference=1, evaluation_definition=definition,
            evaluation_definition_id=content_hash(definition), truth_qualification=truth,
            **record, acquisition_id="A1", original_evaluable_count=10, region_count=4,
            status="complete", metrics=metrics, calibration=[], cap_audit=[],
            detailed_membership_transitions={"C3": {"stage1_reference_recheck": [dict(
                baseline_state="Mc", final_state="Mw", count=transitions)]}})

    def publish_case(directory, evaluations):
        contract = make_contract(mutations=12, segments=2, regions=4, workers=1)
        contract["families"] = ["S01"]
        roster = [job(index) for index in range(len(evaluations))]
        contract["acquisitions"] = ["A1"]
        contract["replicate_roster"].update(main=list(range(1, len(roster)+1)), development=[1])
        save_json(directory / "study_contract.json", contract)
        payload = dict(schema="clipp2.simulation-roster.v1", families=["S01"],
            identity_policy="semantic_jobs_without_cohort_path", jobs=roster)
        save_json(directory / "main_roster.json", dict(payload, roster_id=content_hash(payload)))
        for record, evaluation in zip(roster, evaluations):
            attempt = Path("cells") / record["cell_id"] / "attempt-001"
            relative = attempt / "evaluation-1.json"
            save_json(directory / relative, evaluation)
            save_json(directory / "cells" / record["cell_id"] / "completion.json", dict(job=record,
                status="execution_failed" if evaluation["status"] == "execution_failed" else "complete",
                attempt=str(attempt), evaluations=[str(relative)], cost_ledgers=[], elapsed_seconds=0.,
                peak_resident_bytes=0, output_bytes=0))
        originals = {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}
        matrices = []
        save_figure = figures._save

        def capture(figure, output, name, paths):
            if name == "07_population_accounting":
                matrices.extend(np.array(axis.images[0].get_array()) for axis in figure.axes[2:] if axis.images)
            return save_figure(figure, output, name, paths)

        with ExitStack() as stack:
            for name in ("CliPP2.fit_execution.fit_prepared", "CliPP2.assignment.assign_mutations",
                         "CliPP2.kernel.model.RegionalModel", "CliPP2.simulation.study._history",
                         "CliPP2.simulation.study._acquire", "CliPP2.simulation.study.assess_development"):
                stack.enter_context(patch(name, side_effect=AssertionError("Report attempted inference: " + name)))
            stack.enter_context(patch("CliPP2.simulation.figures._save", side_effect=capture))
            result = report(directory)
        assert all(path.read_bytes() == contents for path, contents in originals.items())
        assert all(row["executed"] == sum(state.get("execution_available") == 1
                   for state in result["summary"]["availability"] if state["arm"] == row["arm"])
                   for row in result["summary"]["population_counts"])
        output = Path(result["report"]).parent
        required = ("aggregate.json", "tumor_metrics.tsv", "detailed_transitions.tsv", "bound_audit.tsv",
                    "execution_ledger.tsv", "comparison_report.html")
        assert all((output / name).is_file() for name in required)
        assert len(list(output.glob("*.png"))) == len(list(output.glob("*.pdf"))) == 7
        assert result["summary"]["fit_count"] == result["summary"]["assignment_pass_count"] == 0
        return result, output, matrices

    failure_states = _operational_states(dict(status="failed", failed_stage="stage_one"), {})
    failure_states["C3"]["decision_population"] = dict(scope="whole_original_evaluable", coverage="complete",
        evaluable_count=10, accepted_count=0, evidence="failure_before_stage_one")
    failure = unavailable_evaluation(job(1), "execution_failed", evaluation_config={},
        execution_state="execution_failed", original_evaluable_count=10,
        arm_applicability=failure_states)
    assert "region_count" not in failure
    failure.pop("calibration", None)
    failure.pop("cap_audit", None)
    zero = next(row for row in failure["metrics"] if row["name"] == "binary_map_correct_per_original_evaluable" and row["arm"] == "C3")
    assert zero["value"] == 0 and zero["status"] == "known_no_decisions"
    result, output, _ = publish_case(root / "known_zero", [successful(0), failure])
    assert {row["arm"]: row["executed"] for row in result["summary"]["population_counts"]} == {
        "A": 1, "B": 0, "C": 0, "B3": 0, "C3": 1}
    metric = next(row for row in result["summary"]["metrics"]
                  if row["name"] == "binary_map_correct_per_original_evaluable" and row["arm"] == "C3")
    assert metric["planned_population_estimate"] == .5 and metric["planned_population_status"] == "available"
    assert len(result["summary"]["planned_weights"]) == 2
    exported = pd.read_csv(output / "tumor_metrics.tsv", sep="\t")
    assert set(exported.truth_qualification) == {"verified_history_binding", "missing_truth_evidence"}
    for value in (None, True, "4", [4], {}, float("nan"), float("inf"), 0, -1, 1.5):
        assert figures._sensitivity_covariate(dict(failure, region_count=value), "region_count") is None
    assert figures._sensitivity_covariate(dict(failure, region_count=4), "region_count") == 4
    assert figures._values([failure], "binary_map_correct_per_original_evaluable", "C3")[0][1] == 0
    unsupported = deepcopy(failure)
    unsupported["truth_qualification"] = "verified_history_binding"
    for arm in unsupported["applicability"]:
        arm.pop("decision_population", None)
    assert figures._values([unsupported], "binary_map_correct_per_original_evaluable", "C3") == []
    result, output, matrices = publish_case(root / "truth_qualified", [successful(0),
        successful(1, truth="missing_truth_evidence", transitions=37)])
    transitions = pd.read_csv(output / "detailed_transitions.tsv", sep="\t")
    assert transitions.set_index("truth_qualification")["count"].to_dict() == {
        "verified_history_binding": 3, "missing_truth_evidence": 37}
    metrics = pd.read_csv(output / "tumor_metrics.tsv", sep="\t")
    assert set(metrics.truth_qualification) == {"verified_history_binding", "missing_truth_evidence"}
    assert metrics.groupby("truth_qualification").size().to_dict() == {
        "verified_history_binding": 7, "missing_truth_evidence": 7}
    assert len(matrices) == 1 and matrices[0].sum() == matrices[0][0, 1] == 3
    html = (output / "comparison_report.html").read_text()
    section = html.split("<h2>Detailed membership transitions by target origin</h2>")[1].split("<h2>")[0]
    assert "verified history bindings only" in section
    assert "<td>3</td>" in section and "<td>37</td>" not in section and "<td>40</td>" not in section
    return dict(status="passed", known_zero_preserves_planned_weight=True,
        upstream_failure_preserves_per_arm_execution_counts=True, source_inputs_unchanged=True,
        optional_covariates_do_not_block_publication=True, primary_html_and_heatmap_verified_only=True,
        raw_exports_preserve_truth_qualification=True, fit_count=0, assignment_pass_count=0)


def _offline_report_check(root, history, job):
    """An archived result can acquire new metrics without inference or source IO."""
    from contextlib import ExitStack
    from CliPP2.simulation.reporting import reevaluate_cell
    from CliPP2.simulation.study import _latest_completion
    _prepare_fixture(root, history, job)
    save_json(root / "job_manifests/batch-001.json", [job])
    # Offline reports use retained evidence without reopening the cohort source.
    contract = read_json(root / "study_contract.json")
    contract["cohort"]["path"] = str(root / "not_present_original_cohort.json")
    (root / "study_contract.json").write_text(json.dumps(contract))
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    destination = root.parent / "offline_report"
    config = dict(reference_match_atol=.025, reference_distance="l2")
    forbidden = ("CliPP2.fit_execution.fit_prepared", "CliPP2.assignment.assign_mutations",
        "CliPP2.kernel.model.RegionalModel", "CliPP2.simulation.study._history", "CliPP2.simulation.study._acquire",
        "CliPP2.simulation.study.assess_development", "CliPP2.simulation.cohort.load_cohort")
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
    assert manifest["source_cohort_content_id"] == contract["cohort"]["content_id"]
    assert manifest["fit_count"] == manifest["assignment_pass_count"] == 0
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
    for case in ("stage_one", "capped_stage_two", "assignment", "no_reference", "blocked", "empty", "authoritative_assignments"):
        directory = root / case
        current = dict(status="complete", stage_one=None,
            arms={"B": dict(status="complete", retained_source_evidence="B"), "C": dict(status="complete")},
            cost_ledger=dict(fit_count=1, assignment_pass_count=0))
        if case == "authoritative_assignments":
            current["arms"]["B"]["assignment"] = dict(assignments=[
                dict(evidence_status="observed", final_assignment_label=1, map_cluster_label=1,
                     posterior=.51, diagnostic_status="failed"),
                dict(evidence_status="prior_only", final_assignment_label=None, map_cluster_label=0)])
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
                    execution_configs=[asdict(workflow_config())])
        _prepare_fixture(directory, history, task["job"])
        def infer(_input, output, config, **kwargs):
            save_json(Path(output) / "paired_result.json", current)
            return deepcopy(current)
        with patch("CliPP2.simulation.study._history", return_value=history), \
                patch("CliPP2.simulation.study._acquire", return_value=frame), \
                patch("CliPP2.simulation.experiment.run_paired_experiment", side_effect=infer) as runner, \
                patch("CliPP2.simulation.evaluation.evaluate", side_effect=lambda h, c, e, **k: dict(metrics=[], arms=e["arms"])):
            first = _run_cell(task)
            failed = case in ("stage_one", "capped_stage_two", "assignment")
            assert (first["status"] == "execution_failed") == failed
            assert first["cost_ledgers"] == [current["cost_ledger"]]
            assert "objective_path" not in first
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


def _independent_controls_checks(root, contract):
    frozen = deepcopy(contract)
    with patch.object(controls, "run_controls", side_effect=AssertionError("automatic controls")), \
            patch("CliPP2.simulation.study.run_jobs", return_value=[]) as run, \
            patch("CliPP2.simulation.study.run_optional_panels"), \
            patch("CliPP2.simulation.study.report", return_value={"status": "fixture"}) as summarize:
        assert _execute(root, frozen) == {"status": "fixture"}
        assert run.call_count == 2 and summarize.call_count == 1
        assert not (root / "qualification.json").exists()
        assert not (root / "qualification_steps").exists()
        roster = (root / "main_roster.json").read_bytes()
        assessment = (root / "development_assessment.json").read_bytes()
        assert _execute(root, frozen, resume=True) == {"status": "fixture"}
        assert run.call_count == 4 and summarize.call_count == 2
        assert (root / "main_roster.json").read_bytes() == roster
        assert (root / "development_assessment.json").read_bytes() == assessment
    return root


def run_lifecycle_checks(root, settings):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    contract = _design_check(root, settings)
    job = next(jobs(contract, "development"))
    histories = root / "history_reuse"
    writer = _fixture()[0]
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
    frozen_root = _independent_controls_checks(root / "independent_controls", contract)
    frozen = read_json(frozen_root / "study_contract.json")
    content = read_json(frozen_root / "cohort_settings.json")
    content["acquisitions"]["A1"]["mean_depth"] += 1
    (frozen_root / "cohort_settings.json").write_text(json.dumps(content))
    with patch("CliPP2.simulation.study.run_jobs", side_effect=AssertionError("changed cohort reached execution")):
        try:
            _execute(frozen_root, frozen, resume=True)
        except ValueError as error:
            assert "content" in str(error)
        else:
            raise AssertionError("Changed frozen cohort resumed")
    return dict(status="passed", native_fit_count=0, assignment_pass_count=0,
        checks=["cohort_job_and_history_identity", "matching_history_reuse_and_changed_manifest_rejection",
                "frozen_content_verified_before_resume_reuse", "caught_stage1_stage2_assignment_failures_retryable",
                "partial_arm_evaluation_cost_evidence_retained", "scientific_rejections_not_retried",
                "study_execution_and_resume_independent_of_controls", "historical_control_receipts_preserved",
                "historical_control_costs_remain_reported"])


def run_checks(directory, cohort_settings):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    contract = _design_check(root, cohort_settings)
    lifecycle = run_lifecycle_checks(root / "lifecycle", cohort_settings)
    from check_simulation_resume import run_checks as check_resume
    resume = check_resume(root / "resume")
    for protocol, schema in (("2.1", "clipp2.simulation-contract.v1"), ("2.2", "clipp2.simulation-contract.v2"), ("2.2", "clipp2.simulation-contract.v3"), ("2.2", "clipp2.simulation-contract.v4")):
        old = dict(contract, protocol_version=protocol, schema=schema)
        with patch("CliPP2.simulation.study.read_json", side_effect=AssertionError("old contract reached resume IO")):
            try:
                _execute(root / "old_protocol", old, resume=True)
            except ValueError as error:
                assert "original installed environment" in str(error)
            else:
                raise AssertionError("An older contract resumed under the revised implementation")
    history, frame, job = _fixture()
    _native_failure_check(root / "native_failure", history, frame, job, contract)
    _offline_report_check(root / "native_failure", history, job)
    _planned_report_check(root / "unexecuted_report", contract)
    report_integrity = _report_integrity_check(root / "report_integrity")
    optional_execution = _optional_execution_report_check(root / "optional_execution_reports")
    partial_reports = _partial_evidence_report_checks(root / "partial_evidence_reports")
    assert controls.generate_history is generate_history, "History mock leaked into real controls"
    mechanisms = controls.run_controls(root / "actual_mechanism_controls")
    assert mechanisms["status"] == "passed" and mechanisms["cost_ledger"]["fit_count"] == 0
    mechanism_checks = dict(status=mechanisms["status"], schema=mechanisms["schema"],
        checks=mechanisms["checks"], native_fit_count=0,
        assignment_pass_count=mechanisms["assignment_pass_count"])
    result = dict(status="passed", directory=str(root), native_fit_count=6, assignment_pass_count=4+mechanisms["assignment_pass_count"],
        mechanism_checks=mechanism_checks,
        lifecycle=lifecycle, resume=resume, partial_evidence_reports=partial_reports,
        report_integrity=report_integrity,
        optional_execution_reports=optional_execution,
        checks=["exact_60_variant_720_history_2640_cell_design", "full_10_20_unrestricted_all_target_contract",
                "source_protocol21_execution24_and_cross_protocol_resume_rejection",
                "authoritative_observed_MAP_ignores_reporting_diagnostics",
                "poor_finite_development_does_not_block_main", "development_failures_remain_accounted", "post_evaluator_failure_retains_native_costs",
                "default_resume_terminal_failure_immutable", "changed_execution_config_rejected",
                "changed_reference_evaluation_tolerance_rejected",
                "explicit_retry_appends_attempt_and_preserves_prior_costs", "exactly_one_declared_configuration",
                "successful_retry_archives_complete_numerical_receipts", "archive_roundtrip_preserves_paired_state_and_costs",
                "full_truth_evaluator_accepts_protocol25_MAP_and_final_membership",
                "runtime_counts_archived_failed_and_interrupted_work_once", "runtime_status_does_not_modify_outputs",
                "frozen_configuration_reaches_real_runner", "unexecuted_main_cells_remain_in_planned_denominator",
                "offline_archived_evaluation_preserves_sources_and_contract",
                "offline_missing_or_ambiguous_evidence_unavailable_without_regeneration"])
    save_json(root / "study_checks.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--cohort-settings", type=Path, default=Path("CliPP2_phylogeny_cohort_settings.json.gz"))
    args = parser.parse_args()
    print(json.dumps(run_checks(args.outdir or tempfile.mkdtemp(prefix="clipp2-study-check-"), args.cohort_settings), indent=2))
