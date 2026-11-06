"""Mock-only checks for frozen-roster retry and optional-panel attempt attribution."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from CliPP2.simulation.design import make_contract
from CliPP2.simulation.study import (_execute, _latest_completion, read_json, run_optional_panels,
                                    save_json)


class _DevelopmentIncomplete(Exception):
    pass


def _roster_retry(root):
    contract = make_contract(mutations=12, segments=2, regions=2, workers=1,
                             main_replicates=1, development_replicates=1)
    contract["families"] = ["S01"]
    forwarded = []

    def controls(output):
        save_json(output / "evidence.json", dict(checked=True))
        save_json(output / "controls_summary.json", dict(status="passed", paths=["evidence.json"],
            cost_ledger=dict(fit_count=0, assignment_pass_count=0, events=[])))

    def run_jobs(directory, jobs, *, retry_failed=False, **kwargs):
        rows = []
        job_list = list(jobs)
        forwarded.append((job_list[0]["phase"], retry_failed))
        for job in job_list:
            cell = directory / "cells" / job["cell_id"]
            prior = _latest_completion(cell)
            if prior is not None and (prior["status"] == "complete" or not retry_failed):
                rows.append(prior)
                continue
            number = 1 if prior is None else 2
            attempt = cell / f"attempt-{number:03}"
            value = dict(job=job, status="complete" if retry_failed or job["phase"] == "main" else "execution_failed",
                attempt=str(attempt.relative_to(directory)), elapsed_seconds=1., output_bytes=100,
                cost_ledgers=[dict(fit_count=0, assignment_pass_count=0)])
            save_json(attempt / "evidence.json", value)
            save_json(cell / ("completion.json" if number == 1 else "completion.002.json"), value)
            rows.append(value)
        return rows

    def selection(directory, cells, configuration):
        if any(cell["status"] != "complete" for cell in cells):
            raise _DevelopmentIncomplete
        return dict(selected=dict(contributor=.95, posterior=.95), configuration_scores=[])

    with patch("CliPP2.simulation.biology.generate_history", return_value=SimpleNamespace(validate=lambda: True)), \
            patch("CliPP2.simulation.controls.run_controls", side_effect=controls), \
            patch("CliPP2.simulation.study.run_jobs", side_effect=run_jobs), \
            patch("CliPP2.simulation.study.select_thresholds", side_effect=selection), \
            patch("CliPP2.simulation.study.run_optional_panels"), \
            patch("CliPP2.simulation.study.report", return_value=dict(status="complete")):
        for resume in (False, True):
            try:
                _execute(root, contract, resume=resume)
            except _DevelopmentIncomplete:
                pass
            else:
                raise AssertionError("Failed development was silently replaced")
            assert (root / "main_roster.json").exists()
            assert not (root / "threshold_development.json").exists()
        original = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*.json")}
        _execute(root, contract, resume=True, retry_failed=True)
        receipt = read_json(root / "threshold_development.json")
        assert receipt["selected_workflow_config"]["assignment"]["posterior_min"] == .95
        with patch("CliPP2.simulation.study.select_thresholds", side_effect=AssertionError("Frozen settings reselected")):
            _execute(root, contract, resume=True)
    assert forwarded[:3] == [("development", False), ("development", False), ("development", True)]
    assert all((root / name).read_bytes() == value for name, value in original.items())
    assert not (root / "allocation.json").exists() and not (root / "study_plan.json").exists()
    from CliPP2.simulation.runtime import status
    summary = status(root)
    assert summary["thresholds_frozen"] and summary["planned_main_cells"] == 4
    assert summary["planned_development_cells"] == 2 and summary["conditional_development_extension_cells"] == 2
    return dict(default_keeps_failure=True, explicit_retry_preserves_roster=True,
                prior_evidence_unchanged=True, thresholds_reused_from_authoritative_receipt=True)


def _development_blocks(root):
    from CliPP2.simulation.design import jobs
    outcomes = []
    for qualified in (True, False):
        directory = root / ("qualified" if qualified else "rejected")
        contract = make_contract(mutations=12, segments=2, regions=2, workers=1,
                                 main_replicates=1, development_replicates=1)
        contract["families"] = ["S01"]
        initial = list(jobs(contract, "development"))
        extension = list(jobs(contract, "development", extension=True))
        assert {j["history_id"] for j in initial}.isdisjoint(j["history_id"] for j in extension)
        expected = [initial, extension]
        seen = []

        def controls(output):
            save_json(output / "controls_summary.json", dict(status="passed", paths=[],
                cost_ledger=dict(fit_count=0, assignment_pass_count=0, events=[])))

        def run_jobs(output, population, **kwargs):
            values = list(population)
            seen.append(values)
            assert read_json(output / "main_roster.json")["jobs"] == list(jobs(contract, "main"))
            return [dict(job=job, status="complete") for job in values]

        selections = [dict(selected=None), dict(selected=dict(contributor=.95, posterior=.95) if qualified else None)]
        with patch("CliPP2.simulation.biology.generate_history", return_value=SimpleNamespace(validate=lambda: True)), \
                patch("CliPP2.simulation.controls.run_controls", side_effect=controls), \
                patch("CliPP2.simulation.study.run_jobs", side_effect=run_jobs), \
                patch("CliPP2.simulation.study.select_thresholds", side_effect=selections), \
                patch("CliPP2.simulation.study.run_optional_panels"), \
                patch("CliPP2.simulation.study.report", return_value=dict(status="complete")):
            if qualified:
                _execute(directory, contract)
                assert seen == expected + [list(jobs(contract, "main"))]
            else:
                try:
                    _execute(directory, contract)
                except RuntimeError as error:
                    assert "No development setting qualified" in str(error)
                else:
                    raise AssertionError("Rejected development allowed main execution")
                assert seen == expected
                assert read_json(directory / "threshold_development.json")["selected"] is None
        outcomes.append(dict(qualified=qualified, development_blocks=2, main_executed=qualified))
    return outcomes


def _optional_retries(root):
    root.mkdir(parents=True)
    contract = make_contract(mutations=12, segments=2, regions=2, workers=1)
    selected = dict(contributor=.95, posterior=.95)
    contract['optional_panels']['technical_sensitivity'] = True
    with patch('CliPP2.simulation.panels.sensitivity_jobs', return_value=[{'technical': True}]), \
            patch('CliPP2.simulation.study.run_jobs') as runner:
        run_optional_panels(root, contract, selected, 1)
        run_optional_panels(root, contract, selected, 1, retry_failed=True)
    assert [call.kwargs['retry_failed'] for call in runner.call_args_list] == [False, True]
    contract['optional_panels'].update(technical_sensitivity=False, external_reference=True)
    cell_id = 'main-S03-H001-A4'
    directory = root / 'cells' / cell_id
    job = dict(cell_id=cell_id, family='S03')
    first = dict(job=job, attempt=f'cells/{cell_id}/attempt-001', status='complete', cost_ledgers=[{}])
    second = dict(first, attempt=f'cells/{cell_id}/attempt-002')
    save_json(directory / 'completion.json', first)
    calls = []

    def audit(history, catalog, paired, config, output, **kwargs):
        calls.append(paired['selected_attempt'])
        save_json(output / 'reference_audit.json', dict(status='passed', selected_attempt=paired['selected_attempt']))

    with patch('CliPP2.simulation.study._history', return_value=object()), \
            patch('CliPP2.simulation.study.load_original_catalog', return_value=object()), \
            patch('CliPP2.simulation.study._paired_result', side_effect=lambda p: dict(selected_attempt=p.name)), \
            patch('CliPP2.simulation.controls.run_reference_audit', side_effect=audit):
        run_optional_panels(root, contract, selected, 1)
        audit_root = root / 'optional_panels/reference_audits/S03'
        original = {str(path.relative_to(root)): path.read_bytes() for path in audit_root.rglob('*.json')}
        save_json(directory / 'completion.002.json', second)
        run_optional_panels(root, contract, selected, 1, retry_failed=True)
        run_optional_panels(root, contract, selected, 1, retry_failed=True)
    assert calls == ['attempt-001', 'attempt-002']
    assert all((root / name).read_bytes() == value for name, value in original.items())
    newest = read_json(audit_root / 'attempt-002/source_attempt.json')
    assert newest['selected_attempt'] == second['attempt']
    assert read_json(audit_root / 'attempt-002/reference_audit.json')['source_attempt'] == newest
    return dict(technical_retry_forwarded=True, reference_audit_uses_latest_attempt=True,
                matching_attempt_reused=True, earlier_audit_retained=True)


def run_checks(root):
    root = Path(root)
    return dict(status='passed', roster=_roster_retry(root / 'roster'),
                development_blocks=_development_blocks(root / 'development_blocks'),
                optional=_optional_retries(root / 'optional'), fit_count=0, assignment_pass_count=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(run_checks(parser.parse_args().output), indent=2))
