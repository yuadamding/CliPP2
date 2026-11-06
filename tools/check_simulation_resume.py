"""Mock-only checks for pilot retry and optional-panel attempt attribution."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from CliPP2.simulation.design import make_contract
from CliPP2.simulation.study import (_execute, _latest_completion, read_json, run_optional_panels,
                                    save_json)


class _Allocated(Exception):
    pass


def _pilot_retry(root):
    contract = make_contract(mutations=12, segments=2, regions=2, workers=1,
                             budget_minutes=60, main_histories=1, development_histories=1)
    forwarded = []

    def controls(output):
        save_json(output / 'evidence.json', dict(checked=True))
        save_json(output / 'controls_summary.json', dict(status='passed', paths=['evidence.json'],
            cost_ledger=dict(fit_count=0, assignment_pass_count=0, events=[])))

    def run_jobs(directory, jobs, *, retry_failed=False, **kwargs):
        forwarded.append(retry_failed)
        rows = []
        for job in jobs:
            assert job['phase'] == 'pilot'
            cell = directory / 'cells' / job['cell_id']
            prior = _latest_completion(cell)
            if prior is not None and not retry_failed:
                rows.append(prior)
                continue
            number = 1 if prior is None else 2
            attempt = cell / f'attempt-{number:03}'
            value = dict(job=job, status='complete' if retry_failed else 'execution_failed',
                attempt=str(attempt.relative_to(directory)), elapsed_seconds=1., output_bytes=100,
                cost_ledgers=[dict(fit_count=0, assignment_pass_count=0)])
            save_json(attempt / 'evidence.json', value)
            save_json(cell / ('completion.json' if number == 1 else 'completion.002.json'), value)
            rows.append(value)
        return rows

    with patch('CliPP2.simulation.biology.generate_history', return_value=SimpleNamespace(validate=lambda: True)), \
            patch('CliPP2.simulation.controls.run_controls', side_effect=controls), \
            patch('CliPP2.simulation.study.run_jobs', side_effect=run_jobs), \
            patch('CliPP2.simulation.study._freeze_main_roster', side_effect=_Allocated):
        for resume in (False, True):
            try:
                _execute(root, contract, resume=resume)
            except RuntimeError as error:
                assert 'Timing pilot execution failed' in str(error)
            else:
                raise AssertionError('Failed pilot permitted allocation')
            assert not (root / 'allocation.json').exists()
        original = {str(path.relative_to(root)): path.read_bytes()
                    for path in (root / 'cells').rglob('*.json')}
        try:
            _execute(root, contract, resume=True, retry_failed=True)
        except _Allocated:
            pass
        else:
            raise AssertionError('Successful retried pilot did not reach allocation')
    assert forwarded == [False, False, True]
    assert (root / 'allocation.json').exists()
    assert all((root / name).read_bytes() == value for name, value in original.items())
    for directory in (root / 'cells').iterdir():
        value = _latest_completion(directory)
        assert value['status'] == 'complete' and value['attempt'].endswith('attempt-002')
    return dict(default_keeps_failure=True, explicit_retry_reaches_allocation=True, prior_evidence_unchanged=True)


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
    return dict(status='passed', pilot=_pilot_retry(root / 'pilot'),
                optional=_optional_retries(root / 'optional'), fit_count=0, assignment_pass_count=0)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(run_checks(parser.parse_args().output), indent=2))
