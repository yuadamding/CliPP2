#!/usr/bin/env python3
"""Bounded attributable comparisons; never launch the full biological cohort."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import html
import io
import json
import os
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pandas as pd

from CliPP2.assignment import AssignmentBundle, assign_mutations, derive_current_bundle
from CliPP2.config import FitConfig, WorkflowConfig
from CliPP2.io.assignment import load_assignment_bundle, save_assignment_bundle
from CliPP2.io.data import json_value
from CliPP2.io.tumor_txt import write_tumor_txt
from CliPP2.simulation.experiment import run_paired_experiment
from CliPP2.workflow import run_workflow


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(json_value(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def reporting_comparison(root):
    with zipfile.ZipFile(Path(__file__).parent/'fixtures/protocol22_baseline.zip') as archive:
        frozen = AssignmentBundle.from_dict(json.loads(archive.read('assignment_bundle.json'))['payload'])
        original = pd.read_csv(io.BytesIO(archive.read('assignment/workflow_protocol22_mutation_assignments.tsv')),
                               sep='\t', float_precision='round_trip').set_index('mutation_id')
    request = derive_current_bundle(frozen)
    save_assignment_bundle(request, root/'derivative_bundle.json')
    result = assign_mutations(request, outdir=root/'assignment')
    comparisons = []
    for row in result.assignments:
        before = original.loc[row['mutation_id']]
        np.testing.assert_array_equal(row['log_scores'], [before[f'log_score_{k}'] for k in request.reference.labels])
        np.testing.assert_array_equal(row['probabilities'], [before[f'probability_{k}'] for k in request.reference.labels])
        comparisons.append(dict(mutation_id=row['mutation_id'], historical_status=before['status'],
            current_evidence_status=row['evidence_status'], final_label=row['final_assignment_label'],
            scores_exact=True, probabilities_exact=True))
    return dict(comparison='reporting_only', historical_schema=frozen.payload['schema'],
                historical_bundle_identity=frozen.content_hash, derivative_identity=request.content_hash,
                references_and_target_population_unchanged=True, records=comparisons)


def fixture(path, alt, major=1, minor=1, extra=()):
    rows = []
    for i, values in enumerate(alt):
        for j, value in enumerate(values):
            rows.append(dict(mutation_id=f'm{i:03}', sample_id=f'r{j}', alt_count=value,
                ref_count=100-value, count_observed=1, purity=1., normal_cn=2,
                segment_id=f's{major}', cn_state_id='0', cn_state_fraction=1., allele_a_cn=major, allele_b_cn=minor))
    for i, (cn, excluded, observed) in enumerate(extra):
        for j in range(len(alt[0])):
            rows.append(dict(rows[j], mutation_id=f'extra{i}', allele_a_cn=cn, allele_b_cn=0,
                segment_id=f's{cn}', input_excluded=int(excluded), count_observed=int(observed),
                alt_count=20 if observed else 0, ref_count=80 if observed else 0))
    frame = pd.DataFrame(rows)
    if 'input_excluded' in frame:
        frame['input_excluded'] = frame.input_excluded.fillna(0).astype(int)
    return write_tumor_txt(path, frame, metadata=dict(tumor_id=path.stem))


def summary(directory):
    receipt = json.loads((directory/'workflow_receipt.json').read_text())
    envelope = json.loads((directory/'assignment_bundle.json').read_text())
    payload = envelope['payload']
    table = pd.read_csv(directory/'all_original_mutations.tsv', sep='\t')
    return dict(status=receipt['status'], counts=receipt['counts'],
        reference_centers=payload['reference']['centers'], reference_priors=payload['reference']['priors'],
        selected_native_center=receipt['reference']['native_center'], target_ids=payload['target_ids'],
        non_target_ids=payload['non_target_ids'],
        final_labels={row['mutation_id']:None if pd.isna(row['final_assignment_label']) else int(row['final_assignment_label'])
                      for row in table.to_dict('records')},
        caps=pd.read_csv(directory/'stage2_segment_multiplicity_bounds.tsv', sep='\t').to_dict('records'))


def historical_workflow(python, source, input_path, output, settings, revision):
    # The external comparison harness pins its baseline; inference does not scan source.
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(source))
    code = '''import json,sys
from CliPP2.config import WorkflowConfig
from CliPP2.workflow import run_workflow
request=json.load(sys.stdin)
run_workflow(request['input'],request['output'],WorkflowConfig.from_dict(request['settings']),run_manifest_id=request['revision'])
'''
    completed = subprocess.run([str(python), '-c', code], input=json.dumps(dict(input=str(input_path), output=str(output),
        settings=settings, revision=revision)), text=True, env=environment, capture_output=True)
    (output.parent/'baseline.log').write_text(completed.stdout+completed.stderr)
    if completed.returncode:
        raise RuntimeError(f'Baseline workflow failed; see {output.parent / "baseline.log"}')


def compare(args):
    root = args.outdir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = WorkflowConfig(fit=FitConfig(max_major_cn=10, max_clusters=3, cpu_threads=1), stage_two_max_clusters=3)
    plan = dict(schema='clipp2.protocol23-comparison-plan.v1', baseline_revision=args.baseline_revision,
        current_revision=args.current_revision, fixed_population_config=asdict(config),
        cases=['diploid', 'subclonal_source', 'cap_sensitive'],
        full_support_comparator_cases=['subclonal_source', 'cap_sensitive'],
        population_expansion_case='high_cn_with_explicit_exclusion_and_missing_reads',
        reporting_only_fixture='protocol22_baseline.zip', no_performance_selection=True, full_cohort=False)
    save(root/'plan.json', plan)
    reporting = reporting_comparison(root/'reporting_only')
    patterns = dict(diploid=([[50, 50]]*4+[[20, 15]]*4, 1, 1),
        subclonal_source=([[20, 20]]*4+[[10, 10]]*4, 1, 1),
        cap_sensitive=([[100, 25], [25, 100]]*2+[[25, 25]]*16+[[12, 12]]*16, 4, 0))
    discovery, comparators = {}, {}
    for name, (values, major, minor) in patterns.items():
        case = root/name
        case.mkdir()
        path = fixture(case/'input.tsv', values, major, minor)
        historical_workflow(args.baseline_python, args.baseline_source, path, case/'baseline',
                            asdict(config)|{'stage_two_near_clonal_distance_max':.05}, args.baseline_revision)
        run_workflow(path, case/'revised', config, run_manifest_id=args.current_revision)
        before, after = summary(case/'baseline'), summary(case/'revised')
        before_bundle = load_assignment_bundle(case/'baseline/assignment_bundle.json').to_dict()
        after_bundle = load_assignment_bundle(case/'revised/assignment_bundle.json').to_dict()
        assert before_bundle['catalog']['mutation_ids'] == after_bundle['catalog']['mutation_ids']
        assert before_bundle['catalog']['eligible'] == after_bundle['catalog']['eligible']
        changed = [mutation for mutation, label in after['final_labels'].items() if label != before['final_labels'][mutation]]
        # Labels are inventory-relative, so preserve centers as well as raw label differences.
        discovery[name] = dict(comparison='workflow_on_frozen_population', population_and_masks_equal=True,
            before=before, after=after, inventory_relative_label_changes=changed)
        if name in plan['full_support_comparator_cases']:
            paired = run_paired_experiment(path, case/'full_support_comparator', config, run_manifest_id=args.current_revision)
            if paired['status'] != 'complete':
                raise RuntimeError(f'Comparator did not complete: {case}')
            comparators[name] = dict(cost_ledger=paired['cost_ledger'],
                full_support_centers=paired['arms']['B']['stage_two']['centers'],
                capped_centers=paired['arms']['C']['stage_two']['centers'],
                full_support_reference=paired['arms']['B']['reference']['centers'],
                capped_reference=paired['arms']['C']['reference']['centers'],
                interpretation='bounded native refits; no biological performance threshold')
    population = root/'population'
    population.mkdir()
    path = fixture(population/'input.tsv', patterns['diploid'][0], extra=((12,False,True),(13,True,True),(11,False,False)))
    run_workflow(path, population/'restricted', config, run_manifest_id=args.current_revision)
    expanded = WorkflowConfig(fit=FitConfig(max_clusters=3, cpu_threads=1), stage_two_max_clusters=3)
    run_workflow(path, population/'unrestricted', expanded, run_manifest_id=args.current_revision)
    limited, full = summary(population/'restricted'), summary(population/'unrestricted')
    assert limited['counts']['N1']+1 == full['counts']['N1']
    assert full['final_labels']['extra1'] is None and full['final_labels']['extra2'] is None
    population_change = dict(comparison='population_expansion', same_observed_file=True,
        before=limited, after=full, formerly_cn_excluded_fitted=['extra0'], explicit_exclusion_preserved=['extra1'],
        prior_only_outside_fitted_population=['extra2'], refitted=True)
    report = dict(schema='clipp2.protocol23-comparison.v1', status='complete', plan=plan,
        reporting_only=reporting, frozen_population=discovery, population_expansion=population_change,
        full_support_comparators=comparators,
        interpretation='Implementation fixtures, not TRACERx cohort outcomes or evidence of biological superiority.')
    save(root/'comparison.json', report)
    rows = ''.join('<tr>'+''.join(f'<td>{html.escape(str(value))}</td>' for value in
        (name, data['before']['counts']['targets'], data['after']['counts']['targets'],
         data['before']['reference_centers'],data['after']['reference_centers']))+'</tr>' for name,data in discovery.items())
    (root/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>Protocol 2.3 comparison</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:40px auto}td,th{padding:12px;border:1px solid #ccc}'
        'table{border-collapse:collapse}</style><h1>Reference-preserving Protocol 2.3</h1>'
        '<p>Fixed-reference reporting preserves scores and probabilities exactly. The following comparisons use the same original population and masks.</p>'
        '<table><tr><th>Case</th><th>Previous targets</th><th>Current targets</th><th>Previous references</th><th>Current references</th></tr>'+rows+'</table>'
        '<p>Population expansion retains one additional informative major-CN12 mutation, preserves the explicit exclusion, and records the missing-read case as prior-only. Both population policies were refitted.</p>'
        '<p>Two predeclared full-support Stage 2 comparators completed. See <a href="comparison.json">full-precision results</a> and <a href="plan.json">fixed plan</a>.</p>'
        '<p>These are bounded implementation fixtures, not cohort-wide biological performance claims. No cohort was launched.</p>')
    return dict(status='complete', report=str(root/'comparison.json'), html=str(root/'comparison.html'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--baseline-python', type=Path, required=True)
    parser.add_argument('--baseline-source', type=Path, required=True, help='Frozen prior source package directory (parent of CliPP2)')
    parser.add_argument('--baseline-revision', required=True)
    parser.add_argument('--current-revision', required=True)
    print(json.dumps(compare(parser.parse_args()), indent=2))


if __name__=='__main__':
    main()
