#!/usr/bin/env python3
"""Deterministic scientific evaluator contracts; no inferred biological calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy, xlog1py

from CliPP2.simulation.evaluation import (aggregate, candidate_correspondence, cap_audit,
    information_diagnostics, mapped_probabilities, partition_scores, probability_scores,
    recovery_reach, reference_center_diagnostics, transition, json_safe)
from CliPP2.simulation.evaluation import log_probability_mass


def _emission(alt, depth, major, phi, upper):
    p = np.arange(1, upper+1)*phi/major
    logs = (gammaln(depth+1)-gammaln(alt+1)-gammaln(depth-alt+1)
            + xlogy(alt, p)+xlog1py(depth-alt, -p))
    return float(np.exp(logsumexp(logs)-np.log(upper))), np.exp(logs), np.exp(logs-logsumexp(logs))


def run_checks():
    full, masses, probabilities = _emission(2, 3, 4, 1., 4)
    cap, _, _ = _emission(2, 3, 4, 1., 2)
    np.testing.assert_allclose(masses, [.140625, .375, .421875, 0.], rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose([full, cap, probabilities[2]], [15/64, 33/128, .45], rtol=1e-14)
    a, ma, _ = _emission(1, 2, 2, 1., 2)
    b, mb, _ = _emission(1, 2, 2, .6, 2)
    np.testing.assert_allclose([a/(a+b), b/(a+b)], [5/14, 9/14], rtol=1e-14)
    assert a < b and ma.max() > mb.max()
    # Different acquisition branches with equal sampled vectors are one observable class.
    assert partition_scores(['same', 'same', 'other'], [2, 2, 7])['ari'] == 1.
    assert partition_scores([], [])['pair_precision'] is None
    assert partition_scores(['a', 'b'], [0, 0])['pair_precision'] == 0.
    assert partition_scores(['a', 'b'], [0, 1])['pair_precision'] is None
    mapping, audit = candidate_correspondence((0, 1, 2, 3), 0,
        {0: ('a',), 1: ('a', 'b', 'c', 'd', 'e'), 2: ('f', 'g'), 3: ('b',)},
        {'a': 'trunk', 'b': 'other', 'c': 'other', 'd': 'other', 'e': 'other', 'f': 'x', 'g': 'y'}, 'trunk')
    assert mapping == {0: 'trunk', 1: 'other', 2: None, 3: 'other'}
    assert audit[2]['source_purity'] == .5
    # Pure source overlap cannot establish correct geometric center recovery.
    geometric = reference_center_diagnostics((1,), [[.9, .9]], {'truth': [.4, .4]})
    assert geometric['recovered_count'] is None and geometric['reference_recall'] is None
    assert geometric['status'] == 'unassessed_no_declared_tolerance'
    np.testing.assert_allclose(geometric['mean_one_to_one_l2_error'], np.sqrt(.5))
    rejected = reference_center_diagnostics((1,), [[.9, .9]], {'truth': [.4, .4]}, match_atol=.05)
    assert rejected['reference_recall'] == 0
    one = reference_center_diagnostics((7,), [[.5, .5]], {'a': [.4, .4], 'b': [.6, .6]}, match_atol=.11)
    assert one['recovered_count'] == 1 and one['reference_recall'] == .5
    # Maximum qualified cardinality is evaluated independently of overlap labels.
    two = reference_center_diagnostics((7, 8), [[.5, .5], [.4, .4]],
                                       {'a': [.4, .4], 'b': [.6, .6]}, match_atol=.11)
    assert two['recovered_count'] == 2 and two['reference_recall'] == 1
    q = mapped_probabilities([.1, .2, .3, .4], (0, 1, 2, 3), mapping, ('trunk', 'other', 'absent'))
    np.testing.assert_allclose([q['other'], q['__unmapped__']], [.6, .3])
    assert np.isposinf(probability_scores(q, None, truth_class='absent')['log_loss'])
    assert log_probability_mass([0., -1000.], [1]) == -1000.
    assert probability_scores(1., False, true_log_probability=-1000.)['log_loss'] == 1000.
    # Exact aliases survive arbitrary depth and vanish only in distinguishing regions.
    phi = np.array([[1., 1.], [.5, .65]])
    multiplicity = np.array([[1., 1.], [2., 2.]])
    args = (phi, multiplicity, np.full((2, 2), 2), np.ones((2, 2)), np.ones((2, 2)),
            np.full((2, 2), 2), np.full((2, 2), 100), np.ones((2, 2), bool), ('clonal', 'sub'))
    distinct = information_diagnostics(*args)
    assert not distinct[0]['exact_alias'] and distinct[0]['expected_kl'] > 0
    masked_args = list(args)
    masked_args[7] = np.array([[1, 0], [1, 0]], bool)
    masked = information_diagnostics(*masked_args)
    assert all(row['exact_alias'] for row in masked)
    # Fixed contributor IDs cannot silently discard a contributor with zero carriers.
    h = SimpleNamespace(mutation_ids=('source', 'target'), region_ids=('r',),
        phi=np.array([[0.], [1.]]), multiplicity=np.array([[np.nan], [2.]]))
    catalog = SimpleNamespace(eligible=np.ones((2, 1), bool), segment_ids=np.array([['s'], ['s']]),
        major_cn=np.full((2, 1), 2), count_observed=np.ones((2, 1), bool), total_counts=np.full((2, 1), 100))
    rows = cap_audit(h, catalog, ('target',), [dict(mutation_id='source', region_id='r', segment_id='s', usable=True)], np.ones((2, 1)))
    assert rows[0]['additional_exclusion'] and not rows[0]['truth_replacement_evaluable']
    assert rows[0]['truth_replacement_upper'] is None
    h.phi[0, 0], h.multiplicity[0, 0] = 1., 1.
    rows = cap_audit(h, catalog, ('target',), [dict(mutation_id='source', region_id='r', segment_id='s', usable=True)], np.ones((2, 1)))
    assert rows[0]['truth_replacement_exclusion'] and not rows[0]['harmful_minimum_call_error']
    assert [transition(x, y) for x, y in ((True, True), (False, True), (False, False),
        (True, False), (True, None), (False, None))] == ['correct_retained', 'error_corrected',
        'error_persisted', 'error_introduced', 'correct_resolution_lost', 'error_withheld']
    states = {'B': dict(available=True, baseline={'t': True, 'f': True}),
        'C': dict(available=True, assignment_available=True, baseline={'t': False, 'f': False},
                  targets={'t'}, non_targets={'f'}, final={'t': True, 'f': False})}
    reach = recovery_reach(states, np.ones(2, bool), {'t': 0, 'f': 1})
    assert reach['H'] == ['f', 't'] and reach['recovery_ceiling'] == {'numerator': 1, 'denominator': 2}
    assert reach['recovered_H']['numerator'] == 1
    # Different mutation/acquisition counts never redefine independent histories.
    cells = []
    for family, histories in (('S01', [(0, [0., 0.]), (1, [1.])]), ('S02', [(0, [1.]), (1, [1.])])):
        for history, values in histories:
            for acquisition, value in enumerate(values):
                cells.append(dict(family=family, history_id=str(history), acquisition_id=str(acquisition),
                    metrics=[dict(name='score', arm='B', value=value), dict(name='score', arm='C', value=value+.1)]))
    summary = aggregate(cells, bootstrap_replicates=100, seed=1, families=('S01', 'S02'))
    b = next(row for row in summary['metrics'] if row['arm'] == 'B')
    paired = next(row for row in summary['metrics'] if row['arm'] == 'C_minus_B')
    assert b['equal_family_mean'] == .75
    np.testing.assert_allclose([paired['equal_family_mean'], paired['lower_95'], paired['upper_95']], .1)
    incomplete = aggregate(cells, bootstrap_replicates=0)
    assert all(row['equal_family_mean'] is None for row in incomplete['metrics'])
    report = dict(schema='clipp2.simulation.evaluation-checks.v1', full_emission=full, capped_emission=cap,
        conditional_map=3, conditional_probability=probabilities[2], marginalization_reverses_profile=True,
        aliases_respect_observed_regions=True, fixed_contributor_identity=True, cap_counterexample=True,
        no_probability_renormalization=True, absent_truth_log_loss='Infinity', recovery_ceiling=True,
        observable_partition=True, history_level_equal_family_aggregation=True, unavailable_families_explicit=True)
    report.update(continuous_reference_geometry=True, binary_recovery_requires_declared_tolerance=True,
                  one_to_one_reference_recall=True)
    json.dumps(json_safe(report), allow_nan=False)
    return json_safe(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run_checks()
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
