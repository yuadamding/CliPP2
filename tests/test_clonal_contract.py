"""The only new prior is an occupied, joint all-region CCF-one cluster."""
from dataclasses import replace

import numpy as np
import pytest

from CliPP2.config import FitConfig, _resolve_fit_options
from CliPP2.core import scalar
from CliPP2.core.bic import fixed_partition_dirichlet_score
from CliPP2.core.clonal import (
    ClonalConstraintInfeasibleError, clonal_eligible_rows, clonal_members,
    make_clonal_witness_bounds, validate_clonal_feasibility,
)
from CliPP2.core.fusion import partition_starts, solver
from CliPP2.core.objective import compile_observed_model, observed_terms_numpy
from CliPP2.model_selection.candidates import evaluate_partition
from CliPP2.model_selection.partitions import extract_certified_fusion_partition

from ._fixtures import context, options, tumor, two_blocks


def refit(data, labels, guide=False, **kwargs):
    method = (partition_starts._fit_guide_centers if guide
              else scalar.partition_constrained_observed_refit)
    if guide:
        kwargs['_model'] = compile_observed_model(data, eps=1e-6)
    return method(data, labels, eps=1e-6, tol=1e-6, max_iter=32, **kwargs)


def test_joint_existence_is_exact_without_restricting_other_centers():
    separate = np.array([[1., .6], [.8, 1.]])
    with pytest.raises(ValueError):
        validate_clonal_feasibility(separate, np.zeros_like(separate), np.ones_like(separate))
    phi = np.array([[1., 1.], [1., .6], [np.nextafter(1., 0.), 1.]])
    np.testing.assert_array_equal(clonal_members(phi), [True, False, False])
    validate_clonal_feasibility(phi, np.zeros_like(phi), np.ones_like(phi))


def test_eligibility_uses_original_float64_domain_and_never_expands_it(tmp_path):
    upper = np.array([[1., np.nextafter(1., 0.)], [.8, 1.]])
    assert np.float32(upper[0, 1]) == 1.
    before = upper.copy()
    assert not clonal_eligible_rows(np.zeros_like(upper), upper).any()
    with pytest.raises(ClonalConstraintInfeasibleError):
        make_clonal_witness_bounds(np.zeros_like(upper), upper, 0)
    data = replace(tumor(tmp_path/'ineligible.tsv'), phi_upper=upper)
    with pytest.raises(ClonalConstraintInfeasibleError):
        context(data, dtype='float32')
    np.testing.assert_array_equal(data.phi_upper, before)


def test_missing_counts_do_not_relax_all_region_equality(tmp_path):
    data = tumor(tmp_path/'missing.tsv', ((24., 0.),))
    data = replace(data, count_observed=np.array([[True, False]]))
    result = refit(data, np.array([0]))
    np.testing.assert_array_equal(result.phi, [[1., 1.]])
    score = fixed_partition_dirichlet_score(loglik=result.loglik, num_clusters=1,
        data=data, labels=np.array([0]), partition_signature='singleton')
    assert score.n_eff == 1 and score.degrees_of_freedom == 2


def test_profile_matches_independent_mle_enumeration_and_preserves_free_centers(tmp_path):
    data = tumor(tmp_path/'profile.tsv', ((8., 12.), (23., 24.), (30., 12.)))
    labels = np.arange(3)
    # Independent singleton binomial MLE: p=.4*phi and depth=60.
    free = np.clip(data.alt_counts/24., 1e-6, 1.)
    candidates = []
    for fixed in range(3):
        centers = free.copy()
        centers[fixed] = 1.
        p = .4*centers
        loss = -np.sum(data.alt_counts*np.log(p)+(60-data.alt_counts)*np.log1p(-p))
        candidates.append((float(loss), fixed, centers))
    expected_loss, expected_fixed, expected = min(candidates, key=lambda item: item[0])
    actual = refit(data, labels)
    assert actual.clonal_cluster_id == expected_fixed == 1
    assert actual.fit_loss == pytest.approx(expected_loss, abs=2e-7)
    np.testing.assert_allclose(actual.cluster_centers, expected, atol=2e-5, rtol=0)
    np.testing.assert_array_equal(actual.phi, actual.cluster_centers[actual.labels])
    assert actual.cluster_centers[2, 0] == 1. and actual.cluster_centers[2, 1] < 1.


def test_same_labels_ignore_raw_witness_and_retain_original_score_penalty(tmp_path):
    data = tumor(tmp_path/'witnesses.tsv', ((24., 24.), (24., 24.)))
    original = context(data)
    evaluations = []
    cache = {}
    for witness in (0, 1):
        raw = solver._fit_prepared_box(solver._prepare_witness_problem(original, witness),
                                      .1, options())
        partition = extract_certified_fusion_partition(raw, graph=original.graph_spec,
            tolerance=8e-4, mutation_ids=data.mutation_ids)
        evaluations.append(evaluate_partition(data=data, partition=partition,
            selection_options=_resolve_fit_options(FitConfig(device='cpu')), refit_cache=cache))
    first, second = evaluations
    np.testing.assert_array_equal(first.refit.labels, second.refit.labels)
    np.testing.assert_array_equal(first.refit.phi, second.refit.phi)
    assert first.score == second.score
    ordinary_loss = observed_terms_numpy(compile_observed_model(data, eps=1e-6),
                                        np.ones((2, 2)), eps=1e-6).loss.sum()
    k = len(np.unique(first.refit.labels))
    ordinary = fixed_partition_dirichlet_score(loglik=-float(ordinary_loss), num_clusters=k,
        data=data, labels=first.refit.labels, partition_signature=first.score.partition_signature)
    assert first.score.degrees_of_freedom == k*2
    assert first.score.penalty == ordinary.penalty
    assert first.score.value-ordinary.value == pytest.approx(
        2*(-first.refit.loglik-ordinary_loss), abs=1e-10)


@pytest.mark.parametrize('guide', [False, True])
def test_single_cluster_never_runs_a_free_optimizer(tmp_path, monkeypatch, guide):
    data = two_blocks(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('The sole occupied center is fixed, not freely optimized.')
    monkeypatch.setattr(scalar, '_fit_coordinate', forbidden)
    monkeypatch.setattr(partition_starts, 'certify_scalar_minimum', forbidden)
    result = refit(data, np.zeros(2, dtype=int), guide)
    assert result.finite_candidate_found and result.free_fit_failures == ()
    np.testing.assert_array_equal(result.phi, np.ones((2, 2)))


@pytest.mark.parametrize('guide', [False, True])
@pytest.mark.parametrize('failure', ['returned', 'raised'])
def test_only_the_fixed_block_can_recover_its_failed_free_coordinate(tmp_path, monkeypatch, guide, failure):
    data = two_blocks(tmp_path)
    baseline = refit(data, np.array([0, 1]), guide)
    module, name = ((partition_starts, 'certify_scalar_minimum') if guide
                    else (scalar, '_fit_coordinate'))
    original = getattr(module, name)
    calls = []
    def fail_one(problem, **kwargs):
        calls.append(float(problem.alt[0]))
        if problem.alt[0] == 18. and failure == 'raised':
            raise FloatingPointError('controlled numerical failure')
        value = original(problem, **kwargs)
        if problem.alt[0] == 18.:
            return (replace(value, argmin=np.nan, attained_value=np.inf) if guide
                    else replace(value, beta=np.nan, loss=np.inf, finite_candidate_found=False))
        return value
    monkeypatch.setattr(module, name, fail_one)
    cache = scalar._RefitCoordinateCache()
    result = refit(data, np.array([0, 1]), guide, _coordinate_cache=cache)
    assert result.finite_candidate_found and result.clonal_cluster_id == 0
    np.testing.assert_array_equal(result.cluster_centers[0], [1., 1.])
    np.testing.assert_array_equal(result.cluster_centers[1], baseline.cluster_centers[1])
    assert result.fit_loss == baseline.fit_loss
    if failure == 'raised':
        assert result.free_fit_failures == ((0, 1, 'FloatingPointError: controlled numerical failure'),)
        calls.clear()
        refit(data, np.array([0, 1]), guide, _coordinate_cache=cache)
        assert calls == [18.]  # Interrupted coordinates never contaminate cache.


@pytest.mark.parametrize('guide', [False, True])
@pytest.mark.parametrize('cause', ['two_failures', 'ineligible_fixed_block'])
def test_unavailable_nonclonal_blocks_are_not_hidden(tmp_path, monkeypatch, guide, cause):
    data = two_blocks(tmp_path)
    if cause == 'ineligible_fixed_block':
        data = replace(data, phi_upper=np.array([[.9, 1.], [1., 1.]]))
    module, name = ((partition_starts, 'certify_scalar_minimum') if guide
                    else (scalar, '_fit_coordinate'))
    original = getattr(module, name)
    def broken(problem, **kwargs):
        if problem.alt[0] == 18. or (cause == 'two_failures' and problem.alt[0] == 9.):
            raise FloatingPointError('unavailable free coordinate')
        return original(problem, **kwargs)
    monkeypatch.setattr(module, name, broken)
    result = refit(data, np.array([0, 1]), guide)
    assert not result.finite_candidate_found and np.isinf(result.fit_loss)


@pytest.mark.parametrize('guide', [False, True])
@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, MemoryError])
def test_model_programming_and_resource_errors_propagate(tmp_path, monkeypatch, guide, error_type):
    data = two_blocks(tmp_path)
    module, name = ((partition_starts, 'certify_scalar_minimum') if guide
                    else (scalar, '_fit_coordinate'))
    error = error_type('not a recoverable arithmetic failure')
    def broken(*args, **kwargs):
        raise error
    monkeypatch.setattr(module, name, broken)
    with pytest.raises(error_type) as caught:
        refit(data, np.array([0, 1]), guide)
    assert caught.value is error
