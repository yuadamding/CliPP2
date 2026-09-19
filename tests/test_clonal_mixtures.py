"""Real canonical multiplicity mixtures preserve support and clonal feasibility."""
import numpy as np
import pandas as pd
import pytest

from CliPP2.core.clonal import clonal_eligible_rows, validate_clonal_feasibility
from CliPP2.core.fusion import solver
from CliPP2.core.objective import has_global_supporting_tangent, has_proven_convex_observed_loss
from CliPP2.core.scalar import partition_constrained_observed_refit
from CliPP2.io.tumor_txt import load_tumor_txt, write_tumor_txt

from ._fixtures import context, options


def canonical_mixture(path, *, mixed_cn=False):
    rows = []
    for mutation in range(2):
        for region in range(2 if mixed_cn else 1):
            if not mixed_cn:
                states, alt = [(2, 1, 1.)], 45
            elif mutation == 0:
                states, alt = [(1, 1, 1.)], 30
            elif region == 0:
                # Strictly increasing likelihood makes the genuine sub-unit
                # upper bound active, so widening it would change this fit.
                states, alt = [(4, 0, .25), (1, 0, .75)], 60
            else:
                states, alt = [(2, 1, 1.)], 45
            for state, (major, minor, fraction) in enumerate(states):
                rows.append(dict(mutation_id=f'm{mutation}', sample_id=f'R{region}',
                    alt_count=alt, ref_count=60-alt, count_observed=1, purity=.8, normal_cn=2,
                    segment_id=f's{mutation}', cn_state_id=f'c{state}', cn_state_fraction=fraction,
                    allele_a_cn=major, allele_b_cn=minor))
    write_tumor_txt(path, pd.DataFrame(rows))
    return load_tumor_txt(path)


def snapshot(model):
    return {name: getattr(model, name).copy() for name in ('valid', 'slope', 'log_prior', 'lower', 'upper')}


def assert_unchanged(model, before):
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(model, name), value)


def test_actual_multiplicity_mixture_declines_reuse_then_solves_each_box(tmp_path, execution_device, monkeypatch):
    data = canonical_mixture(tmp_path/'mixture.tsv')
    original = context(data, device=execution_device)
    model, before = original.source_model, snapshot(original.source_model)
    np.testing.assert_array_equal(model.valid, np.ones((2, 1, 2), dtype=bool))
    np.testing.assert_allclose(model.slope, np.broadcast_to([2/7, 4/7], (2, 1, 2)), rtol=1e-15)
    np.testing.assert_array_equal(model.log_prior, np.full((2, 1, 2), -np.log(2.)))
    np.testing.assert_array_equal(model.upper, np.ones((2, 1)))
    assert not has_proven_convex_observed_loss(model, eps=original.eps)
    assert not has_global_supporting_tangent(model, np.ones((2, 1)), eps=original.eps)

    audit, fit_box = solver._audit_reusable_witness, solver._fit_prepared_box
    audited, solved = [], []

    def observe_audit(branch, incumbent, **kwargs):
        witness = branch.optimization_box.witness_index
        # Establish the shortcut reaches the unsupported-mixture guard, not
        # an earlier inadmissible-incumbent or inexact-membership rejection.
        assert incumbent.certificate.admissible
        np.testing.assert_array_equal(incumbent.phi[witness], [1.])
        assert branch.source_model is model and branch.graph is original.graph
        assert not has_global_supporting_tangent(model, incumbent.phi, eps=original.eps)
        covered, work, failure = audit(branch, incumbent, **kwargs)
        assert not covered and failure is None and work.full_certificate_audit_passes == 0
        audited.append(witness)
        return covered, work, failure

    def observe_solve(branch, *args, **kwargs):
        solved.append(branch.optimization_box.witness_index)
        return fit_box(branch, *args, **kwargs)

    monkeypatch.setattr(solver, '_audit_reusable_witness', observe_audit)
    monkeypatch.setattr(solver, '_fit_prepared_box', observe_solve)
    result = solver.fit_prepared(original, 2., options())
    assert audited == [1] and solved == [0, 1]
    assert result.certificate.witness_branches_reused == ()
    assert result.certificate.witness_search_complete and result.certificate.admissible
    assert result.certificate.audit_dtype == 'float64' and result.certificate.components.residual <= .004
    assert not result.certificate.global_optimum
    validate_clonal_feasibility(result.phi, model.lower, model.upper)
    assert_unchanged(model, before)


def test_real_mixed_cn_bound_below_one_keeps_separate_joint_clonal_witness(tmp_path, execution_device):
    data = canonical_mixture(tmp_path/'mixed_cn.tsv', mixed_cn=True)
    original = context(data, device=execution_device)
    model, before = original.source_model, snapshot(original.source_model)
    np.testing.assert_array_equal(data.cn_state_count, [[1, 1], [2, 1]])
    assert data.mean_total_cn[1, 0] == .25*4+.75*1 == 1.75
    assert data.major_cn[1, 0] == 4 and data.minor_cn[1, 0] == 0
    assert data.cn_filter_report.retained_mutation_count == 2
    assert not data.cn_filter_report.excluded_mutation_ids
    np.testing.assert_array_equal(model.valid.sum(axis=-1), [[1, 1], [4, 2]])
    expected_scale = .8/((1-.8)*2+.8*1.75)
    np.testing.assert_allclose(model.slope[1, 0], expected_scale*np.arange(1, 5), rtol=1e-15)
    expected_upper = (1-1e-6)/(4*expected_scale)
    assert model.upper[1, 0] == pytest.approx(expected_upper, abs=1e-15) and expected_upper < 1.
    np.testing.assert_array_equal(clonal_eligible_rows(model.lower, model.upper), [True, False])
    branch = solver._prepare_witness_problem(original, 0)
    assert branch.source_model is model and branch.graph is original.graph
    np.testing.assert_array_equal(branch.optimization_box.upper[1], model.upper[1])

    result = solver.fit_prepared(original, .1, options())
    assert result.certificate.admissible and result.certificate.witness_search_complete
    assert result.certificate.witness_branches_eligible == (0,)
    assert result.certificate.components.residual <= .004 and result.certificate.audit_dtype == 'float64'
    np.testing.assert_array_equal(result.phi[0], [1., 1.])
    np.testing.assert_array_equal(result.phi[1], model.upper[1])
    assert result.phi[1, 0] <= expected_upper < 1.
    validate_clonal_feasibility(result.phi, model.lower, model.upper)
    # The production final fixed-label refit obeys the same genuine CN domain.
    refit = partition_constrained_observed_refit(data, np.array([0, 1]), eps=1e-6, tol=1e-6, max_iter=32)
    assert refit.finite_candidate_found and refit.clonal_cluster_id == 0
    np.testing.assert_array_equal(refit.cluster_centers[0], [1., 1.])
    np.testing.assert_array_equal(refit.phi[1], model.upper[1])
    validate_clonal_feasibility(refit.phi, model.lower, model.upper)
    assert_unchanged(model, before)
    np.testing.assert_array_equal(data.phi_upper, before['upper'])
