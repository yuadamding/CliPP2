"""Independent small oracles and real CPU/allocated-CUDA certificate paths."""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
import torch

from CliPP2.config import FitConfig, _resolve_fit_options
from CliPP2.core.fusion import solver
from CliPP2.core.fusion.graph import build_likelihood_noise_regularized_adaptive_graph
from CliPP2.core.fusion.graph_ops import (
    build_likelihood_noise_regularized_adaptive_tensor_graph, tensor_graph_to_pairwise_graph,
)
from CliPP2.core.fusion.types import KKTDiagnostics, WorkCounters
from CliPP2.core.objective import observed_terms_numpy
from CliPP2.model_selection.proposals import build_partition_guided_graph_with_resource_policy

from ._fixtures import context, options, tumor


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_adaptive_graph_preserves_original_bounds_and_ignores_witness(tmp_path, execution_device, dtype):
    data = tumor(tmp_path/'graph.tsv', ((12.,),)*5)
    data = replace(data, phi_upper=np.array([[1.], [1.], [.72], [.72], [.72]]))
    original = context(data, device=execution_device, dtype=dtype, initialize=False)
    phi = torch.linspace(.1, .7, 5, dtype=original.runtime.dtype, device=execution_device)[:, None]
    curvature = torch.full_like(phi, .01)  # All variance caps use interval widths.
    fit_options = _resolve_fit_options(FitConfig(device=execution_device))
    kwargs = dict(gamma=fit_options.graph.adaptive_weight_gamma,
        minimum_tau=max(fit_options.graph.adaptive_weight_floor, fit_options.eps),
        baseline=fit_options.graph.adaptive_weight_baseline, noise_divisor=4)
    # Independently reconstruct the original round-to-nearest graph inputs.
    lower = torch.as_tensor(original.source_model.lower.copy(), dtype=original.runtime.dtype, device=execution_device)
    upper = torch.as_tensor(original.source_model.upper.copy(), dtype=original.runtime.dtype, device=execution_device)
    if dtype == 'float32':
        assert original.upper[2, 0].item() < .72 < upper[2, 0].item()
    if execution_device == 'cuda':
        expected, tau = build_likelihood_noise_regularized_adaptive_tensor_graph(
            phi, curvature, original.runtime, lower=lower, upper=upper,
            count_observed=original.model.observed, **kwargs)
        expected = tensor_graph_to_pairwise_graph(expected)
    else:
        expected, tau = build_likelihood_noise_regularized_adaptive_graph(
            phi.numpy(), curvature.numpy(), lower=lower.numpy(), upper=upper.numpy(),
            count_observed=original.model.observed.numpy(), **kwargs)
    for witness in (None, 0, 1):
        branch = original if witness is None else solver._prepare_witness_problem(original, witness)
        actual, _, actual_tau = build_partition_guided_graph_with_resource_policy(
            guide_phi=phi, guide_curvature=curvature, solver_context=branch,
            fit_options=fit_options, noise_divisor=4)
        assert actual_tau == tau
        for name in ('edge_u', 'edge_v', 'edge_w'):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize('lambda_value', [0., .1])
def test_witness_search_matches_independent_binomial_enumeration(tmp_path, execution_device, lambda_value):
    data = tumor(tmp_path/'oracle.tsv', ((6.,), (18.,)))
    def loss(phi):
        p = .4*np.asarray(phi)
        counts = np.array([6., 18.])
        return float(-np.sum(counts*np.log(p)+(60-counts)*np.log1p(-p))
                     + lambda_value*abs(phi[0]-phi[1]))
    candidates = []
    for witness in (0, 1):
        def objective(free):
            phi = np.ones(2)
            phi[1-witness] = free
            return loss(phi)
        best = minimize_scalar(objective, bounds=(1e-6, 1.), method='bounded',
                               options={'xatol': 1e-13})
        candidates.append((best.fun, witness))
    expected_loss, expected_witness = min(candidates)
    actual = solver.fit_prepared(context(data, device=execution_device), lambda_value, options())
    assert actual.objective.total == pytest.approx(expected_loss, abs=2e-5)
    assert actual.provenance.witness_index == expected_witness
    np.testing.assert_array_equal(actual.phi[expected_witness], [1.])
    assert actual.certificate.admissible and actual.certificate.conditional_kkt_certified
    assert actual.certificate.audit_dtype == 'float64'
    assert actual.certificate.components.residual <= .004
    assert actual.certificate.witness_search_complete


def reusable(tmp_path, device):
    # Strictly active upper bounds: the free MLE is 1.25, not exactly 1 with
    # zero gradient, so this fixture genuinely exercises exact-point reuse.
    return context(tumor(tmp_path/'reuse.tsv', ((30.,),)*4), device=device)


def test_accepted_reuse_matches_exhaustive_solves_without_inventing_globality(tmp_path, execution_device, monkeypatch):
    original = reusable(tmp_path, execution_device)
    fast = solver.fit_prepared(original, 2., options())
    assert fast.certificate.witness_branches_reused == (1, 2, 3)
    assert fast.certificate.witness_reuse_basis == 'fresh_float64_kkt_with_global_support'
    assert fast.certificate.witness_search_complete and fast.certificate.witness_search_work_complete
    assert fast.work.full_certificate_audit_passes >= 3
    monkeypatch.setattr(solver, '_audit_reusable_witness', lambda *a, **k: (False, WorkCounters(), None))
    exhaustive = solver.fit_prepared(original, 2., options())
    assert exhaustive.certificate.witness_branches_reused == ()
    np.testing.assert_array_equal(fast.phi, exhaustive.phi)
    assert fast.objective == exhaustive.objective
    assert not fast.certificate.global_optimum
    assert fast.provenance.global_optimality_basis == 'not_certified'


def test_failed_new_box_audit_triggers_real_ordinary_solves(tmp_path, execution_device, monkeypatch):
    original = reusable(tmp_path, execution_device)
    audit, terminal, fit_box = (solver._audit_reusable_witness,
        solver._terminal_backward_error_audit_float64, solver._fit_prepared_box)
    called, audited = [], []
    def reject(**kwargs):
        audited.append(kwargs['problem'].optimization_box.witness_index)
        diagnostics, scope, direction, objective = terminal(**kwargs)
        return replace(diagnostics, backward_error_stationarity_residual=1.), scope, direction, objective
    def audit_only(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(solver, '_terminal_backward_error_audit_float64', reject)
            return audit(*args, **kwargs)
    def record(branch, *args, **kwargs):
        called.append(branch.optimization_box.witness_index)
        return fit_box(branch, *args, **kwargs)
    monkeypatch.setattr(solver, '_audit_reusable_witness', audit_only)
    monkeypatch.setattr(solver, '_fit_prepared_box', record)
    result = solver.fit_prepared(original, 2., options())
    assert audited == [1, 2, 3]  # A declined precondition cannot pass vacuously.
    assert called == [0, 1, 2, 3]
    assert result.certificate.witness_branches_reused == ()
    assert result.certificate.admissible and result.certificate.witness_search_complete


def test_zero_gradient_boundary_records_exact_reuse_conditions(tmp_path, execution_device, record_property):
    original = context(tumor(tmp_path/'boundary.tsv', ((24.,),)*4), device=execution_device)
    first = solver._fit_prepared_box(solver._prepare_witness_problem(original, 0), 2., options())
    target = solver._prepare_witness_problem(original, 1)
    diagnostics, _, direction, objective = solver._terminal_backward_error_audit_float64(
        problem=target, phi=torch.tensor(first.phi, device=execution_device),
        certificate=first.certificate.witness, lambda_value=2., tol=8e-4)
    support = (solver.has_proven_convex_observed_loss(original.source_model, eps=original.eps)
               or solver.has_global_supporting_tangent(original.source_model, first.phi, eps=original.eps))
    covered, work, failure = solver._audit_reusable_witness(target, first, lambda_value=2., tolerance=8e-4)
    conditions = dict(incumbent_admissible=first.certificate.admissible,
        exact_target_clonal=bool(np.all(first.phi[1] == 1.)), global_support=bool(support),
        directional=bool(direction), box_zero=diagnostics.box_residual == 0.,
        kkt_pass=bool(np.isfinite(diagnostics.backward_error_kkt_residual)
                      and diagnostics.backward_error_kkt_residual <= .004),
        objective_unchanged=objective == first.objective.total)
    evidence = dict(device=execution_device, phi_hex=[[float(x).hex() for x in row] for row in first.phi],
        objective_hex=first.objective.total.hex(), audit_objective_hex=objective.hex(),
        audit_kkt=diagnostics.backward_error_kkt_residual, audit_box=diagnostics.box_residual,
        conditions=conditions, reused=covered, reuse_audit_passes=work.full_certificate_audit_passes,
        failure=failure, rejected_conditions=[name for name, passed in conditions.items() if not passed])
    record_property('clonal_boundary_diagnostics', json.dumps(evidence, sort_keys=True))
    assert failure is None and covered == all(conditions.values())
    assert first.certificate.admissible and first.certificate.conditional_kkt_certified
    assert first.certificate.components.residual <= .004
    np.testing.assert_array_equal(first.phi[0], [1.])
    assert np.all(first.phi >= original.source_model.lower) and np.all(first.phi <= original.source_model.upper)
    # Independent analytic optimum: all four unconstrained MLEs equal one.
    optimum = -4*(24*np.log(.4)+36*np.log(.6))
    assert first.objective.total == pytest.approx(optimum, abs=2e-5)


def test_optional_audit_failure_preserves_partial_work_after_recovery(tmp_path, execution_device, monkeypatch):
    original = reusable(tmp_path, execution_device)
    monkeypatch.setattr(solver, '_audit_reusable_witness',
        lambda *a, **k: (False, WorkCounters(), 'ValueError: controlled audit failure'))
    result = solver.fit_prepared(original, 2., options())
    certificate = result.certificate
    assert certificate.witness_search_complete and certificate.witness_branches_unresolved == ()
    assert not certificate.witness_search_work_complete
    assert certificate.witness_reuse_audit_failures == tuple(
        (index, 'ValueError: controlled audit failure') for index in (1, 2, 3))


def test_conditional_certificate_is_not_search_or_global_proof(tmp_path, execution_device, monkeypatch):
    original = context(tumor(tmp_path/'incomplete.tsv', ((6.,), (18.,))), device=execution_device)
    fit_box = solver._fit_prepared_box
    def fail_one(branch, *args, **kwargs):
        if branch.optimization_box.witness_index == 0:
            raise FloatingPointError('controlled failed witness')
        return fit_box(branch, *args, **kwargs)
    monkeypatch.setattr(solver, '_fit_prepared_box', fail_one)
    result = solver.fit_prepared(original, .1, options())
    cert = result.certificate
    assert cert.admissible and cert.conditional_kkt_certified
    assert cert.witness_branches_unresolved == (0,)
    assert not cert.witness_search_complete and not cert.witness_search_work_complete
    assert not cert.global_optimum
    for changed in (replace(cert, witness_search_complete=True), replace(cert, global_optimum=True)):
        with pytest.raises(ValueError):
            changed.validate_clonal_search((0, 1), witness_index=1, global_basis='not_certified')


def test_effective_boxes_preserve_objective_graph_and_exact_primal_gate(tmp_path, execution_device):
    data = tumor(tmp_path/'identity.tsv')
    original = context(data, device=execution_device)
    a, b = (solver._prepare_witness_problem(original, index) for index in (0, 1))
    assert a.model is b.model is original.model and a.graph is b.graph is original.graph
    assert a.objective_spec_hash != b.objective_spec_hash
    phi = np.ones((2, 2))
    for branch in (a, b):
        assert observed_terms_numpy(branch.source_model, phi, eps=1e-6).loss.sum() == observed_terms_numpy(original.source_model, phi, eps=1e-6).loss.sum()
        audit = solver._float64_audit_context(branch)
        assert audit.lower[branch.optimization_box.witness_index].eq(1.).all()
        value = torch.ones((2, 2), dtype=torch.float64, device=execution_device)
        value[branch.optimization_box.witness_index, 0] = np.nextafter(1., 0.)
        zero = KKTDiagnostics(0., 0., 0., 0., 0., 0., 0.)
        assert solver._with_explicit_primal_check(zero, branch, value).box_residual > 0
