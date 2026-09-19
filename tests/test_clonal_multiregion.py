"""Independent two-dimensional group-fusion oracle; no production loss calls."""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.optimize import minimize

from CliPP2.core.fusion import solver

from ._fixtures import context, options, tumor


COUNTS = np.array([[5., 21.], [18., 7.]])
DEPTH, SCALE = 60., .4  # Diploid, purity .8: p = purity*phi/2.


def binomial_loss(phi):
    probability = SCALE*np.asarray(phi)
    return float(-np.sum(COUNTS*np.log(probability)
                         + (DEPTH-COUNTS)*np.log1p(-probability)))


def objective(phi, lambda_value, *, order=2):
    return binomial_loss(phi) + lambda_value*np.linalg.norm(phi[0]-phi[1], ord=order)


def witness_oracle(witness, lambda_value, *, order=2):
    """Solve the free 2D vector independently for each joint fixed-one witness.

    Raising a free coordinate below its binomial MLE decreases both likelihood
    loss and distance to the fixed-one row. Therefore no optimum lies below the
    MLE; on [MLE,1]^2 clipping is inactive and the likelihood is strictly convex.
    Adding either convex norm leaves a unique global solution. Two starts and a
    directly evaluated gradient check guard against a false optimizer success.
    """
    counts = COUNTS[1-witness]
    mle = counts/(DEPTH*SCALE)

    def phi_at(free):
        phi = np.ones((2, 2))
        phi[1-witness] = free
        return phi

    def gradient(free):
        value = -counts/free + SCALE*(DEPTH-counts)/(1-SCALE*free)
        delta = free-1.
        if order == 1:
            return value-lambda_value
        distance = np.linalg.norm(delta)
        # At the coincident corner, zero is a valid Euclidean-norm subgradient.
        return value + (lambda_value*delta/distance if distance else 0.)

    fits = [minimize(lambda free: objective(phi_at(free), lambda_value, order=order),
                    start, method='SLSQP', jac=gradient,
                    bounds=list(zip(mle, np.ones(2))),
                    options={'ftol': 1e-13, 'maxiter': 500})
            for start in (mle, (mle+1.)/2)]
    for fit in fits:
        assert fit.success, fit.message
        # Independently polish the strictly interior optimum using its analytic
        # Hessian: SLSQP may stop on objective rounding before gradient accuracy.
        for _ in range(3):
            hessian = np.diag(counts/fit.x**2
                + SCALE**2*(DEPTH-counts)/(1-SCALE*fit.x)**2)
            if order == 2:
                delta = fit.x-1.
                distance = np.linalg.norm(delta)
                hessian += lambda_value*(np.eye(2)/distance
                                         - np.outer(delta, delta)/distance**3)
            fit.x -= np.linalg.solve(hessian, gradient(fit.x))
            assert np.all(fit.x > mle) and np.all(fit.x < 1.)
        fit.fun = objective(phi_at(fit.x), lambda_value, order=order)
        assert np.all(fit.x > mle) and np.all(fit.x < 1.)
        assert np.max(np.abs(gradient(fit.x))) < 1e-10
    np.testing.assert_allclose(fits[0].x, fits[1].x, atol=1e-7, rtol=0)
    best = min(fits, key=lambda fit: fit.fun)
    return float(best.fun), witness, phi_at(best.x)


@pytest.mark.parametrize('lambda_value, solve_tolerance', [(2., 1e-7), (4., 1e-7), (4., 8e-4)])
def test_group_l2_search_matches_independent_joint_witness_oracle(
        tmp_path, execution_device, lambda_value, solve_tolerance, record_property):
    branches = [witness_oracle(witness, lambda_value) for witness in (0, 1)]
    expected_loss, expected_witness, expected_phi = min(branches, key=lambda item: item[0])
    assert expected_witness == 1  # No advance assignment to the first mutation.
    assert branches[0][0]-expected_loss > 2.

    # This crossed-count fixture discriminates L2 from coordinatewise L1 fusion.
    l1_loss, _, l1_phi = min((witness_oracle(witness, lambda_value, order=1)
                             for witness in (0, 1)), key=lambda item: item[0])
    assert l1_loss-expected_loss > .1
    assert np.max(np.abs(l1_phi-expected_phi)) > .01
    # The weaker, regionwise-only constraint admits this cheaper *nonjoint*
    # construction. Merely having max_i phi_ir == 1 in each region is not enough.
    regionwise = np.array([[COUNTS[0, 0]/(DEPTH*SCALE), 1.],
                          [1., COUNTS[1, 1]/(DEPTH*SCALE)]])
    assert np.all(np.max(regionwise, axis=0) == 1.)
    assert not np.any(np.all(regionwise == 1., axis=1))
    assert objective(regionwise, lambda_value) < expected_loss-8.

    original = context(tumor(tmp_path/'group.tsv', COUNTS), device=execution_device)
    lower, upper = original.source_model.lower.copy(), original.source_model.upper.copy()
    np.testing.assert_array_equal(upper, np.ones((2, 2)))
    assert np.all(lower <= COUNTS/(DEPTH*SCALE))
    np.testing.assert_array_equal(original.graph_spec.edge_u, [0])
    np.testing.assert_array_equal(original.graph_spec.edge_v, [1])
    np.testing.assert_array_equal(original.graph_spec.edge_w, [1.])
    # Two high-accuracy test-only solves plus a normal-tolerance lambda=4 solve.
    # Every arm retains the original certification tolerance and 0.004 gate;
    # tighter optimization is not a new production configuration or relaxed gate.
    oracle_options = options()
    if solve_tolerance != oracle_options.tolerance:
        oracle_options = replace(oracle_options, tolerance=solve_tolerance,
                                 certification_tolerance=8e-4, outer_max_iter=32)
    result = solver.fit_prepared(original, lambda_value, oracle_options)
    objective_tolerance = 2e-5
    counts = COUNTS[1-expected_witness]
    # For phi<=1: ell''=a/phi^2+.16*(n-a)/(1-.4*phi)^2 >= a+.16*(n-a).
    # Strong convexity converts the independent objective gap budget to a
    # justified Euclidean CCF error bound, rather than backend-tuned digits.
    curvature_lower_bound = float(np.min(counts+SCALE**2*(DEPTH-counts)))
    phi_tolerance = np.sqrt(2*objective_tolerance/curvature_lower_bound)
    record_property('multiregion_group_l2_oracle', json.dumps(dict(
        device=execution_device, lambda_value=lambda_value, solve_tolerance=solve_tolerance,
        outer_max_iter=oracle_options.outer_max_iter,
        oracle_objective=expected_loss, fitted_objective=result.objective.total,
        objective_gap=result.objective.total-expected_loss,
        phi_l2_error=float(np.linalg.norm(result.phi-expected_phi)),
        phi_l2_tolerance=float(phi_tolerance), curvature_lower_bound=curvature_lower_bound,
        witness_index=result.provenance.witness_index,
        kkt_residual=result.certificate.components.residual,
        admissible=result.certificate.admissible), sort_keys=True))
    assert result.provenance.witness_index == expected_witness
    np.testing.assert_array_equal(result.phi[expected_witness], [1., 1.])
    assert np.all(result.phi[1-expected_witness] < 1.)
    assert np.linalg.norm(result.phi-expected_phi) <= phi_tolerance
    assert result.objective.total == pytest.approx(expected_loss, abs=objective_tolerance)
    assert result.objective.total == pytest.approx(objective(result.phi, lambda_value), abs=2e-10)
    assert np.all(result.phi >= lower) and np.all(result.phi <= upper)
    np.testing.assert_array_equal(original.source_model.lower, lower)
    np.testing.assert_array_equal(original.source_model.upper, upper)
    certificate = result.certificate
    assert certificate.admissible and certificate.conditional_kkt_certified
    assert certificate.audit_dtype == 'float64'
    certification_tolerance = (oracle_options.tolerance if oracle_options.certification_tolerance is None
                               else oracle_options.certification_tolerance)
    assert certificate.components.residual <= 5*certification_tolerance == .004
    assert certificate.witness_search_complete
