"""Full-step recovery must decrease the objective AND majorize the loss."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion import solver
from CliPP2.core.objective import observed_terms_torch
from test_integer_likelihood import EPS, integer_data


def _recovery(monkeypatch, *, attempts=None):
    data = replace(
        integer_data(((1,),), purity=1.),
        alt_counts=np.array([[10.]]), phi_init=np.array([[.4]]),
    )
    options = resolve_fit_config(
        device="cpu", dtype="float64", outer_max_iter=1, inner_max_iter=16,
        objective_shape="unimodal_full_step_backtracking",
    )
    context = solver.prepare_torch_problem_with_resource_policy(
        data, options, exact_pilot=data.phi_init, pooled_start=data.phi_init,
        scalar_well_starts=(),
    )
    trials = []
    original = solver._solve_inner_subproblem

    def recorded_solve(**kwargs):
        result = original(**kwargs)
        current = observed_terms_torch(context.model, kwargs["phi"], eps=EPS)
        trial = observed_terms_torch(context.model, result.phi, eps=EPS)
        delta = result.phi - kwargs["phi"]
        quadratic_gap = torch.sum(current.gradient * delta + .5 * kwargs["h"] * delta.square())
        trials.append((
            result.phi.clone(),
            float((trial.loss - current.loss).sum()),
            float((trial.loss - current.loss).sum() - quadratic_gap),
        ))
        return result

    monkeypatch.setattr(solver, "_solve_inner_subproblem", recorded_solve)
    if attempts is not None:
        monkeypatch.setattr(solver, "_FULL_STEP_MAX_CURVATURE_ATTEMPTS", attempts)
    fit = solver._fit_from_start(
        context, 0., options.solver, solver._StartAttempt(data.phi_init, None, "majorization_fixture"),
    )
    return fit, trials, data, options


def test_recovery_rejects_descent_without_majorization_and_retries_full_endpoint(monkeypatch):
    fit, trials, data, options = _recovery(monkeypatch)
    assert trials[0][0].item() == pytest.approx(.08)
    assert trials[0][1] == pytest.approx(-.314561, abs=1e-6)
    assert trials[0][2] == pytest.approx(4.685439, abs=1e-6)
    assert len(trials) >= 2  # The old code accepted the first Armijo-passing endpoint.
    np.testing.assert_array_equal(fit.phi, trials[-1][0].numpy())
    assert trials[-1][1] < 0.
    current_loss = -10. * np.log(.2) - 90. * np.log(.8)
    allowance = max(64 * np.finfo(np.float64).eps, options.solver.tolerance ** 2) * (1. + current_loss)
    assert trials[-1][2] <= allowance
    assert not np.array_equal(fit.phi, data.phi_init)
    assert fit.convergence.accepted_full_steps == 1
    assert fit.convergence.accepted_damped_steps == 0


def test_recovery_exhaustion_keeps_original_primal_without_damping(monkeypatch):
    fit, trials, data, _ = _recovery(monkeypatch, attempts=1)
    assert len(trials) == 1 and trials[0][2] > 4.
    np.testing.assert_array_equal(fit.phi, data.phi_init)
    assert fit.convergence.accepted_full_steps == fit.convergence.accepted_damped_steps == 0
    assert not fit.certificate.certified
