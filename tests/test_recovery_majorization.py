"""Full-step recovery must decrease the objective AND majorize the loss."""

from dataclasses import replace
import os

import numpy as np
import pytest
import torch

from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion import solver
from CliPP2.core.fusion.graph import build_complete_uniform_graph
from CliPP2.core.objective import observed_terms_torch
from CliPP2.io.data import tumor_data_fingerprint
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


def _positive_lambda_recovery(monkeypatch, *, dtype, device="cpu", attempts=None):
    """Run the real complete-graph ADMM recovery, recording but not replacing it."""
    data = replace(
        integer_data(((1,),) * 4, purity=1.),
        alt_counts=np.array([[10.], [12.], [14.], [16.]]),
        phi_init=np.full((4, 1), .4),
    )
    graph = build_complete_uniform_graph(4)
    options = resolve_fit_config(
        lambda_value=2., device=device, dtype=dtype, graph=graph,
        outer_max_iter=1, inner_max_iter=128, certificate_max_iter=96,
        objective_shape="unimodal_full_step_backtracking",
    )
    context = solver.prepare_torch_problem_with_resource_policy(
        data, options, exact_pilot=data.phi_init, pooled_start=data.phi_init,
        scalar_well_starts=(),
    )
    assert context.runtime.device.type == device
    source_hash, graph_hash = tumor_data_fingerprint(data), graph.fingerprint
    objective_hash = context.objective_spec_hash
    initial = torch.tensor(data.phi_init.copy(), device=device, dtype=context.runtime.dtype)
    current = observed_terms_torch(context.model, initial, eps=EPS)
    trials, audits, events = [], [], []
    solve_inner, certify = solver._solve_inner_subproblem, solver.certify

    def recorded_solve(**kwargs):
        result = solve_inner(**kwargs)
        trial = observed_terms_torch(context.model, result.phi, eps=EPS)
        delta = result.phi - initial  # One outer iteration: anchor never changes.
        loss_change = float((trial.loss.sum() - current.loss.sum()).item())
        majorization_gap = loss_change - float(torch.sum(
            current.gradient * delta + .5 * kwargs["h"] * delta.square(),
        ).item())
        trials.append(dict(
            phi=result.phi.detach().cpu().numpy().copy(),
            loss_change=loss_change, majorization_gap=majorization_gap,
            surrogate_gradient=(kwargs["h"] * (result.phi - kwargs["U"])).detach().cpu().numpy(),
            surrogate_residual=result.surrogate_kkt.backward_error_kkt_residual,
            surrogate_scope=result.surrogate_certificate.gradient_scope,
        ))
        assert result.backend_name == "admm_complete_graph"
        assert kwargs["lambda_value"] == 2. and kwargs["spectral_rho"]
        assert kwargs["graph_hash"] == graph_hash and kwargs["edge_u"].numel() == 6
        events.append("inner")
        return result

    def recorded_audit(**kwargs):
        result = certify(**kwargs)
        audits.append(dict(
            phi=kwargs["phi"].detach().cpu().numpy().copy(),
            gradient=kwargs["gradient"].value.detach().cpu().numpy().copy(),
            gradient_scope=kwargs["gradient"].scope,
            residual=result.diagnostics.backward_error_kkt_residual,
        ))
        assert kwargs["problem"].graph_hash == graph_hash
        assert kwargs["problem"].lambda_value == 2.
        events.append("raw_audit")
        return result

    monkeypatch.setattr(solver, "_solve_inner_subproblem", recorded_solve)
    monkeypatch.setattr(solver, "certify", recorded_audit)
    if attempts is not None:
        monkeypatch.setattr(solver, "_FULL_STEP_MAX_CURVATURE_ATTEMPTS", attempts)
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        fit = solver._fit_from_start(
            context, 2., options.solver,
            solver._StartAttempt(data.phi_init, None, "positive_lambda_majorization_fixture"),
        )
    finally:
        torch.set_num_threads(original_threads)
    context.validate()
    assert tumor_data_fingerprint(data) == fit.provenance.source_data_hash == source_hash
    assert graph.fingerprint == fit.provenance.original_graph_hash == graph_hash
    assert context.objective_spec_hash == fit.provenance.objective_spec_hash == objective_hash
    assert fit.provenance.lambda_value == 2.
    assert fit.convergence.stage_inner_solve_calls == len(trials)
    assert events[-1] == "raw_audit" and events.index("inner") < len(events) - 1
    return data, options, context, fit, initial, current, trials, audits


def _binomial_objective(data, graph, phi, lambda_value):
    phi = np.asarray(phi, dtype=np.float64)
    probability = np.clip(data.scaling * phi, EPS, 1. - EPS)
    loss = float(np.sum(
        -data.alt_counts * np.log(probability)
        - (data.total_counts - data.alt_counts) * np.log1p(-probability),
    ))
    penalty = float(lambda_value * np.sum(
        graph.edge_w * np.linalg.norm(phi[graph.edge_u] - phi[graph.edge_v], axis=1),
    ))
    return loss, penalty


def _assert_positive_lambda_audit(data, options, context, fit, audits):
    # The final call must evaluate the nonlinear observed gradient at the actual
    # returned point, not reuse the inner quadratic surrogate's residual.
    terminal = audits[-1]
    np.testing.assert_array_equal(terminal["phi"], fit.phi)
    probability = data.scaling * np.asarray(fit.phi, dtype=np.float64)
    gradient = data.scaling * (
        (data.total_counts - data.alt_counts) / (1. - probability)
        - data.alt_counts / probability
    )
    np.testing.assert_allclose(terminal["gradient"], gradient, rtol=1e-12, atol=1e-12)
    assert terminal["gradient_scope"] == fit.certificate.gradient_scope == "observed_objective"
    assert fit.certificate.scope == "full_original_graph"
    assert fit.certificate.audit_dtype == "float64"
    assert fit.certificate.residual_method == "componentwise_box_cone_backward_error_v1"
    assert fit.certificate.tolerance == .004
    assert terminal["residual"] == fit.certificate.components.residual
    assert fit.certificate.components.residual > .004
    assert not fit.certificate.certified and not fit.certificate.admissible
    # Re-audit independently using the frozen source/context and final raw dual.
    diag, scope, directional, objective = solver._terminal_backward_error_audit_float64(
        problem=context, phi=fit.state.phi, certificate=fit.certificate.witness,
        lambda_value=2., tol=options.solver.tolerance,
    )
    assert scope == "observed_objective" and directional
    assert diag.backward_error_kkt_residual == fit.certificate.components.residual
    assert objective == pytest.approx(fit.objective.total, rel=1e-13)


def _check_positive_lambda_endpoint(monkeypatch, *, dtype, device="cpu"):
    data, options, context, fit, initial, current, trials, audits = _positive_lambda_recovery(
        monkeypatch, dtype=dtype, device=device,
    )
    initial_loss, initial_penalty = _binomial_objective(data, context.graph_spec, initial.cpu().numpy(), 2.)
    allowance = max(
        64 * torch.finfo(initial.dtype).eps, options.solver.tolerance**2,
    ) * (1. + abs(float(current.loss.sum().item())))
    first_loss, first_penalty = _binomial_objective(data, context.graph_spec, trials[0]["phi"], 2.)
    assert first_loss + first_penalty < initial_loss + initial_penalty
    assert trials[0]["majorization_gap"] > allowance
    assert len(trials) >= 2
    np.testing.assert_array_equal(fit.phi, trials[-1]["phi"])
    assert trials[-1]["majorization_gap"] <= allowance
    final_loss, final_penalty = _binomial_objective(data, context.graph_spec, fit.phi, 2.)
    assert final_penalty > 0.  # The positive-lambda graph term is genuinely active.
    assert final_loss + final_penalty < initial_loss + initial_penalty
    assert fit.objective.total == pytest.approx(final_loss + final_penalty, rel=1e-13)
    assert fit.convergence.accepted_full_steps == 1
    assert fit.convergence.accepted_damped_steps == 0
    assert fit.convergence.mm_consistency_violations == 0
    assert trials[-1]["surrogate_scope"] == "mm_surrogate"
    assert trials[-1]["surrogate_residual"] <= .004
    assert not np.allclose(audits[-1]["gradient"], trials[-1]["surrogate_gradient"])
    _assert_positive_lambda_audit(data, options, context, fit, audits)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_positive_lambda_full_endpoint_majorization_and_independent_raw_audit(monkeypatch, dtype):
    _check_positive_lambda_endpoint(monkeypatch, dtype=dtype)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_positive_lambda_rejected_endpoint_exhaustion_does_not_damp_or_inherit_audit(monkeypatch, dtype):
    data, options, context, fit, initial, _, trials, audits = _positive_lambda_recovery(
        monkeypatch, dtype=dtype, attempts=1,
    )
    assert len(trials) == 1 and trials[0]["majorization_gap"] > 4.
    assert trials[0]["surrogate_residual"] <= .004
    np.testing.assert_array_equal(fit.phi, initial.cpu().numpy())
    assert fit.convergence.accepted_full_steps == fit.convergence.accepted_damped_steps == 0
    assert fit.convergence.rejected_outer_steps == 1
    assert fit.objective.total == pytest.approx(sum(_binomial_objective(
        data, context.graph_spec, fit.phi, 2.,
    )), rel=1e-13)
    _assert_positive_lambda_audit(data, options, context, fit, audits)


@pytest.mark.skipif(os.environ.get("CLIPP2_TEST_CUDA") != "1", reason="LSF CUDA opt-in required")
def test_positive_lambda_cuda_full_endpoint_requires_raw_audit(monkeypatch):
    assert torch.cuda.is_available(), "CLIPP2_TEST_CUDA=1 requires qualified CUDA"
    _check_positive_lambda_endpoint(monkeypatch, dtype="float32", device="cuda")
