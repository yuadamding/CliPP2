"""A saturated admission residual must not discard useful fixed-primal work."""

from dataclasses import replace
import os

import numpy as np
import pytest
import torch

from CliPP2.core.fusion import torch_backend as backend
from CliPP2.core.objective import compile_observed_model, model_to_torch, observed_terms_torch
from test_integer_likelihood import EPS, integer_data


def _case(dtype, device="cpu"):
    data = replace(
        integer_data(((1,),) * 4, purity=1.),
        alt_counts=np.array([[24.], [25.], [24.], [25.]]),
        total_counts=np.array([[99.], [97.], [99.], [97.]]),
    )
    model = model_to_torch(compile_observed_model(data, eps=EPS),
                           backend.resolve_runtime(device, dtype=dtype), eps=EPS)
    phi = torch.full_like(model.alt, .5)
    grad = observed_terms_torch(model, phi, eps=EPS).gradient
    torch.testing.assert_close(grad, phi.new_tensor([[2.], [-2.], [2.], [-2.]]), rtol=0, atol=0)
    u, v = torch.triu_indices(4, 4, 1, device=phi.device)
    dual = phi.new_zeros((6, 1))
    dual[0] = 4.
    return dict(phi=phi, grad_smooth=grad, dual_kkt=dual, lower=model.lower, upper=model.upper,
                edge_u=u, edge_v=v, edge_w=phi.new_full((6,), 1 / 3), lambda_value=12., atol=8e-4)


def _violation(kwargs, dual):
    # Independent incidence multiplication and exact-bound cone projection.
    n = kwargs["phi"].shape[0]
    incidence = np.zeros((len(dual), n))
    incidence[np.arange(len(dual)), kwargs["edge_u"].cpu().numpy()] = 1.
    incidence[np.arange(len(dual)), kwargs["edge_v"].cpu().numpy()] = -1.
    total = kwargs["grad_smooth"].cpu().numpy() + incidence.T @ dual.cpu().numpy()
    phi, lo, hi = (kwargs[name].cpu().numpy() for name in ("phi", "lower", "upper"))
    cone = np.where(phi == lo, np.maximum(total, 0.), 0.)
    cone = np.where(phi == hi, np.minimum(total, 0.), cone)
    cone = np.where(lo == hi, total, cone)
    return float(np.linalg.norm(total - cone))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("chunk_edges", [1, 2, 6])
def test_saturated_binomial_case_reaches_gate_without_primal_change(monkeypatch, dtype, chunk_edges):
    kwargs = _case(dtype)
    before = {key: value.clone() for key, value in kwargs.items() if torch.is_tensor(value)}
    budget = chunk_edges * kwargs["phi"].element_size()
    start = backend.graph_fusion_kkt_residual_from_grad_torch(**kwargs)
    assert start.backward_error_kkt_residual == 1.
    assert _violation(kwargs, kwargs["dual_kkt"]) == pytest.approx(np.sqrt(80))
    exact = torch.zeros_like(kwargs["dual_kkt"])
    exact[0] = exact[-1] = -2.
    exact_audit = backend.graph_fusion_kkt_residual_from_grad_torch(**dict(kwargs, dual_kkt=exact))
    assert exact_audit.backward_error_kkt_residual == 0.

    # Diagnostic control only: same update equations, disabling just early abort.
    plateau = backend._update_certificate_refinement_plateau
    def never_abort(**values):
        result = plateau(**values)
        return (*result[:-1], False)
    with monkeypatch.context() as patch:
        patch.setattr(backend, "_update_certificate_refinement_plateau", never_abort)
        control = backend.refine_graph_fusion_dual_certificate_torch(
            **kwargs, max_iter=96, edge_work_bytes=budget,
        )
    assert control["diag"].backward_error_kkt_residual <= .004
    assert control["refinement_iterations"] == 42

    result = backend.refine_graph_fusion_dual_certificate_torch(
        **kwargs, max_iter=96, edge_work_bytes=budget,
    )
    assert result["diag"].backward_error_kkt_residual <= .004
    assert result["refinement_iterations"] == 42
    torch.testing.assert_close(result["dual"], control["dual"], rtol=0, atol=0)
    assert _violation(kwargs, result["dual"]) == pytest.approx(.018394, abs=1e-6)
    for key, value in before.items():
        torch.testing.assert_close(kwargs[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("iterations", [8, 16, 24])
@pytest.mark.parametrize("chunk_edges", [1, 6])
def test_budget_exhaustion_keeps_useful_backward_error_tied_witness(iterations, chunk_edges):
    kwargs = _case("float64")
    result = backend.refine_graph_fusion_dual_certificate_torch(
        **kwargs, max_iter=iterations, edge_work_bytes=chunk_edges * 8,
    )
    assert result["refinement_iterations"] == iterations
    assert result["diag"].backward_error_kkt_residual == 1.
    assert _violation(kwargs, result["dual"]) < _violation(kwargs, kwargs["dual_kkt"])
    assert result["status"] == "refined_fused_edge_dual"
    audit = backend.graph_fusion_kkt_residual_from_grad_torch(**dict(kwargs, dual_kkt=result["dual"]))
    assert audit == result["diag"]


@pytest.mark.parametrize("residuals,merits,best_index", [
    ([1., 1., 1., 1., 1.], [8., 7., 6., 5., 6.], 3),
    ([.8, .9, .85, .8, .8], [10., 1., 0., 9., 8.], 4),
    ([.8, .9, .7, .8, .75], [10., 1., 100., 0., 0.], 2),
    ([1., 1., float("nan"), float("inf"), 1.], [8., 7., 0., 0., 6.], 4),
    ([1., 1., 1., 1., 1.], [8., float("inf"), 7., float("inf"), 9.], 2),
])
def test_witness_order_is_primary_backward_error_then_exact_tie_merit(
    monkeypatch, residuals, merits, best_index,
):
    kwargs = _case("float64")
    audits, duals = [], []
    def audit(**values):
        index = len(audits)
        # Legacy order deliberately prefers the last witness, independently.
        diag = backend.KKTDiagnostics(1 / (index + 1), 0., 0., 0., residuals[index], 0., 0.)
        values["_progress_out"]["cone_violation_norm"] = merits[index]
        audits.append(diag)
        duals.append(values["dual_kkt"].clone())
        return diag
    monkeypatch.setattr(backend, "graph_fusion_kkt_residual_from_grad_torch", audit)
    result = backend.refine_graph_fusion_dual_certificate_torch(**kwargs, max_iter=3)
    assert result["refinement_iterations"] == 3
    assert result["diag"] == audits[best_index]
    torch.testing.assert_close(result["dual"], duals[best_index], rtol=0, atol=0)
    assert result["diag"].backward_error_kkt_residual > .004


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_secondary_progress_prevents_abort_but_mapping_motion_alone_does_not(dtype):
    anchor, merit, stalled = 1., 10., 0
    for iteration in range(32):
        anchor, merit, stalled, stop = backend._update_certificate_refinement_plateau(
            anchor_residual=anchor, best_residual=1., anchor_merit=merit,
            best_merit=10. - .1 * (iteration + 1), mapping_delta=0.,
            stalled_iterations=stalled, atol=8e-4, dtype=dtype,
        )
        assert not stop and stalled == 0 and anchor == 1.
    for iteration in range(16):
        anchor, merit, stalled, stop = backend._update_certificate_refinement_plateau(
            anchor_residual=anchor, best_residual=1., anchor_merit=merit,
            best_merit=merit, mapping_delta=1., stalled_iterations=stalled,
            atol=8e-4, dtype=dtype,
        )
        assert stop == (iteration == 15)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_secondary_merit_uses_exact_box_cone_and_nonfinite_fails_closed(dtype):
    phi = torch.tensor([[0.], [0.], [1.], [1.], [.5], [.5], [0.], [1.]], dtype=dtype)
    # Even the closest representable interior points have the singleton cone.
    phi[-2] = torch.nextafter(phi[-2], torch.ones_like(phi[-2]))
    phi[-1] = torch.nextafter(phi[-1], torch.zeros_like(phi[-1]))
    assert 0. < phi[-2].item() < 8e-4 and 0. < 1. - phi[-1].item() < 8e-4
    lower, upper = torch.zeros_like(phi), torch.ones_like(phi)
    lower[5] = upper[5] = .5  # fixed coordinate permits the full normal cone
    kwargs = dict(phi=phi, lower=lower, upper=upper,
                  grad_smooth=phi.new_tensor([[2.], [-3.], [-4.], [5.], [6.], [9.], [7.], [-8.]]),
                  dual_kkt=phi.new_zeros((0, 1)), edge_u=torch.empty(0, dtype=torch.long),
                  edge_v=torch.empty(0, dtype=torch.long), edge_w=phi.new_zeros(0),
                  lambda_value=0., atol=8e-4)
    progress = {}
    ordinary = backend.graph_fusion_kkt_residual_from_grad_torch(**kwargs)
    diagnostic = backend.graph_fusion_kkt_residual_from_grad_torch(**kwargs, _progress_out=progress)
    assert ordinary == diagnostic
    assert progress["cone_violation_norm"] == pytest.approx(np.sqrt(183))
    assert progress["cone_violation_norm"] == pytest.approx(_violation(kwargs, kwargs["dual_kkt"]))
    kwargs["grad_smooth"][4] = float("nan")
    failed = backend.graph_fusion_kkt_residual_from_grad_torch(**kwargs, _progress_out=progress)
    assert not np.isfinite(failed.backward_error_kkt_residual)
    assert progress["cone_violation_norm"] == float("inf")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mapping_delta,patience", [
    (0., 8), (1., 16), (float("nan"), 16), (float("inf"), 16),
])
@pytest.mark.parametrize("best_residual,best_merit", [
    (1., float("nan")), (1., float("inf")),
    (float("nan"), 0.), (float("inf"), 0.),
    (1.1, 0.),  # Secondary improvement cannot offset a worse admission merit.
    (1., 10. - 1e-8),  # A non-material numerical change cannot reset patience.
])
def test_plateau_invalid_or_nonmaterial_merits_do_not_extend_patience(
    dtype, mapping_delta, patience, best_residual, best_merit,
):
    stalled = 0
    for iteration in range(patience):
        anchor, merit, stalled, stop = backend._update_certificate_refinement_plateau(
            anchor_residual=1., best_residual=best_residual, anchor_merit=10.,
            best_merit=best_merit, mapping_delta=mapping_delta,
            stalled_iterations=stalled, atol=8e-4, dtype=dtype,
        )
        assert (anchor, merit, stalled) == (1., 10., iteration + 1)
        assert stop == (iteration == patience - 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("anchor_residual,anchor_merit", [
    (float("inf"), 10.), (1., float("inf")), (1., float("nan")),
])
def test_plateau_finite_progress_replaces_nonfinite_anchor(dtype, anchor_residual, anchor_merit):
    result = backend._update_certificate_refinement_plateau(
        anchor_residual=anchor_residual, best_residual=1., anchor_merit=anchor_merit,
        best_merit=5., mapping_delta=0., stalled_iterations=7,
        atol=8e-4, dtype=dtype,
    )
    assert result == (1., 5., 0, False)


@pytest.mark.skipif(os.environ.get("CLIPP2_TEST_CUDA") != "1", reason="Explicit LSF CUDA opt-in required")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cuda_saturated_refinement_stays_within_fixed_budget(dtype):
    assert torch.cuda.is_available()
    kwargs = _case(dtype, "cuda")
    for chunk_edges in (1, 6):
        result = backend.refine_graph_fusion_dual_certificate_torch(
            **kwargs, max_iter=96, edge_work_bytes=chunk_edges * kwargs["phi"].element_size(),
        )
        assert result["diag"].backward_error_kkt_residual <= .004
        assert 24 < result["refinement_iterations"] <= 96
