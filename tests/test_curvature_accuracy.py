"""Accuracy evidence, separate from historical proposal/graph parity goldens.

Source-float64 finite differences are evaluated as a possible numerical revision,
not substituted into the production graph recipe by these tests.
"""

import numpy as np
import pytest
import torch
from scipy.special import logsumexp

from CliPP2.core.fusion.partition_starts import observed_curvature_at_pilot_torch
from CliPP2.core.fusion.torch_backend import resolve_runtime
from CliPP2.core.objective import model_to_torch
from test_curvature_preparation import _context, _prepared_case
from test_integer_likelihood import integer_data


def _analytic_curvature(model, phi, eps):
    """Independent mixture Hessian; there is no ordinary Hessian at a kink."""
    raw = np.asarray(phi)[..., None] * model.slope
    if np.any(model.valid & ((raw == eps) | (raw == 1. - eps))):
        raise ValueError("Curvature is not defined at a clipping transition.")
    probability = np.clip(raw, eps, 1. - eps)
    slope = np.where((raw > eps) & (raw < 1. - eps), model.slope, 0.)
    alt, ref = model.alt[..., None], model.nonalt[..., None]
    joint = alt * np.log(probability) + ref * np.log1p(-probability) + model.log_prior
    joint = np.where(model.valid, joint, -np.inf)
    posterior = np.exp(joint - logsumexp(joint, axis=-1, keepdims=True))
    score = slope * (alt / probability - ref / (1. - probability))
    component_curvature = slope ** 2 * (alt / probability ** 2 + ref / (1. - probability) ** 2)
    mean_score = np.sum(posterior * score, axis=-1, keepdims=True)
    curvature = np.sum(posterior * (component_curvature - (score - mean_score) ** 2), axis=-1)
    return np.where(model.observed, curvature, 0.)


def _source_float64_case():
    context = _prepared_case("float32")
    model = model_to_torch(
        context.source_model, resolve_runtime("cpu", dtype="float64"), eps=context.eps,
    )
    # The SAME rounded pilot isolates curvature arithmetic from pilot changes.
    phi = context.exact_pilot.double()
    return context, model, phi


def test_independent_mixture_curvature_matches_float64_autodiff():
    context, model, phi = _source_float64_case()
    phi = phi.requires_grad_()
    probability = torch.clamp(phi[..., None] * model.slope, context.eps, 1. - context.eps)
    joint = (model.alt[..., None] * probability.log()
             + model.nonalt[..., None] * torch.log1p(-probability) + model.log_prior)
    loss = torch.where(model.observed, -torch.logsumexp(joint.masked_fill(~model.valid, -torch.inf), dim=-1), 0.)
    gradient = torch.autograd.grad(loss.sum(), phi, create_graph=True)[0]
    hessian = torch.autograd.grad(gradient.sum(), phi)[0]
    analytic = _analytic_curvature(context.source_model, phi.detach().numpy(), context.eps)
    np.testing.assert_allclose(hessian.detach().numpy(), analytic, rtol=2e-12, atol=2e-12)
    assert analytic[1, 1] == pytest.approx(53.12934791609, abs=1e-9)


def test_source_float64_stencil_is_accurate_at_same_pilot_without_cap_masking():
    context, model, phi = _source_float64_case()
    expected = np.maximum(_analytic_curvature(context.source_model, phi.numpy(), context.eps), 1e-6)
    actual = observed_curvature_at_pilot_torch(model, phi, eps=context.eps, curvature_cap_quantile=1.)
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-4, atol=1e-10)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "Known float32 loss-stencil cancellation (~15.9% on the reviewed coordinate); "
    "changing production curvature requires separate proposal/graph qualification."
))
def test_production_float32_curvature_meets_independent_accuracy_target():
    context, _, phi = _source_float64_case()
    expected = np.maximum(_analytic_curvature(context.source_model, phi.numpy(), context.eps), 1e-6)
    actual = observed_curvature_at_pilot_torch(
        context.model, context.exact_pilot, eps=context.eps, curvature_cap_quantile=1.,
    )
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-4, atol=1e-10)


def test_clipped_plateau_and_box_boundary_keep_floor_not_a_smooth_hessian():
    context = _context(integer_data(((1,),), purity=1.), dtype="float64", eps=.1)
    for value in (.1, .11):
        phi = torch.tensor([[value]], dtype=torch.float64)
        assert _analytic_curvature(context.source_model, phi.numpy(), context.eps).item() == 0.
        actual = observed_curvature_at_pilot_torch(context.model, phi, eps=context.eps)
        assert actual.item() == 1e-6
    with pytest.raises(ValueError, match="clipping transition"):
        _analytic_curvature(context.source_model, np.array([[.2]]), context.eps)


def test_stencil_crossing_clipping_transition_is_not_an_analytic_hessian():
    context = _context(integer_data(((1,),), purity=1.), dtype="float64", eps=.1)
    phi = torch.tensor([[.2001]], dtype=torch.float64)
    # The default 0.001 stencil straddles the 0.2 probability breakpoint.
    analytic = _analytic_curvature(context.source_model, phi.numpy(), context.eps).item()
    stencil = observed_curvature_at_pilot_torch(context.model, phi, eps=context.eps).item()
    assert analytic > 600.
    assert stencil == 1e-6
