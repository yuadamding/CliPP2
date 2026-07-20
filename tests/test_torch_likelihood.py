from __future__ import annotations

import numpy as np
import pytest
import torch

from CliPP2.core.fusion.torch_backend import (
    TorchTumorData,
    mutation_region_loss_grid_torch,
    mutation_region_terms_torch,
    objective_terms_torch,
    objective_value_torch,
)
from CliPP2.core.fusion.starts import _mutation_region_loss_grid_numpy


def _toy_torch_data(dtype: torch.dtype = torch.float64) -> TorchTumorData:
    alt = torch.tensor([[4.0, 2.0], [7.0, 1.0]], dtype=dtype)
    total = torch.tensor([[10.0, 10.0], [12.0, 8.0]], dtype=dtype)
    return TorchTumorData(
        alt=alt,
        total=total,
        nonalt=total - alt,
        phi_upper=torch.ones_like(alt),
        ambiguous=torch.tensor([[False, True], [True, False]], dtype=torch.bool),
        b_minus=torch.full_like(alt, 0.35),
        b_plus=torch.full_like(alt, 0.7),
        b_fixed=torch.full_like(alt, 0.5),
    )


def test_mutation_region_terms_match_independent_observed_likelihood_formula() -> None:
    data = _toy_torch_data()
    phi = torch.tensor([[0.4, 0.6], [0.8, 0.3]], dtype=torch.float64)
    prior = 0.65
    eps = 1e-6

    terms = mutation_region_terms_torch(data, phi, major_prior=prior, eps=eps)

    p_fixed = torch.clamp(data.b_fixed * phi, min=eps, max=1.0 - eps)
    fixed_loss = -(data.alt * torch.log(p_fixed) + data.nonalt * torch.log1p(-p_fixed))
    p_minus = torch.clamp(data.b_minus * phi, min=eps, max=1.0 - eps)
    p_plus = torch.clamp(data.b_plus * phi, min=eps, max=1.0 - eps)
    log_minor = (
        data.alt * torch.log(p_minus)
        + data.nonalt * torch.log1p(-p_minus)
        + np.log1p(-prior)
    )
    log_major = (
        data.alt * torch.log(p_plus)
        + data.nonalt * torch.log1p(-p_plus)
        + np.log(prior)
    )
    amb_loss = -torch.logaddexp(log_minor, log_major)
    expected_loss = torch.where(data.ambiguous, amb_loss, fixed_loss)
    expected_gamma = torch.where(
        data.ambiguous,
        torch.sigmoid(log_major - log_minor),
        torch.ones_like(phi),
    )

    assert torch.allclose(terms.loss, expected_loss)
    assert torch.allclose(terms.gamma_major, expected_gamma)
    assert torch.all(torch.isfinite(terms.grad))
    assert torch.all(terms.hess_upper >= 1e-8)


def test_objective_terms_keep_tensor_scalars_and_scalar_wrapper_matches() -> None:
    data = _toy_torch_data()
    phi = torch.tensor([[0.4, 0.6], [0.8, 0.3]], dtype=torch.float64)
    edge_u = torch.tensor([0], dtype=torch.long)
    edge_v = torch.tensor([1], dtype=torch.long)
    edge_w = torch.tensor([2.0], dtype=torch.float64)

    tensor_terms = objective_terms_torch(
        data,
        phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=3.0,
        major_prior=0.5,
        eps=1e-6,
    )
    fit, penalty, total, gamma = objective_value_torch(
        data,
        phi,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=3.0,
        major_prior=0.5,
        eps=1e-6,
    )

    assert tensor_terms.total.ndim == 0
    assert fit == pytest.approx(float(tensor_terms.fit.item()))
    assert penalty == pytest.approx(float(tensor_terms.penalty.item()))
    assert total == pytest.approx(float(tensor_terms.total.item()))
    assert torch.allclose(gamma, tensor_terms.gamma_major)


def test_numpy_scalar_loss_uses_exact_extreme_prior_like_torch() -> None:
    beta = np.array([0.02, 0.4, 0.9], dtype=np.float64)
    prior = 1e-12
    eps = 1e-6
    numpy_loss = _mutation_region_loss_grid_numpy(
        beta,
        alt=4.0,
        total=10.0,
        b_minus=0.25,
        b_plus=0.75,
        b_fixed=0.5,
        ambiguous=True,
        major_prior=prior,
        eps=eps,
    )
    torch_loss = mutation_region_loss_grid_torch(
        torch.tensor(beta, dtype=torch.float64),
        alt=torch.tensor(4.0, dtype=torch.float64),
        total=torch.tensor(10.0, dtype=torch.float64),
        b_minus=torch.tensor(0.25, dtype=torch.float64),
        b_plus=torch.tensor(0.75, dtype=torch.float64),
        b_fixed=torch.tensor(0.5, dtype=torch.float64),
        ambiguous=torch.tensor(True),
        major_prior=prior,
        eps=eps,
    )

    assert np.allclose(
        numpy_loss, torch_loss.detach().cpu().numpy(), rtol=1e-12, atol=1e-12
    )

    prob_minus = np.clip(beta * 0.25, eps, 1.0 - eps)
    prob_plus = np.clip(beta * 0.75, eps, 1.0 - eps)
    clipped_log_minor = (
        4.0 * np.log(prob_minus)
        + 6.0 * np.log1p(-prob_minus)
        + np.log(max(1.0 - prior, eps))
    )
    clipped_log_major = (
        4.0 * np.log(prob_plus) + 6.0 * np.log1p(-prob_plus) + np.log(max(prior, eps))
    )
    clipped_loss = -np.logaddexp(clipped_log_minor, clipped_log_major)
    assert not np.allclose(numpy_loss, clipped_loss, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("prior", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_major_prior_must_be_strict_probability(prior: float) -> None:
    data = _toy_torch_data()
    phi = torch.full((2, 2), 0.5, dtype=torch.float64)

    with pytest.raises(ValueError, match="major_prior"):
        mutation_region_terms_torch(data, phi, major_prior=prior, eps=1e-6)
