"""Selective host reductions preserve marginalization and production consumers."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from CliPP2.core import objective
from CliPP2.core.fusion import partition_starts as partitions
from CliPP2.core.fusion.torch_backend import resolve_runtime
from test_integer_likelihood import EPS, enumeration, integer_data


def _data(purity=.7):
    major = np.array([[1, 6], [2, 5], [3, 4], [4, 3], [5, 2], [6, 1]])
    data = integer_data(major, purity=purity)
    minor = np.where(np.arange(12).reshape(6, 2) % 3 == 0, 0, data.minor_cn)
    minor[0, 1] = 6
    scaling = purity / (purity * (major + minor) + (1 - purity) * 2)
    alt = np.array([[0., 100.], [0., 0.], [10., 30.], [60., 45.], [5., 99.], [11., 8.]])
    total = np.full((6, 2), 100.)
    total[1] = 0
    observed = np.ones((6, 2), dtype=bool)
    observed[0, 1] = False
    upper = np.minimum(1., (1 - EPS) / (scaling * major))
    return replace(data, minor_cn=minor, scaling=scaling, phi_upper=upper,
                   alt_counts=alt, total_counts=total, count_observed=observed)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("purity", [.3, 1.])
@pytest.mark.parametrize("ccf", [EPS, .2, .75, 1 - EPS])
def test_loss_and_posterior_match_full_and_independent_mixture(dtype, purity, ccf):
    data = _data(purity)
    model = objective.compile_observed_model(data, eps=EPS)
    phi = np.full(model.shape, ccf, dtype=dtype)
    full = objective.observed_terms_numpy(model, phi, eps=EPS)
    loss = objective._observed_reduction_numpy(model, phi, eps=EPS, output="loss")
    posterior = objective._observed_reduction_numpy(model, phi, eps=EPS, output="posterior")
    expected_loss, expected_posterior = enumeration(data, phi.astype(np.float64))
    np.testing.assert_array_equal(loss, full.loss)
    np.testing.assert_array_equal(posterior, full.posterior)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(posterior, expected_posterior, rtol=1e-12, atol=1e-13)
    # Host reductions stay source-float64 even when working Phi was float32.
    assert loss.dtype == posterior.dtype == np.float64
    assert np.isfinite(loss).all() and np.isfinite(posterior).all()


@pytest.mark.parametrize("output", ["loss", "posterior", "terms"])
def test_derivative_array_footprint_is_measured_separately(monkeypatch, output):
    model = objective.compile_observed_model(_data(), eps=EPS)
    original = objective.candidate_terms_numpy
    arrays = []

    def measured(alt, nonalt, probability, slope, *, derivative_order=2):
        result = original(alt, nonalt, probability, slope, derivative_order=derivative_order)
        arrays.extend(array for array in (slope, result[1], result[2]) if array is not None)
        return result

    monkeypatch.setattr(objective, "candidate_terms_numpy", measured)
    objective._observed_reduction_numpy(model, np.full(model.shape, .5), eps=EPS, output=output)
    expected_count = 3 if output == "terms" else 0
    assert len(arrays) == expected_count
    assert sum(array.nbytes for array in arrays) == expected_count * model.slope.nbytes


def test_host_emission_without_derivatives_never_constructs_clipped_slope(monkeypatch):
    model = objective.compile_observed_model(_data(), eps=EPS)

    def unexpected(*args, **kwargs):
        pytest.fail("Loss/posterior-only emissions must not allocate clipped-slope masks.")

    monkeypatch.setattr(np, "where", unexpected)
    kernel = objective._emission_kernel_numpy(model, np.full(model.shape, .5),
                                              eps=EPS, derivatives=False)
    assert kernel.slope is None


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_center_costs_cem_and_map_match_full_derivative_route(monkeypatch, dtype):
    data = _data()
    model = objective.compile_observed_model(data, eps=EPS)
    runtime = objective.model_to_torch(model, resolve_runtime("cpu", dtype=dtype), eps=EPS)
    centers = torch.tensor([[.2, .3], [.55, .7], [.9, .95]],
                           dtype=runtime.alt.dtype).double().numpy()
    cost = partitions._loss_to_centers(data, centers, eps=EPS)
    expected = np.stack([objective.observed_terms_numpy(
        model, np.broadcast_to(center, model.shape), eps=EPS).loss.sum(axis=1)
        for center in centers], axis=1)
    np.testing.assert_array_equal(cost, expected)
    labels = np.array([0, 1, 0, 1, 0, 1])
    actual = partitions.refine_partition_likelihood(
        data, labels, eps=EPS, tol=1e-6, max_iter=3, refit_max_iter=32,
    )

    def full_reduction(model, phi, *, eps, output):
        return getattr(objective.observed_terms_numpy(model, phi, eps=eps), output)

    monkeypatch.setattr(partitions, "_observed_reduction_numpy", full_reduction)
    reference = partitions.refine_partition_likelihood(
        data, labels, eps=EPS, tol=1e-6, max_iter=3, refit_max_iter=32,
    )
    for name in ("labels", "phi", "cluster_centers"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(reference, name))
    assert actual.loglik == reference.loglik
    assert partitions._classification_refit_score(data, actual.labels, actual) == (
        partitions._classification_refit_score(data, reference.labels, reference))

    phi = np.minimum(np.full(model.shape, .75), model.upper)
    posterior = objective.observed_terms_numpy(model, phi, eps=EPS).posterior
    result = objective.infer_integer_multiplicity_posterior_numpy(data, phi, eps=EPS)
    np.testing.assert_array_equal(result.posterior, posterior)
    np.testing.assert_array_equal(result.multiplicity_call, np.argmax(posterior, axis=-1) + 1)
    np.testing.assert_array_equal(result.map_probability, np.max(posterior, axis=-1))


def test_reporting_uses_posterior_only_reduction(monkeypatch):
    data = _data()
    original = objective._observed_reduction_numpy
    outputs = []

    def checked(*args, **kwargs):
        outputs.append(kwargs["output"])
        return original(*args, **kwargs)

    monkeypatch.setattr(objective, "_observed_reduction_numpy", checked)
    objective.infer_integer_multiplicity_posterior_numpy(data, np.full((6, 2), .7), eps=EPS)
    assert outputs == ["posterior"]


def test_retired_emission_names_have_no_compatibility_aliases():
    for cls in (objective.ObservedModel, objective.TorchObservedModel):
        assert not hasattr(cls, "path_shape")
    for name in ("_path_kernel_numpy", "_path_kernel_torch", "_NumpyPathKernel", "_TorchPathKernel"):
        assert not hasattr(objective, name)
