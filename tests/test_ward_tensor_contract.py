"""Ward normalizes inputs once at preparation and preserves exact merge order."""

import numpy as np
import pytest
import torch

from CliPP2.config import resolve_fit_config
from CliPP2.core.fusion import partition_starts as partitions, torch_backend as backend
from test_curvature_preparation import _context
from test_integer_likelihood import integer_data


@pytest.mark.parametrize("change", ["numpy_pilot", "numpy_curvature", "shape", "rank",
                                    "dtype", "device", "float16", "integer"])
def test_ward_rejects_nonprepared_inputs(change):
    phi = torch.full((4, 2), .5, dtype=torch.float64)
    curvature = torch.ones_like(phi)
    if change == "numpy_pilot":
        phi = phi.numpy()
    elif change == "numpy_curvature":
        curvature = curvature.numpy()
    elif change == "shape":
        curvature = curvature[:1]
    elif change == "rank":
        phi, curvature = phi.flatten(), curvature.flatten()
    elif change == "dtype":
        curvature = curvature.float()
    elif change == "device":
        curvature = torch.empty_like(curvature, device="meta")
    elif change == "float16":
        phi, curvature = phi.half(), curvature.half()
    else:
        phi, curvature = phi.long(), curvature.long()
    with pytest.raises((TypeError, ValueError), match="Ward requires|shape|dtype and device"):
        partitions.hessian_weighted_ward_label_sets_torch(phi, curvature, K_grid=(1,))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("regions", [1, 3])
@pytest.mark.parametrize("budget", [1, 12, 4_000_000])
def test_tied_ward_merges_preserve_row_major_order_without_runtime_resolution(monkeypatch, dtype, regions, budget):
    phi = torch.full((4, regions), .5, dtype=dtype)
    curvature = torch.ones_like(phi)
    original_phi, original_curvature = phi.clone(), curvature.clone()
    monkeypatch.setattr(backend, "resolve_runtime", lambda *a, **k: pytest.fail("resolved a runtime"))
    actual = partitions.hessian_weighted_ward_label_sets_torch(
        phi, curvature, K_grid=(4, 3, 2, 1), initial_pairwise_work_elements=budget,
    )
    # Equal initial costs merge (0,1), then untouched (2,3), then the two blocks.
    assert {k: labels.tolist() for k, labels in actual.items()} == {
        4: [0, 1, 2, 3], 3: [0, 0, 1, 2], 2: [0, 0, 1, 1], 1: [0, 0, 0, 0],
    }
    assert torch.equal(phi, original_phi) and torch.equal(curvature, original_curvature)
    assert "resolve_runtime" not in vars(partitions)


@pytest.mark.parametrize("supplied", [False, True])
def test_proposal_boundary_converts_each_optional_array_once(monkeypatch, supplied):
    context = _context(integer_data(((1, 2), (3, 4), (2, 1))))
    options = resolve_fit_config(device="cpu", dtype="float64")
    pilot = context.exact_pilot.numpy().copy()
    curvature = np.ones_like(pilot)
    conversions, ward_inputs = [], []
    convert = partitions.as_runtime_tensor

    def boundary(value, runtime):
        conversions.append(value)
        return convert(value, runtime)

    def ward(phi, h, *, K_grid):
        assert phi.dtype == h.dtype == context.model.alt.dtype
        assert phi.device == h.device == context.model.alt.device
        assert phi.shape == h.shape == context.model.shape
        ward_inputs.append((phi, h))
        return {1: np.zeros(3, dtype=int)}

    monkeypatch.setattr(partitions, "as_runtime_tensor", boundary)
    monkeypatch.setattr(partitions, "hessian_weighted_ward_label_sets_torch", ward)
    monkeypatch.setattr(partitions, "generate_likelihood_partition_starts", lambda *a, **k: [])
    monkeypatch.setattr(backend, "resolve_runtime", lambda *a, **k: pytest.fail("resolved a runtime"))
    partitions.generate_partition_initializer_pool(
        context=context, pilot_phi=pilot, fit_options=options,
        curvature=curvature if supplied else None,
    )
    assert len(ward_inputs) == 1
    assert len(conversions) == (2 if supplied else 1)
    assert conversions[0] is pilot
    if supplied:
        assert conversions[1] is curvature


def test_ward_retains_empty_grid_and_invalid_work_budget_contract():
    phi = torch.ones((1, 2), dtype=torch.float64)
    assert partitions.hessian_weighted_ward_label_sets_torch(phi, phi, K_grid=(0, 2)) == {}
    with pytest.raises(ValueError, match="initial_pairwise_work_elements"):
        partitions.hessian_weighted_ward_label_sets_torch(
            phi, phi, K_grid=(1,), initial_pairwise_work_elements=0,
        )
