"""Exact Ward merge-history oracle using the former logical-node cost matrix."""

import os

import numpy as np
import pytest
import torch

from CliPP2.core.fusion.partition_starts import hessian_weighted_ward_label_sets_torch
from CliPP2.core.scalar import canonical_partition_labels


@torch.no_grad()
def _logical_matrix_reference(phi, curvature):
    """Frozen eda308d cost arithmetic with a dense logical row-major minimum."""
    count, regions = phi.shape
    nodes = 2 * count - 1
    weights = phi.new_zeros((nodes, regions))
    centers = torch.zeros_like(weights)
    weights[:count], centers[:count] = curvature, phi
    sentinel = torch.finfo(phi.dtype).max / 16.
    tiny = torch.finfo(phi.dtype).tiny
    costs = phi.new_full((nodes, nodes), sentinel)
    # Row-at-a-time evaluation retains the former initializer's operations.
    for row in range(count):
        denom = weights[row] + weights[:count]
        scale = weights[row] * weights[:count]
        scale.div_(denom.clamp_min(tiny))
        scale.masked_fill_(denom <= 0., 0.)
        difference = centers[row] - centers[:count]
        difference.square_().mul_(scale)
        cost = .5 * torch.sum(difference, dim=1)
        costs[row, row + 1:count] = cost[row + 1:]
    active = np.zeros(nodes, dtype=bool)
    active[:count] = True
    membership = torch.arange(count, device=phi.device)
    out = {count: canonical_partition_labels(membership.cpu().numpy())}
    for new_id in range(count, nodes):
        flat = int(torch.argmin(costs).item())
        left, right = divmod(flat, nodes)
        combined = weights[left] + weights[right]
        weights[new_id] = combined
        centers[new_id] = torch.where(
            combined > 0.,
            (weights[left] * centers[left] + weights[right] * centers[right])
            / combined.clamp_min(tiny),
            .5 * (centers[left] + centers[right]),
        )
        membership = torch.where(
            (membership == left) | (membership == right),
            torch.full_like(membership, new_id), membership,
        )
        active[[left, right]], active[new_id] = False, True
        costs[left, :], costs[:, left] = sentinel, sentinel
        costs[right, :], costs[:, right] = sentinel, sentinel
        other = torch.as_tensor(np.flatnonzero(active[:new_id]), device=phi.device)
        denom = weights[new_id].unsqueeze(0) + weights[other]
        scale = torch.where(
            denom > 0., weights[new_id].unsqueeze(0) * weights[other]
            / denom.clamp_min(tiny), torch.zeros_like(denom),
        )
        difference = centers[new_id].unsqueeze(0) - centers[other]
        costs[other, new_id] = .5 * torch.sum(scale * torch.square(difference), dim=1)
        out[nodes - new_id] = canonical_partition_labels(membership.cpu().numpy())
    return out


def _fixture(count, regions, dtype, kind, device="cpu"):
    generator = torch.Generator().manual_seed(174 + count + regions)
    phi = torch.rand((count, regions), dtype=dtype, generator=generator)
    curvature = torch.rand((count, regions), dtype=dtype, generator=generator) * 15
    if kind == "ties":
        phi.fill_(.5)
        curvature.fill_(1.)
    elif kind == "duplicates":
        phi[::2] = phi[0].clone()
        curvature[::2] = curvature[0].clone()
    elif kind == "zero_weights":
        curvature[::2] = 0.
    return phi.to(device), curvature.to(device)


@pytest.mark.parametrize("count,regions", [(2, 1), (7, 3), (19, 7), (37, 1), (41, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["random", "ties", "duplicates", "zero_weights"])
def test_every_ward_cut_matches_logical_matrix_reference(count, regions, dtype, kind):
    phi, curvature = _fixture(count, regions, dtype, kind)
    expected = _logical_matrix_reference(phi, curvature)
    actual = hessian_weighted_ward_label_sets_torch(
        phi, curvature, K_grid=range(1, count + 1), initial_pairwise_work_elements=23,
    )
    assert actual.keys() == expected.keys()
    for count in expected:
        np.testing.assert_array_equal(actual[count], expected[count])


@pytest.mark.skipif(os.environ.get("CLIPP2_TEST_CUDA") != "1", reason="LSF CUDA opt-in required")
def test_cuda_ward_slot_reuse_preserves_every_logical_merge():
    assert torch.cuda.is_available(), "CLIPP2_TEST_CUDA=1 requires qualified CUDA"
    for dtype in (torch.float32, torch.float64):
        for kind in ("random", "ties", "duplicates", "zero_weights"):
            phi, curvature = _fixture(37, 3, dtype, kind, "cuda")
            expected = _logical_matrix_reference(phi, curvature)
            actual = hessian_weighted_ward_label_sets_torch(
                phi, curvature, K_grid=range(1, 38), initial_pairwise_work_elements=51,
            )
            for count in expected:
                np.testing.assert_array_equal(actual[count], expected[count])


def test_ward_reuses_m_by_m_cost_storage_without_full_matrix_argmin(monkeypatch):
    allocated = []
    original = torch.Tensor.new_full

    def record_allocation(tensor, size, *args, **kwargs):
        allocated.append(tuple(size))
        return original(tensor, size, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "new_full", record_allocation)
    monkeypatch.setattr(torch, "argmin", lambda *a, **k: pytest.fail("global Ward argmin"))
    phi, curvature = _fixture(41, 3, torch.float64, "duplicates")
    hessian_weighted_ward_label_sets_torch(phi, curvature, K_grid=(1,))
    assert allocated == [(41, 41)]
