"""Missingness invariance tests (audit §5.2).

When count_observed=False for a mutation_region:
  - its log-likelihood loss contribution must be exactly zero;
  - its gradient contribution must be exactly zero;
  - the BIC effective mutation_region count must exclude it (total_counts == 0 for unobserved mutation_regions).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from CliPP2.core.fusion.torch_backend import TorchTumorData, mutation_region_terms_torch
from CliPP2.io.data import TumorData
from CliPP2.runners.selection import (
    effective_bic_mutation_region_count,
    effective_bic_depth_count,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_torch_data(
    alt: list[float],
    total: list[float],
    count_observed: list[bool] | None = None,
    b_scale: float = 0.5,
) -> TorchTumorData:
    """Minimal TorchTumorData with S=1 sample per mutation."""
    M = len(alt)
    alt_t = torch.tensor([[float(a)] for a in alt], dtype=torch.float64)
    total_t = torch.tensor([[float(t)] for t in total], dtype=torch.float64)
    obs: torch.Tensor | None = None
    if count_observed is not None:
        obs = torch.tensor([[bool(o)] for o in count_observed], dtype=torch.bool)
    return TorchTumorData(
        alt=alt_t,
        total=total_t,
        nonalt=total_t - alt_t,
        phi_upper=torch.ones(M, 1, dtype=torch.float64),
        ambiguous=torch.zeros(M, 1, dtype=torch.bool),
        b_minus=torch.full((M, 1), b_scale * 0.5, dtype=torch.float64),
        b_plus=torch.full((M, 1), b_scale, dtype=torch.float64),
        b_fixed=torch.full((M, 1), b_scale * 0.75, dtype=torch.float64),
        count_observed=obs,
    )


def _make_tumor_data(
    alt: list[float],
    total: list[float],
    count_observed: list[bool],
) -> TumorData:
    M = len(alt)
    alt_np = np.array([[float(a)] for a in alt], dtype=np.float64)
    total_np = np.array([[float(t)] for t in total], dtype=np.float64)
    purity = np.ones((M, 1), dtype=np.float64) * 0.8
    major = np.ones((M, 1), dtype=np.float64) * 2.0
    minor = np.ones((M, 1), dtype=np.float64)
    normal = np.ones((M, 1), dtype=np.float64) * 2.0
    scaling = purity / (purity * (major + minor) + (1 - purity) * normal)
    return TumorData(
        tumor_id="test",
        mutation_ids=[f"m{i}" for i in range(M)],
        region_ids=["s1"],
        alt_counts=alt_np,
        total_counts=total_np,
        purity=purity,
        major_cn=major,
        minor_cn=minor,
        normal_cn=normal,
        has_cna=np.ones((M, 1), dtype=bool),
        scaling=scaling,
        phi_upper=np.ones((M, 1), dtype=np.float64),
        phi_init=np.full((M, 1), 0.5, dtype=np.float64),
        init_major_mask=np.ones((M, 1), dtype=bool),
        count_observed=np.array([[bool(o)] for o in count_observed], dtype=bool),
    )


# ── mutation_region_terms_torch masking ──────────────────────────────────────────────────


def test_unobserved_mutation_region_contributes_exactly_zero_loss() -> None:
    phi = torch.tensor([[0.5], [0.5]], dtype=torch.float64)
    data = _make_torch_data(alt=[5, 3], total=[10, 8], count_observed=[True, False])
    terms = mutation_region_terms_torch(data, phi, major_prior=0.7, eps=1e-6)

    assert float(terms.loss[1, 0]) == 0.0


def test_unobserved_mutation_region_contributes_exactly_zero_grad() -> None:
    phi = torch.tensor([[0.5], [0.5]], dtype=torch.float64)
    data = _make_torch_data(alt=[5, 3], total=[10, 8], count_observed=[True, False])
    terms = mutation_region_terms_torch(data, phi, major_prior=0.7, eps=1e-6)

    assert float(terms.grad[1, 0]) == 0.0


def test_unobserved_mutation_region_contributes_exactly_zero_curvature() -> None:
    phi = torch.tensor([[0.5], [0.5]], dtype=torch.float64)
    data = _make_torch_data(alt=[5, 3], total=[10, 8], count_observed=[True, False])
    terms = mutation_region_terms_torch(data, phi, major_prior=0.7, eps=1e-6)

    assert float(terms.hess_upper[1, 0]) == 0.0


def test_observed_mutation_region_loss_unchanged_by_adjacent_unobserved_mutation_region() -> (
    None
):
    phi = torch.tensor([[0.4], [0.5]], dtype=torch.float64)

    data_full = _make_torch_data(alt=[4, 3], total=[10, 8], count_observed=None)
    data_masked = _make_torch_data(
        alt=[4, 3], total=[10, 8], count_observed=[True, False]
    )

    terms_full = mutation_region_terms_torch(data_full, phi, major_prior=0.7, eps=1e-6)
    terms_masked = mutation_region_terms_torch(
        data_masked, phi, major_prior=0.7, eps=1e-6
    )

    assert float(terms_masked.loss[0, 0]) == pytest.approx(float(terms_full.loss[0, 0]))
    assert float(terms_masked.grad[0, 0]) == pytest.approx(float(terms_full.grad[0, 0]))


def test_total_loss_reduced_by_masking_one_mutation_region() -> None:
    phi = torch.tensor([[0.5], [0.5]], dtype=torch.float64)
    data_full = _make_torch_data(alt=[5, 3], total=[10, 8], count_observed=None)
    data_masked = _make_torch_data(
        alt=[5, 3], total=[10, 8], count_observed=[True, False]
    )

    terms_full = mutation_region_terms_torch(data_full, phi, major_prior=0.7, eps=1e-6)
    terms_masked = mutation_region_terms_torch(
        data_masked, phi, major_prior=0.7, eps=1e-6
    )

    assert float(terms_masked.loss.sum()) < float(terms_full.loss.sum())


def test_count_observed_none_equivalent_to_all_true() -> None:
    phi = torch.tensor([[0.4]], dtype=torch.float64)
    data_none = _make_torch_data(alt=[4], total=[10], count_observed=None)
    data_true = _make_torch_data(alt=[4], total=[10], count_observed=[True])

    terms_none = mutation_region_terms_torch(data_none, phi, major_prior=0.7, eps=1e-6)
    terms_true = mutation_region_terms_torch(data_true, phi, major_prior=0.7, eps=1e-6)

    assert float(terms_none.loss[0, 0]) == pytest.approx(float(terms_true.loss[0, 0]))
    assert float(terms_none.grad[0, 0]) == pytest.approx(float(terms_true.grad[0, 0]))


# ── BIC effective mutation_region count ──────────────────────────────────────────────────


def test_effective_bic_mutation_region_count_excludes_zero_depth_mutation_regions() -> (
    None
):
    """Cells with total_counts=0 (as produced by conversion.py for missing data)
    must not contribute to the BIC effective mutation_region count."""
    data = _make_tumor_data(
        alt=[5, 0],
        total=[10, 0],  # second mutation_region: depth=0 → unobserved
        count_observed=[True, False],
    )
    assert effective_bic_mutation_region_count(data) == 1


def test_effective_bic_counts_exclude_masked_positive_depth_mutation_regions() -> None:
    data = _make_tumor_data(
        alt=[5, 3],
        total=[10, 8],
        count_observed=[True, False],
    )

    assert effective_bic_mutation_region_count(data) == 1
    assert effective_bic_depth_count(data) == 10.0


def test_effective_bic_mutation_region_count_all_observed(tmp_path) -> None:
    data = _make_tumor_data(alt=[5, 3], total=[10, 8], count_observed=[True, True])
    assert effective_bic_mutation_region_count(data) == 2


def test_effective_bic_mutation_region_count_all_missing_returns_one() -> None:
    data = _make_tumor_data(alt=[0, 0], total=[0, 0], count_observed=[False, False])
    assert effective_bic_mutation_region_count(data) == 1  # max(..., 1) floor


# ── Missingness invariance: same loglik as omitting the mutation entirely ─────


def test_adding_unobserved_mutation_region_does_not_change_observed_loglik() -> None:
    """A fully-unobserved mutation_region (all regions masked) must contribute 0 to sum loss."""
    phi = torch.tensor([[0.5], [0.4], [0.6]], dtype=torch.float64)

    # Two fully observed mutations
    data_two = _make_torch_data(alt=[5, 3], total=[10, 8], count_observed=[True, True])
    # Three mutations; the third is entirely unobserved
    data_three = _make_torch_data(
        alt=[5, 3, 7], total=[10, 8, 12], count_observed=[True, True, False]
    )
    phi_two = phi[:2]

    terms_two = mutation_region_terms_torch(
        data_two, phi_two, major_prior=0.7, eps=1e-6
    )
    terms_three = mutation_region_terms_torch(
        data_three, phi, major_prior=0.7, eps=1e-6
    )

    # Total loss must match (unobserved mutation_region contributes nothing)
    assert float(terms_two.loss.sum()) == pytest.approx(float(terms_three.loss.sum()))
