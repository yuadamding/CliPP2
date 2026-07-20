from __future__ import annotations

import numpy as np
import pytest
import torch

from CliPP2.core.bic import cluster_sizes_from_labels, compute_partition_icl
from CliPP2.core.fusion.partition_starts import (
    _classification_assignment_cost,
    _classification_log_cluster_weights,
    _loss_to_centers,
    _loss_to_centers_torch,
    generate_likelihood_partition_starts,
    hessian_weighted_ward_label_sets,
    hessian_weighted_ward_label_sets_torch,
    refine_partition_likelihood,
    refine_partition_likelihood_torch,
)
from CliPP2.io.data import TumorData


def _identical_mutation_tumor() -> TumorData:
    alt = np.full((3, 1), 5.0, dtype=np.float64)
    total = np.full((3, 1), 10.0, dtype=np.float64)
    ones = np.ones_like(alt)
    return TumorData(
        tumor_id="identical",
        mutation_ids=["m0", "m1", "m2"],
        region_ids=["r0"],
        alt_counts=alt,
        total_counts=total,
        purity=ones,
        major_cn=ones,
        minor_cn=ones,
        normal_cn=np.full_like(alt, 2.0),
        has_cna=np.zeros_like(alt, dtype=bool),
        scaling=np.full_like(alt, 0.5),
        phi_upper=ones,
        phi_init=ones,
        init_major_mask=np.ones_like(alt, dtype=bool),
        count_observed=np.ones_like(alt, dtype=bool),
    )


def _multi_region_tumor(
    *,
    masked_alt: float = 0.0,
    observe_second_region: bool = False,
) -> TumorData:
    alt = np.asarray(
        [[2.0, masked_alt], [3.0, masked_alt], [7.0, masked_alt], [8.0, masked_alt]],
        dtype=np.float64,
    )
    total = np.full_like(alt, 10.0)
    ones = np.ones_like(alt)
    return TumorData(
        tumor_id="multi-region",
        mutation_ids=["m0", "m1", "m2", "m3"],
        region_ids=["r0", "r1"],
        alt_counts=alt,
        total_counts=total,
        purity=ones,
        major_cn=ones,
        minor_cn=ones,
        normal_cn=np.full_like(alt, 2.0),
        has_cna=np.zeros_like(alt, dtype=bool),
        scaling=np.full_like(alt, 0.5),
        phi_upper=ones,
        phi_init=np.asarray(
            [[0.4, 0.3], [0.6, 0.3], [1.0, 0.7], [1.0, 0.7]],
            dtype=np.float64,
        ),
        init_major_mask=np.ones_like(alt, dtype=bool),
        count_observed=np.asarray(
            [
                [True, observe_second_region],
                [True, observe_second_region],
                [True, observe_second_region],
                [True, observe_second_region],
            ],
            dtype=bool,
        ),
    )


def test_classification_assignment_cost_uses_smoothed_cluster_proportions() -> None:
    labels = np.asarray([0, 0, 0, 1], dtype=np.int64)
    log_weights = _classification_log_cluster_weights(
        labels,
        num_clusters=2,
        alpha=1.0,
    )
    assert np.allclose(np.exp(log_weights), np.asarray([4.0 / 6.0, 2.0 / 6.0]))

    count_cost = np.zeros((4, 2), dtype=np.float64)
    assignment_cost = _classification_assignment_cost(
        count_cost,
        labels,
        alpha=1.0,
    )
    assert np.all(assignment_cost[:, 0] < assignment_cost[:, 1])

    with pytest.raises(ValueError, match="one entry per mutation"):
        _classification_assignment_cost(count_cost, labels[:-1], alpha=1.0)
    with pytest.raises(ValueError, match="positive and finite"):
        _classification_log_cluster_weights(labels, num_clusters=2, alpha=0.0)


def test_classification_refinement_can_remove_an_unsupported_component() -> None:
    data = _identical_mutation_tumor()
    labels = np.asarray([0, 0, 1], dtype=np.int64)

    refined_labels, refit = refine_partition_likelihood(
        data,
        labels,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-6,
        max_iter=3,
        refit_max_iter=32,
        classification_weight_alpha=1.0,
        allow_component_death=True,
    )

    assert np.unique(refined_labels).size == 1
    assert refit.n_clusters == 1


def test_legacy_refinement_keeps_requested_components() -> None:
    data = _identical_mutation_tumor()
    labels = np.asarray([0, 0, 1], dtype=np.int64)

    refined_labels, refit = refine_partition_likelihood(
        data,
        labels,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-6,
        max_iter=3,
        refit_max_iter=32,
    )

    assert np.unique(refined_labels).size == 2
    assert refit.n_clusters == 2


def test_cpu_and_torch_classification_refinement_agree() -> None:
    data = _identical_mutation_tumor()
    labels = np.asarray([0, 0, 1], dtype=np.int64)
    kwargs = dict(
        major_prior=0.5,
        eps=1e-6,
        tol=1e-6,
        max_iter=3,
        refit_max_iter=32,
        classification_weight_alpha=1.0,
        allow_component_death=True,
    )

    cpu_labels, cpu_refit = refine_partition_likelihood(data, labels, **kwargs)
    torch_labels, torch_refit = refine_partition_likelihood_torch(
        data,
        labels,
        device="cpu",
        dtype="float64",
        **kwargs,
    )

    assert np.array_equal(cpu_labels, torch_labels)
    assert np.allclose(cpu_refit.phi, torch_refit.phi, atol=1e-6)
    assert np.isclose(cpu_refit.fit_loss, torch_refit.fit_loss, atol=1e-6)


def test_masked_counts_do_not_affect_assignment_cost_or_refinement() -> None:
    low = _multi_region_tumor(masked_alt=0.0)
    high = _multi_region_tumor(masked_alt=10.0)
    centers = np.asarray([[0.4, 0.2], [1.0, 0.8]], dtype=np.float64)

    low_cost = _loss_to_centers(low, centers, major_prior=0.5, eps=1e-6)
    high_cost = _loss_to_centers(high, centers, major_prior=0.5, eps=1e-6)
    np.testing.assert_allclose(low_cost, high_cost, atol=1e-12)
    np.testing.assert_allclose(
        _loss_to_centers_torch(
            low,
            centers,
            major_prior=0.5,
            eps=1e-6,
            device="cpu",
            dtype="float64",
        ).numpy(),
        _loss_to_centers_torch(
            high,
            centers,
            major_prior=0.5,
            eps=1e-6,
            device="cpu",
            dtype="float64",
        ).numpy(),
        atol=1e-12,
    )

    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    kwargs = dict(
        major_prior=0.5,
        eps=1e-6,
        tol=1e-6,
        max_iter=3,
        refit_max_iter=32,
        classification_weight_alpha=1.0,
        allow_component_death=True,
    )
    low_labels, low_refit = refine_partition_likelihood(low, labels, **kwargs)
    high_labels, high_refit = refine_partition_likelihood(high, labels, **kwargs)
    assert np.array_equal(low_labels, high_labels)
    np.testing.assert_allclose(low_refit.phi, high_refit.phi, atol=1e-8)


def test_non_degenerate_cpu_and_torch_refinement_agree() -> None:
    data = _multi_region_tumor(masked_alt=4.0, observe_second_region=True)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    kwargs = dict(
        major_prior=0.5,
        eps=1e-6,
        tol=1e-6,
        max_iter=3,
        refit_max_iter=32,
        classification_weight_alpha=1.0,
        allow_component_death=True,
    )
    cpu_labels, cpu_refit = refine_partition_likelihood(data, labels, **kwargs)
    torch_labels, torch_refit = refine_partition_likelihood_torch(
        data,
        labels,
        device="cpu",
        dtype="float64",
        **kwargs,
    )
    assert np.array_equal(cpu_labels, torch_labels)
    np.testing.assert_allclose(cpu_refit.phi, torch_refit.phi, atol=1e-6)
    assert cpu_refit.fit_loss == pytest.approx(torch_refit.fit_loss, abs=1e-6)


def test_chunked_torch_ward_initial_costs_match_cpu_merges() -> None:
    rng = np.random.default_rng(12)
    pilot = rng.uniform(0.05, 0.95, size=(9, 4))
    curvature = rng.lognormal(mean=0.0, sigma=0.8, size=pilot.shape)
    k_grid = [1, 2, 4, 7, 9]

    expected = hessian_weighted_ward_label_sets(
        pilot,
        curvature,
        K_grid=k_grid,
    )
    observed = hessian_weighted_ward_label_sets_torch(
        torch.as_tensor(pilot, dtype=torch.float64),
        torch.as_tensor(curvature, dtype=torch.float64),
        K_grid=k_grid,
        device="cpu",
        dtype="float64",
        # Force one row at a time so the test exercises block boundaries.
        initial_pairwise_work_elements=pilot.shape[0] * pilot.shape[1],
    )

    assert observed.keys() == expected.keys()
    for k in k_grid:
        assert np.array_equal(
            observed[k][:, None] == observed[k][None, :],
            expected[k][:, None] == expected[k][None, :],
        )


def test_torch_ward_rejects_an_empty_initial_work_budget() -> None:
    pilot = np.asarray([[0.2], [0.8]], dtype=np.float64)
    curvature = np.ones_like(pilot)
    with pytest.raises(ValueError, match="must be positive"):
        hessian_weighted_ward_label_sets_torch(
            pilot,
            curvature,
            K_grid=[1],
            device="cpu",
            dtype="float64",
            initial_pairwise_work_elements=0,
        )


def test_generator_records_realized_k_and_realized_partition_icl() -> None:
    data = _identical_mutation_tumor()
    initial_labels = np.asarray([0, 0, 1], dtype=np.int64)
    candidates = generate_likelihood_partition_starts(
        data,
        exact_pilot=data.phi_init,
        major_prior=0.5,
        eps=1e-6,
        K_grid=[2],
        max_candidates_per_K=5,
        cem_max_iter=3,
        refit_max_iter=32,
        tol=1e-6,
        label_sets={2: initial_labels},
        use_torch=False,
        classification_weight_alpha=1.0,
        allow_component_death=True,
    )

    cem = next(candidate for candidate in candidates if "cem" in candidate.source)
    assert cem.K == np.unique(cem.labels).size
    assert cem.K == 1
    expected = compute_partition_icl(
        -cem.fit_loss,
        cluster_sizes_from_labels(cem.labels),
        data,
        alpha=1.0,
    )
    assert cem.bic == pytest.approx(expected)

    ward = next(candidate for candidate in candidates if "cem" not in candidate.source)
    assert cem.bic <= ward.bic
