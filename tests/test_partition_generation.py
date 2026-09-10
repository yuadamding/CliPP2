"""Host-only Ward/CEM proposals retain the statistical and ordering contract."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
import torch

from CliPP2.config import DIRICHLET_ALPHA, DIRICHLET_CODE_WEIGHT, resolve_fit_config
from CliPP2.core.bic import effective_bic_mutation_region_count, fixed_partition_dirichlet_score
from CliPP2.core.fusion import partition_starts as partitions
from CliPP2.core.objective import compile_observed_model
from CliPP2.core.scalar import PartitionRefitResult, partition_constrained_observed_refit
from test_integer_likelihood import EPS, integer_data


def _refit(data, labels):
    return partition_constrained_observed_refit(data, labels, eps=EPS, tol=1e-6, max_iter=64)


def test_conditional_allocation_cost_removes_each_mutations_own_count():
    labels = np.array([0, 0, 0, 1])
    probability = np.exp(partitions._classification_leave_one_out_log_cluster_weights(
        labels, num_clusters=2,
    ))
    expected = np.array([[3 / 5, 2 / 5]] * 3 + [[4 / 5, 1 / 5]])
    np.testing.assert_allclose(probability, expected, rtol=0, atol=1e-16)
    count_cost = np.arange(8, dtype=float).reshape(4, 2)
    np.testing.assert_array_equal(
        partitions._classification_assignment_cost(count_cost, labels),
        count_cost - DIRICHLET_CODE_WEIGHT * np.log(expected),
    )
    assert DIRICHLET_ALPHA == 1.0 and DIRICHLET_CODE_WEIGHT == .7
    with pytest.raises(ValueError, match="one entry per mutation"):
        partitions._classification_assignment_cost(count_cost, labels[:-1])


def test_empty_cluster_repair_keeps_k_and_chooses_lowest_cost_eligible_donor():
    costs = np.array([[0., 10., 7.], [0., 9., 6.], [0., 8., 5.], [0., 7., 4.]])
    original = np.zeros(4, dtype=int)
    repaired = partitions._repair_empty_clusters(original, costs)
    np.testing.assert_array_equal(repaired, [0, 0, 2, 1])
    np.testing.assert_array_equal(original, np.zeros(4))
    assert len(np.unique(repaired)) == 3


@pytest.mark.parametrize("cost_mode", ["finite", "tied", "infeasible"])
def test_empty_cluster_repair_preserves_every_occupied_count(cost_mode):
    # Every assignment (including all-empty-but-one and singleton donors) for
    # N >= K must be repairable: an empty block implies a nonsingleton donor.
    for n in range(1, 5):
        for k in range(1, n + 1):
            costs = np.zeros((n, k))
            if cost_mode == "finite":
                costs[:] = np.arange(n * k).reshape(n, k) % 7
            elif cost_mode == "infeasible":
                costs[:, 1:] = np.inf
            for values in product(range(k), repeat=n):
                labels = np.array(values)
                repaired = partitions._repair_empty_clusters(labels, costs)
                np.testing.assert_array_equal(np.unique(repaired), np.arange(k))
                np.testing.assert_array_equal(labels, values)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_cem_returns_accepted_refit_with_canonical_fixed_k_labels(monkeypatch, k):
    data = integer_data(((1,),) * 6)
    labels = (np.arange(6) % k) * 7 + 8
    template = _refit(data, labels)
    refits = []

    def refit(current_labels):
        assert np.array_equal(current_labels, partitions._canonical_labels(current_labels))
        assert np.unique(current_labels).size == k
        loss = 200. if not refits else 100.
        result = replace(template, labels=current_labels, loglik=-loss, fit_loss=loss,
                         phi=template.cluster_centers[current_labels])
        refits.append(result)
        return result

    costs = np.full((6, k), 100.)
    costs[:, 0] = 0.
    monkeypatch.setattr(partitions, "_loss_to_centers", lambda *args, **kwargs: costs)
    result = partitions.refine_partition_likelihood(
        data, labels, eps=EPS, tol=1e-6, max_iter=1, _refit_labels=refit,
    )
    assert isinstance(result, PartitionRefitResult)
    assert result is refits[-1]
    assert result.n_clusters == k
    np.testing.assert_array_equal(np.unique(result.labels), np.arange(k))


@pytest.mark.parametrize("proposed_loss,accepted", [(250., False), (100., True)])
def test_cem_accepts_only_improvement_after_fixed_label_refit(monkeypatch, proposed_loss, accepted):
    data = integer_data(((1,),) * 4)
    initial = np.array([0, 0, 1, 1])
    proposed = np.array([0, 0, 0, 1])
    template = _refit(data, initial)
    calls = []

    def refit(labels):
        calls.append(labels.copy())
        loss = 200. if np.array_equal(labels, initial) else proposed_loss
        return replace(template, labels=labels, loglik=-loss, fit_loss=loss,
                       phi=template.cluster_centers[labels])

    monkeypatch.setattr(partitions, "_loss_to_centers",
                        lambda *args, **kwargs: np.array([[0., 8.]] * 3 + [[8., 0.]]))
    result = partitions.refine_partition_likelihood(
        data, initial, eps=EPS, tol=1e-6, max_iter=3, _refit_labels=refit,
    )
    np.testing.assert_array_equal(result.labels, proposed if accepted else initial)
    assert result.n_clusters == 2
    assert result.fit_loss == (proposed_loss if accepted else 200.)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], initial)
    np.testing.assert_array_equal(calls[1], proposed)


@pytest.mark.parametrize("proposed,current,accepted", [
    (100., 100., False),
    (np.nextafter(100., 0.), 100., False),
    (100. - 1e-12, 100., False),
    (100. - 1e-10, 100., True),
    (101., 100., False),
    (np.nan, 100., False),
    (np.inf, np.inf, False),
    (100., np.inf, True),
    (100., np.nan, True),
])
def test_cem_strict_score_gate_keeps_roundoff_and_nonfinite_contract(proposed, current, accepted):
    assert partitions._classification_score_strictly_improves(proposed, current) is accepted


def test_missing_positive_depth_rows_do_not_enter_refit_or_score():
    masked = replace(integer_data(((1,),) * 3), alt_counts=np.array([[2.], [3.], [9.]]),
                     total_counts=np.array([[10.], [12.], [14.]]),
                     count_observed=np.array([[True], [True], [False]]))
    dropped = replace(integer_data(((1,),) * 2), alt_counts=masked.alt_counts[:2],
                      total_counts=masked.total_counts[:2])
    masked_refit = _refit(masked, np.zeros(3, dtype=int))
    dropped_refit = _refit(dropped, np.zeros(2, dtype=int))
    np.testing.assert_array_equal(masked_refit.cluster_centers, dropped_refit.cluster_centers)
    assert masked_refit.loglik == dropped_refit.loglik
    assert effective_bic_mutation_region_count(masked) == effective_bic_mutation_region_count(dropped) == 2
    scores = [fixed_partition_dirichlet_score(
        loglik=refit.loglik, num_clusters=1, data=data, labels=refit.labels, partition_signature="",
    ).value for data, refit in ((masked, masked_refit), (dropped, dropped_refit))]
    assert scores[0] == scores[1]
    all_observed = replace(masked, count_observed=np.ones((3, 1), dtype=bool))
    assert masked_refit.loglik > _refit(all_observed, np.zeros(3, dtype=int)).loglik


def test_masked_ambiguous_region_does_not_affect_host_assignment_or_cem():
    data = integer_data(((1, 6), (2, 4), (6, 3), (4, 2)),
                        observed=np.array([[True, False]] * 4))
    low = replace(data, alt_counts=np.column_stack((data.alt_counts[:, 0], np.zeros(4))))
    high = replace(data, alt_counts=np.column_stack((data.alt_counts[:, 0], data.total_counts[:, 1])))
    centers = np.array([[.2, .3], [.8, .7]])
    np.testing.assert_array_equal(partitions._loss_to_centers(low, centers, eps=EPS),
                                  partitions._loss_to_centers(high, centers, eps=EPS))
    labels = np.array([0, 0, 1, 1])
    first, second = [partitions.refine_partition_likelihood(
        source, labels, eps=EPS, tol=1e-6, max_iter=3,
    ) for source in (low, high)]
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(first.phi, second.phi)
    assert first.loglik == second.loglik
    assert first.n_clusters == second.n_clusters == 2


def test_generator_keeps_both_families_one_refit_cache_and_ordered_dedup(monkeypatch):
    data = integer_data(((1,),) * 4)
    initial, improved = np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1])
    template = _refit(data, initial)
    source = compile_observed_model(data, eps=EPS)
    refit_calls, cem_calls = [], []

    def refit(data, labels, **kwargs):
        assert kwargs["_model"] is source
        refit_calls.append(labels.copy())
        loss = 20. if np.array_equal(labels, initial) else 10.
        return replace(template, labels=labels, phi=template.cluster_centers[labels],
                       loglik=-loss, fit_loss=loss)

    def cem(data, labels, **kwargs):
        assert kwargs["_model"] is source
        cem_calls.append(labels.copy())
        return kwargs["_refit_labels"](improved)

    monkeypatch.setattr(partitions, "partition_constrained_observed_refit", refit)
    monkeypatch.setattr(partitions, "refine_partition_likelihood", cem)
    # Different requested K with duplicate attained partitions must not create
    # repeated candidates or re-run the same immutable-label refit.
    candidates = partitions.generate_likelihood_partition_starts(
        data, eps=EPS, label_sets={3: initial, 2: initial}, tol=1e-6,
    )
    assert [candidate.source for candidate in candidates] == ["hessian_ward_cem_K2", "hessian_ward_K2"]
    assert [candidate.requested_k for candidate in candidates] == [2, 2]
    assert len(refit_calls) == 2 and len(cem_calls) == 2
    for candidate in candidates:
        assert candidate.K == 2 and candidate.component_death_count == 0
        np.testing.assert_array_equal(candidate.phi_start, template.cluster_centers[candidate.labels])
        assert candidate.bic == fixed_partition_dirichlet_score(
            loglik=-candidate.fit_loss, num_clusters=2, labels=candidate.labels,
            data=data, partition_signature="",
        ).value
    kept = partitions.generate_likelihood_partition_starts(
        data, eps=EPS, label_sets={2: initial}, tol=1e-6, max_candidates_per_K=1,
    )
    assert [candidate.source for candidate in kept] == ["hessian_ward_cem_K2"]


@pytest.mark.parametrize("removed", ["use_torch", "allow_component_death", "include_plain_ward",
                                      "include_ward_cem", "classification_weight_alpha",
                                      "classification_code_weight"])
def test_retired_generation_modes_are_not_silently_redirected(removed):
    data = integer_data(((1,), (1,)))
    with pytest.raises(TypeError, match=removed):
        partitions.generate_likelihood_partition_starts(
            data, eps=EPS, label_sets={1: np.zeros(2, dtype=int)}, **{removed: True},
        )


def test_retired_torch_refit_and_cem_symbols_are_absent():
    for name in ("partition_constrained_observed_refit_torch",
                 "refine_partition_likelihood_torch_with_trace", "_loss_to_centers_torch",
                 "PartitionRefinementResult", "refine_partition_likelihood_with_trace"):
        assert not hasattr(partitions, name)


def test_torch_ward_remains_chunk_independent():
    rng = np.random.default_rng(12)
    phi = torch.tensor(rng.uniform(.05, .95, (9, 4)), dtype=torch.float64)
    curvature = torch.tensor(rng.lognormal(0, .8, (9, 4)), dtype=torch.float64)
    kwargs = dict(K_grid=[1, 2, 4, 7, 9])
    full = partitions.hessian_weighted_ward_label_sets_torch(phi, curvature, **kwargs)
    chunked = partitions.hessian_weighted_ward_label_sets_torch(
        phi, curvature, **kwargs, initial_pairwise_work_elements=36,
    )
    assert full.keys() == chunked.keys()
    for k in full:
        np.testing.assert_array_equal(full[k], chunked[k])


def test_final_phi_pool_keeps_declared_k_grid_and_host_source(monkeypatch):
    data = integer_data(((2, 3), (3, 6), (6, 2), (4, 4)))
    from test_curvature_preparation import _context
    options = resolve_fit_config(device="cpu", dtype="float64")
    context = _context(data)
    seen = []
    ward = partitions.hessian_weighted_ward_label_sets_torch

    def capture(phi, curvature, **kwargs):
        seen.append(kwargs["K_grid"])
        return ward(phi, curvature, **kwargs)

    monkeypatch.setattr(partitions, "hessian_weighted_ward_label_sets_torch", capture)
    candidates = partitions.generate_partition_initializer_pool(
        context=context, pilot_phi=data.phi_init, fit_options=options,
        declared_k_grid=(4, 2, 2, 0, 7),
    )
    assert seen == [[2, 4]]
    assert candidates and all(candidate.requested_k in (2, 4) for candidate in candidates)
    assert isinstance(candidates, tuple)
