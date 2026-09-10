"""Exact cluster-region reuse without changing scalar policy or certificates."""

from dataclasses import fields, replace
from time import perf_counter
import weakref

import numpy as np
import pytest

from CliPP2.core import scalar
from CliPP2.core.fusion import partition_starts as starts
from test_integer_likelihood import EPS, integer_data


P = np.repeat(np.arange(4), 2)
Q = np.array([0, 0, 1, 2, 1, 2, 3, 3])


def _data():
    data = integer_data(((1, 2), (2, 1), (1, 3), (2, 2)) * 2)
    return replace(data, alt_counts=np.tile(
        np.array([5, 5, 10, 30, 10, 30, 35, 35], dtype=float)[:, None], (1, 2),
    ))


def _refit(data, labels=P, cache=None, **kwargs):
    options = dict(eps=EPS, tol=1e-4, max_iter=16)
    options.update(kwargs)
    return scalar.partition_constrained_observed_refit(
        data, labels, _coordinate_cache=cache, **options,
    )


def _assert_equal(left, right):
    assert type(left) is type(right)
    for item in fields(left):
        first, second = getattr(left, item.name), getattr(right, item.name)
        if isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second, err_msg=item.name)
        else:
            assert first == second, item.name


@pytest.fixture
def coordinate_calls(monkeypatch):
    calls = []
    original = scalar._fit_coordinate

    def counted(problem, **kwargs):
        calls.append(kwargs)
        return original(problem, **kwargs)

    monkeypatch.setattr(scalar, "_fit_coordinate", counted)
    return calls


@pytest.mark.parametrize("mode", ["interval_certified", "grid_local"])
def test_cache_reuses_exact_clusters_and_all_logical_refit_fields(mode, coordinate_calls):
    data = _data()
    expected = [_refit(data, labels, scalar_mode=mode) for labels in (P, Q)]
    assert len(coordinate_calls) == 16
    coordinate_calls.clear()
    cache = scalar._RefitCoordinateCache()
    actual = [_refit(data, labels, cache, scalar_mode=mode) for labels in (P, Q)]
    assert len(coordinate_calls) == 12  # Two shared clusters, in both regions.
    for left, right in zip(expected, actual):
        _assert_equal(left, right)
    _assert_equal(_refit(data, P * 7 + 10, cache, scalar_mode=mode), expected[0])
    assert len(coordinate_calls) == 12


def test_different_k_changes_coordinate_tolerance_and_cannot_reuse(coordinate_calls):
    data = _data()
    cache = scalar._RefitCoordinateCache()
    _refit(data, P, cache)
    # Preserve three blocks but split the last one: every coordinate now has
    # tolerance tol/(5*S), so even unchanged memberships must be solved again.
    split = np.array([0, 0, 1, 1, 2, 2, 3, 4])
    actual = _refit(data, split, cache)
    assert len(coordinate_calls) == 18
    assert {call["tolerance"] for call in coordinate_calls} == {1e-4 / 8, 1e-4 / 10}
    _assert_equal(actual, _refit(data, split))


@pytest.mark.parametrize("change", [
    "counts", "ids", "bounds", "observed", "eps", "tol", "max_iter",
    "scalar_mode", "scalar_grid_points", "scalar_local_steps",
])
def test_source_bounds_and_numerical_policy_never_reuse_stale_result(change, coordinate_calls):
    data = _data()
    cache = scalar._RefitCoordinateCache()
    _refit(data, cache=cache)
    options = {}
    if change == "counts":
        data = replace(data, alt_counts=data.alt_counts + 1.0)
    elif change == "ids":
        data = replace(data, mutation_ids=tuple(f"new{i}" for i in range(8)))
    elif change == "bounds":
        data = replace(data, phi_upper=np.full((8, 2), .8))
    elif change == "observed":
        data = replace(data, count_observed=np.zeros((8, 2), dtype=bool))
    else:
        options[change] = dict(
            eps=2 * EPS, tol=2e-4, max_iter=32, scalar_mode="grid_local",
            scalar_grid_points=32, scalar_local_steps=2,
        )[change]
    actual = _refit(data, cache=cache, **options)
    assert len(coordinate_calls) == 16
    _assert_equal(actual, _refit(data, **options))


def test_missing_and_zero_depth_coordinates_retain_exact_certificate_fields(coordinate_calls):
    data = _data()
    alt, total = data.alt_counts.copy(), data.total_counts.copy()
    alt[4:6] = total[4:6] = 0.0
    observed = np.ones((8, 2), dtype=bool)
    observed[:2, 1] = False
    data = replace(data, alt_counts=alt, total_counts=total, count_observed=observed)
    cache = scalar._RefitCoordinateCache()
    expected = _refit(data)
    actual = _refit(data, cache=cache)
    _assert_equal(actual, expected)
    _assert_equal(_refit(data, cache=cache), expected)
    assert len(coordinate_calls) == 16
    assert np.all(actual.cluster_centers[2] == (1.0 + EPS) / 2.0)
    assert actual.cluster_centers[0, 1] == (1.0 + EPS) / 2.0


def test_exhausted_certificate_is_reused_without_becoming_resolved(monkeypatch, coordinate_calls):
    def exhausted(problem, **kwargs):
        return scalar.ScalarGlobalMinimumCertificate(
            .4, 123.0, 120.0, 3.0, False, "interval_binomial_mixture_bound_v1",
            kwargs["max_intervals"],
        )

    monkeypatch.setattr(scalar, "certify_scalar_minimum", exhausted)
    data, cache = _data(), scalar._RefitCoordinateCache()
    expected = _refit(data)
    actual = _refit(data, cache=cache)
    _assert_equal(actual, expected)
    _assert_equal(_refit(data, cache=cache), expected)
    assert len(coordinate_calls) == 16
    assert not actual.global_optimum_certified
    assert actual.global_optimality_gap == 24.0
    assert actual.global_certificate_intervals == 8 * 4096
    assert actual.refit_total_refined_candidates == expected.refit_total_refined_candidates


def test_cache_bounds_entry_overhead_and_membership_bytes_and_evicts_lru():
    data, populated = _data(), scalar._RefitCoordinateCache()
    _refit(data, cache=populated)
    entries = list(populated._entries.items())
    cache = scalar._RefitCoordinateCache(max_entries=2, max_membership_bytes=32)
    for key, value in entries[:2]:
        cache.put(key, value)
    assert cache.get(entries[0][0]) is entries[0][1]  # Refresh the oldest entry.
    cache.put(*entries[2])
    assert cache.get(entries[1][0]) is None
    assert cache.get(entries[0][0]) is entries[0][1]
    assert len(cache._entries) == 2 and cache._membership_bytes == 32
    cache.put(*entries[0])
    assert len(cache._entries) == 2 and cache._membership_bytes == 32
    bytes_only = scalar._RefitCoordinateCache(max_entries=100, max_membership_bytes=16)
    for key, value in entries:
        bytes_only.put(key, value)
    assert list(bytes_only._entries) == [entries[-1][0]]
    assert bytes_only._membership_bytes == 16
    # An oversize cluster is not cached and must not evict useful small entries.
    huge = replace(entries[-1][0], members=b"x" * 17)
    bytes_only.put(huge, entries[-1][1])
    assert list(bytes_only._entries) == [entries[-1][0]]
    assert bytes_only.get(replace(entries[-1][0], include_breakpoints=False)) is None
    empty = scalar._RefitCoordinateCache(max_entries=0)
    empty.put(*entries[0])
    assert not empty._entries


def test_cache_retains_no_source_or_compiled_model():
    data, cache = _data(), scalar._RefitCoordinateCache()
    model = scalar.compile_observed_model(data, eps=EPS)
    source_ref, array_ref = weakref.ref(data), weakref.ref(model.alt)
    _refit(data, cache=cache)
    del data, model
    assert source_ref() is None and array_ref() is None
    assert cache._entries


def test_proposal_pool_cache_preserves_proposals_and_reduces_scalar_work(
    monkeypatch, coordinate_calls,
):
    data = replace(integer_data(((1, 1),) * 8), alt_counts=np.tile(
        np.array([5, 5, 10, 30, 10, 30, 35, 35], dtype=float)[:, None], (1, 2),
    ))
    timings, counts, outputs = [], [], []
    for enabled in (False, True):
        monkeypatch.setattr(starts, "_RefitCoordinateCache", lambda: scalar._RefitCoordinateCache(
            max_entries=1024 if enabled else 0,
        ))
        coordinate_calls.clear()
        before = perf_counter()
        outputs.append(starts.generate_likelihood_partition_starts(
            data, eps=EPS, label_sets={4: P}, cem_max_iter=8, tol=1e-4,
        ))
        timings.append(perf_counter() - before)
        counts.append(len(coordinate_calls))
    assert len(outputs[0]) == len(outputs[1])
    for left, right in zip(*outputs):
        _assert_equal(left, right)
    assert counts[1] < counts[0]
    print(f"proposal scalar calls uncached/cached={counts}; seconds={timings}")
