"""Exact cluster-region reuse without changing scalar policy or certificates."""

from dataclasses import asdict, fields, replace
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
    uncached = scalar._ScalarWorkStats()
    expected = [
        _refit(data, labels, scalar_mode=mode, _work_stats=uncached) for labels in (P, Q)
    ]
    assert len(coordinate_calls) == 16
    coordinate_calls.clear()
    cache = scalar._RefitCoordinateCache()
    actual = [_refit(data, labels, cache, scalar_mode=mode) for labels in (P, Q)]
    assert len(coordinate_calls) == 12  # Two shared clusters, in both regions.
    for left, right in zip(expected, actual):
        _assert_equal(left, right)
    assert uncached.scalar_solves == 16 and cache.work.scalar_solves == 12
    assert (cache.work.cache_hits, cache.work.cache_misses) == (4, 12)
    work_field = "interval_evaluations" if mode == "interval_certified" else "grid_points_evaluated"
    logical_field = "global_certificate_intervals" if mode == "interval_certified" else "refit_total_grid_points"
    assert 0 < getattr(cache.work, work_field) < getattr(uncached, work_field)
    assert getattr(uncached, work_field) == sum(getattr(item, logical_field) for item in expected)
    before = asdict(cache.work)
    _assert_equal(_refit(data, P * 7 + 10, cache, scalar_mode=mode), expected[0])
    assert len(coordinate_calls) == 12
    assert asdict(cache.work) == {**before, "cache_hits": 12}
    assert uncached.scalar_seconds > 0 and cache.work.scalar_seconds > 0


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
    before = asdict(cache.work)
    _assert_equal(_refit(data, cache=cache), expected)
    assert asdict(cache.work) == {**before, "cache_hits": 8}
    assert cache.work.scalar_solves == 8
    assert cache.work.interval_evaluations == expected.global_certificate_intervals
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
    assert (cache.work.scalar_solves, cache.work.cache_hits, cache.work.cache_misses) == (8, 8, 8)
    # The injected certificate reports logical work, but executes no bounds.
    assert cache.work.interval_evaluations == 0


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
    assert cache.work.cache_evictions == 1
    assert bytes_only.work.cache_evictions == len(entries) - 1


def test_cache_retains_no_source_or_compiled_model():
    data, cache = _data(), scalar._RefitCoordinateCache()
    model = scalar.compile_observed_model(data, eps=EPS)
    source_ref, array_ref = weakref.ref(data), weakref.ref(model.alt)
    _refit(data, cache=cache)
    del data, model
    assert source_ref() is None and array_ref() is None
    assert cache._entries
    assert all(isinstance(value, (int, float)) for value in asdict(cache.work).values())


@pytest.mark.parametrize("capacity", [0, 1])
def test_disabled_and_evicted_cache_repeats_physical_work_without_logical_change(capacity):
    data, cache = _data(), scalar._RefitCoordinateCache(max_entries=capacity)
    first = _refit(data, cache=cache, scalar_mode="grid_local")
    before = asdict(cache.work)
    _assert_equal(_refit(data, cache=cache, scalar_mode="grid_local"), first)
    assert cache.work.cache_hits == 0
    for field in ("cache_misses", "scalar_solves", "grid_points_evaluated"):
        assert getattr(cache.work, field) == 2 * before[field]
    assert cache.work.cache_evictions == (15 if capacity else 0)
    assert len(cache._entries) == capacity


@pytest.mark.parametrize("mode", ["interval_certified", "grid_local"])
def test_failed_scalar_work_is_measured_and_never_cached(mode, monkeypatch):
    data, cache = _data(), scalar._RefitCoordinateCache()
    calls = []
    original = scalar._interval_lower_bound

    def failing_bound(problem, left, right):
        calls.append(1)
        if len(calls) % 4 == 0:
            raise RuntimeError("injected bound failure")
        return original(problem, left, right)

    def failing_grid(problem, beta):
        calls.append(np.asarray(beta).size)
        raise RuntimeError("injected grid failure")

    clock = iter([10.0, 12.0, 20.0, 25.0])
    monkeypatch.setattr(scalar, "perf_counter", lambda: next(clock))
    if mode == "interval_certified":
        monkeypatch.setattr(scalar, "_interval_lower_bound", failing_bound)
    else:
        monkeypatch.setattr(scalar, "scalar_loss", failing_grid)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="injected"):
            _refit(data, cache=cache, scalar_mode=mode)
    work = cache.work
    assert not cache._entries
    assert (work.cache_hits, work.cache_misses, work.scalar_solves, work.scalar_failures) == (0, 2, 2, 2)
    assert work.scalar_seconds == 7.0
    assert work.interval_evaluations == (sum(calls) if mode == "interval_certified" else 0)
    assert work.grid_points_evaluated == (sum(calls) if mode == "grid_local" else 0)


def test_scalar_shortcuts_and_unresolved_budget_count_only_dispatched_bounds():
    data = _data()
    model = scalar.compile_observed_model(data, eps=EPS)
    problem = scalar.scalar_problem_from_model(
        model, np.array([2, 3]), 0, lower=EPS, upper=1.0, eps=EPS,
    )
    work = scalar._ScalarWorkStats()
    fixed = scalar.certify_scalar_minimum(
        replace(problem, lower=0.4, upper=0.4), tolerance=1e-4, max_intervals=1,
        _work_stats=work,
    )
    assert fixed.intervals_evaluated == 1 and work.interval_evaluations == 0
    missing = scalar.certify_scalar_minimum(
        replace(problem, observed=np.zeros(2, dtype=bool)), tolerance=1e-4,
        max_intervals=1, _work_stats=work,
    )
    assert missing.intervals_evaluated == work.interval_evaluations == 0
    unresolved = scalar.certify_scalar_minimum(
        problem, tolerance=1e-30, max_intervals=1, _work_stats=work,
    )
    assert not unresolved.globally_certified
    assert unresolved.intervals_evaluated == work.interval_evaluations == 1


def test_conflicting_work_sinks_fail_before_work():
    cache, other = scalar._RefitCoordinateCache(), scalar._ScalarWorkStats()
    with pytest.raises(ValueError, match="share one physical-work sink"):
        _refit(_data(), cache=cache, _work_stats=other)
    assert cache.work.scalar_solves == other.scalar_solves == 0


def test_proposal_pool_cache_preserves_proposals_and_reduces_scalar_work(
    monkeypatch, coordinate_calls,
):
    data = replace(integer_data(((1, 1),) * 8), alt_counts=np.tile(
        np.array([5, 5, 10, 30, 10, 30, 35, 35], dtype=float)[:, None], (1, 2),
    ))
    timings, counts, outputs, diagnostics = [], [], [], []
    for enabled in (False, True):
        monkeypatch.setattr(starts, "_RefitCoordinateCache", lambda **kwargs: scalar._RefitCoordinateCache(
            max_entries=1024 if enabled else 0, **kwargs,
        ))
        work = scalar._ScalarWorkStats()
        coordinate_calls.clear()
        before = perf_counter()
        outputs.append(starts.generate_likelihood_partition_starts(
            data, eps=EPS, label_sets={4: P}, cem_max_iter=8, tol=1e-4, _work_stats=work,
        ))
        timings.append(perf_counter() - before)
        counts.append(len(coordinate_calls))
        diagnostics.append(work)
    assert len(outputs[0]) == len(outputs[1])
    for left, right in zip(*outputs):
        _assert_equal(left, right)
    assert counts[1] < counts[0]
    assert [work.scalar_solves for work in diagnostics] == counts
    assert diagnostics[1].interval_evaluations < diagnostics[0].interval_evaluations
    assert diagnostics[0].cache_hits == 0 < diagnostics[1].cache_hits
    print(f"proposal scalar calls uncached/cached={counts}; seconds={timings}")
    print(f"physical work uncached/cached={[asdict(work) for work in diagnostics]}")
