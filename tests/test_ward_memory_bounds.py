"""Bound temporary Ward storage without changing any logical merge or tie."""

import os
import json

import numpy as np
import pytest
import torch

from CliPP2.core.fusion import partition_starts as ward
from test_ward_compact_storage import _fixture, _logical_matrix_reference


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["random", "ties", "duplicates", "zero_weights"])
@pytest.mark.parametrize("budget", [1, 97, 1_000_000])
def test_batched_refresh_and_compaction_preserve_every_cut(monkeypatch, dtype, kind, budget):
    # Force repeated compaction, including exact ties and reused slot order.
    monkeypatch.setattr(ward, "_WARD_HEAP_MIN_ENTRIES", 1)
    monkeypatch.setattr(ward, "_WARD_HEAP_ACTIVE_MULTIPLIER", 1)
    phi, curvature = _fixture(41, 3, dtype, kind)
    originals = phi.clone(), curvature.clone()
    expected = _logical_matrix_reference(phi, curvature)
    actual = ward.hessian_weighted_ward_label_sets_torch(
        phi, curvature, K_grid=range(1, 42), initial_pairwise_work_elements=257,
        refresh_work_elements=budget,
    )
    for count in expected:
        np.testing.assert_array_equal(actual[count], expected[count])
    assert torch.equal(phi, originals[0]) and torch.equal(curvature, originals[1])


@pytest.mark.parametrize("budget", [1, 13, 51, 1_000_000])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_each_refresh_gather_has_explicit_element_bound(monkeypatch, budget, dtype):
    # Logical IDs are sorted; physical slots intentionally are not. Every row
    # is tied, so the smallest logical column must win in every row batch.
    slots = np.random.default_rng(741).permutation(13)
    active_ids = np.array([0, 2, 3, 6, 8, 9, 11])
    rows = np.array([0, 1, 2, 3, 4, 6, 7, 9, 11])
    matrix = torch.ones((13, 13), dtype=dtype)
    best_cost, best_column = np.zeros(13), np.zeros(13, dtype=np.int64)
    minimum, shapes = torch.min, []

    def measured_minimum(value, *args, **kwargs):
        shapes.append(tuple(value.shape))
        return minimum(value, *args, **kwargs)

    monkeypatch.setattr(torch, "min", measured_minimum)
    ward._ward_refresh_minima(matrix, rows, active_ids, slots, best_cost, best_column, budget)
    assert shapes and sum(shape[0] for shape in shapes) == rows.size
    assert max(np.prod(shape) for shape in shapes) <= max(budget, active_ids.size)
    np.testing.assert_array_equal(best_cost[rows], 1)
    np.testing.assert_array_equal(best_column[rows], active_ids[0])


def test_lazy_heap_is_linear_bounded_and_compaction_removes_only_stale_entries(monkeypatch):
    phi, curvature = _fixture(151, 3, torch.float64, "random")
    push, compact = ward.heapq.heappush, ward._ward_compact_heap
    sizes, compactions = [], []

    def measured_push(heap, value):
        push(heap, value)
        sizes.append(len(heap))

    def checked_compact(heap, active, costs, columns, versions, sentinel):
        current = sorted(
            entry for entry in heap
            if active[entry[1]] and active[entry[2]]
            and versions[entry[1]] == entry[3] and columns[entry[1]] == entry[2]
        )
        compact(heap, active, costs, columns, versions, sentinel)
        assert sorted(heap) == current
        assert len(heap) <= np.count_nonzero(active)
        compactions.append(len(heap))

    monkeypatch.setattr(ward.heapq, "heappush", measured_push)
    monkeypatch.setattr(ward, "_ward_compact_heap", checked_compact)
    actual = ward.hessian_weighted_ward_label_sets_torch(phi, curvature, K_grid=(1, 4, 9))
    assert compactions
    # A previous bounded heap plus at most M updates can coexist just before
    # compaction; the replacement list contributes at most M additional tuples.
    assert max(sizes) <= max(ward._WARD_HEAP_MIN_ENTRIES, 4 * len(phi)) + len(phi)
    monkeypatch.setattr(ward, "_WARD_HEAP_MIN_ENTRIES", 10**12)
    expected = ward.hessian_weighted_ward_label_sets_torch(phi, curvature, K_grid=(1, 4, 9))
    for count in expected:
        np.testing.assert_array_equal(actual[count], expected[count])


def test_refresh_work_budget_must_be_positive():
    phi = torch.ones((3, 1))
    with pytest.raises(ValueError, match="refresh_work_elements"):
        ward.hessian_weighted_ward_label_sets_torch(phi, phi, K_grid=(1,), refresh_work_elements=0)


def test_benchmark_smoke_receipt_binds_source_environment_settings_and_repeated_cuts(capsys):
    from tools.benchmark_ward import main

    main(["--mutations", "17", "--regions", "3", "--repeats", "2", "--fixture", "refresh-stress"])
    record = json.loads(capsys.readouterr().out)
    assert len(record["seconds"]) == 2
    for name in ("environment_sha256", "settings_sha256", "input_sha256", "result_sha256"):
        assert len(record[name]) == 64
    assert len(record["source"]["runtime_sha256"]) == 64
    assert record["allocations"]["persistent_cost_matrix_bytes"] == 17 * 17 * 4
    assert record["allocations"]["refresh_gather_peak_elements"] == 15 * 16
    assert record["allocations"]["cuda_peak_allocated_delta_bytes"] is None
    assert record["instrumented_cpu_phases"]


def test_benchmark_cuda_requires_explicit_source_and_lsf_opt_in(monkeypatch):
    from tools.benchmark_ward import main

    monkeypatch.delenv("CLIPP2_TEST_CUDA", raising=False)
    with pytest.raises(SystemExit, match="2"):
        main(["--device", "cuda"])


@pytest.mark.skipif(os.environ.get("CLIPP2_TEST_CUDA") != "1", reason="LSF CUDA opt-in required")
def test_cuda_bounded_ward_refresh_matches_every_logical_cut(monkeypatch):
    assert torch.cuda.is_available(), "CLIPP2_TEST_CUDA=1 requires qualified LSF CUDA"
    monkeypatch.setattr(ward, "_WARD_HEAP_MIN_ENTRIES", 1)
    monkeypatch.setattr(ward, "_WARD_HEAP_ACTIVE_MULTIPLIER", 1)
    for dtype in (torch.float32, torch.float64):
        for kind in ("random", "ties", "duplicates", "zero_weights"):
            phi, curvature = _fixture(37, 3, dtype, kind, "cuda")
            expected = _logical_matrix_reference(phi, curvature)
            actual = ward.hessian_weighted_ward_label_sets_torch(
                phi, curvature, K_grid=range(1, 38), initial_pairwise_work_elements=51,
                refresh_work_elements=41,
            )
            for count in expected:
                np.testing.assert_array_equal(actual[count], expected[count])
