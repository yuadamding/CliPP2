"""Evaluation ID-alignment tests (audit §5.10, P1.3).

Verifies that load_simulation_truth uses mutation_id-based row alignment
when the truth file contains a mutation_id column, rather than relying on
positional alignment (which silently misaligns when orderings differ).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from CliPP2.metrics.evaluation import (
    SimulationTruth,
    _reindex_by_mutation_id,
    evaluate_fit_against_simulation,
    load_simulation_truth,
)
from CliPP2.io.data import TumorData


# ── _reindex_by_mutation_id unit tests ───────────────────────────────────────


def test_reindex_positional_passthrough_if_no_ids() -> None:
    values = np.array([10, 20, 30], dtype=int)
    data_ids = ["a", "b", "c"]
    result = _reindex_by_mutation_id(values, None, data_ids, "test", "tumor1")
    np.testing.assert_array_equal(result, values)


def test_reindex_positional_raises_on_shape_mismatch() -> None:
    values = np.array([10, 20], dtype=int)
    data_ids = ["a", "b", "c"]
    with pytest.raises(ValueError, match="shape mismatch"):
        _reindex_by_mutation_id(values, None, data_ids, "test", "tumor1")


def test_reindex_by_id_reorders_to_match_data_order() -> None:
    # truth is in order [c, a, b]; data wants [a, b, c]
    truth_ids = ["c", "a", "b"]
    values = np.array([30, 10, 20], dtype=int)
    data_ids = ["a", "b", "c"]
    result = _reindex_by_mutation_id(values, truth_ids, data_ids, "test", "tumor1")
    np.testing.assert_array_equal(result, np.array([10, 20, 30]))


def test_reindex_by_id_same_order_is_identity() -> None:
    truth_ids = ["a", "b", "c"]
    values = np.array([10, 20, 30], dtype=int)
    data_ids = ["a", "b", "c"]
    result = _reindex_by_mutation_id(values, truth_ids, data_ids, "test", "tumor1")
    np.testing.assert_array_equal(result, values)


def test_reindex_by_id_raises_on_missing_mutation() -> None:
    truth_ids = ["a", "b"]
    values = np.array([10, 20], dtype=int)
    data_ids = ["a", "b", "c"]  # c not in truth
    with pytest.raises(ValueError, match="not found"):
        _reindex_by_mutation_id(values, truth_ids, data_ids, "test", "tumor1")


def test_reindex_by_id_raises_on_duplicate_truth_ids() -> None:
    values = np.array([10, 20, 30], dtype=int)
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        _reindex_by_mutation_id(
            values, ["a", "a", "b"], ["a", "b", "c"], "test", "tumor1"
        )


def test_reindex_by_id_raises_on_extra_truth_ids() -> None:
    values = np.array([10, 20, 30], dtype=int)
    with pytest.raises(ValueError, match="not found in data"):
        _reindex_by_mutation_id(values, ["a", "b", "x"], ["a", "b"], "test", "tumor1")


# ── load_simulation_truth integration tests ──────────────────────────────────


def _minimal_tumor_data(mutation_ids: list[str]) -> TumorData:
    M = len(mutation_ids)
    ones = np.ones((M, 1), dtype=np.float64)
    return TumorData(
        tumor_id="tumor1",
        mutation_ids=mutation_ids,
        region_ids=["s1"],
        alt_counts=ones * 5,
        total_counts=ones * 10,
        purity=ones * 0.8,
        major_cn=ones * 2,
        minor_cn=ones,
        normal_cn=ones * 2,
        has_cna=np.ones((M, 1), dtype=bool),
        scaling=ones * 0.2857,
        phi_upper=ones,
        phi_init=ones * 0.5,
        init_major_mask=np.ones((M, 1), dtype=bool),
    )


def _write_truth_with_ids(
    tumor_dir: Path,
    mutation_ids: list[str],
    cluster_ids: list[int],
    ccf_values: list[float],
) -> None:
    """Write truth.txt and truth_cp.txt with mutation_id column."""
    tumor_dir.mkdir(parents=True, exist_ok=True)
    with open(tumor_dir / "truth.txt", "w") as f:
        f.write("mutation_id\tcluster_id\n")
        for mid, cid in zip(mutation_ids, cluster_ids):
            f.write(f"{mid}\t{cid}\n")
    with open(tumor_dir / "truth_cp.txt", "w") as f:
        f.write("mutation_id\tccf\n")
        for mid, ccf in zip(mutation_ids, ccf_values):
            f.write(f"{mid}\t{ccf}\n")


def test_load_simulation_truth_identity_when_ids_match(tmp_path: Path) -> None:
    data = _minimal_tumor_data(["a", "b", "c"])
    _write_truth_with_ids(
        tmp_path / "tumor1",
        mutation_ids=["a", "b", "c"],
        cluster_ids=[0, 1, 1],
        ccf_values=[0.9, 0.4, 0.4],
    )
    truth = load_simulation_truth(data, tmp_path)
    np.testing.assert_array_equal(truth.truth_clusters, [0, 1, 1])
    np.testing.assert_allclose(truth.truth_phi[:, 0], [0.9, 0.4, 0.4], rtol=1e-5)


def test_load_simulation_truth_reorders_by_mutation_id(tmp_path: Path) -> None:
    # Data orders mutations [a, b, c]; truth file stores them in order [c, b, a]
    data = _minimal_tumor_data(["a", "b", "c"])
    _write_truth_with_ids(
        tmp_path / "tumor1",
        mutation_ids=["c", "b", "a"],
        cluster_ids=[2, 1, 0],
        ccf_values=[0.3, 0.5, 0.9],
    )
    truth = load_simulation_truth(data, tmp_path)

    # After alignment: a→0,0.9; b→1,0.5; c→2,0.3
    np.testing.assert_array_equal(truth.truth_clusters, [0, 1, 2])
    np.testing.assert_allclose(truth.truth_phi[:, 0], [0.9, 0.5, 0.3], rtol=1e-5)


def test_load_simulation_truth_positional_fallback_without_ids(tmp_path: Path) -> None:
    # truth.txt has NO mutation_id column → positional alignment
    tumor_dir = tmp_path / "tumor1"
    tumor_dir.mkdir(parents=True)
    (tumor_dir / "truth.txt").write_text("cluster_id\n0\n1\n1\n")
    (tumor_dir / "truth_cp.txt").write_text("ccf\n0.9\n0.4\n0.4\n")

    data = _minimal_tumor_data(["a", "b", "c"])
    truth = load_simulation_truth(data, tmp_path)
    np.testing.assert_array_equal(truth.truth_clusters, [0, 1, 1])
    np.testing.assert_allclose(truth.truth_phi[:, 0], [0.9, 0.4, 0.4], rtol=1e-5)


def test_load_simulation_truth_raises_when_mutation_missing(tmp_path: Path) -> None:
    data = _minimal_tumor_data(["a", "b", "c"])
    _write_truth_with_ids(
        tmp_path / "tumor1",
        mutation_ids=["a", "b"],  # missing "c"
        cluster_ids=[0, 1],
        ccf_values=[0.9, 0.4],
    )
    with pytest.raises(ValueError, match="not found"):
        load_simulation_truth(data, tmp_path)


def test_evaluation_reports_raw_summary_and_bic_refit_cp_rmse_separately() -> None:
    data = _minimal_tumor_data(["a", "b"])
    truth = SimulationTruth(
        truth_clusters=np.asarray([0, 1], dtype=np.int64),
        truth_phi=np.asarray([[0.0], [1.0]], dtype=np.float64),
        truth_multiplicity=None,
    )
    fit = SimpleNamespace(
        phi=np.asarray([[0.1], [0.9]], dtype=np.float64),
        phi_clustered=np.asarray([[0.2], [0.8]], dtype=np.float64),
        cluster_labels=np.asarray([0, 1], dtype=np.int64),
        n_clusters=2,
    )

    evaluation = evaluate_fit_against_simulation(
        fit=fit,
        data=data,
        simulation_truth=truth,
        bic_refit_phi=np.asarray([[0.05], [0.95]], dtype=np.float64),
        bic_partition_labels=np.asarray([0, 1], dtype=np.int64),
    )

    assert evaluation.raw_cp_rmse == pytest.approx(0.1)
    assert evaluation.summary_cp_rmse == pytest.approx(0.2)
    assert evaluation.cp_rmse == pytest.approx(evaluation.summary_cp_rmse)
    assert evaluation.bic_refit_cp_rmse == pytest.approx(0.05)
