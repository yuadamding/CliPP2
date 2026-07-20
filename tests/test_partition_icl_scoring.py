from __future__ import annotations

from math import lgamma

import numpy as np
import pandas as pd
import pytest

from CliPP2.core.bic import (
    cluster_sizes_from_labels,
    compute_classic_bic,
    compute_partition_icl,
    compute_unlabeled_dirichlet_partition_log_evidence,
)
from CliPP2.core.fusion.types import PairwiseFusionGraph
from CliPP2.core.model import FitOptions
from CliPP2.io.data import TumorData, compute_phi_init_from_counts
from CliPP2.model_selection.adaptive import (
    _adaptive_score_column,
    _best_candidate_rows_by_lambda,
)
from CliPP2.model_selection.config import (
    DEFAULT_SELECTION_SCORE,
    PARTITION_ICL_DIRICHLET_ALPHA,
)
from CliPP2.model_selection.partitions import _likelihood_partition_k_grid
from CliPP2.model_selection.scoring import (
    _add_bic_selection_eligible,
    _bic_selection_eligible_mask,
    _lambda_applicable_mask,
    _normalize_selection_score_name,
    _row_bic_selection_eligible,
    _row_lambda_applicable,
    _selection_score_value,
)
from CliPP2.runners.model_selection import select_model


def _tumor(num_mutations: int = 4) -> TumorData:
    alt = np.asarray(
        [[5.0, 7.0], [6.0, 8.0], [20.0, 18.0], [21.0, 19.0]][:num_mutations],
        dtype=np.float64,
    )
    total = np.full_like(alt, 50.0)
    purity = np.full_like(alt, 0.7)
    major = np.ones_like(alt)
    minor = np.ones_like(alt)
    normal = np.full_like(alt, 2.0)
    scaling = purity / (purity * (major + minor) + (1.0 - purity) * normal)
    phi_upper = np.ones_like(alt)
    phi_init, init_major_mask = compute_phi_init_from_counts(
        alt_counts=alt,
        total_counts=total,
        scaling=scaling,
        major_cn=major,
        minor_cn=minor,
        phi_upper=phi_upper,
    )
    return TumorData(
        tumor_id="partition-icl",
        mutation_ids=[f"m{i}" for i in range(num_mutations)],
        region_ids=["r0", "r1"],
        alt_counts=alt,
        total_counts=total,
        purity=purity,
        major_cn=major,
        minor_cn=minor,
        normal_cn=normal,
        has_cna=np.zeros_like(alt, dtype=bool),
        scaling=scaling,
        phi_upper=phi_upper,
        phi_init=phi_init,
        init_major_mask=init_major_mask,
        count_observed=np.ones_like(alt, dtype=bool),
    )


def _line_graph() -> PairwiseFusionGraph:
    return PairwiseFusionGraph(
        edge_u=np.asarray([0, 1, 2], dtype=np.int32),
        edge_v=np.asarray([1, 2, 3], dtype=np.int32),
        edge_w=np.ones(3, dtype=np.float64),
        name="line",
        degree_bound=2,
    )


def test_unlabeled_dirichlet_partition_code_matches_closed_form() -> None:
    sizes = np.asarray([2, 2], dtype=np.int64)
    observed = compute_unlabeled_dirichlet_partition_log_evidence(sizes, alpha=1.0)
    expected = (
        lgamma(2.0) - lgamma(6.0) + 2.0 * (lgamma(3.0) - lgamma(1.0)) + lgamma(3.0)
    )
    assert observed == pytest.approx(expected)


def test_partition_icl_is_label_invariant_and_adds_assignment_code() -> None:
    data = _tumor()
    first = cluster_sizes_from_labels(np.asarray([10, 10, 3, 3]))
    relabeled = cluster_sizes_from_labels(np.asarray([0, 0, 99, 99]))

    assert np.array_equal(first, relabeled)
    classic = compute_classic_bic(-100.0, 2, data)
    icl_first = compute_partition_icl(-100.0, first, data, alpha=1.0)
    icl_relabeled = compute_partition_icl(-100.0, relabeled, data, alpha=1.0)
    assert icl_first == pytest.approx(icl_relabeled)
    assert icl_first > classic


def test_one_cluster_has_zero_assignment_code() -> None:
    data = _tumor()
    sizes = np.asarray([data.num_mutations], dtype=np.int64)
    assert compute_unlabeled_dirichlet_partition_log_evidence(
        sizes, alpha=1.0
    ) == pytest.approx(0.0)
    assert compute_partition_icl(-10.0, sizes, data) == pytest.approx(
        compute_classic_bic(-10.0, 1, data)
    )


def test_selection_boolean_columns_parse_strictly_and_fail_closed() -> None:
    values = [True, "true", 1, False, "false", 0, np.nan, "nan", "unknown", 2]
    frame = pd.DataFrame({"bic_selection_eligible": values})

    assert _bic_selection_eligible_mask(frame).tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert not _row_bic_selection_eligible(
        pd.Series({"bic_selection_eligible": np.nan})
    )
    assert not _row_bic_selection_eligible(
        pd.Series({"bic_selection_eligible": "false"})
    )


def test_derived_eligibility_and_lambda_applicability_fail_closed() -> None:
    eligibility = pd.DataFrame(
        {
            "raw_kkt_eligible": ["true", "false", np.nan, "unknown"],
            "bic_refit_finite_candidate_found": ["true"] * 4,
            "classic_bic": [1.0] * 4,
        }
    )
    applicability = pd.DataFrame(
        {
            "lambda_applicable": ["true", "false", np.nan, "unknown"],
            "lambda": [1.0] * 4,
        }
    )

    assert _add_bic_selection_eligible(eligibility)[
        "bic_selection_eligible"
    ].tolist() == [True, False, False, False]
    assert _lambda_applicable_mask(applicability).tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert _row_lambda_applicable(pd.Series({"lambda": 1.0}))
    assert not _row_lambda_applicable(
        pd.Series({"lambda": 1.0, "lambda_applicable": np.nan})
    )


def test_selection_score_modes_preserve_all_diagnostics() -> None:
    data = _tumor()
    kwargs = dict(
        loglik=-50.0,
        num_clusters=2,
        data=data,
        bic_df_scale=2.0,
        bic_cluster_penalty=3.0,
        cluster_sizes=np.asarray([2, 2]),
    )

    bic_selected, classic, extended, partition_icl = _selection_score_value(
        **kwargs,
        selection_score="bic",
    )
    extended_selected, classic_again, extended_again, icl_again = (
        _selection_score_value(
            **kwargs,
            selection_score="ebic",
        )
    )
    icl_selected, _, _, _ = _selection_score_value(
        **kwargs,
        selection_score="partition_icl",
    )

    assert bic_selected == pytest.approx(classic)
    assert extended_selected == pytest.approx(extended)
    assert icl_selected == pytest.approx(partition_icl)
    assert classic_again == pytest.approx(classic)
    assert extended_again == pytest.approx(extended)
    assert icl_again == pytest.approx(partition_icl)
    assert _normalize_selection_score_name("ebic") == "extended_bic"
    assert DEFAULT_SELECTION_SCORE == "partition_icl"
    assert PARTITION_ICL_DIRICHLET_ALPHA == 1.0


def test_partition_icl_requires_cluster_sizes_but_classic_bic_does_not() -> None:
    data = _tumor()
    _, _, _, icl = _selection_score_value(
        loglik=-10.0,
        num_clusters=1,
        data=data,
        bic_df_scale=1.0,
        bic_cluster_penalty=0.0,
        selection_score="bic",
    )
    assert np.isnan(icl)
    with pytest.raises(ValueError, match="requires candidate cluster sizes"):
        _selection_score_value(
            loglik=-10.0,
            num_clusters=1,
            data=data,
            bic_df_scale=1.0,
            bic_cluster_penalty=0.0,
            selection_score="partition_icl",
        )


def test_adaptive_per_lambda_ranking_uses_active_score_column() -> None:
    frame = pd.DataFrame(
        [
            {
                "lambda": 1.0,
                "selection_step": 0,
                "raw_kkt_eligible": True,
                "bic_refit_converged": True,
                "selection_eligible": True,
                "converged": True,
                "classic_bic": 10.0,
                "extended_bic": 30.0,
                "partition_icl": 100.0,
                "bic": 100.0,
                "penalized_objective": 2.0,
            },
            {
                "lambda": 1.0,
                "selection_step": 1,
                "raw_kkt_eligible": True,
                "bic_refit_converged": True,
                "selection_eligible": True,
                "converged": True,
                "classic_bic": 20.0,
                "extended_bic": 5.0,
                "partition_icl": 50.0,
                "bic": 50.0,
                "penalized_objective": 2.0,
            },
        ]
    )

    classic = _best_candidate_rows_by_lambda(frame, normalized_score="bic")
    extended = _best_candidate_rows_by_lambda(frame, normalized_score="extended_bic")
    icl = _best_candidate_rows_by_lambda(frame, normalized_score="partition_icl")
    assert int(classic.iloc[0].selection_step) == 0
    assert int(extended.iloc[0].selection_step) == 1
    assert int(icl.iloc[0].selection_step) == 1
    assert _adaptive_score_column("partition_icl") == "partition_icl"


def test_partition_k_grid_is_dense_through_fifteen() -> None:
    grid = _likelihood_partition_k_grid(50)
    assert grid[:15] == list(range(1, 16))


def test_partition_icl_drives_end_to_end_selected_score() -> None:
    result = select_model(
        data=_tumor(),
        simulation_root=None,
        lambda_grid=[0.1],
        lambda_grid_mode="adaptive_bic",
        fit_options=FitOptions(
            lambda_value=0.0,
            graph=_line_graph(),
            outer_max_iter=2,
            inner_max_iter=16,
            tol=1e-4,
            device="cpu",
            dtype="float64",
        ),
        bic_df_scale=1.0,
        bic_cluster_penalty=0.0,
        selection_score="partition_icl",
        use_warm_starts=True,
        evaluate_all_candidates=False,
        finalize_selected_fit=False,
    )

    selected = result.search_df.loc[
        result.search_df["is_selected_best_row"].astype(bool)
    ].iloc[0]
    assert selected["selection_score_name"] == "partition_icl"
    assert result.selection_metric_value == pytest.approx(
        float(selected["partition_icl"])
    )
    assert result.selected_artifact.bic == pytest.approx(
        float(selected["partition_icl"])
    )
    assert result.selected_artifact.classic_bic == pytest.approx(
        float(selected["classic_bic"])
    )
    assert result.selected_artifact.partition_icl == pytest.approx(
        float(selected["partition_icl"])
    )
