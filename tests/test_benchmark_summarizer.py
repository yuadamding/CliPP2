from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _load_summarizer():
    path = Path(__file__).parents[1] / "sim_v2_results" / "summarize_guided_admm.py"
    spec = importlib.util.spec_from_file_location("benchmark_summarizer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUMMARIZER = _load_summarizer()


def _design_row(depth: int, samples: int, lambda_mut: int) -> dict[str, float | int]:
    return {
        "depth": depth,
        "name_k": 3,
        "nominal_purity": 0.6,
        "amp_rate": 0.2,
        "S": samples,
        "lambda_mut": lambda_mut,
        "name_m": lambda_mut,
        "replicate": 0,
    }


def test_factorial_validation_accepts_dynamic_factor_levels() -> None:
    design = {
        f"case_{depth}_{samples}_{lambda_mut}": _design_row(depth, samples, lambda_mut)
        for depth in (50, 1000)
        for samples in (1, 20)
        for lambda_mut in (300, 4000)
    }

    SUMMARIZER._validate_factorial_design(
        design,
        expected_tumors=8,
        expected_per_s=4,
    )

    design.pop(next(iter(design)))
    with pytest.raises(ValueError, match="complete factorial"):
        SUMMARIZER._validate_factorial_design(
            design,
            expected_tumors=7,
            expected_per_s=None,
        )


def test_parse_tumor_id_preserves_mutation_rate_setting() -> None:
    parsed = SUMMARIZER._parse_tumor_id("50_7_0.6_0.2_S20_Lm4000_M3972_rep1")

    assert parsed["lambda_mut"] == 4000
    assert parsed["depth"] == 50
    assert parsed["S"] == 20


def test_multiplicity_calls_must_equal_major_or_minor_copy_number() -> None:
    major = np.array([[2.0, 3.0], [1.0, 2.0]])
    minor = np.array([[1.0, 0.0], [1.0, 1.0]])
    valid = np.array([[2.0, 0.0], [1.0, 1.0]])
    SUMMARIZER._validate_multiplicity_calls(
        valid,
        major,
        minor,
        source="test",
    )

    invalid = valid.copy()
    invalid[1, 1] = 1.5
    with pytest.raises(ValueError, match="invalid multiplicity call"):
        SUMMARIZER._validate_multiplicity_calls(
            invalid,
            major,
            minor,
            source="test",
        )


def _selected_contract_rows() -> tuple[pd.Series, pd.Series]:
    shared = {
        "n_clusters": 3,
        "tol": 1e-4,
        "inner_solver": "admm_complete_graph",
        "graph_name": "complete_adaptive_test",
        "admm_iterations": 12,
        "converged": True,
        "stationarity_certified": True,
        "selection_eligible": True,
        "raw_kkt_eligible": True,
        "lambda_path_prespecified": False,
        "selection_method": "online_partition_guided_admm",
        "selection_score_name": "partition_icl",
        "lambda_search_mode": "partition_guided_admm",
        "lambda_source": "online_partition_guide_kkt",
        "initialization_mode": "ward_cem_partition_icl_kkt",
        "initializer_selection_score": "partition_icl",
        "initializer_source": "hessian_ward_cem_K3",
    }
    summary = pd.Series(
        {
            **shared,
            "tumor_id": "tumor",
            "selected_lambda": 2.5,
            "selected_kkt_residual": 2e-4,
            "selected_candidate_pool_source": "raw_fused_lambda_path",
            "evaluate_all_candidates": False,
        }
    )
    selected = pd.Series(
        {
            **shared,
            "tumor_id": "tumor",
            "lambda": 2.5,
            "bic_n_clusters": 3,
            "raw_kkt_residual": 2e-4,
            "candidate_pool_source": "raw_fused_lambda_path",
            "num_edges": 6,
        }
    )
    return summary, selected


def _schema_v1_contract_rows() -> tuple[pd.Series, pd.Series]:
    summary, selected = _selected_contract_rows()
    exact = {
        "inner_solver": "quotient_workset_complete_graph",
        "inner_backend": "quotient_workset_complete_graph",
        "admm_iterations": 0,
        "backend_iterations": 19,
        "exactness_provenance_version": 1,
        "estimator_role": "raw_fused_lambda_path",
        "objective_faithful": True,
        "objective_spec_hash": "objective-v1",
        "original_graph_hash": "graph-v1",
        "certificate_problem_hash": "problem-v1",
        "certificate_scope": "full_original_graph",
        "certificate_gradient_scope": "observed_objective",
        "full_kkt_certified": True,
        "full_kkt_certificate_status": "certified",
        "full_kkt_tolerance": 5e-4,
    }
    for key, value in exact.items():
        summary[key] = value
        selected[key] = value
    return summary, selected


def test_selected_solver_contract_cross_checks_and_enforces_requested_path() -> None:
    summary, selected = _selected_contract_rows()

    contract = SUMMARIZER._selected_solver_contract(
        summary_row=summary,
        selected_row=selected,
        tumor_id="tumor",
        num_mutations=4,
    )

    assert contract["strict_solver_contract"] is True
    assert contract["complete_edge_count"] is True
    assert contract["postselection_truth_only"] is True

    mismatched = selected.copy()
    mismatched["lambda"] = 3.0
    with pytest.raises(ValueError, match="selected lambda"):
        SUMMARIZER._selected_solver_contract(
            summary_row=summary,
            selected_row=mismatched,
            tumor_id="tumor",
            num_mutations=4,
        )

    wrong_edges = selected.copy()
    wrong_edges["num_edges"] = 5
    incomplete = SUMMARIZER._selected_solver_contract(
        summary_row=summary,
        selected_row=wrong_edges,
        tumor_id="tumor",
        num_mutations=4,
    )
    assert incomplete["complete_edge_count"] is False
    assert incomplete["strict_solver_contract"] is False

    truth_leaking = summary.copy()
    truth_leaking["evaluate_all_candidates"] = True
    leaking = SUMMARIZER._selected_solver_contract(
        summary_row=truth_leaking,
        selected_row=selected,
        tumor_id="tumor",
        num_mutations=4,
    )
    assert leaking["postselection_truth_only"] is False
    assert leaking["strict_solver_contract"] is False


def test_selected_solver_contract_accepts_schema_v1_quotient_without_admm_gate() -> (
    None
):
    summary, selected = _schema_v1_contract_rows()

    contract = SUMMARIZER._selected_solver_contract(
        summary_row=summary,
        selected_row=selected,
        tumor_id="tumor",
        num_mutations=4,
    )

    assert contract["strict_solver_contract"] is True
    assert contract["backend_neutral_exact_provenance"] is True
    assert contract["explicit_exactness_provenance"] is True
    assert contract["admm_solver"] is False
    assert contract["admm_used"] is False
    assert contract["backend_iterations"] == 19

    rejected = selected.copy()
    rejected["full_kkt_certificate_status"] = "not_certified"
    rejected_summary = summary.copy()
    rejected_summary["full_kkt_certificate_status"] = "not_certified"
    invalid = SUMMARIZER._selected_solver_contract(
        summary_row=rejected_summary,
        selected_row=rejected,
        tumor_id="tumor",
        num_mutations=4,
    )
    assert invalid["certificate_status_accepted"] is False
    assert invalid["strict_solver_contract"] is False

    mismatched = selected.copy()
    mismatched["original_graph_hash"] = "other-graph"
    with pytest.raises(
        ValueError, match="exactness field 'original_graph_hash' mismatch"
    ):
        SUMMARIZER._selected_solver_contract(
            summary_row=summary,
            selected_row=mismatched,
            tumor_id="tumor",
            num_mutations=4,
        )


def test_cluster_centers_must_match_mutation_labels_sizes_and_values() -> None:
    data = SimpleNamespace(tumor_id="tumor", region_ids=["tumor_sample0"])
    mutations = pd.DataFrame(
        {
            "tumor_id": ["tumor"] * 3,
            "cluster_label": [1, 1, 2],
            "cluster_size": [2, 2, 1],
            "summary_phi_tumor_region0": [0.2, 0.2, 0.8],
        }
    )
    centers = pd.DataFrame(
        {
            "tumor_id": ["tumor", "tumor"],
            "cluster_label": [1, 2],
            "cluster_size": [2, 1],
            "phi_tumor_region0": [0.2, 0.8],
        }
    )
    SUMMARIZER._validate_center_consistency(
        centers=centers,
        mutations=mutations,
        data=data,
    )

    invalid = centers.copy()
    invalid.loc[0, "cluster_size"] = 1
    with pytest.raises(ValueError, match="cluster sizes"):
        SUMMARIZER._validate_center_consistency(
            centers=invalid,
            mutations=mutations,
            data=data,
        )


def test_pooled_ccf_rmse_is_not_tumor_macro_rmse() -> None:
    group = pd.DataFrame(
        {
            "ccf_cells": [1, 9],
            "summary_ccf_sse": [1.0, 0.0],
        }
    )

    assert SUMMARIZER._pooled_rmse(group, "summary_ccf_sse") == pytest.approx(
        np.sqrt(0.1)
    )
    assert np.mean([1.0, 0.0]) == pytest.approx(0.5)


def test_simulation_evaluation_metric_is_cross_checked_when_required() -> None:
    row = pd.Series({"ARI": 0.75})
    SUMMARIZER._assert_evaluation_metric(
        row=row,
        key="ARI",
        expected=0.75,
        tumor_id="tumor",
        required=True,
    )
    with pytest.raises(ValueError, match="simulation_eval ARI"):
        SUMMARIZER._assert_evaluation_metric(
            row=row,
            key="ARI",
            expected=0.5,
            tumor_id="tumor",
            required=True,
        )
    with pytest.raises(ValueError, match="is missing"):
        SUMMARIZER._assert_evaluation_metric(
            row=row,
            key="cp_rmse",
            expected=0.1,
            tumor_id="tumor",
            required=True,
        )


def test_oracle_input_noise_matches_binomial_fisher_information() -> None:
    data = SimpleNamespace(
        total_counts=np.asarray([[100.0], [100.0]]),
        scaling=np.asarray([[0.5], [0.5]]),
        count_observed=np.ones((2, 1), dtype=bool),
    )
    truth = SimpleNamespace(
        truth_clusters=np.asarray([0, 1]),
        truth_phi=np.asarray([[0.2], [0.6]]),
        truth_multiplicity=np.ones((2, 1)),
    )

    metrics = SUMMARIZER._oracle_input_noise_metrics(data, truth)

    se_left = 0.06
    se_right = np.sqrt(0.0084)
    expected_z = 0.4 / np.sqrt(se_left**2 + se_right**2)
    assert metrics["oracle_fisher_se_mean"] == pytest.approx(0.5 * (se_left + se_right))
    assert metrics["oracle_zero_information_cluster_regions"] == 0
    assert metrics["oracle_weakest_pair_mahalanobis"] == pytest.approx(expected_z)
    assert metrics["oracle_weakest_pair_max_region_z"] == pytest.approx(expected_z)

    data.total_counts[1, 0] = 0.0
    zero_information = SUMMARIZER._oracle_input_noise_metrics(data, truth)
    assert zero_information["oracle_zero_information_cluster_regions"] == 1
    assert zero_information["oracle_weakest_pair_mahalanobis"] == 0.0
