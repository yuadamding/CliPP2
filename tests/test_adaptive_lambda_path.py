from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import CliPP2.runners.model_selection as model_selection
from CliPP2.core.model import FitOptions
from CliPP2.core.fusion.refit import partition_constrained_observed_refit
from CliPP2.core.fusion.torch_backend import (
    objective_value_torch,
    resolve_runtime,
    to_torch_tumor_data,
)
from CliPP2.io.data import TumorData, compute_phi_init_from_counts
from CliPP2.model_selection.adaptive import (
    _best_candidate_rows_by_lambda,
    _full_fusion_box_residual_with_dual_balls,
)
from CliPP2.model_selection.config import (
    ADAPTIVE_FIRST_PASS_INNER_MAX_ITER,
    ADAPTIVE_FIRST_PASS_OUTER_MAX_ITER,
)
from CliPP2.model_selection.partitions import _partition_is_coarsening
from CliPP2.model_selection.scoring import (
    _add_bic_selection_eligible,
    _is_bic_selection_eligible,
)
from CliPP2.runners.model_selection import (
    ADAPTIVE_PATH_MAX_CANDIDATES,
    ADAPTIVE_PATH_MAX_ROUNDS,
    ADAPTIVE_PATH_REFINE_PER_ROUND,
    ADAPTIVE_PATH_TRANSITION_PROBE_MAX_CANDIDATES,
    BICSelectionResult,
    SimulationDiagnostics,
    _adaptive_first_pass_options,
    _adaptive_interval_proposal_records,
    _adaptive_transition_probe_records,
    _lambda_warm_start_distance,
    _partition_signature,
    _sorted_unique_lambdas,
    select_model,
)
from CliPP2.runners.selection import LambdaBracket


def _tiny_tumor() -> TumorData:
    alt = np.asarray(
        [
            [5.0, 8.0],
            [11.0, 13.0],
            [22.0, 18.0],
        ],
        dtype=np.float32,
    )
    total = np.full_like(alt, 50.0)
    purity = np.full_like(alt, 0.7)
    major_cn = np.ones_like(alt)
    minor_cn = np.ones_like(alt)
    normal_cn = np.full_like(alt, 2.0)
    has_cna = np.zeros_like(alt, dtype=bool)
    total_cn = major_cn + minor_cn
    scaling = purity / (purity * total_cn + (1.0 - purity) * normal_cn)
    max_prob_scale = np.maximum(scaling * major_cn, scaling * minor_cn)
    phi_upper = np.minimum(
        1.0, (1.0 - 1e-6) / np.clip(max_prob_scale, 1e-6, None)
    ).astype(np.float32)
    phi_init, init_major_mask = compute_phi_init_from_counts(
        alt_counts=alt,
        total_counts=total,
        scaling=scaling,
        major_cn=major_cn,
        minor_cn=minor_cn,
        phi_upper=phi_upper,
    )
    return TumorData(
        tumor_id="tiny_adaptive",
        mutation_ids=["m0", "m1", "m2"],
        region_ids=["s0", "s1"],
        alt_counts=alt,
        total_counts=total,
        purity=purity,
        major_cn=major_cn,
        minor_cn=minor_cn,
        normal_cn=normal_cn,
        has_cna=has_cna,
        scaling=scaling.astype(np.float32),
        phi_upper=phi_upper,
        phi_init=phi_init,
        init_major_mask=init_major_mask,
    )


def _two_mutation_one_sample_tumor() -> TumorData:
    alt = np.asarray([[5.0], [45.0]], dtype=np.float32)
    total = np.asarray([[50.0], [50.0]], dtype=np.float32)
    purity = np.ones_like(alt, dtype=np.float32)
    major_cn = np.full_like(alt, 2.0, dtype=np.float32)
    minor_cn = np.zeros_like(alt, dtype=np.float32)
    normal_cn = np.full_like(alt, 2.0, dtype=np.float32)
    has_cna = np.zeros_like(alt, dtype=bool)
    total_cn = major_cn + minor_cn
    scaling = purity / (purity * total_cn + (1.0 - purity) * normal_cn)
    max_prob_scale = np.maximum(scaling * major_cn, scaling * minor_cn)
    phi_upper = np.minimum(
        1.0, (1.0 - 1e-6) / np.clip(max_prob_scale, 1e-6, None)
    ).astype(np.float32)
    phi_init, init_major_mask = compute_phi_init_from_counts(
        alt_counts=alt,
        total_counts=total,
        scaling=scaling,
        major_cn=major_cn,
        minor_cn=minor_cn,
        phi_upper=phi_upper,
    )
    return TumorData(
        tumor_id="two_mutation_refit",
        mutation_ids=["m0", "m1"],
        region_ids=["s0"],
        alt_counts=alt,
        total_counts=total,
        purity=purity,
        major_cn=major_cn,
        minor_cn=minor_cn,
        normal_cn=normal_cn,
        has_cna=has_cna,
        scaling=scaling.astype(np.float32),
        phi_upper=phi_upper,
        phi_init=phi_init,
        init_major_mask=init_major_mask,
    )


def test_partition_refit_is_not_raw_cluster_average() -> None:
    data = _two_mutation_one_sample_tumor()
    raw_phi = np.asarray([[0.2], [0.2]], dtype=np.float64)

    runtime = resolve_runtime("cpu", dtype="float64")
    torch_data = to_torch_tumor_data(data, runtime)
    empty_edges = torch.empty(0, dtype=torch.long, device=runtime.device)
    empty_weights = torch.empty(0, dtype=runtime.dtype, device=runtime.device)
    cluster_mean_fit_loss, _, _, _ = objective_value_torch(
        torch_data,
        torch.as_tensor(raw_phi, dtype=runtime.dtype, device=runtime.device),
        edge_u=empty_edges,
        edge_v=empty_edges,
        edge_w=empty_weights,
        lambda_value=0.0,
        major_prior=0.5,
        eps=1e-6,
    )

    refit = partition_constrained_observed_refit(
        data,
        np.asarray([0, 0], dtype=np.int64),
        major_prior=0.5,
        eps=1e-6,
        tol=1e-6,
        max_iter=80,
        hint_phi=raw_phi,
    )

    assert refit.loglik_source == "partition_constrained_observed_mle"
    assert refit.converged
    assert refit.fit_loss <= cluster_mean_fit_loss + 1e-8
    assert np.isclose(float(refit.cluster_centers[0, 0]), 0.5, atol=1e-3)
    assert not np.isclose(
        float(refit.cluster_centers[0, 0]), float(np.mean(raw_phi[:, 0])), atol=1e-2
    )


def test_adaptive_bic_path_produces_bracketed_candidates() -> None:
    result = select_model(
        data=_tiny_tumor(),
        simulation_root=None,
        lambda_grid=None,
        lambda_grid_mode="adaptive_bic",
        fit_options=FitOptions(
            lambda_value=0.0,
            outer_max_iter=2,
            inner_max_iter=12,
            tol=1e-4,
            device="cpu",
            dtype="float64",
            summary_tol=1e-4,
            bic_partition_tol=1e-4,
        ),
        bic_df_scale=1.0,
        bic_cluster_penalty=0.0,
        selection_score="bic",
        use_warm_starts=True,
        evaluate_all_candidates=False,
        finalize_selected_fit=True,
    )

    assert isinstance(result, BICSelectionResult)
    assert isinstance(result.simulation, SimulationDiagnostics)
    assert result.lambda_search_mode == "adaptive_bic"
    assert result.lambda_bracket_eq is not None and result.lambda_bracket_eq > 0.0
    assert (
        result.lambda_bracket_full is not None
        and result.lambda_bracket_full >= result.lambda_bracket_eq
    )
    assert result.search_df["lambda_search_mode"].eq("adaptive_bic").all()
    assert result.search_df["lambda_bracket_eq"].notna().all()
    assert result.selected_lambda_left is not None
    assert result.selected_lambda_right is not None
    assert (
        result.selected_lambda_left
        <= result.selected_lambda_representative
        <= result.selected_lambda_right
    )
    selected = result.search_df.loc[
        result.search_df["is_selected_best_row"].astype(bool)
    ].iloc[0]
    assert bool(selected["selection_eligible"])
    assert selected["bic_loglik_source"] == "partition_constrained_observed_mle"
    assert "bic_refit_cache_hit" in result.search_df.columns
    assert np.isclose(
        float(selected["classic_bic"]),
        -2.0 * float(selected["bic_loglik"])
        + float(selected["bic_df"]) * np.log(float(selected["bic_n_eff"])),
    )
    assert np.isclose(
        float(selected["classic_bic_active_df"]),
        -2.0 * float(selected["bic_loglik"])
        + float(selected["bic_active_df"]) * np.log(float(selected["bic_n_eff"])),
    )
    assert np.isclose(
        float(selected["classic_bic_active_df_depth_n"]),
        -2.0 * float(selected["bic_loglik"])
        + float(selected["bic_active_df"]) * np.log(float(selected["bic_depth_n_eff"])),
    )
    assert float(selected["bic_partition_tol"]) == 1e-4
    evaluated_lambdas = result.search_df["lambda"].to_numpy(dtype=float)
    assert np.any(np.isclose(evaluated_lambdas, 0.0))
    positive_lambdas = evaluated_lambdas[evaluated_lambdas > 0.0]
    assert positive_lambdas.size > 0
    assert positive_lambdas.min() <= max(
        float(result.lambda_bracket_eq) / 128.0, 1e-6
    ) * (1.0 + 1e-8)
    assert "bic_penalty" in result.search_df.columns
    assert "delta_bic_vs_one_cluster" in result.search_df.columns
    assert np.isclose(
        float(selected["bic_penalty"]),
        float(selected["bic_df"]) * np.log(float(selected["bic_n_eff"])),
    )


def test_full_fusion_residual_respects_iteration_budget() -> None:
    grad = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    phi = torch.full((2, 1), 0.5, dtype=torch.float64)
    lower = torch.zeros_like(phi)
    upper = torch.ones_like(phi)
    edge_u = torch.tensor([0], dtype=torch.long)
    edge_v = torch.tensor([1], dtype=torch.long)
    edge_w = torch.tensor([1.0], dtype=torch.float64)

    r1, info1 = _full_fusion_box_residual_with_dual_balls(
        grad_smooth=grad,
        phi=phi,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=10.0,
        atol=1e-12,
        max_iter=1,
        return_info=True,
    )
    r80, info80 = _full_fusion_box_residual_with_dual_balls(
        grad_smooth=grad,
        phi=phi,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=10.0,
        atol=1e-12,
        max_iter=80,
        return_info=True,
    )

    assert info1.iterations == 1
    assert info80.iterations > 1
    assert r80 <= r1 + 1e-10


def test_full_fusion_residual_uses_precomputed_degree_bound(monkeypatch) -> None:
    grad = torch.tensor([[1.0], [-1.0]], dtype=torch.float64)
    phi = torch.full((2, 1), 0.5, dtype=torch.float64)
    lower = torch.zeros_like(phi)
    upper = torch.ones_like(phi)
    edge_u = torch.tensor([0], dtype=torch.long)
    edge_v = torch.tensor([1], dtype=torch.long)
    edge_w = torch.tensor([1.0], dtype=torch.float64)

    def fail_bincount(*args, **kwargs):
        raise AssertionError("full-fusion KKT probe should use supplied degree_bound")

    monkeypatch.setattr(torch, "bincount", fail_bincount)

    residual, info = _full_fusion_box_residual_with_dual_balls(
        grad_smooth=grad,
        phi=phi,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=10.0,
        atol=1e-12,
        max_iter=4,
        degree_bound=1,
        return_info=True,
    )

    assert info.iterations > 0
    assert residual >= 0.0


def test_adaptive_first_pass_uses_bic_budget_floor() -> None:
    options = _adaptive_first_pass_options(
        FitOptions(lambda_value=0.0, outer_max_iter=2, inner_max_iter=12)
    )

    assert options.outer_max_iter == ADAPTIVE_FIRST_PASS_OUTER_MAX_ITER
    assert options.inner_max_iter == ADAPTIVE_FIRST_PASS_INNER_MAX_ITER


def test_adaptive_bic_search_budget_is_not_smoke_sized() -> None:
    assert ADAPTIVE_PATH_MAX_CANDIDATES >= 40
    assert ADAPTIVE_PATH_MAX_ROUNDS >= 4
    assert ADAPTIVE_PATH_REFINE_PER_ROUND >= 5
    assert ADAPTIVE_PATH_TRANSITION_PROBE_MAX_CANDIDATES >= 3


def test_model_selection_hashes_static_candidate_metadata_once(monkeypatch) -> None:
    calls = {"edge": 0, "data": 0}
    original_edge_hash = model_selection._edge_list_hash
    original_data_hash = model_selection._input_data_hash

    def counting_edge_hash(*args, **kwargs):
        calls["edge"] += 1
        return original_edge_hash(*args, **kwargs)

    def counting_data_hash(*args, **kwargs):
        calls["data"] += 1
        return original_data_hash(*args, **kwargs)

    monkeypatch.setattr(model_selection, "_edge_list_hash", counting_edge_hash)
    monkeypatch.setattr(model_selection, "_input_data_hash", counting_data_hash)

    result = model_selection.select_model(
        data=_tiny_tumor(),
        simulation_root=None,
        lambda_grid=[0.0, 0.2],
        lambda_grid_mode="adaptive_bic",
        fit_options=FitOptions(
            lambda_value=0.0,
            outer_max_iter=2,
            inner_max_iter=16,
            tol=1e-4,
            device="cpu",
            dtype="float64",
            summary_tol=1e-4,
            bic_partition_tol=1e-4,
        ),
        bic_df_scale=1.0,
        bic_cluster_penalty=0.0,
        selection_score="bic",
        use_warm_starts=True,
        evaluate_all_candidates=False,
        finalize_selected_fit=True,
    )

    assert calls == {"edge": 1, "data": 1}
    assert result.search_df["edge_list_hash"].nunique() == 1
    assert result.search_df["input_data_hash"].nunique() == 1


def test_model_selection_keeps_prepared_starts_tensor_resident(monkeypatch) -> None:
    observed_start_types: list[dict[str, bool]] = []
    original_evaluate_candidate = model_selection._evaluate_candidate

    def recording_evaluate_candidate(**kwargs):
        scalar_starts = kwargs.get("scalar_well_starts") or []
        observed_start_types.append(
            {
                "phi_start": torch.is_tensor(kwargs.get("phi_start")),
                "exact_pilot": torch.is_tensor(kwargs.get("exact_pilot")),
                "pooled_start": torch.is_tensor(kwargs.get("pooled_start")),
                "scalar_well_starts": all(
                    torch.is_tensor(start) for start in scalar_starts
                ),
            }
        )
        return original_evaluate_candidate(**kwargs)

    monkeypatch.setattr(
        model_selection, "_evaluate_candidate", recording_evaluate_candidate
    )

    result = model_selection.select_model(
        data=_tiny_tumor(),
        simulation_root=None,
        lambda_grid=[0.0, 0.2],
        lambda_grid_mode="adaptive_bic",
        fit_options=FitOptions(
            lambda_value=0.0,
            outer_max_iter=2,
            inner_max_iter=16,
            tol=1e-4,
            device="cpu",
            dtype="float64",
            summary_tol=1e-4,
            bic_partition_tol=1e-4,
        ),
        bic_df_scale=1.0,
        bic_cluster_penalty=0.0,
        selection_score="bic",
        use_warm_starts=True,
        evaluate_all_candidates=False,
        finalize_selected_fit=True,
    )

    assert not result.search_df.empty
    assert observed_start_types
    assert all(all(record.values()) for record in observed_start_types)


def test_adaptive_transition_probe_targets_sparse_lambda_eq_gaps() -> None:
    bracket = LambdaBracket(
        lambda_min=100.0 / 128.0,
        lambda_eq=100.0,
        lambda_full=400.0,
        anchors=[0.0, 100.0 / 2.0, 100.0, 200.0, 400.0],
        diagnostics={},
    )

    records = _adaptive_transition_probe_records(
        bracket,
        bracket.anchors,
        max_new=3,
    )

    assert [record.reason for record in records] == [
        "lambda_eq_upper_transition_probe",
        "lambda_eq_lower_transition_probe",
        "lambda_eq_high_transition_probe",
    ]
    assert np.allclose(
        [record.lambda_value for record in records],
        [np.sqrt(100.0 * 200.0), np.sqrt(50.0 * 100.0), np.sqrt(200.0 * 400.0)],
    )
    assert all(record.priority_key[0] == -1 for record in records)


def test_adaptive_transition_probe_skips_already_evaluated_midpoints() -> None:
    bracket = LambdaBracket(
        lambda_min=100.0 / 128.0,
        lambda_eq=100.0,
        lambda_full=400.0,
        anchors=[0.0, 50.0, 100.0, 200.0, 400.0],
        diagnostics={},
    )

    records = _adaptive_transition_probe_records(
        bracket,
        [*bracket.anchors, np.sqrt(100.0 * 200.0)],
        max_new=2,
    )

    assert [record.reason for record in records] == [
        "lambda_eq_lower_transition_probe",
        "lambda_eq_high_transition_probe",
    ]


def test_lambda_sort_keeps_zero_diagnostic_endpoint() -> None:
    assert _sorted_unique_lambdas([1.0, 0.0, -1.0, np.inf, 0.0]) == [0.0, 1.0]


def test_lambda_warm_start_distance_prefers_positive_log_neighbors() -> None:
    assert _lambda_warm_start_distance(
        source_lambda=10.0, target_lambda=100.0
    ) == pytest.approx(np.log(10.0))
    assert _lambda_warm_start_distance(source_lambda=0.0, target_lambda=0.0) == 0.0
    assert _lambda_warm_start_distance(source_lambda=2.0, target_lambda=0.0) == 2.0
    assert np.isinf(_lambda_warm_start_distance(source_lambda=0.0, target_lambda=2.0))


def test_bic_selection_requires_raw_kkt_and_refit_convergence() -> None:
    assert _is_bic_selection_eligible(
        raw_kkt_eligible=True,
        bic_refit_finite_candidate_found=True,
        classic_bic=1.0,
    )
    assert _is_bic_selection_eligible(
        raw_kkt_eligible=True,
        bic_refit_converged=True,
        classic_bic=1.0,
    )
    assert not _is_bic_selection_eligible(
        raw_kkt_eligible=False,
        bic_refit_converged=True,
        classic_bic=1.0,
    )
    assert not _is_bic_selection_eligible(
        raw_kkt_eligible=True,
        bic_refit_converged=False,
        classic_bic=1.0,
    )
    assert not _is_bic_selection_eligible(
        raw_kkt_eligible=True,
        bic_refit_converged=True,
        classic_bic=float("nan"),
    )


def test_bic_selection_eligible_column_requires_refit_and_finite_bic() -> None:
    search_df = pd.DataFrame(
        [
            {
                "raw_kkt_eligible": True,
                "bic_refit_converged": True,
                "classic_bic": 1.0,
            },
            {
                "raw_kkt_eligible": True,
                "bic_refit_converged": False,
                "classic_bic": 0.0,
            },
            {
                "raw_kkt_eligible": False,
                "bic_refit_converged": True,
                "classic_bic": 0.0,
            },
            {
                "raw_kkt_eligible": True,
                "bic_refit_converged": True,
                "classic_bic": np.nan,
            },
        ]
    )

    annotated = _add_bic_selection_eligible(search_df)

    assert annotated["bic_selection_eligible"].tolist() == [True, False, False, False]


def test_bic_selection_eligible_prefers_finite_refit_candidate_column() -> None:
    search_df = pd.DataFrame(
        [
            {
                "raw_kkt_eligible": True,
                "bic_refit_finite_candidate_found": False,
                "bic_refit_converged": True,
                "classic_bic": 1.0,
            },
            {
                "raw_kkt_eligible": True,
                "bic_refit_finite_candidate_found": True,
                "bic_refit_converged": False,
                "classic_bic": 1.0,
            },
        ]
    )

    annotated = _add_bic_selection_eligible(search_df)

    assert annotated["bic_selection_eligible"].tolist() == [False, True]


def test_duplicate_lambda_keeps_bic_preferred_candidate() -> None:
    search_df = pd.DataFrame(
        [
            {
                "lambda": 1.0,
                "selection_step": 0,
                "selection_eligible": True,
                "raw_kkt_eligible": True,
                "bic_refit_converged": True,
                "converged": True,
                "classic_bic": 100.0,
                "penalized_objective": 0.0,
            },
            {
                "lambda": 1.0,
                "selection_step": 1,
                "selection_eligible": True,
                "raw_kkt_eligible": True,
                "bic_refit_converged": True,
                "converged": True,
                "classic_bic": 50.0,
                "penalized_objective": 10.0,
            },
            {
                "lambda": 1.0,
                "selection_step": 2,
                "selection_eligible": True,
                "raw_kkt_eligible": True,
                "bic_refit_converged": False,
                "converged": True,
                "classic_bic": 1.0,
                "penalized_objective": -10.0,
            },
        ]
    )

    best = _best_candidate_rows_by_lambda(search_df)

    assert best.shape[0] == 1
    assert int(best.iloc[0]["selection_step"]) == 1


def _interval_row(
    *,
    lambda_value: float,
    selection_step: int,
    partition_signature: str,
    n_clusters: int,
    classic_bic: float = 1000.0,
    profile_penalty: float = 1.0,
    penalized_objective: float = 100.0,
    selection_eligible: bool = True,
) -> dict[str, float | int | str | bool]:
    return {
        "lambda": float(lambda_value),
        "selection_step": int(selection_step),
        "_candidate_id": int(selection_step),
        "selection_eligible": bool(selection_eligible),
        "bic_selection_eligible": bool(selection_eligible),
        "bic_refit_converged": True,  # explicit: absent = False per audit
        "converged": True,
        "penalized_objective": float(penalized_objective),
        "profile_penalty": float(profile_penalty),
        "fixed_objective_kkt_residual": 0.2,
        "classic_bic": float(classic_bic),
        "partition_signature": str(partition_signature),
        "n_clusters": int(n_clusters),
    }


def test_partition_signature_uses_canonical_blocks() -> None:
    labels = np.asarray([5, 2, 5, 7], dtype=np.int64)
    relabeled = np.asarray([10, 0, 10, 3], dtype=np.int64)
    different = np.asarray([0, 0, 1, 2], dtype=np.int64)

    assert _partition_signature(labels) == _partition_signature(relabeled)
    assert _partition_signature(labels) != _partition_signature(different)
    assert _partition_is_coarsening(
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        np.asarray([0, 0, 1, 1], dtype=np.int64),
    )
    assert not _partition_is_coarsening(
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([0, 1, 0, 1], dtype=np.int64),
    )


def test_adaptive_interval_refines_zero_left_partition_change() -> None:
    search_df = pd.DataFrame(
        [
            _interval_row(
                lambda_value=0.0,
                selection_step=0,
                partition_signature=_partition_signature(
                    np.asarray([0, 1, 2], dtype=np.int64)
                ),
                n_clusters=3,
            ),
            _interval_row(
                lambda_value=1.0,
                selection_step=1,
                partition_signature=_partition_signature(
                    np.asarray([0, 0, 1], dtype=np.int64)
                ),
                n_clusters=2,
            ),
        ]
    )

    proposals = [
        proposal.lambda_value
        for proposal in _adaptive_interval_proposal_records(
            search_df, normalized_score="bic", tol=1e-4, max_new=1
        )
    ]

    assert np.allclose(proposals, [0.5])


def test_adaptive_interval_refines_wide_same_partition_plateau() -> None:
    signature = _partition_signature(np.asarray([0, 0, 1], dtype=np.int64))
    search_df = pd.DataFrame(
        [
            _interval_row(
                lambda_value=1.0,
                selection_step=0,
                partition_signature=signature,
                n_clusters=2,
            ),
            _interval_row(
                lambda_value=100.0,
                selection_step=1,
                partition_signature=signature,
                n_clusters=2,
            ),
        ]
    )

    proposals = [
        proposal.lambda_value
        for proposal in _adaptive_interval_proposal_records(
            search_df, normalized_score="bic", tol=1e-4, max_new=1
        )
    ]

    assert np.allclose(proposals, [10.0])


def test_adaptive_interval_prioritizes_nonnested_partition_change() -> None:
    left_labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    right_labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    fused_labels = np.asarray([0, 0, 0, 0], dtype=np.int64)
    search_df = pd.DataFrame(
        [
            _interval_row(
                lambda_value=1.0,
                selection_step=0,
                partition_signature=_partition_signature(left_labels),
                n_clusters=2,
            ),
            _interval_row(
                lambda_value=10.0,
                selection_step=1,
                partition_signature=_partition_signature(right_labels),
                n_clusters=2,
                penalized_objective=95.0,
            ),
            _interval_row(
                lambda_value=100.0,
                selection_step=2,
                partition_signature=_partition_signature(fused_labels),
                n_clusters=1,
                penalized_objective=90.0,
            ),
        ]
    )

    records = _adaptive_interval_proposal_records(
        search_df,
        normalized_score="bic",
        tol=1e-4,
        max_new=1,
        partition_labels_by_candidate_id={
            0: left_labels,
            1: right_labels,
            2: fused_labels,
        },
    )

    assert len(records) == 1
    assert np.isclose(records[0].lambda_value, np.sqrt(10.0))
    assert records[0].reason == "nonnested_partition_change"
    assert records[0].nonagglomerative_or_numerically_inconsistent


def test_adaptive_interval_kkt_risk_uses_selection_eligible_not_converged() -> None:
    search_df = pd.DataFrame(
        [
            {
                "lambda": 1.0,
                "selection_step": 0,
                "selection_eligible": True,
                "converged": True,
                "penalized_objective": 100.0,
                "profile_penalty": 1.0,
                "fixed_objective_kkt_residual": 0.2,
                "classic_bic": 1000.0,
                "partition_signature": "1:a",
            },
            {
                "lambda": 10.0,
                "selection_step": 1,
                "selection_eligible": True,
                "converged": True,
                "penalized_objective": 90.0,
                "profile_penalty": 1.0,
                "fixed_objective_kkt_residual": 0.2,
                "classic_bic": 1000.0,
                "partition_signature": "1:a",
            },
            {
                "lambda": 100.0,
                "selection_step": 2,
                "selection_eligible": False,
                "converged": True,
                "penalized_objective": 80.0,
                "profile_penalty": 1.0,
                "fixed_objective_kkt_residual": 0.2,
                "classic_bic": 100.0,
                "partition_signature": "1:a",
            },
        ]
    )

    proposals = [
        proposal.lambda_value
        for proposal in _adaptive_interval_proposal_records(
            search_df, normalized_score="bic", tol=1e-4, max_new=1
        )
    ]

    assert np.allclose(proposals, [np.sqrt(10.0 * 100.0)])


@pytest.mark.parametrize(
    ("selection_score", "lambda_grid_mode"),
    [
        ("oracle_ari", "adaptive_bic"),
        ("partition_refit_ebic", "adaptive_bic"),
        ("classic_bic", "adaptive_bic"),
        ("bic", "adaptive_cv_stability"),
        ("bic", "adaptive_ebic_path"),
    ],
)
def test_non_bic_selection_modes_are_rejected(
    selection_score: str, lambda_grid_mode: str
) -> None:
    with pytest.raises(ValueError):
        select_model(
            data=_tiny_tumor(),
            simulation_root=None,
            lambda_grid=None,
            lambda_grid_mode=lambda_grid_mode,
            fit_options=FitOptions(
                lambda_value=0.0,
                outer_max_iter=1,
                inner_max_iter=8,
                tol=1e-4,
                device="cpu",
                dtype="float64",
                summary_tol=1e-4,
                bic_partition_tol=1e-4,
            ),
            bic_df_scale=1.0,
            bic_cluster_penalty=0.0,
            selection_score=selection_score,
            use_warm_starts=True,
            evaluate_all_candidates=False,
            finalize_selected_fit=True,
        )
