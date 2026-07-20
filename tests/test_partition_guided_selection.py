from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import CliPP2.runners.model_selection as runner
from CliPP2.core.fusion.graph import build_complete_uniform_graph
from CliPP2.core.fusion.partition_starts import PartitionCandidate
from CliPP2.core.model import FitOptions
from CliPP2.core.fusion.types import (
    DenseEdgeCertificate,
    DenseWarmState,
    QuotientFailureProvenance,
    SolverState,
    TorchRuntime,
)
from CliPP2.io.data import TumorData, compute_phi_init_from_counts
from CliPP2.model_selection.scoring import _positive_exact_fusion_selection_mask


def _two_group_tumor() -> TumorData:
    alt = np.asarray(
        [
            [5.0, 7.0],
            [6.0, 6.0],
            [5.0, 8.0],
            [20.0, 17.0],
            [21.0, 18.0],
            [20.0, 19.0],
        ],
        dtype=np.float64,
    )
    total = np.full_like(alt, 50.0)
    purity = np.full_like(alt, 0.7)
    major_cn = np.ones_like(alt)
    minor_cn = np.ones_like(alt)
    normal_cn = np.full_like(alt, 2.0)
    scaling = purity / (purity * (major_cn + minor_cn) + (1.0 - purity) * normal_cn)
    phi_upper = np.ones_like(alt)
    phi_init, init_major_mask = compute_phi_init_from_counts(
        alt_counts=alt,
        total_counts=total,
        scaling=scaling,
        major_cn=major_cn,
        minor_cn=minor_cn,
        phi_upper=phi_upper,
    )
    return TumorData(
        tumor_id="partition-guided-selection-test",
        mutation_ids=[f"m{i}" for i in range(int(alt.shape[0]))],
        region_ids=["r0", "r1"],
        alt_counts=alt,
        total_counts=total,
        purity=purity,
        major_cn=major_cn,
        minor_cn=minor_cn,
        normal_cn=normal_cn,
        has_cna=np.zeros_like(alt, dtype=bool),
        scaling=scaling,
        phi_upper=phi_upper,
        phi_init=phi_init,
        init_major_mask=init_major_mask,
        count_observed=np.ones_like(alt, dtype=bool),
    )


def test_partition_candidate_records_effective_runtime() -> None:
    data = _two_group_tumor()
    graph = build_complete_uniform_graph(data.num_mutations)
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    centers = np.asarray([[0.2, 0.25], [0.8, 0.75]], dtype=np.float64)
    candidate = PartitionCandidate(
        labels=labels,
        K=2,
        source="test",
        theta=centers,
        phi_start=centers[labels],
        fit_loss=10.0,
        bic=20.0,
    )
    runtime = TorchRuntime(
        device=torch.device("cpu"),
        device_name="cpu",
        dtype=torch.float32,
    )

    fit, _, row, _ = runner._evaluate_partition_candidate(
        data=data,
        fit_options=FitOptions(
            lambda_value=0.0,
            graph=graph,
            device="cuda",
            dtype="float64",
        ),
        candidate=candidate,
        candidate_rank=1,
        bic_df_scale=1.0,
        bic_cluster_penalty=0.0,
        simulation_truth=None,
        evaluate_candidate=False,
        selection_method="test",
        profile_name="test",
        selection_step=0,
        selection_score="partition_icl",
        static_metadata=runner._candidate_static_metadata(
            data, graph, candidate.phi_start
        ),
        runtime=runtime,
    )

    assert (fit.device, fit.dtype) == ("cpu", "float32")
    assert (row["device"], row["dtype"]) == ("cpu", "float32")


def test_partition_guided_mode_selects_only_positive_complete_graph_admm(
    monkeypatch,
) -> None:
    def _partition_candidate_must_not_be_evaluated(*args, **kwargs):
        raise AssertionError("A direct partition proposal entered final selection.")

    monkeypatch.setattr(
        runner,
        "_evaluate_partition_candidate",
        _partition_candidate_must_not_be_evaluated,
    )
    result = runner.select_model(
        data=_two_group_tumor(),
        simulation_root=None,
        lambda_grid=None,
        lambda_grid_mode="partition_guided_admm",
        fit_options=FitOptions(
            lambda_value=0.0,
            outer_max_iter=8,
            inner_max_iter=30,
            tol=1e-4,
            device="cpu",
            dtype="float64",
            summary_tol=1e-4,
            bic_partition_tol=1e-4,
        ),
        bic_df_scale=1.0,
        bic_cluster_penalty=0.0,
        selection_score="partition_icl",
        use_warm_starts=True,
        evaluate_all_candidates=False,
        finalize_selected_fit=True,
    )

    frame = result.search_df
    assert not frame.empty
    assert frame["candidate_pool_source"].eq("raw_fused_lambda_path").all()
    assert frame["lambda"].gt(0.0).all()
    assert frame["lambda_path_prespecified"].eq(False).all()  # noqa: E712
    assert frame["inner_solver"].eq("admm_complete_graph").all()
    assert frame["admm_iterations"].gt(0).all()
    assert frame["persistent_solver_state_device"].eq("cpu").all()
    assert int(frame["num_edges"].iloc[0]) == 6 * 5 // 2
    selected = frame.loc[frame["is_selected_best_row"]].iloc[0]
    assert bool(selected["raw_kkt_eligible"])
    assert bool(selected["bic_selection_eligible"])
    assert int(selected["exactness_provenance_version"]) == 1
    assert bool(selected["objective_faithful"])
    assert bool(selected["full_kkt_certified"])
    assert selected["certificate_scope"] == "full_original_graph"
    assert selected["certificate_gradient_scope"] == "observed_objective"
    assert selected["objective_spec_hash"]
    assert selected["original_graph_hash"]
    assert selected["certificate_problem_hash"]
    assert (
        selected["partition_signature"] == selected["initializer_partition_signature"]
    )
    assert result.best_fit.lambda_value > 0.0
    assert result.best_fit.inner_solver == "admm_complete_graph"
    assert result.best_fit.admm_iterations > 0
    assert result.lambda_search_mode == "partition_guided_admm"
    # Reproducing the initializer is not a stopping rule: the online controller
    # must observe a genuine neighboring partition/score basin.
    assert frame["lambda"].nunique() > 1
    assert frame["partition_signature"].nunique() > 1
    assert frame["search_phase"].eq("initial").any()
    assert (
        result.adaptive_search_stop_reason != "online_lambda_guide_partition_certified"
    )
    assert (
        frame["fusion_graph_source"]
        .eq("partition_guide_likelihood_noise_degree_regularized")
        .all()
    )
    assert np.isfinite(frame["fusion_graph_likelihood_noise_tau"]).all()
    assert frame["fusion_graph_likelihood_noise_tau"].gt(0.0).all()
    assert np.allclose(
        frame["fusion_graph_likelihood_noise_divisor"],
        5.0**1.05,
    )
    assert frame["fusion_graph_likelihood_noise_degree_exponent"].eq(1.05).all()
    assert (
        frame["fusion_graph_pilot_matrix_hash"].iloc[0]
        == frame["initializer_matrix_hash"].iloc[0]
    )
    assert (
        frame["scalar_likelihood_pilot_matrix_hash"].iloc[0]
        != frame["initializer_matrix_hash"].iloc[0]
    )


def test_persistent_solver_state_offload_preserves_values_and_shared_storage() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    phi = torch.arange(6, dtype=torch.float64, device=device).reshape(3, 2)
    dual = torch.arange(12, dtype=torch.float64, device=device).reshape(6, 2)
    state = SolverState(
        phi=phi.detach(),
        dual=dual.detach(),
        previous_lambda=3.5,
        warm_state=DenseWarmState(
            phi=phi.detach(),
            dual=dual.detach(),
            previous_lambda=3.5,
            graph_hash="shared-storage-test",
        ),
        certificate=DenseEdgeCertificate(
            dual=dual.detach(),
            graph_hash="shared-storage-test",
            gradient_scope="observed_objective",
        ),
        quotient_failure=QuotientFailureProvenance(
            lambda_value=3.5,
            graph_hash="shared-storage-test",
            reason="test-failure",
        ),
    )

    observed = runner._offload_solver_state_to_cpu(state)

    assert observed is not None
    assert observed.phi.device.type == "cpu"
    assert observed.dual is not None and observed.dual.device.type == "cpu"
    assert not hasattr(observed, "curvature")
    assert not hasattr(observed, "split")
    assert observed.previous_lambda == pytest.approx(3.5)
    assert observed.quotient_failure == state.quotient_failure
    assert isinstance(observed.warm_state, DenseWarmState)
    assert isinstance(observed.certificate, DenseEdgeCertificate)
    assert observed.phi is observed.warm_state.phi
    assert (
        observed.phi.untyped_storage().data_ptr()
        == observed.warm_state.phi.untyped_storage().data_ptr()
    )
    assert observed.dual is not None
    assert observed.warm_state.dual is not None
    assert observed.dual is observed.warm_state.dual is observed.certificate.dual
    assert (
        observed.dual.untyped_storage().data_ptr()
        == observed.warm_state.dual.untyped_storage().data_ptr()
        == observed.certificate.dual.untyped_storage().data_ptr()
    )
    np.testing.assert_array_equal(observed.phi.numpy(), phi.detach().cpu().numpy())
    np.testing.assert_array_equal(observed.dual.numpy(), dual.detach().cpu().numpy())


@pytest.mark.parametrize("selection_score", ["bic", "extended_bic"])
def test_guided_mode_rejects_scores_not_used_by_online_controller(
    monkeypatch, selection_score: str
) -> None:
    def _initializer_must_not_run(*args, **kwargs):
        raise AssertionError("Validation occurred after initializer generation.")

    monkeypatch.setattr(
        runner,
        "generate_partition_initializer_pool",
        _initializer_must_not_run,
    )
    with pytest.raises(ValueError, match="requires selection_score='partition_icl'"):
        runner.select_model(
            data=_two_group_tumor(),
            simulation_root=None,
            lambda_grid=None,
            lambda_grid_mode="partition_guided_admm",
            fit_options=FitOptions(lambda_value=0.0, device="cpu", dtype="float64"),
            bic_df_scale=1.0,
            bic_cluster_penalty=0.0,
            selection_score=selection_score,
            use_warm_starts=True,
            evaluate_all_candidates=False,
        )


def test_guided_mode_rejects_prespecified_lambda_path_before_initialization(
    monkeypatch,
) -> None:
    def _initializer_must_not_run(*args, **kwargs):
        raise AssertionError("Validation occurred after initializer generation.")

    monkeypatch.setattr(
        runner,
        "generate_partition_initializer_pool",
        _initializer_must_not_run,
    )
    with pytest.raises(ValueError, match="does not accept a prespecified lambda grid"):
        runner.select_model(
            data=_two_group_tumor(),
            simulation_root=None,
            lambda_grid=[1.0, 2.0],
            lambda_grid_mode="partition_guided_admm",
            fit_options=FitOptions(lambda_value=0.0, device="cpu", dtype="float64"),
            bic_df_scale=1.0,
            bic_cluster_penalty=0.0,
            selection_score="partition_icl",
            use_warm_starts=True,
            evaluate_all_candidates=False,
        )


def test_strict_final_mask_uses_exact_provenance_with_legacy_dense_fallback() -> None:
    legacy_dense = {
        "raw_kkt_eligible": True,
        "bic_refit_finite_candidate_found": True,
        "classic_bic": 1.0,
        "bic": 1.0,
        "candidate_pool_source": "raw_fused_lambda_path",
        "lambda": 2.0,
        "inner_solver": "admm_complete_graph",
        "admm_iterations": 4,
    }
    exact_quotient = {
        **legacy_dense,
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
        "fixed_objective_kkt_residual": 2e-4,
        "full_kkt_tolerance": 5e-4,
        "inner_backend": "quotient_workset_complete_graph",
        "backend_iterations": 0,
        "inner_solver": "quotient_workset_complete_graph",
        "admm_iterations": 0,
    }
    rows = [
        dict(legacy_dense),
        dict(exact_quotient),
        {
            **legacy_dense,
            "lambda": 0.0,
            "inner_solver": "closed_form_projection",
            "admm_iterations": 0,
        },
        {
            **legacy_dense,
            "candidate_pool_source": "likelihood_partition",
            "lambda": np.nan,
        },
        {**legacy_dense, "inner_solver": "pdhg"},
        {**legacy_dense, "admm_iterations": 0},
        {
            **exact_quotient,
            # Explicit provenance is authoritative: the legacy ADMM fields
            # cannot rescue a failed versioned certificate.
            "inner_solver": "admm_complete_graph",
            "admm_iterations": 4,
            "full_kkt_certified": False,
        },
        {**exact_quotient, "certificate_scope": "workset_only"},
        {**exact_quotient, "certificate_gradient_scope": "mm_surrogate"},
        {**exact_quotient, "objective_faithful": False},
        {**exact_quotient, "fixed_objective_kkt_residual": 6e-4},
        {**exact_quotient, "objective_spec_hash": ""},
        {
            **exact_quotient,
            "full_kkt_certificate_status": "not_certified",
        },
    ]
    mask = _positive_exact_fusion_selection_mask(pd.DataFrame(rows))
    assert mask.tolist() == [True, True] + [False] * 11
