from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from CliPP2.core.model import FitOptions, fit_fixed_objective
from CliPP2.runners import pipeline

from test_adaptive_lambda_path import _tiny_tumor


def _selection_artifact() -> SimpleNamespace:
    return SimpleNamespace(
        bic=12.0,
        classic_bic=10.0,
        extended_bic=12.0,
        selection_score_name="bic",
    )


def test_summary_ari_reports_selected_lambda_ari(monkeypatch, tmp_path) -> None:
    data = _tiny_tumor()
    fit = fit_fixed_objective(
        data,
        FitOptions(
            lambda_value=0.1,
            outer_max_iter=2,
            inner_max_iter=16,
            tol=1e-4,
            device="cpu",
            dtype="float64",
        ),
    )

    selection_result = SimpleNamespace(
        best_fit=fit,
        selected_artifact=_selection_artifact(),
        best_evaluation=None,
        search_df=pd.DataFrame({"lambda": [0.1, 0.5]}),
        profile_name="manual",
        selection_method="lambda_path_grid",
        lambda_search_mode="explicit_grid",
        selected_lambda_representative=0.1,
        selected_lambda_left=0.1,
        selected_lambda_right=0.1,
        selected_lambda_interval_log10_width=0.0,
        lambda_bracket_min=None,
        lambda_bracket_eq=None,
        lambda_bracket_full=None,
        adaptive_refinement_rounds_completed=0,
        selection_used_convergence_fallback=False,
        num_candidates=2,
        num_converged_candidates=1,
        selection_metric_value=12.0,
        selection_lambda_min=0.1,
        selection_lambda_max=0.1,
        selection_lambda_count=1,
        selection_hits_lower_boundary=True,
        selection_hits_upper_boundary=False,
        selection_boundary_unresolved=False,
        selection_optimum_resolved=True,
        selected_ari=0.2,
        best_ari=0.9,
        best_converged_ari=0.9,
        best_converged_lambda_min=0.5,
        best_converged_lambda_max=0.5,
        best_converged_lambda_count=1,
        ari_optimal_lambda_min=0.5,
        ari_optimal_lambda_max=0.5,
        ari_optimal_lambda_count=1,
        ari_hits_lower_boundary=False,
        ari_hits_upper_boundary=True,
        ari_boundary_unresolved=False,
        ari_optimum_resolved=True,
        adaptive_search_rounds_completed=0,
        adaptive_search_stop_reason="not_applicable",
        bic_df_scale=8.0,
        bic_cluster_penalty=4.0,
    )

    monkeypatch.setattr(pipeline, "load_tumor_tsv", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        pipeline, "select_model", lambda *args, **kwargs: selection_result
    )

    summary, _ = pipeline.process_one_file_bundle(
        file_path=tmp_path / "fake.tsv",
        outdir=tmp_path,
        simulation_root=None,
        write_outputs=False,
    )

    assert summary["selected_ari"] == 0.2
    assert summary["best_ari"] == 0.9
    assert summary["ARI"] == 0.2
    assert "selected_validation_loglik_mean" not in summary
    assert "cv_stability_replicates" not in summary


def test_summary_ari_uses_final_evaluation_when_selection_ari_missing(
    monkeypatch, tmp_path
) -> None:
    data = _tiny_tumor()
    fit = fit_fixed_objective(
        data,
        FitOptions(
            lambda_value=0.1,
            outer_max_iter=2,
            inner_max_iter=16,
            tol=1e-4,
            device="cpu",
            dtype="float64",
        ),
    )

    selection_result = SimpleNamespace(
        best_fit=fit,
        selected_artifact=_selection_artifact(),
        best_evaluation=None,
        search_df=pd.DataFrame({"lambda": [0.1]}),
        profile_name="manual",
        selection_method="lambda_path_grid",
        lambda_search_mode="explicit_grid",
        selected_lambda_representative=0.1,
        selected_lambda_left=0.1,
        selected_lambda_right=0.1,
        selected_lambda_interval_log10_width=0.0,
        lambda_bracket_min=None,
        lambda_bracket_eq=None,
        lambda_bracket_full=None,
        adaptive_refinement_rounds_completed=0,
        selection_used_convergence_fallback=False,
        num_candidates=1,
        num_converged_candidates=1,
        selection_metric_value=12.0,
        selection_lambda_min=0.1,
        selection_lambda_max=0.1,
        selection_lambda_count=1,
        selection_hits_lower_boundary=True,
        selection_hits_upper_boundary=True,
        selection_boundary_unresolved=False,
        selection_optimum_resolved=True,
        selected_ari=None,
        best_ari=None,
        best_converged_ari=None,
        best_converged_lambda_min=None,
        best_converged_lambda_max=None,
        best_converged_lambda_count=0,
        ari_optimal_lambda_min=None,
        ari_optimal_lambda_max=None,
        ari_optimal_lambda_count=0,
        ari_hits_lower_boundary=False,
        ari_hits_upper_boundary=False,
        ari_boundary_unresolved=False,
        ari_optimum_resolved=True,
        adaptive_search_rounds_completed=0,
        adaptive_search_stop_reason="not_applicable",
        bic_df_scale=8.0,
        bic_cluster_penalty=4.0,
    )
    final_eval = SimpleNamespace(
        ari=0.77,
        cp_rmse=0.1,
        multiplicity_f1=0.2,
        estimated_clonal_fraction=0.3,
        true_clonal_fraction=0.4,
        clonal_fraction_error=0.1,
        n_eval_mutations=3,
        n_filtered_mutations=0,
    )

    monkeypatch.setattr(pipeline, "load_tumor_tsv", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        pipeline, "select_model", lambda *args, **kwargs: selection_result
    )
    monkeypatch.setattr(
        pipeline, "evaluate_fit_against_simulation", lambda *args, **kwargs: final_eval
    )
    (tmp_path / data.tumor_id).mkdir()

    summary, _ = pipeline.process_one_file_bundle(
        file_path=tmp_path / "fake.tsv",
        outdir=tmp_path,
        simulation_root=tmp_path,
        write_outputs=False,
        finalize_selected_fit=True,
    )

    assert summary["selected_ari"] == 0.77
    assert summary["ARI"] == 0.77


def test_evaluate_all_candidates_is_decoupled_from_output_writing(
    monkeypatch, tmp_path
) -> None:
    data = _tiny_tumor()
    fit = fit_fixed_objective(
        data,
        FitOptions(
            lambda_value=0.1,
            outer_max_iter=2,
            inner_max_iter=16,
            tol=1e-4,
            device="cpu",
            dtype="float64",
        ),
    )
    captured: dict[str, object] = {}
    selection_result = SimpleNamespace(
        best_fit=fit,
        selected_artifact=_selection_artifact(),
        best_evaluation=None,
        search_df=pd.DataFrame({"lambda": [0.1], "is_selected_best_row": [True]}),
        profile_name="manual",
        selection_method="lambda_path_grid",
        lambda_search_mode="explicit_grid",
        selected_lambda_representative=0.1,
        selected_lambda_left=0.1,
        selected_lambda_right=0.1,
        selected_lambda_interval_log10_width=0.0,
        lambda_bracket_min=None,
        lambda_bracket_eq=None,
        lambda_bracket_full=None,
        adaptive_refinement_rounds_completed=0,
        selection_used_convergence_fallback=False,
        num_candidates=1,
        num_converged_candidates=1,
        selection_metric_value=12.0,
        selection_lambda_min=0.1,
        selection_lambda_max=0.1,
        selection_lambda_count=1,
        selection_hits_lower_boundary=True,
        selection_hits_upper_boundary=True,
        selection_boundary_unresolved=False,
        selection_optimum_resolved=True,
        selected_ari=0.5,
        best_ari=0.5,
        best_converged_ari=0.5,
        best_converged_lambda_min=0.1,
        best_converged_lambda_max=0.1,
        best_converged_lambda_count=1,
        ari_optimal_lambda_min=0.1,
        ari_optimal_lambda_max=0.1,
        ari_optimal_lambda_count=1,
        ari_hits_lower_boundary=False,
        ari_hits_upper_boundary=False,
        ari_boundary_unresolved=False,
        ari_optimum_resolved=True,
        adaptive_search_rounds_completed=0,
        adaptive_search_stop_reason="not_applicable",
        bic_df_scale=1.0,
        bic_cluster_penalty=0.0,
    )

    def fake_select_model(**kwargs):
        captured.update(kwargs)
        return selection_result

    monkeypatch.setattr(pipeline, "load_tumor_tsv", lambda *args, **kwargs: data)
    monkeypatch.setattr(pipeline, "select_model", fake_select_model)
    (tmp_path / data.tumor_id).mkdir()

    summary, _ = pipeline.process_one_file_bundle(
        file_path=tmp_path / "fake.tsv",
        outdir=tmp_path,
        simulation_root=tmp_path,
        write_outputs=False,
        evaluate_all_candidates=True,
        finalize_selected_fit=False,
    )

    assert captured["evaluate_all_candidates"] is True
    assert summary["evaluate_all_candidates"] is True
