from __future__ import annotations

import numpy as np

import CliPP2
from CliPP2.core.model import (
    FitOptions,
    FitResult,
    Problem,
    SolverOptions,
    fit_fixed_objective,
)
from CliPP2.runners.model_selection import SelectionArtifact, select_model

from test_adaptive_lambda_path import _tiny_tumor


def _weighted_line_graph() -> CliPP2.PairwiseFusionGraph:
    return CliPP2.PairwiseFusionGraph(
        edge_u=np.asarray([0, 1], dtype=np.int32),
        edge_v=np.asarray([1, 2], dtype=np.int32),
        edge_w=np.asarray([0.25, 2.0], dtype=np.float64),
        name="weighted_line",
        degree_bound=2,
    )


def test_fit_result_excludes_selection_owned_fields() -> None:
    forbidden = {
        "bic",
        "classic_bic",
        "extended_bic",
        "bic_refit_phi",
        "bic_refit_cluster_centers",
        "bic_partition_labels",
        "selection_score_name",
    }

    assert forbidden.isdisjoint(FitResult.__dataclass_fields__)


def test_summary_property_is_opt_in_for_compact_fit_view() -> None:
    data = _tiny_tumor()
    graph = _weighted_line_graph()
    problem = Problem(data=data, graph=graph, lambda_value=0.1)
    options = SolverOptions(
        outer_max_iter=2,
        inner_max_iter=16,
        tol=1e-4,
        device="cpu",
        dtype="float64",
        compute_summary=False,
    )

    result = CliPP2.fit(problem, options)
    assert result.summary is None

    with_summary = CliPP2.fit(
        problem, SolverOptions(**{**options.__dict__, "compute_summary": True})
    )
    assert with_summary.summary is not None


def test_problem_graph_weights_are_explicit_and_recorded_in_result() -> None:
    data = _tiny_tumor()
    graph = _weighted_line_graph()
    problem = Problem(data=data, graph=graph, lambda_value=0.2)

    result = CliPP2.fit(
        problem,
        SolverOptions(
            outer_max_iter=1,
            inner_max_iter=16,
            tol=1e-4,
            device="cpu",
            dtype="float64",
        ),
    )

    assert np.allclose(problem.graph.edge_w, np.asarray([0.25, 2.0], dtype=np.float64))
    assert result.graph_name == "weighted_line"


def test_selection_artifact_does_not_replace_primary_optimizer() -> None:
    data = _tiny_tumor()
    fit = fit_fixed_objective(
        data,
        FitOptions(
            lambda_value=0.1,
            graph=_weighted_line_graph(),
            outer_max_iter=2,
            inner_max_iter=16,
            tol=1e-4,
            device="cpu",
            dtype="float64",
        ),
        compute_summary=False,
    )
    raw_phi = fit.estimate.phi.copy()
    artifact = SelectionArtifact(
        bic_refit_phi=np.zeros_like(raw_phi),
        bic_partition_labels=np.zeros((raw_phi.shape[0],), dtype=np.int64),
    )

    assert not hasattr(fit, "bic_refit_phi")
    assert np.allclose(fit.estimate.phi, raw_phi)
    assert not np.allclose(artifact.bic_refit_phi, fit.estimate.phi)


def test_model_selection_returns_bic_sidecar_not_fit_fields() -> None:
    data = _tiny_tumor()
    result = select_model(
        data=data,
        simulation_root=None,
        lambda_grid=[0.1],
        lambda_grid_mode="adaptive_bic",
        fit_options=FitOptions(
            lambda_value=0.0,
            graph=_weighted_line_graph(),
            outer_max_iter=2,
            inner_max_iter=16,
            tol=1e-4,
            device="cpu",
            dtype="float64",
        ),
        bic_df_scale=1.0,
        bic_cluster_penalty=0.0,
        selection_score="bic",
        use_warm_starts=True,
        evaluate_all_candidates=False,
        finalize_selected_fit=False,
    )

    assert result.selected_artifact.classic_bic is not None
    assert not hasattr(result.best_fit, "classic_bic")
