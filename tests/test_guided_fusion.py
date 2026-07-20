from __future__ import annotations

import numpy as np
import pytest
import torch

from CliPP2.core.fusion.solver import (
    fit_observed_data_pairwise_fusion,
    prepare_torch_problem,
)
from CliPP2.core.fusion.types import PairwiseFusionGraph
from CliPP2.io.data import TumorData, compute_phi_init_from_counts
from CliPP2.model_selection.guided_fusion import (
    build_guided_fusion_initialization,
)


def _tumor(num_mutations: int, num_regions: int) -> TumorData:
    alt = np.arange(
        5.0, 5.0 + float(num_mutations * num_regions), dtype=np.float64
    ).reshape(num_mutations, num_regions)
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
        tumor_id="guided-fusion-test",
        mutation_ids=[f"m{i}" for i in range(num_mutations)],
        region_ids=[f"r{i}" for i in range(num_regions)],
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


def _context(
    guide_phi: np.ndarray,
    *,
    graph: PairwiseFusionGraph | None = None,
) -> tuple[TumorData, object]:
    data = _tumor(int(guide_phi.shape[0]), int(guide_phi.shape[1]))
    context = prepare_torch_problem(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-4,
        inner_max_iter=30,
        graph=graph,
        exact_pilot=guide_phi,
        pooled_start=guide_phi,
        scalar_well_starts=(),
        device="cpu",
        dtype="float64",
        objective_shape="unimodal",
    )
    return data, context


def test_guided_initialization_uses_observed_within_block_dual_capacity() -> None:
    guide = np.asarray([[0.25, 0.35], [0.25, 0.35], [0.75, 0.65]], dtype=np.float64)
    labels = np.asarray([7, 7, 19], dtype=np.int64)
    grad = np.asarray([[1.0, 2.0], [-1.0, -2.0], [0.0, 0.0]])
    _, context = _context(guide)

    result = build_guided_fusion_initialization(
        guide,
        labels,
        solver_context=context,
        grad_smooth=grad,
    )

    edge_u, edge_v = context.graph.edge_u, context.graph.edge_v
    within = (edge_u == 0) & (edge_v == 1)
    within_weight = float(context.graph.weight[within].item())
    expected_lambda = np.sqrt(5.0) / within_weight
    expected_dual = torch.tensor([-1.0, -2.0], dtype=torch.float64)

    assert result.lambda_value == pytest.approx(expected_lambda)
    assert result.diagnostics.required_lambda_without_between_edges == pytest.approx(
        expected_lambda
    )
    assert result.solver_state.previous_lambda == result.lambda_value
    assert not hasattr(result.solver_state, "split")
    assert not hasattr(result.solver_state, "curvature")
    assert torch.equal(result.solver_state.phi[0], result.solver_state.phi[1])
    assert torch.allclose(result.solver_state.dual[within][0], expected_dual)
    assert result.diagnostics.max_dual_ball_ratio <= 1.0 + 1e-12
    assert result.diagnostics.max_within_dual_ball_ratio == pytest.approx(1.0)
    assert result.diagnostics.max_between_dual_ball_ratio == pytest.approx(1.0)
    assert result.diagnostics.dual_ball_residual == 0.0
    assert result.diagnostics.block_flow_balance_max_abs < 1e-12
    assert result.diagnostics.gradient_source == "provided"


def test_exact_lower_bound_absorbs_an_outward_gradient() -> None:
    provisional = np.full((2, 1), 1e-6, dtype=np.float64)
    _, context = _context(provisional)
    lower = float(context.lower[0, 0].item())
    guide = np.full((2, 1), lower, dtype=np.float64)
    grad = np.asarray([[3.0], [1.0]], dtype=np.float64)

    result = build_guided_fusion_initialization(
        guide,
        np.asarray([0, 0]),
        solver_context=context,
        grad_smooth=grad,
    )

    assert result.diagnostics.required_lambda_without_between_edges == 0.0
    assert result.lambda_value == result.diagnostics.numerical_lambda_floor
    assert result.lambda_value > 0.0
    assert torch.count_nonzero(result.solver_state.dual).item() == 0
    assert result.diagnostics.num_exact_lower_active_coordinates == 2
    assert result.diagnostics.stationarity_residual == 0.0
    assert result.diagnostics.kkt_residual == 0.0


def test_near_lower_bound_remains_interior_for_stationarity() -> None:
    provisional = np.full((2, 1), 1e-6, dtype=np.float64)
    _, context = _context(provisional)
    lower = float(context.lower[0, 0].item())
    guide = np.full((2, 1), lower + 1e-10, dtype=np.float64)
    grad = np.asarray([[3.0], [1.0]], dtype=np.float64)

    result = build_guided_fusion_initialization(
        guide,
        np.asarray([0, 0]),
        solver_context=context,
        grad_smooth=grad,
    )

    edge_weight = float(context.graph.weight.item())
    assert result.diagnostics.required_lambda_without_between_edges == pytest.approx(
        1.0 / edge_weight
    )
    assert result.diagnostics.num_exact_lower_active_coordinates == 0
    assert result.diagnostics.stationarity_residual > 0.0
    assert result.diagnostics.kkt_residual > 0.0


def test_observed_gradient_state_is_consumed_by_complete_graph_solver() -> None:
    guide = np.asarray([[0.25], [0.25], [0.65]], dtype=np.float64)
    labels = np.asarray([0, 0, 1], dtype=np.int64)
    data, context = _context(guide)

    initialization = build_guided_fusion_initialization(
        guide,
        labels,
        solver_context=context,
    )
    artifacts = fit_observed_data_pairwise_fusion(
        data,
        lambda_value=initialization.lambda_value,
        major_prior=0.5,
        eps=1e-6,
        outer_max_iter=8,
        inner_max_iter=30,
        tol=1e-4,
        phi_start=initialization.solver_state.phi,
        exact_pilot=guide,
        pooled_start=guide,
        scalar_well_starts=(),
        start_mode="warm_only",
        device="cpu",
        dtype="float64",
        solver_context=context,
        solver_state=initialization.solver_state,
        objective_shape="unimodal",
    )

    assert initialization.diagnostics.gradient_source == "observed_likelihood"
    assert np.isfinite(initialization.lambda_value)
    assert initialization.lambda_value > 0.0
    assert np.isfinite(artifacts.penalized_objective)
    assert artifacts.solver_state is not None
    assert artifacts.solver_state.dual is not None


def test_guided_initialization_rejects_noncomplete_graph() -> None:
    guide = np.asarray([[0.2], [0.2], [0.7]], dtype=np.float64)
    line_graph = PairwiseFusionGraph(
        edge_u=np.asarray([0, 1], dtype=np.int32),
        edge_v=np.asarray([1, 2], dtype=np.int32),
        edge_w=np.ones(2, dtype=np.float64),
        name="line",
        degree_bound=2,
    )
    _, context = _context(guide, graph=line_graph)

    with pytest.raises(ValueError, match="complete pairwise-fusion graph"):
        build_guided_fusion_initialization(
            guide,
            np.asarray([0, 0, 1]),
            solver_context=context,
        )


def test_guided_initialization_rejects_nonconstant_guide_block() -> None:
    guide = np.asarray([[0.2], [0.3], [0.7]], dtype=np.float64)
    _, context = _context(guide)

    with pytest.raises(ValueError, match="share a center"):
        build_guided_fusion_initialization(
            guide,
            np.asarray([0, 0, 1]),
            solver_context=context,
        )
