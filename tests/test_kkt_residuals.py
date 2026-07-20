from __future__ import annotations

import pytest
import torch

import CliPP2.core.fusion.torch_backend as torch_backend
from CliPP2.core.fusion.torch_backend import (
    _box_qp_sweeps_for_atol,
    graph_fusion_kkt_residual_from_grad_torch,
    inner_kkt_residual_torch,
    project_stationarity_cone_torch,
    refine_graph_fusion_dual_certificate_torch,
    resolve_runtime,
    solve_majorized_subproblem_alm_torch,
    stationarity_residual_torch,
)
from CliPP2.core.fusion.graph_ops import graph_adjoint_edges, graph_forward_edges


def _residual(
    *,
    phi: list[list[float]],
    grad: list[list[float]],
    dual: list[list[float]] | None,
    lambda_value: float,
    lower: list[list[float]] | None = None,
    upper: list[list[float]] | None = None,
) -> dict[str, float]:
    dtype = torch.float64
    phi_t = torch.tensor(phi, dtype=dtype)
    edge_u = (
        torch.tensor([0], dtype=torch.long)
        if len(phi) == 2
        else torch.empty((0,), dtype=torch.long)
    )
    edge_v = (
        torch.tensor([1], dtype=torch.long)
        if len(phi) == 2
        else torch.empty((0,), dtype=torch.long)
    )
    edge_w = (
        torch.tensor([3.0], dtype=dtype)
        if len(phi) == 2
        else torch.empty((0,), dtype=dtype)
    )
    return graph_fusion_kkt_residual_from_grad_torch(
        phi=phi_t,
        grad_smooth=torch.tensor(grad, dtype=dtype),
        dual_kkt=None if dual is None else torch.tensor(dual, dtype=dtype),
        lower=torch.tensor(lower, dtype=dtype)
        if lower is not None
        else torch.zeros_like(phi_t),
        upper=torch.tensor(upper, dtype=dtype)
        if upper is not None
        else torch.full_like(phi_t, 2.0),
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lambda_value,
        atol=1e-8,
    )


def test_stationarity_cone_projection_covers_every_coordinate_type() -> None:
    total_grad = torch.tensor(
        [[-3.0, 4.0], [-3.0, 4.0], [-3.0, 4.0], [-3.0, 4.0]],
        dtype=torch.float64,
    )
    phi = torch.tensor(
        [[0.5, 0.5], [0.0, 0.0], [1.0, 1.0], [0.25, 0.25]],
        dtype=total_grad.dtype,
    )
    lower = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.25, 0.25]],
        dtype=total_grad.dtype,
    )
    upper = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.25, 0.25]],
        dtype=total_grad.dtype,
    )

    projected = project_stationarity_cone_torch(
        total_grad,
        phi=phi,
        lower=lower,
        upper=upper,
    )

    expected = torch.tensor(
        [[0.0, 0.0], [0.0, 4.0], [-3.0, 0.0], [-3.0, 4.0]],
        dtype=total_grad.dtype,
    )
    torch.testing.assert_close(projected, expected, rtol=0.0, atol=0.0)


def test_stationarity_cone_projection_uses_exact_boundary_membership() -> None:
    eps = 1e-12
    total_grad = torch.tensor([[3.0], [-4.0]], dtype=torch.float64)
    phi = torch.tensor([[eps], [1.0 - eps]], dtype=total_grad.dtype)
    lower = torch.zeros_like(phi)
    upper = torch.ones_like(phi)

    projected = project_stationarity_cone_torch(
        total_grad,
        phi=phi,
        lower=lower,
        upper=upper,
    )

    torch.testing.assert_close(
        projected,
        torch.zeros_like(total_grad),
        rtol=0.0,
        atol=0.0,
    )


def test_stationarity_cone_residual_is_not_clipped_by_the_box_width() -> None:
    total_grad = torch.tensor([[-10.0], [10.0], [10.0]], dtype=torch.float64)
    phi = torch.tensor([[0.0], [1.0], [0.5]], dtype=total_grad.dtype)
    lower = torch.zeros_like(phi)
    upper = torch.ones_like(phi)

    cone_residual = total_grad - project_stationarity_cone_torch(
        total_grad,
        phi=phi,
        lower=lower,
        upper=upper,
    )
    legacy_residual = stationarity_residual_torch(
        total_grad=total_grad,
        phi=phi,
        lower=lower,
        upper=upper,
        atol=1e-8,
    )

    torch.testing.assert_close(
        cone_residual,
        total_grad,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        legacy_residual,
        torch.tensor([[-1.0], [1.0], [0.5]], dtype=total_grad.dtype),
        rtol=0.0,
        atol=0.0,
    )


def test_squared_cone_distance_has_the_expected_edge_gradient() -> None:
    dtype = torch.float64
    phi = torch.tensor([[0.0, 0.5], [1.0, 0.5]], dtype=dtype)
    lower = torch.zeros_like(phi)
    upper = torch.ones_like(phi)
    base_grad = torch.tensor([[-1.2, 0.7], [0.9, -0.4]], dtype=dtype)
    edge_u = torch.tensor([0], dtype=torch.long)
    edge_v = torch.tensor([1], dtype=torch.long)
    edge_dual = torch.tensor([[0.2, -0.1]], dtype=dtype)

    def objective(dual: torch.Tensor) -> torch.Tensor:
        total_grad = base_grad + graph_adjoint_edges(
            dual,
            edge_u=edge_u,
            edge_v=edge_v,
            num_nodes=2,
        )
        residual = total_grad - project_stationarity_cone_torch(
            total_grad,
            phi=phi,
            lower=lower,
            upper=upper,
        )
        return 0.5 * torch.sum(residual.square())

    total_grad = base_grad + graph_adjoint_edges(
        edge_dual,
        edge_u=edge_u,
        edge_v=edge_v,
        num_nodes=2,
    )
    cone_residual = total_grad - project_stationarity_cone_torch(
        total_grad,
        phi=phi,
        lower=lower,
        upper=upper,
    )
    expected = graph_forward_edges(
        cone_residual,
        edge_u=edge_u,
        edge_v=edge_v,
    )

    step = 1e-6
    finite_difference = torch.empty_like(edge_dual)
    for region in range(int(edge_dual.shape[1])):
        perturbation = torch.zeros_like(edge_dual)
        perturbation[0, region] = step
        finite_difference[0, region] = (
            objective(edge_dual + perturbation) - objective(edge_dual - perturbation)
        ) / (2.0 * step)

    torch.testing.assert_close(
        finite_difference,
        expected,
        rtol=1e-8,
        atol=1e-9,
    )


def test_active_edge_certificate_uses_oriented_penalty_subgradient() -> None:
    # diff = phi_0 - phi_1 = -0.5, so q = lambda * w * sign(diff) = -6.
    # grad = -D^T q balances the smooth and nonsmooth terms.
    diag = _residual(
        phi=[[0.5], [1.0]],
        grad=[[6.0], [-6.0]],
        dual=[[-6.0]],
        lambda_value=2.0,
    )
    assert diag["kkt_residual"] == 0.0
    assert diag["projected_stationarity_residual"] == diag["stationarity_residual"]
    assert diag["projected_stationarity_norm"] == 0.0
    assert diag["stationarity_normalizer"] > 1.0
    assert diag["smooth_gradient_norm"] > 0.0
    assert diag["fusion_adjustment_norm"] > 0.0
    assert diag["box_primal_violation"] == 0.0
    assert diag["num_interior_coordinates"] == 2
    assert diag["num_lower_active_coordinates"] == 0
    assert diag["num_upper_active_coordinates"] == 0
    assert diag["num_frozen_coordinates"] == 0
    assert diag["edge_subgradient_residual"] == 0.0


def test_fused_edge_accepts_interior_dual_ball_certificate() -> None:
    diag = _residual(
        phi=[[0.5], [0.5]],
        grad=[[1.0], [-1.0]],
        dual=[[-1.0]],
        lambda_value=2.0,
    )
    assert diag["kkt_residual"] == 0.0
    assert diag["dual_ball_residual"] == 0.0


def test_fused_edge_dual_refinement_finds_nonzero_certificate() -> None:
    dtype = torch.float64
    phi = torch.tensor([[0.5], [0.5]], dtype=dtype)
    grad = torch.tensor([[1.0], [-1.0]], dtype=dtype)
    edge_u = torch.tensor([0], dtype=torch.long)
    edge_v = torch.tensor([1], dtype=torch.long)
    edge_w = torch.tensor([1.0], dtype=dtype)

    zero_diag = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad,
        dual_kkt=None,
        lower=torch.zeros_like(phi),
        upper=torch.ones_like(phi),
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=2.0,
        atol=1e-8,
    )
    audit = refine_graph_fusion_dual_certificate_torch(
        phi=phi,
        grad_smooth=grad,
        dual_kkt=None,
        lower=torch.zeros_like(phi),
        upper=torch.ones_like(phi),
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=2.0,
        atol=1e-8,
        max_iter=200,
    )

    assert zero_diag["kkt_residual"] > 0.0
    assert audit["status"] == "refined_fused_edge_dual"
    assert audit["dual_refined"]
    assert audit["fused_edges"] == 1
    assert audit["diag"]["kkt_residual"] < 1e-6
    assert audit["stationarity_after"] < audit["stationarity_before"]


def test_fused_edge_rejects_dual_outside_ball() -> None:
    diag = _residual(
        phi=[[0.5], [0.5]],
        grad=[[0.0], [0.0]],
        dual=[[7.0]],
        lambda_value=2.0,
    )
    assert diag["kkt_residual"] > 0.0
    assert diag["dual_ball_residual"] > 0.0


def test_dual_certificate_refinement_never_degrades_incoming_kkt() -> None:
    dtype = torch.float64
    phi = torch.tensor(
        [
            [0.5803276078096351, 0.5174833393279104],
            [0.5088978095025879, 0.4386281992270642],
            [0.5046182451684760, 0.36317408223340164],
        ],
        dtype=dtype,
    )
    incoming_dual = torch.tensor(
        [
            [0.6620388477967750, 0.7494695217338186],
            [0.3939990588925312, 0.9191108429301657],
            [0.0696379971743047, 0.9975723278788117],
        ],
        dtype=dtype,
    )
    grad = torch.tensor(
        [
            [-1.0560379066893062, -1.6685803646639843],
            [0.5924008506224703, -0.2481028061449930],
            [0.4636370560668360, 1.9166831708089773],
        ],
        dtype=dtype,
    )
    edge_u = torch.tensor([0, 0, 1], dtype=torch.long)
    edge_v = torch.tensor([1, 2, 2], dtype=torch.long)
    edge_w = torch.ones(3, dtype=dtype)
    lower = torch.zeros_like(phi)
    upper = torch.ones_like(phi)
    before = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad,
        dual_kkt=incoming_dual,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=1.0,
        atol=1e-4,
    )

    audit = refine_graph_fusion_dual_certificate_torch(
        phi=phi,
        grad_smooth=grad,
        dual_kkt=incoming_dual,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=1.0,
        atol=1e-4,
        max_iter=1,
    )

    assert audit["status"] == "input_dual_retained"
    assert not audit["dual_refined"]
    assert audit["diag"]["kkt_residual"] <= before["kkt_residual"]
    assert torch.equal(audit["dual"], incoming_dual)


def test_box_normal_cone_signs() -> None:
    lower = [[0.0]]
    upper = [[2.0]]
    lower_ok = _residual(
        phi=[[0.0]], grad=[[1.0]], dual=None, lambda_value=0.0, lower=lower, upper=upper
    )
    lower_bad = _residual(
        phi=[[0.0]],
        grad=[[-1.0]],
        dual=None,
        lambda_value=0.0,
        lower=lower,
        upper=upper,
    )
    upper_ok = _residual(
        phi=[[2.0]],
        grad=[[-1.0]],
        dual=None,
        lambda_value=0.0,
        lower=lower,
        upper=upper,
    )
    upper_bad = _residual(
        phi=[[2.0]], grad=[[1.0]], dual=None, lambda_value=0.0, lower=lower, upper=upper
    )

    assert lower_ok["kkt_residual"] == 0.0
    assert lower_bad["kkt_residual"] > 0.0
    assert upper_ok["kkt_residual"] == 0.0
    assert upper_bad["kkt_residual"] > 0.0
    assert lower_ok["num_lower_active_coordinates"] == 1
    assert lower_ok["num_upper_active_coordinates"] == 0
    assert upper_ok["num_lower_active_coordinates"] == 0
    assert upper_ok["num_upper_active_coordinates"] == 1
    assert lower_ok["box_primal_violation"] == 0.0


def test_near_lower_bound_interior_uses_projected_box_residual() -> None:
    # phi is close to the lower bound but strictly interior. A positive gradient
    # would be acceptable only at the exact lower bound, not merely within atol.
    diag = _residual(
        phi=[[5e-9]],
        grad=[[1.0]],
        dual=None,
        lambda_value=0.0,
        lower=[[0.0]],
        upper=[[1.0]],
    )

    assert diag["num_lower_active_coordinates"] == 1
    assert diag["stationarity_residual"] > 0.0
    assert diag["kkt_residual"] > 0.0


def test_kkt_residual_includes_box_primal_violation() -> None:
    diag = _residual(
        phi=[[-0.5]],
        grad=[[0.0]],
        dual=None,
        lambda_value=0.0,
        lower=[[0.0]],
        upper=[[1.0]],
    )

    assert diag["stationarity_residual"] > 0.0
    assert diag["box_primal_violation"] == pytest.approx(0.5)
    assert diag["box_residual"] == pytest.approx(0.25)
    assert diag["kkt_residual"] == pytest.approx(
        max(diag["stationarity_residual"], diag["box_residual"])
    )


@pytest.mark.parametrize("bad_lambda", [-1.0, float("nan"), float("inf")])
def test_kkt_rejects_invalid_lambda_values(bad_lambda: float) -> None:
    with pytest.raises(ValueError, match="lambda_value"):
        _residual(phi=[[0.5]], grad=[[0.0]], dual=None, lambda_value=bad_lambda)


def test_inner_kkt_residual_uses_majorized_surrogate_gradient() -> None:
    dtype = torch.float64
    phi = torch.tensor([[0.5], [1.0]], dtype=dtype)
    h = torch.full_like(phi, 2.0)
    surrogate_grad = torch.tensor([[6.0], [-6.0]], dtype=dtype)
    U = phi - surrogate_grad / h
    residual = inner_kkt_residual_torch(
        phi=phi,
        dual=torch.tensor([[-6.0]], dtype=dtype),
        U=U,
        h=h,
        lower=torch.zeros_like(phi),
        upper=torch.full_like(phi, 2.0),
        lambda_value=2.0,
        edge_u=torch.tensor([0], dtype=torch.long),
        edge_v=torch.tensor([1], dtype=torch.long),
        edge_w=torch.tensor([3.0], dtype=dtype),
        atol=1e-8,
    )
    assert residual == 0.0


def test_box_qp_sweeps_are_accuracy_driven_not_alm_budget() -> None:
    assert _box_qp_sweeps_for_atol(1e-6, max_iter=32) == 21
    assert _box_qp_sweeps_for_atol(1e-8, max_iter=32) == 28
    assert _box_qp_sweeps_for_atol(1e-12, max_iter=32) == 32


def test_zero_penalty_alm_shortcut_reports_no_admm_iterations() -> None:
    runtime = resolve_runtime("cpu", dtype="float64")
    U = torch.tensor([[0.0], [1.0]], dtype=runtime.dtype, device=runtime.device)
    h = torch.full_like(U, 4.0)
    lower = torch.full_like(U, -10.0)
    upper = torch.full_like(U, 10.0)

    _, _, _, iterations, converged, residual = solve_majorized_subproblem_alm_torch(
        runtime=runtime,
        num_mutations=2,
        U=U,
        h=h,
        lower=lower,
        upper=upper,
        lambda_value=0.0,
        edge_u=torch.tensor([0], dtype=torch.long, device=runtime.device),
        edge_v=torch.tensor([1], dtype=torch.long, device=runtime.device),
        edge_w=torch.tensor([1.0], dtype=runtime.dtype, device=runtime.device),
        tol=1e-8,
        max_iter=16,
        phi_start=U,
        dual_start=None,
    )

    assert iterations == 0
    assert converged
    assert residual == 0.0


def test_alm_uses_periodic_kkt_audits_and_decoupled_box_budget(monkeypatch) -> None:
    runtime = resolve_runtime("cpu", dtype="float64")
    dtype = runtime.dtype
    device = runtime.device
    U = torch.tensor([[0.0], [1.0]], dtype=dtype, device=device)
    h = torch.full_like(U, 4.0)
    lower = torch.full_like(U, -10.0)
    upper = torch.full_like(U, 10.0)
    edge_u = torch.tensor([0], dtype=torch.long, device=device)
    edge_v = torch.tensor([1], dtype=torch.long, device=device)
    edge_w = torch.tensor([1.0], dtype=dtype, device=device)
    original_box_qp = torch_backend._complete_graph_isotropic_box_qp_torch
    box_iters: list[int] = []
    audit_calls = 0

    def counting_box_qp(**kwargs):
        box_iters.append(int(kwargs["max_iter"]))
        return original_box_qp(**kwargs)

    def counting_kkt(**kwargs):
        nonlocal audit_calls
        audit_calls += 1
        return 1.0

    monkeypatch.setattr(
        torch_backend, "_complete_graph_isotropic_box_qp_torch", counting_box_qp
    )
    monkeypatch.setattr(torch_backend, "inner_kkt_residual_torch", counting_kkt)

    _, _, _, iterations, converged, residual = (
        torch_backend.solve_majorized_subproblem_alm_torch(
            runtime=runtime,
            num_mutations=2,
            U=U,
            h=h,
            lower=lower,
            upper=upper,
            lambda_value=0.5,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            tol=1e-12,
            max_iter=12,
            phi_start=U,
            dual_start=None,
            kkt_check_every=4,
            box_phi_atol=1e-6,
        )
    )

    assert not converged
    assert residual == 1.0
    assert iterations == 12
    assert audit_calls == 3
    assert len(box_iters) == 12
    assert set(box_iters) == {21}


def test_pdhg_uses_precomputed_tau_node_without_rebuilding_degree(monkeypatch) -> None:
    runtime = resolve_runtime("cpu", dtype="float64")
    dtype = runtime.dtype
    device = runtime.device
    U = torch.tensor([[0.0], [1.0]], dtype=dtype, device=device)
    h = torch.full_like(U, 4.0)
    lower = torch.full_like(U, -10.0)
    upper = torch.full_like(U, 10.0)
    edge_u = torch.tensor([0], dtype=torch.long, device=device)
    edge_v = torch.tensor([1], dtype=torch.long, device=device)
    edge_w = torch.tensor([1.0], dtype=dtype, device=device)
    tau_node = torch.full((2, 1), 0.99, dtype=dtype, device=device)

    def fail_bincount(*args, **kwargs):
        raise AssertionError("PDHG should use the supplied tau_node")

    monkeypatch.setattr(torch, "bincount", fail_bincount)

    phi, dual, dual_kkt, iterations, _, residual = (
        torch_backend.solve_majorized_subproblem_pdhg_torch(
            runtime=runtime,
            num_mutations=2,
            U=U,
            h=h,
            lower=lower,
            upper=upper,
            lambda_value=0.5,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            degree_bound=1,
            tol=1e-8,
            max_iter=16,
            phi_start=U,
            dual_start=None,
            tau_node=tau_node,
        )
    )

    assert iterations > 0
    assert torch.isfinite(phi).all()
    assert torch.isfinite(dual).all()
    assert torch.equal(dual, dual_kkt)
    assert residual >= 0.0


def test_false_certificate_s2_perpendicular_dual_is_rejected() -> None:
    """Audit Finding 3 regression (false KKT certificate).

    Construct a 2-mutation, 2-sample case where phi[0] - phi[1] = [delta, 0]
    with delta < atol.  The WRONG dual [0, lambda*w] has the same L2 norm as
    the correct dual [lambda*w, 0] but points in the wrong direction.

    Old code path:
      diff_norm = delta < atol  →  "inside ball" branch
      |wrong_dual| = lambda*w   =  radius  →  ball check passes  →  FALSE cert.

    New proximal fixed-point path:
      v = diff + wrong_dual = [delta, lambda*w]
      |v| ≈ lambda*w = radius  (float64 rounds diff^2 to 0 in sum of squares)
      but big=(|v|>=radius)  →  r = -wrong_dual + radius*v/|v| = [delta, 0] ≠ 0
      →  edge_subgradient_residual > 0  →  false cert detected.
    """
    dtype = torch.float64
    delta = 5e-9  # small enough to be inside atol=1e-8
    lam = 2.0
    w = 3.0  # radius = lam * w = 6.0

    phi = torch.tensor([[0.5 + delta, 0.5], [0.5, 0.5]], dtype=dtype)

    # grad_smooth chosen so stationarity=0 with the WRONG dual [[0, 6]]:
    #   total_grad = grad_smooth + D^T * wrong_dual = 0
    #   D^T * [[0,6]] = [[0,6],[0,-6]]  →  grad_smooth = [[0,-6],[0,6]]
    grad_wrong_dual = torch.tensor([[0.0, -lam * w], [0.0, lam * w]], dtype=dtype)
    wrong_dual = torch.tensor([[0.0, lam * w]], dtype=dtype)

    # grad_smooth chosen so stationarity=0 with the CORRECT dual [[6, 0]]:
    #   D^T * [[6,0]] = [[6,0],[-6,0]]  →  grad_smooth = [[-6,0],[6,0]]
    grad_correct_dual = torch.tensor([[-lam * w, 0.0], [lam * w, 0.0]], dtype=dtype)
    correct_dual = torch.tensor([[lam * w, 0.0]], dtype=dtype)

    edge_u = torch.tensor([0], dtype=torch.long)
    edge_v = torch.tensor([1], dtype=torch.long)
    edge_w = torch.tensor([w], dtype=dtype)
    lower = torch.zeros_like(phi)
    upper = torch.full_like(phi, 2.0)

    diag_wrong = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad_wrong_dual,
        dual_kkt=wrong_dual,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lam,
        atol=1e-8,
    )
    diag_correct = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad_correct_dual,
        dual_kkt=correct_dual,
        lower=lower,
        upper=upper,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        lambda_value=lam,
        atol=1e-8,
    )

    # Wrong dual: stationarity=0 (by construction) but edge residual > 0
    assert diag_wrong["stationarity_residual"] == pytest.approx(0.0, abs=1e-12)
    assert diag_wrong["edge_subgradient_residual"] > 0.0
    assert diag_wrong["kkt_residual"] > 0.0

    # Correct dual: both stationarity and edge residual are zero
    assert diag_correct["stationarity_residual"] == pytest.approx(0.0, abs=1e-12)
    assert diag_correct["edge_subgradient_residual"] == pytest.approx(0.0, abs=1e-15)
    assert diag_correct["kkt_residual"] == pytest.approx(0.0, abs=1e-15)


def test_alm_returns_kkt_scaled_dual_separate_from_warm_start() -> None:
    runtime = resolve_runtime("cpu", dtype="float64")
    dtype = runtime.dtype
    device = runtime.device
    U = torch.tensor([[0.0], [1.0]], dtype=dtype, device=device)
    h = torch.full_like(U, 4.0)
    lower = torch.full_like(U, -10.0)
    upper = torch.full_like(U, 10.0)
    edge_u = torch.tensor([0], dtype=torch.long, device=device)
    edge_v = torch.tensor([1], dtype=torch.long, device=device)
    edge_w = torch.tensor([1.0], dtype=dtype, device=device)

    phi, warm_dual, dual_kkt, _, converged, residual = (
        solve_majorized_subproblem_alm_torch(
            runtime=runtime,
            num_mutations=2,
            U=U,
            h=h,
            lower=lower,
            upper=upper,
            lambda_value=0.5,
            edge_u=edge_u,
            edge_v=edge_v,
            edge_w=edge_w,
            tol=1e-8,
            max_iter=1000,
            phi_start=U,
            dual_start=None,
        )
    )

    wrong_scaled_residual = inner_kkt_residual_torch(
        phi=phi,
        dual=warm_dual,
        U=U,
        h=h,
        lower=lower,
        upper=upper,
        lambda_value=0.5,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        atol=1e-8,
    )

    assert converged
    assert residual < 5e-8
    assert torch.allclose(
        phi, torch.tensor([[0.125], [0.875]], dtype=dtype, device=device), atol=1e-7
    )
    assert torch.allclose(dual_kkt, 4.0 * warm_dual, atol=1e-8)
    assert torch.allclose(
        dual_kkt, torch.tensor([[-0.5]], dtype=dtype, device=device), atol=5e-8
    )
    assert wrong_scaled_residual > 1e-2


def test_spectral_rho_admm_is_equivariant_to_objective_scaling() -> None:
    runtime = resolve_runtime("cpu", dtype="float64")
    dtype = runtime.dtype
    device = runtime.device
    U = torch.tensor([[0.0], [1.0]], dtype=dtype, device=device)
    h = torch.full_like(U, 4.0)
    lower = torch.full_like(U, -10.0)
    upper = torch.full_like(U, 10.0)
    edge_u = torch.tensor([0], dtype=torch.long, device=device)
    edge_v = torch.tensor([1], dtype=torch.long, device=device)
    edge_w = torch.tensor([1.0], dtype=dtype, device=device)

    base = solve_majorized_subproblem_alm_torch(
        runtime=runtime,
        num_mutations=2,
        U=U,
        h=h,
        lower=lower,
        upper=upper,
        lambda_value=0.5,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        tol=1e-7,
        max_iter=2000,
        phi_start=U,
        dual_start=None,
        spectral_rho=True,
    )
    factor = 100.0
    scaled = solve_majorized_subproblem_alm_torch(
        runtime=runtime,
        num_mutations=2,
        U=U,
        h=factor * h,
        lower=lower,
        upper=upper,
        lambda_value=factor * 0.5,
        edge_u=edge_u,
        edge_v=edge_v,
        edge_w=edge_w,
        tol=1e-7,
        max_iter=2000,
        phi_start=U,
        dual_start=None,
        spectral_rho=True,
    )

    phi_base, scaled_dual_base, actual_dual_base, _, converged_base, _ = base
    phi_scaled, scaled_dual_scaled, actual_dual_scaled, _, converged_scaled, _ = scaled
    assert converged_base and converged_scaled
    assert torch.allclose(phi_scaled, phi_base, atol=1e-6)
    assert torch.allclose(scaled_dual_scaled, scaled_dual_base, atol=1e-6)
    assert torch.allclose(
        actual_dual_scaled,
        factor * actual_dual_base,
        atol=1e-5,
    )
