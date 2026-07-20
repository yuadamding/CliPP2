from __future__ import annotations

import pytest
import torch

import CliPP2.core.fusion.torch_backend as backend
from CliPP2.core.fusion.torch_backend import (
    graph_fusion_kkt_residual_from_grad_torch,
    pairwise_penalty_torch,
    refine_graph_fusion_dual_certificate_torch,
    resolve_runtime,
    solve_majorized_subproblem_alm_torch,
)


def _complete_problem() -> dict[str, object]:
    runtime = resolve_runtime("cpu", dtype="float64")
    phi_start = torch.tensor(
        [
            [0.10, 0.80, 0.30],
            [0.20, 0.70, 0.40],
            [0.50, 0.40, 0.60],
            [0.80, 0.20, 0.70],
            [0.90, 0.10, 0.90],
        ],
        dtype=runtime.dtype,
    )
    edge_index = torch.triu_indices(5, 5, offset=1)
    return {
        "runtime": runtime,
        "num_mutations": 5,
        "U": phi_start,
        "h": torch.full_like(phi_start, 4.0),
        "lower": torch.zeros_like(phi_start),
        "upper": torch.ones_like(phi_start),
        "lambda_value": 0.2,
        "edge_u": edge_index[0],
        "edge_v": edge_index[1],
        "edge_w": torch.linspace(0.7, 1.3, edge_index.shape[1], dtype=runtime.dtype),
        "tol": 1e-8,
        "max_iter": 1000,
        "phi_start": phi_start,
        "dual_start": None,
        "spectral_rho": True,
    }


def test_streamed_complete_graph_admm_matches_dense_one_edge_chunks() -> None:
    problem = _complete_problem()
    dense = solve_majorized_subproblem_alm_torch(**problem)
    # Three float64 values are 24 bytes, so this forces exactly one edge/chunk.
    streamed = solve_majorized_subproblem_alm_torch(**problem, edge_work_bytes=24)

    assert dense[4] and streamed[4]
    assert dense[5] < 5e-8
    assert streamed[5] < 5e-8
    assert torch.allclose(streamed[0], dense[0], atol=2e-10, rtol=2e-10)
    assert torch.allclose(streamed[1], dense[1], atol=2e-9, rtol=2e-9)
    assert torch.allclose(streamed[2], dense[2], atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("dual_start_is_actual", [False, True])
def test_streamed_admm_matches_dense_with_nonspectral_warm_dual(
    dual_start_is_actual: bool,
) -> None:
    problem = _complete_problem()
    problem["spectral_rho"] = False
    num_dual_values = int(problem["edge_u"].numel()) * int(problem["U"].shape[1])
    # The actual-dual case deliberately exceeds some lambda-weight balls so the
    # chunked warm-start projection is exercised. The scaled-u case consumes
    # exactly the same values without projection, covering both API units.
    problem["dual_start"] = torch.linspace(
        -0.5,
        0.5,
        num_dual_values,
        dtype=problem["U"].dtype,
    ).reshape(int(problem["edge_u"].numel()), int(problem["U"].shape[1]))
    problem["dual_start_is_actual"] = dual_start_is_actual

    dense = solve_majorized_subproblem_alm_torch(**problem)
    streamed = solve_majorized_subproblem_alm_torch(
        **problem,
        # One three-region float64 edge per chunk.
        edge_work_bytes=24,
    )

    assert dense[3:] == pytest.approx(streamed[3:], abs=1e-13)
    for dense_tensor, streamed_tensor in zip(dense[:3], streamed[:3], strict=True):
        torch.testing.assert_close(
            streamed_tensor,
            dense_tensor,
            rtol=5e-8,
            atol=5e-9,
        )


def test_chunked_kkt_scaled_dual_matches_materialized_actual_dual() -> None:
    problem = _complete_problem()
    result = solve_majorized_subproblem_alm_torch(**problem, edge_work_bytes=24)
    phi, scaled_dual, actual_dual = result[:3]
    h = problem["h"]
    U = problem["U"]
    common = {
        "phi": phi,
        "grad_smooth": h * (phi - U),
        "lower": problem["lower"],
        "upper": problem["upper"],
        "edge_u": problem["edge_u"],
        "edge_v": problem["edge_v"],
        "edge_w": problem["edge_w"],
        "lambda_value": problem["lambda_value"],
        "atol": problem["tol"],
        "edge_work_bytes": 24,
    }
    actual_diag = graph_fusion_kkt_residual_from_grad_torch(
        **common,
        dual_kkt=actual_dual,
    )
    # Infer the terminal rho from y=rho*u without relying on solver internals.
    nonzero = torch.abs(scaled_dual) > 1e-14
    rho = float(torch.median(actual_dual[nonzero] / scaled_dual[nonzero]).item())
    scaled_diag = graph_fusion_kkt_residual_from_grad_torch(
        **common,
        dual_kkt=scaled_dual,
        dual_scale=rho,
    )

    for key in (
        "stationarity_residual",
        "edge_subgradient_residual",
        "dual_ball_residual",
        "box_residual",
        "kkt_residual",
    ):
        assert scaled_diag[key] == pytest.approx(actual_diag[key], abs=1e-13)


def test_pairwise_penalty_streaming_matches_dense(monkeypatch) -> None:
    problem = _complete_problem()
    phi = problem["phi_start"]
    kwargs = {
        "edge_u": problem["edge_u"],
        "edge_v": problem["edge_v"],
        "edge_w": problem["edge_w"],
        "lambda_value": problem["lambda_value"],
    }
    dense = pairwise_penalty_torch(phi, **kwargs)
    monkeypatch.setattr(backend, "DEFAULT_EDGE_WORK_BYTES", 24)
    streamed = pairwise_penalty_torch(phi, **kwargs)
    assert streamed == pytest.approx(dense, rel=1e-14, abs=1e-14)


def test_nonpositive_edge_work_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="edge work budget must be positive"):
        solve_majorized_subproblem_alm_torch(
            **_complete_problem(),
            edge_work_bytes=0,
        )


def test_streamed_final_certificate_matches_dense_on_active_edge() -> None:
    dtype = torch.float64
    phi = torch.tensor([[0.25], [0.75]], dtype=dtype)
    grad = torch.tensor([[0.5], [-0.5]], dtype=dtype)
    edge_u = torch.tensor([0], dtype=torch.long)
    edge_v = torch.tensor([1], dtype=torch.long)
    edge_w = torch.ones(1, dtype=dtype)
    common = {
        "phi": phi,
        "grad_smooth": grad,
        "dual_kkt": None,
        "lower": torch.zeros_like(phi),
        "upper": torch.ones_like(phi),
        "edge_u": edge_u,
        "edge_v": edge_v,
        "edge_w": edge_w,
        "lambda_value": 0.5,
        "atol": 1e-8,
    }
    dense = refine_graph_fusion_dual_certificate_torch(**common)
    streamed = refine_graph_fusion_dual_certificate_torch(
        **common,
        edge_work_bytes=1,
    )
    assert streamed["status"] == dense["status"] == "analytic_nonfused_dual"
    assert torch.equal(streamed["dual"], dense["dual"])
    assert streamed["diag"]["kkt_residual"] == dense["diag"]["kkt_residual"]


def test_streamed_final_certificate_retains_certified_incoming_dual() -> None:
    dtype = torch.float64
    phi = torch.tensor([[0.5], [0.5]], dtype=dtype)
    incoming = torch.tensor([[-1.0]], dtype=dtype)
    audit = refine_graph_fusion_dual_certificate_torch(
        phi=phi,
        grad_smooth=torch.tensor([[1.0], [-1.0]], dtype=dtype),
        dual_kkt=incoming,
        lower=torch.zeros_like(phi),
        upper=torch.ones_like(phi),
        edge_u=torch.tensor([0], dtype=torch.long),
        edge_v=torch.tensor([1], dtype=torch.long),
        edge_w=torch.ones(1, dtype=dtype),
        lambda_value=2.0,
        atol=1e-8,
        edge_work_bytes=1,
    )
    assert audit["status"] == "input_dual_retained"
    assert audit["diag"]["kkt_residual"] == 0.0
    assert audit["fused_edges"] == 1
    assert audit["nonzero_edges"] == 0
    assert audit["dual"] is incoming


def test_streamed_final_certificate_refines_a_suboptimal_certified_input() -> None:
    # This seed supplies an incoming multiplier inside the downstream 5*atol
    # eligibility gate, while analytic fused-edge refinement produces a
    # materially stronger certificate. Streaming must not return early merely
    # because the incoming state clears that gate.
    torch.manual_seed(0)
    num_mutations, num_regions = 5, 2
    edge_index = torch.triu_indices(num_mutations, num_mutations, offset=1)
    phi = torch.rand(num_mutations, num_regions, dtype=torch.float64)
    phi[1] = phi[0]
    phi[3] = phi[2]
    grad = 0.2 * torch.randn_like(phi)
    incoming = 0.2 * torch.randn(
        edge_index.shape[1],
        num_regions,
        dtype=phi.dtype,
    )
    edge_w = 0.5 + torch.rand(edge_index.shape[1], dtype=phi.dtype)
    atol = 0.1
    common = {
        "phi": phi,
        "grad_smooth": grad,
        "dual_kkt": incoming,
        "lower": torch.zeros_like(phi),
        "upper": torch.ones_like(phi),
        "edge_u": edge_index[0],
        "edge_v": edge_index[1],
        "edge_w": edge_w,
        "lambda_value": 1.0,
        "atol": atol,
    }
    incoming_diag = graph_fusion_kkt_residual_from_grad_torch(**common)
    dense = refine_graph_fusion_dual_certificate_torch(**common, max_iter=96)
    streamed = refine_graph_fusion_dual_certificate_torch(
        **common,
        max_iter=96,
        edge_work_bytes=1,
    )

    assert incoming_diag["kkt_residual"] <= 5.0 * atol
    assert dense["diag"]["kkt_residual"] < incoming_diag["kkt_residual"]
    assert dense["status"] == streamed["status"] == "refined_fused_edge_dual"
    for key, dense_value in dense["diag"].items():
        assert streamed["diag"][key] == pytest.approx(
            dense_value,
            rel=1e-14,
            abs=1e-14,
        )
    torch.testing.assert_close(
        streamed["dual"],
        dense["dual"],
        rtol=1e-14,
        atol=1e-14,
    )


def test_spectral_rho_adaptation_is_objective_scale_equivariant() -> None:
    torch.manual_seed(0)
    runtime = resolve_runtime("cpu", dtype="float64")
    num_mutations, num_regions = 6, 3
    edge_index = torch.triu_indices(num_mutations, num_mutations, offset=1)
    U = torch.rand(num_mutations, num_regions, dtype=runtime.dtype)
    h = 0.1 + 20.0 * torch.rand_like(U)
    edge_w = 0.1 + torch.rand(edge_index.shape[1], dtype=runtime.dtype)
    lambda_value = 0.2
    common = {
        "runtime": runtime,
        "num_mutations": num_mutations,
        "U": U,
        "lower": torch.zeros_like(U),
        "upper": torch.ones_like(U),
        "edge_u": edge_index[0],
        "edge_v": edge_index[1],
        "edge_w": edge_w,
        "tol": 1e-15,
        "max_iter": 10,
        "phi_start": U,
        "dual_start": None,
        "spectral_rho": True,
        "kkt_check_every": 10,
    }
    base = solve_majorized_subproblem_alm_torch(
        **common,
        h=h,
        lambda_value=lambda_value,
    )
    factor = 100.0
    scaled = solve_majorized_subproblem_alm_torch(
        **common,
        h=factor * h,
        lambda_value=factor * lambda_value,
    )

    assert base[3] == scaled[3] == 10
    assert torch.allclose(scaled[0], base[0], atol=2e-7, rtol=2e-7)
    assert torch.allclose(scaled[1], base[1], atol=2e-7, rtol=2e-7)
    assert torch.allclose(
        scaled[2],
        factor * base[2],
        atol=1e-5,
        rtol=2e-7,
    )
