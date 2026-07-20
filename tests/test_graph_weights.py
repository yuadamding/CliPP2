from __future__ import annotations

import numpy as np
import pytest
import torch

import CliPP2.core.fusion.graph_ops as graph_ops_module
import CliPP2.core.fusion.solver as solver_module
from CliPP2.core.fusion.graph import (
    build_complete_adaptive_graph,
    build_complete_uniform_graph,
    build_likelihood_noise_regularized_adaptive_graph,
    coerce_graph,
    likelihood_noise_distance_floor,
)
from CliPP2.core.fusion.graph_ops import (
    PDHG_PRECONDITIONER_ETA,
    build_complete_adaptive_tensor_graph,
    build_likelihood_noise_regularized_adaptive_tensor_graph,
    build_complete_uniform_tensor_graph,
    estimate_complete_tensor_graph_bytes,
    estimate_dense_complete_solver_peak_bytes,
    graph_adjoint,
    graph_forward,
    likelihood_noise_distance_floor_torch,
    project_dual_ball,
    tensor_graph_to_pairwise_graph,
    tensorize_graph,
)
from CliPP2.core.fusion.torch_backend import resolve_runtime
from CliPP2.core.fusion.types import PairwiseFusionGraph


def test_complete_adaptive_graph_weights_are_mean_degree_normalized() -> None:
    pilot_phi = np.asarray(
        [
            [0.1, 0.2],
            [0.1, 0.2],
            [0.4, 0.5],
            [0.8, 0.9],
        ],
        dtype=np.float64,
    )

    uniform = build_complete_uniform_graph(pilot_phi.shape[0])
    adaptive = build_complete_adaptive_graph(
        pilot_phi, gamma=1.0, tau=1e-6, baseline=1.0
    )

    assert adaptive.edge_w.max() < 10.0
    assert np.isclose(float(adaptive.edge_w.mean()), float(uniform.edge_w.mean()))
    assert "mean_normalized" in adaptive.name


def test_likelihood_noise_floor_uses_pairwise_local_standard_error() -> None:
    curvature = np.full((3, 2), 4.0, dtype=np.float64)
    lower = np.zeros_like(curvature)
    upper = np.ones_like(curvature)

    # Each mutation has variance 1/4 in two regions; the difference of two
    # mutation vectors therefore has expected squared scale 2*(1/4+1/4)=1.
    tau = likelihood_noise_distance_floor(
        curvature,
        lower=lower,
        upper=upper,
    )

    assert tau == pytest.approx(1.0)


def test_likelihood_noise_regularized_graph_is_guide_independent_and_bounded() -> None:
    pilot = np.asarray([[0.1], [0.11], [0.7], [0.9]], dtype=np.float64)
    curvature = np.full_like(pilot, 50.0)
    lower = np.zeros_like(pilot)
    upper = np.ones_like(pilot)

    graph, tau = build_likelihood_noise_regularized_adaptive_graph(
        pilot,
        curvature,
        lower=lower,
        upper=upper,
    )

    assert tau == pytest.approx(np.sqrt(2.0 / 50.0))
    assert np.isclose(graph.edge_w.mean(), 1.0 / (pilot.shape[0] - 1))
    assert graph.edge_w.max() / graph.edge_w.min() <= 0.8 / tau + 1e-12
    assert "likelihood_noise" in graph.name

    degree_graph, degree_tau = build_likelihood_noise_regularized_adaptive_graph(
        pilot,
        curvature,
        lower=lower,
        upper=upper,
        noise_divisor=pilot.shape[0] - 1,
    )
    assert degree_tau == pytest.approx(tau / (pilot.shape[0] - 1))
    assert degree_graph.edge_w.max() / degree_graph.edge_w.min() > (
        graph.edge_w.max() / graph.edge_w.min()
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_torch_likelihood_noise_graph_has_exact_reference_spec_and_hash(
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    # Binary-exact inputs make the host and Torch graph specs byte-identical,
    # so this test covers the order-sensitive provenance fingerprint too.
    pilot = np.asarray([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]], dtype=np.float64)
    curvature = np.full_like(pilot, 16.0)
    lower = np.zeros_like(pilot)
    upper = np.ones_like(pilot)
    runtime = resolve_runtime(device, dtype="float64")

    reference, reference_tau = build_likelihood_noise_regularized_adaptive_graph(
        pilot,
        curvature,
        lower=lower,
        upper=upper,
    )
    tensor_graph, observed_tau = (
        build_likelihood_noise_regularized_adaptive_tensor_graph(
            torch.as_tensor(pilot, dtype=runtime.dtype, device=runtime.device),
            torch.as_tensor(curvature, dtype=runtime.dtype, device=runtime.device),
            runtime,
            lower=torch.as_tensor(lower, dtype=runtime.dtype, device=runtime.device),
            upper=torch.as_tensor(upper, dtype=runtime.dtype, device=runtime.device),
        )
    )
    observed = tensor_graph_to_pairwise_graph(tensor_graph)

    assert observed_tau == reference_tau == 0.5
    np.testing.assert_array_equal(observed.edge_u, reference.edge_u)
    np.testing.assert_array_equal(observed.edge_v, reference.edge_v)
    np.testing.assert_array_equal(observed.edge_w, reference.edge_w)
    assert observed.name == reference.name
    assert observed.degree_bound == reference.degree_bound
    assert solver_module._graph_fingerprint(
        observed
    ) == solver_module._graph_fingerprint(reference)
    for value in (
        tensor_graph.edge_index,
        tensor_graph.weight,
        tensor_graph.degree,
        tensor_graph.pdhg_tau_node,
    ):
        assert value.device.type == device


def test_torch_likelihood_noise_floor_matches_numpy_even_median() -> None:
    curvature = np.asarray(
        [[4.0, 9.0], [16.0, 25.0], [36.0, 49.0], [64.0, 81.0]],
        dtype=np.float64,
    )
    lower = np.zeros_like(curvature)
    upper = np.ones_like(curvature)
    expected = likelihood_noise_distance_floor(
        curvature,
        lower=lower,
        upper=upper,
    )

    observed = likelihood_noise_distance_floor_torch(
        torch.as_tensor(curvature),
        lower=torch.as_tensor(lower),
        upper=torch.as_tensor(upper),
    )

    assert observed.device.type == "cpu"
    assert float(observed.item()) == pytest.approx(expected, rel=1e-15, abs=1e-15)


def test_torch_complete_adaptive_graph_matches_numpy_reference() -> None:
    pilot_phi = np.asarray(
        [
            [0.1, 0.2],
            [0.1, 0.2],
            [0.4, 0.5],
            [0.8, 0.9],
        ],
        dtype=np.float64,
    )
    runtime = resolve_runtime("cpu", dtype="float64")

    reference = build_complete_adaptive_graph(
        pilot_phi, gamma=1.25, tau=1e-5, baseline=0.75
    )
    tensor_graph = build_complete_adaptive_tensor_graph(
        torch.as_tensor(pilot_phi, dtype=runtime.dtype),
        runtime,
        gamma=1.25,
        tau=1e-5,
        baseline=0.75,
    )
    host_graph = tensor_graph_to_pairwise_graph(tensor_graph)

    assert host_graph.edge_u.tolist() == reference.edge_u.tolist()
    assert host_graph.edge_v.tolist() == reference.edge_v.tolist()
    assert np.allclose(host_graph.edge_w, reference.edge_w)
    assert np.isclose(float(tensor_graph.weight.mean()), float(reference.edge_w.mean()))


def test_torch_complete_adaptive_graph_matches_numpy_reference_when_chunked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot_phi = np.asarray(
        [
            [0.10, 0.20, 0.30],
            [0.12, 0.19, 0.31],
            [0.40, 0.50, 0.60],
            [0.70, 0.65, 0.55],
            [0.90, 0.80, 0.75],
        ],
        dtype=np.float64,
    )
    runtime = resolve_runtime("cpu", dtype="float64")
    chunk_bytes = (
        torch.empty((), dtype=runtime.dtype).element_size() * pilot_phi.shape[1]
    )
    monkeypatch.setattr(
        graph_ops_module, "COMPLETE_ADAPTIVE_WEIGHT_CHUNK_BYTES", chunk_bytes
    )

    reference = build_complete_adaptive_graph(
        pilot_phi, gamma=1.5, tau=1e-4, baseline=0.5
    )
    tensor_graph = build_complete_adaptive_tensor_graph(
        torch.as_tensor(pilot_phi, dtype=runtime.dtype),
        runtime,
        gamma=1.5,
        tau=1e-4,
        baseline=0.5,
    )
    host_graph = tensor_graph_to_pairwise_graph(tensor_graph)

    assert host_graph.edge_u.tolist() == reference.edge_u.tolist()
    assert host_graph.edge_v.tolist() == reference.edge_v.tolist()
    assert np.allclose(host_graph.edge_w, reference.edge_w)


def test_graph_canonicalization_omits_zero_weight_edges() -> None:
    graph = PairwiseFusionGraph(
        edge_u=np.asarray([0, 0, 1], dtype=np.int32),
        edge_v=np.asarray([1, 2, 2], dtype=np.int32),
        edge_w=np.asarray([1.0, 0.0, 2.0], dtype=np.float64),
        name="with_zero",
    )

    coerced = coerce_graph(3, graph)

    assert coerced.edge_u.tolist() == [0, 1]
    assert coerced.edge_v.tolist() == [1, 2]
    assert coerced.edge_w.tolist() == [1.0, 2.0]


def test_complete_tensor_graph_memory_estimate_accounts_for_adaptive_pairs() -> None:
    uniform_bytes = estimate_complete_tensor_graph_bytes(
        4,
        num_regions=2,
        dtype=torch.float64,
        adaptive=False,
    )
    adaptive_bytes = estimate_complete_tensor_graph_bytes(
        4,
        num_regions=2,
        dtype=torch.float64,
        adaptive=True,
    )

    assert uniform_bytes > 0
    assert adaptive_bytes > uniform_bytes


def test_dense_complete_solver_estimate_includes_persistent_edge_states() -> None:
    graph_only = estimate_dense_complete_solver_peak_bytes(
        8,
        num_regions=3,
        dtype=torch.float64,
        include_dual=False,
        include_split=False,
    )
    solver_peak = estimate_dense_complete_solver_peak_bytes(
        8,
        num_regions=3,
        dtype=torch.float64,
    )

    edge_count = 8 * 7 // 2
    one_edge_state = (
        edge_count * 3 * torch.empty((), dtype=torch.float64).element_size()
    )
    assert solver_peak >= graph_only + 3 * one_edge_state


def test_complete_tensor_graph_memory_guard_runs_before_edge_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = resolve_runtime("cpu", dtype="float64")

    def fail_triu_indices(*args, **kwargs):
        raise AssertionError(
            "complete graph memory guard should run before edge allocation"
        )

    monkeypatch.setattr(torch, "triu_indices", fail_triu_indices)

    with pytest.raises(MemoryError, match="complete uniform tensor graph"):
        build_complete_uniform_tensor_graph(
            100,
            runtime,
            memory_limit_bytes=1,
        )


def test_tensor_graph_complete_flag_requires_complete_edge_structure() -> None:
    runtime = resolve_runtime("cpu", dtype="float64")
    malformed = PairwiseFusionGraph(
        edge_u=np.asarray([0, 0, 0], dtype=np.int32),
        edge_v=np.asarray([1, 1, 2], dtype=np.int32),
        edge_w=np.ones((3,), dtype=np.float64),
        name="duplicate_with_complete_edge_count",
    )

    tensor_graph = tensorize_graph(malformed, runtime, num_nodes=3)

    assert not tensor_graph.is_complete
    assert build_complete_uniform_tensor_graph(3, runtime).is_complete


def test_graph_canonicalization_rejects_negative_weights() -> None:
    graph = PairwiseFusionGraph(
        edge_u=np.asarray([0], dtype=np.int32),
        edge_v=np.asarray([1], dtype=np.int32),
        edge_w=np.asarray([-1.0], dtype=np.float64),
        name="negative",
    )

    with pytest.raises(ValueError, match="nonnegative"):
        coerce_graph(2, graph)


def test_host_graph_does_not_own_torch_cache() -> None:
    graph = build_complete_uniform_graph(3)

    assert not hasattr(graph, "torch_cache")
    assert graph.clear_torch_cache() is None


def test_explicit_cuda_request_does_not_silently_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_runtime("cuda", dtype="float64")

    assert resolve_runtime("auto", dtype="float64").device.type == "cpu"


def test_auto_dtype_is_float64_on_cpu_and_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_runtime("cpu", dtype="auto").dtype == torch.float64
    assert resolve_runtime("auto", dtype="auto").dtype == torch.float64

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    runtime = resolve_runtime("cuda", dtype="auto")

    assert runtime.device.type == "cuda"
    assert runtime.dtype == torch.float64


def test_float16_dtype_is_cuda_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Float16 runtime dtype"):
        resolve_runtime("cpu", dtype="float16")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    runtime = resolve_runtime("cuda", dtype="float16")

    assert runtime.device.type == "cuda"
    assert runtime.dtype == torch.float16


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for float16 graph weights"
)
def test_float16_complete_adaptive_graph_weights_stay_finite_on_cuda() -> None:
    runtime = resolve_runtime("cuda", dtype="float16")
    pilot_phi = torch.tensor(
        [
            [0.10, 0.20],
            [0.10, 0.20],
            [0.30, 0.45],
            [0.90, 0.80],
        ],
        dtype=torch.float16,
        device=runtime.device,
    )

    graph = build_complete_adaptive_tensor_graph(
        pilot_phi,
        runtime,
        gamma=1.0,
        tau=1e-6,
        baseline=1.0,
    )

    assert graph.weight.dtype == torch.float16
    assert bool(torch.isfinite(graph.weight).all().item())
    assert np.isclose(float(graph.weight.float().mean().item()), 1.0 / 3.0, rtol=5e-3)


def test_tensor_graph_ops_satisfy_adjoint_identity_and_dual_projection() -> None:
    graph = build_complete_uniform_graph(3)
    runtime = resolve_runtime("cpu", dtype="float64")
    tensor_graph = tensorize_graph(graph, runtime, num_nodes=3)
    phi = torch.tensor([[1.0, 0.0], [0.5, 2.0], [-1.0, 1.0]], dtype=runtime.dtype)
    dual = torch.tensor([[0.25, -0.5], [1.0, 0.0], [-0.25, 0.75]], dtype=runtime.dtype)

    expected_tau = (PDHG_PRECONDITIONER_ETA / tensor_graph.degree.clamp_min(1.0))[
        :, None
    ]
    assert torch.allclose(tensor_graph.pdhg_tau_node, expected_tau)

    forward = graph_forward(phi, tensor_graph)
    adjoint = graph_adjoint(dual, tensor_graph)

    assert torch.allclose(torch.sum(forward * dual), torch.sum(phi * adjoint))

    radius = torch.full((dual.shape[0],), 0.5, dtype=runtime.dtype)
    projected = project_dual_ball(dual, radius)
    assert torch.all(torch.linalg.norm(projected, dim=1) <= radius + 1e-12)
