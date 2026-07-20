from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import CliPP2.core.fusion.solver as solver_module
import CliPP2.model_selection.guided_fusion as guided_module
import CliPP2.runners.model_selection as model_selection_runner
from CliPP2.core.fusion.certificates import refine_graph_fusion_certificate
from CliPP2.core.fusion.quotient_workset import (
    QuotientCacheResourceError,
    aggregate_exact_quotient_problem,
    estimate_exact_quotient_cache_bytes,
    lifted_inner_objective,
    quotient_inner_objective,
)
from CliPP2.core.fusion.solver import prepare_torch_problem
from CliPP2.core.fusion.torch_backend import (
    graph_fusion_kkt_residual_from_grad_torch,
    project_stationarity_cone_torch,
)
from CliPP2.core.fusion.types import (
    CertificateOptions,
    CompressedEdgeCertificate,
    ExactSolverResourceLimit,
    PrimalOnlyWarmState,
    QuotientWorksetWarmState,
    TensorFusionGraph,
    TorchRuntime,
)
from CliPP2.core.model import FitOptions, fit_fixed_objective
from CliPP2.io.data import TumorData
from CliPP2.model_selection.guided_fusion import build_guided_fusion_initialization
from CliPP2.model_selection.scoring import _positive_exact_fusion_selection_mask
from CliPP2.runners import pipeline


def _complete_graph(num_nodes: int, *, dtype: torch.dtype) -> TensorFusionGraph:
    edge_index = torch.triu_indices(num_nodes, num_nodes, offset=1)
    weight = torch.linspace(0.3, 1.1, int(edge_index.shape[1]), dtype=dtype)
    degree = torch.zeros(num_nodes, dtype=dtype)
    degree.index_add_(0, edge_index[0], weight)
    degree.index_add_(0, edge_index[1], weight)
    return TensorFusionGraph(
        edge_index=edge_index,
        weight=weight,
        degree=degree,
        pdhg_tau_node=0.49 / degree.clamp_min(1e-12),
        num_nodes=num_nodes,
        is_complete=True,
        is_uniform=False,
        name="tracked_test_complete",
    )


def _toy_data() -> TumorData:
    alt = np.array([[3.0], [3.0], [7.0]], dtype=np.float64)
    total = np.array([[10.0], [10.0], [12.0]], dtype=np.float64)
    return TumorData(
        tumor_id="tracked-quotient-toy",
        mutation_ids=["m0", "m1", "m2"],
        region_ids=["r0"],
        alt_counts=alt,
        total_counts=total,
        purity=np.ones_like(alt),
        major_cn=np.ones_like(alt),
        minor_cn=np.ones_like(alt),
        normal_cn=np.full_like(alt, 2.0),
        has_cna=np.zeros_like(alt, dtype=bool),
        scaling=np.full_like(alt, 0.5),
        phi_upper=np.ones_like(alt),
        phi_init=np.full_like(alt, 0.5),
        init_major_mask=np.ones_like(alt, dtype=bool),
    )


class _NoDenseEdgeTensor(TorchDispatchMode):
    def __init__(self, shape: tuple[int, int]) -> None:
        super().__init__()
        self.shape = shape
        self.operations: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))

        def inspect(value) -> None:
            if torch.is_tensor(value) and tuple(value.shape) == self.shape:
                self.operations.append(str(func))
            elif isinstance(value, (tuple, list)):
                for item in value:
                    inspect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    inspect(item)

        inspect(output)
        return output


def test_exact_cone_projection_uses_exact_active_boundaries() -> None:
    phi = torch.tensor([[0.0, 0.5, 1.0, 0.3]], dtype=torch.float64)
    lower = torch.tensor([[0.0, 0.0, 0.0, 0.3]], dtype=torch.float64)
    upper = torch.tensor([[1.0, 1.0, 1.0, 0.3]], dtype=torch.float64)
    gradient = torch.tensor([[-2.0, 3.0, 4.0, -7.0]], dtype=torch.float64)

    projected = project_stationarity_cone_torch(
        gradient,
        phi=phi,
        lower=lower,
        upper=upper,
    )

    torch.testing.assert_close(
        projected,
        torch.tensor([[0.0, 0.0, 0.0, -7.0]], dtype=torch.float64),
    )


def test_exact_quotient_identity_and_cache_preflight() -> None:
    dtype = torch.float64
    graph = _complete_graph(5, dtype=dtype)
    labels = torch.tensor([0, 0, 1, 1, 2])
    h = torch.tensor(
        [[1.0, 2.0], [3.0, 1.5], [2.0, 4.0], [5.0, 1.0], [2.5, 3.0]],
        dtype=dtype,
    )
    center = torch.tensor(
        [[0.1, 0.8], [0.4, 0.5], [0.7, 0.2], [0.6, 0.4], [0.9, 0.1]],
        dtype=dtype,
    )
    lower = torch.zeros_like(h)
    upper = torch.ones_like(h)
    quotient = aggregate_exact_quotient_problem(
        h=h,
        U=center,
        lower=lower,
        upper=upper,
        labels=labels,
        graph=graph,
    )
    theta = torch.tensor([[0.3, 0.4], [0.6, 0.2], [0.8, 0.1]], dtype=dtype)
    lifted = theta.index_select(0, quotient.labels)

    torch.testing.assert_close(
        quotient_inner_objective(
            centers=theta,
            problem=quotient,
            lambda_value=0.37,
        ),
        lifted_inner_objective(
            phi=lifted,
            h=h,
            U=center,
            graph=graph,
            lambda_value=0.37,
        ),
        rtol=1e-13,
        atol=1e-13,
    )
    minimum = estimate_exact_quotient_cache_bytes(
        num_nodes=5,
        num_blocks=3,
        num_regions=2,
        value_dtype=dtype,
    )
    with pytest.raises(QuotientCacheResourceError):
        aggregate_exact_quotient_problem(
            h=h,
            U=center,
            lower=lower,
            upper=upper,
            labels=labels,
            graph=graph,
            max_cache_bytes=minimum - 1,
        )


def test_compressed_observed_refinement_matches_dense_without_dense_allocation() -> (
    None
):
    dtype = torch.float64
    graph = _complete_graph(4, dtype=dtype)
    phi = torch.tensor(
        [[0.1, 0.2], [0.3, 0.25], [0.5, 0.4], [0.7, 0.65]],
        dtype=dtype,
    )
    labels = torch.arange(4)
    certificate = CompressedEdgeCertificate(
        labels=labels,
        centers=phi,
        internal_edge_ids=torch.empty(0, dtype=torch.long),
        internal_dual=torch.empty((0, 2), dtype=dtype),
        graph_hash="tracked-no-dense",
        gradient_scope="mm_surrogate",
    )
    lambda_value = 0.4
    diff = phi.index_select(0, graph.edge_u) - phi.index_select(0, graph.edge_v)
    norm = torch.linalg.vector_norm(diff, dim=1)
    dense_dual = lambda_value * graph.weight[:, None] * diff / norm[:, None]
    adjoint = torch.zeros_like(phi)
    adjoint.index_add_(0, graph.edge_u, dense_dual)
    adjoint.index_add_(0, graph.edge_v, dense_dual, alpha=-1.0)
    expected = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=-adjoint,
        dual_kkt=dense_dual,
        lower=torch.zeros_like(phi),
        upper=torch.ones_like(phi),
        edge_u=graph.edge_u,
        edge_v=graph.edge_v,
        edge_w=graph.weight,
        lambda_value=lambda_value,
        atol=1e-9,
    )
    allocation_audit = _NoDenseEdgeTensor(
        (int(graph.edge_u.numel()), int(phi.shape[1]))
    )

    with allocation_audit:
        result = refine_graph_fusion_certificate(
            certificate=certificate,
            phi=phi,
            grad_smooth=-adjoint,
            gradient_scope="observed_objective",
            graph=graph,
            graph_hash="tracked-no-dense",
            lower=torch.zeros_like(phi),
            upper=torch.ones_like(phi),
            lambda_value=lambda_value,
            atol=1e-9,
            options=CertificateOptions(max_iter=32, max_expansions=2),
        )

    assert result.status == "certified"
    assert result.diagnostics.as_dict() == expected
    assert isinstance(result.certificate, CompressedEdgeCertificate)
    assert result.certificate.gradient_scope == "observed_objective"
    assert result.work_counters.workset_iterations == 0
    assert allocation_audit.operations == []


def test_guided_quotient_state_retains_sparse_flow_without_dense_dual() -> None:
    data = _toy_data()
    guide = np.array([[0.3], [0.3], [0.7]], dtype=np.float64)
    context = prepare_torch_problem(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-5,
        inner_max_iter=32,
        exact_pilot=guide,
        device="cpu",
        dtype="float64",
    )
    guided = build_guided_fusion_initialization(
        guide,
        np.array([0, 0, 1]),
        solver_context=context,
        materialize_dense_dual=False,
    )

    assert guided.solver_state.dual is None
    assert isinstance(guided.solver_state.warm_state, QuotientWorksetWarmState)
    assert tuple(guided.solver_state.warm_state.internal_dual.shape) == (1, 1)
    assert guided.solver_state.warm_state.graph_hash == context.graph_hash


def test_dense_guided_initialization_fails_before_dual_allocation(
    monkeypatch,
) -> None:
    data = _toy_data()
    guide = np.array([[0.3], [0.3], [0.7]], dtype=np.float64)
    context = prepare_torch_problem(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-5,
        inner_max_iter=32,
        exact_pilot=guide,
        device="cpu",
        dtype="float64",
    )
    monkeypatch.setattr(
        guided_module,
        "dense_complete_solver_memory_preflight",
        lambda **_kwargs: (False, 4096, 1024),
    )

    with pytest.raises(ExactSolverResourceLimit, match="dense guided"):
        build_guided_fusion_initialization(
            guide,
            np.array([0, 0, 1]),
            solver_context=context,
            materialize_dense_dual=True,
        )


def test_graph_construction_memory_error_uses_dense_cpu_fallback(monkeypatch) -> None:
    data = _toy_data()
    fake_gpu_runtime = TorchRuntime(
        device=torch.device("cuda"),
        device_name="cuda",
        dtype=torch.float64,
    )
    original_prepare = solver_module.prepare_torch_problem
    prepared_devices: list[str] = []

    def fail_device_graph_once(*args, **kwargs):
        runtime = kwargs["runtime"]
        prepared_devices.append(runtime.device_name)
        if len(prepared_devices) == 1:
            raise MemoryError("simulated device graph allocation failure")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        solver_module,
        "prepare_torch_problem",
        fail_device_graph_once,
    )
    artifacts = solver_module.fit_observed_data_pairwise_fusion(
        data,
        lambda_value=0.2,
        major_prior=0.5,
        eps=1e-6,
        outer_max_iter=2,
        inner_max_iter=40,
        tol=1e-5,
        exact_pilot=data.phi_init,
        runtime=fake_gpu_runtime,
        dense_fallback_policy="cpu_allowed",
        compute_summary=False,
    )

    assert prepared_devices == ["cuda", "cpu"]
    assert artifacts.inner_solver == "admm_complete_graph_cpu_fallback"
    assert artifacts.torch_result is not None
    assert artifacts.torch_result.phi_raw.device.type == "cpu"


def test_prepared_context_resource_policy_carries_cpu_fallback(monkeypatch) -> None:
    data = _toy_data()
    fake_gpu_runtime = TorchRuntime(
        device=torch.device("cuda"),
        device_name="cuda",
        dtype=torch.float64,
    )
    original_prepare = solver_module.prepare_torch_problem
    prepared_devices: list[str] = []

    def fail_device_graph_once(*args, **kwargs):
        runtime = kwargs["runtime"]
        prepared_devices.append(runtime.device_name)
        if len(prepared_devices) == 1:
            raise MemoryError("simulated model-selection graph allocation failure")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        solver_module,
        "prepare_torch_problem",
        fail_device_graph_once,
    )
    context = solver_module.prepare_torch_problem_with_resource_policy(
        data,
        dense_fallback_policy="cpu_allowed",
        major_prior=0.5,
        eps=1e-6,
        tol=1e-5,
        inner_max_iter=32,
        exact_pilot=data.phi_init,
        runtime=fake_gpu_runtime,
    )

    assert prepared_devices == ["cuda", "cpu"]
    assert context.runtime.device.type == "cpu"
    assert context.resource_fallback == "dense_cpu"


def test_likelihood_only_context_defers_complete_graph_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _toy_data()

    def unexpected_graph_build(*_args, **_kwargs):
        raise AssertionError("the deferred pilot must not materialize a graph")

    monkeypatch.setattr(
        solver_module,
        "build_complete_adaptive_tensor_graph",
        unexpected_graph_build,
    )
    context = solver_module.prepare_torch_problem(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-5,
        inner_max_iter=32,
        exact_pilot=data.phi_init,
        device="cpu",
        dtype="float64",
        defer_graph=True,
    )

    assert context.graph.name == "deferred_likelihood_pilot"
    assert context.graph.edge_u.numel() == 0
    assert context.graph_spec.edge_u.size == 0


def test_likelihood_only_context_rejects_a_resolved_graph() -> None:
    data = _toy_data()
    with pytest.raises(ValueError, match="defer_graph=True"):
        solver_module.prepare_torch_problem(
            data,
            major_prior=0.5,
            eps=1e-6,
            tol=1e-5,
            inner_max_iter=32,
            graph=solver_module.resolve_pairwise_fusion_graph(
                data.num_mutations,
                graph=None,
                pilot_phi=data.phi_init,
            ),
            exact_pilot=data.phi_init,
            device="cpu",
            dtype="float64",
            defer_graph=True,
        )


def test_prebuilt_tensor_graph_is_reused_without_reupload(monkeypatch) -> None:
    data = _toy_data()
    runtime = solver_module.resolve_runtime("cpu", dtype="float64")
    tensor_graph = solver_module.build_complete_adaptive_tensor_graph(
        torch.as_tensor(data.phi_init, dtype=runtime.dtype),
        runtime,
        gamma=1.25,
        tau=1e-5,
        baseline=0.75,
    )
    graph = solver_module.tensor_graph_to_pairwise_graph(tensor_graph)

    def unexpected_reupload(*_args, **_kwargs):
        raise AssertionError("the prebuilt graph must not be tensorized again")

    monkeypatch.setattr(solver_module, "tensorize_graph", unexpected_reupload)
    context = solver_module.prepare_torch_problem(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-5,
        inner_max_iter=32,
        graph=graph,
        prebuilt_tensor_graph=tensor_graph,
        exact_pilot=data.phi_init,
        device="cpu",
        dtype="float64",
    )

    assert context.graph is tensor_graph
    assert context.graph_hash == solver_module._graph_fingerprint(context.graph_spec)
    np.testing.assert_array_equal(context.graph_spec.edge_u, graph.edge_u)
    np.testing.assert_array_equal(context.graph_spec.edge_v, graph.edge_v)
    np.testing.assert_array_equal(context.graph_spec.edge_w, graph.edge_w)


def test_prebuilt_tensor_graph_cannot_claim_a_different_host_graph() -> None:
    data = _toy_data()
    runtime = solver_module.resolve_runtime("cpu", dtype="float64")
    tensor_graph = solver_module.build_complete_adaptive_tensor_graph(
        torch.as_tensor(data.phi_init, dtype=runtime.dtype),
        runtime,
    )
    graph = solver_module.tensor_graph_to_pairwise_graph(tensor_graph)
    mismatched_weight = np.asarray(graph.edge_w).copy()
    mismatched_weight[0] = np.nextafter(mismatched_weight[0], np.inf)

    with pytest.raises(ValueError, match="weights do not match graph"):
        solver_module.prepare_torch_problem(
            data,
            major_prior=0.5,
            eps=1e-6,
            tol=1e-5,
            inner_max_iter=32,
            graph=replace(graph, edge_w=mismatched_weight),
            prebuilt_tensor_graph=tensor_graph,
            exact_pilot=data.phi_init,
            device="cpu",
            dtype="float64",
        )


def _guided_resource_policy_context():
    data = _toy_data()
    guide = np.array([[0.3], [0.3], [0.7]], dtype=np.float64)
    cpu_context = prepare_torch_problem(
        data,
        major_prior=0.5,
        eps=1e-6,
        tol=1e-5,
        inner_max_iter=32,
        exact_pilot=guide,
        pooled_start=guide,
        scalar_well_starts=(),
        device="cpu",
        dtype="float64",
    )
    device_context = replace(
        cpu_context,
        runtime=TorchRuntime(
            device=torch.device("cuda"),
            device_name="cuda",
            dtype=torch.float64,
        ),
    )
    return data, guide, cpu_context, device_context


@pytest.mark.parametrize("failure_type", [MemoryError, torch.OutOfMemoryError])
def test_guided_initialization_resource_policy_retries_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
    failure_type: type[BaseException],
) -> None:
    data, guide, cpu_context, device_context = _guided_resource_policy_context()
    original_build = model_selection_runner.build_guided_fusion_initialization
    attempted_devices: list[str] = []

    def fail_device_build_once(phi, labels, **kwargs):
        context = kwargs["solver_context"]
        attempted_devices.append(context.runtime.device_name)
        if context.runtime.device.type != "cpu":
            raise failure_type("simulated guided device allocation failure")
        return original_build(phi, labels, **kwargs)

    monkeypatch.setattr(
        model_selection_runner,
        "build_guided_fusion_initialization",
        fail_device_build_once,
    )
    options = FitOptions(
        lambda_value=0.0,
        inner_max_iter=32,
        tol=1e-5,
        major_prior=0.5,
        eps=1e-6,
        graph=cpu_context.graph_spec,
        device="cuda",
        dtype="float64",
        dense_fallback_policy="cpu_allowed",
        materialize_full_dual=False,
    )

    guided, returned_context, returned_guide = (
        model_selection_runner._build_guided_initialization_with_resource_policy(
            data=data,
            guide_phi=guide,
            guide_labels=np.array([0, 0, 1], dtype=np.int64),
            solver_context=device_context,
            fit_options=options,
        )
    )

    assert attempted_devices == ["cuda", "cpu"]
    assert returned_context.runtime.device.type == "cpu"
    assert returned_context.resource_fallback == "dense_cpu"
    assert returned_context.graph_hash == cpu_context.graph_hash
    assert returned_context.objective_spec_hash == cpu_context.objective_spec_hash
    assert isinstance(returned_guide, np.ndarray)
    assert guided.solver_state.phi.device.type == "cpu"


@pytest.mark.parametrize("policy", ["device_only", "error"])
@pytest.mark.parametrize("failure_type", [MemoryError, torch.OutOfMemoryError])
def test_guided_initialization_resource_policy_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    failure_type: type[BaseException],
) -> None:
    data, guide, cpu_context, device_context = _guided_resource_policy_context()
    attempted_devices: list[str] = []

    def fail_guided_build(_phi, _labels, **kwargs):
        attempted_devices.append(kwargs["solver_context"].runtime.device_name)
        raise failure_type("simulated guided allocation failure")

    monkeypatch.setattr(
        model_selection_runner,
        "build_guided_fusion_initialization",
        fail_guided_build,
    )
    options = FitOptions(
        lambda_value=0.0,
        graph=cpu_context.graph_spec,
        device="cuda",
        dtype="float64",
        dense_fallback_policy=policy,
        materialize_full_dual=False,
    )

    with pytest.raises(
        ExactSolverResourceLimit,
        match="guided initialization exhausted memory on cuda",
    ) as error:
        model_selection_runner._build_guided_initialization_with_resource_policy(
            data=data,
            guide_phi=guide,
            guide_labels=np.array([0, 0, 1], dtype=np.int64),
            solver_context=device_context,
            fit_options=options,
        )

    assert attempted_devices == ["cuda"]
    assert isinstance(error.value.__cause__, failure_type)


def test_quotient_fit_is_eligible_only_with_observed_full_graph_certificate() -> None:
    data = _toy_data()
    common = dict(
        lambda_value=1.0,
        outer_max_iter=4,
        inner_max_iter=80,
        tol=1e-5,
        device="cpu",
        dtype="float64",
        certificate_max_iter=1000,
        workset_max_expansions=8,
    )
    fit = fit_fixed_objective(
        data,
        FitOptions(inner_backend="quotient_workset", **common),
        compute_summary=False,
    )
    dense = fit_fixed_objective(
        data,
        FitOptions(inner_backend="dense", **common),
        compute_summary=False,
    )

    assert fit.inner_backend == "quotient_workset_complete_graph"
    assert fit.admm_iterations == 0
    assert fit.certificate_scope == "full_original_graph"
    assert fit.certificate_gradient_scope == "observed_objective"
    assert fit.full_kkt_certified
    assert fit.selection_eligible
    np.testing.assert_allclose(fit.phi, dense.phi, rtol=0.0, atol=1e-7)
    assert fit.penalized_objective == pytest.approx(
        dense.penalized_objective,
        rel=1e-12,
        abs=1e-12,
    )

    row = {
        "bic_selection_eligible": True,
        "raw_kkt_eligible": True,
        "candidate_pool_source": "raw_fused_lambda_path",
        "lambda": 1.0,
        "exactness_provenance_version": 1,
        "estimator_role": "raw_fused_lambda_path",
        "objective_faithful": True,
        "objective_spec_hash": "objective",
        "original_graph_hash": "graph",
        "certificate_problem_hash": "problem",
        "certificate_scope": "full_original_graph",
        "certificate_gradient_scope": "observed_objective",
        "full_kkt_certified": True,
        "full_kkt_certificate_status": "not_certified",
        "fixed_objective_kkt_residual": 1e-7,
        "full_kkt_tolerance": 1e-5,
    }
    assert not _positive_exact_fusion_selection_mask(pd.DataFrame([row]))[0]


def test_damped_state_is_primal_only_and_resource_batch_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    labels = torch.zeros(2, dtype=torch.long)
    trial = QuotientWorksetWarmState(
        phi=torch.full((2, 1), 0.2),
        labels=labels,
        centers=torch.full((1, 1), 0.2),
        quotient_dual=torch.full((1, 1), 0.7),
        internal_edge_ids=torch.tensor([0]),
        internal_dual=torch.full((1, 1), 0.3),
        graph_hash="graph",
        previous_lambda=1.0,
    )
    dual, dual_kkt, certificate, warm, dual_is_actual = (
        solver_module._invalidate_damped_trial_state(
            phi=torch.full((2, 1), 0.4),
            trial_warm_state=trial,
        )
    )
    assert dual is dual_kkt is certificate is None
    assert not dual_is_actual
    assert isinstance(warm, PrimalOnlyWarmState)
    assert warm.structure_hint_is_heuristic

    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    (input_dir / "limited.tsv").touch()
    (input_dir / "ok.tsv").touch()

    def fake_process(*, file_path, **_kwargs):
        path = Path(file_path)
        if path.stem == "limited":
            raise ExactSolverResourceLimit("exact_solver_resource_limit: test")
        return {"tumor_id": "ok", "selection_eligible": True}

    monkeypatch.setattr(pipeline, "process_one_file", fake_process)
    summary = pipeline.run_directory(
        input_dir,
        output_dir,
        workers=1,
        write_outputs=False,
    )
    limited = summary.loc[summary["tumor_id"] == "limited"].iloc[0]
    assert not bool(limited["selection_eligible"])
    assert limited["failure_reason"] == "exact_solver_resource_limit"
