from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

import CliPP2.core.fusion.quotient_workset as quotient_module
from CliPP2.core.fusion.solver import prepare_torch_problem
from CliPP2.core.fusion.quotient_workset import (
    QuotientCacheResourceError,
    aggregate_exact_quotient_problem,
    estimate_exact_quotient_cache_bytes,
    heuristic_residual_split,
    lifted_inner_objective,
    quotient_inner_objective,
    solve_majorized_subproblem_quotient_workset_torch,
)
from CliPP2.core.fusion.types import (
    CertificateOptions,
    ExactSolverResourceLimit,
    QuotientWorksetWarmState,
    TensorFusionGraph,
    TorchRuntime,
)
from CliPP2.core.model import FitOptions, fit_fixed_objective
from CliPP2.io.data import TumorData


def _complete_graph(num_nodes: int, *, dtype: torch.dtype) -> TensorFusionGraph:
    pairs = [(u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes)]
    edge_index = torch.tensor(pairs, dtype=torch.long).T.contiguous()
    weight = torch.linspace(0.2, 1.2, len(pairs), dtype=dtype)
    degree = torch.zeros(num_nodes, dtype=dtype)
    degree.index_add_(0, edge_index[0], weight)
    degree.index_add_(0, edge_index[1], weight)
    return TensorFusionGraph(
        edge_index=edge_index,
        weight=weight,
        degree=degree,
        pdhg_tau_node=0.49 / degree,
        num_nodes=num_nodes,
        is_complete=True,
        is_uniform=False,
        name="complete_test",
    )


def test_exact_quotient_preserves_lifted_objective_and_boxes() -> None:
    dtype = torch.float64
    graph = _complete_graph(5, dtype=dtype)
    labels = torch.tensor([7, 7, 2, 2, 9], dtype=torch.long)
    h = torch.tensor(
        [[1.0, 2.0], [3.0, 1.5], [2.0, 4.0], [5.0, 1.0], [2.5, 3.0]],
        dtype=dtype,
    )
    U = torch.tensor(
        [[0.1, 0.8], [0.4, 0.5], [0.7, 0.2], [0.6, 0.4], [0.9, 0.1]],
        dtype=dtype,
    )
    lower = torch.tensor(
        [[0.05, 0.2], [0.1, 0.3], [0.2, 0.1], [0.3, 0.15], [0.4, 0.0]],
        dtype=dtype,
    )
    upper = torch.tensor(
        [[0.8, 0.95], [0.9, 0.9], [0.85, 0.8], [0.8, 0.75], [1.0, 0.7]],
        dtype=dtype,
    )
    quotient = aggregate_exact_quotient_problem(
        h=h,
        U=U,
        lower=lower,
        upper=upper,
        labels=labels,
        graph=graph,
    )
    centers = torch.tensor([[0.5, 0.4], [0.3, 0.6], [0.7, 0.2]], dtype=dtype)
    lifted = centers.index_select(0, quotient.labels)

    original = lifted_inner_objective(
        phi=lifted,
        h=h,
        U=U,
        graph=graph,
        lambda_value=0.37,
    )
    compressed = quotient_inner_objective(
        centers=centers,
        problem=quotient,
        lambda_value=0.37,
    )

    torch.testing.assert_close(compressed, original, rtol=1e-13, atol=1e-13)
    assert quotient.common_box_feasible
    for block in range(quotient.num_blocks):
        members = quotient.labels == block
        torch.testing.assert_close(
            quotient.lower[block], torch.max(lower[members], dim=0).values
        )
        torch.testing.assert_close(
            quotient.upper[block], torch.min(upper[members], dim=0).values
        )


def test_exact_quotient_cache_budget_fails_before_large_cache_allocation() -> None:
    dtype = torch.float64
    graph = _complete_graph(5, dtype=dtype)
    values = torch.ones((5, 2), dtype=dtype)
    labels = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
    minimum = estimate_exact_quotient_cache_bytes(
        num_nodes=5,
        num_blocks=3,
        num_regions=2,
        value_dtype=dtype,
    )
    quotient_edges = 3
    quotient_edge_state_bytes = (
        quotient_edges * 2 * torch.empty((), dtype=dtype).element_size()
    )
    assert minimum >= 11 * quotient_edge_state_bytes

    with pytest.raises(
        QuotientCacheResourceError,
        match="exact_quotient_cache_resource_limit",
    ):
        aggregate_exact_quotient_problem(
            h=values,
            U=torch.zeros_like(values),
            lower=torch.zeros_like(values),
            upper=values,
            labels=labels,
            graph=graph,
            max_cache_bytes=minimum - 1,
        )


def test_heuristic_split_is_deterministic_and_strictly_refines_structure() -> None:
    labels = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    signal = torch.tensor([[0.0], [10.0], [0.0], [2.0], [2.0]])

    proposed = heuristic_residual_split(labels=labels, residual_signal=signal)

    assert proposed is not None
    assert proposed.tolist() == [0, 1, 0, 2, 2]
    assert int(torch.unique(proposed).numel()) == int(torch.unique(labels).numel()) + 1
    assert (
        heuristic_residual_split(
            labels=torch.arange(3),
            residual_signal=torch.zeros((3, 1)),
        )
        is None
    )


def test_compatible_warm_certificate_skips_quotient_and_workset_iterations() -> None:
    dtype = torch.float64
    graph = _complete_graph(3, dtype=dtype)
    labels = torch.tensor([0, 0, 1], dtype=torch.long)
    centers = torch.tensor([[0.3], [0.7]], dtype=dtype)
    phi = centers.index_select(0, labels)
    lambda_value = 0.5

    # The two inter-block analytic edge multipliers exactly cancel this smooth
    # gradient; the sole internal edge therefore needs a zero multiplier.
    grad = (
        lambda_value
        * torch.stack(
            [
                graph.weight[1],
                graph.weight[2],
                -(graph.weight[1] + graph.weight[2]),
            ]
        )[:, None]
    )
    h = torch.ones_like(phi)
    U = phi - grad
    warm = QuotientWorksetWarmState(
        phi=phi,
        labels=labels,
        centers=centers,
        quotient_dual=None,
        internal_edge_ids=torch.tensor([0], dtype=torch.long),
        internal_dual=torch.zeros((1, 1), dtype=dtype),
        graph_hash="warm-fast-path",
        previous_lambda=lambda_value,
    )

    attempt = solve_majorized_subproblem_quotient_workset_torch(
        runtime=TorchRuntime(
            device=torch.device("cpu"), device_name="cpu", dtype=dtype
        ),
        U=U,
        h=h,
        lower=torch.zeros_like(phi),
        upper=torch.ones_like(phi),
        lambda_value=lambda_value,
        graph=graph,
        graph_hash="warm-fast-path",
        tol=1e-8,
        max_iter=80,
        phi_start=phi,
        warm_state=warm,
        certificate_options=CertificateOptions(),
        partition_tolerance=1e-8,
    )

    result = attempt.certified_result
    work = attempt.work_counters
    assert attempt.status == "certified"
    assert result is not None
    torch.testing.assert_close(result.phi, phi, rtol=0.0, atol=0.0)
    assert result.surrogate_kkt.kkt_residual <= 5e-8
    assert result.inner_iterations == 0
    assert work.quotient_iterations == 0
    assert work.workset_iterations == 0
    assert work.dense_iterations == 0
    assert work.streamed_edge_passes == 2


def test_unconverged_quotient_returns_no_worse_seed_without_workset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dtype = torch.float64
    graph = _complete_graph(3, dtype=dtype)
    phi_start = torch.tensor([[0.2], [0.2], [0.9]], dtype=dtype)
    U = torch.tensor([[0.3], [0.3], [0.7]], dtype=dtype)

    def unconverged_quotient(**kwargs):
        centers = kwargs["U"].clone()
        dual = torch.zeros(
            (int(kwargs["edge_u"].numel()), 1),
            dtype=dtype,
        )
        return centers, dual, dual, 3, False, 1.0

    monkeypatch.setattr(
        quotient_module,
        "solve_majorized_subproblem_alm_torch",
        unconverged_quotient,
    )
    monkeypatch.setattr(
        quotient_module,
        "refine_graph_fusion_certificate",
        lambda **_kwargs: pytest.fail("workset refinement must be convergence-gated"),
    )

    attempt = solve_majorized_subproblem_quotient_workset_torch(
        runtime=TorchRuntime(
            device=torch.device("cpu"), device_name="cpu", dtype=dtype
        ),
        U=U,
        h=torch.ones_like(U),
        lower=torch.zeros_like(U),
        upper=torch.ones_like(U),
        lambda_value=0.01,
        graph=graph,
        graph_hash="unconverged-seed",
        tol=1e-8,
        max_iter=20,
        phi_start=phi_start,
        warm_state=None,
        certificate_options=CertificateOptions(),
        partition_tolerance=1e-8,
    )

    assert attempt.status == "quotient_unconverged"
    torch.testing.assert_close(attempt.phi_candidate, U)
    assert attempt.certified_result is None
    assert attempt.work_counters.quotient_iterations == 3
    assert attempt.exact_inner_objective <= float(
        lifted_inner_objective(
            phi=phi_start,
            h=torch.ones_like(U),
            U=U,
            graph=graph,
            lambda_value=0.01,
        ).item()
    )


def test_opt_in_quotient_backend_returns_observed_full_graph_certificate() -> None:
    alt = np.array([[3.0], [3.0], [7.0]], dtype=np.float64)
    total = np.array([[10.0], [10.0], [12.0]], dtype=np.float64)
    data = TumorData(
        tumor_id="quotient-toy",
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

    fit = fit_fixed_objective(
        data,
        FitOptions(
            lambda_value=1.0,
            outer_max_iter=4,
            inner_max_iter=80,
            tol=1e-5,
            device="cpu",
            dtype="float64",
            inner_backend="quotient_workset",
            certificate_max_iter=1000,
            workset_max_expansions=8,
        ),
        compute_summary=False,
    )

    assert fit.inner_solver == "quotient_workset_complete_graph"
    assert fit.inner_backend == "quotient_workset_complete_graph"
    assert fit.admm_iterations == 0
    assert fit.quotient_iterations > 0
    # A fresh quotient proposal first passes the cheap retained-edge and omitted-
    # column gates; only then does it pay for the authoritative full-graph audit.
    assert fit.workset_iterations == 1
    assert fit.workset_expansions == 0
    assert fit.streamed_edge_passes > 0
    assert fit.dense_iterations == 0
    assert fit.fallback_reason == ""
    assert fit.solver_state is not None
    assert fit.solver_state.dual is None
    assert fit.solver_state.certificate is not None
    assert fit.certificate_scope == "full_original_graph"
    assert fit.certificate_gradient_scope == "observed_objective"
    assert fit.full_kkt_certified
    assert fit.fixed_objective_kkt_residual <= fit.full_kkt_tolerance

    dense = fit_fixed_objective(
        data,
        replace(
            FitOptions(
                lambda_value=1.0,
                outer_max_iter=4,
                inner_max_iter=80,
                tol=1e-5,
                device="cpu",
                dtype="float64",
            ),
            inner_backend="dense",
        ),
        compute_summary=False,
    )
    np.testing.assert_allclose(fit.phi, dense.phi, rtol=0.0, atol=1e-7)
    assert fit.penalized_objective == pytest.approx(
        dense.penalized_objective,
        rel=1e-12,
        abs=1e-12,
    )
    assert dense.selection_eligible


def test_resource_limit_is_explicit_when_workset_and_dense_fallback_do_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alt = np.array([[3.0], [3.0], [7.0]], dtype=np.float64)
    total = np.array([[10.0], [10.0], [12.0]], dtype=np.float64)
    data = TumorData(
        tumor_id="quotient-resource-limit",
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
    options = FitOptions(
        lambda_value=1.0,
        outer_max_iter=1,
        inner_max_iter=20,
        tol=1e-5,
        device="cpu",
        dtype="float64",
        inner_backend="quotient_workset",
        workset_max_bytes=1,
        dense_fallback_policy="auto",
    )
    context = prepare_torch_problem(
        data,
        major_prior=options.major_prior,
        eps=options.eps,
        tol=options.tol,
        inner_max_iter=options.inner_max_iter,
        device=options.device,
        dtype=options.dtype,
    )
    with pytest.raises(ExactSolverResourceLimit, match="dense fallback is disabled"):
        fit_fixed_objective(
            data,
            replace(
                options,
                inner_backend="quotient_workset",
                dense_fallback_policy="error",
            ),
            phi_start=np.full_like(alt, 0.5),
            solver_context=context,
            compute_summary=False,
        )

    fallback_start = np.array([[0.5], [0.5], [0.7]], dtype=np.float64)
    dense_fallback = fit_fixed_objective(
        data,
        options,
        phi_start=fallback_start,
        solver_context=context,
        start_mode="warm_only",
        compute_summary=False,
    )
    assert dense_fallback.quotient_iterations > 0
    assert dense_fallback.dense_iterations > 0
    assert dense_fallback.admm_iterations == dense_fallback.dense_iterations
    assert dense_fallback.inner_iterations > dense_fallback.admm_iterations
    assert "dense_current_device_after_resource_limit" in dense_fallback.fallback_reason

    continued = fit_fixed_objective(
        data,
        options,
        solver_context=context,
        solver_state=dense_fallback.solver_state,
        start_mode="warm_only",
        compute_summary=False,
    )
    assert continued.quotient_iterations == 0
    assert (
        "dense_current_device_after_prior_quotient_failure" in continued.fallback_reason
    )

    monkeypatch.setenv("CLIPP2_MAX_COMPLETE_GRAPH_BYTES", "1")

    with pytest.raises(MemoryError, match="exact_solver_resource_limit"):
        fit_fixed_objective(
            data,
            options,
            phi_start=np.full_like(alt, 0.5),
            solver_context=context,
            compute_summary=False,
        )
