from __future__ import annotations

import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from CliPP2.core.fusion.certificates import (
    _initial_internal_tree_ids,
    audit_graph_fusion_certificate,
    refine_graph_fusion_certificate,
)
from CliPP2.core.fusion.torch_backend import (
    graph_fusion_kkt_residual_from_grad_torch,
)
from CliPP2.core.fusion.types import (
    CertificateOptions,
    CompressedEdgeCertificate,
    TensorFusionGraph,
)


def _complete_graph(num_nodes: int, *, dtype: torch.dtype) -> TensorFusionGraph:
    pairs = [(u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes)]
    edge_index = torch.tensor(pairs, dtype=torch.long).T.contiguous()
    weight = torch.linspace(0.3, 1.1, len(pairs), dtype=dtype)
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
        name="test_complete",
    )


def _materialize_compressed_dual(
    *,
    certificate: CompressedEdgeCertificate,
    graph: TensorFusionGraph,
    phi: torch.Tensor,
    lambda_value: float,
) -> torch.Tensor:
    diff = phi.index_select(0, graph.edge_u) - phi.index_select(0, graph.edge_v)
    diff_norm = torch.linalg.vector_norm(diff, dim=1)
    dual = torch.zeros_like(diff)
    between = certificate.labels.index_select(0, graph.edge_u) != (
        certificate.labels.index_select(0, graph.edge_v)
    )
    active = between & (diff_norm > 0.0)
    radius = float(lambda_value) * graph.weight
    dual[active] = radius[active, None] * diff[active] / diff_norm[active, None]
    dual.index_copy_(0, certificate.internal_edge_ids, certificate.internal_dual)
    return dual


def test_complete_graph_initial_workset_is_a_balanced_tree_per_block() -> None:
    graph = _complete_graph(11, dtype=torch.float64)
    labels = torch.tensor([0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1], dtype=torch.long)

    edge_ids, edge_passes = _initial_internal_tree_ids(
        labels=labels,
        graph=graph,
        dtype=torch.float64,
    )
    actual_pairs = set(
        zip(
            graph.edge_u.index_select(0, edge_ids).tolist(),
            graph.edge_v.index_select(0, edge_ids).tolist(),
        )
    )

    assert actual_pairs == {
        (0, 2),
        (0, 3),
        (2, 5),
        (2, 6),
        (3, 7),
        (3, 8),
        (1, 4),
        (1, 9),
        (4, 10),
    }
    degree = torch.zeros(11, dtype=torch.long)
    ones = torch.ones(9, dtype=torch.long)
    degree.index_add_(0, graph.edge_u.index_select(0, edge_ids), ones)
    degree.index_add_(0, graph.edge_v.index_select(0, edge_ids), ones)
    assert int(torch.max(degree).item()) <= 3
    assert edge_passes == 0


class _NoFullEdgeRegionTensor(TorchDispatchMode):
    def __init__(self, *, edge_shape: tuple[int, int]) -> None:
        super().__init__()
        self.edge_shape = edge_shape
        self.operations: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))

        def inspect(value) -> None:
            if torch.is_tensor(value):
                if tuple(value.shape) == self.edge_shape:
                    self.operations.append(str(func))
                return
            if isinstance(value, (tuple, list)):
                for item in value:
                    inspect(item)
            elif isinstance(value, dict):
                for item in value.values():
                    inspect(item)

        inspect(output)
        return output


def test_compressed_audit_matches_materialized_dense_certificate() -> None:
    dtype = torch.float64
    graph = _complete_graph(5, dtype=dtype)
    labels = torch.tensor([0, 0, 0, 1, 2], dtype=torch.long)
    centers = torch.tensor([[0.2, 0.4], [0.8, 0.6], [0.5, 0.1]], dtype=dtype)
    phi = centers.index_select(0, labels)
    support_ids = torch.tensor([0, 4], dtype=torch.long)
    support_dual = torch.tensor([[0.02, -0.03], [-0.01, 0.015]], dtype=dtype)
    certificate = CompressedEdgeCertificate(
        labels=labels,
        centers=centers,
        internal_edge_ids=support_ids,
        internal_dual=support_dual,
        graph_hash="graph-v1",
        gradient_scope="observed_objective",
    )
    grad = torch.tensor(
        [[0.7, -0.3], [-0.5, 0.2], [0.1, 0.4], [-0.2, -0.1], [0.3, 0.6]],
        dtype=dtype,
    )
    lower = torch.zeros_like(phi)
    upper = torch.ones_like(phi)
    lambda_value = 0.9
    full_dual = _materialize_compressed_dual(
        certificate=certificate,
        graph=graph,
        phi=phi,
        lambda_value=lambda_value,
    )

    expected = graph_fusion_kkt_residual_from_grad_torch(
        phi=phi,
        grad_smooth=grad,
        dual_kkt=full_dual,
        lower=lower,
        upper=upper,
        edge_u=graph.edge_u,
        edge_v=graph.edge_v,
        edge_w=graph.weight,
        lambda_value=lambda_value,
        atol=1e-8,
    )
    actual = audit_graph_fusion_certificate(
        certificate=certificate,
        phi=phi,
        grad_smooth=grad,
        graph=graph,
        graph_hash="graph-v1",
        lower=lower,
        upper=upper,
        lambda_value=lambda_value,
        atol=1e-8,
    ).as_dict()

    assert actual.keys() == expected.keys()
    for key, expected_value in expected.items():
        assert actual[key] == expected_value, key


def test_compressed_audit_rejects_stale_primal_and_graph() -> None:
    dtype = torch.float64
    graph = _complete_graph(3, dtype=dtype)
    labels = torch.tensor([0, 0, 1], dtype=torch.long)
    centers = torch.tensor([[0.2], [0.8]], dtype=dtype)
    certificate = CompressedEdgeCertificate(
        labels=labels,
        centers=centers,
        internal_edge_ids=torch.empty(0, dtype=torch.long),
        internal_dual=torch.empty((0, 1), dtype=dtype),
        graph_hash="expected",
        gradient_scope="observed_objective",
    )
    common = dict(
        certificate=certificate,
        grad_smooth=torch.zeros((3, 1), dtype=dtype),
        graph=graph,
        lower=torch.zeros((3, 1), dtype=dtype),
        upper=torch.ones((3, 1), dtype=dtype),
        lambda_value=1.0,
        atol=1e-8,
    )

    try:
        audit_graph_fusion_certificate(
            **common,
            phi=centers.index_select(0, labels),
            graph_hash="wrong",
        )
    except ValueError as exc:
        assert "graph hash" in str(exc)
    else:
        raise AssertionError("Expected graph-hash mismatch to fail closed.")

    stale = centers.index_select(0, labels).clone()
    stale[0, 0] += 1e-12
    try:
        audit_graph_fusion_certificate(
            **common,
            phi=stale,
            graph_hash="expected",
        )
    except ValueError as exc:
        assert "stale" in str(exc)
    else:
        raise AssertionError("Expected stale compressed primal to fail closed.")


def test_internal_workset_expands_and_certifies_full_graph() -> None:
    dtype = torch.float64
    graph = _complete_graph(3, dtype=dtype)
    labels = torch.zeros(3, dtype=torch.long)
    centers = torch.tensor([[0.5]], dtype=dtype)
    phi = centers.index_select(0, labels)
    initial = CompressedEdgeCertificate(
        labels=labels,
        centers=centers,
        internal_edge_ids=torch.empty(0, dtype=torch.long),
        internal_dual=torch.empty((0, 1), dtype=dtype),
        graph_hash="expand",
        gradient_scope="mm_surrogate",
    )

    result = refine_graph_fusion_certificate(
        certificate=initial,
        phi=phi,
        grad_smooth=torch.tensor([[0.0], [1.0], [-1.0]], dtype=dtype),
        gradient_scope="observed_objective",
        graph=graph,
        graph_hash="expand",
        lower=torch.zeros_like(phi),
        upper=torch.ones_like(phi),
        lambda_value=1.0,
        atol=1e-7,
        options=CertificateOptions(
            max_iter=2000,
            refinement_rounds=2,
            max_expansions=4,
            add_batch=1,
            mapping_tolerance=1e-9,
            column_tolerance=1e-9,
        ),
    )

    assert result.status == "certified"
    assert result.diagnostics.kkt_residual <= 5e-7
    assert isinstance(result.certificate, CompressedEdgeCertificate)
    assert result.certificate.gradient_scope == "observed_objective"
    assert 2 in result.certificate.internal_edge_ids.tolist()
    assert result.work_counters.workset_expansions >= 1
    assert result.fused_edges == 3
    assert result.nonzero_edges == 0


def test_unconverged_workset_refinement_fails_closed_after_three_attempts() -> None:
    dtype = torch.float64
    graph = _complete_graph(8, dtype=dtype)
    labels = torch.zeros(8, dtype=torch.long)
    centers = torch.tensor([[0.5]], dtype=dtype)
    phi = centers.index_select(0, labels)
    initial = CompressedEdgeCertificate(
        labels=labels,
        centers=centers,
        internal_edge_ids=torch.empty(0, dtype=torch.long),
        internal_dual=torch.empty((0, 1), dtype=dtype),
        graph_hash="bounded-unconverged-workset",
        gradient_scope="mm_surrogate",
    )

    result = refine_graph_fusion_certificate(
        certificate=initial,
        phi=phi,
        grad_smooth=torch.arange(8, dtype=dtype).sub(3.5)[:, None],
        gradient_scope="observed_objective",
        graph=graph,
        graph_hash="bounded-unconverged-workset",
        lower=torch.zeros_like(phi),
        upper=torch.ones_like(phi),
        lambda_value=100.0,
        atol=1e-12,
        options=CertificateOptions(
            max_iter=1,
            max_expansions=10,
            add_batch=1,
            mapping_tolerance=1e-30,
            column_tolerance=1e-30,
        ),
    )

    assert result.status == "workset_incomplete"
    assert result.diagnostics.kkt_residual > 5e-12
    assert result.work_counters.workset_iterations == 3
    assert result.work_counters.workset_expansions == 2


def test_mm_certificate_is_not_reused_as_observed_objective_authority() -> None:
    dtype = torch.float64
    graph = _complete_graph(3, dtype=dtype)
    labels = torch.zeros(3, dtype=torch.long)
    centers = torch.tensor([[0.5]], dtype=dtype)
    phi = centers.index_select(0, labels)
    all_edge_ids = torch.arange(int(graph.edge_u.numel()), dtype=torch.long)
    initial = CompressedEdgeCertificate(
        labels=labels,
        centers=centers,
        internal_edge_ids=all_edge_ids,
        internal_dual=torch.zeros((int(all_edge_ids.numel()), 1), dtype=dtype),
        graph_hash="gradient-scope",
        gradient_scope="mm_surrogate",
    )
    common = dict(
        phi=phi,
        graph=graph,
        graph_hash="gradient-scope",
        lower=torch.zeros_like(phi),
        upper=torch.ones_like(phi),
        lambda_value=1.0,
        atol=1e-8,
        options=CertificateOptions(max_iter=64, max_expansions=2),
    )

    mm_result = refine_graph_fusion_certificate(
        certificate=initial,
        grad_smooth=torch.zeros_like(phi),
        gradient_scope="mm_surrogate",
        **common,
    )
    assert mm_result.status == "certified"
    assert isinstance(mm_result.certificate, CompressedEdgeCertificate)
    assert mm_result.certificate.gradient_scope == "mm_surrogate"

    observed_result = refine_graph_fusion_certificate(
        certificate=mm_result.certificate,
        grad_smooth=torch.ones_like(phi),
        gradient_scope="observed_objective",
        **common,
    )
    assert observed_result.status == "not_certified"
    assert math.isfinite(observed_result.diagnostics.kkt_residual)
    assert observed_result.diagnostics.kkt_residual > 5e-8
    assert isinstance(observed_result.certificate, CompressedEdgeCertificate)
    assert observed_result.certificate.gradient_scope == "observed_objective"


def test_periodic_and_final_style_refinement_never_allocates_dense_edge_dual() -> None:
    dtype = torch.float64
    graph = _complete_graph(4, dtype=dtype)
    phi = torch.tensor(
        [[0.1, 0.2], [0.3, 0.25], [0.5, 0.4], [0.7, 0.65]],
        dtype=dtype,
    )
    labels = torch.arange(4, dtype=torch.long)
    certificate = CompressedEdgeCertificate(
        labels=labels,
        centers=phi,
        internal_edge_ids=torch.empty(0, dtype=torch.long),
        internal_dual=torch.empty((0, 2), dtype=dtype),
        graph_hash="no-dense-audit",
        gradient_scope="mm_surrogate",
    )
    lambda_value = 0.4
    analytic_dual = _materialize_compressed_dual(
        certificate=certificate,
        graph=graph,
        phi=phi,
        lambda_value=lambda_value,
    )
    adjoint = torch.zeros_like(phi)
    adjoint.index_add_(0, graph.edge_u, analytic_dual)
    adjoint.index_add_(0, graph.edge_v, analytic_dual, alpha=-1.0)
    audit = _NoFullEdgeRegionTensor(
        edge_shape=(int(graph.edge_u.numel()), int(phi.shape[1]))
    )

    with audit:
        result = refine_graph_fusion_certificate(
            certificate=certificate,
            phi=phi,
            grad_smooth=-adjoint,
            gradient_scope="observed_objective",
            graph=graph,
            graph_hash="no-dense-audit",
            lower=torch.zeros_like(phi),
            upper=torch.ones_like(phi),
            lambda_value=lambda_value,
            atol=1e-9,
            options=CertificateOptions(max_iter=32, max_expansions=2),
        )

    assert result.status == "certified"
    assert isinstance(result.certificate, CompressedEdgeCertificate)
    assert tuple(result.certificate.internal_dual.shape) == (0, 2)
    assert audit.operations == []
