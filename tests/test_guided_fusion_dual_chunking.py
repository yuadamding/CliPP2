from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from CliPP2.core.fusion.solver import prepare_torch_problem
from CliPP2.core.fusion.types import PairwiseFusionGraph, QuotientWorksetWarmState
from CliPP2.io.data import TumorData, compute_phi_init_from_counts
import CliPP2.model_selection.guided_fusion as guided_fusion
from CliPP2.model_selection.guided_fusion import _assemble_actual_dual


def _dual_inputs() -> dict[str, torch.Tensor]:
    phi = torch.tensor(
        [
            [0.0, 0.25, 0.40],
            [0.0, 0.25, 0.40],
            [0.70, 1.00, 0.55],
            [0.70, 1.00, 0.55],
            [0.0, 0.25, 0.40],
            [0.35, 0.60, 0.85],
        ],
        dtype=torch.float64,
    )
    edge_index = torch.triu_indices(6, 6, offset=1)
    return {
        "phi": phi,
        "labels": torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.long),
        "grad_smooth": torch.tensor(
            [
                [1.0, -0.5, 0.2],
                [-0.3, 0.7, -1.1],
                [0.4, -0.8, 1.3],
                [-1.2, 0.6, 0.1],
                [0.9, -0.4, -0.7],
                [-0.6, 1.1, 0.5],
            ],
            dtype=torch.float64,
        ),
        "lower": torch.zeros_like(phi),
        "upper": torch.ones_like(phi),
        "edge_u": edge_index[0],
        "edge_v": edge_index[1],
        "edge_w": torch.linspace(0.2, 1.6, edge_index.shape[1], dtype=torch.float64),
    }


@pytest.mark.parametrize("lambda_value", [0.0, 2.75])
def test_chunked_actual_dual_is_exactly_vectorized_equivalent(
    lambda_value: float,
) -> None:
    inputs = _dual_inputs()

    vectorized = _assemble_actual_dual(lambda_value=lambda_value, **inputs)
    one_edge_chunks = _assemble_actual_dual(
        lambda_value=lambda_value,
        _work_memory_bytes=1,
        **inputs,
    )

    for expected, observed in zip(vectorized, one_edge_chunks, strict=True):
        torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


class _FullEdgeTemporaryAudit(TorchDispatchMode):
    def __init__(self, *, edge_shape: tuple[int, int]) -> None:
        super().__init__()
        self.edge_shape = edge_shape
        self.full_edge_work_ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        name = str(func)
        tensors = output if isinstance(output, (tuple, list)) else (output,)
        if "index_select" in name or "neg" in name:
            for tensor in tensors:
                if torch.is_tensor(tensor) and tuple(tensor.shape) == self.edge_shape:
                    self.full_edge_work_ops.append(name)
        return output


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


class _NoFullEdgeVectorTensor(TorchDispatchMode):
    def __init__(self, *, edge_count: int) -> None:
        super().__init__()
        self.edge_count = int(edge_count)
        self.operations: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))

        def inspect(value) -> None:
            if torch.is_tensor(value):
                # The complete graph stores edge endpoints as row views of its
                # persistent edge-index tensor.  Those `select` views are not
                # temporaries; flag every other full-E vector result.
                if (
                    "select.int" not in str(func)
                    and value.ndim == 1
                    and int(value.numel()) == self.edge_count
                ):
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


def test_chunked_actual_dual_avoids_full_edge_region_work_tensors() -> None:
    inputs = _dual_inputs()
    edge_shape = (int(inputs["edge_u"].numel()), int(inputs["phi"].shape[1]))
    audit = _FullEdgeTemporaryAudit(edge_shape=edge_shape)

    with audit:
        dual, _, _ = _assemble_actual_dual(
            lambda_value=2.75,
            _work_memory_bytes=1,
            **inputs,
        )

    assert tuple(dual.shape) == edge_shape
    assert audit.full_edge_work_ops == []


def _guided_context(
    guide_phi: np.ndarray,
    *,
    graph: PairwiseFusionGraph | None = None,
):
    num_mutations, num_regions = guide_phi.shape
    alt = np.arange(
        5.0,
        5.0 + float(num_mutations * num_regions),
        dtype=np.float64,
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
    data = TumorData(
        tumor_id="guided-chunking-test",
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
    return prepare_torch_problem(
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


def test_full_guided_builder_matches_when_all_edge_helpers_are_chunked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guide_phi = np.asarray(
        [
            [0.20, 0.30, 0.40],
            [0.20, 0.30, 0.40],
            [0.70, 0.60, 0.55],
            [0.70, 0.60, 0.55],
            [0.45, 0.50, 0.65],
            [0.85, 0.25, 0.75],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64)
    grad = _dual_inputs()["grad_smooth"].numpy()
    context = _guided_context(guide_phi)
    vectorized = guided_fusion.build_guided_fusion_initialization(
        guide_phi,
        labels,
        solver_context=context,
        grad_smooth=grad,
    )

    original_assemble = guided_fusion._assemble_actual_dual
    assemble_calls = 0

    def one_edge_assemble(*args, **kwargs):
        nonlocal assemble_calls
        assemble_calls += 1
        kwargs["_work_memory_bytes"] = 1
        return original_assemble(*args, **kwargs)

    # One byte forces the ratio, projection, and separation helpers through
    # their bounded branches. The wrapper also forces `_assemble_actual_dual`,
    # whose default work budget was bound when that function was defined.
    monkeypatch.setattr(guided_fusion, "_GUIDED_FUSION_WORK_BYTES", 1)
    monkeypatch.setattr(guided_fusion, "_assemble_actual_dual", one_edge_assemble)
    chunked = guided_fusion.build_guided_fusion_initialization(
        guide_phi,
        labels,
        solver_context=context,
        grad_smooth=grad,
    )

    assert assemble_calls >= 2
    assert chunked.lambda_value == vectorized.lambda_value
    assert chunked.diagnostics == vectorized.diagnostics
    assert (
        chunked.solver_state.previous_lambda == vectorized.solver_state.previous_lambda
    )
    assert chunked.solver_state.split is vectorized.solver_state.split is None
    assert chunked.solver_state.curvature is vectorized.solver_state.curvature is None
    torch.testing.assert_close(
        chunked.solver_state.phi,
        vectorized.solver_state.phi,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        chunked.solver_state.dual,
        vectorized.solver_state.dual,
        rtol=0.0,
        atol=0.0,
    )


def test_compressed_guided_builder_matches_dense_diagnostics_and_state() -> None:
    guide_phi = np.asarray(
        [
            [0.20, 0.30, 0.40],
            [0.20, 0.30, 0.40],
            [0.70, 0.60, 0.55],
            [0.70, 0.60, 0.55],
            [0.45, 0.50, 0.65],
            [0.85, 0.25, 0.75],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64)
    grad = _dual_inputs()["grad_smooth"].numpy()
    context = _guided_context(guide_phi)

    default_dense = guided_fusion.build_guided_fusion_initialization(
        guide_phi,
        labels,
        solver_context=context,
        grad_smooth=grad,
    )
    explicit_dense = guided_fusion.build_guided_fusion_initialization(
        guide_phi,
        labels,
        solver_context=context,
        grad_smooth=grad,
        materialize_dense_dual=True,
    )
    compressed = guided_fusion.build_guided_fusion_initialization(
        guide_phi,
        labels,
        solver_context=context,
        grad_smooth=grad,
        materialize_dense_dual=False,
    )

    assert explicit_dense.lambda_value == default_dense.lambda_value
    assert explicit_dense.diagnostics == default_dense.diagnostics
    torch.testing.assert_close(
        explicit_dense.solver_state.dual,
        default_dense.solver_state.dual,
        rtol=0.0,
        atol=0.0,
    )
    assert compressed.lambda_value == default_dense.lambda_value
    assert compressed.diagnostics == default_dense.diagnostics
    assert compressed.solver_state.dual is None
    assert compressed.solver_state.certificate is None
    assert isinstance(compressed.solver_state.warm_state, QuotientWorksetWarmState)
    warm_state = compressed.solver_state.warm_state
    assert warm_state.graph_hash == context.graph_hash
    assert warm_state.previous_lambda == compressed.lambda_value
    assert warm_state.phi is compressed.solver_state.phi
    assert warm_state.quotient_dual is None
    assert tuple(warm_state.internal_edge_ids.shape) == (2,)
    assert tuple(warm_state.internal_dual.shape) == (2, guide_phi.shape[1])
    torch.testing.assert_close(
        warm_state.internal_dual,
        default_dense.solver_state.dual.index_select(0, warm_state.internal_edge_ids),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        warm_state.labels,
        torch.as_tensor(labels, dtype=torch.long),
        rtol=0.0,
        atol=0.0,
    )
    expected_centers = torch.as_tensor(guide_phi[[0, 2, 4, 5]], dtype=torch.float64)
    torch.testing.assert_close(
        warm_state.centers,
        expected_centers,
        rtol=0.0,
        atol=0.0,
    )


def test_compressed_guided_builder_never_materializes_full_edge_dual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guide_phi = np.asarray(
        [
            [0.20, 0.30, 0.40],
            [0.20, 0.30, 0.40],
            [0.70, 0.60, 0.55],
            [0.70, 0.60, 0.55],
            [0.45, 0.50, 0.65],
            [0.85, 0.25, 0.75],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64)
    grad = _dual_inputs()["grad_smooth"].numpy()
    context = _guided_context(guide_phi)
    edge_shape = (
        int(context.graph.edge_u.numel()),
        int(guide_phi.shape[1]),
    )
    audit = _NoFullEdgeRegionTensor(edge_shape=edge_shape)

    def reject_dense_assembly(*args, **kwargs):
        raise AssertionError("compressed initialization called dense dual assembly")

    monkeypatch.setattr(
        guided_fusion,
        "_assemble_actual_dual",
        reject_dense_assembly,
    )
    with audit:
        result = guided_fusion.build_guided_fusion_initialization(
            guide_phi,
            labels,
            solver_context=context,
            grad_smooth=grad,
            materialize_dense_dual=False,
        )

    assert result.solver_state.dual is None
    assert audit.operations == []


def test_compressed_guided_builder_avoids_full_edge_index_temporaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guide_phi = np.asarray(
        [
            [0.20, 0.30, 0.40],
            [0.20, 0.30, 0.40],
            [0.70, 0.60, 0.55],
            [0.70, 0.60, 0.55],
            [0.45, 0.50, 0.65],
            [0.85, 0.25, 0.75],
        ],
        dtype=np.float64,
    )
    labels = np.asarray([0, 0, 1, 1, 2, 3], dtype=np.int64)
    context = _guided_context(guide_phi)
    grad = _dual_inputs()["grad_smooth"]
    edge_count = int(context.graph.edge_u.numel())
    audit = _NoFullEdgeVectorTensor(edge_count=edge_count)
    monkeypatch.setattr(
        guided_fusion, "_validate_complete_graph", lambda *_a, **_k: None
    )

    with audit:
        result = guided_fusion.build_guided_fusion_initialization(
            guide_phi,
            labels,
            solver_context=context,
            grad_smooth=grad,
            materialize_dense_dual=False,
        )

    assert result.diagnostics.within_edge_count == 2
    assert result.diagnostics.between_edge_count == edge_count - 2
    assert audit.operations == []


def test_compressed_guided_builder_matches_projected_capacity_fallback() -> None:
    guide_phi = np.asarray([[0.20], [0.20], [0.80]], dtype=np.float64)
    labels = np.asarray([0, 0, 1], dtype=np.int64)
    grad = np.asarray([[1.0], [-1.0], [0.0]], dtype=np.float64)
    graph = PairwiseFusionGraph(
        edge_u=np.asarray([0, 0, 1], dtype=np.int32),
        edge_v=np.asarray([1, 2, 2], dtype=np.int32),
        edge_w=np.asarray([0.1, 10.0, 0.1], dtype=np.float64),
        name="complete_heterogeneous",
        degree_bound=2,
    )
    context = _guided_context(guide_phi, graph=graph)

    dense = guided_fusion.build_guided_fusion_initialization(
        guide_phi,
        labels,
        solver_context=context,
        grad_smooth=grad,
        max_capacity_iterations=4,
    )
    compressed = guided_fusion.build_guided_fusion_initialization(
        guide_phi,
        labels,
        solver_context=context,
        grad_smooth=grad,
        max_capacity_iterations=4,
        materialize_dense_dual=False,
    )

    assert not dense.diagnostics.capacity_converged
    assert dense.diagnostics.capacity_status == "projected_zero_between_capacity_scale"
    assert compressed.lambda_value == dense.lambda_value
    assert compressed.diagnostics == dense.diagnostics
