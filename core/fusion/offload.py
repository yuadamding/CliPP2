"""Host retention of raw fits, shared by fusion and model selection."""

from dataclasses import dataclass, fields, is_dataclass, replace
import math

import torch

from .graph_ops import dense_complete_solver_memory_preflight
from .types import (
    CompressedEdgeCertificate,
    DenseEdgeCertificate,
    DenseWarmState,
    PrimalOnlyWarmState,
    RawFit,
    SolverState,
)


def _warm_identity(value):
    """Metadata/version snapshot without copying or synchronizing tensor data."""
    if torch.is_tensor(value):
        if value.is_inference() or value.requires_grad or value.layout != torch.strided:
            raise ValueError("Unversioned or differentiable warm tensors cannot be retained.")
        return (id(value), value._version, value.dtype, value.device, tuple(value.shape),
                tuple(value.stride()), value.storage_offset())
    if is_dataclass(value):
        return (id(value), tuple((field.name, _warm_identity(getattr(value, field.name)))
                                for field in fields(value)))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("Unknown warm-state metadata cannot enter the active owner.")


def _warm_problem_identity(problem):
    return (id(problem), problem.objective_spec_hash, problem.runtime.device,
            problem.runtime.dtype, problem.source_data.num_mutations,
            problem.source_data.num_regions, problem.optimization_box.witness_index,
            tuple(_warm_identity(value) for value in (
                problem.graph.edge_index, problem.graph.weight)))


@dataclass(slots=True)
class _ActiveWarmState:
    """One call-local upload; archives and certificate ownership stay on CPU."""

    resident: SolverState | None
    source_identity: tuple
    resident_identity: tuple
    problem_identity: tuple

    def resolve(self, state, problem):
        if self.resident is None:
            return state
        try:
            valid = (
                _warm_identity(state) == self.source_identity
                and _warm_identity(self.resident) == self.resident_identity
                and _warm_problem_identity(problem) == self.problem_identity
            )
        except (AttributeError, RuntimeError, ValueError):
            valid = False
        if not valid:
            # Changed source input still follows the ordinary solver's own
            # validation; an edited upload is never reused as its replacement.
            self.resident = None
            return state
        return self.resident


def prepare_active_warm_state(state, problem) -> _ActiveWarmState | None:
    """Upload one source-bound CPU guide state for this witness loop, if safe.

    No objective, lambda, projection or certificate changes are made. The
    caller must resolve the owner before every branch and never retain it in
    search history. Unknown capacity is not permission to retain more VRAM.
    """
    if (
        state is None or problem.runtime.device.type != "cuda"
        or torch.is_inference_mode_enabled()
        or problem.optimization_box.witness_index is not None
        or state.objective_spec_hash != problem.objective_spec_hash
        or not math.isfinite(float(state.previous_lambda))
    ):
        return None
    n, regions = problem.source_data.num_mutations, problem.source_data.num_regions
    edges = n * (n - 1) // 2
    if edges == 0 or any(
        not torch.is_tensor(value) or value.device.type != "cpu"
        or value.dtype != problem.runtime.dtype or tuple(value.shape) != shape
        or value.layout != torch.strided or not value.is_contiguous()
        or value.is_shared()
        for value, shape in ((state.phi, (n, regions)), (state.dual, (edges, regions)))
    ):
        return None
    try:
        source_identity = _warm_identity(state)
        problem_identity = _warm_problem_identity(problem)
    except (AttributeError, RuntimeError, ValueError):
        return None
    upload_bytes = sum(value.numel() * value.element_size() for value in (state.phi, state.dual))
    # The node FP64-QP allowance can make a tiny FP32 solve's estimate larger
    # than FP64's; qualify both instead of assuming promotion always dominates.
    for dtype in dict.fromkeys((problem.runtime.dtype, torch.float64)):
        try:
            fits, _, limit = dense_complete_solver_memory_preflight(
                num_nodes=n, num_regions=regions,
                runtime=replace(problem.runtime, dtype=dtype),
                resident_edges=(problem.graph.edge_u, problem.graph.edge_v),
                extra_allocation_bytes=upload_bytes,
            )
        except (RuntimeError, MemoryError, ValueError):
            return None
        if not fits or limit is None:
            return None
    try:
        resident = replace(
            state, phi=state.phi.detach().to(device=problem.runtime.device),
            dual=state.dual.detach().to(device=problem.runtime.device),
        )
    except torch.OutOfMemoryError:
        return None  # Optional ownership optimization; ordinary solve remains authoritative.
    owner = _ActiveWarmState(resident, source_identity, _warm_identity(resident), problem_identity)
    return owner if owner.resolve(state, problem) is resident else None


def offload_raw_fit_to_cpu(fit: RawFit) -> RawFit:
    """Detach persistent tensors to CPU without changing computation provenance.

    Memoize identical storage views across solver state, warm hints and terminal
    certificates. Storage identity (not its data pointer) distinguishes empty
    tensors, whose pointers can all be zero. Distinct strided views retain their
    own values; exact view aliases share a single host tensor.
    """
    tensors: dict[tuple, torch.Tensor] = {}

    def host(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        key = (
            tensor.device, tensor.dtype, tensor.layout, tensor.untyped_storage(),
            tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()),
        )
        if key not in tensors:
            tensors[key] = tensor.detach().to(device="cpu")
        return tensors[key]

    def certificate_host(certificate):
        if isinstance(certificate, DenseEdgeCertificate):
            return replace(certificate, dual=host(certificate.dual))
        if isinstance(certificate, CompressedEdgeCertificate):
            return replace(
                certificate, labels=host(certificate.labels), centers=host(certificate.centers),
                internal_edge_ids=host(certificate.internal_edge_ids),
                internal_dual=host(certificate.internal_dual),
            )
        return certificate

    state = fit.state
    if state is not None:
        warm = state.warm_state
        if isinstance(warm, DenseWarmState):
            warm = replace(warm, phi=host(warm.phi), dual=host(warm.dual))
        elif isinstance(warm, PrimalOnlyWarmState):
            warm = replace(
                warm, phi=host(warm.phi), structure_hint=host(warm.structure_hint),
                certificate_hint=certificate_host(warm.certificate_hint),
            )
        state = replace(
            state, phi=host(state.phi), dual=host(state.dual), warm_state=warm,
            certificate=certificate_host(state.certificate),
        )
    return replace(
        fit, state=state,
        certificate=replace(fit.certificate, witness=certificate_host(fit.certificate.witness)),
    )
