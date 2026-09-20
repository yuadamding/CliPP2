"""Host retention of raw fits, shared by fusion and model selection."""

from dataclasses import replace

import torch

from .types import (
    CompressedEdgeCertificate,
    DenseEdgeCertificate,
    DenseWarmState,
    PrimalOnlyWarmState,
    RawFit,
)


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
