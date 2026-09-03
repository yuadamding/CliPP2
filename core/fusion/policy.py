from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Literal

import numpy as np

from .types import CompressedEdgeCertificate, RawFit


class NextAction(Enum):
    ACCEPT = auto()
    RETRY_SAME_RUNTIME = auto()
    DENSE_CURRENT_DEVICE = auto()
    CPU_FALLBACK = auto()
    FLOAT64_POLISH = auto()
    FAIL = auto()


PolicyPhase = Literal["working", "selected", "precision_polish"]


@dataclass(slots=True)
class PolicyState:
    phase: PolicyPhase
    result: RawFit | None = None
    resource_error: BaseException | None = None
    runtime_device_type: str = "cpu"
    fallback_policy: str = "error"
    representation_retry_done: bool = False

_AUDIT_COMPATIBLE_STATUSES = {
    "certified",
    "input_dual_retained",
    "analytic_nonfused_dual",
    "refined_fused_edge_dual",
    "zero_penalty_no_dual_needed",
}


def _compressed_representation_incomplete(result: RawFit) -> bool:
    terminal = result.certificate
    if not isinstance(terminal.witness, CompressedEdgeCertificate):
        return False
    return terminal.status in {"resource_limit", "workset_incomplete"}


def _precision_residual_is_only_blocker(result: RawFit) -> bool:
    terminal = result.certificate
    status_supported = terminal.status in _AUDIT_COMPATIBLE_STATUSES or (
        isinstance(terminal.witness, CompressedEdgeCertificate)
        and terminal.status == "not_certified"
    )
    return (
        np.isfinite(result.objective.total)
        and result.convergence.mm_consistency_violations == 0
        and terminal.directional_admissible
        and status_supported
        and np.isfinite(terminal.components.residual)
        and terminal.components.residual > terminal.tolerance
    )


def decide_next_action(state: PolicyState) -> NextAction:
    """Choose one estimator-preserving retry, fallback, or terminal action."""

    if state.resource_error is not None:
        cpu_allowed = (
            state.fallback_policy == "cpu_allowed"
            and state.runtime_device_type != "cpu"
        )
        return NextAction.CPU_FALLBACK if cpu_allowed else NextAction.FAIL
    if state.result is None:
        return NextAction.RETRY_SAME_RUNTIME
    if state.phase == "working":
        if (
            not state.representation_retry_done
            and _compressed_representation_incomplete(state.result)
        ):
            if state.fallback_policy == "error":
                return NextAction.FAIL
            return NextAction.DENSE_CURRENT_DEVICE
        return NextAction.ACCEPT
    if state.phase == "selected":
        if (
            state.result.provenance.dtype != "float64"
            and not state.result.certificate.admissible
            and _precision_residual_is_only_blocker(state.result)
        ):
            return NextAction.FLOAT64_POLISH
        return NextAction.ACCEPT
    if state.phase == "precision_polish":
        return NextAction.ACCEPT
    raise ValueError(f"Unknown policy phase: {state.phase}")


def combine_fallback_reasons(*reasons: str) -> str:
    normalized = (str(reason).strip() for reason in reasons)
    return ";".join(dict.fromkeys(reason for reason in normalized if reason))


def record_attempt(
    result: RawFit,
    *,
    attempted: RawFit | None = None,
    reason: str = "",
    backend_name: str | None = None,
) -> RawFit:
    backend = str(backend_name or result.provenance.inner_solver)
    attempted_reason = "" if attempted is None else attempted.certificate.fallback_reason
    return replace(
        result,
        work=result.work if attempted is None else result.work + attempted.work,
        certificate=replace(
            result.certificate,
            fallback_reason=combine_fallback_reasons(
                attempted_reason,
                result.certificate.fallback_reason,
                reason,
            ),
        ),
        provenance=replace(result.provenance, inner_solver=backend),
    )
