from __future__ import annotations

from dataclasses import replace

import numpy as np

from .types import CompressedEdgeCertificate, RawFit


_AUDIT_COMPATIBLE_STATUSES = {
    "certified",
    "input_dual_retained",
    "analytic_nonfused_dual",
    "refined_fused_edge_dual",
    "zero_penalty_no_dual_needed",
}




def _precision_residual_is_only_blocker(result: RawFit) -> bool:
    terminal = result.certificate
    status_supported = terminal.status in _AUDIT_COMPATIBLE_STATUSES or (
        isinstance(terminal.witness, CompressedEdgeCertificate)
        and terminal.status == "not_certified"
        and result.work.full_certificate_audit_passes > 0
    )
    return (
        np.isfinite(result.objective.total)
        and result.convergence.mm_consistency_violations == 0
        and terminal.directional_admissible
        and status_supported
        and np.isfinite(terminal.components.residual)
        and terminal.components.residual > terminal.tolerance
    )


def needs_precision_polish(result: RawFit) -> bool:
    """Only a finite, otherwise admissible float32 residual miss can be polished."""
    return bool(result.provenance.dtype != "float64"
                and not result.certificate.admissible
                and _precision_residual_is_only_blocker(result))


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
