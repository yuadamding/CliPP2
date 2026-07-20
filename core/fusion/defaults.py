"""Canonical execution defaults and normalized solver policy names."""

from __future__ import annotations

from typing import Final, Literal, TypeAlias, cast


InnerBackend: TypeAlias = Literal["auto", "dense", "quotient_workset"]
DenseFallbackPolicy: TypeAlias = Literal["device_only", "cpu_allowed", "error"]

DEFAULT_DEVICE: Final = "cuda"
DEFAULT_DTYPE: Final = "float64"
DEFAULT_INNER_BACKEND: Final[InnerBackend] = "dense"
DEFAULT_DENSE_FALLBACK_POLICY: Final[DenseFallbackPolicy] = "device_only"

INNER_BACKENDS: Final = ("auto", "dense", "quotient_workset")
DENSE_FALLBACK_POLICIES: Final = ("device_only", "cpu_allowed", "error")
DENSE_FALLBACK_POLICY_INPUTS: Final = ("auto", *DENSE_FALLBACK_POLICIES)

DEFAULT_WORKSET_MAX_BYTES: Final = 256 * 1024 * 1024
DEFAULT_COMPRESSED_CACHE_MAX_BYTES: Final = 256 * 1024 * 1024
DEFAULT_WORKSET_ADD_BATCH: Final = 64
DEFAULT_WORKSET_MAX_EXPANSIONS: Final = 16
DEFAULT_CERTIFICATE_MAX_ITER: Final = 512
DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS: Final = 2
DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE: Final = 1.0


def normalize_inner_backend(value: str) -> InnerBackend:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in INNER_BACKENDS:
        raise ValueError("inner_backend must be one of: auto, dense, quotient_workset.")
    if normalized == "auto":
        normalized = DEFAULT_INNER_BACKEND
    return cast(InnerBackend, normalized)


def normalize_dense_fallback_policy(value: str) -> DenseFallbackPolicy:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "auto":
        normalized = DEFAULT_DENSE_FALLBACK_POLICY
    if normalized not in DENSE_FALLBACK_POLICIES:
        raise ValueError(
            "dense_fallback_policy must be device_only, cpu_allowed, or error."
        )
    return cast(DenseFallbackPolicy, normalized)
