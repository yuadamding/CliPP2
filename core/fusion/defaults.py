"""Canonical execution defaults and normalized solver policy names."""

from __future__ import annotations

from typing import Final, Literal, TypeAlias, cast


DenseFallbackPolicy: TypeAlias = Literal["device_only", "cpu_allowed", "error"]

DEFAULT_DEVICE: Final = "cuda"
DEFAULT_DTYPE: Final = "float32"
DEFAULT_OPTIMIZATION_TOLERANCE: Final = 8e-4
DEFAULT_DENSE_FALLBACK_POLICY: Final[DenseFallbackPolicy] = "device_only"

DENSE_FALLBACK_POLICIES: Final = ("device_only", "cpu_allowed", "error")

DEFAULT_WORKSET_MAX_BYTES: Final = 256 * 1024 * 1024
DEFAULT_COMPRESSED_CACHE_MAX_BYTES: Final = 256 * 1024 * 1024
DEFAULT_WORKSET_ADD_BATCH: Final = 64
DEFAULT_WORKSET_MAX_EXPANSIONS: Final = 16
DEFAULT_CERTIFICATE_MAX_ITER: Final = 512
DEFAULT_CERTIFICATE_REFINEMENT_ROUNDS: Final = 2
DEFAULT_CERTIFICATE_COLUMN_TOL_SCALE: Final = 1.0


def normalize_dense_fallback_policy(value: str) -> DenseFallbackPolicy:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized == "auto":
        normalized = DEFAULT_DENSE_FALLBACK_POLICY
    if normalized not in DENSE_FALLBACK_POLICIES:
        raise ValueError(
            "dense_fallback_policy must be device_only, cpu_allowed, or error."
        )
    return cast(DenseFallbackPolicy, normalized)
