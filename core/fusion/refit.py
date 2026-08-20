"""Backward-compatible imports for canonical scalar partition refitting."""

from ..scalar import (
    PartitionRefitResult,
    canonical_partition_labels,
    partition_constrained_observed_refit,
)

_canonical_labels = canonical_partition_labels

__all__ = ["PartitionRefitResult", "partition_constrained_observed_refit"]
