"""One immutable, fully resolved configuration for a CliPP2 fit."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Final

DEFAULT_MAX_MAJOR_CN: Final = None
DEFAULT_MAX_CLUSTERS: Final = 20
DEFAULT_STAGE_ONE_MAX_CLUSTERS: Final = 10
DEFAULT_DEVICE: Final = "cpu"
DEFAULT_AUDIT_SCOPE: Final = "selected"
DEFAULT_MISSING_COUNTS: Final = "zero_alt"
PROTOCOL_VERSION: Final = "2.3"
STAGE_THREE_REFERENCE_POLICY: Final = "single_fixed_clonal_stage_one_source_v1"
COMPATIBLE_NATIVE_FIT_PROTOCOLS: Final = ("2.2", "2.3")


def validate_audit_scope(value):
    if value not in ("selected", "all"):
        raise ValueError("audit_scope must be selected or all.")
    return value


def validate_missing_counts(value):
    if value not in ("zero_alt", "unobserved"):
        raise ValueError("missing_counts must be zero_alt or unobserved.")
    return value


def validate_max_major_cn(value: int | None) -> int | None:
    """None keeps all valid CN; a positive integer declares a population cutoff."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("max_major_cn must be None (unrestricted) or a positive integer.")
    return int(value)


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Public execution and CN eligibility options; numerical policy is fixed."""

    device: str = DEFAULT_DEVICE
    max_major_cn: int | None = DEFAULT_MAX_MAJOR_CN
    max_clusters: int = DEFAULT_MAX_CLUSTERS
    cpu_threads: int | None = None
    audit_scope: str = DEFAULT_AUDIT_SCOPE

    def __post_init__(self) -> None:
        validate_audit_scope(self.audit_scope)
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda.")
        object.__setattr__(self, "max_major_cn", validate_max_major_cn(self.max_major_cn))
        if type(self.max_clusters) is not int or self.max_clusters < 1:
            raise ValueError("max_clusters must be a positive integer.")
        if self.cpu_threads is not None and (
                type(self.cpu_threads) is not int or not 1 <= self.cpu_threads <= 2**31 - 1):
            raise ValueError("cpu_threads must be a positive int32 or None (auto).")


def resolve_fit_config(
    *, device: str = DEFAULT_DEVICE,
    max_major_cn: int | None = DEFAULT_MAX_MAJOR_CN,
    max_clusters: int = DEFAULT_MAX_CLUSTERS,
    cpu_threads: int | None = None,
    audit_scope: str = DEFAULT_AUDIT_SCOPE,
) -> FitConfig:
    """Construct the only public fit configuration; removed knobs raise TypeError."""
    return FitConfig(device=device, max_major_cn=max_major_cn,
                     max_clusters=max_clusters, cpu_threads=cpu_threads, audit_scope=audit_scope)


__all__ = ["FitConfig", "resolve_fit_config"]


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    """Protocol 2.3 estimation policy; numerical verification remains mandatory."""

    fit: FitConfig = field(default_factory=lambda: FitConfig(
        device="cpu", max_clusters=DEFAULT_STAGE_ONE_MAX_CLUSTERS))
    stage_two_max_clusters: int = DEFAULT_MAX_CLUSTERS
    normal_cn_policy: str | None = None
    missing_counts: str = DEFAULT_MISSING_COUNTS

    def __post_init__(self):
        validate_missing_counts(self.missing_counts)
        if type(self.fit) is not FitConfig:
            raise TypeError("Workflow requires a validated FitConfig")
        if self.normal_cn_policy not in (None, "diploid"):
            raise ValueError("normal_cn_policy must be None or the explicit 'diploid' assumption")
        if type(self.stage_two_max_clusters) is not int or self.stage_two_max_clusters < 1:
            raise ValueError("stage_two_max_clusters must be a positive integer")

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        fit_values = dict(device="cpu", max_clusters=DEFAULT_STAGE_ONE_MAX_CLUSTERS)
        fit_values.update(values.get("fit", {}))
        values["fit"] = FitConfig(**fit_values)
        return cls(**values)


__all__ += ["WorkflowConfig"]
