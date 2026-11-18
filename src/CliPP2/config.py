"""Protocol 3.0 configuration: one role-constrained likelihood."""

from dataclasses import dataclass
from numbers import Integral
from typing import Final

DEFAULT_MAX_MAJOR_CN: Final = None
DEFAULT_MAX_CLUSTERS: Final = 10
DEFAULT_MISSING_COUNTS: Final = "zero_alt"
DEFAULT_PROPOSAL_BUDGET: Final = 32


def validate_missing_counts(value):
    if value not in ("zero_alt", "unobserved"):
        raise ValueError("missing_counts must be zero_alt or unobserved.")
    return value


def validate_max_major_cn(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("max_major_cn must be None (unrestricted) or a positive integer.")
    return int(value)


def _integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Discovery search, eligibility, and input policies for one CPU fit."""

    max_major_cn: int | None = DEFAULT_MAX_MAJOR_CN
    max_clusters: int = DEFAULT_MAX_CLUSTERS
    cpu_threads: int | None = None
    proposal_budget: int = DEFAULT_PROPOSAL_BUDGET
    normal_cn_policy: str | None = None
    missing_counts: str = DEFAULT_MISSING_COUNTS

    def __post_init__(self):
        object.__setattr__(self, "max_major_cn", validate_max_major_cn(self.max_major_cn))
        _integer(self.max_clusters, "max_clusters")
        _integer(self.proposal_budget, "proposal_budget", minimum=0)
        if self.cpu_threads is not None:
            _integer(self.cpu_threads, "cpu_threads")
            if self.cpu_threads > 2**31 - 1:
                raise ValueError("cpu_threads must fit a positive int32.")
        if self.normal_cn_policy not in (None, "diploid"):
            raise ValueError("normal_cn_policy must be None or 'diploid'.")
        validate_missing_counts(self.missing_counts)

__all__ = ["FitConfig"]
