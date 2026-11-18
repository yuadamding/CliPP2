"""CPU discovery settings for fixed-multiplicity clustering."""

from dataclasses import dataclass
from typing import Final

DEFAULT_MAX_CLUSTERS: Final = 10
DEFAULT_MISSING_COUNTS: Final = "zero_alt"
DEFAULT_PROPOSAL_BUDGET: Final = 32


def validate_missing_counts(value):
    if value not in ("zero_alt", "unobserved"):
        raise ValueError("missing_counts must be zero_alt or unobserved.")
    return value


def _integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


@dataclass(frozen=True, slots=True)
class FitConfig:
    """Discovery search and input settings for one CPU fit."""

    max_clusters: int = DEFAULT_MAX_CLUSTERS
    cpu_threads: int | None = None
    proposal_budget: int = DEFAULT_PROPOSAL_BUDGET
    missing_counts: str = DEFAULT_MISSING_COUNTS

    def __post_init__(self):
        _integer(self.max_clusters, "max_clusters")
        _integer(self.proposal_budget, "proposal_budget", minimum=0)
        if self.cpu_threads is not None:
            _integer(self.cpu_threads, "cpu_threads")
            if self.cpu_threads > 2**31 - 1:
                raise ValueError("cpu_threads must fit a positive int32.")
        validate_missing_counts(self.missing_counts)

__all__ = ["FitConfig"]
