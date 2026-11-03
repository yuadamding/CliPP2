"""Versioned proposal budgets, not numerical tolerances or runtime deadlines."""
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchPolicy:
    name: str = "bounded_diverse_search_v2"
    tree_width: int = 4
    repair_width: int = 2
    split_slots: int = 32
    exchange_parents: int = 8
    exchange_slots: int = 16
    boundary_slots: int = 16
    polish_starts_per_q: int = 2
    polish_rounds: int = 2

    def limits(self, k):
        return {"native": k, "repair": self.repair_width*k*(k-1)//2,
                "coarsening": self.tree_width*k*(k-1)//2,
                "boundary": self.tree_width*self.boundary_slots*k,
                "polishing": self.polish_starts_per_q*self.polish_rounds*2*k,
                "split": self.split_slots, "exchange": self.exchange_slots}


POLICY = SearchPolicy()


class SearchWork:
    """Charge absent, duplicate, rejected and nonadvancing slots alike."""

    def __init__(self, k):
        self.limits = POLICY.limits(k)
        self.used = dict.fromkeys(self.limits, 0)

    def spend(self, kind, count=1):
        if count < 0 or self.used[kind]+count > self.limits[kind]:
            raise AssertionError(f"Bounded search exceeded its {kind} slots")
        self.used[kind] += count

    def summary(self):
        return {"policy": POLICY.name, "slots": dict(self.used),
                "limits": dict(self.limits), "total_slots": sum(self.used.values()),
                "maximum_slots": sum(self.limits.values()),
                "excludes": ["continuation", "proposal_ranking", "terminal_audit", "verification"]}
