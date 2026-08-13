from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Literal


SelectionContractId = Literal[
    "raw-fusion-only-v0.3",
    "hybrid-ward-cem-v1",
    "legacy-0.1-selection-compat",
]


@dataclass(frozen=True)
class PartitionCandidateConfig:
    k_anchors: tuple[int, ...]
    max_candidates_per_k: int
    cem_max_iter: int
    generation_refit_max_iter: int
    classification_alpha: float
    classification_code_weight: float
    allow_component_death: bool
    include_plain_ward: bool
    include_ward_cem: bool
    include_final_phi_ladder: bool
    final_phi_ladder_kmax: int
    final_phi_parent_count: int
    adaptive_k_refinement: bool = False


@dataclass(frozen=True)
class SelectionContract:
    contract_id: SelectionContractId
    selectable_partition_pool: bool
    raw_partition_rule: Literal[
        "certified_complete_link",
        "legacy_connected_components",
    ]
    graph_pilot_source: Literal[
        "partition_guide",
        "zero_penalty_pilot",
        "profile_default",
    ]
    partition_config: PartitionCandidateConfig
    force_float64: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


_K_ANCHORS = (*range(1, 16), 20, 25, 30, 40, 50)


RAW_FUSION_ONLY_V03 = SelectionContract(
    contract_id="raw-fusion-only-v0.3",
    selectable_partition_pool=False,
    raw_partition_rule="certified_complete_link",
    # Preserve 0.3.0 exactly: strict used the guide and approximate profiles
    # used the zero-penalty pilot to construct the frozen graph.
    graph_pilot_source="profile_default",
    partition_config=PartitionCandidateConfig(
        k_anchors=_K_ANCHORS,
        max_candidates_per_k=5,
        cem_max_iter=8,
        generation_refit_max_iter=32,
        classification_alpha=1.0,
        classification_code_weight=0.7,
        allow_component_death=True,
        include_plain_ward=True,
        include_ward_cem=True,
        include_final_phi_ladder=False,
        final_phi_ladder_kmax=0,
        final_phi_parent_count=0,
        adaptive_k_refinement=True,
    ),
    force_float64=False,
)


HYBRID_WARD_CEM_V1 = SelectionContract(
    contract_id="hybrid-ward-cem-v1",
    selectable_partition_pool=True,
    raw_partition_rule="certified_complete_link",
    graph_pilot_source="zero_penalty_pilot",
    partition_config=PartitionCandidateConfig(
        k_anchors=_K_ANCHORS,
        max_candidates_per_k=5,
        cem_max_iter=8,
        generation_refit_max_iter=32,
        classification_alpha=1.0,
        classification_code_weight=0.7,
        allow_component_death=False,
        include_plain_ward=True,
        include_ward_cem=True,
        include_final_phi_ladder=True,
        final_phi_ladder_kmax=30,
        final_phi_parent_count=1,
    ),
    force_float64=False,
)


LEGACY_01_SELECTION_COMPAT = SelectionContract(
    contract_id="legacy-0.1-selection-compat",
    selectable_partition_pool=True,
    raw_partition_rule="legacy_connected_components",
    graph_pilot_source="partition_guide",
    partition_config=PartitionCandidateConfig(
        k_anchors=_K_ANCHORS,
        max_candidates_per_k=5,
        cem_max_iter=8,
        generation_refit_max_iter=32,
        classification_alpha=1.0,
        classification_code_weight=1.0,
        allow_component_death=True,
        include_plain_ward=True,
        include_ward_cem=True,
        include_final_phi_ladder=False,
        final_phi_ladder_kmax=0,
        final_phi_parent_count=0,
    ),
    force_float64=True,
)


SELECTION_CONTRACTS: dict[str, SelectionContract] = {
    item.contract_id: item
    for item in (
        RAW_FUSION_ONLY_V03,
        HYBRID_WARD_CEM_V1,
        LEGACY_01_SELECTION_COMPAT,
    )
}
SELECTION_CONTRACT_IDS = tuple(SELECTION_CONTRACTS)
# The raw-only contract remains available as the exact v0.3 compatibility
# surface.  Production defaults to the hybrid union because the deterministic
# Ward/CEM partitions cover statistically competitive low-K models that the
# bounded non-convex raw lambda path can skip entirely.
DEFAULT_SELECTION_CONTRACT = HYBRID_WARD_CEM_V1.contract_id


def normalize_selection_contract_id(value: str) -> str:
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "raw-fusion-only-v03": "raw-fusion-only-v0.3",
        "legacy-01-selection-compat": "legacy-0.1-selection-compat",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SELECTION_CONTRACTS:
        allowed = ", ".join(SELECTION_CONTRACT_IDS)
        raise ValueError(f"Unknown selection contract {value!r}; expected {allowed}.")
    return normalized


def get_selection_contract(value: str) -> SelectionContract:
    return SELECTION_CONTRACTS[normalize_selection_contract_id(value)]


__all__ = [
    "DEFAULT_SELECTION_CONTRACT",
    "get_selection_contract",
    "HYBRID_WARD_CEM_V1",
    "LEGACY_01_SELECTION_COMPAT",
    "normalize_selection_contract_id",
    "PartitionCandidateConfig",
    "RAW_FUSION_ONLY_V03",
    "SELECTION_CONTRACT_IDS",
    "SelectionContract",
]
