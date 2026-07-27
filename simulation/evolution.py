"""Pure joint SNV and copy-number evolutionary mechanics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CopyNumberEvolutionConfig, _validate_copy_number_config


@dataclass(frozen=True)
class GenomeSegment:
    segment_id: int
    chromosome: int
    start: int
    end: int


@dataclass(frozen=True)
class CNAEvent:
    event_id: int
    clone_id: int
    parent_clone_id: int
    branch_time: float
    segment_ids: tuple[int, ...]
    allele: int
    event_type: str = "gain"


@dataclass(frozen=True)
class JointEvolutionResult:
    clone_allele_cn: np.ndarray
    clone_total_cn: np.ndarray
    mutation_dosage_numeric: np.ndarray
    mutation_carrier: np.ndarray
    mutation_branch_time: np.ndarray
    mutation_origin_allele: np.ndarray
    mutation_origin_physical_copy: np.ndarray
    mutation_cn_at_origin: np.ndarray
    cna_event_history: pd.DataFrame


def simulate_genome_segments(
    n_segments: int,
    *,
    segment_size_bp: int = 1_000_000,
    chromosome: int = 1,
) -> list[GenomeSegment]:
    if n_segments < 1:
        raise ValueError("n_segments must be at least 1.")
    if segment_size_bp < 1:
        raise ValueError("segment_size_bp must be at least 1.")
    return [
        GenomeSegment(
            segment_id=segment_id,
            chromosome=int(chromosome),
            start=segment_id * int(segment_size_bp) + 1,
            end=(segment_id + 1) * int(segment_size_bp),
        )
        for segment_id in range(int(n_segments))
    ]


def assign_mutations_to_segments(
    no_mutations: int,
    segments: list[GenomeSegment],
    *,
    random_state=None,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    if no_mutations < 0:
        raise ValueError("no_mutations must be nonnegative.")
    if not segments:
        raise ValueError("At least one genomic segment is required.")
    if [segment.segment_id for segment in segments] != list(range(len(segments))):
        raise ValueError("Segment IDs must be contiguous and aligned to list order.")

    lengths = np.asarray(
        [segment.end - segment.start + 1 for segment in segments], dtype=float
    )
    if np.any(lengths <= 0):
        raise ValueError("Every segment must have positive length.")
    probabilities = lengths / lengths.sum()
    mutation_segment = rng.choice(
        len(segments), size=no_mutations, p=probabilities
    ).astype(int)
    mutation_position = np.empty(no_mutations, dtype=int)
    used_by_segment: list[set[int]] = [set() for _ in segments]
    for mutation_id, segment_id in enumerate(mutation_segment):
        segment = segments[int(segment_id)]
        if len(used_by_segment[int(segment_id)]) >= segment.end - segment.start + 1:
            raise ValueError(
                f"Segment {segment_id} has fewer positions than assigned mutations."
            )
        while True:
            position = int(rng.integers(segment.start, segment.end + 1))
            if position not in used_by_segment[int(segment_id)]:
                used_by_segment[int(segment_id)].add(position)
                mutation_position[mutation_id] = position
                break
    return mutation_segment, mutation_position


def simulate_branch_cna_events(
    parent: np.ndarray,
    config: CopyNumberEvolutionConfig,
    *,
    constrained_segment_ids: np.ndarray | None = None,
    random_state=None,
) -> list[CNAEvent]:
    """Sample contiguous gain events on each incoming clone branch.

    When ``max_local_cn_states_per_mutation`` is configured, callers may
    provide the mutation-bearing segments in ``constrained_segment_ids``.
    When they are omitted, all segments are constrained.  At most
    ``max_local_cn_states_per_mutation - 1`` gain events are retained per
    constrained segment.  Because branch-event descendant sets are laminar on
    one clone tree, this conservatively guarantees no more than the requested
    number of final allele-specific states at those loci.
    """

    _validate_copy_number_config(config)
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    parent = np.asarray(parent, dtype=int)
    events: list[CNAEvent] = []
    event_id = 0
    geometric_p = 1.0 / float(config.mean_cna_span_segments)
    for clone_id, parent_clone_id in enumerate(parent):
        rate_multiplier = (
            config.trunk_cna_rate_multiplier if parent_clone_id == -1 else 1.0
        )
        expected_events = (
            float(config.cna_event_rate)
            * float(config.n_segments)
            * rate_multiplier
            / float(config.mean_cna_span_segments)
        )
        n_events = int(rng.poisson(expected_events))
        for _ in range(n_events):
            span = min(int(rng.geometric(geometric_p)), int(config.n_segments))
            start = int(rng.integers(0, config.n_segments - span + 1))
            events.append(
                CNAEvent(
                    event_id=event_id,
                    clone_id=int(clone_id),
                    parent_clone_id=int(parent_clone_id),
                    branch_time=float(rng.random()),
                    segment_ids=tuple(range(start, start + span)),
                    allele=int(rng.integers(0, 2)),
                )
            )
            event_id += 1

    max_local_states = config.max_local_cn_states_per_mutation
    if max_local_states is None:
        return events
    if constrained_segment_ids is None:
        constrained_segment_ids = np.arange(int(config.n_segments), dtype=int)

    constrained = np.zeros(int(config.n_segments), dtype=bool)
    constrained_ids = np.asarray(constrained_segment_ids, dtype=int).reshape(-1)
    if np.any((constrained_ids < 0) | (constrained_ids >= int(config.n_segments))):
        raise ValueError("constrained_segment_ids contains an invalid segment ID.")
    constrained[constrained_ids] = True

    event_budget = int(max_local_states) - 1
    retained_event_count = np.zeros(int(config.n_segments), dtype=int)
    accepted: list[CNAEvent] = []
    for event_index in rng.permutation(len(events)):
        event = events[int(event_index)]
        segment_ids = np.asarray(event.segment_ids, dtype=int)
        affected_constrained = segment_ids[constrained[segment_ids]]
        if affected_constrained.size and np.any(
            retained_event_count[affected_constrained] >= event_budget
        ):
            continue
        accepted.append(event)
        retained_event_count[affected_constrained] += 1
    return sorted(accepted, key=lambda event: event.event_id)


def _copy_genome_state(genome):
    return [
        [list(map(set, homolog_copies)) for homolog_copies in segment]
        for segment in genome
    ]


def _tree_order_and_ancestry(parent: np.ndarray) -> tuple[list[int], np.ndarray]:
    parent = np.asarray(parent, dtype=int)
    K = int(parent.shape[0])
    if K < 1 or parent[0] != -1 or np.any(parent[1:] < 0):
        raise ValueError("parent must describe one rooted tree with clone 0 as root.")

    children = [[] for _ in range(K)]
    for clone_id in range(1, K):
        parent_id = int(parent[clone_id])
        if parent_id >= K or parent_id == clone_id:
            raise ValueError(f"Invalid parent[{clone_id}]={parent_id}.")
        children[parent_id].append(clone_id)

    order: list[int] = []
    stack = [0]
    while stack:
        clone_id = stack.pop()
        order.append(clone_id)
        stack.extend(reversed(children[clone_id]))
    if len(order) != K:
        raise ValueError(
            "parent contains a cycle or a clone disconnected from the root."
        )

    ancestry = np.zeros((K, K), dtype=bool)
    for clone_id in order:
        ancestry[clone_id, clone_id] = True
        parent_id = int(parent[clone_id])
        if parent_id >= 0:
            ancestry[:, clone_id] |= ancestry[:, parent_id]
    return order, ancestry


def simulate_joint_snv_cna_evolution(
    *,
    parent: np.ndarray,
    mutation_origin_clone: np.ndarray,
    mutation_segment: np.ndarray,
    branch_cna_events: list[CNAEvent],
    n_segments: int,
    max_allele_cn: int = 6,
    ensure_positive_descendant_dosage: bool = True,
    mutation_branch_time: np.ndarray | None = None,
    mutation_origin_allele: np.ndarray | None = None,
    random_state=None,
) -> JointEvolutionResult:
    """Interleave SNVs and gain events while inheriting physical copies down the tree."""

    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    if n_segments < 1:
        raise ValueError("n_segments must be at least 1.")
    if max_allele_cn < 1:
        raise ValueError("max_allele_cn must be at least 1.")

    parent = np.asarray(parent, dtype=int)
    mutation_origin_clone = np.asarray(mutation_origin_clone, dtype=int)
    mutation_segment = np.asarray(mutation_segment, dtype=int)
    if (
        mutation_origin_clone.ndim != 1
        or mutation_segment.shape != mutation_origin_clone.shape
    ):
        raise ValueError(
            "Mutation origin clones and segments must be aligned one-dimensional arrays."
        )
    K = int(parent.shape[0])
    M = int(mutation_origin_clone.shape[0])
    if np.any((mutation_origin_clone < 0) | (mutation_origin_clone >= K)):
        raise ValueError("mutation_origin_clone contains an invalid clone ID.")
    if np.any((mutation_segment < 0) | (mutation_segment >= n_segments)):
        raise ValueError("mutation_segment contains an invalid segment ID.")

    order, ancestry = _tree_order_and_ancestry(parent)
    if mutation_branch_time is None:
        mutation_branch_time = rng.random(M)
    else:
        mutation_branch_time = np.asarray(mutation_branch_time, dtype=float)
    if mutation_origin_allele is None:
        mutation_origin_allele = rng.integers(0, 2, size=M, dtype=int)
    else:
        mutation_origin_allele = np.asarray(mutation_origin_allele, dtype=int)
    if mutation_branch_time.shape != (M,) or np.any(
        (mutation_branch_time < 0.0) | (mutation_branch_time > 1.0)
    ):
        raise ValueError(
            "mutation_branch_time must contain one value in [0, 1] per mutation."
        )
    if mutation_origin_allele.shape != (M,) or np.any(
        (mutation_origin_allele < 0) | (mutation_origin_allele > 1)
    ):
        raise ValueError("mutation_origin_allele must contain only 0 (A) or 1 (B).")

    events_by_clone: list[list[CNAEvent]] = [[] for _ in range(K)]
    seen_event_ids: set[int] = set()
    for event in branch_cna_events:
        if event.event_id in seen_event_ids:
            raise ValueError(f"Duplicate CNA event ID {event.event_id}.")
        seen_event_ids.add(event.event_id)
        if event.event_type != "gain":
            raise ValueError("The matched simulator accepts gain events only.")
        if not 0 <= event.clone_id < K:
            raise ValueError(f"CNA event {event.event_id} has an invalid clone ID.")
        if event.parent_clone_id != int(parent[event.clone_id]):
            raise ValueError(
                f"CNA event {event.event_id} does not match the clone parent vector."
            )
        if not 0.0 <= event.branch_time <= 1.0:
            raise ValueError(
                f"CNA event {event.event_id} has a branch time outside [0, 1]."
            )
        if event.allele not in (0, 1):
            raise ValueError(f"CNA event {event.event_id} has an invalid allele.")
        if (
            not event.segment_ids
            or len(set(event.segment_ids)) != len(event.segment_ids)
            or any(
                segment_id < 0 or segment_id >= n_segments
                for segment_id in event.segment_ids
            )
        ):
            raise ValueError(f"CNA event {event.event_id} has an invalid segment span.")
        events_by_clone[event.clone_id].append(event)

    normal_genome = [[[set()], [set()]] for _ in range(n_segments)]
    clone_genomes = [None] * K
    clone_allele_cn = np.empty((K, n_segments, 2), dtype=int)
    mutation_copy_index = np.full(M, -1, dtype=int)
    mutation_cn_at_origin = np.full(M, -1, dtype=int)
    event_rows: list[dict[str, object]] = []

    mutations_by_clone = [
        np.flatnonzero(mutation_origin_clone == clone_id) for clone_id in range(K)
    ]
    for clone_id in order:
        parent_id = int(parent[clone_id])
        genome = _copy_genome_state(
            normal_genome if parent_id == -1 else clone_genomes[parent_id]
        )
        timeline: list[tuple[float, int, int, object]] = []
        for event in events_by_clone[clone_id]:
            timeline.append((float(event.branch_time), 0, int(event.event_id), event))
        for mutation_id in mutations_by_clone[clone_id]:
            timeline.append(
                (
                    float(mutation_branch_time[mutation_id]),
                    1,
                    int(mutation_id),
                    int(mutation_id),
                )
            )
        timeline.sort(key=lambda item: (item[0], item[1], item[2]))

        for _, kind, _, payload in timeline:
            if kind == 0:
                event = payload
                for segment_id in event.segment_ids:
                    copies = genome[int(segment_id)][int(event.allele)]
                    if len(copies) >= max_allele_cn:
                        continue
                    source_copy_index = int(rng.integers(0, len(copies)))
                    carried_mutations = set(copies[source_copy_index])
                    cn_before = len(copies)
                    copies.append(set(carried_mutations))
                    event_rows.append(
                        {
                            "event_id": int(event.event_id),
                            "clone_id": int(clone_id),
                            "parent_clone_id": int(parent_id),
                            "branch_time": float(event.branch_time),
                            "segment_id": int(segment_id),
                            "allele": "A" if event.allele == 0 else "B",
                            "event_type": "gain",
                            "cn_before": int(cn_before),
                            "cn_after": int(cn_before + 1),
                            "source_copy_index": int(source_copy_index),
                            "duplicated_copy_carried_mutations": ",".join(
                                str(mutation_id)
                                for mutation_id in sorted(carried_mutations)
                            ),
                        }
                    )
            else:
                mutation_id = int(payload)
                segment_id = int(mutation_segment[mutation_id])
                allele = int(mutation_origin_allele[mutation_id])
                copies = genome[segment_id][allele]
                copy_index = int(rng.integers(0, len(copies)))
                mutation_copy_index[mutation_id] = copy_index
                mutation_cn_at_origin[mutation_id] = len(copies)
                copies[copy_index].add(mutation_id)

        clone_genomes[clone_id] = genome
        for segment_id in range(n_segments):
            clone_allele_cn[clone_id, segment_id, 0] = len(genome[segment_id][0])
            clone_allele_cn[clone_id, segment_id, 1] = len(genome[segment_id][1])

    mutation_dosage = np.zeros((M, K), dtype=int)
    for clone_id, genome in enumerate(clone_genomes):
        for segment in genome:
            for homolog_copies in segment:
                for physical_copy in homolog_copies:
                    if physical_copy:
                        mutation_ids = np.fromiter(physical_copy, dtype=int)
                        mutation_dosage[mutation_ids, clone_id] += 1
    mutation_carrier = mutation_dosage > 0
    expected_carrier = ancestry[mutation_origin_clone, :]
    if not np.array_equal(mutation_carrier, expected_carrier):
        raise AssertionError("Gain-only evolution violated mutation inheritance.")
    if ensure_positive_descendant_dosage and np.any(
        mutation_dosage[mutation_carrier] < 1
    ):
        raise AssertionError("A carrier clone has nonpositive mutation dosage.")

    applied_gains = np.zeros_like(clone_allele_cn)
    for row in event_rows:
        allele = 0 if row["allele"] == "A" else 1
        applied_gains[int(row["clone_id"]), int(row["segment_id"]), allele] += 1
    diploid_profile = np.ones((n_segments, 2), dtype=int)
    for clone_id in order:
        parent_id = int(parent[clone_id])
        inherited_profile = (
            diploid_profile if parent_id == -1 else clone_allele_cn[parent_id]
        )
        if not np.array_equal(
            clone_allele_cn[clone_id],
            inherited_profile + applied_gains[clone_id],
        ):
            raise AssertionError(
                "Clone CN profile is inconsistent with its inherited branch-event history."
            )

    event_columns = [
        "event_id",
        "clone_id",
        "parent_clone_id",
        "branch_time",
        "segment_id",
        "allele",
        "event_type",
        "cn_before",
        "cn_after",
        "source_copy_index",
        "duplicated_copy_carried_mutations",
    ]
    event_history = pd.DataFrame(event_rows, columns=event_columns)
    clone_total_cn = np.sum(clone_allele_cn, axis=2)
    return JointEvolutionResult(
        clone_allele_cn=clone_allele_cn,
        clone_total_cn=clone_total_cn,
        mutation_dosage_numeric=mutation_dosage,
        mutation_carrier=mutation_carrier,
        mutation_branch_time=np.asarray(mutation_branch_time, dtype=float),
        mutation_origin_allele=np.asarray(mutation_origin_allele, dtype=int),
        mutation_origin_physical_copy=mutation_copy_index,
        mutation_cn_at_origin=mutation_cn_at_origin,
        cna_event_history=event_history,
    )


def canonicalize_cn_clone_profiles(
    clone_allele_cn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    profiles = np.asarray(clone_allele_cn, dtype=int)
    if profiles.ndim != 3 or profiles.shape[2] != 2:
        raise ValueError("clone_allele_cn must have shape (K, L, 2).")

    profile_to_id: dict[tuple[int, ...], int] = {}
    cn_clone_id = np.empty(profiles.shape[0], dtype=int)
    unique_profiles: list[np.ndarray] = []
    for clone_id, profile in enumerate(profiles):
        key = tuple(int(value) for value in profile.reshape(-1))
        if key not in profile_to_id:
            profile_to_id[key] = len(unique_profiles)
            unique_profiles.append(profile.copy())
        cn_clone_id[clone_id] = profile_to_id[key]
    return cn_clone_id, np.stack(unique_profiles, axis=0)


def aggregate_cn_clone_fractions(
    exclusive_clone_fraction_samples: np.ndarray,
    cn_clone_id: np.ndarray,
) -> np.ndarray:
    exclusive = np.asarray(exclusive_clone_fraction_samples, dtype=float)
    profile_id = np.asarray(cn_clone_id, dtype=int)
    if exclusive.ndim != 2 or profile_id.shape != (exclusive.shape[0],):
        raise ValueError("Exclusive clone fractions and CN clone IDs are not aligned.")
    n_cn_clones = int(np.max(profile_id)) + 1
    fractions = np.zeros((n_cn_clones, exclusive.shape[1]), dtype=float)
    for clone_id, cn_id in enumerate(profile_id):
        fractions[int(cn_id)] += exclusive[clone_id]
    if not np.allclose(np.sum(fractions, axis=0), 1.0, atol=1e-8, rtol=0.0):
        raise AssertionError("CN-clone fractions do not sum to one in every sample.")
    return fractions


def compute_mutation_sample_truth(
    *,
    clone_fraction: np.ndarray,
    clone_total_cn: np.ndarray,
    mutation_segment: np.ndarray,
    mutation_dosage: np.ndarray,
    mutation_carrier: np.ndarray,
    purity: float,
    normal_cn: np.ndarray,
) -> dict[str, np.ndarray]:
    clone_fraction = np.asarray(clone_fraction, dtype=float)
    clone_total_cn = np.asarray(clone_total_cn, dtype=float)
    mutation_segment = np.asarray(mutation_segment, dtype=int)
    mutation_dosage = np.asarray(mutation_dosage, dtype=float)
    mutation_carrier = np.asarray(mutation_carrier, dtype=bool)
    normal_cn = np.asarray(normal_cn, dtype=float)
    K = clone_fraction.shape[0]
    M = mutation_segment.shape[0]
    if (
        clone_fraction.ndim != 1
        or clone_total_cn.ndim != 2
        or clone_total_cn.shape[0] != K
    ):
        raise ValueError("Clone fractions and clone total-CN profiles are not aligned.")
    if mutation_dosage.shape != (M, K) or mutation_carrier.shape != (M, K):
        raise ValueError("Mutation dosage/carrier matrices must have shape (M, K).")
    if normal_cn.shape != (M,):
        raise ValueError("normal_cn must contain one value per mutation.")
    if np.any((mutation_segment < 0) | (mutation_segment >= clone_total_cn.shape[1])):
        raise ValueError("mutation_segment contains an invalid segment ID.")
    if np.any(clone_fraction < -1e-10) or not np.isclose(
        np.sum(clone_fraction), 1.0, atol=1e-8
    ):
        raise ValueError("clone_fraction must be nonnegative and sum to one.")
    if not 0.0 < float(purity) < 1.0:
        raise ValueError("purity must lie strictly between zero and one.")

    clone_total_at_mutation = clone_total_cn[:, mutation_segment]
    mean_tumor_total_cn = clone_fraction @ clone_total_at_mutation
    mutant_copy_mass = mutation_dosage @ clone_fraction
    ccf = mutation_carrier.astype(float) @ clone_fraction
    denominator = (1.0 - float(purity)) * normal_cn + float(
        purity
    ) * mean_tumor_total_cn
    expected_vaf = float(purity) * mutant_copy_mass / denominator
    effective_multiplicity = np.divide(
        mutant_copy_mass,
        ccf,
        out=np.full(M, np.nan, dtype=float),
        where=ccf > 0.0,
    )

    if np.any(mean_tumor_total_cn <= 0.0):
        raise AssertionError("Mean tumor total CN must be positive.")
    if np.any(mutant_copy_mass < -1e-10) or np.any(
        mutant_copy_mass > mean_tumor_total_cn + 1e-8
    ):
        raise AssertionError("Mutant-copy mass is outside the total-copy mass.")
    if np.any((expected_vaf < -1e-10) | (expected_vaf > 1.0 + 1e-10)):
        raise AssertionError("Expected VAF is outside [0, 1].")

    return {
        "ccf": ccf,
        "mutant_copy_mass": mutant_copy_mass,
        "effective_multiplicity": effective_multiplicity,
        "mean_tumor_total_cn": mean_tumor_total_cn,
        "expected_vaf": np.clip(expected_vaf, 0.0, 1.0),
    }


def _simulate_constrained_cn_evolution(
    *,
    cna_rng: np.random.Generator,
    mutation_time_rng: np.random.Generator,
    physical_copy_rng: np.random.Generator,
    parent: np.ndarray,
    mutation_origin_clone: np.ndarray,
    mutation_segment: np.ndarray,
    config: CopyNumberEvolutionConfig,
    max_rejection_tries: int,
) -> tuple[JointEvolutionResult, np.ndarray, np.ndarray, dict[str, int]]:
    _validate_copy_number_config(config)
    K = int(np.asarray(parent).shape[0])
    required_cn_clones = (
        K if config.require_unique_cn_profiles else int(config.min_cn_clone_count)
    )
    if required_cn_clones > K:
        raise ValueError(
            "The requested minimum CN-clone count exceeds the evolutionary clone count."
        )
    if config.cna_event_rate == 0.0 and required_cn_clones > 1:
        raise ValueError(
            "A zero CNA event rate can generate only one diploid CN profile."
        )
    if config.max_allele_cn == 1 and required_cn_clones > 1:
        raise ValueError("max_allele_cn=1 can generate only one diploid CN profile.")
    min_two_state_fraction = config.min_two_state_snv_fraction
    if min_two_state_fraction is not None:
        if np.asarray(mutation_segment).size == 0:
            raise ValueError(
                "min_two_state_snv_fraction requires at least one mutation."
            )
        if K < 2:
            raise ValueError(
                "min_two_state_snv_fraction requires at least two evolutionary clones."
            )
        if config.cna_event_rate == 0.0:
            raise ValueError(
                "A positive min_two_state_snv_fraction cannot be met with a zero CNA event rate."
            )

    rejection_limit = max(int(max_rejection_tries), 1)
    best_two_state_fraction = float("-inf")
    best_cn_clone_count = 0
    mutation_count = int(np.asarray(mutation_origin_clone).shape[0])
    mutation_branch_time = mutation_time_rng.random(mutation_count)
    mutation_origin_allele = mutation_time_rng.integers(
        0,
        2,
        size=mutation_count,
        dtype=int,
    )
    for attempt in range(1, rejection_limit + 1):
        events = simulate_branch_cna_events(
            parent,
            config,
            constrained_segment_ids=np.unique(mutation_segment),
            random_state=cna_rng,
        )
        evolution = simulate_joint_snv_cna_evolution(
            parent=parent,
            mutation_origin_clone=mutation_origin_clone,
            mutation_segment=mutation_segment,
            branch_cna_events=events,
            n_segments=config.n_segments,
            max_allele_cn=config.max_allele_cn,
            ensure_positive_descendant_dosage=config.ensure_positive_descendant_dosage,
            mutation_branch_time=mutation_branch_time,
            mutation_origin_allele=mutation_origin_allele,
            random_state=physical_copy_rng,
        )
        cn_clone_id, unique_profiles = canonicalize_cn_clone_profiles(
            evolution.clone_allele_cn
        )
        best_cn_clone_count = max(best_cn_clone_count, int(unique_profiles.shape[0]))
        state_count_by_segment = np.asarray(
            [
                np.unique(unique_profiles[:, segment_id, :], axis=0).shape[0]
                for segment_id in range(int(config.n_segments))
            ],
            dtype=int,
        )
        max_local_states = config.max_local_cn_states_per_mutation
        if max_local_states is not None:
            mutation_state_counts = state_count_by_segment[
                np.asarray(mutation_segment, dtype=int)
            ]
            if np.any(mutation_state_counts > int(max_local_states)):
                raise AssertionError(
                    "The constrained CNA event generator exceeded "
                    "max_local_cn_states_per_mutation."
                )
        else:
            mutation_state_counts = state_count_by_segment[
                np.asarray(mutation_segment, dtype=int)
            ]

        two_state_fraction = float(np.mean(mutation_state_counts == 2))
        best_two_state_fraction = max(best_two_state_fraction, two_state_fraction)
        meets_two_state_fraction = (
            min_two_state_fraction is None
            or two_state_fraction >= float(min_two_state_fraction)
        )
        if unique_profiles.shape[0] >= required_cn_clones and meets_two_state_fraction:
            return (
                evolution,
                cn_clone_id,
                unique_profiles,
                {
                    "attempts": int(attempt),
                    "accepted_cna_events": int(len(events)),
                },
            )
    raise RuntimeError(
        "Failed to generate copy-number evolution satisfying the configured "
        f"invariants within {rejection_limit} attempts: "
        f"required_cn_clone_count={required_cn_clones}, "
        f"best_cn_clone_count={best_cn_clone_count}, "
        "min_two_state_snv_fraction="
        f"{min_two_state_fraction}, "
        "best_two_state_snv_fraction="
        f"{best_two_state_fraction:.6f}, "
        "max_local_cn_states_per_mutation="
        f"{config.max_local_cn_states_per_mutation}. "
        "The two-state fraction is weighted by mutation IDs, not unique segments."
    )


__all__ = [
    "CNAEvent",
    "GenomeSegment",
    "JointEvolutionResult",
    "aggregate_cn_clone_fractions",
    "assign_mutations_to_segments",
    "canonicalize_cn_clone_profiles",
    "compute_mutation_sample_truth",
    "simulate_branch_cna_events",
    "simulate_genome_segments",
    "simulate_joint_snv_cna_evolution",
]
