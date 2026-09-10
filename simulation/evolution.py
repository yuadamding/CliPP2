"""Tree-inherited carriers, clonal trunk CN, and sampled integer multiplicity."""

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
    mutation_multiplicity: np.ndarray
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
    random_state=None,
) -> list[CNAEvent]:
    """Sample trunk-only gains; all clones inherit the resulting CN profile."""
    _validate_copy_number_config(config)
    _tree_order_and_ancestry(parent)
    rng = np.random.default_rng(random_state)
    expected_events = (
        config.cna_event_rate * config.n_segments / config.mean_cna_span_segments
    )
    events = []
    for event_id in range(int(rng.poisson(expected_events))):
        span = min(
            int(rng.geometric(1.0 / config.mean_cna_span_segments)),
            int(config.n_segments),
        )
        start = int(rng.integers(0, config.n_segments - span + 1))
        events.append(
            CNAEvent(
                event_id,
                0,
                -1,
                float(rng.random()),
                tuple(range(start, start + span)),
                int(rng.integers(0, 2)),
            )
        )
    return events


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


def sample_multiplicity(
    major_cn: np.ndarray,
    minor_cn: np.ndarray,
    *,
    random_state=None,
) -> np.ndarray:
    """One uniform integer draw per unequal-CN mutation; equal-CN dosage is one."""
    major = np.asarray(major_cn)
    minor = np.asarray(minor_cn)
    if (
        major.ndim != 1
        or minor.shape != major.shape
        or not np.all(np.isfinite(major))
        or not np.all(np.isfinite(minor))
        or np.any(major != np.rint(major))
        or np.any(minor != np.rint(minor))
        or np.any(major < 1)
        or np.any(minor < 0)
        or np.any(minor > major)
    ):
        raise ValueError(
            "major_cn/minor_cn must be aligned, ordered integer CN vectors."
        )
    rng = np.random.default_rng(random_state)
    multiplicity = np.ones(major.size, dtype=int)
    eligible = major != minor
    multiplicity[eligible] = rng.integers(
        1,
        np.minimum(major[eligible], 6).astype(int) + 1,
    )
    return multiplicity


def simulate_joint_snv_cna_evolution(
    *,
    parent: np.ndarray,
    mutation_origin_clone: np.ndarray,
    mutation_segment: np.ndarray,
    branch_cna_events: list[CNAEvent],
    n_segments: int,
    max_allele_cn: int = 6,
    random_state=None,
) -> JointEvolutionResult:
    """Inherit clonal CN and carriers, then sample dosage independently of timing.

    The tree defines mutation presence, not amplification history. Each mutation
    has the same sampled dosage in every carrier clone and therefore every region.
    """
    _validate_copy_number_config(
        CopyNumberEvolutionConfig(
            n_segments=n_segments,
            max_allele_cn=max_allele_cn,
        )
    )
    parent = np.asarray(parent, dtype=int)
    _, ancestry = _tree_order_and_ancestry(parent)
    origins = np.asarray(mutation_origin_clone, dtype=int)
    segments = np.asarray(mutation_segment, dtype=int)
    if origins.ndim != 1 or segments.shape != origins.shape:
        raise ValueError("Mutation origin clones and segments must be aligned vectors.")
    if np.any((origins < 0) | (origins >= len(parent))):
        raise ValueError("mutation_origin_clone contains an invalid clone ID.")
    if np.any((segments < 0) | (segments >= n_segments)):
        raise ValueError("mutation_segment contains an invalid segment ID.")

    profile = np.ones((n_segments, 2), dtype=int)
    seen = set()
    event_rows = []
    for event in sorted(branch_cna_events, key=lambda e: (e.branch_time, e.event_id)):
        if event.event_id in seen:
            raise ValueError(f"Duplicate CNA event ID {event.event_id}.")
        seen.add(event.event_id)
        if event.clone_id != 0 or event.parent_clone_id != -1:
            raise ValueError(
                "Subclonal CN is not supported: all gains must be on the trunk."
            )
        if event.event_type != "gain" or event.allele not in (0, 1):
            raise ValueError("Only allele A/B gain events are supported.")
        if not np.isfinite(event.branch_time) or not 0 <= event.branch_time <= 1:
            raise ValueError("CNA branch_time must be finite and in [0, 1].")
        if (
            not event.segment_ids
            or len(set(event.segment_ids)) != len(event.segment_ids)
            or any(s < 0 or s >= n_segments for s in event.segment_ids)
        ):
            raise ValueError("CNA event has an invalid segment span.")
        for segment_id in event.segment_ids:
            before = int(profile[segment_id, event.allele])
            if before >= max_allele_cn:
                continue
            profile[segment_id, event.allele] += 1
            event_rows.append(
                {
                    "event_id": event.event_id,
                    "clone_id": 0,
                    "parent_clone_id": -1,
                    "branch_time": event.branch_time,
                    "segment_id": segment_id,
                    "allele": "A" if event.allele == 0 else "B",
                    "event_type": "gain",
                    "cn_before": before,
                    "cn_after": before + 1,
                }
            )
    cn = profile[segments]
    multiplicity = sample_multiplicity(
        cn.max(axis=1),
        cn.min(axis=1),
        random_state=random_state,
    )
    carrier = ancestry[origins, :]
    clone_cn = np.repeat(profile[None, :, :], len(parent), axis=0)
    return JointEvolutionResult(
        clone_allele_cn=clone_cn,
        clone_total_cn=clone_cn.sum(axis=2),
        mutation_dosage_numeric=carrier * multiplicity[:, None],
        mutation_carrier=carrier,
        mutation_multiplicity=multiplicity,
        cna_event_history=pd.DataFrame(
            event_rows,
            columns=[
                "event_id",
                "clone_id",
                "parent_clone_id",
                "branch_time",
                "segment_id",
                "allele",
                "event_type",
                "cn_before",
                "cn_after",
            ],
        ),
    )


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

    if not np.all(clone_total_cn == clone_total_cn[:1]):
        raise ValueError("Subclonal CN is not supported.")
    multiplicity = np.max(mutation_dosage, axis=1, initial=0)
    if (
        np.any(multiplicity < 1)
        or np.any(multiplicity != np.rint(multiplicity))
        or not np.array_equal(mutation_dosage, mutation_carrier * multiplicity[:, None])
    ):
        raise ValueError(
            "Every carrier must have the same positive integer multiplicity."
        )
    mean_tumor_total_cn = clone_total_cn[0, mutation_segment]
    ccf = mutation_carrier.astype(float) @ clone_fraction
    ccf[np.all(mutation_carrier, axis=1)] = 1.0
    mutant_copy_mass = multiplicity * ccf
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


__all__ = [
    "CNAEvent",
    "GenomeSegment",
    "JointEvolutionResult",
    "assign_mutations_to_segments",
    "compute_mutation_sample_truth",
    "sample_multiplicity",
    "simulate_branch_cna_events",
    "simulate_genome_segments",
    "simulate_joint_snv_cna_evolution",
]
