"""Canonical CliPP2 simulation tables, manifests, and contract validation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..io.tumor_input import (
    REGION_TABLE_COLUMNS,
    REQUIRED_REGION_FILES,
    REQUIRED_ROOT_FILES,
    ROOT_TABLE_COLUMNS,
    load_tumor_directory,
)
from .config import (
    GENERATOR_VERSION,
    OUTPUT_SCHEMA_VERSION,
)
from .evolution import (
    GenomeSegment,
    JointEvolutionResult,
    _tree_order_and_ancestry,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _generator_provenance() -> tuple[str, str | None]:
    source_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source_path in sorted(source_dir.glob("*.py")):
        digest.update(source_path.name.encode("utf-8"))
        digest.update(b"\0")
        with source_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    source_hash = digest.hexdigest()
    commit = os.environ.get("GIT_COMMIT") or os.environ.get("CI_COMMIT_SHA")
    if commit:
        return source_hash, commit
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return source_hash, None
    return source_hash, completed.stdout.strip() or None


def _output_file_hashes(data_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    input_hashes: dict[str, str] = {}
    truth_hashes: dict[str, str] = {}
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name == "scenario_manifest.json":
            continue
        relative_path = path.relative_to(data_dir).as_posix()
        destination = truth_hashes if path.name.startswith("truth") else input_hashes
        destination[relative_path] = _sha256_file(path)
    return input_hashes, truth_hashes


def _require_exact_columns(path: Path, expected: tuple[str, ...]) -> None:
    observed = tuple(pd.read_csv(path, sep="\t", nrows=0).columns)
    if observed != expected:
        raise ValueError(
            f"{path} columns must be exactly {list(expected)}; observed {list(observed)}."
        )


def validate_generated_tumor_directory(tumor_dir: str | Path) -> None:
    """Require a generated bundle to contain exactly the public input contract."""

    tumor_dir = Path(tumor_dir)
    for filename, columns in ROOT_TABLE_COLUMNS.items():
        _require_exact_columns(tumor_dir / filename, columns)

    unexpected_root = {
        path.name
        for path in tumor_dir.iterdir()
        if path.is_file()
        and not path.name.startswith("truth")
        and path.name != "scenario_manifest.json"
        and path.name not in REQUIRED_ROOT_FILES
    }
    if unexpected_root:
        raise ValueError(
            f"Generated tumor has non-contract root files: {sorted(unexpected_root)}."
        )

    data = load_tumor_directory(tumor_dir)
    for region_id in data.region_ids:
        region_dir = tumor_dir / region_id
        for filename, columns in REGION_TABLE_COLUMNS.items():
            _require_exact_columns(region_dir / filename, columns)
        unexpected_region = {
            path.name
            for path in region_dir.iterdir()
            if path.is_file()
            and not path.name.startswith("truth")
            and path.name not in REQUIRED_REGION_FILES
        }
        if unexpected_region:
            raise ValueError(
                f"{region_id} has non-contract files: {sorted(unexpected_region)}."
            )

    unsupported = np.asarray(data.path_unsupported_reason, dtype=object)
    if any(value not in {None, ""} for value in unsupported.reshape(-1)):
        raise ValueError(
            "Generated tumor contains unsupported local copy-number states."
        )


def _cn_clone_profile_table(
    profiles: np.ndarray,
    segments: list[GenomeSegment],
    *,
    clone_ids: np.ndarray | None = None,
    cn_clone_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    profiles = np.asarray(profiles, dtype=int)
    rows: list[dict[str, int]] = []
    for profile_index, profile in enumerate(profiles):
        for segment in segments:
            allele_a = int(profile[segment.segment_id, 0])
            allele_b = int(profile[segment.segment_id, 1])
            row = {
                "segment_id": int(segment.segment_id),
                "chromosome": int(segment.chromosome),
                "start": int(segment.start),
                "end": int(segment.end),
                "allele_a_cn": allele_a,
                "allele_b_cn": allele_b,
                "major_cn": max(allele_a, allele_b),
                "minor_cn": min(allele_a, allele_b),
                "total_cn": allele_a + allele_b,
            }
            if clone_ids is None:
                row = {"cn_clone_id": int(profile_index), **row}
            else:
                row = {
                    "clone_id": int(clone_ids[profile_index]),
                    "cn_clone_id": int(cn_clone_ids[profile_index]),
                    **row,
                }
            rows.append(row)
    return pd.DataFrame(rows)


def _local_cn_state_table(
    *,
    sample_id: int,
    unique_profiles: np.ndarray,
    cn_clone_fraction: np.ndarray,
    segments: list[GenomeSegment],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, object]] = []
    dominant_a = np.empty(len(segments), dtype=int)
    dominant_b = np.empty(len(segments), dtype=int)
    dominant_fraction = np.empty(len(segments), dtype=float)
    dominant_state_id = np.empty(len(segments), dtype=int)
    for segment in segments:
        groups: dict[tuple[int, int], dict[str, object]] = {}
        for cn_clone_id, profile in enumerate(unique_profiles):
            state = (
                int(profile[segment.segment_id, 0]),
                int(profile[segment.segment_id, 1]),
            )
            group = groups.setdefault(state, {"fraction": 0.0, "members": []})
            group["fraction"] = float(group["fraction"]) + float(
                cn_clone_fraction[cn_clone_id]
            )
            group["members"].append(int(cn_clone_id))

        ordered_groups = list(groups.items())
        for state_id, ((allele_a, allele_b), group) in enumerate(ordered_groups):
            rows.append(
                {
                    "sample_id": int(sample_id),
                    "segment_id": int(segment.segment_id),
                    "state_id": int(state_id),
                    "allele_a_cn": allele_a,
                    "allele_b_cn": allele_b,
                    "major_cn": max(allele_a, allele_b),
                    "minor_cn": min(allele_a, allele_b),
                    "tumor_fraction": float(group["fraction"]),
                    "member_cn_clone_ids": ",".join(
                        str(value) for value in group["members"]
                    ),
                }
            )
        best_state_id, (best_state, best_group) = max(
            enumerate(ordered_groups),
            key=lambda item: float(item[1][1]["fraction"]),
        )
        dominant_a[segment.segment_id] = int(best_state[0])
        dominant_b[segment.segment_id] = int(best_state[1])
        dominant_fraction[segment.segment_id] = float(best_group["fraction"])
        dominant_state_id[segment.segment_id] = int(best_state_id)

    table = pd.DataFrame(rows)
    if not np.allclose(
        table.groupby("segment_id", sort=False)["tumor_fraction"].sum().to_numpy(),
        1.0,
        atol=1e-8,
        rtol=0.0,
    ):
        raise AssertionError("Region-local CN-state fractions do not sum to one.")
    return table, dominant_a, dominant_b, dominant_fraction, dominant_state_id


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return {"minimum": 0.0, "mean": 0.0, "maximum": 0.0}
    return {
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


def _local_state_distribution(state_counts: np.ndarray) -> dict[str, object]:
    counts = np.asarray(state_counts, dtype=int).reshape(-1)
    denominator = max(int(counts.size), 1)
    buckets = {
        "1": int(np.sum(counts == 1)),
        "2": int(np.sum(counts == 2)),
        "3": int(np.sum(counts == 3)),
        "4_plus": int(np.sum(counts >= 4)),
    }
    return {
        "count": int(counts.size),
        "fractions": {
            name: float(value / denominator) for name, value in buckets.items()
        },
        "histogram": {
            str(value): int(np.sum(counts == value)) for value in np.unique(counts)
        },
    }


def _snv_cna_timing_summary(
    *,
    parent: np.ndarray,
    mutation_origin_clone: np.ndarray,
    mutation_segment: np.ndarray,
    mutation_branch_time: np.ndarray,
    event_history: pd.DataFrame,
) -> dict[str, float | int]:
    mutation_count = int(np.asarray(mutation_origin_clone).shape[0])
    if mutation_count == 0 or event_history.empty:
        return {
            "comparable_snv_cna_pairs": 0,
            "pre_gain_pair_fraction": 0.0,
            "post_gain_pair_fraction": 0.0,
            "fraction_mutations_with_pre_gain_event": 0.0,
            "fraction_mutations_with_post_gain_event": 0.0,
        }

    _, ancestry = _tree_order_and_ancestry(np.asarray(parent, dtype=int))
    event_rows = event_history[
        ["event_id", "clone_id", "segment_id", "branch_time"]
    ].drop_duplicates()
    events_by_segment = {
        int(segment_id): group
        for segment_id, group in event_rows.groupby("segment_id", sort=False)
    }
    pre_gain_mutation = np.zeros(mutation_count, dtype=bool)
    post_gain_mutation = np.zeros(mutation_count, dtype=bool)
    pre_gain_pairs = 0
    post_gain_pairs = 0
    for mutation_id in range(mutation_count):
        origin_clone = int(mutation_origin_clone[mutation_id])
        segment_id = int(mutation_segment[mutation_id])
        segment_events = events_by_segment.get(segment_id)
        if segment_events is None:
            continue
        for event in segment_events.itertuples(index=False):
            event_clone = int(event.clone_id)
            if event_clone == origin_clone:
                mutation_precedes = float(mutation_branch_time[mutation_id]) < float(
                    event.branch_time
                )
            elif ancestry[origin_clone, event_clone]:
                mutation_precedes = True
            elif ancestry[event_clone, origin_clone]:
                mutation_precedes = False
            else:
                continue
            if mutation_precedes:
                pre_gain_pairs += 1
                pre_gain_mutation[mutation_id] = True
            else:
                post_gain_pairs += 1
                post_gain_mutation[mutation_id] = True

    pair_count = pre_gain_pairs + post_gain_pairs
    return {
        "comparable_snv_cna_pairs": int(pair_count),
        "pre_gain_pair_fraction": float(pre_gain_pairs / max(pair_count, 1)),
        "post_gain_pair_fraction": float(post_gain_pairs / max(pair_count, 1)),
        "fraction_mutations_with_pre_gain_event": float(np.mean(pre_gain_mutation)),
        "fraction_mutations_with_post_gain_event": float(np.mean(post_gain_mutation)),
    }


def _realized_cn_complexity(
    *,
    parent: np.ndarray,
    mutation_origin_clone: np.ndarray,
    mutation_segment: np.ndarray,
    evolution: JointEvolutionResult,
    unique_cn_profiles: np.ndarray,
    cn_clone_fraction_samples: np.ndarray,
    accepted_cna_events: int,
    mutation_sample_truth: pd.DataFrame,
) -> dict[str, object]:
    profiles = np.asarray(unique_cn_profiles, dtype=int)
    state_count_by_segment = np.asarray(
        [
            np.unique(profiles[:, segment_id, :], axis=0).shape[0]
            for segment_id in range(profiles.shape[1])
        ],
        dtype=int,
    )
    mutation_state_counts = state_count_by_segment[
        np.asarray(mutation_segment, dtype=int)
    ]
    altered_segments = np.any(profiles != 1, axis=(0, 2))
    carrier_dosages = evolution.mutation_dosage_numeric[evolution.mutation_carrier]
    effective_multiplicity = mutation_sample_truth["effective_multiplicity"].to_numpy(
        dtype=float
    )
    finite_multiplicity = np.isfinite(effective_multiplicity)
    noninteger = (
        np.abs(
            effective_multiplicity[finite_multiplicity]
            - np.rint(effective_multiplicity[finite_multiplicity])
        )
        > 1e-8
    )
    fractions = np.asarray(cn_clone_fraction_samples, dtype=float)
    safe_fractions = np.where(fractions > 0.0, fractions, 1.0)
    entropy = -np.sum(fractions * np.log(safe_fractions), axis=0)
    event_history = evolution.cna_event_history
    applied_event_count = (
        int(event_history["event_id"].nunique()) if not event_history.empty else 0
    )
    mutation_on_altered_segment = altered_segments[
        np.asarray(mutation_segment, dtype=int)
    ]
    maximum_dosage_by_mutation = np.max(
        evolution.mutation_dosage_numeric,
        axis=1,
        initial=0,
    )
    return {
        "accepted_cna_event_count": int(accepted_cna_events),
        "applied_cna_event_count": applied_event_count,
        "applied_cna_segment_event_count": int(event_history.shape[0]),
        "altered_segment_fraction": float(np.mean(altered_segments)),
        "whole_genome_cn_clone_count": int(profiles.shape[0]),
        "local_state_counts": {
            "mutation_weighted": _local_state_distribution(mutation_state_counts),
            "segment_weighted": _local_state_distribution(state_count_by_segment),
        },
        "maximum_local_state_count": int(np.max(state_count_by_segment)),
        "maximum_allele_cn": int(np.max(profiles)),
        "maximum_clone_specific_dosage": (
            int(np.max(carrier_dosages)) if carrier_dosages.size else 0
        ),
        "fraction_mutations_on_altered_segments": float(
            np.mean(mutation_on_altered_segment)
        ),
        "fraction_mutations_with_dosage_gt_one": float(
            np.mean(maximum_dosage_by_mutation > 1)
        ),
        "fraction_noninteger_effective_multiplicity": (
            float(np.mean(noninteger)) if noninteger.size else 0.0
        ),
        "cn_clone_fraction_entropy": _numeric_summary(entropy),
        "snv_cna_timing": _snv_cna_timing_summary(
            parent=parent,
            mutation_origin_clone=mutation_origin_clone,
            mutation_segment=mutation_segment,
            mutation_branch_time=evolution.mutation_branch_time,
            event_history=event_history,
        ),
    }


def _write_scenario_manifest(
    *,
    data_dir: Path,
    intended_factors: dict[str, object],
    realized_factors: dict[str, object],
    rejection_counts: dict[str, int],
    rng_metadata: dict[str, object],
) -> None:
    input_hashes, truth_hashes = _output_file_hashes(data_dir)
    generator_hash, git_commit = _generator_provenance()
    manifest = {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "git_commit": git_commit,
        "generator_source_sha256": generator_hash,
        "scenario_id": data_dir.name,
        "intended_factors": intended_factors,
        "rng": rng_metadata,
        "realized_factors": realized_factors,
        "rejection_counts": rejection_counts,
        "input_file_hashes": input_hashes,
        "truth_file_hashes": truth_hashes,
    }
    (data_dir / "scenario_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["validate_generated_tumor_directory"]
