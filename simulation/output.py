"""Canonical CliPP2 simulation tables, manifests, and contract validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from CliPP2.io.tumor_txt import (
    SCHEMA_COLUMNS as TUMOR_TXT_COLUMNS,
    TUMOR_TXT_SCHEMA,
    load_tumor_txt,
)
from .config import (
    GENERATOR_VERSION,
    OUTPUT_SCHEMA_VERSION,
)
from .evolution import (
    GenomeSegment,
    JointEvolutionResult,
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
    project = source_dir.parents[1]
    if not (project / ".git").exists():
        return source_hash, None
    try:
        completed = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return source_hash, None
    return source_hash, completed.stdout.strip() or None


def _canonical_input_path(data_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file() and path.name.endswith(".clipp2.txt")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Generated tumor must contain exactly one root .clipp2.txt input; "
            f"observed {[path.name for path in candidates]}."
        )
    return candidates[0]


def _output_file_hashes(data_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    canonical_input = _canonical_input_path(data_dir)
    input_hashes = {
        canonical_input.relative_to(data_dir).as_posix(): _sha256_file(canonical_input)
    }
    truth_hashes: dict[str, str] = {}
    for path in sorted(data_dir.rglob("*")):
        if (
            not path.is_file()
            or path == canonical_input
            or path.name == "scenario_manifest.json"
        ):
            continue
        relative_path = path.relative_to(data_dir).as_posix()
        if not path.name.startswith("truth"):
            raise ValueError(
                "Generated bundles may contain only the canonical input, "
                f"truth files, and scenario_manifest.json; found {relative_path!r}."
            )
        truth_hashes[relative_path] = _sha256_file(path)
    return input_hashes, truth_hashes


def validate_generated_tumor_directory(tumor_dir: str | Path) -> None:
    """Validate a benchmark bundle with one canonical observed tumor input."""

    tumor_dir = Path(tumor_dir)
    canonical_input = _canonical_input_path(tumor_dir)
    unexpected_root = sorted(
        path.name
        for path in tumor_dir.iterdir()
        if path.is_file()
        and path != canonical_input
        and path.name != "scenario_manifest.json"
        and not path.name.startswith("truth")
    )
    if unexpected_root:
        raise ValueError(
            f"Generated tumor has non-contract root files: {unexpected_root}."
        )

    data = load_tumor_txt(canonical_input)
    if data.tumor_id != tumor_dir.name:
        raise ValueError(
            "Canonical tumor_id must match its benchmark bundle directory: "
            f"{data.tumor_id!r} != {tumor_dir.name!r}."
        )
    for region_id in data.region_ids:
        region_dir = tumor_dir / region_id
        if not region_dir.is_dir():
            raise ValueError(f"Missing truth directory for sample {region_id!r}.")
        unexpected_region = sorted(
            path.name
            for path in region_dir.iterdir()
            if path.is_file() and not path.name.startswith("truth")
        )
        if unexpected_region:
            raise ValueError(f"{region_id} has non-truth files: {unexpected_region}.")

    report = data.cn_filter_report
    if report is None or report.excluded_mutation_ids:
        raise ValueError(
            "Generated tumor must retain every mutation after CN filtering."
        )
    truth = pd.read_csv(tumor_dir / "truth.txt", sep="\t")
    if truth.mutation_id.duplicated().any() or set(truth.mutation_id) != set(
        data.mutation_ids
    ):
        raise ValueError("Mutation truth must match the complete retained input.")
    observed = pd.read_csv(canonical_input, sep="\t", comment="#")
    region_numbers = {f"region{i + 1}": i for i in range(len(data.region_ids))}
    if set(data.region_ids) != set(region_numbers):
        raise ValueError("Generated samples must be region1 through regionN.")
    observed["sample_id"] = observed.sample_id.map(region_numbers)
    keys = ["mutation_id", "sample_id"]
    expected_rows = len(truth) * len(region_numbers)
    if len(observed) != expected_rows or observed.duplicated(keys).any():
        raise ValueError(
            "Matched simulation input requires one clonal CN row per unit."
        )
    sample_truth = pd.read_csv(tumor_dir / "truth_mutation_sample.tsv", sep="\t")
    if (
        len(sample_truth) != expected_rows
        or sample_truth.duplicated(keys).any()
        or set(map(tuple, sample_truth[keys].to_numpy()))
        != set(map(tuple, observed[keys].to_numpy()))
    ):
        raise ValueError(
            "Mutation-sample truth must match every input unit exactly once."
        )
    aligned = observed.merge(sample_truth, on=keys, validate="one_to_one")
    major = aligned[["allele_a_cn", "allele_b_cn"]].max(axis=1)
    minor = aligned[["allele_a_cn", "allele_b_cn"]].min(axis=1)
    total = aligned.allele_a_cn + aligned.allele_b_cn
    multiplicity = aligned.multiplicity.to_numpy(dtype=float)
    ccf = aligned.ccf.to_numpy(dtype=float)
    if (
        not np.all(np.isfinite(multiplicity))
        or np.any(multiplicity != np.rint(multiplicity))
        or np.any((multiplicity < 1) | (multiplicity > np.minimum(major, 6)))
        or np.any(multiplicity[major == minor] != 1)
        or not np.all(np.isfinite(ccf))
        or np.any((ccf <= 0) | (ccf > 1 + 1e-8))
    ):
        raise ValueError(
            "Truth requires positive CCF and integer multiplicity in 1..major CN."
        )
    expected_vaf = (
        aligned.purity
        * ccf
        * multiplicity
        / (aligned.purity * total + (1 - aligned.purity) * aligned.normal_cn)
    )
    for name, expected in (
        ("effective_multiplicity", multiplicity),
        ("mutant_copy_mass", ccf * multiplicity),
        ("mean_tumor_total_cn", total),
        ("expected_vaf", expected_vaf),
    ):
        if not np.allclose(aligned[name], expected, atol=1e-8, rtol=0.0):
            raise ValueError(
                f"Truth {name} disagrees with the clonal integer-CN model."
            )
    if not aligned.groupby("mutation_id").multiplicity.nunique().eq(1).all():
        raise ValueError("Mutation truth multiplicity must be shared across regions.")
    if not observed.cn_state_fraction.eq(1.0).all():
        raise ValueError("Clonal CN fractions must be exactly one.")
    if (
        not observed.groupby("segment_id")[["allele_a_cn", "allele_b_cn"]]
        .nunique()
        .eq(1)
        .all()
        .all()
    ):
        raise ValueError("Clonal CN profiles must be identical across regions.")
    clone_truth = pd.read_csv(tumor_dir / "truth_clone_sample.txt", sep="\t")
    linked = aligned.merge(truth, on="mutation_id", validate="many_to_one").merge(
        clone_truth,
        left_on=["cluster_id", "sample_id"],
        right_on=["clone_id", "sample_id"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_clone"),
    )
    if not np.allclose(linked.ccf, linked.ccf_clone, atol=1e-8, rtol=0.0):
        raise ValueError(
            "Mutation CCF truth disagrees with acquisition-clone CCF truth."
        )
    root = clone_truth.loc[clone_truth.clone_id == 0]
    if (
        len(root) != len(region_numbers)
        or set(root.sample_id) != set(region_numbers.values())
        or not root.ccf.eq(1.0).all()
        or not truth.cluster_id.eq(0).any()
        or not linked.loc[linked.cluster_id == 0, "ccf"].eq(1.0).all()
    ):
        raise ValueError(
            "Truth requires a nonempty clonal cluster 0 with CCF exactly one."
        )
    manifest_path = tumor_dir / "scenario_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        input_hashes, truth_hashes = _output_file_hashes(tumor_dir)
        if (
            manifest["output_schema_version"] != OUTPUT_SCHEMA_VERSION
            or manifest["generator_version"] != GENERATOR_VERSION
            or manifest["intended_factors"]["mutation_count"] != len(truth)
            or manifest["realized_factors"]["retained_mutation_count"] != len(truth)
            or manifest["realized_factors"]["excluded_mutation_count"] != 0
            or manifest["input_file_hashes"] != input_hashes
            or manifest["truth_file_hashes"] != truth_hashes
        ):
            raise ValueError("Generated bundle does not match its versioned manifest.")


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
    profile: np.ndarray,
    segments: list[GenomeSegment],
) -> pd.DataFrame:
    """Exactly one clonal CN state with fraction one per genomic segment."""
    rows = []
    for segment in segments:
        allele_a, allele_b = map(int, profile[segment.segment_id])
        rows.append(
            {
                "sample_id": sample_id,
                "segment_id": segment.segment_id,
                "state_id": 0,
                "allele_a_cn": allele_a,
                "allele_b_cn": allele_b,
                "major_cn": max(allele_a, allele_b),
                "minor_cn": min(allele_a, allele_b),
                "tumor_fraction": 1.0,
                "member_cn_clone_ids": "0",
            }
        )
    return pd.DataFrame(rows)


def _canonical_observation_table(
    *,
    mutation_ids: np.ndarray,
    mutation_segment: np.ndarray,
    alt_count: np.ndarray,
    ref_count: np.ndarray,
    purity: float,
    sample_id: str,
    local_state_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build one sample's rows in the canonical long tumor schema."""

    mutation_ids = np.asarray(mutation_ids, dtype=object)
    mutation_segment = np.asarray(mutation_segment, dtype=int)
    alt_count = np.asarray(alt_count, dtype=int)
    ref_count = np.asarray(ref_count, dtype=int)
    expected_shape = (mutation_ids.size,)
    for name, values in (
        ("mutation_segment", mutation_segment),
        ("alt_count", alt_count),
        ("ref_count", ref_count),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, not {values.shape}."
            )

    if local_state_table.segment_id.duplicated().any():
        raise ValueError("Only one clonal CN state per segment is supported.")
    if not local_state_table.tumor_fraction.eq(1.0).all():
        raise ValueError("Every clonal CN state must have fraction exactly one.")
    states = local_state_table.set_index("segment_id").reindex(mutation_segment)
    if states[["allele_a_cn", "allele_b_cn"]].isna().any().any():
        raise ValueError("A mutation segment has no clonal CN state.")
    return pd.DataFrame(
        {
            "mutation_id": mutation_ids,
            "sample_id": str(sample_id),
            "alt_count": alt_count,
            "ref_count": ref_count,
            "count_observed": 1,
            "purity": float(purity),
            "normal_cn": 2,
            "segment_id": mutation_segment,
            "cn_state_id": "state1",
            "cn_state_fraction": 1.0,
            "allele_a_cn": states[["allele_a_cn", "allele_b_cn"]]
            .max(axis=1)
            .to_numpy(),
            "allele_b_cn": states[["allele_a_cn", "allele_b_cn"]]
            .min(axis=1)
            .to_numpy(),
        },
        columns=TUMOR_TXT_COLUMNS,
    )


def _numeric_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return {"minimum": 0.0, "mean": 0.0, "maximum": 0.0}
    return {
        "minimum": float(np.min(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
    }


def _realized_cn_complexity(
    *,
    mutation_segment: np.ndarray,
    evolution: JointEvolutionResult,
    accepted_cna_events: int,
) -> dict[str, object]:
    profile = evolution.clone_allele_cn[0]
    cn = profile[mutation_segment]
    eligible = cn[:, 0] != cn[:, 1]
    multiplicity = evolution.mutation_multiplicity
    values, counts = np.unique(multiplicity[eligible], return_counts=True)
    events = evolution.cna_event_history
    return {
        "accepted_cna_event_count": accepted_cna_events,
        "applied_cna_event_count": int(events.event_id.nunique()),
        "applied_cna_segment_event_count": len(events),
        "whole_genome_cn_clone_count": 1,
        "maximum_local_state_count": 1,
        "maximum_allele_cn": int(profile.max()),
        "altered_segment_fraction": float(np.any(profile != 1, axis=1).mean()),
        "unequal_cn_mutation_count": int(eligible.sum()),
        "unequal_cn_multiplicity_histogram": {
            str(value): int(count) for value, count in zip(values, counts, strict=True)
        },
        "maximum_clone_specific_dosage": int(multiplicity.max()),
        "fraction_mutations_with_dosage_gt_one": float((multiplicity > 1).mean()),
        "fraction_noninteger_effective_multiplicity": 0.0,
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
        "input_schema": TUMOR_TXT_SCHEMA,
        "canonical_input_file": next(iter(input_hashes)),
        "input_file_hashes": input_hashes,
        "truth_file_hashes": truth_hashes,
    }
    (data_dir / "scenario_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["validate_generated_tumor_directory"]
