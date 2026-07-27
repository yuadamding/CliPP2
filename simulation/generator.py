"""High-level generation of canonical CliPP2 tumor and cohort bundles."""

from __future__ import annotations

import itertools as its
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    CopyNumberEvolutionConfig,
    SimulationGridConfig,
    TumorSimulationConfig,
    _validate_copy_number_config,
)
from .evolution import (
    _simulate_constrained_cn_evolution,
    aggregate_cn_clone_fractions,
    assign_mutations_to_segments,
    compute_mutation_sample_truth,
    simulate_genome_segments,
)
from .output import (
    _cn_clone_profile_table,
    _local_cn_state_table,
    _numeric_summary,
    _realized_cn_complexity,
    _write_scenario_manifest,
    validate_generated_tumor_directory,
)
from .tree import sample_mutations_per_clone, simulate_clonal_tree_ccf


RNG_STREAM_NAMES = (
    "seed_tree",
    "seed_clone_fractions",
    "seed_mutation_counts",
    "seed_mutation_segments",
    "seed_cna_events",
    "seed_mutation_times",
    "seed_physical_copy_choices",
    "seed_purity",
    "seed_depth",
    "seed_alt_counts",
    "seed_input_noise",
    "seed_calling",
)
_RESERVED_RNG_STREAMS = frozenset({"seed_input_noise", "seed_calling"})


def _json_seed_entropy(entropy: Any) -> int | list[int]:
    values = np.asarray(entropy)
    if values.ndim == 0:
        return int(values)
    return [int(value) for value in values.reshape(-1)]


def _seed_sequence_from_generator(rng: np.random.Generator) -> np.random.SeedSequence:
    entropy = rng.integers(0, 2**32, size=4, dtype=np.uint32)
    return np.random.SeedSequence([int(value) for value in entropy])


def _named_random_streams(
    seed_sequence: np.random.SeedSequence,
) -> tuple[dict[str, np.random.Generator], dict[str, object]]:
    child_sequences = seed_sequence.spawn(len(RNG_STREAM_NAMES))
    streams = {
        name: np.random.default_rng(child)
        for name, child in zip(RNG_STREAM_NAMES, child_sequences, strict=True)
    }
    stream_metadata = {
        name: {
            "spawn_key": [int(value) for value in child.spawn_key],
            "seed_words": [
                int(value) for value in child.generate_state(4, dtype=np.uint32)
            ],
            "used": name not in _RESERVED_RNG_STREAMS,
        }
        for name, child in zip(RNG_STREAM_NAMES, child_sequences, strict=True)
    }
    metadata = {
        "bit_generator": "PCG64",
        "root_seed_sequence": {
            "entropy": _json_seed_entropy(seed_sequence.entropy),
            "spawn_key": [int(value) for value in seed_sequence.spawn_key],
            "pool_size": int(seed_sequence.pool_size),
        },
        "streams": stream_metadata,
    }
    return streams, metadata


def _write_patient_simulation(
    rng: np.random.Generator,
    out_dir: Path,
    N_mean: int,
    simu_purity: float,
    amp_rate: float,
    n_samples: int,
    sim: int,
    K_min: int,
    K_max: int,
    lambda_mut: int,
    alpha_mut: float,
    alpha_split: float,
    alpha_lambda: float,
    tau_lineage_min: float,
    tau_lineage_max: float,
    purity_conc: float,
    lineage_zero_prob: float,
    min_clone_ccf: float,
    min_clone_ccf_l2_norm: float,
    min_mutations_per_clone: int,
    min_clone_ccf_distance: float,
    max_rejection_tries: int,
    copy_number_config: CopyNumberEvolutionConfig | None = None,
    seed_sequence: np.random.SeedSequence | None = None,
    mutation_count: int | None = None,
    tumor_id: str | None = None,
) -> Path:
    if not 0.0 < float(simu_purity) < 1.0:
        raise ValueError("simu_purity must lie strictly between zero and one.")
    if purity_conc <= 0.0:
        raise ValueError("purity_conc must be positive.")
    explicit_directory_name: str | None = None
    if tumor_id is not None:
        explicit_directory_name = str(tumor_id).strip()
        if (
            not explicit_directory_name
            or explicit_directory_name in {".", ".."}
            or Path(explicit_directory_name).name != explicit_directory_name
        ):
            raise ValueError("tumor_id must be one nonempty directory name.")
        explicit_data_dir = out_dir / explicit_directory_name
        if explicit_data_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing tumor: {explicit_data_dir}"
            )
    copy_number_config = replace(
        copy_number_config or CopyNumberEvolutionConfig(),
        cna_event_rate=float(amp_rate),
    )
    _validate_copy_number_config(copy_number_config)
    if seed_sequence is None:
        seed_sequence = _seed_sequence_from_generator(rng)
    streams, rng_metadata = _named_random_streams(seed_sequence)

    K = streams["seed_tree"].integers(K_min, K_max + 1)
    tau_vec = streams["seed_clone_fractions"].uniform(
        tau_lineage_min,
        tau_lineage_max,
        size=n_samples,
    )

    sim_tree = simulate_clonal_tree_ccf(
        K=K,
        n_samples=n_samples,
        alpha_split=alpha_split,
        tau=tau_vec,
        lineage_zero_prob=lineage_zero_prob,
        random_state=streams["seed_tree"],
        fraction_random_state=streams["seed_clone_fractions"],
        alpha_lambda=alpha_lambda,
        min_clone_ccf=min_clone_ccf,
        min_clone_ccf_l2_norm=min_clone_ccf_l2_norm,
        min_clone_ccf_distance=min_clone_ccf_distance,
        max_rejection_tries=max_rejection_tries,
    )

    ccf_patient_clones = sim_tree["ccf_patient_clones"]
    ccf_samples_clones = sim_tree["ccf_samples_clones"]
    ccf_patient_lineage = sim_tree["ccf_patient_lineages"]
    ccf_samples_lineage = sim_tree["ccf_samples_lineages"]
    lineages = sim_tree["lineages"]
    lineage_terminals = sim_tree["lineage_terminals"]
    exclusive_clone_fraction_patient = sim_tree["exclusive_clone_fraction_patient"]
    exclusive_clone_fraction_samples = sim_tree["exclusive_clone_fraction_samples"]
    if not np.allclose(
        exclusive_clone_fraction_patient, sim_tree["lambda_k"], atol=1e-8, rtol=0.0
    ):
        raise AssertionError(
            "Patient exclusive fractions disagree with the original lineage-mass construction."
        )

    cluster_id, _cluster_size, no_mutations = sample_mutations_per_clone(
        ccf_patient_clones=ccf_patient_clones,
        lambda_mut=lambda_mut,
        mutation_count=mutation_count,
        alpha_mut=alpha_mut,
        min_mutations_per_clone=min_mutations_per_clone,
        random_state=streams["seed_mutation_counts"],
    )
    segments = simulate_genome_segments(
        copy_number_config.n_segments,
        segment_size_bp=copy_number_config.segment_size_bp,
    )
    mutation_segment, mutation_position = assign_mutations_to_segments(
        no_mutations,
        segments,
        random_state=streams["seed_mutation_segments"],
    )
    (
        evolution,
        cn_clone_id,
        unique_cn_profiles,
        cn_generation,
    ) = _simulate_constrained_cn_evolution(
        cna_rng=streams["seed_cna_events"],
        mutation_time_rng=streams["seed_mutation_times"],
        physical_copy_rng=streams["seed_physical_copy_choices"],
        parent=sim_tree["parent"],
        mutation_origin_clone=cluster_id,
        mutation_segment=mutation_segment,
        config=copy_number_config,
        max_rejection_tries=max_rejection_tries,
    )
    cn_clone_fraction_samples = aggregate_cn_clone_fractions(
        exclusive_clone_fraction_samples,
        cn_clone_id,
    )

    if explicit_directory_name is None:
        directory_name = (
            f"{N_mean}_{K}_{simu_purity}_{amp_rate}_S{n_samples}_"
            f"Lm{int(lambda_mut)}_M{no_mutations}_rep{sim}"
        )
    else:
        directory_name = explicit_directory_name
    data_dir = out_dir / directory_name
    data_dir.mkdir(parents=True, exist_ok=True)
    region_labels = tuple(f"region{j + 1}" for j in range(n_samples))
    for region_id in region_labels:
        (data_dir / region_id).mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"cluster_id": cluster_id}).to_csv(
        data_dir / "truth.txt", sep="\t", index=False
    )
    pd.DataFrame(
        {"clone_id": np.arange(K, dtype=int), "ccf": ccf_patient_clones}
    ).to_csv(data_dir / "truth_clone_patient.txt", sep="\t", index=False)
    pd.DataFrame(
        {
            "clone_id": np.arange(K, dtype=int),
            "lambda": exclusive_clone_fraction_patient,
        }
    ).to_csv(data_dir / "truth_lambda.txt", sep="\t", index=False)
    pd.DataFrame(
        {
            "lineage_id": np.arange(len(lineages), dtype=int),
            "terminal_clone_id": lineage_terminals,
            "ccf": ccf_patient_lineage,
        }
    ).to_csv(data_dir / "truth_lineage_patient.txt", sep="\t", index=False)
    pd.DataFrame(
        {"cluster_id": cluster_id, "ccf": ccf_patient_clones[cluster_id]}
    ).to_csv(data_dir / "truth_cp_patient.txt", sep="\t", index=False)
    parent = np.asarray(sim_tree["parent"], dtype=int)
    pd.DataFrame(
        {
            "clone_id": np.arange(K, dtype=int),
            "parent_clone_id": parent,
            "exclusive_fraction_patient": exclusive_clone_fraction_patient,
            "ccf_patient": ccf_patient_clones,
        }
    ).to_csv(data_dir / "truth_clone_tree.tsv", sep="\t", index=False)

    clone_ids = np.repeat(np.arange(K, dtype=int), n_samples)
    sample_ids = np.tile(np.arange(n_samples, dtype=int), K)
    pd.DataFrame(
        {
            "clone_id": clone_ids,
            "sample_id": sample_ids,
            "ccf": ccf_samples_clones.reshape(-1),
        }
    ).to_csv(data_dir / "truth_clone_sample.txt", sep="\t", index=False)
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "clone_id": clone_ids,
            "exclusive_fraction": exclusive_clone_fraction_samples.reshape(-1),
            "cumulative_ccf": ccf_samples_clones.reshape(-1),
        }
    ).to_csv(data_dir / "truth_clone_fraction_sample.tsv", sep="\t", index=False)

    L_count = len(lineages)
    lineage_ids = np.repeat(np.arange(L_count, dtype=int), n_samples)
    sample_ids_L = np.tile(np.arange(n_samples, dtype=int), L_count)
    pd.DataFrame(
        {
            "lineage_id": lineage_ids,
            "sample_id": sample_ids_L,
            "ccf": ccf_samples_lineage.reshape(-1),
        }
    ).to_csv(data_dir / "truth_lineage_sample.txt", sep="\t", index=False)

    sample_targets = np.full(n_samples, float(simu_purity), dtype=float)
    alpha_p = sample_targets * purity_conc
    beta_p = (1.0 - sample_targets) * purity_conc
    sample_purities = streams["seed_purity"].beta(alpha_p, beta_p)
    pd.DataFrame(
        {
            "sample_id": region_labels,
            "purity": sample_purities,
        }
    ).to_csv(data_dir / "purity.txt", sep="\t", index=False)
    pd.DataFrame(
        {
            "sample_id": np.arange(n_samples, dtype=int),
            "tau_lineage": tau_vec,
        }
    ).to_csv(data_dir / "truth_region_parameters.tsv", sep="\t", index=False)

    truth_cn_profile = _cn_clone_profile_table(
        evolution.clone_allele_cn,
        segments,
        clone_ids=np.arange(K, dtype=int),
        cn_clone_ids=cn_clone_id,
    )
    truth_cn_profile.to_csv(
        data_dir / "truth_cn_clone_profile.tsv", sep="\t", index=False
    )
    evolution.cna_event_history.to_csv(
        data_dir / "truth_cna_events.tsv", sep="\t", index=False
    )

    mutation_ids = np.arange(no_mutations, dtype=int)
    mutation_chromosome = np.asarray(
        [segments[int(segment_id)].chromosome for segment_id in mutation_segment],
        dtype=int,
    )
    pd.DataFrame(
        {
            "mutation_id": mutation_ids,
            "origin_clone_id": cluster_id,
            "segment_id": mutation_segment,
            "origin_allele": np.where(evolution.mutation_origin_allele == 0, "A", "B"),
            "branch_time": evolution.mutation_branch_time,
            "cn_at_origin": evolution.mutation_cn_at_origin,
            "physical_copy_index": evolution.mutation_origin_physical_copy,
        }
    ).to_csv(data_dir / "truth_mutation_history.tsv", sep="\t", index=False)

    dosage_carrier = evolution.mutation_carrier.reshape(-1)
    dosage_values = np.where(
        dosage_carrier,
        evolution.mutation_dosage_numeric.reshape(-1).astype(object),
        None,
    )
    pd.DataFrame(
        {
            "mutation_id": np.repeat(mutation_ids, K),
            "clone_id": np.tile(np.arange(K, dtype=int), no_mutations),
            "carrier": dosage_carrier.astype(int),
            "dosage": pd.array(dosage_values, dtype="Int64"),
        }
    ).to_csv(data_dir / "truth_mutation_clone_dosage.tsv", sep="\t", index=False)

    observed_cn_profiles = _cn_clone_profile_table(unique_cn_profiles, segments)
    observed_cn_profiles[
        [
            "cn_clone_id",
            "segment_id",
            "chromosome",
            "start",
            "end",
            "allele_a_cn",
            "allele_b_cn",
        ]
    ].to_csv(data_dir / "cn_clone_profiles.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "sample_id": np.repeat(region_labels, unique_cn_profiles.shape[0]),
            "cn_clone_id": np.tile(
                np.arange(unique_cn_profiles.shape[0], dtype=int), n_samples
            ),
            "tumor_fraction": cn_clone_fraction_samples.T.reshape(-1),
        }
    ).to_csv(data_dir / "cn_clone_fractions.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "mutation_id": mutation_ids,
            "segment_id": mutation_segment,
            "chromosome": mutation_chromosome,
            "position": mutation_position,
        }
    ).to_csv(data_dir / "mutation_segments.tsv", sep="\t", index=False)

    truth_mutation_sample_tables: list[pd.DataFrame] = []
    depth_arrays: list[np.ndarray] = []
    for j in range(n_samples):
        region_id = region_labels[j]
        (
            local_state_table,
            dominant_a,
            dominant_b,
            _dominant_fraction,
            _dominant_state_id,
        ) = _local_cn_state_table(
            sample_id=j,
            unique_profiles=unique_cn_profiles,
            cn_clone_fraction=cn_clone_fraction_samples[:, j],
            segments=segments,
        )
        local_state_table.drop(columns="sample_id").to_csv(
            data_dir / region_id / "truth_cn_states.tsv",
            sep="\t",
            index=False,
        )
        major_cn_segments = np.maximum(dominant_a, dominant_b)
        minor_cn_segments = np.minimum(dominant_a, dominant_b)
        pd.DataFrame(
            {
                "chromosome_index": [segment.chromosome for segment in segments],
                "start_position": [segment.start for segment in segments],
                "end_position": [segment.end for segment in segments],
                "major_cn": major_cn_segments,
                "minor_cn": minor_cn_segments,
            }
        ).to_csv(data_dir / region_id / "cna.txt", sep="\t", index=False)

        purity_j = sample_purities[j]
        truth = compute_mutation_sample_truth(
            clone_fraction=exclusive_clone_fraction_samples[:, j],
            clone_total_cn=evolution.clone_total_cn,
            mutation_segment=mutation_segment,
            mutation_dosage=evolution.mutation_dosage_numeric,
            mutation_carrier=evolution.mutation_carrier,
            purity=purity_j,
            normal_cn=np.full(no_mutations, 2.0, dtype=float),
        )
        mutation_ccf_sample_j = ccf_samples_clones[cluster_id, j]
        if not np.allclose(truth["ccf"], mutation_ccf_sample_j, atol=1e-8, rtol=0.0):
            raise AssertionError(
                "Mutation carrier CCF no longer matches the acquisition-clone CCF."
            )
        if np.any((truth["ccf"] > 0.0) & (truth["mutant_copy_mass"] <= 0.0)):
            raise AssertionError("A present mutation has zero mutant-copy mass.")

        n_j = streams["seed_depth"].poisson(N_mean, size=no_mutations)
        r_j = streams["seed_alt_counts"].binomial(
            n_j,
            truth["expected_vaf"],
        )
        ref_j = n_j - r_j
        depth_arrays.append(n_j)

        pd.DataFrame(
            {
                "chromosome_index": mutation_chromosome,
                "position": mutation_position,
                "alt_count": r_j,
                "ref_count": ref_j,
            }
        ).to_csv(data_dir / region_id / "snv.txt", sep="\t", index=False)

        with open(data_dir / region_id / "purity.txt", "w", encoding="utf-8") as handle:
            handle.write(f"{purity_j}\n")

        pd.DataFrame(
            {"cluster_id": cluster_id, "ccf": mutation_ccf_sample_j, "sample_id": j}
        ).to_csv(data_dir / region_id / "truth_cp.txt", sep="\t", index=False)
        truth_mutation_sample_tables.append(
            pd.DataFrame(
                {
                    "mutation_id": mutation_ids,
                    "sample_id": np.full(no_mutations, j, dtype=int),
                    "ccf": truth["ccf"],
                    "mutant_copy_mass": truth["mutant_copy_mass"],
                    "effective_multiplicity": truth["effective_multiplicity"],
                    "mean_tumor_total_cn": truth["mean_tumor_total_cn"],
                    "expected_vaf": truth["expected_vaf"],
                }
            )
        )

    mutation_sample_truth = pd.concat(
        truth_mutation_sample_tables,
        ignore_index=True,
    )
    mutation_sample_truth.to_csv(
        data_dir / "truth_mutation_sample.tsv",
        sep="\t",
        index=False,
    )
    intended_factors = {
        "mean_depth": int(N_mean),
        "purity_mean": float(simu_purity),
        "cna_event_rate": float(amp_rate),
        "sample_count": int(n_samples),
        "replicate": int(sim),
        "clone_count_min": int(K_min),
        "clone_count_max": int(K_max),
        "mutation_count_mode": "fixed" if mutation_count is not None else "poisson",
        "mutation_count": (
            int(mutation_count) if mutation_count is not None else int(lambda_mut)
        ),
        "mutation_count_poisson_mean": (
            None if mutation_count is not None else int(lambda_mut)
        ),
        "fixed_mutation_count": (
            int(mutation_count) if mutation_count is not None else None
        ),
        "alpha_mut": float(alpha_mut),
        "alpha_split": float(alpha_split),
        "alpha_lambda": float(alpha_lambda),
        "tau_lineage_min": float(tau_lineage_min),
        "tau_lineage_max": float(tau_lineage_max),
        "purity_concentration": float(purity_conc),
        "lineage_zero_probability": float(lineage_zero_prob),
        "min_clone_ccf": float(min_clone_ccf),
        "min_clone_ccf_l2_norm": float(min_clone_ccf_l2_norm),
        "min_mutations_per_clone": int(min_mutations_per_clone),
        "min_clone_ccf_distance": float(min_clone_ccf_distance),
        "max_rejection_tries": int(max_rejection_tries),
        "copy_number": asdict(copy_number_config),
    }
    realized_factors = {
        "clone_count": int(K),
        "mutation_count": int(no_mutations),
        "sample_count": int(n_samples),
        "sample_purity": _numeric_summary(sample_purities),
        "depth": _numeric_summary(np.concatenate(depth_arrays)),
        "cn_complexity": _realized_cn_complexity(
            parent=parent,
            mutation_origin_clone=cluster_id,
            mutation_segment=mutation_segment,
            evolution=evolution,
            unique_cn_profiles=unique_cn_profiles,
            cn_clone_fraction_samples=cn_clone_fraction_samples,
            accepted_cna_events=cn_generation["accepted_cna_events"],
            mutation_sample_truth=mutation_sample_truth,
        ),
    }
    rejection_counts = {
        "tree_ccf": int(sim_tree["generation_attempts"]) - 1,
        "tree_topology": int(sim_tree["topology_rejections"]),
        "copy_number": int(cn_generation["attempts"]) - 1,
    }
    validate_generated_tumor_directory(data_dir)
    _write_scenario_manifest(
        data_dir=data_dir,
        intended_factors=intended_factors,
        realized_factors=realized_factors,
        rejection_counts=rejection_counts,
        rng_metadata=rng_metadata,
    )
    return data_dir


def write_patient_simulation(
    rng: np.random.Generator,
    out_dir: str | Path,
    N_mean: int,
    simu_purity: float,
    amp_rate: float,
    n_samples: int,
    sim: int,
    K_min: int = 2,
    K_max: int = 10,
    lambda_mut: int = 2000,
    alpha_mut: float = 10.0,
    alpha_split: float = 1.0,
    alpha_lambda: float = 5.0,
    tau_lineage_min: float = 1.0,
    tau_lineage_max: float = 50.0,
    purity_conc: float = 50.0,
    lineage_zero_prob: float = 0.0,
    min_clone_ccf: float = 0.02,
    min_clone_ccf_l2_norm: float = 0.05,
    min_mutations_per_clone: int = 15,
    min_clone_ccf_distance: float = 0.10,
    max_rejection_tries: int = 1024,
    copy_number_config: CopyNumberEvolutionConfig | None = None,
    mutation_count: int | None = None,
    tumor_id: str | None = None,
) -> Path:
    return _write_patient_simulation(
        rng=rng,
        out_dir=Path(out_dir),
        N_mean=N_mean,
        simu_purity=simu_purity,
        amp_rate=amp_rate,
        n_samples=n_samples,
        sim=sim,
        K_min=K_min,
        K_max=K_max,
        lambda_mut=lambda_mut,
        alpha_mut=alpha_mut,
        alpha_split=alpha_split,
        alpha_lambda=alpha_lambda,
        tau_lineage_min=tau_lineage_min,
        tau_lineage_max=tau_lineage_max,
        purity_conc=purity_conc,
        lineage_zero_prob=lineage_zero_prob,
        min_clone_ccf=min_clone_ccf,
        min_clone_ccf_l2_norm=min_clone_ccf_l2_norm,
        min_mutations_per_clone=min_mutations_per_clone,
        min_clone_ccf_distance=min_clone_ccf_distance,
        max_rejection_tries=max_rejection_tries,
        copy_number_config=copy_number_config,
        mutation_count=mutation_count,
        tumor_id=tumor_id,
    )


def simulate_tumor(
    config: TumorSimulationConfig = TumorSimulationConfig(),
) -> Path:
    """Generate one named, exact-size tumor and validate its public input bundle."""

    if config.mutation_count < 1:
        raise ValueError("mutation_count must be positive.")
    if config.mean_depth < 1:
        raise ValueError("mean_depth must be positive.")
    if config.region_count < 1:
        raise ValueError("region_count must be positive.")
    if config.clone_count < 2:
        raise ValueError("clone_count must be at least 2.")
    if config.copy_number.max_local_cn_states_per_mutation != 2:
        raise ValueError(
            "Canonical CliPP2 simulations require exactly the supported two-state cap."
        )
    minimum_two_state = config.copy_number.min_two_state_snv_fraction
    if minimum_two_state is None or minimum_two_state <= 0.5:
        raise ValueError(
            "Canonical CliPP2 simulations require more than 50% two-state SNVs."
        )

    seed_sequence = np.random.SeedSequence(int(config.seed))
    return _write_patient_simulation(
        rng=np.random.default_rng(seed_sequence),
        out_dir=Path(config.out_dir),
        N_mean=int(config.mean_depth),
        simu_purity=float(config.purity),
        amp_rate=float(config.copy_number.cna_event_rate),
        n_samples=int(config.region_count),
        sim=0,
        K_min=int(config.clone_count),
        K_max=int(config.clone_count),
        lambda_mut=int(config.mutation_count),
        alpha_mut=10.0,
        alpha_split=1.0,
        alpha_lambda=5.0,
        tau_lineage_min=1.0,
        tau_lineage_max=50.0,
        purity_conc=50.0,
        lineage_zero_prob=0.0,
        min_clone_ccf=0.02,
        min_clone_ccf_l2_norm=0.05,
        min_mutations_per_clone=int(config.min_mutations_per_clone),
        min_clone_ccf_distance=0.10,
        max_rejection_tries=int(config.max_rejection_tries),
        copy_number_config=config.copy_number,
        seed_sequence=seed_sequence,
        mutation_count=int(config.mutation_count),
        tumor_id=config.tumor_id,
    )


def run_simulation_grid(
    out_dir: str | Path = "CliPP2Sim",
    purity_list: list[float] | None = None,
    amp_rate_list: list[float] | None = None,
    N_list: list[int] | None = None,
    n_samples_list: list[int] | None = None,
    reps: int = 20,
    seed: int | None = None,
    K_min: int = 2,
    K_max: int = 10,
    lambda_mut: int = 2000,
    lambda_mut_list: list[int] | None = None,
    alpha_mut: float = 10.0,
    alpha_split: float = 1.0,
    alpha_lambda: float = 5.0,
    tau_lineage_min: float = 1.0,
    tau_lineage_max: float = 50.0,
    purity_conc: float = 50.0,
    lineage_zero_prob: float = 0.0,
    min_clone_ccf: float = 0.02,
    min_clone_ccf_l2_norm: float = 0.05,
    min_mutations_per_clone: int = 15,
    min_clone_ccf_distance: float = 0.10,
    max_rejection_tries: int = 1024,
    copy_number_config: CopyNumberEvolutionConfig | None = None,
) -> list[Path]:
    if purity_list is None:
        purity_list = list(SimulationGridConfig.purity_list)
    if amp_rate_list is None:
        amp_rate_list = list(SimulationGridConfig.amp_rate_list)
    if N_list is None:
        N_list = list(SimulationGridConfig.N_list)
    if n_samples_list is None:
        n_samples_list = list(SimulationGridConfig.n_samples_list)
    if lambda_mut_list is None:
        config_default = SimulationGridConfig.lambda_mut_list
        lambda_mut_list = (
            list(config_default) if config_default is not None else [int(lambda_mut)]
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_seed_sequence = np.random.SeedSequence(seed)
    written_dirs: list[Path] = []

    for N_mean, simu_purity, amp_rate, n_samples, lambda_mut_value in its.product(
        N_list,
        purity_list,
        amp_rate_list,
        n_samples_list,
        lambda_mut_list,
    ):
        for sim in range(reps):
            patient_seed_sequence = master_seed_sequence.spawn(1)[0]
            child_rng = np.random.default_rng(patient_seed_sequence)
            written_dirs.append(
                _write_patient_simulation(
                    rng=child_rng,
                    out_dir=out_dir,
                    N_mean=N_mean,
                    simu_purity=simu_purity,
                    amp_rate=amp_rate,
                    n_samples=n_samples,
                    sim=sim,
                    K_min=K_min,
                    K_max=K_max,
                    lambda_mut=int(lambda_mut_value),
                    alpha_mut=alpha_mut,
                    alpha_split=alpha_split,
                    alpha_lambda=alpha_lambda,
                    tau_lineage_min=tau_lineage_min,
                    tau_lineage_max=tau_lineage_max,
                    purity_conc=purity_conc,
                    lineage_zero_prob=lineage_zero_prob,
                    min_clone_ccf=min_clone_ccf,
                    min_clone_ccf_l2_norm=min_clone_ccf_l2_norm,
                    min_mutations_per_clone=min_mutations_per_clone,
                    min_clone_ccf_distance=min_clone_ccf_distance,
                    max_rejection_tries=max_rejection_tries,
                    copy_number_config=copy_number_config,
                    seed_sequence=patient_seed_sequence,
                )
            )

    return written_dirs


def run_simulation_grid_from_config(config: SimulationGridConfig) -> list[Path]:
    return run_simulation_grid(
        out_dir=config.out_dir,
        purity_list=list(config.purity_list),
        amp_rate_list=list(config.amp_rate_list),
        N_list=list(config.N_list),
        n_samples_list=list(config.n_samples_list),
        reps=config.reps,
        seed=config.seed,
        K_min=config.K_min,
        K_max=config.K_max,
        lambda_mut=config.lambda_mut,
        lambda_mut_list=list(config.lambda_mut_list)
        if config.lambda_mut_list is not None
        else None,
        alpha_mut=config.alpha_mut,
        alpha_split=config.alpha_split,
        alpha_lambda=config.alpha_lambda,
        tau_lineage_min=config.tau_lineage_min,
        tau_lineage_max=config.tau_lineage_max,
        purity_conc=config.purity_conc,
        lineage_zero_prob=config.lineage_zero_prob,
        min_clone_ccf=config.min_clone_ccf,
        min_clone_ccf_l2_norm=config.min_clone_ccf_l2_norm,
        min_mutations_per_clone=config.min_mutations_per_clone,
        min_clone_ccf_distance=config.min_clone_ccf_distance,
        max_rejection_tries=config.max_rejection_tries,
        copy_number_config=config.copy_number,
    )


__all__ = [
    "RNG_STREAM_NAMES",
    "run_simulation_grid",
    "run_simulation_grid_from_config",
    "simulate_tumor",
    "write_patient_simulation",
]
