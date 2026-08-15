"""High-level generation of canonical CliPP2 tumor and cohort bundles."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..io.tumor_txt import write_tumor_txt
from .config import TumorSimulationConfig, _validate_copy_number_config
from .evolution import (
    _simulate_constrained_cn_evolution,
    aggregate_cn_clone_fractions,
    assign_mutations_to_segments,
    compute_mutation_sample_truth,
    simulate_genome_segments,
)
from .output import (
    _canonical_observation_table,
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
    config: TumorSimulationConfig,
) -> Path:
    out_dir = Path(config.out_dir)
    N_mean = int(config.mean_depth)
    simu_purity = float(config.purity)
    n_samples = int(config.region_count)
    K_min = K_max = int(config.clone_count)
    lambda_mut = mutation_count = int(config.mutation_count)
    min_mutations_per_clone = int(config.min_mutations_per_clone)
    max_rejection_tries = int(config.max_rejection_tries)
    copy_number_config = config.copy_number
    alpha_mut, alpha_split, alpha_lambda = 10.0, 1.0, 5.0
    tau_lineage_min, tau_lineage_max = 1.0, 50.0
    purity_conc, lineage_zero_prob = 50.0, 0.0
    min_clone_ccf, min_clone_ccf_l2_norm = 0.02, 0.05
    min_clone_ccf_distance = 0.10
    if not 0.0 < float(simu_purity) < 1.0:
        raise ValueError("simu_purity must lie strictly between zero and one.")
    if purity_conc <= 0.0:
        raise ValueError("purity_conc must be positive.")
    directory_name = str(config.tumor_id).strip()
    if (
        not directory_name
        or directory_name in {".", ".."}
        or Path(directory_name).name != directory_name
    ):
        raise ValueError("tumor_id must be one nonempty directory name.")
    data_dir = out_dir / directory_name
    if data_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing tumor: {data_dir}")
    _validate_copy_number_config(copy_number_config)
    seed_sequence = np.random.SeedSequence(int(config.seed))
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

    data_dir.mkdir(parents=True, exist_ok=True)
    region_labels = tuple(f"region{j + 1}" for j in range(n_samples))
    for region_id in region_labels:
        (data_dir / region_id).mkdir(parents=True, exist_ok=True)

    mutation_id_width = max(3, len(str(no_mutations)))
    mutation_ids = np.asarray(
        [
            f"m{mutation_index + 1:0{mutation_id_width}d}"
            for mutation_index in range(no_mutations)
        ],
        dtype=object,
    )
    pd.DataFrame(
        {
            "mutation_id": mutation_ids,
            "cluster_id": cluster_id,
        }
    ).to_csv(data_dir / "truth.txt", sep="\t", index=False)
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
        {
            "mutation_id": mutation_ids,
            "cluster_id": cluster_id,
            "ccf": ccf_patient_clones[cluster_id],
        }
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

    pd.DataFrame(
        {
            "mutation_id": mutation_ids,
            "segment_id": mutation_segment,
            "chromosome": mutation_chromosome,
            "position": mutation_position,
        }
    ).to_csv(data_dir / "truth_mutation_segments.tsv", sep="\t", index=False)

    truth_mutation_sample_tables: list[pd.DataFrame] = []
    canonical_observation_tables: list[pd.DataFrame] = []
    depth_arrays: list[np.ndarray] = []
    for j in range(n_samples):
        region_id = region_labels[j]
        (
            local_state_table,
            _dominant_a,
            _dominant_b,
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

        canonical_observation_tables.append(
            _canonical_observation_table(
                mutation_ids=mutation_ids,
                mutation_segment=mutation_segment,
                alt_count=r_j,
                ref_count=ref_j,
                purity=float(purity_j),
                sample_id=region_id,
                local_state_table=local_state_table,
            )
        )

        pd.DataFrame(
            {
                "mutation_id": mutation_ids,
                "cluster_id": cluster_id,
                "ccf": mutation_ccf_sample_j,
                "sample_id": j,
            }
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
    canonical_observations = pd.concat(
        canonical_observation_tables,
        ignore_index=True,
    ).sort_values(
        ["mutation_id", "sample_id", "cn_state_id"],
        kind="stable",
        ignore_index=True,
    )
    canonical_state_counts = canonical_observations.groupby(
        ["mutation_id", "sample_id"],
        sort=False,
    ).size()
    maximum_local_states = copy_number_config.max_local_cn_states_per_mutation
    if maximum_local_states is not None and bool(
        (canonical_state_counts > int(maximum_local_states)).any()
    ):
        raise AssertionError(
            "Canonical output exceeded max_local_cn_states_per_mutation."
        )
    minimum_two_state_fraction = copy_number_config.min_two_state_snv_fraction
    canonical_two_state_fraction = float(np.mean(canonical_state_counts == 2))
    if minimum_two_state_fraction is not None and canonical_two_state_fraction < float(
        minimum_two_state_fraction
    ):
        raise RuntimeError(
            "Canonical unphased output did not satisfy "
            "min_two_state_snv_fraction: "
            f"required={minimum_two_state_fraction}, "
            f"observed={canonical_two_state_fraction:.6f}."
        )
    write_tumor_txt(data_dir / f"{directory_name}.clipp2.txt", canonical_observations)
    intended_factors = {
        "mean_depth": int(N_mean),
        "purity_mean": float(simu_purity),
        "cna_event_rate": float(copy_number_config.cna_event_rate),
        "sample_count": int(n_samples),
        "replicate": 0,
        "clone_count_min": int(K_min),
        "clone_count_max": int(K_max),
        "mutation_count_mode": "fixed",
        "mutation_count": int(mutation_count),
        "mutation_count_poisson_mean": None,
        "fixed_mutation_count": int(mutation_count),
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

    return _write_patient_simulation(config)


__all__ = [
    "RNG_STREAM_NAMES",
    "simulate_tumor",
]
