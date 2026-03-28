from __future__ import annotations

import argparse
import itertools as its
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationGridConfig:
    out_dir: str | Path = "CliPP2Sim"
    purity_list: tuple[float, ...] = (0.3, 0.6, 0.9)
    amp_rate_list: tuple[float, ...] = (0.0, 0.1, 0.2)
    N_list: tuple[int, ...] = (50, 75, 100, 200, 300, 400, 500, 1000)
    n_samples_list: tuple[int, ...] = (2, 5, 10, 15)
    reps: int = 20
    seed: int | None = None
    K_min: int = 2
    K_max: int = 10
    lambda_mut: int = 2000
    lambda_mut_list: tuple[int, ...] | None = (300, 600, 1000, 2000, 4000)
    alpha_mut: float = 10.0
    alpha_split: float = 1.0
    alpha_lambda: float = 5.0
    tau_lineage_min: float = 1.0
    tau_lineage_max: float = 50.0
    purity_conc: float = 50.0
    lineage_zero_prob: float = 0.0
    min_clone_ccf: float = 0.02
    min_clone_ccf_l2_norm: float = 0.05
    min_mutations_per_clone: int = 15
    min_clone_ccf_distance: float = 0.10
    max_rejection_tries: int = 1024


def parse_int_list(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def _check_patient_tree_and_ccf(parent, children, ccf_patient_clones, tol=1e-8):
    parent = np.asarray(parent, dtype=int)
    ccf = np.asarray(ccf_patient_clones, dtype=float)
    K = parent.shape[0]

    assert K >= 1, "Tree must have at least one node."
    assert parent[0] == -1, "Root must have parent -1."
    for k in range(1, K):
        p = parent[k]
        assert 0 <= p < K, f"Invalid parent[{k}]={p}; must be in [0,{K-1}]."

    assert np.all(ccf >= -tol), "Some ccf_patient_clones < 0."
    assert np.all(ccf <= 1.0 + tol), "Some ccf_patient_clones > 1."
    assert abs(ccf[0] - 1.0) <= tol, f"Root CCF must be ~1, got {ccf[0]}."

    for k in range(1, K):
        p = parent[k]
        assert ccf[k] <= ccf[p] + tol, (
            f"Descendant clone {k} has ccf {ccf[k]:.4g} > parent {p} ccf {ccf[p]:.4g}."
        )

    for k in range(K):
        ch = children[k]
        if ch:
            s_children = float(sum(ccf[c] for c in ch))
            assert s_children <= ccf[k] + tol, (
                f"Mass mismatch at node {k}: ccf={ccf[k]:.6g}, sum(children)={s_children:.6g}."
            )

    for k in range(K):
        stack = [k]
        desc_leaves = []
        while stack:
            node = stack.pop()
            if len(children[node]) == 0:
                desc_leaves.append(node)
            else:
                stack.extend(children[node])
        s_leaves = float(sum(ccf[leaf] for leaf in desc_leaves))
        assert s_leaves <= ccf[k] + tol, (
            f"Descendant-leaf mismatch at node {k}: ccf={ccf[k]:.6g}, sum(desc_leaves)={s_leaves:.6g}."
        )

    return True


def _check_sample_ccf_against_tree(parent, children, ccf_samples_clones, tol=1e-8):
    parent = np.asarray(parent, dtype=int)
    C = np.asarray(ccf_samples_clones, dtype=float)
    K, M = C.shape

    assert np.allclose(C[0, :], 1.0, atol=tol), "Root CCF must be ~1 in all samples."
    assert np.all(C >= -tol), "Some sample CCFs < 0."
    assert np.all(C <= 1.0 + tol), "Some sample CCFs > 1."

    for j in range(M):
        for k in range(1, K):
            p = parent[k]
            assert C[k, j] <= C[p, j] + tol, (
                f"Sample {j}: clone {k} has CCF {C[k, j]:.4g} > parent {p} CCF {C[p, j]:.4g}."
            )
        for k in range(K):
            ch = children[k]
            if ch:
                s_children = float(sum(C[c, j] for c in ch))
                assert s_children <= C[k, j] + tol, (
                    f"Sample {j}, node {k}: CCF={C[k, j]:.6g}, sum(children)={s_children:.6g}."
                )
    return True


def _min_pairwise_clone_distance(ccf_samples_clones: np.ndarray) -> float:
    ccf = np.asarray(ccf_samples_clones, dtype=float)
    if ccf.shape[0] <= 1:
        return float("inf")
    diff = ccf[:, None, :] - ccf[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return float(np.min(dist))


def _min_clone_region_ccf(ccf_samples_clones: np.ndarray) -> float:
    ccf = np.asarray(ccf_samples_clones, dtype=float)
    return float(np.min(ccf))


def _min_clone_l2_norm(ccf_samples_clones: np.ndarray) -> float:
    ccf = np.asarray(ccf_samples_clones, dtype=float)
    norms = np.linalg.norm(ccf, axis=1)
    return float(np.min(norms))


def simulate_clonal_tree_ccf(
    K,
    n_samples,
    alpha_split=1.0,
    tau=50.0,
    lineage_zero_prob=0.0,
    random_state=None,
    eps=1e-8,
    alpha_lambda=5.0,
    lineage_eps=1e-8,
    min_clone_ccf=0.02,
    min_clone_ccf_l2_norm=0.05,
    min_clone_ccf_distance=0.10,
    max_rejection_tries=1024,
):
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    if lineage_zero_prob > eps and min_clone_ccf > 0.0:
        raise ValueError(
            "lineage_zero_prob must be 0 when enforcing a strictly positive minimum clone CCF in every region."
        )

    for _ in range(max(int(max_rejection_tries), 1)):
        while True:
            parent = np.empty(K, dtype=int)
            parent[0] = -1
            for k in range(1, K):
                parent[k] = rng.integers(0, k)

            children = [[] for _ in range(K)]
            for k, p in enumerate(parent):
                if p >= 0:
                    children[p].append(k)

            leaves_all = [k for k in range(K) if len(children[k]) == 0]
            is_pure_chain = (K > 1 and len(leaves_all) == 1 and all(len(ch) <= 1 for ch in children))
            if not is_pure_chain or K <= 5:
                break

        if is_pure_chain:
            base_min = 0.2
            if base_min * K > 1.0 + 1e-10:
                raise ValueError(f"Cannot enforce λ_k >= 0.2 for pure chain with K={K}.")
            if abs(base_min * K - 1.0) <= 1e-10:
                lambda_k = np.full(K, base_min, dtype=float)
            else:
                leftover = 1.0 - base_min * K
                lambda_k = base_min + leftover * rng.dirichlet(np.ones(K))
        else:
            lambda_k = rng.dirichlet(alpha_lambda * np.ones(K))

        ccf_patient_clones = np.zeros(K, dtype=float)
        for k in reversed(range(K)):
            ccf_patient_clones[k] = lambda_k[k] + sum(ccf_patient_clones[c] for c in children[k])

        _check_patient_tree_and_ccf(parent, children, ccf_patient_clones, tol=1e-8)

        lineage_terminals = [k for k in range(K) if lambda_k[k] > lineage_eps]
        if len(lineage_terminals) == 0:
            lineage_terminals = [k for k in range(K) if len(children[k]) == 0]

        lineages = []
        for terminal in lineage_terminals:
            path = []
            node = terminal
            while node != -1:
                path.append(node)
                node = parent[node]
            path.reverse()
            lineages.append(path)

        L = len(lineages)
        A = np.zeros((K, L), dtype=float)
        for ell_idx, path in enumerate(lineages):
            for k in path:
                A[k, ell_idx] = 1.0

        idx_term = np.array(lineage_terminals, dtype=int)
        ccf_patient_lineages = lambda_k[idx_term].copy()
        u_safe = np.maximum(ccf_patient_lineages.astype(float), eps)
        u_safe = u_safe / u_safe.sum()
        lineage_floor = max(float(min_clone_ccf), float(min_clone_ccf_l2_norm) / np.sqrt(float(n_samples)))
        if lineage_floor * L >= 1.0 - eps:
            raise ValueError(
                f"Cannot enforce lineage floor {lineage_floor:.4f} with {L} lineages; floor * lineages must stay below 1."
            )
        leftover_mass = 1.0 - lineage_floor * L

        tau_arr = np.asarray(tau, dtype=float)
        if tau_arr.ndim == 0:
            tau_vec = np.full(n_samples, float(tau_arr), dtype=float)
        else:
            if tau_arr.shape[0] != n_samples:
                raise ValueError("If `tau` is array-like, its length must be n_samples.")
            tau_vec = tau_arr

        ccf_samples_lineages = np.zeros((L, n_samples), dtype=float)
        for j in range(n_samples):
            if tau_vec[j] <= 0:
                lambda_sample = u_safe.copy()
            else:
                alpha_present = tau_vec[j] * u_safe
                lambda_sample = rng.dirichlet(alpha_present)

            ccf_samples_lineages[:, j] = lineage_floor + leftover_mass * lambda_sample

        ccf_samples_clones = A @ ccf_samples_lineages
        _check_sample_ccf_against_tree(parent, children, ccf_samples_clones, tol=1e-8)
        if _min_clone_region_ccf(ccf_samples_clones) < float(min_clone_ccf) - eps:
            continue
        if _min_clone_l2_norm(ccf_samples_clones) < float(min_clone_ccf_l2_norm) - eps:
            continue
        if _min_pairwise_clone_distance(ccf_samples_clones) < float(min_clone_ccf_distance) - eps:
            continue

        return {
            "parent": parent,
            "children": children,
            "lineage_terminals": idx_term,
            "lineages": lineages,
            "A": A,
            "ccf_patient_clones": ccf_patient_clones,
            "lambda_k": lambda_k,
            "ccf_patient_lineages": ccf_patient_lineages,
            "ccf_samples_lineages": ccf_samples_lineages,
            "ccf_samples_clones": ccf_samples_clones,
        }

    raise RuntimeError(
        "Failed to generate a clonal tree satisfying the region-level clone CCF and pairwise clone-separation constraints."
    )


def sample_mutations_per_clone(
    ccf_patient_clones,
    lambda_mut=800,
    alpha_mut=10.0,
    min_mutations_per_clone=1,
    random_state=None,
):
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    ccf = np.asarray(ccf_patient_clones, dtype=float)
    K = ccf.shape[0]

    min_mutations_per_clone = max(int(min_mutations_per_clone), 1)
    N_mut = rng.poisson(lambda_mut)
    min_total = K * min_mutations_per_clone
    if N_mut < min_total:
        N_mut = min_total

    base = np.maximum(ccf, 0.0)
    if base.sum() <= 0:
        p0 = np.full(K, 1.0 / K, dtype=float)
    else:
        base = base + 1e-6
        p0 = base / base.sum()

    theta = rng.dirichlet(alpha_mut * p0)
    base_counts = np.full(K, min_mutations_per_clone, dtype=int)
    remaining = N_mut - min_total
    if remaining > 0:
        extra = rng.multinomial(remaining, theta)
        cluster_size = base_counts + extra
    else:
        cluster_size = base_counts

    cluster_id = np.repeat(np.arange(K), cluster_size)
    rng.shuffle(cluster_id)
    return cluster_id.astype(int), cluster_size.astype(int), int(N_mut)


def _sample_cna_states(
    rng: np.random.Generator,
    no_mutations: int,
    amp_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    major_cn = np.ones(no_mutations, dtype=int)
    minor_cn = np.ones(no_mutations, dtype=int)
    multiplicity = np.ones(no_mutations, dtype=int)
    multiplicity_source = np.full(no_mutations, "fixed", dtype=object)

    if amp_rate <= 0.0 or no_mutations <= 0:
        return major_cn, minor_cn, multiplicity, multiplicity_source

    amp_mask = rng.random(no_mutations) < amp_rate
    n_amp = int(amp_mask.sum())
    if n_amp == 0:
        return major_cn, minor_cn, multiplicity, multiplicity_source

    major_amp = np.empty(n_amp, dtype=int)
    minor_amp = np.empty(n_amp, dtype=int)
    multiplicity_amp = np.empty(n_amp, dtype=int)
    source_amp = np.empty(n_amp, dtype=object)

    filled = 0
    while filled < n_amp:
        batch_size = max(2 * (n_amp - filled), 16)
        allele_a = rng.integers(0, 5, size=batch_size)
        allele_b = rng.integers(0, 5, size=batch_size)
        cand_major = np.maximum(allele_a, allele_b)
        cand_minor = np.minimum(allele_a, allele_b)

        valid = ((cand_major + cand_minor) > 0) & ~((cand_major == 1) & (cand_minor == 1))
        if not np.any(valid):
            continue

        valid_idx = np.flatnonzero(valid)
        take = min(valid_idx.size, n_amp - filled)
        idx = valid_idx[:take]

        maj = cand_major[idx]
        minr = cand_minor[idx]
        ambiguous = (maj > minr) & (minr > 0)
        choose_major = np.ones(take, dtype=bool)
        if ambiguous.any():
            choose_major[ambiguous] = rng.random(int(ambiguous.sum())) < 0.5

        mult = np.where(choose_major, maj, minr)
        src = np.full(take, "fixed", dtype=object)
        src[minr == 0] = "major"
        if ambiguous.any():
            src[ambiguous] = np.where(choose_major[ambiguous], "major", "minor")

        end = filled + take
        major_amp[filled:end] = maj
        minor_amp[filled:end] = minr
        multiplicity_amp[filled:end] = mult
        source_amp[filled:end] = src
        filled = end

    major_cn[amp_mask] = major_amp
    minor_cn[amp_mask] = minor_amp
    multiplicity[amp_mask] = multiplicity_amp
    multiplicity_source[amp_mask] = source_amp
    return major_cn, minor_cn, multiplicity, multiplicity_source


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
) -> Path:
    K = rng.integers(K_min, K_max + 1)
    tau_vec = rng.uniform(tau_lineage_min, tau_lineage_max, size=n_samples)

    sim_tree = simulate_clonal_tree_ccf(
        K=K,
        n_samples=n_samples,
        alpha_split=alpha_split,
        tau=tau_vec,
        lineage_zero_prob=lineage_zero_prob,
        random_state=rng,
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
    lambda_k = sim_tree["lambda_k"]

    cluster_id, cluster_size, no_mutations = sample_mutations_per_clone(
        ccf_patient_clones=ccf_patient_clones,
        lambda_mut=lambda_mut,
        alpha_mut=alpha_mut,
        min_mutations_per_clone=min_mutations_per_clone,
        random_state=rng,
    )

    data_dir = out_dir / (
        f"{N_mean}_{K}_{simu_purity}_{amp_rate}_S{n_samples}_Lm{int(lambda_mut)}_M{no_mutations}_rep{sim}"
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    for j in range(n_samples):
        (data_dir / f"sample{j}").mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"cluster_id": cluster_id}).to_csv(data_dir / "truth.txt", sep="\t", index=False)
    pd.DataFrame({"clone_id": np.arange(K, dtype=int), "ccf": ccf_patient_clones}).to_csv(
        data_dir / "truth_clone_patient.txt", sep="\t", index=False
    )
    pd.DataFrame({"clone_id": np.arange(K, dtype=int), "lambda": lambda_k}).to_csv(
        data_dir / "truth_lambda.txt", sep="\t", index=False
    )
    pd.DataFrame(
        {
            "lineage_id": np.arange(len(lineages), dtype=int),
            "terminal_clone_id": lineage_terminals,
            "ccf": ccf_patient_lineage,
        }
    ).to_csv(data_dir / "truth_lineage_patient.txt", sep="\t", index=False)
    pd.DataFrame({"cluster_id": cluster_id, "ccf": ccf_patient_clones[cluster_id]}).to_csv(
        data_dir / "truth_cp_patient.txt", sep="\t", index=False
    )

    clone_ids = np.repeat(np.arange(K, dtype=int), n_samples)
    sample_ids = np.tile(np.arange(n_samples, dtype=int), K)
    pd.DataFrame({"clone_id": clone_ids, "sample_id": sample_ids, "ccf": ccf_samples_clones.reshape(-1)}).to_csv(
        data_dir / "truth_clone_sample.txt", sep="\t", index=False
    )

    L_count = len(lineages)
    lineage_ids = np.repeat(np.arange(L_count, dtype=int), n_samples)
    sample_ids_L = np.tile(np.arange(n_samples, dtype=int), L_count)
    pd.DataFrame({"lineage_id": lineage_ids, "sample_id": sample_ids_L, "ccf": ccf_samples_lineage.reshape(-1)}).to_csv(
        data_dir / "truth_lineage_sample.txt", sep="\t", index=False
    )

    alpha_p = simu_purity * purity_conc
    beta_p = (1.0 - simu_purity) * purity_conc
    sample_purities = rng.beta(alpha_p, beta_p, size=n_samples)
    pd.DataFrame(
        {"sample_id": np.arange(n_samples, dtype=int), "purity": sample_purities, "tau_lineage": tau_vec}
    ).to_csv(data_dir / "purity.txt", sep="\t", index=False)

    for j in range(n_samples):
        major_cn_j, minor_cn_j, multiplicity_j, multiplicity_source_j = _sample_cna_states(
            rng=rng,
            no_mutations=no_mutations,
            amp_rate=amp_rate,
        )
        total_j = major_cn_j + minor_cn_j
        pd.DataFrame(
            {
                "chromosome_index": np.ones(no_mutations, dtype=int),
                "start_position": [3 * i + 1 for i in range(no_mutations)],
                "end_position": [3 * i + 3 for i in range(no_mutations)],
                "major_cn": major_cn_j,
                "minor_cn": minor_cn_j,
                "total_cn": total_j,
                "multiplicity": multiplicity_j,
                "multiplicity_source": multiplicity_source_j,
            }
        ).to_csv(data_dir / f"sample{j}" / "cna.txt", sep="\t", index=False)

        mutation_ccf_sample_j = ccf_samples_clones[cluster_id, j]
        purity_j = sample_purities[j]
        n_j = rng.poisson(N_mean, size=no_mutations)
        vaf_j = purity_j * mutation_ccf_sample_j * multiplicity_j / (2 * (1 - purity_j) + purity_j * total_j)
        r_j = rng.binomial(n_j, vaf_j)
        ref_j = n_j - r_j

        pd.DataFrame(
            {
                "chromosome_index": np.ones(no_mutations, dtype=int),
                "position": [3 * i + 2 for i in range(no_mutations)],
                "alt_count": r_j,
                "ref_count": ref_j,
                "sample_id": j,
            }
        ).to_csv(data_dir / f"sample{j}" / "snv.txt", sep="\t", index=False)

        with open(data_dir / f"sample{j}" / "purity.txt", "w", encoding="utf-8") as handle:
            handle.write(f"{purity_j}\n")

        pd.DataFrame({"cluster_id": cluster_id, "ccf": mutation_ccf_sample_j, "sample_id": j}).to_csv(
            data_dir / f"sample{j}" / "truth_cp.txt", sep="\t", index=False
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
        lambda_mut_list = list(config_default) if config_default is not None else [int(lambda_mut)]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.default_rng(seed)
    written_dirs: list[Path] = []

    for N_mean, simu_purity, amp_rate, n_samples, lambda_mut_value in its.product(
        N_list,
        purity_list,
        amp_rate_list,
        n_samples_list,
        lambda_mut_list,
    ):
        for sim in range(reps):
            child_seed = int(rng_master.integers(0, 2**32 - 1))
            child_rng = np.random.default_rng(child_seed)
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
        lambda_mut_list=list(config.lambda_mut_list) if config.lambda_mut_list is not None else None,
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
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate CliPP2 simulation datasets.")
    parser.add_argument("--out-dir", default="CliPP2Sim", help="Directory to write simulated patient folders.")
    parser.add_argument("--purity-list", default="0.3,0.6,0.9", help="Comma-separated purity values.")
    parser.add_argument("--amp-rate-list", default="0.0,0.1,0.2", help="Comma-separated CNA amplification rates.")
    parser.add_argument("--N-list", default="50,75,100,200,300,400,500,1000", help="Comma-separated mean depth values.")
    parser.add_argument("--n-samples-list", default="2,5,10,15", help="Comma-separated sample-count values.")
    parser.add_argument("--reps", type=int, default=20, help="Number of simulation replicates per scenario.")
    parser.add_argument("--seed", type=int, default=None, help="Optional master RNG seed.")
    parser.add_argument("--K-min", type=int, default=2, help="Minimum number of clones.")
    parser.add_argument("--K-max", type=int, default=10, help="Maximum number of clones.")
    parser.add_argument("--lambda-mut", type=int, default=2000, help="Legacy single Poisson mean for mutation count.")
    parser.add_argument(
        "--lambda-mut-list",
        default="300,600,1000,2000,4000",
        help="Comma-separated Poisson means for mutation counts; used by default to cover a wider mutation range.",
    )
    parser.add_argument("--alpha-mut", type=float, default=10.0, help="Dirichlet concentration for mutation allocation.")
    parser.add_argument("--alpha-lambda", type=float, default=5.0, help="Dirichlet concentration for lineage residual masses.")
    parser.add_argument("--tau-lineage-min", type=float, default=1.0, help="Minimum lineage concentration per sample.")
    parser.add_argument("--tau-lineage-max", type=float, default=50.0, help="Maximum lineage concentration per sample.")
    parser.add_argument("--purity-conc", type=float, default=50.0, help="Beta concentration for sample purities.")
    parser.add_argument("--lineage-zero-prob", type=float, default=0.0, help="Probability of zeroing a lineage in a sample. Default is 0.0 because clone CCF is now constrained to stay positive in every region.")
    parser.add_argument("--min-clone-ccf", type=float, default=0.02, help="Minimum allowed clone CCF in every region.")
    parser.add_argument("--min-clone-ccf-l2-norm", type=float, default=0.05, help="Minimum L2 norm of each clone's multiregion CCF vector.")
    parser.add_argument("--min-mutations-per-clone", type=int, default=15, help="Minimum number of mutations assigned to each clone.")
    parser.add_argument(
        "--min-clone-ccf-distance",
        type=float,
        default=0.10,
        help="Minimum L2 distance between any two clones' multiregion CCF profiles.",
    )
    parser.add_argument(
        "--max-rejection-tries",
        type=int,
        default=1024,
        help="Maximum rejection-sampling attempts when enforcing clone CCF constraints.",
    )
    return parser


def simulation_config_from_args(args: argparse.Namespace) -> SimulationGridConfig:
    return SimulationGridConfig(
        out_dir=args.out_dir,
        purity_list=tuple(parse_float_list(args.purity_list)),
        amp_rate_list=tuple(parse_float_list(args.amp_rate_list)),
        N_list=tuple(parse_int_list(args.N_list)),
        n_samples_list=tuple(parse_int_list(args.n_samples_list)),
        reps=args.reps,
        seed=args.seed,
        K_min=args.K_min,
        K_max=args.K_max,
        lambda_mut=args.lambda_mut,
        lambda_mut_list=tuple(parse_int_list(args.lambda_mut_list)) if args.lambda_mut_list else None,
        alpha_mut=args.alpha_mut,
        alpha_lambda=args.alpha_lambda,
        tau_lineage_min=args.tau_lineage_min,
        tau_lineage_max=args.tau_lineage_max,
        purity_conc=args.purity_conc,
        lineage_zero_prob=args.lineage_zero_prob,
        min_clone_ccf=args.min_clone_ccf,
        min_clone_ccf_l2_norm=args.min_clone_ccf_l2_norm,
        min_mutations_per_clone=args.min_mutations_per_clone,
        min_clone_ccf_distance=args.min_clone_ccf_distance,
        max_rejection_tries=args.max_rejection_tries,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    written_dirs = run_simulation_grid_from_config(simulation_config_from_args(args))
    print(f"Generated {len(written_dirs)} simulated patients in {args.out_dir}")
    if written_dirs:
        print(written_dirs[: min(5, len(written_dirs))])


__all__ = [
    "SimulationGridConfig",
    "build_parser",
    "main",
    "parse_float_list",
    "parse_int_list",
    "run_simulation_grid",
    "run_simulation_grid_from_config",
    "sample_mutations_per_clone",
    "simulate_clonal_tree_ccf",
    "simulation_config_from_args",
    "write_patient_simulation",
]
