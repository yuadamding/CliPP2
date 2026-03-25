from __future__ import annotations

import argparse
import itertools as its
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bernoulli


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
    alpha_mut: float = 10.0
    alpha_split: float = 1.0
    alpha_lambda: float = 5.0
    tau_lineage_min: float = 1.0
    tau_lineage_max: float = 50.0
    purity_conc: float = 50.0
    lineage_zero_prob: float = 0.3


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
):
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

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

    tau_arr = np.asarray(tau, dtype=float)
    if tau_arr.ndim == 0:
        tau_vec = np.full(n_samples, float(tau_arr), dtype=float)
    else:
        if tau_arr.shape[0] != n_samples:
            raise ValueError("If `tau` is array-like, its length must be n_samples.")
        tau_vec = tau_arr

    ccf_samples_lineages = np.zeros((L, n_samples), dtype=float)
    for j in range(n_samples):
        if lineage_zero_prob <= 0.0:
            present_mask = np.ones(L, dtype=bool)
        else:
            present_mask = rng.random(L) > lineage_zero_prob
            if not present_mask.any():
                present_mask[int(np.argmax(u_safe))] = True

        present_idx = np.where(present_mask)[0]
        lambda_present = ccf_patient_lineages[present_idx]
        u_present = u_safe[present_idx]

        if tau_vec[j] <= 0:
            v_present = lambda_present.copy()
        else:
            alpha_present = tau_vec[j] * u_present
            w_present = rng.dirichlet(alpha_present)
            v_present = w_present * lambda_present

        total_v = v_present.sum()
        if total_v <= 0:
            lambda_sample_present = lambda_present / lambda_present.sum()
        else:
            lambda_sample_present = v_present / total_v

        lambda_j = np.zeros(L, dtype=float)
        lambda_j[present_idx] = lambda_sample_present
        ccf_samples_lineages[:, j] = lambda_j

    ccf_samples_clones = A @ ccf_samples_lineages
    _check_sample_ccf_against_tree(parent, children, ccf_samples_clones, tol=1e-8)

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


def sample_mutations_per_clone(
    ccf_patient_clones,
    lambda_mut=800,
    alpha_mut=10.0,
    random_state=None,
):
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    ccf = np.asarray(ccf_patient_clones, dtype=float)
    K = ccf.shape[0]

    N_mut = rng.poisson(lambda_mut)
    if N_mut < K:
        N_mut = K

    base = np.maximum(ccf, 0.0)
    if base.sum() <= 0:
        p0 = np.full(K, 1.0 / K, dtype=float)
    else:
        base = base + 1e-6
        p0 = base / base.sum()

    theta = rng.dirichlet(alpha_mut * p0)
    base_counts = np.ones(K, dtype=int)
    remaining = N_mut - K
    if remaining > 0:
        extra = rng.multinomial(remaining, theta)
        cluster_size = base_counts + extra
    else:
        cluster_size = base_counts

    cluster_id = np.repeat(np.arange(K), cluster_size)
    rng.shuffle(cluster_id)
    return cluster_id.astype(int), cluster_size.astype(int), int(N_mut)


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
        random_state=rng,
    )

    data_dir = out_dir / f"{N_mean}_{K}_{simu_purity}_{amp_rate}_S{n_samples}_M{no_mutations}_rep{sim}"
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
        minor_j = np.ones(no_mutations, dtype=int)
        total_j = 2 * np.ones(no_mutations, dtype=int)
        if amp_rate > 0.0:
            tmp = bernoulli.rvs(amp_rate, size=no_mutations, random_state=rng).astype(bool)
            n_amp = int(tmp.sum())
            if n_amp > 0:
                minor_amp = rng.integers(1, 5, size=n_amp)
                ref_amp = rng.integers(0, 5, size=n_amp)
                minor_j[tmp] = minor_amp
                total_j[tmp] = minor_amp + ref_amp

        major_cn_j = np.maximum(total_j - minor_j, minor_j)
        minor_cn_j = np.minimum(total_j - minor_j, minor_j)
        pd.DataFrame(
            {
                "chromosome_index": np.ones(no_mutations, dtype=int),
                "start_position": [3 * i + 1 for i in range(no_mutations)],
                "end_position": [3 * i + 3 for i in range(no_mutations)],
                "major_cn": major_cn_j,
                "minor_cn": minor_cn_j,
                "total_cn": total_j,
                "multiplicity": minor_j,
            }
        ).to_csv(data_dir / f"sample{j}" / "cna.txt", sep="\t", index=False)

        mutation_ccf_sample_j = ccf_samples_clones[cluster_id, j]
        purity_j = sample_purities[j]
        n_j = rng.poisson(N_mean, size=no_mutations)
        vaf_j = purity_j * mutation_ccf_sample_j * minor_j / (2 * (1 - purity_j) + purity_j * total_j)
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
    lineage_zero_prob: float = 0.3,
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
    alpha_mut: float = 10.0,
    alpha_split: float = 1.0,
    alpha_lambda: float = 5.0,
    tau_lineage_min: float = 1.0,
    tau_lineage_max: float = 50.0,
    purity_conc: float = 50.0,
    lineage_zero_prob: float = 0.3,
) -> list[Path]:
    if purity_list is None:
        purity_list = list(SimulationGridConfig.purity_list)
    if amp_rate_list is None:
        amp_rate_list = list(SimulationGridConfig.amp_rate_list)
    if N_list is None:
        N_list = list(SimulationGridConfig.N_list)
    if n_samples_list is None:
        n_samples_list = list(SimulationGridConfig.n_samples_list)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.default_rng(seed)
    written_dirs: list[Path] = []

    for N_mean, simu_purity, amp_rate, n_samples in its.product(N_list, purity_list, amp_rate_list, n_samples_list):
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
                    lambda_mut=lambda_mut,
                    alpha_mut=alpha_mut,
                    alpha_split=alpha_split,
                    alpha_lambda=alpha_lambda,
                    tau_lineage_min=tau_lineage_min,
                    tau_lineage_max=tau_lineage_max,
                    purity_conc=purity_conc,
                    lineage_zero_prob=lineage_zero_prob,
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
        alpha_mut=config.alpha_mut,
        alpha_split=config.alpha_split,
        alpha_lambda=config.alpha_lambda,
        tau_lineage_min=config.tau_lineage_min,
        tau_lineage_max=config.tau_lineage_max,
        purity_conc=config.purity_conc,
        lineage_zero_prob=config.lineage_zero_prob,
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
    parser.add_argument("--lambda-mut", type=int, default=2000, help="Poisson mean for mutation count.")
    parser.add_argument("--alpha-mut", type=float, default=10.0, help="Dirichlet concentration for mutation allocation.")
    parser.add_argument("--alpha-lambda", type=float, default=5.0, help="Dirichlet concentration for lineage residual masses.")
    parser.add_argument("--tau-lineage-min", type=float, default=1.0, help="Minimum lineage concentration per sample.")
    parser.add_argument("--tau-lineage-max", type=float, default=50.0, help="Maximum lineage concentration per sample.")
    parser.add_argument("--purity-conc", type=float, default=50.0, help="Beta concentration for sample purities.")
    parser.add_argument("--lineage-zero-prob", type=float, default=0.3, help="Probability of zeroing a lineage in a sample.")
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
        alpha_mut=args.alpha_mut,
        alpha_lambda=args.alpha_lambda,
        tau_lineage_min=args.tau_lineage_min,
        tau_lineage_max=args.tau_lineage_max,
        purity_conc=args.purity_conc,
        lineage_zero_prob=args.lineage_zero_prob,
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
