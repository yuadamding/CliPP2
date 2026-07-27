"""Clonal-tree and mutation-allocation simulation."""

from __future__ import annotations

import numpy as np


def compute_exclusive_clone_fractions(
    children: list[list[int]],
    clone_ccf: np.ndarray,
    *,
    tol: float = 1e-8,
) -> np.ndarray:
    """Recover terminal-genotype fractions from cumulative clone CCFs."""

    ccf = np.asarray(clone_ccf, dtype=float)
    if ccf.ndim not in (1, 2):
        raise ValueError("clone_ccf must have shape (K,) or (K, S).")
    if ccf.shape[0] != len(children):
        raise ValueError("children and clone_ccf disagree on the clone count.")

    exclusive = ccf.copy()
    for clone_id, child_ids in enumerate(children):
        if child_ids:
            exclusive[clone_id] -= np.sum(ccf[np.asarray(child_ids, dtype=int)], axis=0)

    if np.any(exclusive < -tol):
        minimum = float(np.min(exclusive))
        raise ValueError(
            f"Tree CCFs imply a negative exclusive clone fraction ({minimum:.6g})."
        )
    exclusive[np.abs(exclusive) <= tol] = 0.0

    expected_total = ccf[0]
    actual_total = np.sum(exclusive, axis=0)
    if not np.allclose(actual_total, expected_total, atol=tol, rtol=0.0):
        raise ValueError("Exclusive clone fractions do not reconstruct the root CCF.")
    return exclusive


def _check_patient_tree_and_ccf(parent, children, ccf_patient_clones, tol=1e-8):
    parent = np.asarray(parent, dtype=int)
    ccf = np.asarray(ccf_patient_clones, dtype=float)
    K = parent.shape[0]

    assert K >= 1, "Tree must have at least one node."
    assert parent[0] == -1, "Root must have parent -1."
    for k in range(1, K):
        p = parent[k]
        assert 0 <= p < K, f"Invalid parent[{k}]={p}; must be in [0,{K - 1}]."

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
    fraction_random_state=None,
):
    if isinstance(random_state, np.random.Generator):
        topology_rng = random_state
    else:
        topology_rng = np.random.default_rng(random_state)
    if fraction_random_state is None:
        fraction_rng = topology_rng
    elif isinstance(fraction_random_state, np.random.Generator):
        fraction_rng = fraction_random_state
    else:
        fraction_rng = np.random.default_rng(fraction_random_state)

    if float(alpha_split) <= 0.0:
        raise ValueError("alpha_split must be positive.")
    if not 0.0 <= float(lineage_zero_prob) < 1.0:
        raise ValueError("lineage_zero_prob must be in [0, 1).")
    if lineage_zero_prob > eps and min_clone_ccf > 0.0:
        raise ValueError(
            "lineage_zero_prob must be 0 when enforcing a strictly positive minimum clone CCF in every region."
        )

    topology_rejections = 0
    rejection_limit = max(int(max_rejection_tries), 1)
    for generation_attempt in range(1, rejection_limit + 1):
        while True:
            parent = np.empty(K, dtype=int)
            parent[0] = -1
            children = [[] for _ in range(K)]
            for k in range(1, K):
                if np.isclose(float(alpha_split), 1.0):
                    parent[k] = int(topology_rng.integers(0, k))
                else:
                    child_counts = np.asarray(
                        [len(children[candidate]) for candidate in range(k)],
                        dtype=float,
                    )
                    parent_weights = np.power(
                        1.0 + child_counts, float(alpha_split) - 1.0
                    )
                    parent_weights /= parent_weights.sum()
                    parent[k] = int(topology_rng.choice(k, p=parent_weights))
                children[parent[k]].append(k)

            leaves_all = [k for k in range(K) if len(children[k]) == 0]
            is_pure_chain = (
                K > 1 and len(leaves_all) == 1 and all(len(ch) <= 1 for ch in children)
            )
            if not is_pure_chain or K <= 5:
                break
            topology_rejections += 1

        if is_pure_chain:
            base_min = 0.2
            if base_min * K > 1.0 + 1e-10:
                raise ValueError(
                    f"Cannot enforce λ_k >= 0.2 for pure chain with K={K}."
                )
            if abs(base_min * K - 1.0) <= 1e-10:
                lambda_k = np.full(K, base_min, dtype=float)
            else:
                leftover = 1.0 - base_min * K
                lambda_k = base_min + leftover * fraction_rng.dirichlet(np.ones(K))
        else:
            lambda_k = fraction_rng.dirichlet(alpha_lambda * np.ones(K))

        ccf_patient_clones = np.zeros(K, dtype=float)
        for k in reversed(range(K)):
            ccf_patient_clones[k] = lambda_k[k] + sum(
                ccf_patient_clones[c] for c in children[k]
            )

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
        lineage_floor = max(
            float(min_clone_ccf),
            float(min_clone_ccf_l2_norm) / np.sqrt(float(n_samples)),
        )
        if lineage_floor * L >= 1.0 - eps:
            raise ValueError(
                f"Cannot enforce lineage floor {lineage_floor:.4f} with {L} lineages; floor * lineages must stay below 1."
            )

        tau_arr = np.asarray(tau, dtype=float)
        if tau_arr.ndim == 0:
            tau_vec = np.full(n_samples, float(tau_arr), dtype=float)
        else:
            if tau_arr.shape[0] != n_samples:
                raise ValueError(
                    "If `tau` is array-like, its length must be n_samples."
                )
            tau_vec = tau_arr

        ccf_samples_lineages = np.zeros((L, n_samples), dtype=float)
        for j in range(n_samples):
            if lineage_zero_prob > eps:
                present = fraction_rng.random(L) >= float(lineage_zero_prob)
                if not np.any(present):
                    present[int(fraction_rng.integers(0, L))] = True
            else:
                present = np.ones(L, dtype=bool)

            present_weights = u_safe[present]
            present_weights = present_weights / present_weights.sum()
            if tau_vec[j] <= 0:
                lambda_present = present_weights
            else:
                lambda_present = fraction_rng.dirichlet(tau_vec[j] * present_weights)

            n_present = int(np.sum(present))
            present_leftover = 1.0 - lineage_floor * n_present
            ccf_samples_lineages[present, j] = (
                lineage_floor + present_leftover * lambda_present
            )

        ccf_samples_clones = A @ ccf_samples_lineages
        _check_sample_ccf_against_tree(parent, children, ccf_samples_clones, tol=1e-8)
        if _min_clone_region_ccf(ccf_samples_clones) < float(min_clone_ccf) - eps:
            continue
        if _min_clone_l2_norm(ccf_samples_clones) < float(min_clone_ccf_l2_norm) - eps:
            continue
        if (
            _min_pairwise_clone_distance(ccf_samples_clones)
            < float(min_clone_ccf_distance) - eps
        ):
            continue

        exclusive_clone_fraction_patient = compute_exclusive_clone_fractions(
            children,
            ccf_patient_clones,
        )
        exclusive_clone_fraction_samples = compute_exclusive_clone_fractions(
            children,
            ccf_samples_clones,
        )
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
            "exclusive_clone_fraction_patient": exclusive_clone_fraction_patient,
            "exclusive_clone_fraction_samples": exclusive_clone_fraction_samples,
            "clone_ccf_patient": ccf_patient_clones,
            "clone_ccf_samples": ccf_samples_clones,
            "generation_attempts": int(generation_attempt),
            "topology_rejections": int(topology_rejections),
        }

    raise RuntimeError(
        "Failed to generate a clonal tree satisfying the region-level clone CCF and pairwise clone-separation constraints."
    )


def sample_mutations_per_clone(
    ccf_patient_clones,
    lambda_mut=800,
    mutation_count: int | None = None,
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
    min_total = K * min_mutations_per_clone
    if mutation_count is None:
        N_mut = max(int(rng.poisson(lambda_mut)), min_total)
    else:
        N_mut = int(mutation_count)
        if N_mut < min_total:
            raise ValueError(
                "mutation_count must be at least clone_count * min_mutations_per_clone."
            )

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


__all__ = [
    "compute_exclusive_clone_fractions",
    "sample_mutations_per_clone",
    "simulate_clonal_tree_ccf",
]
