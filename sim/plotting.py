from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


def _ensure_output_dir(output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_retry_table(summary_dir: Path) -> pd.DataFrame:
    path = summary_dir / "seed_retry_log.tsv"
    if not path.exists():
        return pd.DataFrame(columns=["patient_id", "seed_offset_used"])
    return pd.read_csv(path, sep="\t")


def _add_bar_labels(ax, fmt: str = "{:.0f}") -> None:
    for patch in ax.patches:
        h = patch.get_height()
        ax.annotate(
            fmt.format(h),
            (patch.get_x() + patch.get_width() / 2.0, h),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 3),
            textcoords="offset points",
        )


def _savefig(fig: plt.Figure, path: Path, pdf: PdfPages | None = None) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _infer_case_root(summary_dir: Path) -> Path | None:
    if summary_dir.name.endswith("_summary"):
        candidate = summary_dir.parent / summary_dir.name[: -len("_summary")]
        if candidate.exists():
            return candidate
    return None


def _pick_representative_patient_ids(patient_df: pd.DataFrame) -> list[str]:
    reps: list[str] = []
    for n_samples in sorted(patient_df["n_samples"].unique()):
        subset = patient_df.loc[patient_df["n_samples"] == n_samples].copy()
        if subset.empty:
            continue
        score = (
            np.abs(np.log(subset["lambda_mut_setting"] / 1000.0))
            + np.abs(np.log(subset["N_mean"] / 300.0))
            + 2.0 * np.abs(subset["purity"] - 0.9)
            + 2.0 * np.abs(subset["amp_rate"] - 0.2)
            + np.abs(subset["true_K"] - subset["true_K"].median()) / 10.0
            + 0.1 * subset["rep"]
        )
        chosen = subset.loc[score.idxmin(), "patient_id"]
        reps.append(chosen)
    return reps


def _load_case_truth(case_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    truth = pd.read_csv(case_dir / "truth.txt", sep="\t")
    clone_patient = pd.read_csv(case_dir / "truth_clone_patient.txt", sep="\t").sort_values("clone_id")
    clone_sample = pd.read_csv(case_dir / "truth_clone_sample.txt", sep="\t")
    pivot = clone_sample.pivot(index="clone_id", columns="sample_id", values="ccf").sort_index().sort_index(axis=1)
    patient_ccf = clone_patient["ccf"].to_numpy(dtype=float)
    sample_ccf = pivot.to_numpy(dtype=float)
    cluster_sizes = truth["cluster_id"].value_counts().sort_index().reindex(clone_patient["clone_id"], fill_value=0).to_numpy(dtype=int)
    clone_ids = clone_patient["clone_id"].tolist()
    return patient_ccf, sample_ccf, cluster_sizes, clone_ids


def _infer_parent_from_ccf(patient_ccf: np.ndarray, sample_ccf: np.ndarray, eps: float = 1e-8) -> dict[int, int]:
    k_count = len(patient_ccf)
    root = int(np.argmax(patient_ccf))
    parent: dict[int, int] = {root: -1}
    for k in sorted([idx for idx in range(k_count) if idx != root], key=lambda idx: (patient_ccf[idx], idx)):
        candidates: list[tuple[float, float, int]] = []
        for j in range(k_count):
            if j == k:
                continue
            if patient_ccf[j] + eps < patient_ccf[k]:
                continue
            if np.any(sample_ccf[j] + eps < sample_ccf[k]):
                continue
            if not np.any(sample_ccf[j] > sample_ccf[k] + eps) and not (patient_ccf[j] > patient_ccf[k] + eps):
                continue
            candidates.append(
                (
                    float(patient_ccf[j] - patient_ccf[k]),
                    float(np.mean(sample_ccf[j] - sample_ccf[k])),
                    int(j),
                )
            )
        parent[k] = min(candidates)[2] if candidates else root
    return parent


def _children_from_parent(parent: dict[int, int]) -> dict[int, list[int]]:
    children = {node: [] for node in parent}
    for node, p in parent.items():
        if p >= 0:
            children.setdefault(p, []).append(node)
    for node in children:
        children[node] = sorted(children[node])
    return children


def _tree_order(root: int, children: dict[int, list[int]]) -> list[int]:
    order: list[int] = []

    def dfs(node: int) -> None:
        order.append(node)
        for child in children.get(node, []):
            dfs(child)

    dfs(root)
    return order


def _tree_positions(root: int, children: dict[int, list[int]]) -> dict[int, tuple[float, float]]:
    pos: dict[int, tuple[float, float]] = {}
    next_x = 0.0

    def dfs(node: int, depth: int) -> float:
        nonlocal next_x
        child_nodes = children.get(node, [])
        if not child_nodes:
            x = next_x
            next_x += 1.0
        else:
            child_x = [dfs(child, depth + 1) for child in child_nodes]
            x = float(np.mean(child_x))
        pos[node] = (x, -float(depth))
        return x

    dfs(root, 0)
    return pos


def _plot_patient_tree(ax, clone_ids: list[int], patient_ccf: np.ndarray, cluster_sizes: np.ndarray, parent: dict[int, int]) -> list[int]:
    root = next(node for node, p in parent.items() if p < 0)
    children = _children_from_parent(parent)
    pos = _tree_positions(root, children)
    order = _tree_order(root, children)

    for node, p in parent.items():
        if p < 0:
            continue
        x0, y0 = pos[p]
        x1, y1 = pos[node]
        ax.plot([x0, x1], [y0, y1], color="#4c566a", linewidth=2.0, zorder=1)

    node_size = np.clip(cluster_sizes[order] / max(cluster_sizes.max(), 1) * 1400.0, 300.0, 1400.0)
    xs = [pos[node][0] for node in order]
    ys = [pos[node][1] for node in order]
    colors = [patient_ccf[node] for node in order]
    sc = ax.scatter(xs, ys, s=node_size, c=colors, cmap="magma", edgecolors="black", linewidths=0.8, zorder=3)
    for node in order:
        x, y = pos[node]
        ax.text(
            x,
            y - 0.18,
            f"C{clone_ids[node]}\nCCF={patient_ccf[node]:.2f}\nN={cluster_sizes[node]}",
            ha="center",
            va="top",
            fontsize=8,
        )
    ax.set_title("Patient-Level Phylogenetic Tree")
    ax.set_axis_off()
    return order


def _plot_region_ccf_matrix(ax, fig: plt.Figure, clone_ids: list[int], sample_ccf: np.ndarray, cluster_sizes: np.ndarray, patient_ccf: np.ndarray, order: list[int]) -> None:
    mat = sample_ccf[order, :]
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.grid(False)
    n_rows, n_cols = mat.shape
    font_size = 10 if n_cols <= 5 else 9 if n_cols <= 10 else 7
    ax.set_title("Clone CCF Matrix By Region")
    ax.set_xlabel("Region")
    ax.set_ylabel("Clone")
    ax.set_xticks(range(n_cols), [f"R{j}" for j in range(n_cols)])
    ax.set_yticks(
        range(n_rows),
        [f"C{clone_ids[idx]} | N={cluster_sizes[idx]} | P={patient_ccf[idx]:.2f}" for idx in order],
    )
    for i in range(n_rows):
        for j in range(n_cols):
            value = mat[i, j]
            color = "white" if value < 0.45 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=font_size)
    fig.colorbar(im, ax=ax, shrink=0.82, label="CCF")


def plot_simulation_cohort_summary(
    summary_dir: str | Path,
    output_dir: str | Path | None = None,
    cohort_name: str = "CliPP2 Simulation Cohort",
) -> dict[str, str]:
    summary_dir = Path(summary_dir)
    output_dir = _ensure_output_dir(output_dir or (summary_dir / "plots"))

    patient_df = pd.read_csv(summary_dir / "cohort_patients.tsv", sep="\t")
    global_df = pd.read_csv(summary_dir / "cohort_global.tsv", sep="\t")
    design_df = pd.read_csv(summary_dir / "cohort_design.tsv", sep="\t")
    retry_df = _load_retry_table(summary_dir)
    raw_case_root = _infer_case_root(summary_dir)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    pdf_path = output_dir / "simulation_cohort_overview.pdf"
    outputs: dict[str, str] = {"pdf": str(pdf_path)}

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        counts = patient_df["n_samples"].value_counts().sort_index()
        counts.plot(kind="bar", ax=axes[0], color="#355C7D")
        axes[0].set_title("Cases By Number Of Regions")
        axes[0].set_xlabel("Regions")
        axes[0].set_ylabel("Cases")
        _add_bar_labels(axes[0])

        depth_counts = patient_df["N_mean"].value_counts().sort_index()
        depth_counts.plot(kind="bar", ax=axes[1], color="#6C5B7B")
        axes[1].set_title("Cases By Read Depth")
        axes[1].set_xlabel("Nominal Depth")
        axes[1].set_ylabel("Cases")
        _add_bar_labels(axes[1])

        lambda_counts = patient_df["lambda_mut_setting"].value_counts().sort_index()
        lambda_counts.plot(kind="bar", ax=axes[2], color="#C06C84")
        axes[2].set_title("Cases By Mutation-Count Setting")
        axes[2].set_xlabel("lambda_mut")
        axes[2].set_ylabel("Cases")
        _add_bar_labels(axes[2])

        k_counts = patient_df["true_K"].value_counts().sort_index()
        k_counts.plot(kind="bar", ax=axes[3], color="#F67280")
        axes[3].set_title("True Clone Count Distribution")
        axes[3].set_xlabel("True K")
        axes[3].set_ylabel("Cases")
        _add_bar_labels(axes[3])

        purity_counts = patient_df["purity"].value_counts().sort_index()
        purity_counts.plot(kind="bar", ax=axes[4], color="#F8B195")
        axes[4].set_title("Cases By Nominal Purity")
        axes[4].set_xlabel("Nominal Purity")
        axes[4].set_ylabel("Cases")
        _add_bar_labels(axes[4])

        amp_counts = patient_df["amp_rate"].value_counts().sort_index()
        amp_counts.plot(kind="bar", ax=axes[5], color="#99B898")
        axes[5].set_title("Cases By CNA Rate")
        axes[5].set_xlabel("CNA Rate")
        axes[5].set_ylabel("Cases")
        _add_bar_labels(axes[5])

        fig.suptitle(f"{cohort_name}: Design Overview", fontsize=18, y=1.02)
        path = output_dir / "00_design_overview.png"
        outputs["design_overview"] = str(path)
        _savefig(fig, path, pdf)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        axes[0].hist(patient_df["n_mutations"], bins=40, color="#355C7D", edgecolor="white")
        axes[0].set_title("Realized Mutation Count")
        axes[0].set_xlabel("Mutations")
        axes[0].set_ylabel("Cases")

        box_data = [
            patient_df.loc[patient_df["lambda_mut_setting"] == val, "n_mutations"].to_numpy()
            for val in sorted(patient_df["lambda_mut_setting"].unique())
        ]
        axes[1].boxplot(box_data, tick_labels=[str(v) for v in sorted(patient_df["lambda_mut_setting"].unique())])
        axes[1].set_title("Realized Mutations By lambda_mut")
        axes[1].set_xlabel("lambda_mut")
        axes[1].set_ylabel("Mutations")

        scatter = axes[2].scatter(
            patient_df["true_K"],
            patient_df["n_mutations"],
            c=patient_df["n_samples"],
            cmap="viridis",
            alpha=0.65,
            s=26,
        )
        axes[2].set_title("Mutations Vs True K")
        axes[2].set_xlabel("True K")
        axes[2].set_ylabel("Mutations")
        cbar = fig.colorbar(scatter, ax=axes[2])
        cbar.set_label("Regions")

        by_lambda = patient_df.groupby("lambda_mut_setting", as_index=False)["n_mutations"].mean()
        axes[3].plot(by_lambda["lambda_mut_setting"], by_lambda["n_mutations"], marker="o", color="#C06C84")
        axes[3].set_title("Mean Realized Mutations By lambda_mut")
        axes[3].set_xlabel("lambda_mut")
        axes[3].set_ylabel("Mean Mutations")

        fig.suptitle(f"{cohort_name}: Mutation Count Realization", fontsize=18, y=1.02)
        path = output_dir / "01_mutation_realization.png"
        outputs["mutation_realization"] = str(path)
        _savefig(fig, path, pdf)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        axes[0].hist(patient_df["min_clone_ccf_realized"], bins=40, color="#6C5B7B", edgecolor="white")
        axes[0].axvline(patient_df["min_clone_ccf_realized"].min(), color="black", linestyle="--", linewidth=1)
        axes[0].set_title("Minimum Clone CCF Per Case")
        axes[0].set_xlabel("Min Clone CCF")
        axes[0].set_ylabel("Cases")

        axes[1].hist(patient_df["min_clone_ccf_distance_realized"], bins=40, color="#99B898", edgecolor="white")
        axes[1].axvline(patient_df["min_clone_ccf_distance_realized"].min(), color="black", linestyle="--", linewidth=1)
        axes[1].set_title("Minimum Inter-Clone CCF Distance Per Case")
        axes[1].set_xlabel("Min RMS CCF Distance")
        axes[1].set_ylabel("Cases")

        axes[2].hist(patient_df["min_mutations_per_clone_realized"], bins=np.arange(14.5, patient_df["min_mutations_per_clone_realized"].max() + 1.5, 1), color="#F67280", edgecolor="white")
        axes[2].set_title("Minimum Mutations Per Clone Per Case")
        axes[2].set_xlabel("Min Mutations Per Clone")
        axes[2].set_ylabel("Cases")

        for purity, grp in patient_df.groupby("purity"):
            axes[3].hist(grp["sample_purity_mean"], bins=25, alpha=0.65, label=f"nominal={purity}")
        axes[3].set_title("Realized Mean Sample Purity")
        axes[3].set_xlabel("Mean Sample Purity")
        axes[3].set_ylabel("Cases")
        axes[3].legend(frameon=True)

        fig.suptitle(f"{cohort_name}: Constraint Checks", fontsize=18, y=1.02)
        path = output_dir / "02_constraint_checks.png"
        outputs["constraint_checks"] = str(path)
        _savefig(fig, path, pdf)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        pivot_k = (
            patient_df.groupby(["n_samples", "lambda_mut_setting"])["true_K"]
            .mean()
            .unstack("lambda_mut_setting")
            .sort_index()
        )
        pivot_dist = (
            patient_df.groupby(["n_samples", "lambda_mut_setting"])["min_clone_ccf_distance_realized"]
            .mean()
            .unstack("lambda_mut_setting")
            .sort_index()
        )

        im0 = axes[0].imshow(pivot_k.to_numpy(), aspect="auto", cmap="magma")
        axes[0].set_title("Mean True K")
        axes[0].set_xlabel("lambda_mut")
        axes[0].set_ylabel("Regions")
        axes[0].set_xticks(range(len(pivot_k.columns)), [str(c) for c in pivot_k.columns])
        axes[0].set_yticks(range(len(pivot_k.index)), [str(i) for i in pivot_k.index])
        for i in range(pivot_k.shape[0]):
            for j in range(pivot_k.shape[1]):
                axes[0].text(j, i, f"{pivot_k.iat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(im0, ax=axes[0], shrink=0.85)

        im1 = axes[1].imshow(pivot_dist.to_numpy(), aspect="auto", cmap="viridis")
        axes[1].set_title("Mean Min Inter-Clone Distance")
        axes[1].set_xlabel("lambda_mut")
        axes[1].set_ylabel("Regions")
        axes[1].set_xticks(range(len(pivot_dist.columns)), [str(c) for c in pivot_dist.columns])
        axes[1].set_yticks(range(len(pivot_dist.index)), [str(i) for i in pivot_dist.index])
        for i in range(pivot_dist.shape[0]):
            for j in range(pivot_dist.shape[1]):
                axes[1].text(j, i, f"{pivot_dist.iat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(im1, ax=axes[1], shrink=0.85)

        fig.suptitle(f"{cohort_name}: Stratified Heatmaps", fontsize=18, y=1.02)
        path = output_dir / "03_stratified_heatmaps.png"
        outputs["stratified_heatmaps"] = str(path)
        _savefig(fig, path, pdf)

        if raw_case_root is not None:
            representative_ids = _pick_representative_patient_ids(patient_df)
            for page_idx, patient_id in enumerate(representative_ids, start=4):
                case_dir = raw_case_root / patient_id
                if not case_dir.exists():
                    continue
                patient_ccf, sample_ccf, cluster_sizes, clone_ids = _load_case_truth(case_dir)
                parent = _infer_parent_from_ccf(patient_ccf, sample_ccf)
                fig, axes = plt.subplots(1, 2, figsize=(16, 7))
                order = _plot_patient_tree(axes[0], clone_ids, patient_ccf, cluster_sizes, parent)
                _plot_region_ccf_matrix(axes[1], fig, clone_ids, sample_ccf, cluster_sizes, patient_ccf, order)
                meta = patient_df.loc[patient_df["patient_id"] == patient_id].iloc[0]
                fig.suptitle(
                    (
                        f"{cohort_name}: {patient_id}\n"
                        f"S={int(meta['n_samples'])}, true K={int(meta['true_K'])}, "
                        f"N={int(meta['N_mean'])}, lambda_mut={int(meta['lambda_mut_setting'])}, "
                        f"purity={meta['purity']:.1f}, amp={meta['amp_rate']:.1f}"
                    ),
                    fontsize=16,
                    y=1.03,
                )
                path = output_dir / f"{page_idx:02d}_{patient_id}_patient_tree_and_region_ccf.png"
                outputs[f"patient_{patient_id}"] = str(path)
                _savefig(fig, path, pdf)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        retries = retry_df["seed_offset_used"].value_counts().sort_index() if not retry_df.empty else pd.Series(dtype=int)
        if retries.empty:
            axes[0].text(0.5, 0.5, "No seed retries needed", ha="center", va="center", fontsize=14)
            axes[0].set_axis_off()
        else:
            retries.plot(kind="bar", ax=axes[0], color="#355C7D")
            axes[0].set_title("Seed Retry Offsets Used")
            axes[0].set_xlabel("Seed Offset")
            axes[0].set_ylabel("Cases")
            _add_bar_labels(axes[0])

        axes[1].axis("off")
        global_row = global_df.iloc[0]
        design_row = design_df.iloc[0]
        summary_text = "\n".join(
            [
                f"Cases: {int(global_row['n_cases'])}",
                f"TSVs: {int(global_row['n_tsv_files'])}",
                f"Mean true K: {global_row['mean_true_K']:.3f}",
                f"Mean mutations: {global_row['mean_n_mutations']:.1f}",
                f"Mutation range: {int(global_row['min_n_mutations'])}-{int(global_row['max_n_mutations'])}",
                f"Min clone CCF realized: {global_row['min_clone_ccf_realized']:.4f}",
                f"Min clone distance realized: {global_row['min_clone_ccf_distance_realized']:.4f}",
                f"Min mutations/clone realized: {int(global_row['min_mutations_per_clone_realized'])}",
                f"Nominal purity values: {design_row['purity_values']}",
                f"Nominal CNA rates: {design_row['amp_rate_values']}",
                f"Region counts: {design_row['n_samples_values']}",
                f"lambda_mut values: {design_row['lambda_mut_values']}",
                f"Retry cases: {len(retry_df)}",
            ]
        )
        axes[1].text(
            0.0,
            1.0,
            summary_text,
            ha="left",
            va="top",
            fontsize=13,
            family="monospace",
        )

        fig.suptitle(f"{cohort_name}: Summary", fontsize=18, y=1.02)
        path = output_dir / "09_summary_page.png"
        outputs["summary_page"] = str(path)
        _savefig(fig, path, pdf)

    return outputs


__all__ = ["plot_simulation_cohort_summary"]
