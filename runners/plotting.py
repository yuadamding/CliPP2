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


def _savefig(fig: plt.Figure, path: Path, pdf: PdfPages | None = None) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _line_panel(ax, df: pd.DataFrame, x: str, y: str, title: str, ylabel: str, color: str) -> None:
    plot_df = df[[x, y]].dropna().sort_values(x)
    ax.plot(plot_df[x], plot_df[y], marker="o", linewidth=2.2, color=color)
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel)


def _heatmap_panel(
    ax,
    fig: plt.Figure,
    table: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    cmap: str,
    fmt: str,
) -> None:
    im = ax.imshow(table.to_numpy(dtype=float), aspect="auto", cmap=cmap)
    ax.grid(False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(table.shape[1]), [str(v) for v in table.columns])
    ax.set_yticks(range(table.shape[0]), [str(v) for v in table.index])
    ax.tick_params(axis="both", labelsize=13)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.iloc[i, j]
            if pd.isna(value):
                text = "NA"
                color = "black"
            else:
                text = format(value, fmt)
                color = "white" if im.norm(value) > 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.84, label=cbar_label)


def plot_benchmark_outcomes(
    benchmark_dir: str | Path,
    output_dir: str | Path | None = None,
    cohort_name: str = "CliPP2 Benchmark Outcomes",
) -> dict[str, str]:
    benchmark_dir = Path(benchmark_dir)
    output_dir = _ensure_output_dir(output_dir or (benchmark_dir / "plots"))

    global_df = pd.read_csv(benchmark_dir / "benchmark_global.tsv", sep="\t")
    by_regions_path = benchmark_dir / "benchmark_by_n_regions.tsv"
    by_samples_path = benchmark_dir / "benchmark_by_n_samples.tsv"
    tumor_table_path = benchmark_dir / "benchmark_tumors.tsv"
    patient_table_path = benchmark_dir / "benchmark_patients.tsv"

    by_samples_df = pd.read_csv(
        by_regions_path if by_regions_path.exists() else by_samples_path,
        sep="\t",
    )
    by_depth_df = pd.read_csv(benchmark_dir / "benchmark_by_depth.tsv", sep="\t")
    by_true_k_df = pd.read_csv(benchmark_dir / "benchmark_by_true_k.tsv", sep="\t")
    by_purity_df = pd.read_csv(benchmark_dir / "benchmark_by_purity.tsv", sep="\t")
    by_amp_rate_df = pd.read_csv(benchmark_dir / "benchmark_by_amp_rate.tsv", sep="\t")
    scenario_df = pd.read_csv(benchmark_dir / "benchmark_by_scenario.tsv", sep="\t")
    patient_df = pd.read_csv(
        tumor_table_path if tumor_table_path.exists() else patient_table_path,
        sep="\t",
    )

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

    outputs: dict[str, str] = {}
    pdf_path = output_dir / "clipp2_benchmark_outcomes.pdf"
    outputs["pdf"] = str(pdf_path)

    worst20 = patient_df.nsmallest(20, "ARI").copy()
    lambda_df = (
        patient_df.groupby("lambda_mut_setting", dropna=False)
        .agg(
            mean_ARI=("ARI", "mean"),
            median_ARI=("ARI", "median"),
            mean_cp_rmse=("cp_rmse", "mean"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
            n_tumors=("tumor_id", "size"),
        )
        .reset_index()
        .sort_values("lambda_mut_setting")
    )
    ari_heat = scenario_df.pivot_table(index="n_regions", columns="N_mean", values="mean_ARI")
    rmse_heat = scenario_df.pivot_table(index="n_regions", columns="N_mean", values="mean_cp_rmse")
    lambda_heat = scenario_df.pivot_table(index="n_regions", columns="lambda_mut_setting", values="mean_ARI")

    g = global_df.iloc[0]
    worst_case = patient_df.loc[patient_df["ARI"].idxmin()]

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ax = axes[0, 0]
        summary_lines = [
            cohort_name,
            "",
            f"Tumors: {int(g['n_tumors'])}",
            f"Mean ARI: {g['mean_ARI']:.4f}",
            f"Median ARI: {g['median_ARI']:.4f}",
            f"Mean CP RMSE: {g['mean_cp_rmse']:.4f}",
            f"Mean multiplicity F1: {g['mean_multiplicity_f1']:.4f}",
            f"Mean clonal fraction error: {g['mean_clonal_fraction_error']:.4f}",
            f"Mean abs clonal fraction error: {g['mean_abs_clonal_fraction_error']:.4f}",
            f"Mean estimated clusters: {g['mean_estimated_clusters']:.3f}",
            f"Mean cluster-count error: {g['mean_cluster_count_error']:.3f}",
            f"Exact K match rate: {g['exact_cluster_count_match_rate']:.3f}",
            f"Mean elapsed seconds: {g['mean_elapsed_seconds']:.2f}",
            "",
            "Worst case:",
            f"{worst_case['tumor_id']}",
            f"ARI={worst_case['ARI']:.4f}, CP RMSE={worst_case['cp_rmse']:.4f}",
            f"Khat={int(worst_case['n_clusters'])}, Ktrue={int(worst_case['true_K'])}",
            f"lambda={worst_case['selected_lambda']:.4f}",
        ]
        ax.text(0.02, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=11, family="monospace")
        ax.set_axis_off()

        ax = axes[0, 1]
        metric_names = ["mean_ARI", "median_ARI", "mean_cp_rmse", "mean_multiplicity_f1", "exact_cluster_count_match_rate"]
        metric_labels = ["Mean ARI", "Median ARI", "Mean CP RMSE", "Mean mult. F1", "Exact K match"]
        metric_vals = [g[name] for name in metric_names]
        colors = ["#355C7D", "#6C5B7B", "#C06C84", "#F67280", "#99B898"]
        ax.bar(metric_labels, metric_vals, color=colors)
        ax.set_title("Headline Metrics")
        ax.tick_params(axis="x", rotation=20)

        ax = axes[1, 0]
        ax.hist(patient_df["ARI"], bins=30, color="#355C7D", alpha=0.9)
        ax.set_title("ARI Distribution")
        ax.set_xlabel("ARI")
        ax.set_ylabel("Tumors")

        ax = axes[1, 1]
        ax.scatter(
            patient_df["true_clonal_fraction"],
            patient_df["estimated_clonal_fraction"],
            c=patient_df["n_regions"],
            cmap="viridis",
            s=22,
            alpha=0.65,
            edgecolors="none",
        )
        lims = [0.0, 1.0]
        ax.plot(lims, lims, linestyle="--", color="black", linewidth=1.4)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title("Clonal Fraction Calibration")
        ax.set_xlabel("True clonal fraction")
        ax.set_ylabel("Estimated clonal fraction")
        _savefig(fig, output_dir / "00_outcome_summary.png", pdf)
        outputs["summary"] = str(output_dir / "00_outcome_summary.png")

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        _line_panel(axes[0, 0], by_samples_df, "n_regions", "mean_ARI", "Mean ARI By Regions", "Mean ARI", "#355C7D")
        _line_panel(axes[0, 1], by_samples_df, "n_regions", "median_ARI", "Median ARI By Regions", "Median ARI", "#6C5B7B")
        _line_panel(axes[0, 2], by_samples_df, "n_regions", "mean_cp_rmse", "CP RMSE By Regions", "Mean CP RMSE", "#C06C84")
        _line_panel(axes[1, 0], by_samples_df, "n_regions", "mean_multiplicity_f1", "Multiplicity F1 By Regions", "Mean F1", "#F67280")
        _line_panel(axes[1, 1], by_samples_df, "n_regions", "exact_cluster_count_match_rate", "Exact K Match By Regions", "Match rate", "#99B898")
        _line_panel(axes[1, 2], by_samples_df, "n_regions", "mean_elapsed_seconds", "Runtime By Regions", "Mean seconds", "#2A363B")
        _savefig(fig, output_dir / "01_metrics_by_regions.png", pdf)
        outputs["by_regions"] = str(output_dir / "01_metrics_by_regions.png")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        _line_panel(axes[0, 0], by_depth_df, "N_mean", "mean_ARI", "Mean ARI By Depth", "Mean ARI", "#355C7D")
        _line_panel(axes[0, 1], by_true_k_df, "true_K", "mean_ARI", "Mean ARI By True K", "Mean ARI", "#6C5B7B")
        _line_panel(axes[1, 0], by_purity_df, "purity", "mean_ARI", "Mean ARI By Purity", "Mean ARI", "#C06C84")
        _line_panel(axes[1, 1], lambda_df, "lambda_mut_setting", "mean_ARI", "Mean ARI By Mutation Setting", "Mean ARI", "#F67280")
        _savefig(fig, output_dir / "02_ari_across_axes.png", pdf)
        outputs["ari_axes"] = str(output_dir / "02_ari_across_axes.png")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        _line_panel(axes[0, 0], by_true_k_df, "true_K", "mean_estimated_clusters", "Estimated Clusters By True K", "Mean estimated clusters", "#355C7D")
        _line_panel(axes[0, 1], by_true_k_df, "true_K", "mean_cluster_count_error", "Cluster Count Error By True K", "Mean error", "#C06C84")
        _line_panel(axes[1, 0], by_samples_df, "n_regions", "mean_cluster_count_error", "Cluster Count Error By Regions", "Mean error", "#6C5B7B")
        _line_panel(axes[1, 1], by_samples_df, "n_regions", "exact_cluster_count_match_rate", "Exact K Match By Regions", "Match rate", "#99B898")
        _savefig(fig, output_dir / "03_cluster_recovery.png", pdf)
        outputs["cluster_recovery"] = str(output_dir / "03_cluster_recovery.png")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        _heatmap_panel(axes[0, 0], fig, ari_heat, "Mean ARI By Regions And Depth", "Depth", "Regions", "Mean ARI", "viridis", ".2f")
        _heatmap_panel(axes[0, 1], fig, rmse_heat, "Mean CP RMSE By Regions And Depth", "Depth", "Regions", "Mean CP RMSE", "magma_r", ".2f")
        _heatmap_panel(axes[1, 0], fig, lambda_heat, "Mean ARI By Regions And Mutation Setting", "lambda_mut", "Regions", "Mean ARI", "viridis", ".2f")
        axes[1, 1].axis("off")
        axes[1, 1].text(
            0.02,
            0.98,
            "Heatmap notes\n\n"
            "- Rows are number of sampled regions.\n"
            "- Columns are either depth or nominal mutation-count setting.\n"
            "- These summarize where the benchmark is easiest or hardest.\n"
            "- In this cohort, low-region settings are the main failure regime.",
            va="top",
            ha="left",
            fontsize=12,
        )
        _savefig(fig, output_dir / "04_heatmaps.png", pdf)
        outputs["heatmaps"] = str(output_dir / "04_heatmaps.png")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].hist(patient_df["elapsed_seconds"], bins=35, color="#355C7D", alpha=0.9)
        axes[0, 0].set_title("Elapsed Time Distribution")
        axes[0, 0].set_xlabel("Seconds")
        axes[0, 0].set_ylabel("Patients")

        axes[0, 1].scatter(
            patient_df["elapsed_seconds"],
            patient_df["ARI"],
            c=patient_df["n_regions"],
            cmap="viridis",
            s=18,
            alpha=0.6,
            edgecolors="none",
        )
        axes[0, 1].set_title("ARI vs Runtime")
        axes[0, 1].set_xlabel("Elapsed seconds")
        axes[0, 1].set_ylabel("ARI")

        axes[1, 0].hist(patient_df["cluster_count_error"], bins=np.arange(patient_df["cluster_count_error"].min() - 0.5, patient_df["cluster_count_error"].max() + 1.5, 1), color="#C06C84", alpha=0.9)
        axes[1, 0].set_title("Cluster Count Error Distribution")
        axes[1, 0].set_xlabel("Estimated K - True K")
        axes[1, 0].set_ylabel("Tumors")

        axes[1, 1].scatter(
            patient_df["true_K"],
            patient_df["n_clusters"],
            c=patient_df["ARI"],
            cmap="plasma",
            s=24,
            alpha=0.7,
            edgecolors="none",
        )
        lim = [1.5, max(patient_df["true_K"].max(), patient_df["n_clusters"].max()) + 0.5]
        axes[1, 1].plot(lim, lim, linestyle="--", color="black", linewidth=1.4)
        axes[1, 1].set_xlim(lim)
        axes[1, 1].set_ylim(lim)
        axes[1, 1].set_title("Estimated vs True K")
        axes[1, 1].set_xlabel("True K")
        axes[1, 1].set_ylabel("Estimated K")
        _savefig(fig, output_dir / "05_runtime_and_error_distributions.png", pdf)
        outputs["runtime_error"] = str(output_dir / "05_runtime_and_error_distributions.png")

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        worst_plot = worst20.sort_values("ARI", ascending=True).copy()
        labels = [pid.replace("_Lm", "\nLm") for pid in worst_plot["tumor_id"]]
        axes[0].barh(range(len(worst_plot)), worst_plot["ARI"], color="#C06C84")
        axes[0].set_yticks(range(len(worst_plot)), labels)
        axes[0].set_title("Worst 20 Cases By ARI")
        axes[0].set_xlabel("ARI")

        axes[1].scatter(
            worst20["ARI"],
            worst20["cp_rmse"],
            c=worst20["n_regions"],
            cmap="viridis",
            s=75,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.3,
        )
        for _, row in worst20.nsmallest(8, "ARI").iterrows():
            axes[1].annotate(
                row["tumor_id"],
                (row["ARI"], row["cp_rmse"]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
        axes[1].set_title("Worst-Case Tail")
        axes[1].set_xlabel("ARI")
        axes[1].set_ylabel("CP RMSE")
        _savefig(fig, output_dir / "06_worst_cases.png", pdf)
        outputs["worst_cases"] = str(output_dir / "06_worst_cases.png")

    return outputs
