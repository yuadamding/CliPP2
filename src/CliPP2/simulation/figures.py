"""Six data-driven scientific figure groups for the biological simulation study."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import FAMILIES, TRANSITIONS


def _unavailable(ax, message="Unavailable for the executed study cells"):
    ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes, color="0.4", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _values(evaluations, name, arm, *, view="", stratum="all"):
    rows = []
    for evaluation in evaluations:
        matches = [row for row in evaluation["metrics"] if row["name"] == name and row["arm"] == arm
                   and row.get("view", "") == view and row.get("stratum", "all") == stratum]
        if matches and isinstance(matches[0]["value"], (int, float)):
            rows.append((evaluation, float(matches[0]["value"])))
    return rows


def _save(figure, output, name, paths):
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        path = output / (name+"."+suffix)
        if path.exists():
            raise FileExistsError(path)
        figure.savefig(path, dpi=170, bbox_inches="tight")
        paths.append(str(path))
    plt.close(figure)


def _physical(histories, output, paths):
    chosen = [h for h in histories if h.family in ("S03", "S04", "S05")][:3]
    if not chosen:
        chosen = list(histories)[:1]
    figure, axes = plt.subplots(2, max(1, len(chosen)), figsize=(5*max(1, len(chosen)), 7), squeeze=False)
    if not chosen:
        for ax in axes.flat:
            _unavailable(ax, "Physical-history objects unavailable; no schematic substituted")
    for column, history in enumerate(chosen):
        records = [r for r in history.copies if r["segment_id"] == history.segment_keys[0]]
        lookup = {row["copy_id"]: row for row in records}
        clone_ids = list(dict.fromkeys(row["clone_id"] for row in records))
        ax = axes[0, column]
        for row in records:
            y = clone_ids.index(row["clone_id"]) + .12*(row["homolog"]-1)
            parent = lookup.get(row.get("parent_copy_id"))
            if parent:
                py = clone_ids.index(parent["clone_id"]) + .12*(parent["homolog"]-1)
                ax.plot([parent["creation_index"], row["creation_index"]], [py, y], color="0.75", lw=.7)
            ax.scatter(row["creation_index"], y, color=("#1565a8", "#ca6b26")[row["homolog"]-1],
                       s=12+3*len(row["mutation_ids"]), alpha=.9 if row["active"] else .35)
        ax.set(yticks=range(len(clone_ids)), yticklabels=clone_ids, xlabel="Recorded copy-event order",
               title=f"{history.family}: physical copy ancestry")
        ax = axes[1, column]
        positive = history.phi[:, 0] > 0
        coordinates = np.unique(np.column_stack((history.phi[positive, 0], history.multiplicity[positive, 0])), axis=0)
        for phi, multiplicity in coordinates:
            count = int(np.sum((history.phi[:, 0] == phi) & (history.multiplicity[:, 0] == multiplicity)))
            ax.scatter(phi, multiplicity, s=35+6*np.sqrt(count), color="#1565a8")
            ax.annotate(str(count), (phi, multiplicity), xytext=(5, 5), textcoords="offset points")
        ax.set(xlabel="True carrier prevalence, first region", ylabel="Physical mutant copies per carrier",
               title="Observed biological classes (labels: mutation counts)", xlim=(-.03, 1.05))
    figure.suptitle("1 · Physical-copy mechanisms from recorded histories", y=1.02)
    _save(figure, output, "01_physical_copy_mechanisms", paths)


def _heatmaps(evaluations, output, paths):
    acquisitions = sorted({e["acquisition_id"] for e in evaluations})
    figure, axes = plt.subplots(1, 3, figsize=(14, 5), squeeze=False)
    for ax, (metric, title) in zip(axes.flat, (("stage2_truth_exclusion", "True multiplicity excluded"),
            ("cap_activity", "Cap below original major CN"), ("true_call_replacement_exclusion", "Exclusion after fixed-ID true-call replacement"))):
        values = np.full((len(FAMILIES), len(acquisitions)), np.nan)
        for i, family in enumerate(FAMILIES):
            for j, acquisition in enumerate(acquisitions):
                selected = [value for e, value in _values(evaluations, metric, "C", stratum="all_positive_carrier_pairs")
                            if e["family"] == family and e["acquisition_id"] == acquisition]
                if selected:
                    values[i, j] = np.mean(selected)
        if not np.isfinite(values).any():
            _unavailable(ax)
        else:
            cmap = plt.get_cmap("magma").copy()
            cmap.set_bad("#dddddd")
            image = ax.imshow(values, vmin=0, vmax=1, aspect="auto", cmap=cmap)
            ax.set(xticks=range(len(acquisitions)), xticklabels=acquisitions, yticks=range(len(FAMILIES)), yticklabels=FAMILIES)
            figure.colorbar(image, ax=ax, label="Pair fraction; gray = unavailable", shrink=.8)
        ax.set_title(title, fontsize=10)
    figure.suptitle("2 · Stage 2 cap exclusion and activity", y=1.02)
    _save(figure, output, "02_cap_exclusion_activity", paths)


def _paired(evaluations, aggregate_report, output, paths):
    settings = (("ari", "common_native_remainder", "B", "C", "Native partition ARI"),
                ("false_merge_fraction", "", "B", "C", "False clonal merges"),
                ("ccf_rmse", "constructed_reference", "B", "C", "Constructed-reference CCF RMSE"),
                ("correct_per_original_evaluable", "", "B", "B3", "Uncapped reassessment"),
                ("correct_per_original_evaluable", "", "C", "C3", "Capped reassessment"),
                ("correct_per_original_evaluable", "", "B3", "C3", "Downstream cap effect"))
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (name, view, before, after, title) in zip(axes.flat, settings):
        left = {(e["family"], e["history_id"], e["acquisition_id"]): value for e, value in _values(evaluations, name, before, view=view)}
        right = {(e["family"], e["history_id"], e["acquisition_id"]): value for e, value in _values(evaluations, name, after, view=view)}
        keys = sorted(left.keys() & right.keys())
        if keys:
            for key in keys:
                ax.plot([0, 1], [left[key], right[key]], color="#1565a8", alpha=.25, marker="o", ms=3)
            ax.set(xticks=[0, 1], xticklabels=[before, after])
        else:
            _unavailable(ax)
        ax.set_title(title)
        summaries = [row for row in aggregate_report.get("metrics", ()) if row["name"] == name
                     and row["arm"] == after+"_minus_"+before and row["view"] == view and row["stratum"] == "all"]
        if summaries and isinstance(summaries[0]["equal_family_mean"], (float, int)):
            row = summaries[0]
            interval = f" [{row['lower_95']:.3g}, {row['upper_95']:.3g}]" if row['lower_95'] is not None else ""
            ax.text(.02, .02, f"Equal-family Δ {row['equal_family_mean']:.3g}{interval}", transform=ax.transAxes, fontsize=8)
    figure.suptitle("3 · Paired cells: fitting, merging, and reassessment", y=1.01)
    _save(figure, output, "03_paired_effects", paths)


def _decisions(evaluations, aggregate_report, output, paths):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for arm, color in (("B3", "#1565a8"), ("C3", "#ca6b26")):
        coverage = {(e["family"], e["history_id"], e["acquisition_id"]): value for e, value in _values(evaluations, "acceptance_coverage", arm)}
        error = {(e["family"], e["history_id"], e["acquisition_id"]): value for e, value in _values(evaluations, "selective_error", arm)}
        keys = coverage.keys() & error.keys()
        if keys:
            axes[0].scatter([coverage[key] for key in keys], [error[key] for key in keys], label=arm, color=color, alpha=.6, s=18)
        for evaluation in evaluations:
            curve = [row for row in evaluation.get("error_coverage_curves", ()) if row["arm"] == arm
                     and row["acceptance_coverage"] is not None and row["selective_error"] is not None]
            if curve:
                axes[0].plot([row["acceptance_coverage"] for row in curve], [row["selective_error"] for row in curve],
                             color=color, alpha=.2, lw=.8)
        offset = -.18 if arm == "B3" else .18
        for index, name in enumerate(TRANSITIONS):
            values = [value for _, value in _values(evaluations, "transition_fraction", arm, stratum=name)]
            if values:
                axes[1].scatter(index+offset+np.linspace(-.08, .08, len(values)), values, color=color, alpha=.4,
                                s=10, label=arm if index == 0 else None)
            summaries = [row for row in aggregate_report.get("metrics", ()) if row["name"] == "transition_fraction"
                         and row["arm"] == arm and row["stratum"] == name]
            if summaries and isinstance(summaries[0]["equal_family_mean"], (float, int)):
                axes[1].scatter(index+offset, summaries[0]["equal_family_mean"], color=color, marker="D", s=45)
    if not axes[0].collections:
        _unavailable(axes[0])
    else:
        axes[0].legend()
    axes[0].set(xlabel="Accepted / evaluable targets", ylabel="Wrong / accepted targets",
                title="Fixed-reference threshold curves and executed decisions", xlim=(-.02, 1.02), ylim=(-.02, 1.02))
    axes[1].set(xticks=np.arange(6), xticklabels=[name.replace("_", "\n") for name in TRANSITIONS],
                ylabel="Fraction of evaluable targets", title="Combined-baseline transitions")
    if axes[1].collections:
        axes[1].legend()
    else:
        _unavailable(axes[1])
    figure.suptitle("4 · Decisions and abstention must be reported together", y=1.02)
    _save(figure, output, "04_decisions_transitions", paths)


def _weighted_reliability(evaluations, arm, stratum):
    histories, acquisitions = defaultdict(set), defaultdict(set)
    for e in evaluations:
        histories[e["family"]].add(e["history_id"])
        acquisitions[e["family"], e["history_id"]].add(e["acquisition_id"])
    bins = np.zeros((10, 3))
    for e in evaluations:
        selected = [r for r in e["calibration"] if r["arm"] == arm and (stratum == "all"
                    or r["true_class_present"] == (stratum == "true_class_present"))]
        if not selected:
            continue
        weight = 1 / (len(histories)*len(histories[e["family"]])
                      * len(acquisitions[e["family"], e["history_id"]]) * len(selected))
        for row in selected:
            q = row["clonal_probability"]
            bins[min(int(q*10), 9)] += weight*np.array([1., q, row["truth_clonal"]])
    return [(row[1]/row[0], row[2]/row[0]) for row in bins if row[0] > 0]


def _reliability(evaluations, output, paths):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, stratum in zip(axes, ("all", "true_class_present", "true_class_absent")):
        any_rows = False
        for arm, color in (("B3", "#1565a8"), ("C3", "#ca6b26")):
            points = _weighted_reliability(evaluations, arm, stratum)
            if points:
                any_rows = True
                ax.plot(*np.asarray(points).T, "o-", color=color, label=arm)
        if not any_rows:
            _unavailable(ax)
        else:
            ax.plot([0, 1], [0, 1], "--", color="0.5", lw=1)
            ax.legend()
        ax.set(xlabel="Designated-clonal probability", ylabel="Observed biological clonal fraction",
               title=stratum.replace("_", " "), xlim=(0, 1), ylim=(0, 1))
    figure.suptitle("5 · Reliability, including absent biological truth classes", y=1.02)
    _save(figure, output, "05_probability_reliability", paths)


def _sensitivities(evaluations, output, paths):
    figure, axes = plt.subplots(2, 3, figsize=(13, 8))
    definitions = (("mean_depth", "Mean observed depth"), ("region_count", "Requested regions"),
                   ("purity", "Mean supplied purity"), ("major_cn", "Mean original major CN"),
                   ("contributors", "Qualified contributors per audited pair"), ("input_errors", "Input-error panel"))
    for ax, (key, label) in zip(axes.flat, definitions):
        points = []
        for e, value in _values(evaluations, "correct_per_original_evaluable", "C3"):
            calibration = e["calibration"]
            if key == "region_count":
                x = e["region_count"]
            elif key in ("mean_depth", "major_cn"):
                x = np.mean([r[key] for r in calibration]) if calibration else None
            elif key == "purity":
                raw = e.get("acquisition", {}).get("purity")
                x = np.asarray(raw).mean() if raw is not None else None
            elif key == "contributors":
                x = np.mean([r["contributor_count"] for r in e["cap_audit"]]) if e["cap_audit"] else None
            else:
                x = e.get("input_error_fraction")
            if x is not None and np.isfinite(x):
                points.append((x, value))
        if points and len(set(x for x, _ in points)) > 1:
            ax.scatter(*np.asarray(points).T, s=20, alpha=.5, color="#1565a8")
        else:
            _unavailable(ax, "No executed variation in this panel")
        ax.set(xlabel=label, ylabel="C3 correct / original evaluable")
    figure.suptitle("6 · Sensitivity panels: only executed contrasts are shown", y=1.01)
    _save(figure, output, "06_sensitivities", paths)


def make_figures(evaluations, aggregate_report, output_dir, *, histories=()):
    """Produce all six required groups, with neutral panels when data are absent."""
    evaluations, histories = list(evaluations), list(histories)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    _physical(histories, output, paths)
    _heatmaps(evaluations, output, paths)
    _paired(evaluations, aggregate_report, output, paths)
    _decisions(evaluations, aggregate_report, output, paths)
    _reliability(evaluations, output, paths)
    _sensitivities(evaluations, output, paths)
    return paths
