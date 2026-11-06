#!/usr/bin/env python3
"""Small deterministic CPU support-sensitivity experiment, not calibration approval.

Both fits use the same frozen simulated remainder. Paired assignments share one
reference built from the capped fit; their imposed floor is a deliberate stress
policy, not a claim that a complete workflow inferred that floor from donors.
The separate nonidentifiability case exposes latent CCF/multiplicity aliases.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from CliPP2._source import file_sha256, git_source_identity, source_fingerprint  # noqa: E402
from CliPP2.assignment import AssignmentBundle, AssignmentReference, assign_mutations  # noqa: E402
from CliPP2.config import AssignmentThresholds  # noqa: E402
from CliPP2.io.data import MultiplicityBounds  # noqa: E402
from CliPP2.io.reporting import verify_selected  # noqa: E402
from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt  # noqa: E402
from CliPP2.kernel.model import RegionalModel  # noqa: E402
from CliPP2.tree.selection import fit_regions  # noqa: E402


def write_catalog(path, alt, depth, major, *, purity=None):
    alt, depth, major = np.broadcast_arrays(np.atleast_2d(alt), np.atleast_2d(depth), np.atleast_2d(major))
    purity = np.ones(alt.shape[1]) if purity is None else np.asarray(purity)
    records = [dict(mutation_id=f"m{i:03}", sample_id=f"r{r+1}",
        alt_count=int(alt[i, r]), ref_count=int(depth[i, r]-alt[i, r]), count_observed=1,
        purity=float(purity[r]), normal_cn=2, segment_id="shared_segment",
        cn_state_id="0", cn_state_fraction=1., allele_a_cn=int(major[i, r]), allele_b_cn=0)
        for i in range(alt.shape[0]) for r in range(alt.shape[1])]
    write_tumor_txt(path, pd.DataFrame(records))
    return load_original_catalog(path)


def matched_groups(truth, labels):
    groups, clusters = int(np.max(truth))+1, int(np.max(labels))+1
    counts = np.zeros((groups, clusters), dtype=int)
    np.add.at(counts, (truth, labels), 1)
    original, fitted = linear_sum_assignment(-counts)
    mapping = dict(zip(map(int, original), map(int, fitted)))
    matched = sum(counts[a, b] for a, b in mapping.items())
    return mapping, 1-float(matched)/len(truth)


def fit_policy(catalog, support, truth_labels, truth_multiplicities):
    data = catalog.fit_data()
    model = RegionalModel(data, "cpu", cpu_threads=1, multiplicity_support=support)
    try:
        selected = fit_regions(model, 2, audit_scope="selected")
        verification = verify_selected(model, selected)
        fit = selected.selected.fit
        calls = model.multiplicity(fit.centers[fit.labels])
        mapping, error = matched_groups(truth_labels, fit.labels)
        excluded = ((truth_multiplicities < model.multiplicity_min)
                    | (truth_multiplicities > model.multiplicity_max))
        evidence = dict(model_identity=model.identity, support_hash=support.content_hash,
            mutation_count=model.n, occupied_clusters=len(fit.centers), score=float(fit.score),
            mixture_log_likelihood=float(fit.mixture_log_likelihood),
            conditional_log_likelihood=float(fit.conditional_log_likelihood),
            membership_error_after_label_matching=error,
            multiplicity_error_fraction=float(np.mean(calls != truth_multiplicities)),
            true_multiplicity_excluded_fraction=float(np.mean(excluded)),
            independently_verified=bool(verification["independently_verified"]),
            centers=fit.centers.tolist(), labels=fit.labels.tolist())
        baseline = dict(centers=np.array(fit.centers), labels=np.array(fit.labels),
            phi=np.array(fit.centers[fit.labels]), calls=np.array(calls),
            lower=np.array(model.multiplicity_min), upper=np.array(model.multiplicity_max), mapping=mapping)
        return evidence, baseline
    finally:
        model.backend.close()


def assignment_metrics(records, multiplicities, true_labels, true_calls, reference_labels, policy):
    true_labels = np.asarray(true_labels)
    reference = list(reference_labels)
    probabilities = np.asarray([row["probabilities"] for row in records])
    best = np.asarray([row["best_label"] if row["best_label"] is not None else -1 for row in records])
    accepted = np.asarray([row["accepted_label"] if row["accepted_label"] is not None else -1 for row in records])
    finite = np.isfinite(probabilities).all(axis=1)
    y = np.zeros_like(probabilities)
    represented = np.isin(true_labels, reference)
    for i, label in enumerate(true_labels):
        if label in reference:
            y[i, reference.index(label)] = 1
    # A missing true group is an additional outcome with predicted probability
    # zero, not an excuse to drop that target or award a correct classification.
    brier = np.sum((probabilities-y)**2, axis=1) + (~represented)
    correct = (best == true_labels) & represented
    confidence = np.asarray([row["posterior"] for row in records])
    reliability = []
    ece = 0.
    for lo, hi in zip(np.linspace(0, .8, 5), np.linspace(.2, 1., 5), strict=True):
        mask = finite & (confidence >= lo) & ((confidence <= hi) if hi == 1 else (confidence < hi))
        count = int(mask.sum())
        mean_p = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        if count:
            ece += count/max(1, int(finite.sum())) * abs(mean_p-accuracy)
        reliability.append(dict(lower=float(lo), upper=float(hi), count=count,
                                mean_confidence=mean_p, empirical_accuracy=accuracy))
    calls = np.asarray([row[policy+"_conditional_call"] for row in multiplicities]).reshape(true_calls.shape)
    available = np.isfinite(calls)
    return dict(target_count=len(records), unresolved_fraction=float(np.mean(accepted < 0)),
        best_label_error_fraction=float(np.mean(~correct)),
        accepted_label_error_fraction=(float(np.mean(accepted[accepted >= 0] != true_labels[accepted >= 0]))
                                      if np.any(accepted >= 0) else None),
        unrepresented_truth_group_fraction=float(np.mean(~represented)),
        multiplicity_error_on_available_fraction=float(np.mean(calls[available] != true_calls[available]))
            if available.any() else None,
        missing_conditional_multiplicity_fraction=float(np.mean(~available)),
        multiclass_brier_mean=float(brier[finite].mean()) if finite.any() else None,
        expected_calibration_error=float(ece) if finite.any() else None,
        reliability_bins=reliability,
        status_counts={status: sum(row["status"] == status for row in records)
                       for status in sorted({row["status"] for row in records})})


def simulated_case(directory, seed, thresholds):
    rng = np.random.default_rng(seed)
    truth_labels = np.repeat(np.arange(2), 6)
    truth_centers = np.array([[.32, .45], [.72, .80]])
    truth_calls = np.tile([[1, 2], [2, 1], [3, 3]], (4, 1))
    purity = np.array([.8, .7])
    major = np.full((12, 2), 3)
    scaling = purity/((1-purity)*2 + purity*major)
    depths = rng.integers(200, 401, size=(12, 2))
    alt = rng.binomial(depths, scaling*truth_calls*truth_centers[truth_labels])
    catalog = write_catalog(directory/f"simulation_{seed}.tsv", alt, depths, major, purity=purity)
    full = catalog.default_bounds()
    cap = MultiplicityBounds(catalog.mutation_ids, catalog.region_ids, full.lower,
                             np.full_like(full.upper, 2), "synthetic_stage_two_cap_2")
    full_metrics, _ = fit_policy(catalog, full, truth_labels, truth_calls)
    cap_metrics, baseline = fit_policy(catalog, cap, truth_labels, truth_calls)
    reference = AssignmentReference(tuple(range(len(baseline["centers"])+1)), catalog.region_ids,
        np.vstack([np.ones(2), baseline["centers"]]),
        lineage=(dict(source="synthetic_capped_fit", seed=seed),))
    floor = MultiplicityBounds(catalog.mutation_ids, catalog.region_ids,
        np.full_like(full.lower, 2), full.upper, "synthetic_stage_three_floor_stress_2")
    bundle = AssignmentBundle(catalog, reference, catalog.mutation_ids, (), floor, thresholds,
        baseline["phi"], baseline["calls"], baseline["lower"], baseline["upper"],
        ("simulated_stage_two",)*len(truth_labels), tuple(map(int, baseline["labels"])),
        "cpu", 1, dict(qualification="synthetic_stress_policy", floor_inferred_from_donors=False))
    result = assign_mutations(bundle)
    truth_reference_labels = np.asarray([baseline["mapping"].get(int(group), -2)+1 for group in truth_labels])
    paired = {policy: assignment_metrics(getattr(result, policy), result.multiplicities,
        truth_reference_labels, truth_calls, reference.labels, policy)
        for policy in ("restricted", "original")}
    return dict(seed=seed, biological_inputs_shared=True, true_centers=truth_centers.tolist(),
        stage_two=dict(full=full_metrics, capped=cap_metrics),
        fixed_reference=reference.to_dict(), paired_assignment=paired,
        restriction_excludes_native_call_fraction=float(np.mean([
            row["native_call_excluded"] for row in result.multiplicities])),
        restriction_excludes_true_call_fraction=float(np.mean(truth_calls < 2)),
        support_sensitive_decision_fraction=float(np.mean([
            row["support_sensitive_decision"] for row in result.comparison])),
        interpretation="same_remainder_fits_and_fixed_reference_stress_assignment_not_a_donor_calibration")


def nonidentifiability_case(directory, thresholds):
    # phi=1,m=1 and phi=.5,m=2 both imply VAF=.5 at purity1,totalCN2.
    # High depth makes the unsupported alternative component negligible under
    # full-support marginalization; restriction can produce false certainty.
    catalog = write_catalog(directory/"nonidentifiability.tsv", [[200], [200]], 400, 2)
    full = catalog.default_bounds()
    floor = MultiplicityBounds(catalog.mutation_ids, catalog.region_ids,
                               np.full((2, 1), 2), full.upper, "nonidentifiability_floor_2")
    reference = AssignmentReference((0, 1), catalog.region_ids, [[1.], [.5]])
    truth_labels, truth_calls = np.array([0, 1]), np.array([[1.], [2.]])
    bundle = AssignmentBundle(catalog, reference, catalog.mutation_ids, (), floor, thresholds,
        [[1.], [.5]], truth_calls, full.lower, full.upper, ("planted_alias",)*2,
        (0, 1), "cpu", 1, dict(latent_vaf_equal=True, native_baseline="planted_truth_not_fitted"))
    result = assign_mutations(bundle)
    return dict(latent_probability=.5, latent_states=[dict(ccf=1., multiplicity=1), dict(ccf=.5, multiplicity=2)],
        original_decisions=[row["status"] for row in result.original],
        restricted_decisions=[row["status"] for row in result.restricted],
        original_probabilities=[row["probabilities"].tolist() for row in result.original],
        restricted_probabilities=[row["probabilities"].tolist() for row in result.restricted],
        metrics={policy: assignment_metrics(getattr(result, policy), result.multiplicities,
                 truth_labels, truth_calls, reference.labels, policy) for policy in ("restricted", "original")},
        interpretation="identical_reads_do_not_identify_latent_ccf_and_multiplicity;restrictions_can_exclude_truth")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--posterior-min", type=float, required=True)
    parser.add_argument("--mean-deviance-max", type=float, required=True)
    parser.add_argument("--tie-atol", type=float, required=True)
    args = parser.parse_args(argv)
    if args.replicates < 1:
        parser.error("replicates must be positive")
    thresholds = AssignmentThresholds(args.posterior_min, args.mean_deviance_max, args.tie_atol)
    if args.output.exists():
        raise FileExistsError("Qualification output already exists; use a new output path")
    repository = Path(__file__).resolve().parents[1]
    commit, dirty = git_source_identity(repository)
    with tempfile.TemporaryDirectory(prefix="clipp2-qualification-") as temporary:
        directory = Path(temporary)
        report = dict(schema="clipp2.synthetic-qualification.v1", seed=args.seed,
            source_commit=commit, source_dirty=dirty,
            source_sha256=source_fingerprint(repository / "src/CliPP2"),
            qualification_script_sha256=file_sha256(__file__),
            device="cpu", thresholds=asdict(thresholds),
            calibration_approved=False,
            limitations=["Small synthetic experiment, not an empirical biological calibration study",
                "Imposed stress bounds are not inferred from simulated donor qualification",
                "Fixed-reference target read counts also contributed to the capped fit",
                "No CUDA execution; no guarantee across real tumor distributions"],
            cases=[simulated_case(directory, args.seed+i, thresholds) for i in range(args.replicates)],
            nonidentifiability=nonidentifiability_case(directory, thresholds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(f"Wrote {len(report['cases'])} synthetic comparisons to {args.output}")
    print("Calibration remains unapproved; inspect the reported errors, exclusions and unresolved cases.")


if __name__ == "__main__":
    main()
