#!/usr/bin/env python3
"""Protocol 2.2 CPU qualification, separate from biological calibration.

Independent end-to-end replicates repeat fitting, routing and reconciliation.
An optional offline experiment compares capped/uncapped fits on shared inputs.
Run against one installed wheel or editable package; no checkout import is injected.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from CliPP2.assignment import AssignmentBundle, assign_mutations
from CliPP2.config import AssignmentThresholds, DescriptiveScreen, PROTOCOL_VERSION
from CliPP2 import run_workflow
from CliPP2.io.assignment import load_accepted_fit, load_assignment_bundle
from CliPP2.io.data import MultiplicityBounds
from CliPP2.io.reporting import verify_selected
from CliPP2.kernel.model import RegionalModel
from CliPP2.tree.selection import fit_regions
from CliPP2.workflow import classify_groups
from CliPP2.simulation.evaluation import reference_center_diagnostics

from check_workflow import fixture_reference, run_checks, write_catalog, workflow_config


def partition_matching(truth, labels):
    """Overlap measures partition agreement, never center recovery."""
    identities, inverse = np.unique(truth, return_inverse=True)
    counts = np.zeros((len(identities), int(np.max(labels))+1), dtype=int)
    np.add.at(counts, (inverse, labels), 1)
    original, fitted = linear_sum_assignment(-counts)
    mapping = dict(zip(map(int, identities[original]), map(int, fitted)))
    matched = counts[original, fitted].sum()
    return mapping, 1-float(matched)/len(truth)


def reference_errors(truth_centers, reference_centers, atol=None):
    """Use the study's declared full-vector L2 matching, independent of overlap."""
    truth = np.asarray(truth_centers, float)
    reference = np.asarray(reference_centers, float).reshape(-1, truth.shape[1])
    evidence = reference_center_diagnostics(tuple(range(len(reference))), reference,
        dict(enumerate(truth)), match_atol=atol)
    matching = [dict(truth_label=row["truth_class"], reference_index=row["reference_label"],
        l2=row["l2_error"], max_coordinate_error=row["max_absolute_error"])
        for row in evidence["one_to_one_pairs"]]
    recovered = None if atol is None else [False]*len(truth)
    for row in evidence["qualified_pairs"] or ():
        recovered[row["truth_class"]] = True
    delta = truth[:, None, :] - reference[None, :, :]
    distance, maximum = np.linalg.norm(delta, axis=2), np.max(np.abs(delta), axis=2)
    return dict(alignment="one_to_one_minimum_regional_L2_not_member_overlap", matching=matching,
        true_group_count=len(truth), candidate_count=len(reference), reference_match_atol=atol,
        tolerance_metric=evidence["tolerance_metric"], matching_policy=evidence["matching_policy"],
        matching_ambiguous=evidence["matching_ambiguous"], matching_ambiguity=evidence["matching_ambiguity"],
        qualified_pairs=evidence["qualified_pairs"],
        nearest_l2=[float(row.min()) if len(row) else None for row in distance],
        nearest_max_coordinate_error=[float(row.min()) if len(row) else None for row in maximum],
        one_to_one_mean_l2=evidence["mean_one_to_one_l2_error"],
        recovery_status="unassessed_no_declared_tolerance" if atol is None else "assessed_with_declared_tolerance",
        recovered_by_truth=recovered, reference_recall=evidence["reference_recall"])


def fit_policy(catalog, support, truth_labels, truth_multiplicities, truth_centers, reference_match_atol=None):
    data = catalog.fit_data()
    model = RegionalModel(data, "cpu", cpu_threads=1, multiplicity_support=support)
    try:
        selected = fit_regions(model, 2, audit_scope="selected")
        verification = verify_selected(model, selected)
        fit = selected.selected.fit
        calls = model.multiplicity(fit.centers[fit.labels])
        _, error = partition_matching(truth_labels, fit.labels)
        excluded = ((truth_multiplicities < model.multiplicity_min)
                    | (truth_multiplicities > model.multiplicity_max))
        evidence = dict(model_identity=model.identity, support_hash=support.content_hash,
            mutation_count=model.n, occupied_clusters=len(fit.centers), score=float(fit.score),
            mixture_log_likelihood=float(fit.mixture_log_likelihood),
            conditional_log_likelihood=float(fit.conditional_log_likelihood),
            partition_membership_error_after_overlap_matching=error,
            reference_center_errors=reference_errors(truth_centers, fit.centers, reference_match_atol),
            multiplicity_error_fraction=float(np.mean(calls != truth_multiplicities)),
            true_multiplicity_excluded_fraction=float(np.mean(excluded)),
            independently_verified=bool(verification["independently_verified"]),
            centers=fit.centers.tolist(), labels=fit.labels.tolist())
        baseline = dict(centers=np.array(fit.centers), labels=np.array(fit.labels),
            phi=np.array(fit.centers[fit.labels]), calls=np.array(calls),
            lower=np.array(model.multiplicity_min), upper=np.array(model.multiplicity_max))
        diagnostic_fit = SimpleNamespace(source_id="synthetic_stage_two", mutation_ids=model.mutation_ids,
            phi=baseline["phi"], labels=fit.labels, centers=fit.centers,
            support=MultiplicityBounds(model.mutation_ids, model.region_ids,
                                       model.multiplicity_min, model.multiplicity_max, support.policy_id))
        quality_config = workflow_config()
        baseline["groups"] = classify_groups(catalog, diagnostic_fit, quality_config)
        evidence["reference_quality_screen"] = asdict(quality_config.group_screen)
        return evidence, baseline
    finally:
        model.backend.close()


def assignment_metrics(records, multiplicities, true_labels, true_calls, reference_labels):
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
    calls = np.asarray([row["conditional_call"] for row in multiplicities]).reshape(true_calls.shape)
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


def simulated_case(directory, seed, thresholds, reference_match_atol=None):
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
    full_metrics, _ = fit_policy(catalog, full, truth_labels, truth_calls, truth_centers, reference_match_atol)
    cap_metrics, baseline = fit_policy(catalog, cap, truth_labels, truth_calls, truth_centers, reference_match_atol)
    quality, reasons = [[True, True]], [((), ())]
    sources = [dict(kind="declared_all_one_control", stage_one_inference_performed=False)]
    for group in baseline["groups"]:
        coverage, native = group["regional_coverage_pass"], group["native_descriptive_pass"]
        quality.append([bool(c and n) for c, n in zip(coverage, native, strict=True)])
        reasons.append(tuple(tuple(reason for ok, reason in ((c, "source_coverage"),
            (n, "native_center_descriptive_screen")) if not ok) for c, n in zip(coverage, native, strict=True)))
        sources.append(dict(kind="fitted_capped_native_center", quality_support="accepted_capped_support",
                            support_hash=cap.content_hash, source_label=group["source_label"], diagnostics=group))
    reference = fixture_reference(tuple(range(len(baseline["centers"])+1)), catalog.region_ids,
        np.vstack([np.ones(2), baseline["centers"]]),
        lineage=(dict(source="synthetic_capped_fit", seed=seed),), regional_quality=np.asarray(quality, bool),
        regional_quality_reasons=tuple(reasons), center_sources=tuple(sources))
    bundle = AssignmentBundle(catalog, reference, catalog.mutation_ids, (), thresholds,
        baseline["phi"], baseline["calls"], baseline["lower"], baseline["upper"],
        ("simulated_stage_two",)*len(truth_labels), tuple(map(int, baseline["labels"])),
        tuple(int(label)+1 for label in baseline["labels"]), "cpu", 1,
        provenance=dict(qualification="capped_reference_full_support_assignment"))
    result = assign_mutations(bundle)
    errors = reference_errors(truth_centers, reference.centers, reference_match_atol)
    mapping = {row["truth_label"]: reference.labels[row["reference_index"]] for row in errors["matching"]}
    truth_reference_labels = np.asarray([mapping.get(int(group), -1) for group in truth_labels])
    metrics = assignment_metrics(result.assignments, result.multiplicities,
        truth_reference_labels, truth_calls, reference.labels)
    return dict(seed=seed, biological_inputs_shared=True, true_centers=truth_centers.tolist(),
        stage_two=dict(full=full_metrics, capped=cap_metrics), fixed_reference=reference.to_dict(), reference_center_errors=errors,
        stage_three_full_support=metrics,
        interpretation="stage_two_cap_effects_and_full_support_assignment_conditional_on_frozen_reference")


def nonidentifiability_case(directory, thresholds):
    # Both planted (phi,m) states imply the same VAF. Full support retains both.
    catalog = write_catalog(directory/"nonidentifiability.tsv", [[200], [200]], 400, 2)
    full = catalog.default_bounds()
    reference = fixture_reference((0, 1), catalog.region_ids, [[1.], [.5]])
    truth_labels, truth_calls = np.array([0, 1]), np.array([[1.], [2.]])
    bundle = AssignmentBundle(catalog, reference, catalog.mutation_ids, (), thresholds,
        [[1.], [.5]], truth_calls, full.lower, full.upper, ("planted_alias",)*2,
        (0, 1), (0, 1), "cpu", 1,
        provenance=dict(latent_vaf_equal=True, native_baseline="planted_truth_not_fitted"))
    result = assign_mutations(bundle)
    return dict(latent_probability=.5,
        latent_states=[dict(ccf=1., multiplicity=1), dict(ccf=.5, multiplicity=2)],
        decisions=[row["status"] for row in result.assignments],
        probabilities=[row["probabilities"].tolist() for row in result.assignments],
        metrics=assignment_metrics(result.assignments, result.multiplicities,
                                   truth_labels, truth_calls, reference.labels),
        interpretation="identical_reads_do_not_identify_latent_ccf_and_multiplicity")


def end_to_end_case(directory, seed, thresholds, reference_match_atol=None):
    rng = np.random.default_rng(seed)
    truth_labels = np.repeat([0, 1], 4)
    centers = np.array([[1., 1.], [.3, .4]])
    calls = np.ones((8, 2), dtype=int)
    depths = np.full((8, 2), 1000)
    alt = rng.binomial(depths, .5*centers[truth_labels])
    path = directory/f"end_to_end_{seed}.tsv"
    catalog = write_catalog(path, alt, depths, 1, minor=1)
    output = directory/f"end_to_end_{seed}"
    config = workflow_config(assignment=thresholds)
    receipt = run_workflow(path, output, config)
    whole = pd.read_csv(output/"all_original_mutations.tsv", sep="\t")
    assert tuple(whole.mutation_id) == catalog.mutation_ids
    bundle_path = output/"assignment_bundle.json"
    request = load_assignment_bundle(bundle_path) if bundle_path.exists() else None
    errors = reference_errors(centers, [] if request is None else request.reference.centers, reference_match_atol)
    alignment = ({} if request is None else {request.reference.labels[row["reference_index"]]: row["truth_label"]
                                           for row in errors["matching"]})
    current = whole["final_assignment_label"].to_numpy(float)
    accepted = np.isfinite(current)
    matched_truth = np.array([alignment.get(int(label), -1) if np.isfinite(label) else -1 for label in current])
    correct = accepted & (matched_truth == truth_labels)

    def outcomes(mask):
        mask = np.asarray(mask, bool)
        count = int(mask.sum())
        return dict(population_count=count, correct_accepted=int((mask & correct).sum()),
            wrong_accepted=int((mask & accepted & ~correct).sum()), unresolved=int((mask & ~accepted).sum()),
            membership_error=None if count == 0 else float((mask & ~correct).sum()/count))

    target = whole.role == "target"
    unchanged = whole.role == "non_target"
    origins = {origin: outcomes(whole.target_reason == origin)
               for origin in ("stage1_clonal_recheck", "stage2_near_clonal_recheck")}
    stage_two_path = output/"stage_two_accepted.json"
    second = load_accepted_fit(stage_two_path) if stage_two_path.exists() else None
    positions = {mutation: i for i, mutation in enumerate(catalog.mutation_ids)}
    second_rows = [] if second is None else [positions[m] for m in second.mutation_ids]
    routed = np.asarray(whole.target_reason == "stage2_near_clonal_recheck")
    excluded = (np.zeros((0, 2), bool) if second is None else
                (calls[second_rows] < second.support.lower) | (calls[second_rows] > second.support.upper))
    excluded_rows = excluded.any(axis=1)
    source_error = ([] if second is None else
                    np.linalg.norm(second.centers[second.labels]-centers[truth_labels[second_rows]], axis=1))
    group_path = output/"stage_two_groups.tsv"
    groups = pd.read_csv(group_path, sep="\t") if group_path.exists() else None
    label_to_row = {} if request is None else {label: i for i, label in enumerate(request.reference.labels)}
    final_error = [float(np.linalg.norm(request.reference.centers[label_to_row[int(label)]]-centers[truth_labels[i]]))
                   for i, label in enumerate(current) if np.isfinite(label)]
    return dict(seed=seed, status=receipt["status"], counts=receipt["counts"], workflow_config=asdict(config),
        generation_model=dict(reads="binomial", major_cn=1, minor_cn=1, normal_cn=2, purity=1.,
            depth=1000, true_centers=centers.tolist(), true_labels=truth_labels.tolist()),
        membership_alignment="one-to-one center alignment; membership agreement is not binary center recovery",
        targets=outcomes(target), target_origins=origins, unchanged_F=outcomes(unchanged),
        whole_population=outcomes(np.ones(len(whole), bool)), reference_availability_and_error=errors,
        accepted_final_center_mean_l2=float(np.mean(final_error)) if final_error else None,
        routing=dict(stage_two_population=len(second_rows), stage_two_routed=int(routed.sum()),
            geometric_group_count=None if groups is None else int(groups["near_clonal"].sum()),
            nonclonal_truth_stage_two=int(np.sum(truth_labels[second_rows] != 0)),
            nonclonal_truth_stage_two_routed=int(np.sum(routed & (truth_labels != 0))),
            true_multiplicity_excluded_mutations=int(excluded_rows.sum()),
            excluded_truth_routed=int(np.sum(excluded_rows & routed[second_rows])),
            excluded_truth_routing_fraction=(float(np.mean(routed[np.asarray(second_rows)[excluded_rows]]))
                                             if excluded_rows.any() else None),
            native_center_l2_by_mutation=list(map(float, source_error))),
        final_accounting=receipt.get("final_accounting"), true_multiplicities=calls.tolist(),
        interpretation="actual_run_workflow_including_discovery_routing_and_authoritative_final_membership")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--reference-match-atol", type=float, default=None,
                        help="Optional maximum per-coordinate error for one-to-one reference recovery")
    parser.add_argument("--compare-stage-two-support", action="store_true",
                        help="Add an offline full/capped-support comparison; never changes the production workflow")
    parser.add_argument("--posterior-min", type=float, required=True)
    parser.add_argument("--mean-deviance-max", type=float, required=True)
    parser.add_argument("--pair-deviance-max", type=float, required=True)
    parser.add_argument("--outlier-fraction-max", type=float, required=True)
    parser.add_argument("--tie-atol", type=float, required=True)
    parser.add_argument("--multiplicity-tie-atol", type=float, required=True)
    parser.add_argument("--multiplicity-probability-min", type=float, required=True)
    args = parser.parse_args(argv)
    if args.replicates < 1:
        parser.error("replicates must be positive")
    if args.reference_match_atol is not None and (not np.isfinite(args.reference_match_atol) or args.reference_match_atol < 0):
        parser.error("reference-match-atol must be finite and nonnegative")
    thresholds = AssignmentThresholds(args.posterior_min,
        DescriptiveScreen(args.pair_deviance_max, args.mean_deviance_max, args.outlier_fraction_max),
        args.tie_atol, args.multiplicity_tie_atol, args.multiplicity_probability_min)
    if args.output.exists():
        raise FileExistsError("Qualification output already exists; use a new output path")
    with tempfile.TemporaryDirectory(prefix="clipp2-qualification-") as temporary:
        directory = Path(temporary)
        report = dict(schema="clipp2.synthetic-qualification.v5", protocol_version=PROTOCOL_VERSION, seed=args.seed,
            device="cpu", thresholds=asdict(thresholds),
            reference_match_atol=args.reference_match_atol,
            offline_support_comparison_requested=args.compare_stage_two_support,
            calibration_approved=False,
            limitations=["Small synthetic experiment, not an empirical biological calibration study",
                "Diploid end-to-end replicates test orchestration and can have no Stage 2-origin targets or cap-excluded truth",
                "The cap comparison isolates Stage 2; separate end-to-end replicates derive caps through the workflow",
                "When requested, offline fixed-reference target counts also contributed to the capped fit",
                "No CUDA execution; no guarantee across real tumor distributions"],
            workflow_contracts=run_checks(directory / "contracts", args.device),
            cases=([simulated_case(directory, args.seed+i, thresholds, args.reference_match_atol)
                    for i in range(args.replicates)] if args.compare_stage_two_support else []),
            end_to_end=[end_to_end_case(directory, args.seed+i, thresholds, args.reference_match_atol)
                        for i in range(args.replicates)],
            nonidentifiability=nonidentifiability_case(directory, thresholds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(f"Wrote {len(report['end_to_end'])} actual-workflow replicates and {len(report['cases'])} offline support comparisons to {args.output}")
    print("Calibration remains unapproved; inspect the reported errors, exclusions and unresolved cases.")


if __name__ == "__main__":
    main()
