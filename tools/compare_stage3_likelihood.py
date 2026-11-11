#!/usr/bin/env python3
"""Compare hash-bound frozen bundles: zero fits, two assignment passes per case.

Manifest: {"cases": [{"id", "bundle", "bundle_sha256", "truth", "truth_sha256"}]}.
Truth is a keyed TSV: mutation_id, region_id, truth_cluster, true_ccf,
true_multiplicity (NA for ambiguous truth), cna (0/1), observed (0/1).
Paths are relative to the manifest. Truth preparation/eligibility is external;
the manifest must bind the original truth and conversion evidence separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp, xlogy, xlog1py

from CliPP2.assignment import (assign_mutations, conditional_multiplicity,
    derive_assignment_bundle, profile_score_summary)
from CliPP2.io.assignment import load_assignment_bundle, _publish_json, write_supporting_table
from CliPP2.simulation.evaluation import candidate_correspondence, partition_scores


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def macro_f1(truth, prediction, classes):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    return float(np.mean([2*np.sum((truth == k) & (prediction == k)) / max(1, np.sum(truth == k)+np.sum(prediction == k))
                          for k in classes])) if classes else None


def _attribute(request, marginal, profile):
    """Independent finite-component identity; impossible candidates remain -inf."""
    lookup = {m: i for i, m in enumerate(request.catalog.mutation_ids)}
    c = request.catalog
    volume = np.zeros((len(request.target_ids), len(request.reference.labels)))
    for i, mutation in enumerate(request.target_ids):
        source = lookup[mutation]
        for r in np.flatnonzero(c.eligible[source] & c.count_observed[source] & (c.total_counts[source] > 0)):
            width = int(c.major_cn[source, r])
            a = c.purity[source, r] / ((1-c.purity[source, r])*c.normal_cn[source, r]
                + c.purity[source, r]*(c.major_cn[source, r]+c.minor_cn[source, r]))
            p = np.minimum(a*np.arange(1, width+1)[None, :]*request.reference.centers[:, r, None], 1.)
            y, n = c.alt_counts[source, r], c.total_counts[source, r]
            logs = xlogy(y, p)+xlog1py(n-y, -p)
            maximum = np.max(logs, axis=1)
            finite = np.isfinite(maximum)
            volume[i, finite] += logsumexp(logs[finite]-maximum[finite, None], axis=1)-np.log(width)
            volume[i, ~finite] = np.nan
    actual = np.array([r["log_scores"] for r in marginal.assignments])
    profiled = np.array([r["log_scores"] for r in profile.assignments])
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(profiled))
    finite = np.isfinite(actual)
    np.testing.assert_allclose(actual[finite], (profiled+volume)[finite], rtol=1e-10, atol=1e-10)
    return volume, float(np.max(abs(actual[finite]-(profiled+volume)[finite]), initial=0.))


def compare(request, truth, outdir):
    requests = {method: derive_assignment_bundle(request, assignment_method=method) for method in ("marginal", "profile")}
    payloads = [r.to_dict() for r in requests.values()]
    for payload in payloads:
        payload.pop("assignment_method")
        payload.pop("provenance")
    assert payloads[0] == payloads[1], "Methods must share every frozen scientific input"
    results, elapsed = {}, {}
    for method, frozen in requests.items():
        started = time.perf_counter()
        results[method] = assign_mutations(frozen, outdir=outdir / method)
        elapsed[method] = time.perf_counter()-started
    request = requests["marginal"]
    volume, error = _attribute(request, results["marginal"], results["profile"])
    shape = len(request.target_ids), len(request.reference.region_ids)
    keys = pd.MultiIndex.from_product([request.target_ids, request.reference.region_ids], names=["mutation_id", "region_id"])
    if truth.index.has_duplicates or not keys.isin(truth.index).all():
        raise ValueError("Truth must uniquely cover every requested mutation-region pair")
    truth = truth.loc[keys]
    if not truth["observed"].isin([0, 1]).all() or not truth["cna"].isin([0, 1]).all():
        raise ValueError("Truth observation and CNA flags must be explicit 0/1")
    true_phi = truth.true_ccf.to_numpy(float).reshape(shape)
    true_class = truth.truth_cluster.astype(str).to_numpy().reshape(shape)
    if not np.all(true_class == true_class[:, :1]) or not np.isfinite(true_phi).all():
        raise ValueError("Truth clusters must be shared and CCFs finite")
    class_by_id = dict(zip(request.target_ids, true_class[:, 0]))
    clonal = {class_by_id[m] for i, m in enumerate(request.target_ids) if np.all(true_phi[i] == 1)}
    if len(clonal) > 1:
        raise ValueError("Truth must consolidate indistinguishable clonal classes")
    members = {label: [m for m, b in zip(request.target_ids, request.baseline_labels) if b == label]
               for label in request.reference.labels}
    mapping, correspondence = candidate_correspondence(request.reference.labels, request.reference.clonal_label,
        members, class_by_id, next(iter(clonal), None))
    labels, phis, calls = {}, {}, {}
    for method, result in results.items():
        labels[method] = np.array([r["final_assignment_label"] if r["final_assignment_label"] is not None else np.nan for r in result.assignments])
        phis[method] = np.array([r["phi"] for r in result.multiplicities]).reshape(shape)
        calls[method] = np.array([r["assigned_multiplicity"] for r in result.multiplicities]).reshape(shape)
    # Counterfactual tie policy needs no additional native scoring.
    labels["profile_lowest"] = labels["profile"].copy()
    phis["profile_lowest"], calls["profile_lowest"] = phis["profile"].copy(), calls["profile"].copy()
    catalog = request.catalog
    indices = [catalog.mutation_ids.index(m) for m in request.target_ids]
    observed = catalog.eligible[indices] & catalog.count_observed[indices] & (catalog.total_counts[indices] > 0)
    for i, row in enumerate(results["profile"].assignments):
        if row["final_assignment_label"] is None:
            continue
        choice = profile_score_summary(row["log_scores"], request.reference.labels, None)["best_label"]
        labels["profile_lowest"][i] = choice
        if choice == row["final_assignment_label"]:
            continue
        center = request.reference.centers[request.reference.labels.index(choice)]
        for r in np.flatnonzero(observed[i]):
            d = conditional_multiplicity(catalog, indices[i], r, center[r], 1, int(catalog.major_cn[indices[i], r]))
            phis["profile_lowest"][i, r] = center[r]
            calls["profile_lowest"][i, r] = d["map_call"] if center[r] > 0 else np.nan
    biological = observed & truth.observed.to_numpy(bool).reshape(shape)
    tm = truth.true_multiplicity.to_numpy(float).reshape(shape)
    cna = biological & truth.cna.to_numpy(bool).reshape(shape) & np.isfinite(tm) & (tm >= 1) & (tm == np.rint(tm))
    classes = sorted(set(tm[cna].astype(int)) | {int(v) for values in calls.values() for v in values[cna] if np.isfinite(v)})
    metrics, rows, predictions = {}, [], {}
    for method in labels:
        selected = labels[method]
        valid = np.isfinite(selected)
        paired = biological & np.isfinite(phis[method])
        predictions[method] = np.where(np.isfinite(calls[method][cna]), calls[method][cna], -1).astype(int).tolist()
        y, pred = tm[cna].astype(int), predictions[method]
        metrics[method] = dict(ARI=partition_scores(true_class[:, 0][valid], selected[valid])["ari"],
            assigned=int(valid.sum()), missing_assignments=int((~valid).sum()), CNA_pairs=len(y),
            CNA_macro_F1=macro_f1(y, pred, classes) if len(y) else None,
            CNA_accuracy=float(np.mean(y == pred)) if len(y) else None,
            missing_multiplicities=sum(v < 0 for v in pred), CCF_pairs=int(paired.sum()),
            CCF_error_sum=float(abs(phis[method][paired]-true_phi[paired]).sum()),
            CCF_MAE=float(abs(phis[method][paired]-true_phi[paired]).mean()) if paired.any() else None)
    for i, mutation in enumerate(request.target_ids):
        source = indices[i]
        scores = results["profile"].assignments[i]
        truth_id = class_by_id[mutation]
        row = dict(mutation_id=mutation, source_stage=request.source_stages[i], truth_class=truth_id,
            max_major_cn=int(catalog.major_cn[source][observed[i]].max()) if observed[i].any() else None,
            mean_depth=float(catalog.total_counts[source][observed[i]].mean()) if observed[i].any() else None,
            profile_tied_labels=scores["map_set"], profile_unique=scores["map_unique"],
            truth_in_profile_set=any(mapping.get(label) == truth_id for label in scores["map_set"]),
            alias_configuration="exact_profile_tie" if len(scores["map_set"]) > 1 else "unique_profile_maximum" if scores["map_set"] else "no_finite_profile",
            marginal_profile_volume=volume[i].tolist())
        for method, values in labels.items():
            label = int(values[i]) if np.isfinite(values[i]) else None
            row[method+"_label"] = label
            row[method+"_correct"] = None if label is None or mapping.get(label) is None else mapping[label] == truth_id
            row[method+"_ccf_mae"] = float(abs(phis[method][i, biological[i]]-true_phi[i, biological[i]]).mean()) if biological[i].any() and label is not None else None
        row["depth_band"] = "unobserved" if row["mean_depth"] is None else "<100" if row["mean_depth"] < 100 else "100-299" if row["mean_depth"] < 300 else ">=300"
        rows.append(row)
    for method in metrics:
        known = [r for r in rows if r[method+"_correct"] is not None]
        unique = [r for r in known if r["profile_unique"]]
        metrics[method].update(correspondence_coverage=len(known), representative_accuracy=np.mean([r[method+"_correct"] for r in known]) if known else None,
            profile_unique_winner_accuracy=np.mean([r[method+"_correct"] for r in unique]) if unique else None)
    transitions = Counter("unmapped_or_unresolved" if r["marginal_correct"] is None or r["profile_correct"] is None
        else ("correct" if r["marginal_correct"] else "wrong")+"_to_"+("correct" if r["profile_correct"] else "wrong") for r in rows)
    strata = {}
    for field in ("source_stage", "max_major_cn", "depth_band", "alias_configuration"):
        strata[field] = {str(value): dict(count=len(group), changed=sum(r["marginal_label"] != r["profile_label"] for r in group),
            wrong_to_correct=sum(r["marginal_correct"] is False and r["profile_correct"] is True for r in group),
            correct_to_wrong=sum(r["marginal_correct"] is True and r["profile_correct"] is False for r in group))
            for value in sorted({r[field] for r in rows}, key=str) for group in [[r for r in rows if r[field] == value]]}
    write_supporting_table(outdir / "mutations.tsv", rows)
    return dict(metrics=metrics, assignment_seconds=elapsed, transitions=dict(transitions), strata=strata,
        bundle_hashes={m: r.content_hash for m, r in requests.items()}, native_fit_count=0, assignment_pass_count=2,
        attribution_max_error=error, candidate_correspondence=correspondence,
        exact_profile_ties=sum(len(r["profile_tied_labels"]) > 1 for r in rows),
        scoring_changes=sum(r["marginal_label"] != r["profile_lowest_label"] for r in rows),
        tie_policy_changes=sum(r["profile_lowest_label"] != r["profile_label"] for r in rows),
        truth_multiplicity=tm[cna].astype(int).tolist(), predictions=predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest_hash = sha(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    if sha(args.manifest) != manifest_hash:
        raise ValueError("Manifest changed while loading")
    ids = [c["id"] for c in manifest["cases"]]
    if len(set(ids)) != len(ids) or any(Path(v).name != v or v in ("", ".", "..") for v in ids):
        raise ValueError("Case identities must be distinct safe directory names")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reports = []
    for case in manifest["cases"]:
        paths = {key: args.manifest.parent / case[key] for key in ("bundle", "truth")}
        for key, path in paths.items():
            if sha(path) != case[key+"_sha256"]:
                raise ValueError("Bound input changed: " + str(path))
        request = load_assignment_bundle(paths["bundle"])
        truth = pd.read_csv(paths["truth"], sep="\t", float_precision="round_trip").set_index(["mutation_id", "region_id"])
        outdir = args.output_dir / case["id"]
        outdir.mkdir()
        report = compare(request, truth, outdir)
        for key, path in paths.items():
            if sha(path) != case[key+"_sha256"]:
                raise ValueError("Bound input changed during execution: " + str(path))
        _publish_json(outdir / "comparison.json", report)
        reports.append(dict(id=case["id"], **report))
        print(json.dumps(dict(completed=len(reports), total=len(ids), id=case["id"])), flush=True)
    if sha(args.manifest) != manifest_hash:
        raise ValueError("Manifest changed during execution")
    _publish_json(args.output_dir / "summary.json", dict(manifest_sha256=manifest_hash, cases=reports,
        native_fit_count=0, assignment_pass_count=2*len(reports)))


if __name__ == "__main__":
    main()
