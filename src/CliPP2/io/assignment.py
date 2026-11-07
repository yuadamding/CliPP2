"""Versioned frozen requests, accepted-fit receipts, and no-clobber outputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp, xlogy, xlog1py

from ..assignment import (AssignmentBundle, AssignmentResult, POLICY_FIELDS,
                          REGIONAL_BASE_FIELDS, REGIONAL_POLICY_FIELDS, REASSESSMENT_OUTCOMES)
from .data import json_value as _json_value, content_hash
from .reporting import MANIFEST_SCHEMA
from .publication import publish_json
from ..config import PROTOCOL_VERSION
from ..screening import summarize_profile_deviance


def reserve_output(outdir):
    path = Path(outdir)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError("Output directory must be new or empty")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _publish_json(path, value):
    return publish_json(path, json.dumps(_json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_assignment_bundle(bundle, path):
    if type(bundle) is not AssignmentBundle:
        raise TypeError("A validated AssignmentBundle is required")
    payload = bundle.validated_payload()
    return _publish_json(path, dict(payload=payload, content_hash=bundle.content_hash))


def load_assignment_bundle(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(value) != {"payload", "content_hash"} or content_hash(value["payload"]) != value["content_hash"]:
        raise ValueError("Frozen assignment bundle hash mismatch")
    bundle = AssignmentBundle.from_dict(value["payload"])
    if bundle.content_hash != value["content_hash"]:
        raise ValueError("Frozen assignment bundle did not round-trip its content")
    return bundle


def save_original_catalog(catalog, path):
    return _publish_json(path, catalog.to_dict())


def save_accepted_fit(fit, path):
    from ..workflow import AcceptedFit
    if type(fit) is not AcceptedFit:
        raise TypeError("An AcceptedFit is required")
    payload = fit.to_dict()
    return _publish_json(path, dict(payload=payload, content_hash=content_hash(payload)))


def load_accepted_fit(path):
    from ..workflow import AcceptedFit
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(value) != {"payload", "content_hash"} or content_hash(value["payload"]) != value["content_hash"]:
        raise ValueError("An intact full accepted-fit receipt is required; native TSVs alone are insufficient")
    return AcceptedFit.from_dict(value["payload"])


def write_supporting_table(path, records, *, columns=None):
    path = Path(path)
    records = list(records)
    if not records and columns is None:
        raise ValueError("Empty supporting tables require an explicit column schema")
    frame = pd.DataFrame([{key: json.dumps(_json_value(value), separators=(",", ":"), sort_keys=True)
                           if isinstance(value, (tuple, list, Mapping, np.ndarray)) else value
                           for key, value in row.items()} for row in records], columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clipp2-table-", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        frame.to_csv(staged, sep="\t", index=False, float_format="%.17g", na_rep="NA")
        parsed = pd.read_csv(staged, sep="\t", dtype=str, keep_default_na=False)
        if len(parsed) != len(frame) or list(parsed.columns) != list(frame.columns):
            raise ValueError("Supporting TSV did not round-trip its schema/cardinality")
        os.link(staged, path)
    return path


def _matches(actual, expected, *, atol=1e-12):
    if expected is None:
        return actual is None
    if isinstance(expected, Mapping):
        return _json_value(actual) == _json_value(expected)
    if isinstance(expected, (str, bool, np.bool_)):
        return actual == expected
    try:
        left, right = np.asarray(actual, float), np.asarray(expected, float)
        return left.shape == right.shape and np.allclose(left, right, rtol=1e-10, atol=atol, equal_nan=True)
    except (TypeError, ValueError):
        return actual == expected


def _require_fields(record, expected, context, *, atol=1e-12):
    for name, value in expected.items():
        if name not in record or not _matches(record[name], value, atol=atol):
            raise ValueError(f"{context}: inconsistent {name}")


def _independent_deviance(alt, depth, probability):
    """Verify deviance using log1p remainders, independently of screen code.

    For d=p-q, KL=q*(d/q-log1p(d/q)) +
    (1-q)*(-d/(1-q)-log1p(-d/(1-q))). The cancelled linear terms
    never enter the floating calculation. A power series evaluates each small
    remainder; this is distinct from the production symmetric-ratio series.
    """
    q, p = float(alt/depth), float(probability)
    if not np.isfinite(p):
        return np.inf
    if p == q:
        return 0.
    if p <= 0 or p >= 1:
        return np.inf
    if q == 0:
        return float(-2*depth*np.log1p(-p))
    if q == 1:
        return float(-2*depth*np.log(p))
    def remainder(z):
        power, total = z*z, z*z/2
        for exponent in range(3, 101):
            power *= -z
            updated = total + power/exponent
            if updated == total:
                break
            total = updated
        return total
    difference = p-q
    if abs(difference) >= .1*min(q, 1-q):
        return float(2*depth*(q*(np.log(q)-np.log(p))
                             + (1-q)*(np.log1p(-q)-np.log1p(-p))))
    return float(2*depth*(q*remainder(difference/q) + (1-q)*remainder(-difference/(1-q))))


def _conditional_evidence(catalog, source, region, phi, low, high, call, native_call):
    """Independently rebuild the selected pair's binomial terms for publication."""
    if not np.isfinite(call):
        raise ValueError("A defined conditional distribution requires a finite MAP")
    candidates = np.arange(low, high + 1)
    purity = catalog.purity[source, region]
    total_cn = catalog.major_cn[source, region] + catalog.minor_cn[source, region]
    denominator = (1-purity)*catalog.normal_cn[source, region] + purity*total_cn
    p = np.minimum((purity/denominator*candidates)*phi, 1.)
    alt, depth = catalog.alt_counts[source, region], catalog.total_counts[source, region]
    logs = gammaln(depth+1)-gammaln(alt+1)-gammaln(depth-alt+1) + xlogy(alt, p) + xlog1py(depth-alt, -p)
    chosen = int(call)-low
    if phi == 0 and chosen != 0:
        raise ValueError("The prior-only exact multiplicity tie must display the smallest candidate")
    maximum = float(np.max(logs))
    if not np.isclose(logs[chosen], maximum, rtol=1e-10, atol=1e-12):
        raise ValueError("Published conditional MAP does not maximize the component likelihood")
    others = np.delete(logs, chosen)
    log_probability = float(-np.log1p(np.exp(others-logs[chosen]).sum()))
    probabilities = np.exp(logs-logs[chosen]+log_probability)
    gap = float(maximum-np.partition(logs, -2)[-2]) if len(logs) > 1 else np.nan
    native_lp = (float(logs[int(native_call)-low]-logs[chosen]+log_probability)
                 if np.isfinite(native_call) and low <= native_call <= high else -np.inf)
    deviance = _independent_deviance(alt, depth, p[chosen])
    return candidates, probabilities, log_probability, bool(np.isfinite(others).any()), gap, native_lp, deviance


def _independent_scores(catalog, source, reference, informative):
    """Verify absolute marginal masses and both uniform priors from original CN."""
    result = np.full(len(reference.labels), -np.inf)
    if not informative.any():
        return result
    result[:] = -np.log(len(reference.labels))
    for region in np.flatnonzero(informative):
        width = int(catalog.major_cn[source, region])
        candidates = np.arange(1, width + 1)
        purity = catalog.purity[source, region]
        total_cn = catalog.major_cn[source, region] + catalog.minor_cn[source, region]
        denominator = ((1-purity)*catalog.normal_cn[source, region]
                       + purity*total_cn)
        probabilities = np.minimum((purity/denominator*candidates)[None, :]
                                   * reference.centers[:, region, None], 1.)
        alt, depth = catalog.alt_counts[source, region], catalog.total_counts[source, region]
        logs = (gammaln(depth+1)-gammaln(alt+1)-gammaln(depth-alt+1)
                + xlogy(alt, probabilities) + xlog1py(depth-alt, -probabilities))
        result += logsumexp(logs, axis=1)-np.log(width)
    return result


def validate_assignment_result(bundle, result):
    """Check derived evidence and decision gates without fitting or native scoring."""
    from ..assignment import (score_summary, assignment_decision, assignment_outcome, confidence_passes,
                              reference_quality, target_reason)

    if type(bundle) is not AssignmentBundle or type(result) is not AssignmentResult:
        raise ValueError("Assignment output requires validated request/result records")
    bundle.validated_payload()
    if (result.bundle_hash != bundle.content_hash or result.target_ids != bundle.target_ids
            or result.reference.to_dict() != bundle.reference.to_dict()
            or result.summary.get("bundle_hash") != bundle.content_hash):
        raise ValueError("Assignment output must match its live frozen request")
    catalog, labels = bundle.catalog, bundle.reference.labels
    index = {mutation: i for i, mutation in enumerate(catalog.mutation_ids)}
    regional = {(row["mutation_id"], row["region_id"]): row for row in result.multiplicities}
    expected_keys = {(mutation, region) for mutation in bundle.target_ids for region in bundle.reference.region_ids}
    if set(regional) != expected_keys or len(result.multiplicities) != len(expected_keys):
        raise ValueError("Assignment output lost or duplicated target-region keys")
    if len(result.assignments) != len(bundle.target_ids):
        raise ValueError("Assignment output lost target rows")
    for i, (mutation, record) in enumerate(zip(bundle.target_ids, result.assignments)):
        original = index[mutation]
        informative = catalog.eligible[original] & catalog.count_observed[original] & (catalog.total_counts[original] > 0)
        count = int(informative.sum())
        if tuple(record.get("reference_labels", ())) != labels:
            raise ValueError("Assignment probability columns use a different reference-label order")
        scores, probabilities = np.asarray(record["log_scores"], float), np.asarray(record["probabilities"], float)
        if scores.shape != (len(labels),) or probabilities.shape != scores.shape or np.isnan(scores).any() or np.isposinf(scores).any():
            raise ValueError("Assignment scores/probabilities have an invalid shape or value")
        for name in ("best_label", "second_best_label", "accepted_label", "baseline_label",
                     "baseline_reference_label", "final_assignment_label"):
            value = record.get(name)
            if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))):
                raise ValueError("Published assignment labels must be integers or missing")
        _require_fields(record, {"log_scores": _independent_scores(catalog, original, bundle.reference, informative)},
            "Independent assignment emissions", atol=max(1e-12, 32*np.finfo(float).eps*catalog.total_counts[original, informative].sum()))
        defined = count > 0 and np.isfinite(scores).any()
        if defined:
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1., rtol=0, atol=1e-12):
                raise ValueError("Defined assignment probabilities must be nonnegative and normalized")
            expected = score_summary(scores, labels, bundle.thresholds.posterior_min, bundle.thresholds.tie_atol)
            _require_fields(record, {name: expected[name] for name in (
                "best_label", "second_best_label", "posterior", "second_best_probability", "entropy",
                "log_score_margin", "best_log_probability", "posterior_pass", "cluster_tie",
                "has_finite_alternative", "probabilities", "log_competitor_ratio", "comparison_scope")}, "Assignment scores")
        else:
            if not np.isnan(probabilities).all():
                raise ValueError("Uninformative/incompatible assignment probabilities must be missing")
            expected = score_summary(np.full(len(labels), -np.inf), labels,
                                     bundle.thresholds.posterior_min, bundle.thresholds.tie_atol)
            _require_fields(record, {key: value for key, value in expected.items()
                                    if key not in ("best_index", "second_index")}, "Missing assignment")
            _require_fields(record, {"accepted_label": None}, "Missing assignment")
        reference_pass, failed_regions = reference_quality(bundle.reference, expected["best_index"], informative)
        _require_fields(record, dict(reference_quality_pass=reference_pass,
                                    reference_quality_failed_regions=failed_regions), "Reference quality")
        residuals = []
        for j, region in enumerate(bundle.reference.region_ids):
            row = regional[mutation, region]
            low, high = int(bundle.bounds.lower[i, j]), int(bundle.bounds.upper[i, j])
            actual = dict(original_lower=low, original_upper=high,
                original_major_cn=catalog.major_cn[original, j], original_minor_cn=catalog.minor_cn[original, j],
                original_normal_cn=catalog.normal_cn[original, j],
                original_total_cn=catalog.major_cn[original, j] + catalog.minor_cn[original, j],
                alt_count=catalog.alt_counts[original, j], total_depth=catalog.total_counts[original, j],
                ref_count=catalog.total_counts[original, j]-catalog.alt_counts[original, j], purity=catalog.purity[original, j],
                normal_cn_supplied=bool(catalog.normal_cn_supplied[original, j]),
                mapping_genuine=bool(catalog.mapping_genuine[original, j]), segment_id=str(catalog.segment_ids[original, j]),
                count_observed=bool(catalog.count_observed[original, j]), positive_depth=bool(catalog.total_counts[original, j] > 0),
                original_eligible=bool(catalog.eligible[original, j]), excluded=not bool(catalog.eligible[original, j]),
                original_input_excluded=bool(catalog.input_excluded[original, j]),
                informative=bool(informative[j]), native_phi=bundle.native_centers[i, j], native_call=bundle.native_calls[i, j],
                native_lower=bundle.native_lower[i, j], native_upper=bundle.native_upper[i, j],
                source_stage=bundle.source_stages[i], source_label=bundle.source_labels[i],
                best_label=record["best_label"], membership_status=record["status"],
                reference_quality=(bool(bundle.reference.regional_quality[expected["best_index"], j]) if defined else None),
                reference_quality_reasons=(bundle.reference.regional_quality_reasons[expected["best_index"]][j] if defined else ()),
                center_source=(bundle.reference.center_sources[expected["best_index"]] if defined else None),
                multiplicity_tie_atol=bundle.thresholds.multiplicity_tie_atol,
                multiplicity_probability_min=bundle.thresholds.multiplicity_probability_min)
            _require_fields(row, actual, "Regional evidence")
            for name in ("conditional_map", "conditional_call", "assigned_multiplicity"):
                call = float(row[name])
                if not np.isnan(call) and (not np.isfinite(call) or call != np.rint(call) or not low <= call <= high):
                    raise ValueError("Finite conditional calls must belong to the published original support")
            pair_defined = bool(defined and informative[j])
            phi = float(bundle.reference.centers[labels.index(record["best_label"]), j]) if pair_defined else np.nan
            singleton = low == high
            read_informative = bool(pair_defined and phi > 0 and not singleton)
            _require_fields(row, dict(phi=phi, conditional_distribution_defined=pair_defined,
                structural_singleton=singleton, multiplicity_read_informative=read_informative), "Multiplicity information")
            if pair_defined:
                candidates, conditional, log_probability, finite_alternative, gap, native_lp, deviance = _conditional_evidence(
                    catalog, original, j, phi, low, high, row["conditional_map"], bundle.native_calls[i, j])
                _require_fields(row, dict(top_two_gap=gap, native_call_log_probability=native_lp), "Independent conditional evidence",
                    atol=max(1e-12, 16*np.finfo(float).eps*catalog.total_counts[original, j]))
                _require_fields(row, dict(profile_deviance=deviance), "Independent profile deviance", atol=0.)
                actual_probabilities = np.asarray(row["conditional_probabilities"], float)
                if (actual_probabilities.shape != conditional.shape or not np.isfinite(actual_probabilities).all()
                        or np.any(actual_probabilities < 0) or not np.isclose(actual_probabilities.sum(), 1., rtol=0, atol=1e-12)):
                    raise ValueError("Conditional probabilities must be finite, nonnegative, and normalized")
                tied = bool(not singleton and row["top_two_gap"] <= bundle.thresholds.multiplicity_tie_atol)
                concentrated = confidence_passes(log_probability, bundle.thresholds.multiplicity_probability_min, finite_alternative)
                expected_status = ("uninformative_zero_ccf" if phi == 0 else "structural_singleton" if singleton
                                   else "ambiguous_tie" if tied else "low_concentration" if not concentrated else "conditional")
                _require_fields(row, dict(multiplicity_candidates=candidates, conditional_probabilities=conditional,
                    call_probability=float(np.exp(log_probability)), call_log_probability=log_probability,
                    conditional_call=row["conditional_map"] if phi > 0 else np.nan,
                    assigned_multiplicity=(row["conditional_map"] if phi > 0 and record["status"] == "assigned_relative" else np.nan),
                    has_second_component=not singleton, multiplicity_tie=tied, conditional_concentrated=concentrated,
                    multiplicity_status=expected_status), "Conditional distribution")
                if singleton:
                    _require_fields(row, {"top_two_gap": np.nan}, "Singleton uncertainty")
            else:
                if len(row["multiplicity_candidates"]) or len(row["conditional_probabilities"]):
                    raise ValueError("Undefined conditional distributions must have no candidate probabilities")
                expected_status = "excluded" if not catalog.eligible[original, j] else "no_information" if not informative[j] else "incompatible"
                _require_fields(row, dict(conditional_map=np.nan, conditional_call=np.nan, call_probability=np.nan,
                    assigned_multiplicity=np.nan,
                    call_log_probability=np.nan, top_two_gap=np.nan, has_second_component=None,
                    multiplicity_tie=None, conditional_concentrated=None, multiplicity_status=expected_status,
                    native_call_log_probability=np.nan, profile_deviance=np.inf if informative[j] else np.nan), "Undefined multiplicity")
                deviance = np.inf if informative[j] else np.nan
            # The independently verified published value is the scoring value.
            # Distinct stable formulas can differ by one bit at an exact cutoff;
            # apply the strict gate to this canonical value, without widening
            # the scientific threshold. atol=0 above still distinguishes zero
            # from an adjacent positive discrepancy.
            residuals.append(float(row["profile_deviance"]))
        fit_pass = False
        if defined:
            screen = summarize_profile_deviance(np.asarray(residuals), informative, bundle.thresholds.screen, axis=0)
            fit_pass = bool(screen["descriptive_pass"])
            _require_fields(record, dict(mean_profile_deviance=screen["mean_profile_deviance"]), "Mean profile deviance", atol=0.)
            _require_fields(record, dict(outlier_count=int(screen["outlier_count"]), outlier_fraction=screen["outlier_fraction"],
                mean_gate_pass=bool(screen["mean_profile_deviance"] <= bundle.thresholds.screen.mean_deviance_max),
                outlier_gate_pass=bool(screen["outlier_fraction"] <= bundle.thresholds.screen.outlier_fraction_max)), "Descriptive screen")
        status = assignment_decision(expected, bool(catalog.eligible[original].any()), count, fit_pass,
                                     reference_pass=reference_pass)
        accepted = record["best_label"] if status == "assigned_relative" else None
        _require_fields(record, dict(mutation_id=mutation, status=status, accepted_label=accepted,
            informative_regions=count, baseline_label=bundle.baseline_labels[i],
            baseline_reference_label=bundle.baseline_labels[i], final_assignment_label=accepted,
            target_reason=target_reason(bundle.source_stages[i]),
            **assignment_outcome(bundle.source_stages[i], bundle.baseline_labels[i], accepted,
                                 bundle.reference.clonal_label),
            fit_screen_status=("pass" if fit_pass else "fail") if defined else "not_evaluated",
            fit_status="not_tested", pair_deviance_max=bundle.thresholds.screen.pair_deviance_max,
            mean_deviance_max=bundle.thresholds.screen.mean_deviance_max,
            outlier_fraction_max=bundle.thresholds.screen.outlier_fraction_max), "Assignment decision")
    _require_fields(result.summary, dict(schema="clipp2.assignment.result.v4", protocol_version=PROTOCOL_VERSION,
        target_count=len(bundle.target_ids), unchanged_non_target_count=len(bundle.non_target_ids),
        region_count=len(bundle.reference.region_ids), candidate_count=len(labels), support_policy="original_full_1_to_major_cn",
        bounds_hash=bundle.bounds.content_hash, fit_status="not_tested", tumor_id=catalog.tumor_id,
        population_scope="targets_only", comparison_scope="single_candidate" if len(labels) == 1 else "multiple_candidates",
        normal_cn_policy=bundle.normal_cn_policy, thresholds=asdict(bundle.thresholds),
        run_manifest_id=result.run_manifest_id, bundle_run_manifest_id=bundle.run_manifest_id,
        reference=bundle.reference.to_dict()), "Assignment summary")
    _require_fields(result.summary.get("evaluation_receipt", {}), dict(device=bundle.device,
        cpu_threads=bundle.cpu_threads, configuration=asdict(bundle.thresholds),
        run_manifest_id=result.run_manifest_id, bundle_run_manifest_id=bundle.run_manifest_id), "Evaluation receipt")
    aggregates = dict(assigned_occupancy={str(label): sum(row["accepted_label"] == label for row in result.assignments)
                                         for label in labels},
        status_counts={status: sum(row["status"] == status for row in result.assignments)
                       for status in {row["status"] for row in result.assignments}},
        reassessment_counts={status: sum(row["reassessment_status"] == status for row in result.assignments)
                             for status in REASSESSMENT_OUTCOMES})
    for name, expected in aggregates.items():
        if dict(result.summary.get(name, {})) != expected:
            raise ValueError("Assignment aggregate counts do not reconcile its derived records")


def _validated_workflow_accounting(bundle, result, accounting):
    """Reconcile the unchanged map with accepted and unresolved target records."""
    from ..assignment import target_reason
    if any(target_reason(stage) == "fixed_reference_recheck" for stage in bundle.source_stages):
        raise ValueError("Two-origin workflow accounting cannot include generic assignment targets")
    if not isinstance(accounting, Mapping) or not isinstance(accounting.get("unchanged_map"), Mapping):
        raise ValueError("Workflow accounting requires an explicit unchanged mutation-to-reference map")
    labels, unchanged = bundle.reference.labels, dict(accounting["unchanged_map"])
    if set(unchanged) != set(bundle.non_target_ids):
        raise ValueError("Workflow unchanged membership differs from the frozen non-target population")
    if any(isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)) or label not in labels
           for label in unchanged.values()):
        raise ValueError("Unchanged workflow members must have valid integer reference labels")
    fixed = {str(label): sum(value == label for value in unchanged.values()) for label in labels}
    final = {str(label): fixed[str(label)] + sum(row["accepted_label"] == label for row in result.assignments)
             for label in labels}
    expected = dict(population_count=len(bundle.target_ids)+len(unchanged), final_occupancy=final,
        unchanged_non_target_occupancy=fixed,
        unresolved_targets=sum(row["accepted_label"] is None for row in result.assignments))
    def check_counts(actual, wanted):
        if isinstance(wanted, dict):
            return (isinstance(actual, Mapping) and set(actual) == set(wanted)
                    and all(check_counts(actual[key], value) for key, value in wanted.items()))
        return (not isinstance(actual, (bool, np.bool_)) and isinstance(actual, (int, np.integer))
                and actual >= 0 and actual == wanted)
    for key, wanted in expected.items():
        if not check_counts(accounting.get(key), wanted):
            raise ValueError(f"Workflow accounting does not reconcile {key}")
    if sum(final.values()) + expected["unresolved_targets"] != expected["population_count"]:
        raise ValueError("Workflow final occupancy and unresolved targets do not cover T/F")
    if "status" in accounting and accounting["status"] != "complete":
        raise ValueError("Only complete reconciled workflow accounting can be published")
    if "historical_caps_recomputed" in accounting and accounting["historical_caps_recomputed"] is not False:
        raise ValueError("Reassessment must preserve historical caps")
    if "outcomes_by_origin" in accounting:
        origins = {origin: {outcome: sum(row["target_reason"] == origin and row["reassessment_status"] == outcome
            for row in result.assignments) for outcome in ("retained_clonal", "reassigned", "unresolved")}
            for origin in ("stage1_clonal_recheck", "stage2_near_clonal_recheck")}
        if not check_counts(accounting["outcomes_by_origin"], origins):
            raise ValueError("Workflow target-origin outcomes do not reconcile target decisions")
    return _json_value(accounting)


def publish_assignment(bundle, result, outdir, *, workflow_accounting=None):
    validate_assignment_result(bundle, result)
    accounting = (None if workflow_accounting is None else
                  _validated_workflow_accounting(bundle, result, workflow_accounting))
    rows = []
    for i, record in enumerate(result.assignments):
        row = dict(tumor_id=bundle.catalog.tumor_id, mutation_id=bundle.target_ids[i],
            source_stage=bundle.source_stages[i], source_label=bundle.source_labels[i],
            native_phi=json.dumps(_json_value(bundle.native_centers[i])),
            native_calls=json.dumps(_json_value(bundle.native_calls[i])))
        row.update({key: record[key] for key in POLICY_FIELDS})
        for j, label in enumerate(bundle.reference.labels):
            row["probability_" + str(label)] = float(record["probabilities"][j])
            row["log_score_" + str(label)] = float(record["log_scores"][j])
        rows.append(row)
    multiplicities = [dict(tumor_id=bundle.catalog.tumor_id, **row) for row in result.multiplicities]
    target_columns = ["tumor_id", "mutation_id", "source_stage", "source_label", "native_phi", "native_calls", *POLICY_FIELDS]
    for label in bundle.reference.labels:
        target_columns += ["probability_" + str(label), "log_score_" + str(label)]
    regional_columns = ["tumor_id", *REGIONAL_BASE_FIELDS, *REGIONAL_POLICY_FIELDS]
    output = reserve_output(outdir)
    prefix = bundle.catalog.tumor_id + "_"
    paths = [write_supporting_table(output / (prefix + "mutation_assignments.tsv"), rows, columns=target_columns),
             write_supporting_table(output / (prefix + "assignment_multiplicity.tsv"), multiplicities, columns=regional_columns)]
    summary = dict(result.summary, manifest_schema=MANIFEST_SCHEMA, files=[path.name for path in paths])
    if accounting is not None:
        summary.update(population_scope="workflow_combined", workflow_accounting=accounting)
    summary["source_strata"] = {stage: {
        "targets": sum(s == stage for s in bundle.source_stages),
        **{status: sum(s == stage and row["reassessment_status"] == status
                       for s, row in zip(bundle.source_stages, result.assignments))
           for status in REASSESSMENT_OUTCOMES}}
        for stage in sorted(set(bundle.source_stages))}
    _publish_json(output / (prefix + "assignment_summary.json"), summary)
    return summary


def publish_workflow_receipt(outdir, receipt):
    output = Path(outdir)
    files = [path.relative_to(output).as_posix()
             for path in sorted(output.rglob("*")) if path.is_file() and not path.name.startswith(".")
             and path.name != "workflow_receipt.json"]
    return _publish_json(output / "workflow_receipt.json", dict(receipt, manifest_schema=MANIFEST_SCHEMA, files=files))
