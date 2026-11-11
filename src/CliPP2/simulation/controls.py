"""Declared physical-history interventions and fixed-reference scoring audits.

These are mechanism demonstrations, never accepted native fits or estimates of
their frequency in ordinary fitting. A required assignment display baseline is
an explicitly labelled unfitted fixed-center snapshot. Its conditional calls
are evaluated at those coordinates; they are not fabricated native receipts.
"""
from __future__ import annotations

from dataclasses import replace
import os
from time import perf_counter

import numpy as np

from CliPP2.assignment import AssignmentBundle, AssignmentReference, assign_mutations
from CliPP2.config import PROTOCOL_VERSION
from CliPP2.io.assignment import reserve_output, save_assignment_bundle, write_supporting_table
from CliPP2.io.data import MultiplicityBounds, json_value
from CliPP2.io.tumor_txt import load_original_catalog, write_tumor_txt
from CliPP2.workflow import _diagnostics

from .acquisition import acquire
from .biology import generate_history
from .design import workflow_config
from .evaluation import (candidate_correspondence, log_probability_mass,
                         mapped_probabilities, probability_scores, reference_center_diagnostics)
from .experiment import _save_json


def _catalog(history, directory, *, seed, depth=1000, mask_last=False):
    directory.mkdir(parents=True, exist_ok=True)
    frame = acquire(history, seed, purity=.8, depth=depth,
                    perturbation={"mask_distinguishing_region": mask_last})
    path = directory / "input_catalog.tsv"
    write_tumor_txt(path, frame, metadata=frame.attrs["metadata"])
    # The masked-region diagnostic deliberately removes information; it is not
    # an absence-imputation experiment and has no observed depth in that region.
    return load_original_catalog(path, missing_counts="unobserved" if mask_last else "zero_alt")


def _controlled_reference(labels, regions, centers, lineage=(), *, clonal_label=0):
    """Declare controlled coordinates; absent source diagnostics remain missing."""
    return AssignmentReference(tuple(labels), tuple(regions), np.asarray(centers), tuple(lineage),
        clonal_label=clonal_label,
        center_sources=tuple(dict(center_source="declared_fixed_reference_control_not_native_fit",
            reference_label=label) for label in labels))


def _truth_reference(history):
    vectors = [np.ones(len(history.region_ids))]
    for vector in history.phi:
        if not any(np.array_equal(vector, value) for value in vectors):
            vectors.append(vector)
    reference = _controlled_reference(tuple(range(len(vectors))), history.region_ids, np.asarray(vectors),
        tuple(dict(baseline_reference_label=k, classification="declared_true_center_not_native_fit") for k in range(len(vectors))))
    members = {k: [m for m, vector in zip(history.mutation_ids, history.phi)
                   if np.array_equal(vector, value)] for k, value in enumerate(vectors)}
    return reference, members


def _source_arrays(history, accepted_first=None, accepted_second=None):
    """True geometry is a declared control; supplied accepted states stay intact."""
    sources = {m: dict(center=history.phi[i], call=None, lower=None, upper=None,
                      stage="controlled_true_center_snapshot_not_native_fit", label=0)
               for i, m in enumerate(history.mutation_ids)}
    for fit in (accepted_first, accepted_second):
        if fit:
            for i, mutation in enumerate(fit["mutation_ids"]):
                label = int(fit["labels"][i])
                sources[mutation] = dict(center=fit["centers"][label], call=fit["multiplicities"][i],
                    lower=fit["support"]["lower"][i], upper=fit["support"]["upper"][i],
                    stage=fit["source_id"], label=label)
    return sources


def _snapshot_bundle(history, catalog, reference, targets, config, *, sources=None,
                     non_targets=(), independent=False, cap=None, provenance=None):
    """Freeze display baselines without declaring a new fitted/accepted state."""
    targets = tuple(targets)
    sources = _source_arrays(history) if sources is None else sources
    positions = {m: i for i, m in enumerate(catalog.mutation_ids)}
    indices = [positions[m] for m in targets]
    observed = (catalog.eligible & catalog.count_observed & (catalog.total_counts > 0))[indices]
    full = catalog.default_bounds().subset(targets)
    lower, upper = np.array(full.lower), np.array(full.upper)
    phi = np.array([sources[m]["center"] for m in targets], float)
    calls = np.full(phi.shape, np.nan)
    stages, labels, reevaluate = [], [], False
    for i, mutation in enumerate(targets):
        source = sources[mutation]
        stage = source["stage"]
        if source["call"] is not None:
            calls[i] = np.asarray(source["call"], float)
            lower[i], upper[i] = source["lower"], source["upper"]
            if independent and np.any(observed[i] & ~np.isfinite(calls[i])):
                # A newly observed region has no genuine first-read call. Use
                # a declared truth-center display snapshot for this row; keep
                # original source estimates separately in the audit provenance.
                phi[i] = history.phi[history.mutation_ids.index(mutation)]
                lower[i], upper[i] = full.lower[i], full.upper[i]
                stage = "independent_true_center_snapshot_not_native_fit"
                reevaluate = True
        else:
            stage = ("independent_true_center_snapshot_not_native_fit" if independent else
                     "controlled_true_center_snapshot_not_native_fit")
            reevaluate = True
        stages.append(stage)
        labels.append(int(source["label"]))
    if cap is not None:
        upper = np.minimum(upper, cap)
        phi[:] = 1.
        stages = ["controlled_capped_center_snapshot_not_native_fit"] * len(targets)
        reevaluate = True
    if reevaluate:
        bounds = MultiplicityBounds(targets, catalog.region_ids, lower, upper, "fixed_snapshot")
        diagnostics, _ = _diagnostics(catalog, targets, phi, bounds, config)
        evaluated = diagnostics["map_multiplicity"]
        for i, stage in enumerate(stages):
            if "snapshot_not_native_fit" in stage:
                calls[i] = evaluated[i]
    calls[~observed] = np.nan
    return AssignmentBundle(catalog, reference, targets, tuple(non_targets),
        phi, calls, lower, upper, tuple(stages), tuple(labels), (0,) * len(targets),
        device=config.fit.device, cpu_threads=config.fit.cpu_threads,
        assignment_method=config.stage_three_assignment_method,
        normal_cn_policy=config.normal_cn_policy,
        run_manifest_id=os.environ.get("CLIPP2_RUN_MANIFEST_ID"),
        provenance=dict(baseline_status="declared_display_snapshot_not_new_accepted_native_fit",
            native_fit_count=0, independent_count_evaluation=independent,
            **(provenance or {})))


def _score(bundle, path, ledger, name):
    path.mkdir(parents=True, exist_ok=True)
    save_assignment_bundle(bundle, path / "frozen_request.json")
    began = perf_counter()
    ledger["assignment_pass_count"] += 1
    event = dict(panel=name, status="running")
    ledger["events"].append(event)
    try:
        result = assign_mutations(bundle, outdir=path / "assignment")
        event["status"] = "complete"
        _save_json(path / "result.json", result.to_dict())
        return result
    except Exception as error:
        event.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        event["elapsed_seconds"] = perf_counter() - began


def _calibration(history, catalog, result, members, *, reference_match_atol=None):
    truth = dict(zip(history.mutation_ids, history.observable_truth_ids))
    clonal = {m: bool(np.all(phi == 1)) for m, phi in zip(history.mutation_ids, history.phi)}
    clonal_id = next((truth[m] for m in truth if clonal[m]), None)
    mapping, audit = candidate_correspondence(result.reference.labels, result.reference.clonal_label,
        {int(k): value for k, value in members.items()}, truth, clonal_id)
    # An unassayed all-one class remains unmapped; its operational designated
    # probability is reported separately, without inventing biological truth.
    classes = tuple(sorted(set(truth.values())))
    rows = []
    for row in result.assignments:
        probabilities = np.asarray(row["probabilities"], float)
        mutation = row["mutation_id"]
        record = dict(mutation_id=mutation, status=row["status"], final_assignment_label=row["final_assignment_label"], evidence_status=row["evidence_status"],
            truth_class=truth[mutation], true_clonal=clonal[mutation],
            absent_true_class=truth[mutation] not in set(mapping.values()),
            assignment_method=result.assignment_method, score_defined=row["score_defined"],
            K=len(result.reference.labels), probability_available=bool(result.assignment_method == "marginal"
                and row["posterior_defined"] and probabilities.shape == (len(result.reference.labels),)
                and np.isfinite(probabilities).all()),
            data_supported=row["evidence_status"] == "observed")
        if record["probability_available"]:
            values = mapped_probabilities(probabilities, result.reference.labels, mapping, classes)
            designated = float(probabilities[result.reference.labels.index(result.reference.clonal_label)])
            mapped_clonal = values.get(clonal_id, 0.)
            scores = np.asarray(row["log_scores"], float)
            labels = result.reference.labels
            binary_log = log_probability_mass(scores, [i for i, label in enumerate(labels)
                if (label == result.reference.clonal_label) == clonal[mutation]])
            mapped_log = log_probability_mass(scores, [i for i, label in enumerate(labels)
                if (clonal_id is not None and mapping.get(label) == clonal_id) == clonal[mutation]])
            class_log = log_probability_mass(scores, [i for i, label in enumerate(labels)
                if mapping.get(label) == truth[mutation]])
            binary = probability_scores(designated, clonal[mutation], true_log_probability=binary_log)
            multi = probability_scores(values, None, truth_class=truth[mutation], true_log_probability=class_log)
            mapped_binary = probability_scores(mapped_clonal, clonal[mutation], true_log_probability=mapped_log)
            record.update(designated_clonal_probability=designated, mapped_clonal_probability=mapped_clonal,
                mapped_probabilities=values, unmapped_mass=values["__unmapped__"],
                binary_brier=binary["brier"], binary_log_loss=binary["log_loss"],
                mapped_clonal_brier=mapped_binary["brier"], mapped_clonal_log_loss=mapped_binary["log_loss"],
                multiclass_brier=multi["brier"], multiclass_log_loss=multi["log_loss"])
        rows.append(record)
    available = [row for row in rows if row["probability_available"]]
    summary = dict(assignment_method=result.assignment_method,
        tie_policy=result.summary["tie_policy"],
        probability_status="available" if result.assignment_method == "marginal" else "undefined_for_profile",
        target_count=len(rows), probability_count=len(available),
        data_supported_map_count=sum(row["data_supported"] for row in rows),
        data_supported_map_coverage=sum(row["data_supported"] for row in rows) / len(rows) if rows else None,
        absent_true_class_count=sum(row["absent_true_class"] for row in rows),
        K=len(result.reference.labels), single_candidate=len(result.reference.labels) == 1)
    for key in ("binary_brier", "binary_log_loss", "mapped_clonal_brier", "mapped_clonal_log_loss",
                "multiclass_brier", "multiclass_log_loss", "unmapped_mass"):
        summary[key] = float(np.mean([row[key] for row in available])) if available else None
    informative = (catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)).any(axis=1)
    centers = {key: history.phi[i] for i, key in enumerate(history.observable_truth_ids) if informative[i]}
    geometry = reference_center_diagnostics(result.reference.labels, result.reference.centers,
                                           centers, match_atol=reference_match_atol)
    return dict(summary=summary, candidate_mapping=audit, reference_center_geometry=geometry, mutations=rows)


def run_reference_audit(history, catalog, paired_result, config, outdir, second_catalog=None):
    """Score one common panel at B/C/truth, then frozen B/C targets on new reads.

    This optional audit costs at most five assignment passes and zero fits.
    Learned references and actual independent-read target populations are never
    reselected. Missing/inapplicable learned arms retain explicit unavailable rows.
    """
    output = reserve_output(outdir)
    if (set(history.mutation_ids) != set(catalog.mutation_ids)
            or tuple(history.region_ids) != tuple(catalog.region_ids)):
        raise ValueError("Reference audit requires aligned physical truth and original catalog")
    if second_catalog is not None:
        if (second_catalog.biological_identity != catalog.biological_identity
                or second_catalog.mutation_ids != catalog.mutation_ids
                or second_catalog.region_ids != catalog.region_ids):
            raise ValueError("Independent reads must retain tumor and region identities")
        for key in ("major_cn", "minor_cn", "normal_cn", "purity", "segment_ids", "eligible",
                    "mapping_genuine", "normal_cn_supplied", "total_cn"):
            if not np.array_equal(getattr(second_catalog, key), getattr(catalog, key)):
                raise ValueError("Independent reads changed frozen biological inputs")
    informative = catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)
    common = tuple(m for m, keep in zip(catalog.mutation_ids, informative.any(axis=1)) if keep)
    ledger = dict(fit_count=0, assignment_pass_count=0, events=[])
    result = dict(schema="clipp2.fixed-reference-audit.v4", protocol_version=PROTOCOL_VERSION,
                  source_cohort_protocol_version=paired_result.get("source_cohort_protocol_version"),
                  status="complete", common_target_ids=common,
                  cost_ledger=ledger, panels={}, truth_access="evaluation_side_declared_reference_intervention")
    for name in ("B", "C", "truth"):
        arm = paired_result.get("arms", {}).get(name, {})
        if name == "truth":
            reference, members = _truth_reference(history)
            sources = _source_arrays(history)
        elif arm.get("reference"):
            reference = AssignmentReference.from_dict(arm["reference"])
            members = arm["source_members"]
            sources = _source_arrays(history, paired_result.get("stage_one"), arm.get("stage_two"))
        else:
            result["panels"][name] = dict(status="unavailable_reference", source_status=arm.get("status"))
            continue
        bundle = _snapshot_bundle(history, catalog, reference, common, config, sources=sources,
            provenance=dict(panel="common_original_informative", reference_arm=name))
        assigned = _score(bundle, output / name, ledger, name)
        evaluation = _calibration(history, catalog, assigned, members)
        result["panels"][name] = dict(status="complete", targets=common, **evaluation)
        if name != "truth" and second_catalog is not None:
            frozen_targets = tuple(arm["targets"])
            second_bundle = _snapshot_bundle(history, second_catalog, reference, frozen_targets, config,
                sources=sources, non_targets=arm["non_targets"], independent=True,
                provenance=dict(panel="independent_counts_frozen_actual_targets", reference_arm=name,
                    original_native_source_records={m: sources[m] for m in frozen_targets}))
            second = _score(second_bundle, output / (name + "_independent"), ledger, name + "_independent")
            result["panels"][name + "_independent"] = dict(status="complete", targets=frozen_targets,
                **_calibration(history, second_catalog, second, members))
    if ledger["assignment_pass_count"] > 5:
        raise AssertionError("Fixed-reference audit exceeded its declared scoring budget")
    _save_json(output / "reference_audit.json", result)
    return json_value(result)


def _component_alias_count(history, catalog, targets):
    """Count T-versus-A component aliases on the retained observed regions."""
    truth_rows = {m: i for i, m in enumerate(history.mutation_ids)}
    rows = {m: i for i, m in enumerate(catalog.mutation_ids)}
    a_members = set(history.metadata["mutation_classes"]["A"])
    a_center = history.phi[truth_rows[min(a_members)]]
    count = 0
    for mutation in targets:
        i, k = truth_rows[mutation], rows[mutation]
        observed = catalog.eligible[k] & catalog.count_observed[k] & (catalog.total_counts[k] > 0)
        alternative = np.ones(len(catalog.region_ids)) if mutation in a_members else a_center
        product = history.multiplicity[i] * history.phi[i]
        count += bool(observed.any() and all(np.any(np.isclose(
            np.arange(1, int(catalog.major_cn[k, j])+1)*alternative[j], product[j],
            rtol=0, atol=1e-12)) for j in np.flatnonzero(observed)))
    return count


def run_cohort_controls(cohort, outdir):
    """Validate all 60 exact Version 1.2 variants and score declared fixtures.

    First development replicates are chosen by their declared replicate index.
    Physical truth and original mutation ranks are preserved at all 1,500 keys;
    source-set interventions are explicit and never presented as accepted G0.
    No optimization is performed. Optional read masks reuse identical counts.
    """
    from .cohort import acquire_cohort, cohort_history
    output = reserve_output(outdir)
    config = workflow_config()
    ledger = dict(fit_count=0, assignment_pass_count=0, events=[])
    records, panels, retained = [], {}, {}
    selected = {"S05.01", "S04.03", "S05.04", "S12.04", "S11.03"}
    expected_s03 = dict(zip((f"S03.{i:02}" for i in range(1, 6)), (.04, .08, .12, .08, .08)))

    def observation(history, directory, masked=False):
        frame = acquire_cohort(history, cohort, "A4", matched_R4_mask=masked)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "input_catalog.tsv"
        write_tumor_txt(path, frame, metadata=frame.attrs["metadata"])
        return load_original_catalog(path), frame

    for variant_id, variant in sorted(cohort.variants.items()):
        candidates = [record for record in cohort.histories.values()
                      if record["variant_id"] == variant_id and record["split"] == "development"]
        chosen = min(candidates, key=lambda record: (record["replicate"], record["history_id"]))
        history = cohort_history(cohort, chosen["history_id"])
        assert history.phi.shape == (1500, 4) and len(history.segment_keys) == 50
        positions = {m: i for i, m in enumerate(history.mutation_ids)}
        source_ids = tuple(history.metadata["declared_source_ids"])
        source_set = set(source_ids)
        caps, contributors = {}, {}
        for mutation in source_ids:
            i = positions[mutation]
            for j, region in enumerate(history.region_ids):
                if history.phi[i, j] > 0 and np.isfinite(history.multiplicity[i, j]):
                    key = region, str(history.segment_ids[i, j])
                    caps[key] = min(caps.get(key, np.inf), int(history.multiplicity[i, j]))
                    contributors[key] = contributors.get(key, 0) + 1
        denominator = excluded = clonal_excluded = 0
        for i, mutation in enumerate(history.mutation_ids):
            if mutation in source_set:
                continue
            for j, region in enumerate(history.region_ids):
                if history.phi[i, j] <= 0 or not np.isfinite(history.multiplicity[i, j]):
                    continue
                denominator += 1
                cap = caps.get((region, str(history.segment_ids[i, j])), int(history.major_cn[i, j]))
                if history.multiplicity[i, j] > cap:
                    excluded += 1
                    clonal_excluded += int(np.all(history.phi[i] == 1))
        row = dict(variant_id=variant_id, family=variant["family"], history_id=chosen["history_id"],
            tree_id=variant["tree_id"], recipe=variant["recipe"], mutation_count=1500, segment_count=50,
            controlled_reference=variant.get("controlled_reference"), source_count=len(source_ids),
            remainder_count=1500-len(source_ids), positive_carrier_remainder_pairs=denominator,
            truth_cap_excluded_pairs=excluded, clonal_truth_cap_excluded_pairs=clonal_excluded,
            truth_cap_exclusion_rate=excluded/denominator if denominator else None,
            physical_validation="passed", source_status="declared_intervention_not_native_selection")
        if variant_id in expected_s03:
            np.testing.assert_allclose(row["truth_cap_exclusion_rate"], expected_s03[variant_id], atol=1e-15, rtol=0)
            assert excluded == clonal_excluded
            row["expected_exclusion_rate"] = expected_s03[variant_id]
        if variant["family"] == "S09":
            segment = f"S{variant['amplified_segments'][0]:03d}"
            counts = [contributors.get((region, segment), 0) for region in history.region_ids]
            assert counts == [variant["anchor_T_count"]]*4
            row["anchor_contributor_count"] = counts[0]
        directory = output / variant_id
        directory.mkdir()
        _save_json(directory / "declared_source_snapshot.json", dict(
            variant_id=variant_id, history_id=chosen["history_id"], source_ids=source_ids,
            source_status="physical_truth_intervention_not_native_fit",
            segment_caps=[dict(region_id=region, segment_id=segment, truth_cap=caps.get((region, segment)),
                contributor_count=contributors.get((region, segment), 0), fallback=(region, segment) not in caps)
                for region in history.region_ids for segment in history.segment_keys]))
        if variant["family"] == "S10":
            catalog, frame = observation(history, directory / "observation")
            # Rank belongs to the supplied cohort, independent of canonical IO order.
            declared = history.metadata["mutation_records"]
            rank = {record["mutation_id"]: record.get("global_rank", i+1) for i, record in enumerate(declared)}
            indices = {m: i for i, m in enumerate(catalog.mutation_ids)}
            classes = history.metadata["mutation_classes"]
            if variant_id == "S10.01":
                ids = [indices[m] for m in classes["B"]]
                assert np.all(catalog.alt_counts[ids, 0] == 0) and catalog.count_observed[ids, 0].all()
            elif variant_id == "S10.02":
                for mutation, i in indices.items():
                    j = rank[mutation]
                    if j <= 30:
                        assert not catalog.count_observed[i].any() and catalog.eligible[i].all()
                    else:
                        assert bool(catalog.count_observed[i, 1]) == (j % 5 != 0)
                        assert bool(catalog.eligible[i, 2]) == (j % 10 != 0)
                        if j % 10 == 1:
                            assert catalog.total_counts[i, 3] == 0 and catalog.count_observed[i, 3]
                supplied = frame.set_index(["mutation_id", "sample_id"])
                for i, j in np.argwhere(~catalog.eligible & catalog.count_observed):
                    raw = supplied.loc[(catalog.mutation_ids[i], catalog.region_ids[j])]
                    assert catalog.alt_counts[i, j] == float(raw["alt_count"])
            elif variant_id == "S10.03":
                assert not classes.get("T")
            elif variant_id == "S10.04":
                assert np.all(history.phi == 1)
            elif variant_id == "S10.05":
                trunk = set(classes["T"])
                assert all(bool(catalog.count_observed[i, 3]) == (m in trunk)
                           for i, m in enumerate(catalog.mutation_ids))
            row["observation_policy_validation"] = "passed"
        records.append(row)
        if variant_id in selected:
            retained[variant_id] = history
    assert len(records) == 60 and len({row["tree_id"] for row in records}) == 7
    write_supporting_table(output / "structural_controls.tsv", records)

    # Restored m=3 explains clonal T_L at the alternative A=1/3 center.
    history = retained["S05.01"]
    catalog, _ = observation(history, output / "S05.01" / "observation")
    major = dict(zip(history.mutation_ids, history.major_cn[:, 0]))
    targets = tuple(m for m in history.metadata["mutation_batches"]["T_L"] if major[m] == 4)
    ancestor = history.phi[history.mutation_ids.index(history.metadata["mutation_classes"]["A"][0])]
    np.testing.assert_allclose(ancestor, 1/3, rtol=0, atol=1e-15)
    reference = _controlled_reference((0,), history.region_ids, np.asarray([ancestor]), clonal_label=None)
    bundle = _snapshot_bundle(history, catalog, reference, targets, config, cap=2,
        provenance=dict(intervention="S05_T_L_at_fixed_A_center", snapshot_status="unfitted_cap2_source"))
    scored = _score(bundle, output / "S05.01" / "restored_m3", ledger, "S05_restore_m3_at_A")
    assert all(row["native_upper"] == 2 and row["original_upper"] == 4
               and 3 in row["multiplicity_candidates"] for row in scored.multiplicities)
    panels["S05_restore_m3_at_A"] = dict(status="passed", target_count=len(targets),
        restored_pairs=len(scored.multiplicities), observed_map3_pairs=sum(row["conditional_map"] == 3 for row in scored.multiplicities),
        explanation="m3 is a restored alternative at A=1/3, not the true multiplicity of A mutations")

    for variant_id, expected_pairs in (("S04.03", 450), ("S05.04", 400), ("S12.04", 450)):
        history = retained[variant_id]
        reference, members = _truth_reference(history)
        full_catalog, frame = observation(history, output / variant_id / "unmasked")
        masked_catalog, masked_frame = observation(history, output / variant_id / "masked", masked=True)
        original_mask = frame.attrs["acquisition"]["count_observed"]
        changed_mask = masked_frame.attrs["acquisition"]["count_observed"]
        assert int(np.sum(original_mask & ~changed_mask)) == expected_pairs
        for field in ("latent_alt_counts", "latent_depth"):
            np.testing.assert_array_equal(frame.attrs["acquisition"][field], masked_frame.attrs["acquisition"][field])
        targets = tuple(history.metadata["mutation_classes"]["T"] + history.metadata["mutation_classes"]["A"])
        aliases_before = _component_alias_count(history, full_catalog, targets)
        aliases_after = _component_alias_count(history, masked_catalog, targets)
        assert aliases_after > aliases_before
        for suffix, catalog in (("unmasked", full_catalog), ("masked", masked_catalog)):
            assigned = _score(_snapshot_bundle(history, catalog, reference, targets, config),
                output / variant_id / (suffix + "_assignment"), ledger, variant_id + "_" + suffix)
            panels[variant_id + "_" + suffix] = dict(status="complete", masked_pairs=expected_pairs if suffix == "masked" else 0,
                exact_component_alias_count=aliases_after if suffix == "masked" else aliases_before,
                                                    **_calibration(history, catalog, assigned, members))

    history = retained["S11.03"]
    catalog, _ = observation(history, output / "S11.03" / "observation")
    complete, members = _truth_reference(history)
    ancestor = history.phi[history.mutation_ids.index(history.metadata["mutation_classes"]["A"][0])]
    a_label = next(label for label, vector in zip(complete.labels, complete.centers)
                   if np.allclose(vector, ancestor, rtol=0, atol=1e-12))
    for mode in ("complete", "omit_A", "duplicate_A", "distort_A"):
        labels, centers, source_members = list(complete.labels), np.array(complete.centers), dict(members)
        if mode == "omit_A":
            keep = [i for i, label in enumerate(labels) if label != a_label]
            labels, centers = [labels[i] for i in keep], centers[keep]
        elif mode == "duplicate_A":
            label = max(labels)+1
            centers = np.vstack((centers, centers[labels.index(a_label)]))
            labels.append(label)
            source_members[label] = members[a_label]
        elif mode == "distort_A":
            centers[labels.index(a_label)] = np.minimum(1, ancestor+.15)
        reference = _controlled_reference(tuple(labels), history.region_ids, centers)
        assigned = _score(_snapshot_bundle(history, catalog, reference, history.mutation_ids, config),
            output / "S11.03" / mode, ledger, "S11_" + mode)
        panels["S11_" + mode] = dict(status="complete", **_calibration(history, catalog, assigned, source_members))
    result = dict(schema="clipp2.cohort-controls.v4", simulation_plan_version="1.2", status="passed",
        protocol_version=PROTOCOL_VERSION, source_cohort_protocol_version=cohort.settings["workflow_protocol"],
        structural_variant_count=len(records), structural_records=records, panels=panels,
        cost_ledger=ledger, assignment_pass_count=ledger["assignment_pass_count"],
        interpretation="Exact supplied physical histories; declared source/reference fixtures, zero fits")
    _save_json(output / "cohort_controls_summary.json", result)
    return json_value(result)


def run_controls(outdir):
    """Run deterministic physical mechanisms and declared fixed-reference panels."""
    output = reserve_output(outdir)
    config = workflow_config()
    ledger = dict(fit_count=0, assignment_pass_count=0, events=[])
    checks, panels = [], {}

    def record(name, **evidence):
        checks.append(dict(check=name, passed=True, **evidence))

    # The cap counterexample is a physical-copy statement, independent of fitting.
    history = generate_history("S03", 300, n_mutations=120, n_segments=6, n_regions=4)
    history.write_truth(output / "S03")
    positions = {m: i for i, m in enumerate(history.mutation_ids)}
    sources = history.metadata["declared_source_ids"]
    caps = {}
    for mutation in sources:
        i = positions[mutation]
        for j, region in enumerate(history.region_ids):
            key = region, history.segment_ids[i, j]
            caps[key] = min(caps.get(key, np.inf), history.multiplicity[i, j])
    audit = []
    for i, mutation in enumerate(history.mutation_ids):
        for j, region in enumerate(history.region_ids):
            upper = caps.get((region, history.segment_ids[i, j]), history.major_cn[i, j])
            audit.append(dict(mutation_id=mutation, region_id=region, segment_id=history.segment_ids[i, j],
                declared_source=mutation in sources, true_phi=history.phi[i, j],
                true_multiplicity=history.multiplicity[i, j], truth_derived_cap=upper,
                excluded=mutation not in sources and history.multiplicity[i, j] > upper))
    excluded = [row for row in audit if row["excluded"]]
    assert excluded and any(row["true_phi"] == 1 for row in excluded)
    assert all(history.multiplicity[positions[m], j] == 1 for m in sources for j in range(4))
    write_supporting_table(output / "S03" / "truth_cap_audit.tsv", audit)
    record("S03_correct_clonal_reference_call_can_exclude_other_clonal_truth",
           excluded_pairs=len(excluded), clonal_excluded_pairs=sum(row["true_phi"] == 1 for row in excluded))

    # Exact m=1,phi=1 versus m=2,phi=.5 alias from parallel physical gains.
    history = generate_history("S04", 300, n_mutations=120, n_segments=6, n_regions=4)
    history.write_truth(output / "S04")
    catalog = _catalog(history, output / "S04", seed=400)
    targets = tuple(history.metadata["mutation_classes"]["A"])
    indexes = [history.mutation_ids.index(m) for m in targets]
    np.testing.assert_array_equal(history.multiplicity[indexes] * history.phi[indexes], 1.)
    true_reference, members = _truth_reference(history)
    k1 = _controlled_reference((0,), catalog.region_ids, np.ones((1, 4)))
    for name, reference, source_members in (("K1", k1, {0: []}), ("complete", true_reference, members)):
        assigned = _score(_snapshot_bundle(history, catalog, reference, targets, config),
                          output / "S04" / name, ledger, "S04_" + name)
        panels["S04_" + name] = _calibration(history, catalog, assigned, source_members)
        if name == "K1":
            assert all(row["probabilities"] == ((1.,) if assigned.assignment_method == "marginal" else None)
                       for row in assigned.assignments)
            assert any(row["final_assignment_label"] == 0 for row in assigned.assignments)
            record("S04_K1_probability_one_does_not_establish_true_clonality", targets=len(targets))
        else:
            assert len(reference.labels) > 1 and all(
                row["probabilities"][0] < 1 if assigned.assignment_method == "marginal" else row["probabilities"] is None
                for row in assigned.assignments)
            record("S04_complete_reference_preserves_competing_alias", K=len(reference.labels))

    history = generate_history("S05", 300, n_mutations=120, n_segments=6, n_regions=4)
    history.write_truth(output / "S05")
    catalog = _catalog(history, output / "S05", seed=401)
    reference, members = _truth_reference(history)
    targets = tuple(history.metadata["mutation_classes"]["A"])
    capped = _snapshot_bundle(history, catalog, reference, targets, config, cap=1,
        provenance=dict(intervention="declared_cap_induced_all_one_source_geometry_not_fitted"))
    assigned = _score(capped, output / "S05" / "restored", ledger, "S05_restoration")
    assert all(row["native_upper"] == 1 and row["original_upper"] == 2
               and tuple(row["multiplicity_candidates"]) == (1, 2) for row in assigned.multiplicities)
    record("S05_full_support_restored_from_declared_capped_snapshot", restored_pairs=len(assigned.multiplicities))
    panels["S05_restoration"] = _calibration(history, catalog, assigned, members)

    # Changing unused F call records cannot affect a fixed target request.
    sources = _source_arrays(history)
    first = _snapshot_bundle(history, catalog, reference, targets, config, sources=sources)
    for mutation in set(history.mutation_ids) - set(targets):
        sources[mutation]["call"] = [999.] * 4
    second = _snapshot_bundle(history, catalog, reference, targets, config, sources=sources)
    assert first.content_hash == second.content_hash
    record("unused_non_target_multiplicity_records_leave_fixed_request_identical")

    # An informative final region breaks an alias; masking it restores equality.
    history = generate_history("S12", 302, n_mutations=120, n_segments=6, n_regions=4)
    history.write_truth(output / "S12")
    targets = tuple(history.metadata["mutation_classes"]["A"])
    index = history.mutation_ids.index(targets[0])
    product = history.multiplicity[index] * history.phi[index]
    assert np.all(product[:-1] == 1) and product[-1] != 1
    reference = _controlled_reference((0, 1), history.region_ids, np.array([np.ones(4), history.phi[index]]))
    members = {0: list(history.metadata["mutation_classes"]["R"]), 1: list(targets)}
    for name, masked in (("distinguishing_observed", False), ("distinguishing_missing", True)):
        catalog = _catalog(history, output / "S12" / name, seed=402, mask_last=masked)
        assigned = _score(_snapshot_bundle(history, catalog, reference, targets, config),
                          output / "S12" / (name + "_score"), ledger, "S12_" + name)
        panels["S12_" + name] = _calibration(history, catalog, assigned, members)
        if masked:
            assert not catalog.count_observed[:, -1].any()
            assert all(row["multiplicity_status"] == "missing_observation"
                       for row in assigned.multiplicities if row["region_id"] == history.region_ids[-1])
    record("S12_missing_distinguishing_region_restores_exact_component_alias")

    # Equal native vectors remain separate labels, and distorted coordinates alter scores.
    catalog = _catalog(history, output / "reference_perturbation", seed=403)
    duplicate = _controlled_reference((0, 1, 2), history.region_ids,
        np.array([np.ones(4), history.phi[index], history.phi[index]]))
    duplicated = _score(_snapshot_bundle(history, catalog, duplicate, targets, config),
                        output / "reference_perturbation" / "duplicate", ledger, "duplicate_candidate")
    assert all(row["log_scores"][1] == row["log_scores"][2] for row in duplicated.assignments)
    assert any(row["evidence_status"] == "observed" and not row["map_unique"] for row in duplicated.assignments)
    record("duplicate_candidates_preserve_labels_equal_priors_and_ties", K=3)
    distorted = replace(reference, centers=np.array([np.ones(4), np.maximum(0, history.phi[index] - .2)]))
    changed = _score(_snapshot_bundle(history, catalog, distorted, targets, config),
                     output / "reference_perturbation" / "distorted", ledger, "distorted_candidate")
    unperturbed = _score(_snapshot_bundle(history, catalog, reference, targets, config),
                        output / "reference_perturbation" / "unperturbed", ledger, "unperturbed_candidate")
    assert any(not np.array_equal(a["log_scores"], b["log_scores"])
               for a, b in zip(changed.assignments, unperturbed.assignments))
    record("reference_centroid_perturbation_changes_fixed_target_scores")
    result = dict(schema="clipp2.simulation-controls.v4", protocol_version=PROTOCOL_VERSION, status="passed", checks=checks,
        panels=panels, cost_ledger=ledger, assignment_pass_count=ledger["assignment_pass_count"],
        paths=sorted(str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()),
        interpretation="declared physical/source/reference interventions; no native clustering frequency claim")
    _save_json(output / "controls_summary.json", result)
    return json_value(result)
