"""Prespecified study design, deterministic seed streams, and resource allocation."""
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform

import numpy as np

from CliPP2.config import PROTOCOL_VERSION, AssignmentThresholds, DescriptiveScreen, FitConfig, WorkflowConfig
from CliPP2._version import __version__
from CliPP2.provenance import validate_run_manifest_id

from . import STUDY_VERSION

FAMILIES = tuple(f"S{i:02d}" for i in range(1, 13))
ACQUISITIONS = (
    dict(id="A1", purity=.30, depth=100), dict(id="A2", purity=.30, depth=500),
    dict(id="A3", purity=.80, depth=100), dict(id="A4", purity=.80, depth=500),
)
PHASES = dict(pilot=1, development=2, main=3, controls=4, sensitivity=5)
CONTRIBUTOR_GRID = (.90, .95, .99)
POSTERIOR_GRID = (.80, .90, .95, .99)


def seed_for(master, phase, family, history, acquisition=0):
    """Independent history/acquisition streams, unchanged by scheduling or resume."""
    return int(np.random.SeedSequence([int(master), PHASES[phase], int(family[1:]),
                                      int(history), int(acquisition)]).generate_state(1)[0])


def workflow_config(contributor=.95, posterior=.95, *, device="cpu", threads=1):
    # Research screens are deliberately explicit, descriptive, and not p-values.
    # Numerical tolerance is fixed by qualification, never optimized on truth.
    return WorkflowConfig(
        assignment=AssignmentThresholds(posterior_min=posterior,
            screen=DescriptiveScreen(9., 4., .25), tie_atol=1e-10,
            multiplicity_tie_atol=1e-10, multiplicity_probability_min=.95),
        contributor_probability_min=contributor, contributor_deviance_max=9.,
        group_screen=DescriptiveScreen(9., 4., .25), contributor_tie_atol=1e-10,
        fit=FitConfig(device=device, cpu_threads=threads, max_clusters=10),
        stage_two_max_clusters=20, clonal_distance_max=.05,
        min_informative_per_region=1, min_depth_per_region=1)


def resolve_config(contract, contributor, posterior):
    from copy import deepcopy
    values = deepcopy(contract["base_workflow_config"])
    values["contributor_probability_min"] = contributor
    values["assignment"]["posterior_min"] = posterior
    return WorkflowConfig.from_dict(values)


def source_identity():
    """Record environment metadata; a launcher supplies source/artifact evidence."""
    return dict(git_commit=None, working_tree=None,
                implementation_version=__version__, study_version=STUDY_VERSION,
                python=platform.python_version(), platform=platform.platform(),
                numpy=np.__version__, source_policy="External source/build record or not recorded; no automatic source scan.",
                native_identity_policy="Recorded in each accepted-fit and assignment receipt.")


def make_contract(*, master_seed=None, workers=10, budget_minutes=60.,
                  disk_budget_gib=8., mutations=1500, segments=50, regions=4,
                  profile="balanced", main_histories=None, development_histories=None,
                  cohort_settings=None, reference_match_atol=None, run_manifest_id=None):
    if (workers < 1 or not math.isfinite(budget_minutes) or budget_minutes <= 0
            or mutations < 12 or segments < 1 or regions < 1 or disk_budget_gib <= 0
            or profile not in ("balanced", "full")):
        raise ValueError("Invalid study dimensions or resource budget")
    if any(value is not None and value < 1 for value in (main_histories, development_histories)):
        raise ValueError("Every family must retain at least one history")
    if reference_match_atol is not None and (not math.isfinite(reference_match_atol) or reference_match_atol < 0):
        raise ValueError("Reference matching tolerance must be finite and nonnegative, or None")
    cohort = None
    if cohort_settings is not None:
        path = Path(cohort_settings).resolve()
        payload = json.loads(path.read_text())
        if payload.get("schema_version") != "clipp2.phylogeny.v1" or payload.get("workflow_protocol") != "2.1":
            raise ValueError("Unsupported phylogeny cohort settings")
        if (mutations, segments, regions) != (1500, 50, 4):
            raise ValueError("The supplied core cohort requires exactly 1500 mutations, 50 segments and four regions")
        if master_seed is not None and master_seed != payload["master_seed"]:
            raise ValueError("The supplied cohort fixes its master seed; alternative seeds are a separate study")
        master_seed = payload["master_seed"]
        cohort = dict(path=str(path), schema=payload["schema_version"], version=payload["simulation_plan_version"],
            source_protocol_version=payload["workflow_protocol"],
            variant_ids=[v["id"] for v in payload["variants"]],
            region_ids=payload["region_order"], numpy_version=payload["numpy_version"],
            history_count=len(payload["histories"]),
            variants_per_family=5, main_replicates_per_variant=10, development_replicates_per_variant=2,
            input_authority="Physical event logs and recorded realized tip fractions; original IDs and counts retained")
    master_seed = 20261105 if master_seed is None else master_seed
    return dict(schema="clipp2.simulation-contract.v3", study_version=STUDY_VERSION,
        implementation_version=__version__,
        run_manifest_id=validate_run_manifest_id(run_manifest_id if run_manifest_id is not None
                                               else os.environ.get("CLIPP2_RUN_MANIFEST_ID")),
        protocol_version=PROTOCOL_VERSION, source_cohort_protocol_version=cohort["source_protocol_version"] if cohort else None,
        master_seed=int(master_seed), families=list(FAMILIES), cohort=cohort,
        dimensions=dict(mutations=int(mutations), segments=int(segments), regions=int(regions)),
        acquisitions=list(ACQUISITIONS), development_acquisitions=[ACQUISITIONS[0], ACQUISITIONS[-1]],
        proposed_histories_per_family=dict(main=50, development=10), profile=profile,
        requested_histories_per_family=dict(main=main_histories, development=development_histories),
        resource_limits=dict(workers=min(int(workers), os.cpu_count() or 1),
            cpu_threads_per_fit=1, wall_budget_seconds=float(budget_minutes)*60,
            disk_budget_bytes=int(disk_budget_gib*1024**3),
            archive_completed_outputs=True,
            timing_safety_factor=1.5, cpu_device="cpu"),
        threshold_development=dict(contributor_grid=list(CONTRIBUTOR_GRID),
            posterior_grid=list(POSTERIOR_GRID), clonal_distance_grid=[.05],
            clonal_rule="strict unnormalized L2 < 0.05, fixed by the user's specification",
            fixed_population_families=["S01", "S02", "S07"],
            fixed_population_information="complete-truth/original-support separation > 2 natural-log units",
            objective="Correct end-to-end clonal decisions per fixed original evaluable mutation; equal family weights",
            selective_error_limit=.05, reject_zero_acceptance=True,
            missing_outcomes_correct_credit=0,
            tie_breaks=["lower target wrong acceptance", "higher assignment threshold",
                        "higher contributor threshold", "smaller clonal tolerance", "configuration ID"],
            maximum_development_rounds=2,
            revision_if_no_setting_qualifies="Add independent development histories once; retain all first-round cases. If still unqualified, publish development failure and do not claim a qualified main study."),
        base_workflow_config=asdict(workflow_config()),
        evaluation=dict(primary="exact sampled-region biological clonal membership and CCF error",
            reference_match_atol=reference_match_atol,
            reference_matching="one-to-one center geometry; optional per-coordinate absolute tolerance; overlap only labels calibration",
            candidate_mapping_min_purity=.80, alias_probability_atol=1e-12,
            weak_information_max_nats=2., bootstrap_replicates=1000,
            sampling_unit="independent physical history, retaining all acquisitions and arms",
            family_weighting="equal", support_containment_tolerance=0,
            source_call_containment_tolerance=0,
            noninferiority_margins=dict(wrong_acceptance=.01, false_clonal_merge=.01,
                                       additional_multiplicity_exclusion=.005)),
        optional_panels=dict(external_reference=False, independent_counts=False, technical_sensitivity=False,
            assignment_scaling=False, extended_sensitivity=False,
            fixed_total_depth_refitting=False, corrected_cap_refitting=False,
            read_level_BAM=False, historical_CliPPSim4k="unavailable: generator and truth not supplied"),
        source_identity=source_identity())


def allocate(contract, timings, byte_counts):
    """Freeze a balanced allocation from resource measurements, never outcomes."""
    limits = contract["resource_limits"]
    seconds = max(float(x) for x in timings) * limits["timing_safety_factor"]
    nbytes = max(max(map(int, byte_counts)), 1)
    # Each development history has two acquisitions and three upstream settings.
    # The conservative ledger charges three complete fits for every setting even
    # though the runner reuses A/B across contributor thresholds.
    capacity_time = int(limits["wall_budget_seconds"] * limits["workers"] / seconds)
    capacity_disk = int(limits["disk_budget_bytes"] / nbytes)
    capacity = min(capacity_time, capacity_disk)
    wanted = contract["requested_histories_per_family"]
    cohort = contract.get("cohort")
    blocks, main_max, dev_max = (60, 10, 2) if cohort else (12, 50, 10)
    if contract["profile"] == "full":
        main, dev = wanted["main"] or main_max, wanted["development"] or dev_max
    else:
        dev = wanted["development"] or max(1, min(dev_max, capacity // (blocks*6*5)))
        main = wanted["main"] or min(main_max, (capacity - blocks*6*dev) // (blocks*4))
        if main < 1:
            raise RuntimeError("Timing/disk pilot cannot support one history per family with development. "
                               "Increase the explicit resource budget; no unbalanced cohort was generated.")
    if main > main_max or dev > dev_max:
        raise ValueError("Requested allocation exceeds the supplied cohort")
    return dict(schema="clipp2.simulation-allocation.v1", main_histories_per_family=main*(5 if cohort else 1),
        development_histories_per_family=dev*(5 if cohort else 1),
        main_replicates_per_variant=main if cohort else None,
        development_replicates_per_variant=dev if cohort else None,
        timing_seconds_per_dataset=seconds,
        measured_seconds=list(timings), measured_bytes=list(byte_counts),
        dataset_capacity=capacity, reason="Balanced allocation frozen using timing and disk only, before main generation",
        planned_main_datasets=blocks*main*4, planned_development_datasets=blocks*dev*2,
        planned_core_max_fits=blocks*main*4*3+blocks*dev*2*3,
        planned_core_max_assignment_passes=blocks*main*4*2+blocks*dev*2*2,
        development_grid_additional_fits_with_successful_reuse=blocks*dev*2*2,
        development_grid_unconditional_max_fits=blocks*dev*2*9,
        development_grid_additional_max_assignment_passes=blocks*dev*2*4,
        development_threshold_rescoring="Decision-only reuse of fixed probabilities; no additional native scoring",
        reduced_from_proposal=(main < main_max or dev < dev_max),
        full_proposed_core=dict(histories=720, datasets=2640, max_fits=7920, max_assignment_passes=5280))


def jobs(contract, phase, histories_per_family, *, start=0):
    if contract.get("cohort"):
        payload = json.loads(Path(contract["cohort"]["path"]).read_text())
        for row in payload["histories"]:
            if row["split"] != phase or not start < row["replicate"] <= start+histories_per_family:
                continue
            for acquisition_id in row["acquisition_ids"]:
                from .cohort import semantic_seed
                acquisition = payload["acquisitions"][acquisition_id]
                yield dict(cell_id=f"{row['history_id']}.{acquisition_id}", phase=phase,
                    family=row["variant_id"][:3], variant_id=row["variant_id"], history_id=row["history_id"],
                    history_index=row["replicate"]-1, history_seed=int(row["seed128"]),
                    acquisition_seed=None, cohort_settings=contract["cohort"]["path"],
                    prescribed_fit_seeds=dict(stage1=str(semantic_seed(payload, row["split"], row["variant_id"],
                        row["replicate"], "fits", acquisition_id, 1, "stage1", 1)),
                        stage2_common=str(semantic_seed(payload, row["split"], row["variant_id"], row["replicate"],
                            "fits", acquisition_id, 1, "stage2_common", 1)),
                        consumption="not_consumed: this qualified optimizer is deterministic and has no random restart API"),
                    acquisition=dict(id=acquisition_id, purity=acquisition["purity"], depth=acquisition["mean_depth"]),
                    dimensions=contract["dimensions"])
        return
    acquisitions = contract["development_acquisitions"] if phase == "development" else contract["acquisitions"]
    for family in FAMILIES:
        for index in range(start, start+histories_per_family):
            hseed = seed_for(contract["master_seed"], phase, family, index)
            history_id = f"{phase}-{family}-H{index+1:03d}"
            for setting in acquisitions:
                yield dict(cell_id=f"{history_id}-{setting['id']}", phase=phase, family=family,
                    history_id=history_id, history_index=index, history_seed=hseed,
                    acquisition_seed=seed_for(contract["master_seed"], phase, family, index, int(setting["id"][1:])),
                    acquisition=dict(setting), dimensions=contract["dimensions"])
