"""Prespecified study design, deterministic seed streams, and explicit replicate rosters."""
from __future__ import annotations

from dataclasses import asdict
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


def make_contract(*, master_seed=None, workers=10, mutations=1500, segments=50, regions=4,
                  main_replicates=None, development_replicates=None,
                  cohort_settings=None, reference_match_atol=None, run_manifest_id=None):
    if (type(workers) is not int or workers < 1
            or any(type(value) is not int or value < lower for value, lower in
                   ((mutations, 12), (segments, 1), (regions, 1)))):
        raise ValueError("Invalid study dimensions or worker count")
    if any(value is not None and (type(value) is not int or value < 1)
           for value in (main_replicates, development_replicates)):
        raise ValueError("Replicate counts must be positive integers")
    if reference_match_atol is not None and (not math.isfinite(reference_match_atol) or reference_match_atol < 0):
        raise ValueError("Reference matching tolerance must be finite and nonnegative, or None")
    cohort = None
    if cohort_settings is not None:
        from .cohort import load_cohort
        path = Path(cohort_settings).resolve()
        loaded = load_cohort(path)
        payload = loaded.settings
        if (mutations, segments, regions) != (1500, 50, 4):
            raise ValueError("The supplied core cohort requires exactly 1500 mutations, 50 segments and four regions")
        if master_seed is not None and master_seed != payload["master_seed"]:
            raise ValueError("The supplied cohort fixes its master seed; alternative seeds are a separate study")
        master_seed = payload["master_seed"]
        cohort = dict(path=str(path), schema=payload["schema_version"], version=payload["simulation_plan_version"],
            content_id=loaded.content_id,
            source_protocol_version=payload["workflow_protocol"],
            variant_ids=[v["id"] for v in payload["variants"]],
            region_ids=list(payload["region_order"]), numpy_version=payload["numpy_version"],
            history_count=len(payload["histories"]),
            variants_per_family=5, main_replicates_per_variant=10, development_replicates_per_variant=2,
            input_authority="Physical event logs and recorded realized tip fractions; original IDs and counts retained")
    master_seed = 20261105 if master_seed is None else master_seed
    main_max, development_max = (10, 2) if cohort else (50, 10)
    main = main_max if main_replicates is None else main_replicates
    development = development_max if development_replicates is None else development_replicates
    if main > main_max or development > development_max:
        raise ValueError("Declared replicates exceed the supplied study design")
    replicate_roster = dict(unit="variant" if cohort else "family", main=list(range(1, main+1)),
        development=list(range(1, development+1)),
        development_extension=list(range(development+1, min(2*development, development_max)+1)),
        extension_rule="Execute the predeclared second block only if no initial development setting qualifies")
    return dict(schema="clipp2.simulation-contract.v5", study_version=STUDY_VERSION,
        implementation_version=__version__,
        run_manifest_id=validate_run_manifest_id(run_manifest_id if run_manifest_id is not None
                                               else os.environ.get("CLIPP2_RUN_MANIFEST_ID")),
        protocol_version=PROTOCOL_VERSION, source_cohort_protocol_version=cohort["source_protocol_version"] if cohort else None,
        master_seed=int(master_seed), families=list(FAMILIES), cohort=cohort,
        dimensions=dict(mutations=int(mutations), segments=int(segments), regions=int(regions)),
        acquisitions=list(ACQUISITIONS), development_acquisitions=[ACQUISITIONS[0], ACQUISITIONS[-1]],
        replicate_roster=replicate_roster,
        resource_limits=dict(workers=min(workers, os.cpu_count() or 1), cpu_threads_per_fit=1,
            archive_completed_outputs=True, cpu_device="cpu"),
        threshold_development=dict(contributor_grid=list(CONTRIBUTOR_GRID),
            posterior_grid=list(POSTERIOR_GRID), clonal_distance_grid=[.05],
            clonal_rule="strict unnormalized L2 < 0.05, fixed by the user's specification",
            fixed_population_families=["S01", "S02", "S07"],
            fixed_population_information="complete-truth/original-support separation > 2 natural-log units",
            objective="Correct end-to-end clonal decisions per fixed original evaluable mutation; equal family weights",
            endpoint="binary_clonality", constraint_scope="empirical_development_not_population_error_guarantee",
            selective_error_limit=.05, reject_zero_acceptance=True,
            missing_outcomes_correct_credit=0,
            tie_breaks=["lower target wrong acceptance", "higher assignment threshold",
                        "higher contributor threshold", "smaller clonal tolerance", "configuration ID"],
            maximum_development_rounds=2,
            revision_if_no_setting_qualifies="Add independent development histories once; retain all first-round cases. If still unqualified, publish development failure and do not claim a qualified main study."),
        base_workflow_config=asdict(workflow_config()),
        evaluation=dict(record_schema="clipp2.simulation.evaluation.v3",
            primary="exact sampled-region biological clonal membership and CCF error",
            reference_match_atol=reference_match_atol, reference_distance="l2",
            reference_matching="full-vector unnormalized L2 at declared tolerance; maximum valid cardinality then minimum valid distance then lexical tie break",
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


def _replicates(contract):
    roster = contract["replicate_roster"]
    maximums = (10, 2) if contract.get("cohort") else (50, 10)
    if roster.get("unit") != ("variant" if contract.get("cohort") else "family"):
        raise ValueError("Replicate roster unit differs from the study design")
    for name, maximum in zip(("main", "development", "development_extension"),
                             (maximums[0], maximums[1], maximums[1])):
        values = roster.get(name)
        if (not isinstance(values, list) or (not values and name != "development_extension")
                or any(type(value) is not int or not 1 <= value <= maximum for value in values)
                or values != sorted(set(values))):
            raise ValueError(f"Invalid explicit replicate block: {name}")
    if set(roster["development"]) & set(roster["development_extension"]):
        raise ValueError("Development blocks must retain disjoint physical histories")
    return roster


def roster_counts(contract):
    """Population counts depend only on declared replicates and acquisitions."""
    roster = _replicates(contract)
    blocks = len(contract["cohort"]["variant_ids"]) if contract.get("cohort") else len(contract["families"])
    factor = blocks // len(contract["families"])
    main = blocks*len(roster["main"])*len(contract["acquisitions"])
    development = blocks*len(roster["development"])*len(contract["development_acquisitions"])
    extension = blocks*len(roster["development_extension"])*len(contract["development_acquisitions"])
    return dict(planned_main_datasets=main, planned_development_datasets=development,
        conditional_development_extension_datasets=extension,
        main_histories_per_family=factor*len(roster["main"]),
        development_histories_per_family=factor*len(roster["development"]),
        planned_core_max_fits=3*(main+development), planned_core_max_assignment_passes=2*(main+development),
        development_grid_additional_fits_with_successful_reuse=2*development,
        development_grid_unconditional_max_fits=9*development,
        development_grid_additional_max_assignment_passes=4*development)


def jobs(contract, phase, *, extension=False):
    if phase not in ("main", "development") or extension and phase != "development":
        raise ValueError("Core jobs require main or development and an optional development extension")
    replicates = _replicates(contract)["development_extension" if extension else phase]
    if contract.get("cohort"):
        from .cohort import load_cohort
        payload = load_cohort(contract["cohort"]["path"], expected_content_id=contract["cohort"]["content_id"]).settings
        for row in payload["histories"]:
            if row["split"] != phase or row["replicate"] not in replicates:
                continue
            for acquisition_id in row["acquisition_ids"]:
                from .cohort import semantic_seed
                acquisition = payload["acquisitions"][acquisition_id]
                yield dict(cell_id=f"{row['history_id']}.{acquisition_id}", phase=phase,
                    family=row["variant_id"][:3], variant_id=row["variant_id"], history_id=row["history_id"],
                    history_index=row["replicate"]-1, history_seed=int(row["seed128"]),
                    acquisition_seed=None, cohort_settings=contract["cohort"]["path"],
                    cohort_content_id=contract["cohort"]["content_id"],
                    prescribed_fit_seeds=dict(stage1=str(semantic_seed(payload, row["split"], row["variant_id"],
                        row["replicate"], "fits", acquisition_id, 1, "stage1", 1)),
                        stage2_common=str(semantic_seed(payload, row["split"], row["variant_id"], row["replicate"],
                            "fits", acquisition_id, 1, "stage2_common", 1)),
                        consumption="not_consumed: this qualified optimizer is deterministic and has no random restart API"),
                    acquisition=dict(id=acquisition_id, purity=acquisition["purity"], depth=acquisition["mean_depth"]),
                    dimensions=contract["dimensions"])
        return
    acquisitions = contract["development_acquisitions"] if phase == "development" else contract["acquisitions"]
    for family in contract["families"]:
        for replicate in replicates:
            index = replicate-1
            hseed = seed_for(contract["master_seed"], phase, family, index)
            history_id = f"{phase}-{family}-H{index+1:03d}"
            for setting in acquisitions:
                yield dict(cell_id=f"{history_id}-{setting['id']}", phase=phase, family=family,
                    history_id=history_id, history_index=index, history_seed=hseed,
                    acquisition_seed=seed_for(contract["master_seed"], phase, family, index, int(setting["id"][1:])),
                    acquisition=dict(setting), dimensions=contract["dimensions"])
