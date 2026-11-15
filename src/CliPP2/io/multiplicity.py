"""Read-derived multiplicity floors for the two native workflow stages."""
from __future__ import annotations

import numpy as np

from .data import MultiplicityBounds


READ_FLOOR_POLICY = "cell_fraction_weighted_total_cn_read_floor_v1"


def reconcile_workflow_bounds(catalog, raw_read_floors, raw_caps=None, *,
                             max_multiplicity_difference=1, stage="stage_one"):
    """Reconcile empirical floors/caps before entering a numerical context.

    Original major CN and the fixed observed-region range rule are hard.
    Read floors, then donor caps, are retained as far as those rules allow.
    This is only for workflow-generated bounds; explicit support files remain
    hard constraints and must never pass through this relaxation.
    """
    from ..config import validate_max_multiplicity_difference

    validate_max_multiplicity_difference(max_multiplicity_difference)
    requested = raw_read_floors.subset(catalog.mutation_ids, catalog.region_ids)
    major = np.asarray(catalog.major_cn)
    floors = np.asarray(requested.lower)
    caps = major if raw_caps is None else np.asarray(raw_caps, dtype=np.int64)
    if caps.shape != major.shape or np.any(caps < 1) or np.any(caps > major):
        raise ValueError("Raw workflow caps must lie between one and original major CN")
    observed = catalog.eligible & catalog.count_observed & (catalog.total_counts > 0)
    lower = np.array(floors)
    upper = np.maximum(caps, floors)
    if max_multiplicity_difference is not None:
        for i, mask in enumerate(observed):
            if not mask.any():
                continue
            base = int(np.min(major[i, mask]))
            lower[i, mask] = np.minimum(floors[i, mask], base + 1)
            anchor = max(1, int(np.max(lower[i, mask])) - 1)
            upper[i, mask] = np.maximum(np.maximum(caps[i, mask], lower[i, mask]), anchor)
    records = []
    for i, mutation in enumerate(catalog.mutation_ids):
        for j, region in enumerate(catalog.region_ids):
            reasons = []
            if lower[i, j] < floors[i, j]:
                reasons.append("floor_lowered_for_joint_range")
            if caps[i, j] < lower[i, j]:
                reasons.append("cap_raised_to_retained_floor")
            if upper[i, j] > max(caps[i, j], lower[i, j]):
                reasons.append("cap_raised_for_joint_range")
            records.append(dict(mutation_id=mutation, region_id=region,
                segment_id=str(catalog.segment_ids[i, j]), stage=stage,
                informative=bool(observed[i, j]), original_major_cn=int(major[i, j]),
                raw_read_floor=int(floors[i, j]), raw_upper_cap=int(caps[i, j]),
                effective_lower=int(lower[i, j]), effective_upper=int(upper[i, j]),
                max_multiplicity_difference=max_multiplicity_difference,
                adjustment_reason=";".join(reasons) or "unchanged"))
    policy = ("joint_range_one_original_major_then_floor_then_cap_v1"
              if max_multiplicity_difference == 1 else "independent_read_floor_then_cap_v1")
    return MultiplicityBounds(catalog.mutation_ids, catalog.region_ids, lower, upper, policy), records


def read_floor_bounds(catalog):
    """Return original-major upper bounds and independently estimated floors.

    State fractions describe all tumor cells. Original source reads and purity
    are used, before missing-read imputation or regional purity normalization.
    The estimator assumes normal CN=2, independently of the fitting likelihood.
    Missing, excluded and zero-depth pairs have the uninformative lower bound 1.
    """
    from .cn_states import segment_states_from_catalog

    segment_cn_states = segment_states_from_catalog(catalog)
    totals = {} if segment_cn_states is None else segment_cn_states.weighted_totals
    counts = {} if segment_cn_states is None else segment_cn_states.state_counts
    original_total_cn = catalog.total_cn
    sources = {(row["mutation_id"], row["sample_id"]): row
               for row in map(dict, catalog.source_rows)}
    lower = np.ones(catalog.major_cn.shape, dtype=np.int64)
    records = []
    for i, mutation in enumerate(catalog.mutation_ids):
        for j, region in enumerate(catalog.region_ids):
            source = sources[mutation, region]
            key = region, str(catalog.segment_ids[i, j])
            total_cn = totals.get(key, float(original_total_cn[i, j]))
            purity = float(source["purity"])
            major = int(catalog.major_cn[i, j])
            observed = source["count_observed"] == "1"
            alt = None if source["alt_count"] == "." else float(source["alt_count"])
            ref = None if source["ref_count"] == "." else float(source["ref_count"])
            depth = None if alt is None or ref is None else alt + ref
            raw = None
            if not catalog.eligible[i, j]:
                status = "excluded"
            elif not observed:
                status = "missing_observation"
            elif not depth:
                status = "zero_depth"
            else:
                status = "estimated"
                # Preserve the specified floor operation, including values just
                # below integer boundaries. Clamp before integer conversion.
                with np.errstate(over="ignore"):
                    raw = float((np.float64(alt / depth) / purity)
                                * (purity * total_cn + 2 * (1 - purity)))
                lower[i, j] = int(max(1, min(major, np.floor(raw))))
            records.append(dict(mutation_id=mutation, region_id=region, segment_id=key[1],
                cn_state_count=counts.get(key, 1), weighted_total_cn=total_cn,
                weighted_total_cn_source=("supplied_segment_states"
                    if source.get("segment_cn_states") not in (None, ".") else "single_state_input"),
                input_major_cn=major, input_purity=purity, alt_count=alt, total_count=depth,
                normal_cn_for_estimate=2, raw_multiplicity_estimate=raw,
                multiplicity_lower=int(lower[i, j]), multiplicity_upper=major,
                estimate_status=status, policy=READ_FLOOR_POLICY))
    return MultiplicityBounds(catalog.mutation_ids, catalog.region_ids, lower,
                              catalog.major_cn, READ_FLOOR_POLICY), records
