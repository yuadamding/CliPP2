"""Read-derived multiplicity floors for the two native workflow stages."""
from __future__ import annotations

import numpy as np

from .data import MultiplicityBounds


READ_FLOOR_POLICY = "cell_fraction_weighted_total_cn_read_floor_v1"


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
