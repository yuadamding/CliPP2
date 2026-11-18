"""Resolve missing reads once from the original eligible measured population."""

from ..config import validate_missing_counts


def resolve_read_counts(rows, *, policy):
    """Impute eligible missing ALT=0 and regional median depth (half up)."""
    validate_missing_counts(policy)
    depths, missing = {}, []
    for (mutation, region), row in rows.items():
        alt, ref = row["alt_count"] or 0, row["ref_count"] or 0
        depth = alt + ref
        row.update(alt_count=alt, depth=depth, count_imputed=False)
        if policy == "zero_alt" and not row["input_excluded"]:
            if not row["count_observed"]:
                missing.append((mutation, region))
            elif depth > 0:
                depths.setdefault(region, []).append(depth)
    medians = {}
    for mutation, region in sorted(missing):
        if region not in medians:
            values = depths.get(region)
            if not values:
                raise ValueError(f"Missing reads for {mutation!r}/{region!r} require eligible observed "
                                 "positive depths in that region; no regional median is available.")
            values.sort()
            medians[region] = (values[(len(values)-1)//2] + values[len(values)//2] + 1)//2
        rows[mutation, region].update(alt_count=0, depth=medians[region], count_imputed=True)
    return dict(policy=policy, imputed_pairs=len(missing), regional_medians=medians)
