"""Resolve missing reads once; frozen catalogs retain raw rows and imputation evidence."""
from collections import Counter

from ..config import validate_missing_counts


def _eligible(row, cutoff):
    return not row["input_excluded"] and (cutoff is None or row["allele_a_cn"] <= cutoff)


def _supplied_depth(row):
    # With missing ALT, a supplied reference count is the depth under ALT=0.
    # ALT alone cannot establish the total number of reads.
    return (row["alt_count"] or 0) + row["ref_count"] if row["ref_count"] is not None else 0


def resolve_read_counts(rows, *, cutoff, policy):
    """Use supplied positive depth, otherwise the region's eligible observed median.

    Median ties at half a read round upward. Imputed, excluded and zero-depth
    pairs never donate depth. An absent usable regional depth is an error, not
    permission to invent one. Explicit observed zero-depth pairs stay unchanged.
    """
    validate_missing_counts(policy)
    if policy == "unobserved":
        return rows, ()
    depths = {}
    for row in rows.values():
        if _eligible(row, cutoff) and row["count_observed"]:
            depth = row["alt_count"] + row["ref_count"]
            if depth > 0:
                depths.setdefault(row["sample_id"], []).append(depth)
    medians = {}
    for region, values in depths.items():
        values.sort()
        medians[region] = (values[(len(values)-1)//2] + values[len(values)//2] + 1)//2
    records = []
    for (mutation, region), row in sorted(rows.items()):
        if row["count_observed"] or not _eligible(row, cutoff):
            continue
        depth = _supplied_depth(row)
        method = "supplied_depth"
        if depth <= 0:
            depth, method = medians.get(region), "regional_median"
        if depth is None:
            raise ValueError(f"Missing reads for {mutation!r}/{region!r} need a positive supplied depth "
                             "or eligible observed depths in that region; no regional median is available.")
        records.append((mutation, region, depth, method))
    records = tuple(records)
    return replay_read_counts(rows, records, cutoff=cutoff, policy=policy), records


def replay_read_counts(rows, records, *, cutoff, policy):
    """Validate and apply a frozen imputation ledger, never recomputing subset medians."""
    validate_missing_counts(policy)
    resolved = dict(rows)
    expected = {key for key, row in rows.items() if policy == "zero_alt"
                and not row["count_observed"] and _eligible(row, cutoff)}
    seen, medians = set(), {}
    for record in records:
        if not isinstance(record, (tuple, list)) or len(record) != 4:
            raise ValueError("Read-count imputation requires mutation, region, positive depth and depth source")
        mutation, region, depth, method = record
        key = mutation, region
        if key not in expected or key in seen or type(depth) is not int or depth < 1:
            raise ValueError("Invalid, duplicate or ineligible read-count imputation")
        seen.add(key)
        row = rows[key]
        supplied = _supplied_depth(row)
        if method != ("supplied_depth" if supplied > 0 else "regional_median"):
            raise ValueError("Read-count imputation depth source disagrees with the raw observation")
        if supplied > 0 and depth != supplied:
            raise ValueError("Read-count imputation changed the supplied depth")
        if method == "regional_median" and medians.setdefault(region, depth) != depth:
            raise ValueError("Frozen regional median depths must agree within each region")
        resolved[key] = dict(row, alt_count=0, ref_count=depth, count_observed=True)
    if seen != expected:
        raise ValueError("Read-count imputation ledger does not cover exactly the eligible missing pairs")
    return resolved


def read_count_report(policy, records):
    return dict(policy=policy, imputed_pairs=len(records),
                depth_sources=dict(Counter(record[3] for record in records)),
                regional_medians={record[1]: record[2] for record in records if record[3] == "regional_median"},
                median_population="eligible_observed_positive_depth_pairs", median_rounding="half_up")
