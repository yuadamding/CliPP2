"""Independent Python oracle for the regional range-one multiplicity model.

Production inference uses the shared native evaluator. This small implementation
reconstructs its mathematics for publication and numerical qualification.
Inputs to the DP are raw component log likelihoods, without regional priors.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy, xlog1py

from ..config import validate_max_multiplicity_difference


def _combine(values, maximum=False):
    return max(values) if maximum else float(logsumexp(values))


def _messages(values, *, maximum=False):
    """Prefix/suffix messages indexed by whether an anchor has occurred."""
    size = len(values)
    prefix = np.full((size + 1, 2), -np.inf)
    suffix = np.full_like(prefix, -np.inf)
    prefix[0, 0] = suffix[size, 0] = 0.
    for k in range(size):
        x, y = values[k]
        p0, p1 = prefix[k]
        prefix[k+1] = p0+y, _combine((p1+x, p1+y, p0+x), maximum)
        j = size-k-1
        x, y = values[j]
        p0, p1 = suffix[j+1]
        suffix[j] = p0+y, _combine((p1+x, p1+y, p0+x), maximum)
    return prefix, suffix


def _cavity(prefix, suffix, region, contains_anchor, *, maximum=False):
    left, right = prefix[region], suffix[region+1]
    choices = (left[0]+right[1], left[1]+right[0], left[1]+right[1])
    if contains_anchor:
        choices += (left[0]+right[0],)
    return _combine(choices, maximum)


def _anchor_map(values, anchor, fixed=None):
    states = ((0., ()), (-np.inf, ()))
    for region, (x, y) in enumerate(values):
        if fixed is not None and region == fixed[0]:
            if fixed[1] != anchor:
                x = -np.inf
            if fixed[1] != anchor+1:
                y = -np.inf
        candidates = ([], [])
        for used, (score, word) in enumerate(states):
            if not math.isfinite(score):
                continue
            if math.isfinite(x):
                candidates[1].append((score+x, word+(anchor,)))
            if math.isfinite(y):
                candidates[used].append((score+y, word+(anchor+1,)))
        states = tuple(min(items, key=lambda item: (-item[0], item[1])) if items else (-np.inf, ())
                       for items in candidates)
    return states[1]


def joint_components(log_masses, minimum, *, observed=None, max_multiplicity_difference=1):
    """Score one ragged regional vector and return latent joint diagnostics.

    Each ``log_masses[r]`` covers consecutive integers beginning at ``minimum[r]``.
    The observation mask is fixed independently of candidate CCF. ``map`` retains
    latent calls at zero CCF; publication decides which biological calls are NA.
    Exact MAP ties use the lexicographically smallest regional vector.
    """
    validate_max_multiplicity_difference(max_multiplicity_difference)
    masses = tuple(np.asarray(value, dtype=float) for value in log_masses)
    lower = np.asarray(minimum, dtype=np.int64)
    mask = np.ones(len(masses), dtype=bool) if observed is None else np.asarray(observed, dtype=bool)
    if lower.shape != (len(masses),) or mask.shape != lower.shape:
        raise ValueError("Joint multiplicity inputs must have aligned regional axes")
    if any(value.ndim != 1 or not len(value) or np.isnan(value).any() or np.isposinf(value).any()
           for value, used in zip(masses, mask) if used) or np.any(lower < 1):
        raise ValueError("Observed component masses require nonempty finite-or-negative-infinity vectors")
    active = np.flatnonzero(mask)
    calls = np.full(len(masses), np.nan)
    cavity = [np.full(len(value), -np.inf) for value in masses]
    conditional = [np.full(len(value), -np.inf) for value in masses]
    conditional_max = [np.full(len(value), -np.inf) for value in masses]
    probabilities = [np.zeros(len(value)) for value in masses]
    sets = [() for _ in masses]
    log_count = log_mass = profile = 0.
    selected = ()
    if len(active) and max_multiplicity_difference is None:
        totals = [float(logsumexp(masses[r])) for r in active]
        maxima = [float(np.max(masses[r])) for r in active]
        log_count = math.fsum(math.log(len(masses[r])) for r in active)
        log_mass = sum(totals)
        profile = sum(maxima)
        selected = tuple(int(lower[r]+np.argmax(masses[r])) for r in active)
        for j, r in enumerate(active):
            cavity[r][:] = sum(value for k, value in enumerate(totals) if k != j)
            conditional[r][:] = sum(value for k, value in enumerate(maxima) if k != j)
            conditional_max[r][:] = [sum(float(value) if k == j else maxima[k]
                for k in range(len(active))) for value in masses[r]]
    elif len(active):
        largest_lower = int(lower[active].max())
        smallest_upper = min(int(lower[r])+len(masses[r])-1 for r in active)
        if largest_lower > smallest_upper+1:
            raise ValueError("Empty joint multiplicity support")
        log_count = log_mass = profile = -np.inf
        selected = ()
        for anchor in range(max(1, largest_lower-1), smallest_upper+1):
            values, structure = [], []
            for r in active:
                pair = tuple(float(masses[r][m-lower[r]]) if lower[r] <= m < lower[r]+len(masses[r])
                             else -np.inf for m in (anchor, anchor+1))
                values.append(pair)
                structure.append(tuple(0. if lower[r] <= m < lower[r]+len(masses[r]) else -np.inf
                                       for m in (anchor, anchor+1)))
            prefix, suffix = _messages(values)
            max_prefix, max_suffix = _messages(values, maximum=True)
            counts, _ = _messages(structure)
            log_count = float(np.logaddexp(log_count, counts[-1, 1]))
            log_mass = float(np.logaddexp(log_mass, prefix[-1, 1]))
            score, word = _anchor_map(values, anchor)
            if score > profile or (score == profile and word and (not selected or word < selected)):
                profile, selected = score, word
            for j, r in enumerate(active):
                for multiplicity in (anchor, anchor+1):
                    position = multiplicity-int(lower[r])
                    if not 0 <= position < len(masses[r]):
                        continue
                    value = _cavity(prefix, suffix, j, multiplicity == anchor)
                    cavity[r][position] = np.logaddexp(cavity[r][position], value)
                    value = _cavity(max_prefix, max_suffix, j, multiplicity == anchor, maximum=True)
                    conditional[r][position] = max(conditional[r][position], value)
                    # Exact ties require the same regional addition order as the
                    # full MAP score. Prefix + suffix regrouping can lose valid
                    # ties by one ulp, particularly at zero-CCF coordinates.
                    value = _anchor_map(values, anchor, fixed=(j, multiplicity))[0]
                    conditional_max[r][position] = max(conditional_max[r][position], value)
    if math.isfinite(log_mass):
        for j, r in enumerate(active):
            probabilities[r] = np.exp(masses[r]+cavity[r]-log_mass)
            calls[r] = selected[j]
            tied = set(int(lower[r])+int(position) for position in
                       np.flatnonzero(conditional_max[r] == profile))
            tied.add(selected[j])
            sets[r] = tuple(sorted(tied))
    return dict(log_marginal=log_mass-log_count, profile=profile,
                log_normalizer=log_count, log_partition=log_mass, map=calls,
                map_sets=tuple(sets), probabilities=tuple(probabilities),
                cavity_log_marginal=tuple(cavity), conditional_offsets=tuple(conditional),
                conditional_max=tuple(conditional_max),
                status="observed" if len(active) and math.isfinite(log_mass) else
                       "zero_likelihood" if len(active) else "unobserved")


def raw_binomial_components(alt, depth, scaling, minimum, maximum, phi):
    """Unnormalized binomial component log masses, preserving probability endpoints."""
    multiplicities = np.arange(int(minimum), int(maximum)+1, dtype=float)
    probability = np.minimum(1., float(scaling)*multiplicities*float(phi))
    width_log = math.log(len(multiplicities))
    # Reconstruct the existing independent constant, then remove its prior at
    # the raw-component boundary. The order also specifies exact computed ties.
    constant = float(gammaln(depth+1)-gammaln(alt+1)-gammaln(depth-alt+1))-width_log
    constant += width_log
    return xlogy(alt, probability) + xlog1py(depth-alt, -probability) + constant


def joint_row(alt, depth, scaling, minimum, maximum, observed, phi, *, max_multiplicity_difference=1):
    """Independent raw-binomial reconstruction for one mutation's center vector."""
    arrays = tuple(np.asarray(value) for value in (alt, depth, scaling, minimum, maximum, observed, phi))
    alt, depth, scaling, minimum, maximum, observed, phi = arrays
    if any(value.shape != alt.shape for value in arrays) or alt.ndim != 1:
        raise ValueError("Joint row arrays must be aligned regional vectors")
    observed = observed.astype(bool)
    if np.any(~np.isfinite(phi[observed])) or np.any((phi[observed] < 0) | (phi[observed] > 1)):
        raise ValueError("Observed CCFs must be finite and in [0, 1]")
    masses = tuple(raw_binomial_components(alt[r], depth[r], scaling[r], minimum[r], maximum[r], phi[r])
                   if observed[r] else np.zeros(int(maximum[r]-minimum[r]+1)) for r in range(len(alt)))
    result = joint_components(masses, minimum, observed=observed,
                              max_multiplicity_difference=max_multiplicity_difference)
    selected, regional = np.full(len(alt), np.nan), np.full(len(alt), np.nan)
    for r in np.flatnonzero(observed):
        regional[r] = np.max(masses[r])
        if np.isfinite(result["map"][r]):
            selected[r] = masses[r][int(result["map"][r]-minimum[r])]
    independent = sum(regional[observed])
    loss = independent-result["profile"] if np.isfinite(result["profile"]) else np.nan
    result.update(selected_log_mass=selected, regional_profile=regional,
                  coupling_profile_loss=max(0., loss) if np.isfinite(loss) else loss,
                  raw_log_masses=masses)
    return result


def joint_log_columns(alt, depth, scaling, minimum, maximum, observed, centers, *,
                      max_multiplicity_difference=1, aggregation="marginal"):
    """Likelihood-only N-by-K oracle, independent of native caches/emissions.

    The range-one DP partitions assignments by their smallest multiplicity.
    Its two states record whether that anchor has occurred. Vectorizing over
    mutations and centers avoids the MAP/tie/cavity work in ``joint_row``;
    that separate diagnostic oracle remains unchanged. Only O(N*K) workspace
    is needed, and the structural normalizer is evaluated once per row.
    """
    validate_max_multiplicity_difference(max_multiplicity_difference)
    if aggregation not in ("marginal", "profile"):
        raise ValueError("Unknown likelihood aggregation")
    alt, depth, scaling, minimum, maximum, observed = (
        np.asarray(value) for value in (alt, depth, scaling, minimum, maximum, observed))
    centers = np.asarray(centers, dtype=float)
    if (alt.ndim != 2 or any(value.shape != alt.shape for value in
            (depth, scaling, minimum, maximum, observed)) or centers.ndim != 2
            or centers.shape[1] != alt.shape[1]):
        raise ValueError("Joint columns require aligned N-by-R inputs and K-by-R centers")
    observed = observed.astype(bool)
    if (not np.isfinite(minimum).all() or not np.isfinite(maximum).all()
            or np.any(minimum < 1) or np.any(maximum < minimum)
            or np.any(minimum != np.floor(minimum)) or np.any(maximum != np.floor(maximum))):
        raise ValueError("Multiplicity supports require nonempty positive integer intervals")
    used_centers = centers[:, observed.any(axis=0)]
    if (not np.isfinite(used_centers).all()
            or np.any((used_centers < 0) | (used_centers > 1))):
        raise ValueError("Observed CCFs must be finite and in [0, 1]")
    if (any(not np.isfinite(value[observed]).all() for value in (alt, depth, scaling))
            or np.any(alt[observed] < 0) or np.any(depth[observed] < alt[observed])
            or np.any(scaling[observed] < 0)):
        raise ValueError("Observed reads and scaling must be finite valid binomial inputs")
    n, regions = alt.shape
    result = np.zeros((n, len(centers)))
    if not observed.any() or not len(centers):
        return result
    # Match raw_binomial_components' constant reconstruction, including its
    # prior-removal arithmetic. Missing observations are never evaluated.
    constant = np.zeros_like(alt, dtype=float)
    y, d = alt[observed], depth[observed]
    width_log = np.log(maximum[observed]-minimum[observed]+1)
    constant[observed] = gammaln(d+1)-gammaln(y+1)-gammaln(d-y+1)-width_log
    constant[observed] += width_log
    reference = depth-alt

    def component(rows, region, multiplicity):
        probability = np.minimum(1., scaling[rows, region, None] * multiplicity
                                 * centers[None, :, region])
        return (xlogy(alt[rows, region, None], probability)
                + xlog1py(reference[rows, region, None], -probability)
                + constant[rows, region, None])

    combine = np.logaddexp if aggregation == "marginal" else np.maximum
    if max_multiplicity_difference is None:
        for region in range(regions):
            rows = np.flatnonzero(observed[:, region])
            if not len(rows):
                continue
            mass = np.full((len(rows), len(centers)), -np.inf)
            width = maximum[rows, region]-minimum[rows, region]+1
            for offset in range(int(width.max())):
                live = np.flatnonzero(offset < width)
                values = component(rows[live], region, minimum[rows[live], region, None]+offset)
                mass[live] = combine(mass[live], values)
            if aggregation == "marginal":
                mass -= np.log(width)[:, None]
            result[rows] += mass
        return result

    active = observed.any(axis=1)
    # Only these anchors can produce a complete assignment for each row.
    first = np.maximum(1, np.max(np.where(observed, minimum, 1), axis=1)-1)
    last = np.min(np.where(observed, maximum, np.inf), axis=1)
    if np.any(first[active] > last[active]):
        raise ValueError("Empty joint multiplicity support")
    result[active] = -np.inf
    log_count = np.full(n, -np.inf)
    for anchor in range(int(first[active].min()), int(last[active].max())+1):
        rows = np.flatnonzero(active & (first <= anchor) & (anchor <= last))
        if not len(rows):
            continue
        absent = np.zeros((len(rows), len(centers)))
        present = np.full_like(absent, -np.inf)
        count_absent = np.zeros(len(rows))
        count_present = np.full(len(rows), -np.inf)
        for region in range(regions):
            local = np.flatnonzero(observed[rows, region])
            source = rows[local]
            if not len(local):
                continue
            supports_anchor = (minimum[source, region] <= anchor) & (anchor <= maximum[source, region])
            supports_next = ((minimum[source, region] <= anchor+1)
                             & (anchor+1 <= maximum[source, region]))
            x = np.where(supports_anchor[:, None], component(source, region, anchor), -np.inf)
            y = np.where(supports_next[:, None], component(source, region, anchor+1), -np.inf)
            present[local] = combine(present[local] + combine(x, y), absent[local]+x)
            absent[local] += y
            if aggregation == "marginal":
                x_count = np.where(supports_anchor, 0., -np.inf)
                y_count = np.where(supports_next, 0., -np.inf)
                count_present[local] = np.logaddexp(
                    count_present[local]+np.logaddexp(x_count, y_count), count_absent[local]+x_count)
                count_absent[local] += y_count
        result[rows] = combine(result[rows], present)
        if aggregation == "marginal":
            log_count[rows] = np.logaddexp(log_count[rows], count_present)
    if aggregation == "marginal":
        if not np.isfinite(log_count[active]).all():
            raise ValueError("Empty joint multiplicity support")
        result[active] -= log_count[active, None]
    return result
