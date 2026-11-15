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


def _anchor_map(values, anchor):
    states = ((0., ()), (-np.inf, ()))
    for x, y in values:
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
    if math.isfinite(log_mass):
        for j, r in enumerate(active):
            probabilities[r] = np.exp(masses[r]+cavity[r]-log_mass)
            calls[r] = selected[j]
            tied = set(int(lower[r])+int(position) for position in
                       np.flatnonzero(masses[r]+conditional[r] == profile))
            tied.add(selected[j])
            sets[r] = tuple(sorted(tied))
    return dict(log_marginal=log_mass-log_count, profile=profile,
                log_normalizer=log_count, log_partition=log_mass, map=calls,
                map_sets=tuple(sets), probabilities=tuple(probabilities),
                cavity_log_marginal=tuple(cavity), conditional_offsets=tuple(conditional),
                conditional_max=tuple(value+mass for value, mass in zip(conditional, masses)),
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
    """Reconstruct N-by-K columns independently of native caches and emissions."""
    if aggregation not in ("marginal", "profile"):
        raise ValueError("Unknown likelihood aggregation")
    result = np.empty((len(alt), len(centers)))
    field = "log_marginal" if aggregation == "marginal" else "profile"
    for i in range(len(alt)):
        for k, center in enumerate(centers):
            result[i, k] = joint_row(alt[i], depth[i], scaling[i], minimum[i], maximum[i],
                observed[i], center, max_multiplicity_difference=max_multiplicity_difference)[field]
    return result
