"""Paired sequencing acquisitions with biological truth separate from supplied inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .biology import History


def _regional(value, regions, name):
    result = np.broadcast_to(np.asarray(value, float), (regions,)).copy()
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite')
    return result


def acquire(history: History, seed, purity=.8, depth=100, *, missing_rate=0., zero_depth_rate=0.,
            perturbation=None, max_major_cn=10):
    """Return one canonical long table and an explicit acquisition audit in attrs.

    Supported perturbations: purity_offset; cn_error_fraction (one copy in a
    supplied major/minor count); segment_error_fraction; overdispersion (beta-
    binomial ICC); alt_odds_ratio; discovery_min_alt; mask_distinguishing_region;
    and uncovered_fraction (known nongenuine mapping, expected strict rejection).
    Input-only errors reuse exactly the same latent reads at a fixed seed.
    """
    if not isinstance(history, History):
        raise TypeError('acquire requires a validated physical History')
    options = dict(perturbation or {})
    allowed = {'purity_offset', 'cn_error_fraction', 'segment_error_fraction', 'overdispersion',
               'alt_odds_ratio', 'discovery_min_alt', 'mask_distinguishing_region', 'uncovered_fraction'}
    if set(options)-allowed:
        raise ValueError(f'Unknown acquisition perturbations: {sorted(set(options)-allowed)}')
    for name, value in [('missing_rate', missing_rate), ('zero_depth_rate', zero_depth_rate),
            *((key, options.get(key, 0.)) for key in ('cn_error_fraction', 'segment_error_fraction', 'uncovered_fraction'))]:
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be in [0,1]')
    n, r = history.phi.shape
    rho, mean_depth = _regional(purity, r, 'purity'), _regional(depth, r, 'depth')
    if np.any(rho <= 0) or np.any(rho > 1) or np.any(mean_depth < 0):
        raise ValueError('Purity must be in (0,1] and depth nonnegative')
    supplied_purity = rho + _regional(options.get('purity_offset', 0.), r, 'purity_offset')
    if np.any(supplied_purity <= 0) or np.any(supplied_purity > 1):
        raise ValueError('Supplied purity perturbation leaves (0,1]; choose a valid acquisition')
    normal = np.full((n, r), 2.)
    true_denom = (1-rho)*normal + rho*(history.major_cn+history.minor_cn)
    burden = history.mutant_copy_counts @ history.clone_fractions
    biological_probability = rho*burden/true_denom
    reduced = rho*np.nan_to_num(history.multiplicity)*history.phi/true_denom
    np.testing.assert_allclose(biological_probability, reduced, rtol=0, atol=1e-12)
    # Keep exact physical endpoints after the independent floating-point sums.
    assert np.all((biological_probability >= -1e-12) & (biological_probability <= 1+1e-12))
    biological_probability = np.clip(biological_probability, 0., 1.)
    biological_probability[reduced == 0], biological_probability[reduced == 1] = 0., 1.
    # Verify total contributions from the physical copies as well as the scalar reduction.
    segment_position = {value: i for i, value in enumerate(history.segment_keys)}
    total = np.array([history.copy_cn[segment_position[history.segment_ids[i, 0]]].sum(axis=1)
                      @ history.clone_fractions for i in range(n)])
    np.testing.assert_allclose(total, history.major_cn+history.minor_cn, rtol=0, atol=1e-12)
    assert np.all((biological_probability >= 0) & (biological_probability <= 1))
    depth_rng, read_rng, mask_rng, input_rng = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)]
    depths = depth_rng.poisson(mean_depth, size=(n, r))
    depths[mask_rng.random((n, r)) < zero_depth_rate] = 0
    odds = float(options.get('alt_odds_ratio', 1.))
    eta = float(options.get('overdispersion', 0.))
    if not np.isfinite(odds) or odds <= 0 or not np.isfinite(eta) or not 0 <= eta < 1:
        raise ValueError('ALT odds multiplier must be positive and beta-binomial ICC in [0,1)')
    mean_probability = odds*biological_probability/(1-biological_probability+odds*biological_probability)
    sampled_probability = mean_probability.copy()
    if eta:
        interior = (mean_probability > 0) & (mean_probability < 1)
        kappa = 1/eta-1
        sampled_probability[interior] = read_rng.beta(mean_probability[interior]*kappa,
                                                     (1-mean_probability[interior])*kappa)
    alternate = read_rng.binomial(depths, sampled_probability)
    observed = mask_rng.random((n, r)) >= missing_rate
    if history.family == 'S10':
        observed[0] = False
        depths[1, 0], alternate[1, 0] = 0, 0
    if options.get('mask_distinguishing_region', False):
        distinguishing = history.metadata.get('distinguishing_region_index')
        if distinguishing is not None:
            observed[:, distinguishing] = False
    discovery = options.get('discovery_min_alt')
    if discovery is not None:
        if type(discovery) is not int or discovery < 0:
            raise ValueError('discovery_min_alt must be a nonnegative integer')
        observed &= (alternate >= discovery).any(axis=1)[:, None]
    major, minor = np.array(history.major_cn), np.array(history.minor_cn)
    supplied_segments = np.array(history.segment_ids)
    active_segments = tuple(sorted(set(history.segment_ids[:, 0])))
    cn_changed, mapping_changed = np.zeros((n, r), bool), np.zeros((n, r), bool)
    # Errors are shared by an interval, not independently assigned to its mutations.
    count = int(np.ceil(len(active_segments)*options.get('cn_error_fraction', 0.)))
    for segment in input_rng.choice(active_segments, count, replace=False):
        rows = history.segment_ids[:, 0] == segment
        a, b = int(major[rows, 0][0]), int(minor[rows, 0][0])
        alternatives = [(a+1, b), (a-1, b), (a, b+1), (a, b-1)]
        alternatives = [(x, y) for x, y in alternatives if x >= max(1, y) and y >= 0]
        a, b = alternatives[int(input_rng.integers(len(alternatives)))]
        major[rows], minor[rows], cn_changed[rows] = a, b, True
    count = int(np.ceil(len(active_segments)*options.get('segment_error_fraction', 0.)))
    if count and len(active_segments) < 2:
        raise ValueError('Segment-mapping errors require at least two assayed intervals')
    before_major, before_minor = major.copy(), minor.copy()
    for segment in input_rng.choice(active_segments, count, replace=False):
        choices = [value for value in active_segments if value != segment]
        destination = choices[int(input_rng.integers(len(choices)))]
        rows, target = history.segment_ids[:, 0] == segment, np.flatnonzero(history.segment_ids[:, 0] == destination)[0]
        supplied_segments[rows], major[rows], minor[rows] = destination, before_major[target], before_minor[target]
        mapping_changed[rows] = True
    genuine = mask_rng.random((n, r)) >= options.get('uncovered_fraction', 0.)
    records = [dict(mutation_id=mutation, sample_id=region,
        alt_count=int(alternate[i, j]) if observed[i, j] else '.',
        ref_count=int(depths[i, j]-alternate[i, j]) if observed[i, j] else '.',
        count_observed=int(observed[i, j]), purity=float(supplied_purity[j]), normal_cn=2,
        segment_id=str(supplied_segments[i, j]), cn_state_id='0', cn_state_fraction=1.,
        allele_a_cn=int(major[i, j]), allele_b_cn=int(minor[i, j]),
        mapping_genuine=int(genuine[i, j]), normal_cn_supplied=1)
        for i, mutation in enumerate(history.mutation_ids) for j, region in enumerate(history.region_ids)]
    frame = pd.DataFrame(records)
    eligible = major <= max_major_cn
    attrs = dict(schema='clipp2.simulation.acquisition.v1', seed=int(seed),
        true_purity=np.broadcast_to(rho, (n, r)).copy(), normal_cn=normal,
        true_probability=biological_probability, read_mean_probability=mean_probability,
        sampled_probability=sampled_probability, supplied_purity=np.broadcast_to(supplied_purity, (n, r)).copy(),
        alt_counts=np.where(observed, alternate, 0), depth=np.where(observed, depths, 0),
        total_counts=np.where(observed, depths, 0), count_observed=observed,
        latent_alt_counts=alternate, latent_depth=depths, supplied_major_cn=major, supplied_minor_cn=minor,
        supplied_segment_ids=supplied_segments, mapping_genuine=genuine, original_eligible=eligible,
        input_loss=(~observed)|(depths == 0)|(~eligible), biological_truth_available=np.ones((n, r), bool),
        positive_carrier=history.phi > 0, cn_error=cn_changed, segment_mapping_error=mapping_changed,
        inherited_support_exclusion=(history.phi > 0) & (history.multiplicity > major),
        settings=dict(purity=rho.tolist(), depth=mean_depth.tolist(), missing_rate=missing_rate,
                      zero_depth_rate=zero_depth_rate, perturbation=options))
    frame.attrs['acquisition'] = attrs
    frame.attrs['metadata'] = dict(tumor_id=history.history_id, genome_build='synthetic_atomic_intervals',
        coordinate_system='1-based-inclusive', mutation_key_convention='simulation_somatic_snv_id')
    return frame
