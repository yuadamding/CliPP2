"""Protocol 3.0 biological assignments and independent publication verification."""
from dataclasses import dataclass, field, fields
import hashlib
from itertools import product
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp, xlogy, xlog1py

from .data import readonly_array


OUTPUT_SCHEMA = 18
OUTPUT_SUFFIXES = ('mutation_clusters.tsv', 'cluster_centers.tsv',
                   'mutation_region_multiplicity.tsv')
_ENUMERATE_MAX = 512


def _log_binomial(alt, depth, probability):
    p = np.minimum(probability, 1.)
    ref = depth-alt
    return (gammaln(depth+1)-gammaln(alt+1)-gammaln(ref+1)
            + xlogy(alt, p) + xlog1py(ref, -p))


def _first_minimum_log_partition(x, y):
    """Partition into disjoint vectors by the first occurrence of the minimum.

    This verifier uses a different reduction from the fitting evaluator's joint
    DP, and never obtains likelihoods from the model's cached emission columns.
    For a pivot at j, earlier coordinates must be high, j is low, and later
    coordinates are unrestricted. This reduces the partition in linear time.
    """
    both = np.logaddexp(x, y)
    before = np.r_[0., np.cumsum(y[:-1])]
    after = np.r_[np.cumsum(both[:0:-1])[::-1], 0.]
    return float(logsumexp(before+x+after))


def _uniform_joint(masses):
    """Independent exact support, log partition and joint MAP.

    Regional mass vectors start at multiplicity one and need only cover the
    joint support range; at least one region has the smallest original maximum.
    """
    upper = tuple(map(len, masses))
    if not upper:
        raise ValueError('Clonal verification requires an informative region')
    minimum = min(upper)
    count = (minimum-1)*((1 << len(upper))-1) + (1 << sum(u > minimum for u in upper))
    log_count = math.log(count)
    best, best_mass, total = None, -np.inf, -np.inf
    if count <= _ENUMERATE_MAX:
        records = []
        for h in range(1, minimum+1):
            for vector in product(*(range(h, min(h+1, u)+1) for u in upper)):
                if min(vector) == h:
                    log_mass = math.fsum(m[multi-1] for m, multi in zip(masses, vector))
                    records.append((log_mass, vector))
        if len(records) != count:
            raise ValueError('Joint multiplicity enumeration disagrees with support count')
        total = float(logsumexp([entry[0] for entry in records]))
        best_mass, best = min(records, key=lambda pair: (-pair[0], pair[1]))
    else:
        for h in range(1, minimum+1):
            x = np.array([m[h-1] for m in masses])
            y = np.array([m[h] if len(m) > h else -np.inf for m in masses])
            partition = _first_minimum_log_partition(x, y)
            total = float(np.logaddexp(total, partition))
            if not math.isfinite(partition):
                continue
            use_high = y > x  # On an exact tie, low is lexicographically smaller.
            if use_high.all():
                # Independent maxima were all high, which is forbidden because
                # it has no minimum h. Choose the smallest possible score loss.
                use_high[np.argmax(x-y)] = False
            vector = tuple(int(h+v) for v in use_high)
            log_mass = math.fsum(np.where(use_high, y, x))
            if log_mass > best_mass or (log_mass == best_mass and (best is None or vector < best)):
                best_mass, best = log_mass, vector
    return total, log_count, best_mass, best


def _clonal_masses(alt, depth, scaling, upper):
    maximum = int(min(upper))+1  # No admissible vector exceeds this coordinate.
    return tuple(_log_binomial(a, d, scale*np.arange(1, min(maximum, int(u))+1, dtype=float))
                 for a, d, scale, u in zip(alt, depth, scaling, upper))


def _enriched_map(masses, uniform, weight_u, weight_v):
    _, log_count, best_mass, best = uniform
    log_ones = math.fsum(m[0] for m in masses)
    ones = (1,)*len(masses)
    log_u = math.log(weight_u)-log_count
    log_v = math.log(weight_v) if weight_v else -np.inf
    # Compare the uniform MAP with the all-one vector after adding its bonus.
    ones_score = float(np.logaddexp(log_u, log_v))+log_ones
    return best if best is not None and log_u+best_mass > ones_score else ones


@dataclass(frozen=True)
class BiologicalAssignment:
    """Immutable verified biological labels and observed multiplicity calls."""
    mutation_ids: tuple
    region_ids: tuple
    component_labels: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    weights: np.ndarray
    internal_to_biological: np.ndarray
    clonal_prior_beta: float | None
    ccf: np.ndarray
    multiplicities: np.ndarray
    log_likelihood: float
    bic: float
    parameter_count: int
    _data: object = field(repr=False, compare=False)
    _fit: object = field(repr=False, compare=False)
    _proof: str = field(init=False, default='', repr=False, compare=False)

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, np.ndarray):
                object.__setattr__(self, item.name, readonly_array(value))

def _proof(assignment):
    digest = hashlib.sha256(b'clipp2.protocol3.biological-assignment.v1\0')
    for item in fields(assignment):
        if item.name.startswith('_'):
            continue
        value = getattr(assignment, item.name)
        digest.update(item.name.encode()+b'\0')
        if isinstance(value, np.ndarray):
            digest.update(repr((value.shape, value.dtype.str)).encode()+b'\0')
            digest.update(value.tobytes())
        else:
            digest.update(repr(value).encode()+b'\0')
    return digest.hexdigest()


def _axes(requested, source, *, name):
    idx = {value: j for j, value in enumerate(source)}
    if len(set(requested)) != len(requested) or not set(requested) <= idx.keys():
        raise ValueError(f'Publication {name} keys disagree with its input')
    return [idx[key] for key in requested]


def _publication_observations(model, catalog):
    data = model.data
    ids, regions = tuple(model.mutation_ids), tuple(model.region_ids)
    if (model.n, model.r) != (len(ids), len(regions)) or not ids or not regions:
        raise ValueError('Publication requires a nonempty aligned fitting population')
    ij = np.ix_(_axes(ids, data.mutation_ids, name='mutation'),
                _axes(regions, data.region_ids, name='region'))
    alt, depth = data.alt_counts[ij], data.total_counts[ij]
    excluded = data.input_excluded[ij].copy()
    # The loader's CN-filtered cells have inert numerical placeholders. Recover
    # their absent biological CCFs from the original eligibility evidence.
    report = data.cn_filter_report
    if report is not None and report.records:
        i_map, j_map = map(lambda keys: {key: j for j, key in enumerate(keys)}, (ids, regions))
        for record in report.records:
            i, j = i_map.get(record.mutation_id), j_map.get(record.sample_id)
            if i is not None and j is not None:
                excluded[i, j] = True
    informative = data.count_observed[ij] & ~excluded & (depth > 0.)
    major, minor, rho = data.major_cn[ij], data.minor_cn[ij], data.purity[ij]
    denominator = (1.-rho)*data.normal_cn[ij]+rho*(major+minor)
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.) or not informative.any(axis=1).all():
        raise ValueError('Publication inputs lack usable original read evidence or CN')
    scaling = rho/denominator
    if not (np.array_equal(informative, model.observed) and np.array_equal(major, model.major)
            and np.array_equal(alt[informative], model.alt[informative])
            and np.array_equal(depth[informative], model.depth[informative])
            and np.array_equal(scaling, model.scaling)):
        raise ValueError('Fitting observations disagree with independently reconstructed input arrays')
    measured = informative.copy()
    bound = data.original_catalog
    if catalog is not None and bound is not None and catalog.content_hash != bound.content_hash:
        raise ValueError('Publication catalog differs from the preserved original input')
    catalog = bound if catalog is None else catalog
    if catalog is not None:
        if catalog.tumor_id != data.tumor_id:
            raise ValueError('Publication catalog belongs to another tumor')
        i_map, j_map = map(lambda keys: {key: j for j, key in enumerate(keys)}, (ids, regions))
        for mid, rid, _, _ in catalog.read_count_imputations:
            i, j = i_map.get(mid), j_map.get(rid)
            if i is not None and j is not None:
                measured[i, j] = False
    return ids, regions, alt, depth, scaling, major, informative, excluded, measured


def _independent_scores(template, centers, alt, depth, scaling, major, informative):
    one = np.where(informative, _log_binomial(alt, depth, scaling), 0.).sum(axis=1)
    columns, uniform = [], None
    if 'U' in template:
        uniform = []
        marginal = np.empty(len(alt))
        for i, active in enumerate(informative):
            masses = _clonal_masses(alt[i, active], depth[i, active], scaling[i, active], major[i, active])
            details = _uniform_joint(masses)
            marginal[i] = details[0]-details[1]
            uniform.append((masses, details))
        columns.append(marginal)
    if 'V' in template:
        columns.append(one)
    for center in centers:
        regional = _log_binomial(alt, depth, scaling*center)
        columns.append(np.where(informative, regional, 0.).sum(axis=1))
    return np.column_stack(columns), uniform


def evaluate_fit(model, fit, *, catalog=None):
    """Verify a selected mixture and collapse its fixed terms before MAP.

    No fitting or center/weight updates occur. Reconstruct component emissions
    independently using original CN, informative reads and the range-one rule.
    """
    ids, regions, alt, depth, scaling, major, informative, excluded, measured = _publication_observations(model, catalog)
    n, r = len(ids), len(regions)
    template = tuple(fit.template)
    centers, weights = np.asarray(fit.centers, dtype=float), np.asarray(fit.weights, dtype=float)
    q, s = len(template), len(centers)
    singleton = bool(np.all(major[informative] == 1))
    if (template not in ((), ('U',), ('V',), ('U', 'V')) or (singleton and 'U' in template)
            or q+s == 0 or centers.shape != (s, r) or weights.shape != (q+s,)
            or not np.isfinite(centers).all() or np.any((centers < 0.) | (centers > 1.))
            or np.any(np.all(centers == 1., axis=1)) or len(np.unique(centers, axis=0)) != s
            or not np.isfinite(weights).all() or np.any(weights <= 0.)
            or not math.isclose(math.fsum(weights), 1., rel_tol=0., abs_tol=1e-12)
            or fit.biological_count != s+bool(q)):
        raise ValueError('Selected fit is not an active canonical component model')
    columns, uniform = _independent_scores(template, centers, alt, depth, scaling, major, informative)
    log_joint = columns+np.log(weights)[None, :]
    log_observed = logsumexp(log_joint, axis=1)
    log_likelihood = math.fsum(log_observed)
    parameters = s*r+q+s-1
    bic = -2.*log_likelihood+parameters*math.log(n)
    if (not np.isfinite(log_observed).all()
            or not math.isclose(log_likelihood, fit.log_likelihood, rel_tol=1e-10, abs_tol=2e-7)
            or not math.isclose(bic, fit.bic, rel_tol=1e-10, abs_tol=2e-7)):
        raise ValueError('Selected likelihood or BIC does not match independent publication replay')
    internal = np.asarray(fit.responsibilities, dtype=float)
    independent = np.exp(log_joint-log_observed[:, None])
    if (internal.shape != (n, q+s) or not np.isfinite(internal).all() or np.any(internal < 0.)
            or not np.allclose(internal.sum(axis=1), 1., rtol=0., atol=1e-10)
            or not np.allclose(internal, independent, rtol=2e-8, atol=1e-10)):
        raise ValueError('Fitted responsibilities do not match independent publication replay')

    order = np.argsort(-np.linalg.norm(centers, axis=1), kind='stable')
    free_labels = np.empty(s, dtype=int)
    free_labels[order] = np.arange(1, s+1)
    internal_map = np.r_[np.zeros(q, dtype=int), free_labels]
    component_labels = np.arange(0 if q else 1, s+1)
    biological_centers, biological_weights = centers[order], weights[q:][order]
    posterior = internal[:, q:][:, order]
    wc = float(math.fsum(weights[:q]))
    wu = float(weights[template.index('U')]) if 'U' in template else 0.
    wv = float(weights[template.index('V')]) if 'V' in template else 0.
    if q:
        posterior = np.column_stack((internal[:, :q].sum(axis=1), posterior))
        biological_centers = np.vstack((np.ones(r), biological_centers))
        biological_weights = np.r_[wc, biological_weights]
    maximum = np.argmax(posterior, axis=1)
    labels = component_labels[maximum]  # First maximum: smallest public label.
    ccf = biological_centers[maximum].copy()
    valid_calls = measured & (ccf > 0.)
    calls = np.where(valid_calls, 1., np.nan)
    for i in np.flatnonzero(labels == 0):
        if not np.isfinite(columns[i, :q]).any():
            raise ValueError('Clonal MAP label has no finite clonal evidence')
        if wu:
            vector = _enriched_map(*uniform[i], wu, wv)
            for j, multi in zip(np.flatnonzero(informative[i]), vector):
                if valid_calls[i, j]:
                    calls[i, j] = multi
    ccf[excluded] = np.nan
    if not np.array_equal(np.isfinite(calls), valid_calls):
        raise ValueError('Verified regional multiplicity calls do not respect the observed report mask')
    result = BiologicalAssignment(ids, regions, component_labels, labels, biological_centers, biological_weights,
        internal_map, None if not q or singleton else wv/wc, ccf, calls,
        log_likelihood, bic, parameters, model.data, fit)
    object.__setattr__(result, '_proof', _proof(result))
    return result


def publish_stage(model, fit, assignment, outdir):
    """Verify all three TSVs before creating the new output directory."""
    if (type(assignment) is not BiologicalAssignment or assignment._fit is not fit
            or assignment._data is not model.data or assignment._proof != _proof(assignment)
            or tuple(model.mutation_ids) != assignment.mutation_ids or tuple(model.region_ids) != assignment.region_ids):
        raise ValueError('Stage publication requires the verified assignment for this exact fit and model')
    tumor = model.data.tumor_id
    if not tumor or any(c in tumor for c in '/\\\t\r\n'):
        raise ValueError('Tumor ID must be a nonempty single path component')
    labels, cluster_labels = assignment.labels, assignment.component_labels
    ids, regions = assignment.mutation_ids, assignment.region_ids
    wide = pd.DataFrame(dict(tumor_id=tumor, mutation_id=ids, cluster_label=labels))
    cluster = pd.DataFrame(dict(tumor_id=tumor, cluster_label=cluster_labels,
        cluster_size=np.bincount(labels, minlength=int(cluster_labels[-1])+1)[cluster_labels]))
    for j, region in enumerate(regions):
        wide['phi_'+region] = assignment.ccf[:, j]
        cluster['phi_'+region] = assignment.centers[:, j]
    long = pd.DataFrame(dict(tumor_id=tumor, mutation_id=np.repeat(ids, len(regions)),
        region_id=np.tile(regions, len(ids)), phi=assignment.ccf.ravel(),
        multiplicity_call=assignment.multiplicities.ravel()))
    tables = dict(zip(OUTPUT_SUFFIXES, (wide, cluster, long)))
    outdir = Path(outdir)
    outdir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.clipp2-publication-', dir=outdir.parent) as tmp:
        staged = Path(tmp)
        for suffix, table in tables.items():
            target = staged / (tumor+'_'+suffix)
            table.to_csv(target, sep='\t', index=False, float_format='%.17g', na_rep='NA')
            keys = {key: str for key in ('tumor_id', 'mutation_id', 'region_id') if key in table}
            candidate = pd.read_csv(target, sep='\t', dtype=keys, keep_default_na=False,
                na_values={key: ['NA'] for key in table if key not in keys}, float_precision='round_trip')
            pd.testing.assert_frame_equal(table, candidate, check_dtype=False, check_exact=True)
        outdir.mkdir(exist_ok=False)
        for suffix in tables:
            name = tumor+'_'+suffix
            os.link(staged / name, outdir / name)
    return {suffix: outdir/(tumor+'_'+suffix) for suffix in tables}
