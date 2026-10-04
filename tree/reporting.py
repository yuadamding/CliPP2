"""Independent selected-fit checks and the compact three-table public contract."""
from dataclasses import asdict
import hashlib
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp, xlogy, xlog1py

from ..kernel.topology import partition_from_cuts

OUTPUT_SUFFIXES = ('mutation_clusters.tsv', 'cluster_centers.tsv',
                   'mutation_region_multiplicity.tsv')
ALGORITHM = 'regional_frozen_tree_conditional_mixture_v1'


def independent_log_columns(model, centers):
    """Recompute joint emissions from counts/CN, without model caches/kernels."""
    data = model.data
    ix = np.ix_(model.active_indices, model.region_indices)
    alt, depth = data.alt_counts[ix], data.total_counts[ix]
    major, total, purity = data.major_cn[ix], data.mean_total_cn[ix], data.purity[ix]
    normal, mixed = data.normal_cn[ix], data.cn_state_count[ix] > 1
    support = np.where(mixed, np.minimum(major, 4), major).astype(int)
    answer = np.zeros((model.n, len(centers)))
    for region in range(model.r):
        observed = model.observed[:, region]
        y, n = alt[observed, region], depth[observed, region]
        denominator = ((1-purity[observed, region])*normal[observed, region]
                       + purity[observed, region]*total[observed, region])
        scale = purity[observed, region] / denominator
        s = support[observed, region]
        constant = gammaln(n+1)-gammaln(y+1)-gammaln(n-y+1)-np.log(s)
        marginal = np.full((len(y), len(centers)), -np.inf)
        for m in range(1, int(s.max())+1):
            prob = scale[:, None] * m * centers[:, region][None, :]
            prob = np.where(mixed[observed, region, None], np.clip(prob, 1e-6, 1-1e-6),
                            np.minimum(prob, 1.0))
            component = xlogy(y[:, None], prob) + xlog1py((n-y)[:, None], -prob)
            component += constant[:, None]
            component[s < m] = -np.inf
            marginal = np.logaddexp(marginal, component)
        answer[observed] += marginal
    return answer


def verify_selected(model, result):
    fit = result.selected.fit
    labels = np.asarray(fit.labels)
    centers, weights = np.asarray(fit.centers), np.asarray(fit.weights)
    q = len(centers)
    if not fit.eligible or labels.shape != (model.n,) or centers.shape != (q, model.r):
        raise ValueError('Selected result has invalid eligibility or dimensions')
    if not np.issubdtype(labels.dtype, np.integer) or not np.array_equal(np.unique(labels), np.arange(q)):
        raise ValueError('Memberships must be occupied consecutive integer labels')
    phi = centers[labels]
    if not np.isfinite(phi).all() or np.any(phi < model.lower) or np.any(phi > model.upper):
        raise ValueError('Final conditional CCFs violate their original bounds')
    if weights.shape != (q,) or not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError('Published components need positive normalized fitted weights')
    cuts = tuple(e for e, (i, j) in enumerate(result.tree.edges) if labels[i] != labels[j])
    if cuts != tuple(result.selected.cuts):
        raise ValueError('Selected tree-cut provenance disagrees with its memberships')
    connected = partition_from_cuts(result.tree, cuts)
    if len(cuts)+1 != q or any(len(np.unique(connected[labels == k])) != 1 for k in range(q)):
        raise ValueError('Selected blocks must be connected in the frozen tree')
    for k in range(q):
        if not model.observed[labels == k].any(axis=0).all():
            raise ValueError('Unsupported cluster-region center cannot be published')
    columns = independent_log_columns(model, centers)
    kernel = np.exp(columns-columns.max(axis=1)[:, None])
    mass = kernel @ weights
    if not np.all(mass > 0):
        raise ValueError('Every mutation needs positive mixture probability')
    weight_score = (kernel / mass[:, None] / model.n).sum(axis=0)
    weight_gap = model.n * max(0.0, float(weight_score.max()-1.))
    active_gap = float(weight_score.max()-weight_score.min())
    # Same inherited criterion, allowing only floating-reduction roundoff.
    roundoff = 64*np.finfo(float).eps
    if weight_gap > model.n*(1e-8+roundoff) or active_gap > 1e-8+roundoff:
        raise ValueError('Independent mixture-weight optimality check failed')
    conditional = math.fsum(columns[np.arange(model.n), labels])
    mixture = float(logsumexp(columns + np.log(weights), axis=1).sum())
    penalty = (q*model.r+q-1)*math.log(model.n)
    score = -2*mixture+penalty
    for name, actual, expected in (
        ('conditional likelihood', conditional, fit.conditional_log_likelihood),
        ('joint mixture likelihood', mixture, fit.mixture_log_likelihood),
        ('complexity penalty', penalty, fit.complexity_penalty), ('score', score, fit.score)):
        if not np.isfinite(actual) or not np.isclose(actual, expected, atol=1e-7, rtol=1e-10):
            raise ValueError(f'Independent verification disagrees on {name}: {actual} != {expected}')
    bank_scores = [float(candidate['bic']) for candidate in result.candidate_bank
                   if isinstance(candidate, dict) and candidate.get('status') == 'scored'
                   and candidate.get('publication_eligible') is True]
    bank_scores += [float(candidate.score) for candidate in result.candidate_bank
                    if not isinstance(candidate, dict) and candidate.fit.eligible]
    if not bank_scores or not np.isclose(min(bank_scores), score, atol=1e-7, rtol=1e-10):
        raise ValueError('Selected fit is not the best complete-bank score')
    return {'independently_verified': True, 'conditional_log_likelihood': conditional,
            'mixture_log_likelihood': mixture, 'complexity_penalty': penalty, 'score': score,
            'weight_optimality_gap': weight_gap, 'weight_active_score_gap': active_gap,
            'complete_candidate_bank_reconciled': True,
            'num_parameters': q*model.r+q-1, 'num_mutation_vectors': model.n,
            'global_optimality_proven': False, 'joint_mixture_center_mle': False}


def result_tables(model, result, tumor_id):
    fit = result.selected.fit
    labels, centers = fit.labels, fit.centers
    # Cluster zero is largest norm, never an enforced clonal block.
    order = np.argsort(-np.linalg.norm(centers, axis=1), kind='stable')
    mapping = np.empty(len(order), dtype=int)
    mapping[order] = np.arange(len(order))
    public = mapping[labels]
    phi = centers[labels]
    wide = pd.DataFrame(dict(tumor_id=tumor_id, mutation_id=model.mutation_ids, cluster_label=public))
    cluster = pd.DataFrame(dict(tumor_id=tumor_id, cluster_label=np.arange(len(order)),
                                cluster_size=np.bincount(public)))
    for j, region in enumerate(model.region_ids):
        wide['phi_'+region] = phi[:, j]
        cluster['phi_'+region] = centers[order, j]
    ix = np.ix_(model.active_indices, model.region_indices)
    data = model.data
    mixed = data.cn_state_count[ix] > 1
    calls = model.multiplicity(phi)
    calls = np.where(~model.observed & (model.support == 1), 1, calls)
    long = pd.DataFrame(dict(tumor_id=tumor_id,
        mutation_id=np.repeat(model.mutation_ids, model.r), region_id=np.tile(model.region_ids, model.n),
        phi=phi.ravel(), major_cn=np.where(mixed, np.nan, data.major_cn[ix]).ravel(),
        minor_cn=np.where(mixed, np.nan, data.minor_cn[ix]).ravel(), multiplicity_call=calls.ravel()))
    if mixed.any():
        long['mean_total_cn'] = data.mean_total_cn[ix].ravel()
    return dict(zip(OUTPUT_SUFFIXES, (wide, cluster, long)))


def check_destination(outdir, tumor_id):
    if not tumor_id or any(c in tumor_id for c in '/\\\t\r\n'):
        raise ValueError('Tumor ID must be a nonempty single path component')
    outdir = Path(outdir)
    if outdir.exists() and (not outdir.is_dir() or any(
            p.name.startswith(tumor_id+'_') for p in outdir.iterdir())):
        raise FileExistsError('Tumor output namespace already exists; choose a new directory')


def publish(model, result, outdir, tumor_id):
    verification = verify_selected(model, result)
    check_destination(outdir, tumor_id)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tables, hashes = result_tables(model, result, tumor_id), {}
    # Per-file no-clobber publication, not a collective transaction.
    with tempfile.TemporaryDirectory(prefix='.clipp2-publication-', dir=outdir) as temporary:
        staged = Path(temporary)
        for suffix, table in tables.items():
            name = tumor_id+'_'+suffix
            table.to_csv(staged/name, sep='\t', index=False, float_format='%.17g', na_rep='NA')
            parsed = pd.read_csv(staged/name, sep='\t', dtype=str, keep_default_na=False)
            if list(parsed.columns) != list(table.columns) or len(parsed) != len(table):
                raise ValueError('Staged TSV did not round-trip its schema and row count')
            hashes[name] = hashlib.sha256((staged/name).read_bytes()).hexdigest()
        for name in hashes:
            os.link(staged/name, outdir/name)
        for name, digest in hashes.items():
            if hashlib.sha256((outdir/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Published TSV readback hash mismatch')
    return {'status': 'complete', 'algorithm': ALGORITHM, 'output_schema_version': 9, 'files': hashes,
            'verification': verification}


def exclusion_summary(model):
    report = model.data.cn_filter_report
    return {'cn_filter': None if report is None else asdict(report),
            'unidentifiable_mutations': list(model.exclusions)}
