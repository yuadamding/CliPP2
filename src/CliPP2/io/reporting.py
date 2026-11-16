"""Independent selected-fit checks and the compact three-table public contract."""
from dataclasses import asdict
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp, xlogy, xlog1py

from ..kernel import ALGORITHM, LEGACY_SEARCH_ALGORITHMS, SCORE_DEFINITION
from ..kernel.topology import partition_from_cuts
from ..tree.candidates import validate_candidate_bank

OUTPUT_SUFFIXES = ('mutation_clusters.tsv', 'cluster_centers.tsv',
                   'mutation_region_multiplicity.tsv')
OUTPUT_SCHEMA_VERSION = 17
LEGACY_OUTPUT_SCHEMA_VERSION = 16
MANIFEST_SCHEMA = 'clipp2.file-list.v1'
VERIFICATION_SCHEMA_VERSION = 6

_VERIFY_ROWS = 4096
_VERIFY_CENTERS = 32

def _input_observed(model):
    """Read observation eligibility from input, never runtime caches."""
    data = model.data
    ix = np.ix_(model.active_indices, model.region_indices)
    depth = data.total_counts[ix]
    observed = np.isfinite(depth) & (depth > 0)
    observed &= data.count_observed[ix]
    observed &= ~data.input_excluded[ix]
    return observed


def _input_domain(model, centers):
    """Check the common [0,1] domain independently of runtime caches."""
    centers = np.asarray(centers, dtype=float)
    if (centers.ndim != 2 or not len(centers) or centers.shape[1] != model.r
            or not np.isfinite(centers).all()):
        raise ValueError('Centers must be finite nonempty q by R CCFs')
    valid = np.all((centers >= 0) & (centers <= 1), axis=1)
    return _input_observed(model), np.broadcast_to(valid, (model.n, len(centers)))


def independent_log_columns(model, centers, *, aggregation="marginal"):
    """Recompute joint reads/validity columns without model caches or kernels."""
    if aggregation not in ("marginal", "profile"):
        raise ValueError("Unknown independent emission aggregation")
    centers = np.asarray(centers, dtype=float)
    observed_rows, allowed = _input_domain(model, centers)
    data = model.data
    ix = np.ix_(model.active_indices, model.region_indices)
    alt, depth = data.alt_counts[ix], data.total_counts[ix]
    major, total, purity = data.major_cn[ix], data.total_cn[ix], data.purity[ix]
    normal = data.normal_cn[ix]
    bounds = model.multiplicity_support
    if bounds is None:
        minimum = np.ones_like(major, dtype=np.int64)
        maximum = major.astype(np.int64)
    else:
        minimum, maximum = bounds.for_data(data)
        minimum, maximum = np.asarray(minimum)[ix], np.asarray(maximum)[ix]
    if model.max_multiplicity_difference == 1:
        from ..kernel.joint import joint_log_columns
        scale = purity / ((1-purity)*normal+purity*total)
        valid = np.all((centers >= 0) & (centers <= 1), axis=1)
        answer = np.full((model.n, len(centers)), -np.inf)
        if valid.any():
            answer[:, valid] = joint_log_columns(alt, depth, scale, minimum, maximum,
                observed_rows, centers[valid], max_multiplicity_difference=1, aggregation=aggregation)
        return answer
    support = maximum-minimum+1
    answer = np.zeros((model.n, len(centers)))
    for region in range(model.r):
        observed = observed_rows[:, region]
        if not observed.any():
            continue
        y, n = alt[observed, region], depth[observed, region]
        denominator = ((1-purity[observed, region])*normal[observed, region]
                       + purity[observed, region]*total[observed, region])
        scale = purity[observed, region] / denominator
        s = support[observed, region]
        constant = gammaln(n+1)-gammaln(y+1)-gammaln(n-y+1)
        if aggregation == "marginal":
            constant -= np.log(s)
        original_rows = np.flatnonzero(observed)
        for size in np.unique(s):
            bucket = np.flatnonzero(s == size)
            for start in range(0, len(bucket), _VERIFY_ROWS):
                rows = bucket[start:start+_VERIFY_ROWS]
                source = original_rows[rows]
                for first in range(0, len(centers), _VERIFY_CENTERS):
                    cols = slice(first, first+_VERIFY_CENTERS)
                    marginal = np.full((len(rows), len(centers[cols])), -np.inf)
                    # Preserve increasing multiplicity and regional summation
                    # order for every row; never evaluate its unsupported m's.
                    for offset in range(int(size)):
                        m = minimum[source, region, None]+offset
                        prob = scale[rows, None] * m * centers[cols, region][None, :]
                        prob = np.minimum(prob, 1.0)
                        component = (xlogy(y[rows, None], prob)
                                     + xlog1py((n-y)[rows, None], -prob))
                        component += constant[rows, None]
                        marginal = (np.logaddexp(marginal, component) if aggregation == "marginal"
                                    else np.maximum(marginal, component))
                    answer[source, cols] += marginal
    return np.where(allowed, answer, -np.inf)


def _verify_retained_bank(result, score):
    """Audit the retained states and their compact refit-attempt links."""
    links = result.attempt_links
    if links is None:
        raise ValueError('Candidate bank is missing its compact refit-attempt links')
    retained = validate_candidate_bank(result.tree, result.candidate_bank, result.selected, links)
    if not np.isclose(retained.score, score, atol=1e-7, rtol=1e-10):
        raise ValueError('Selected fit is not the best complete-bank score')


def _verify_fitted_numerics(model, fit, *, eligible=True):
    """Independent read/box/weight/score checks for one exact completed state."""
    labels = np.asarray(fit.labels)
    centers, weights = np.asarray(fit.centers), np.asarray(fit.weights)
    q = len(centers)
    if bool(fit.eligible) != eligible or labels.shape != (model.n,) or centers.shape != (q, model.r):
        raise ValueError('Fitted result has invalid eligibility or dimensions')
    if not np.issubdtype(labels.dtype, np.integer) or not np.array_equal(np.unique(labels), np.arange(q)):
        raise ValueError('Memberships must be occupied consecutive integer labels')
    observed, allowed = _input_domain(model, centers)
    if not allowed.any(axis=1).all():
        raise ValueError('A mutation has no admissible center in its original bounds')
    if not allowed[np.arange(model.n), labels].all():
        raise ValueError('Final conditional CCFs violate their original bounds')
    if (weights.shape != (q,) or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, atol=1e-12, rtol=0)
            or (eligible and np.any(weights == 0))):
        raise ValueError('Completed components need valid normalized fitted weights')
    if not eligible and (fit.reason != 'zero_mixture_weight' or not np.any(weights == 0)):
        raise ValueError('Completed ineligible neighbor must have zero mixture weight')
    for k in range(q):
        if not observed[labels == k].any(axis=0).all():
            raise ValueError('Unsupported cluster-region center cannot be published')
    columns = independent_log_columns(model, centers)
    offset = columns.max(axis=1)
    if not np.isfinite(offset).all():
        raise ValueError('A mutation has zero read likelihood at every admissible center')
    active = weights > 0
    log_mass = logsumexp(columns[:, active] + np.log(weights[active]), axis=1)
    if not np.isfinite(log_mass).all():
        raise ValueError('Every mutation needs finite mixture probability')
    # Log-space ratios also cover finite support whose active components would
    # underflow after row-max scaling. Zero-weight ineligible neighbors still
    # need a full inactive-component KKT test and an active-component tie test.
    weight_score = np.exp(columns-log_mass[:, None]).mean(axis=0)
    weight_gap = model.n * max(0.0, float(weight_score.max()-1.))
    active_gap = float(weight_score.max()-weight_score[active].min())
    # Same inherited criterion, allowing only floating-reduction roundoff.
    roundoff = 64*np.finfo(float).eps
    if weight_gap > model.n*(1e-8+roundoff) or active_gap > 1e-8+roundoff:
        raise ValueError('Independent mixture-weight optimality check failed')
    conditional = math.fsum(columns[np.arange(model.n), labels])
    method = model.fit_multiplicity
    fitting_ll = (conditional if method == "marginal" else math.fsum(
        independent_log_columns(model, centers, aggregation="profile")[np.arange(model.n), labels]))
    if (fit.fit_objective_id != model.method_identity["fit_objective_id"]
            or fit.selection_criterion_id != model.method_identity["selection_criterion_id"]):
        raise ValueError("Fitted result has a different fitting/selection identity")
    mixture = float(log_mass.sum())
    penalty = (q*model.r+q-1)*math.log(model.n)
    score = -2*mixture+penalty
    for name, actual, expected in (
        ('fitting likelihood', fitting_ll, fit.fitting_log_likelihood),
        ('conditional likelihood', conditional, fit.conditional_log_likelihood),
        ('joint mixture likelihood', mixture, fit.mixture_log_likelihood),
        ('complexity penalty', penalty, fit.complexity_penalty), ('score', score, fit.score)):
        if not np.isfinite(actual) or not np.isclose(actual, expected, atol=1e-7, rtol=1e-10):
            raise ValueError(f'Independent verification disagrees on {name}: {actual} != {expected}')
    if not eligible:
        diagnostics = fit.diagnostics
        if diagnostics.get('rejection_kind') is not None:
            raise ValueError('Ineligible neighbor lacks a completed numerical fit')
        for field, limit in (('weight_optimality_gap', model.n*(1e-8+roundoff)),
                             ('weight_active_score_gap', 1e-8+roundoff)):
            value = diagnostics.get(field)
            if value is None or not np.isfinite(value) or value < 0 or value > limit:
                raise ValueError('Ineligible neighbor lacks converged weight diagnostics')
    validity = float(np.log(allowed @ weights).sum())
    return {'fitting_log_likelihood': fitting_ll, **dict(model.method_identity),
            'conditional_log_likelihood': conditional,
            'hard_partition_log_likelihood': conditional,
            'mixture_log_likelihood': mixture, 'complexity_penalty': penalty, 'score': score,
            'score_definition': SCORE_DEFINITION,
            'forbidden_assignment_pairs': int((~allowed).sum()),
            'assignment_validity_log_probability': validity,
            'validity_conditioned_mixture_log_likelihood': mixture-validity,
            'weight_optimality_gap': weight_gap, 'weight_active_score_gap': active_gap,
            'num_parameters': q*model.r+q-1, 'num_mutation_vectors': model.n,
            'global_optimality_proven': False, 'joint_mixture_center_mle': False}


def verify_selected(model, result):
    """Independently verify the selected fit and candidate-bank selection.

    The hard-partition likelihood and assignment-validity diagnostic do not
    replace the joint mixture score. This does not certify a local or global
    optimum over memberships.
    """
    if getattr(result, "algorithm", None) == ALGORITHM:
        from ..kernel.mixture_verification import verify_marginal_mixture
        if (result.selected.fit.mixture is not result.search.best
                or not 0 <= result.selected.state_index < len(result.search.candidates)
                or result.search.candidates[result.selected.state_index] is not result.search.best):
            raise ValueError("Published mixture differs from the retained selected search state")
        verified = verify_marginal_mixture(model, result.search.best,
                                          candidates=result.search.candidates)
        fit = result.selected.fit
        for key in ("fitting_log_likelihood", "conditional_log_likelihood", "complexity_penalty", "score"):
            if not np.isclose(verified[key], getattr(fit, key), atol=1e-7, rtol=1e-10):
                raise ValueError(f"Independent soft-mixture verification disagrees on {key}")
        return dict(verified, **dict(fit.method_identity), verification_schema_version=VERIFICATION_SCHEMA_VERSION,
                    independently_verified=True, method_identity=dict(fit.method_identity),
                    membership_kind="soft_mixture", complete_candidate_bank_reconciled=True,
                    selected_fitted_state_retained=True, local_optimality_certified=False,
                    global_optimality_proven=False, hard_refits=0)
    selected = result.selected
    fit, labels = selected.fit, selected.fit.labels
    q = len(fit.centers)
    verified = _verify_fitted_numerics(model, fit)
    _verify_retained_bank(result, verified['score'])
    cuts = tuple(e for e, (i, j) in enumerate(result.tree.edges) if labels[i] != labels[j])
    kind = selected.membership_kind
    if kind == 'tree':
        connected = partition_from_cuts(result.tree, cuts)
        if (cuts != tuple(selected.cuts) or len(cuts)+1 != q
                or any(len(np.unique(connected[labels == k])) != 1 for k in range(q))):
            raise ValueError('Selected blocks must be connected in the frozen tree')
    elif kind != 'general' or list(dict.fromkeys(labels.tolist())) != list(range(q)):
        raise ValueError('Selected general membership must use canonical first-occurrence labels')
    return {'verification_schema_version': VERIFICATION_SCHEMA_VERSION,
            'independently_verified': True, **verified, 'membership_kind': kind,
            'connected_on_proposal_tree': len(cuts)+1 == q,
            'complete_candidate_bank_reconciled': True, 'selected_fitted_state_retained': True,
            'local_optimality_certified': False}


def public_label_order(fit):
    """The single public-label convention used by tables and replay receipts."""
    # Cluster zero is largest norm, never an enforced clonal block.
    order = np.argsort(-np.linalg.norm(fit.centers, axis=1), kind='stable')
    mapping = np.empty(len(order), dtype=int)
    mapping[order] = np.arange(len(order))
    return order, mapping[fit.labels]


def result_tables(model, result, tumor_id):
    fit = result.selected.fit
    labels, centers = fit.labels, fit.centers
    order, public = public_label_order(fit)
    phi = centers[labels]
    wide = pd.DataFrame(dict(tumor_id=tumor_id, mutation_id=model.mutation_ids, cluster_label=public))
    cluster = pd.DataFrame(dict(tumor_id=tumor_id, cluster_label=np.arange(len(order)),
                                cluster_size=np.bincount(public, minlength=len(order))))
    calls = model.multiplicity(phi)
    calls = np.where(~model.observed & (model.support == 1), model.multiplicity_min, calls)
    calls = np.where(phi == 0, np.nan, calls)
    report = model.data.cn_filter_report
    if report is not None and report.records:
        mutations = {name: i for i, name in enumerate(model.mutation_ids)}
        regions = {name: j for j, name in enumerate(model.region_ids)}
        for record in report.records:
            i, j = mutations.get(record.mutation_id), regions.get(record.sample_id)
            if i is not None and j is not None:
                phi[i, j] = calls[i, j] = np.nan
    for j, region in enumerate(model.region_ids):
        wide['phi_'+region] = phi[:, j]
        cluster['phi_'+region] = centers[order, j]
    long = pd.DataFrame(dict(tumor_id=tumor_id,
        mutation_id=np.repeat(model.mutation_ids, model.r), region_id=np.tile(model.region_ids, model.n),
        phi=phi.ravel(), multiplicity_call=calls.ravel()))
    return dict(zip(OUTPUT_SUFFIXES, (wide, cluster, long)))


def check_destination(outdir, tumor_id):
    if not tumor_id or any(c in tumor_id for c in '/\\\t\r\n'):
        raise ValueError('Tumor ID must be a nonempty single path component')
    outdir = Path(outdir)
    if outdir.exists() and (not outdir.is_dir() or any(
            p.name.startswith(tumor_id+'_') for p in outdir.iterdir())):
        raise FileExistsError('Tumor output namespace already exists; choose a new directory')


def publish(model, result, outdir, tumor_id, *, progress=None):
    if progress is not None:
        progress.start_phase('verification')
    started = perf_counter()
    verification = verify_selected(model, result)
    verification_seconds = perf_counter()-started
    if progress is not None:
        progress.emit(complete=True)
        progress.start_phase('output')
    started = perf_counter()
    check_destination(outdir, tumor_id)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tables, files = result_tables(model, result, tumor_id), []
    # Per-file no-clobber publication, not a collective transaction.
    with tempfile.TemporaryDirectory(prefix='.clipp2-publication-', dir=outdir) as temporary:
        staged = Path(temporary)
        for suffix, table in tables.items():
            name = tumor_id+'_'+suffix
            table.to_csv(staged/name, sep='\t', index=False, float_format='%.17g', na_rep='NA')
            files.append(name)
        for name in files:
            os.link(staged/name, outdir/name)
    output_seconds = perf_counter()-started
    if progress is not None:
        progress.emit(complete=True)
    return {'status': 'complete', 'algorithm': getattr(result, 'algorithm', LEGACY_SEARCH_ALGORITHMS[0]),
            'output_schema_version': (OUTPUT_SCHEMA_VERSION if getattr(result, 'algorithm', None) == ALGORITHM
                                      else LEGACY_OUTPUT_SCHEMA_VERSION),
            'manifest_schema': MANIFEST_SCHEMA, 'files': files,
            'verification_seconds': verification_seconds, 'output_seconds': output_seconds,
            'verification': verification}


def exclusion_summary(model):
    report = model.data.cn_filter_report
    return {'cn_filter': None if report is None else asdict(report),
            'unidentifiable_mutations': list(model.exclusions)}
