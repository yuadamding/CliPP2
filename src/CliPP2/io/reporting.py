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

from .._source import file_sha256
from ..config import validate_audit_scope
from ..kernel import ALGORITHM, SCORE_DEFINITION
from ..kernel.model import FitResult
from ..kernel.topology import canonical_partition, partition_from_cuts
from ..tree.candidates import (CandidateBank, RefitAttempt, candidate_record, evidence_attempt_links,
                               score_key, validate_candidate_bank)
from ..tree.membership_audit import (AUDIT_POLICY, MembershipAudit, SingleMove,
                                     fitted_state_fingerprint)
from ..tree.proposals import SCORE_ATOL, SCORE_RTOL

OUTPUT_SUFFIXES = ('mutation_clusters.tsv', 'cluster_centers.tsv',
                   'mutation_region_multiplicity.tsv')
OUTPUT_SCHEMA_VERSION = 13
VERIFICATION_SCHEMA_VERSION = 4

_VERIFY_ROWS = 4096
_VERIFY_CENTERS = 32
_VERIFY_CACHE_BYTES = 16 * 1024 * 1024

def _input_boxes(model):
    """Read support and all-member boxes from input, never runtime caches."""
    data = model.data
    ix = np.ix_(model.active_indices, model.region_indices)
    depth = data.total_counts[ix]
    observed = np.isfinite(depth) & (depth > 0)
    if data.count_observed is not None:
        observed &= np.asarray(data.count_observed, dtype=bool)[ix]
    mixed = data.cn_state_count[ix] > 1
    return observed, np.where(mixed, 1e-6, 0.), np.where(mixed, data.phi_upper[ix], 1.)


def _input_domain(model, centers):
    """Rebuild observations and closed boxes independently of runtime caches."""
    centers = np.asarray(centers, dtype=float)
    if (centers.ndim != 2 or not len(centers) or centers.shape[1] != model.r
            or not np.isfinite(centers).all()):
        raise ValueError('Centers must be finite nonempty q by R CCFs')
    observed, lower, upper = _input_boxes(model)
    allowed = np.ones((model.n, len(centers)), dtype=bool)
    # Keep validity independent of fitting caches without an N-by-q-by-R
    # temporary. Missing observations still impose their original bounds.
    for region in range(model.r):
        for start in range(0, model.n, _VERIFY_ROWS):
            rows = slice(start, start+_VERIFY_ROWS)
            for first in range(0, len(centers), _VERIFY_CENTERS):
                cols = slice(first, first+_VERIFY_CENTERS)
                values = centers[cols, region][None, :]
                allowed[rows, cols] &= ((values >= lower[rows, region, None])
                                        & (values <= upper[rows, region, None]))
    return observed, allowed


def independent_log_columns(model, centers):
    """Recompute joint reads/validity columns without model caches or kernels."""
    centers = np.asarray(centers, dtype=float)
    observed_rows, allowed = _input_domain(model, centers)
    data = model.data
    ix = np.ix_(model.active_indices, model.region_indices)
    alt, depth = data.alt_counts[ix], data.total_counts[ix]
    major, total, purity = data.major_cn[ix], data.mean_total_cn[ix], data.purity[ix]
    normal, mixed = data.normal_cn[ix], data.cn_state_count[ix] > 1
    bounds = getattr(model, 'multiplicity_support', None)
    if bounds is None:
        minimum = np.ones_like(major, dtype=np.int64)
        maximum = np.where(mixed, np.minimum(major, 4), major).astype(np.int64)
    else:
        minimum, maximum = bounds.for_data(data)
        minimum, maximum = np.asarray(minimum)[ix], np.asarray(maximum)[ix]
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
        constant = gammaln(n+1)-gammaln(y+1)-gammaln(n-y+1)-np.log(s)
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
                        prob = np.where(mixed[source, region, None],
                                        np.clip(prob, 1e-6, 1-1e-6), np.minimum(prob, 1.0))
                        component = (xlogy(y[rows, None], prob)
                                     + xlog1py((n-y)[rows, None], -prob))
                        component += constant[rows, None]
                        marginal = np.logaddexp(marginal, component)
                    answer[source, cols] += marginal
    return np.where(allowed, answer, -np.inf)


def _verify_retained_bank(model, result, score):
    """Audit actual retained states; the optional research codec is not required."""
    links = getattr(result, 'attempt_links', None)
    if result.diagnostics.get('diagnostics_schema_version', 0) >= 2 and links is None:
        raise ValueError('Candidate bank is missing its compact refit-attempt links')
    saved = result.diagnostics.get('candidate_evidence')
    if saved is not None:
        # An explicitly requested research export must still describe the current
        # objects. Compare one record at a time, never rebuild a second bank.
        tree = result.tree
        expected_tree = {'identity': tree.identity, 'n': tree.n, 'edges': tree.edges.tolist(),
                         'kind': tree.kind, 'mutation_ids': list(tree.mutation_ids)}
        selected_index = next((i for i, value in enumerate(result.candidate_bank)
                               if value is result.selected), None)
        if (saved.get('version') != 1 or saved.get('tree') != expected_tree
                or saved.get('selected_state_index') != selected_index
                or len(saved.get('states', ())) != len(result.candidate_bank)
                or any(record != candidate_record(candidate)
                       for record, candidate in zip(saved['states'], result.candidate_bank))):
            raise ValueError('Persisted candidate evidence disagrees with the retained bank')
        attempts = saved.get('refit_attempts', ())
        exported_links = evidence_attempt_links(tree, attempts, result.selected.fit.centers.shape[1])
        if links is not None and links != exported_links:
            raise ValueError('Persisted candidate evidence disagrees with retained attempt links')
        if links is None and attempts:
            links = exported_links
    retained = validate_candidate_bank(result.tree, result.candidate_bank, result.selected, links)
    if not np.isclose(retained.score, score, atol=1e-7, rtol=1e-10):
        raise ValueError('Selected fit is not the best complete-bank score')


def _independent_columns(model, centers, cache):
    """Bound verifier-owned columns while retaining the current call's handles."""
    keys = [np.asarray(center, dtype=float).tobytes() for center in centers]
    available = {}
    for key in keys:
        if key in cache:
            # Dict insertion order supplies an exact LRU without a new numerical
            # dependency. Active columns remain alive if later inserts evict them.
            available[key] = cache.pop(key)
            cache[key] = available[key]
    missing = dict((key, center) for key, center in zip(keys, centers) if key not in available)
    if missing:
        computed = independent_log_columns(model, np.asarray(list(missing.values())))
        for index, key in enumerate(missing):
            available[key] = computed[:, index].copy()
            cache[key] = available[key]
    maximum_columns = _VERIFY_CACHE_BYTES // (8*model.n)
    while len(cache) > maximum_columns:
        cache.pop(next(iter(cache)))
    return np.column_stack([available[key] for key in keys])


def _verify_fitted_numerics(model, fit, *, eligible=True, column_cache=None):
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
    columns = (independent_log_columns(model, centers) if column_cache is None else
               _independent_columns(model, centers, column_cache))
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
    mixture = float(log_mass.sum())
    penalty = (q*model.r+q-1)*math.log(model.n)
    score = -2*mixture+penalty
    for name, actual, expected in (
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
    return {'conditional_log_likelihood': conditional,
            'hard_partition_log_likelihood': conditional,
            'mixture_log_likelihood': mixture, 'complexity_penalty': penalty, 'score': score,
            'score_definition': SCORE_DEFINITION,
            'forbidden_assignment_pairs': int((~allowed).sum()),
            'assignment_validity_log_probability': validity,
            'validity_conditioned_mixture_log_likelihood': mixture-validity,
            'weight_optimality_gap': weight_gap, 'weight_active_score_gap': active_gap,
            'num_parameters': q*model.r+q-1, 'num_mutation_vectors': model.n,
            'global_optimality_proven': False, 'joint_mixture_center_mle': False}


def _neighbor_exclusion(model, labels, observed, lower, upper):
    """Independent structural check in canonical group/region order."""
    for group in range(int(labels.max())+1):
        members = labels == group
        for region in range(model.r):
            if not observed[members, region].any():
                return 'unsupported_cluster_region:'+model.region_ids[region]
            if lower[members, region].max() > upper[members, region].min():
                return 'empty_cluster_region_box:'+model.region_ids[region]
    return None


def _verify_membership_audits(model, result, selected_numerics, column_cache):
    """Verify complete terminal neighborhoods, exact attempt starts and scores."""
    audits = getattr(result, 'membership_audits', None)
    attempts = getattr(result, 'attempt_outcomes', None)
    if not isinstance(audits, tuple) or not isinstance(attempts, tuple):
        raise ValueError('Publication requires retained terminal membership audits and outcomes')
    links = result.attempt_links
    if links is None or len(attempts) != len(links):
        raise ValueError('Membership outcomes disagree with retained refit-attempt links')
    keys = set()
    for attempt, link in zip(attempts, links):
        if (not isinstance(attempt, RefitAttempt) or not isinstance(attempt.key, tuple)
                or len(attempt.key) != 3 or attempt.key in keys
                or (attempt.key[0], attempt.outcome == 'eligible', attempt.state_index) != link):
            raise ValueError('Membership outcomes disagree with retained refit-attempt links')
        keys.add(attempt.key)
    bank = result.candidate_bank
    best = {}
    for candidate in bank:
        q = len(candidate.fit.centers)
        if q not in best or score_key(candidate) < score_key(best[q]):
            best[q] = candidate
    scope = result.audit_scope
    validate_audit_scope(scope)
    if result.diagnostics.get('audit_scope', scope) != scope:
        raise ValueError('Membership audit scope disagrees with search diagnostics')
    required = set(best) if scope == 'all' else {len(result.selected.fit.centers)}
    if len(audits) != len(required):
        raise ValueError('Terminal membership audits do not cover the requested scope')
    numerical_policy = CandidateBank.numerical_policy
    numerical_identity = (model.backend.policy, model.backend.identity)
    verified = {fitted_state_fingerprint(model, result.tree, result.selected, numerical_policy):
                selected_numerics}
    observed, lower, upper = _input_boxes(model)
    seen, counts = set(), []
    for audit in audits:
        if (not isinstance(audit, MembershipAudit) or audit.complete is not True
                or type(audit.occupied_q) is not int or audit.occupied_q not in required
                or audit.occupied_q in seen or type(audit.mutation_count) is not int
                or audit.mutation_count != model.n or not isinstance(audit.moves, tuple)):
            raise ValueError('Invalid or incomplete terminal membership audit')
        q = audit.occupied_q
        parent = best[q]
        fingerprint = fitted_state_fingerprint(model, result.tree, parent, numerical_policy)
        if (type(audit.state_index) is not int or audit.state_index != parent.state_index
                or audit.state_fingerprint != fingerprint or audit.model_identity != model.identity
                or audit.numerical_policy != numerical_policy
                or audit.numerical_identity != numerical_identity):
            raise ValueError('Terminal membership audit has stale fitted-state or numerical identity')
        if fingerprint not in verified:
            verified[fingerprint] = _verify_fitted_numerics(model, parent.fit, column_cache=column_cache)
        parent_score = verified[fingerprint]['score']
        labels, centers = parent.fit.labels, parent.fit.centers
        sizes = np.bincount(labels, minlength=q)
        if len(audit.moves) != model.n*(q-1):
            raise ValueError('Terminal membership audit has incomplete move coverage')
        row_counts = {'occupied_q': q, 'state_index': parent.state_index,
                      'examined_pairs': len(audit.moves), 'structural_exclusions': 0,
                      'eligible_neighbors': 0, 'completed_ineligible_neighbors': 0,
                      'unresolved_evaluations': 0, 'improving_eligible_neighbors': 0}
        moves = iter(audit.moves)
        for row in range(model.n):
            for destination in range(q):
                if destination == labels[row]:
                    continue
                move = next(moves)
                if (not isinstance(move, SingleMove) or type(move.row) is not int
                        or type(move.destination) is not int or (move.row, move.destination) != (row, destination)):
                    raise ValueError('Terminal membership audit has duplicate or misordered moves')
                if sizes[labels[row]] == 1:
                    reason, canonical, initial = 'empty_source_cluster', None, None
                else:
                    proposed = labels.copy()
                    proposed[row] = destination
                    canonical, initial = canonical_partition(proposed, centers)
                    reason = _neighbor_exclusion(model, canonical, observed, lower, upper)
                if reason is not None:
                    if (move.outcome != 'structural_exclusion' or move.reason != reason
                            or move.attempt_index is not None):
                        raise ValueError('Membership audit structural exclusion is not justified by input')
                    row_counts['structural_exclusions'] += 1
                    continue
                if (type(move.attempt_index) is not int
                        or not 0 <= move.attempt_index < len(attempts)):
                    raise ValueError('Feasible membership move is missing its exact refit attempt')
                attempt = attempts[move.attempt_index]
                expected = (('general', canonical.tobytes()), initial.tobytes(), numerical_policy)
                if (not isinstance(attempt, RefitAttempt) or attempt.key != expected
                        or attempt.outcome != move.outcome or attempt.reason != move.reason):
                    raise ValueError('Membership audit has mismatched refit identity or outcome')
                if attempt.outcome == 'eligible':
                    index = attempt.state_index
                    if (type(index) is not int or not 0 <= index < len(bank)
                            or attempt.fit is not None or attempt.reason is not None
                            or bank[index].fit.membership.key != expected[0]):
                        raise ValueError('Eligible membership neighbor lacks its retained exact state')
                    neighbor = bank[index]
                    key = fitted_state_fingerprint(model, result.tree, neighbor, numerical_policy)
                    if key not in verified:
                        verified[key] = _verify_fitted_numerics(model, neighbor.fit, column_cache=column_cache)
                    neighbor_score = verified[key]['score']
                    tolerance = SCORE_ATOL+SCORE_RTOL*max(abs(parent_score), abs(neighbor_score))
                    if parent_score-neighbor_score > tolerance:
                        raise ValueError('Terminal membership audit contains an improving eligible neighbor')
                    row_counts['eligible_neighbors'] += 1
                elif attempt.outcome == 'completed_ineligible':
                    fit = attempt.fit
                    if (attempt.state_index is not None or attempt.reason != 'zero_mixture_weight'
                            or not isinstance(fit, FitResult) or fit.membership.key != expected[0]):
                        raise ValueError('Ineligible neighbor lacks its completed exact refit')
                    # Multiple ordered moves may share one canonical refit.
                    key = ('ineligible', move.attempt_index)
                    if key not in verified:
                        verified[key] = _verify_fitted_numerics(model, fit, eligible=False, column_cache=column_cache)
                    row_counts['completed_ineligible_neighbors'] += 1
                else:
                    raise ValueError('Feasible membership neighbor has unresolved numerical evaluation')
        seen.add(q)
        counts.append(row_counts)
    counts.sort(key=lambda row: row['occupied_q'])
    selected = next(row for row in counts if row['state_index'] == result.selected.state_index)
    return {'complete': True, 'scope': scope, 'policy': AUDIT_POLICY, **selected,
            'selected_count_certified': True, 'all_represented_counts_certified': seen == set(best),
            'score_atol': SCORE_ATOL, 'score_rtol': SCORE_RTOL,
            'represented_cluster_counts': sorted(best), 'per_q': counts}


def verify_selected(model, result):
    """Independently verify bank selection and the requested terminal neighborhoods.

    The hard-partition likelihood and assignment-validity diagnostic do not
    replace the joint mixture score. Local closure is not global optimality.
    """
    selected = result.selected
    fit, labels = selected.fit, selected.fit.labels
    q = len(fit.centers)
    column_cache = {}
    verified = _verify_fitted_numerics(model, fit, column_cache=column_cache)
    _verify_retained_bank(model, result, verified['score'])
    cuts = tuple(e for e, (i, j) in enumerate(result.tree.edges) if labels[i] != labels[j])
    kind = selected.membership_kind
    if kind == 'tree':
        connected = partition_from_cuts(result.tree, cuts)
        if (cuts != tuple(selected.cuts) or len(cuts)+1 != q
                or any(len(np.unique(connected[labels == k])) != 1 for k in range(q))):
            raise ValueError('Selected blocks must be connected in the frozen tree')
    elif kind != 'general' or list(dict.fromkeys(labels.tolist())) != list(range(q)):
        raise ValueError('Selected general membership must use canonical first-occurrence labels')
    audit = _verify_membership_audits(model, result, verified, column_cache)
    return {'verification_schema_version': VERIFICATION_SCHEMA_VERSION,
            'independently_verified': True, **verified, 'membership_kind': kind,
            'connected_on_proposal_tree': len(cuts)+1 == q,
            'complete_candidate_bank_reconciled': True, 'selected_fitted_state_retained': True,
            'independent_cached_columns': len(column_cache), 'membership_audit': audit}


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
    calls = model.multiplicity(phi)
    calls = np.where(~model.observed & (model.support == 1), model.multiplicity_min, calls)
    report = getattr(getattr(model, 'data', None), 'cn_filter_report', None)
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
            hashes[name] = file_sha256(staged/name)
        for name in hashes:
            os.link(staged/name, outdir/name)
        for name, digest in hashes.items():
            if file_sha256(outdir/name) != digest:
                raise ValueError('Published TSV readback hash mismatch')
    output_seconds = perf_counter()-started
    if progress is not None:
        progress.emit(complete=True)
    return {'status': 'complete', 'algorithm': ALGORITHM,
            'output_schema_version': OUTPUT_SCHEMA_VERSION, 'files': hashes,
            'verification_seconds': verification_seconds, 'output_seconds': output_seconds,
            'verification': verification}


def exclusion_summary(model):
    report = model.data.cn_filter_report
    return {'cn_filter': None if report is None else asdict(report),
            'unidentifiable_mutations': list(model.exclusions)}
