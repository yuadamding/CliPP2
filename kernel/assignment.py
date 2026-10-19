"""Post-fit assignment diagnostics; not a global or biological certificate."""
import math

import numpy as np
from scipy.special import logsumexp


def feasible_transfer_scan(columns, labels, centers, observed, hard_zero, positive,
                           mutation_ids, *, objective='classification', score_tolerance=0.,
                           lower=None, upper=None):
    """Exhaust every individual move, with an exact profiled count correction.

    ``score_tolerance`` is on the *decrease* in selection score, hence the
    operational likelihood-gain threshold is half that value. Raw positive
    gains remain reported separately. This is not a refitted-neighbor test.
    """
    from .objectives import transfer_count_gain
    if objective not in ('classification', 'assigned_reads'):
        raise ValueError('Exact transfer closure requires an assigned objective')
    if not np.isfinite(score_tolerance) or score_tolerance < 0:
        raise ValueError('Score tolerance must be finite and nonnegative')
    columns, labels, centers = np.asarray(columns), np.asarray(labels, dtype=int), np.asarray(centers)
    observed, hard_zero, positive = (np.asarray(x, dtype=bool) for x in (observed, hard_zero, positive))
    n, q = columns.shape
    if (not n or not q or labels.shape != (n,) or centers.ndim != 2 or len(centers) != q
            or observed.shape != (n, centers.shape[1]) or hard_zero.shape != observed.shape
            or positive.shape != observed.shape or len(mutation_ids) != n
            or not np.array_equal(np.unique(labels), np.arange(q))
            or np.isnan(columns).any() or np.isposinf(columns).any()
            or not np.isfinite(centers).all() or np.any(centers < 0.) or np.any(centers > 1.)):
        raise ValueError('Transfer scan requires aligned occupied finite-domain arrays')
    if (lower is None) != (upper is None):
        raise ValueError('Transfer boxes require both lower and upper bounds')
    lower = np.zeros_like(observed, dtype=float) if lower is None else np.asarray(lower, dtype=float)
    upper = np.ones_like(observed, dtype=float) if upper is None else np.asarray(upper, dtype=float)
    if (lower.shape != observed.shape or upper.shape != observed.shape
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower > upper) or np.any(centers[labels] < lower)
            or np.any(centers[labels] > upper)):
        raise ValueError('Transfer scan requires aligned feasible original boxes')
    assigned = columns[np.arange(n), labels]
    if not np.isfinite(assigned).all():
        raise ValueError('Assigned likelihoods must be finite')
    counts = np.bincount(labels, minlength=q)
    regional = np.zeros((q, observed.shape[1]), dtype=np.int64)
    np.add.at(regional, labels, observed)
    if np.any(regional == 0):
        raise ValueError('Every source cluster needs regional observation support')
    gains = columns-assigned[:, None]
    if objective == 'classification':
        gains += transfer_count_gain(counts, np.arange(q)[:, None], np.arange(q)[None, :])[labels]
    allowed = ((~hard_zero[:, None, :] | (centers[None, :, :] == 0.))
               & (~positive[:, None, :] | (centers[None, :, :] > 0.))).all(axis=2)
    allowed &= np.isfinite(columns)
    allowed &= ((centers[None, :, :] >= lower[:, None, :])
                & (centers[None, :, :] <= upper[:, None, :])).all(axis=2)
    allowed &= (counts[labels] > 1)[:, None]
    allowed &= ((regional[labels]-observed) > 0).all(axis=1)[:, None]
    allowed[np.arange(n), labels] = False
    gains = np.where(allowed, gains, -np.inf)
    best = gains.max(axis=1)
    flat = int(np.argmax(gains))
    row, destination = np.unravel_index(flat, gains.shape)
    maximum = float(gains[row, destination])
    witness = (None if not np.isfinite(maximum) else {
        'mutation_index': int(row), 'mutation_id': str(mutation_ids[row]),
        'source': int(labels[row]), 'destination': int(destination),
        'objective_gain': maximum, 'score_decrease': 2*maximum})
    return {
        'scope': 'all_feasible_fixed_center_fixed_q_single_transfers',
        'objective': objective, 'scanned_pairs': n*(q-1),
        'feasible_pairs': int(allowed.sum()),
        'raw_positive_pair_count': int(np.count_nonzero(gains > 0.)),
        'raw_positive_mutation_count': int(np.count_nonzero(best > 0.)),
        'significant_pair_count': int(np.count_nonzero(gains > score_tolerance/2)),
        'significant_mutation_count': int(np.count_nonzero(best > score_tolerance/2)),
        'maximum_gain': max(0., maximum), 'witness': witness,
        'score_tolerance': float(score_tolerance), 'gain_tolerance': float(score_tolerance/2),
        'closed': bool(not np.any(gains > score_tolerance/2)),
        'refitted_neighborhood_exhausted': False, 'global_optimality_certified': False,
    }


def assignment_audit(columns, labels, centers, weights, observed, hard_zero,
                     positive, clonal, mutation_ids, *, objective='classification', limit=30):
    """Audit original memberships and feasible fixed-center, fixed-q transfers.

    Emission/posterior comparisons intentionally include all centers in the
    physical domain. Constraint admissibility is a separate transfer gate.
    Count gains are exact for a *single* transfer at the current counts; they
    cannot be added across simultaneous moves without updating those counts.
    """
    columns, labels = np.asarray(columns), np.asarray(labels, dtype=int)
    n, q = columns.shape
    counts = np.bincount(labels, minlength=q)
    if n == 0 or np.any(counts == 0):
        raise ValueError('Assignment diagnostics require occupied components')
    row = np.arange(n)
    assigned = columns[row, labels]
    proportions = counts / n
    weighted = columns + np.log(weights)
    logprob = weighted - logsumexp(weighted, axis=1)[:, None]
    assigned_logprob = logprob[row, labels]
    count_weighted = columns + np.log(proportions)
    best_read = columns.argmax(axis=1)
    best_class = count_weighted.argmax(axis=1)
    alternatives = count_weighted.copy()
    alternatives[row, labels] = -np.inf
    best_alternative = alternatives.argmax(axis=1)
    best_alternative_contribution = alternatives[row, best_alternative]
    observed_counts = np.zeros((q, observed.shape[1]), dtype=np.int64)
    np.add.at(observed_counts, labels, observed)
    source_empty = counts[labels] <= 1
    source_unsupported = ((observed_counts[labels] - observed) == 0).any(axis=1)
    permitted = ((~hard_zero[:, None, :] | (centers[None, :, :] == 0.))
                 & (~positive[:, None, :] | (centers[None, :, :] > 0.))).all(axis=2)
    # Positive observations with zero centers are impossible even in likelihood
    # mode and are already excluded by a nonfinite emission.
    permitted &= np.isfinite(columns)
    permitted[row, labels] = False
    def g(x):
        return 0. if x == 0 else x * math.log(x/n)
    count_gains = np.zeros((q, q))
    for a in range(q):
        for b in range(q):
            if a != b:
                count_gains[a, b] = g(counts[a]-1)+g(counts[b]+1)-g(counts[a])-g(counts[b])
    read_gains = columns-assigned[:, None]
    classification_gains = read_gains + count_gains[labels]
    feasible = permitted & ~source_empty[:, None] & ~source_unsupported[:, None]
    feasible[row, labels] = False
    gains = read_gains if objective == 'assigned_reads' else classification_gains
    feasible_gains = np.where(feasible, gains, -np.inf)
    best_feasible = feasible_gains.max(axis=1)
    best_destination = feasible_gains.argmax(axis=1)
    improvement = best_feasible > 1e-10
    count_term = float(np.sum(counts*np.log(proportions)))
    classification = float(math.fsum(assigned)+count_term)
    same_weight_class = float(math.fsum(weighted[row, labels]))
    mixture = float(math.fsum(logsumexp(weighted, axis=1)))
    loss = float(-math.fsum(assigned_logprob))
    if not np.isclose(mixture-same_weight_class, loss, atol=1e-7, rtol=1e-10):
        raise ValueError('Classification-loss identity failed')
    anchor = labels == clonal
    posterior = np.exp(assigned_logprob)
    # Return a bounded list of original IDs, never overwrite memberships with MAP.
    worst = np.argsort(assigned_logprob, kind='stable')[:limit]
    def finite_or_none(x):
        return float(x) if np.isfinite(x) else None
    records = []
    for i in worst:
        records.append({
            'mutation_id': str(mutation_ids[i]), 'assigned_cluster': int(labels[i]),
            'assigned_read_log_likelihood': float(assigned[i]),
            'best_read_cluster': int(best_read[i]), 'best_read_log_likelihood': float(columns[i, best_read[i]]),
            'assigned_component_probability': float(posterior[i]),
            'assigned_log_component_probability': float(assigned_logprob[i]),
            'best_count_weighted_cluster': int(best_class[i]),
            'best_alternative_classification_cluster': int(best_alternative[i]) if q > 1 else None,
            'best_alternative_classification_contribution': finite_or_none(best_alternative_contribution[i]),
            'best_feasible_destination': int(best_destination[i]) if np.isfinite(best_feasible[i]) else None,
            'best_feasible_objective_gain': finite_or_none(best_feasible[i]),
            'source_occupancy_blocks_transfer': bool(source_empty[i]),
            'source_regional_support_blocks_transfer': bool(source_unsupported[i]),
        })
    anchor_prob = posterior[anchor]
    return {
        'scope': 'fixed_centers_fixed_q_single_transfers_not_global_optimality',
        'component_probability_proportions': ('optimized_mixture_weights' if objective == 'legacy_mixture'
                                             else 'occupied_cluster_counts_over_N'),
        'component_probability_domain': 'physical_emission_domain_before_hard_assignment_constraints',
        'classification_proportions': proportions.tolist(),
        'classification_log_likelihood': classification, 'classification_count_term': count_term,
        'mixture_log_likelihood_at_published_proportions': mixture,
        'assigned_log_likelihood_at_published_proportions': same_weight_class,
        'classification_loss': loss, 'classification_loss_identity_verified': True,
        'assigned_not_best_read_count': int(np.count_nonzero(assigned < columns.max(axis=1)-1e-10)),
        'assigned_not_best_count_weighted_count': int(np.count_nonzero(
            count_weighted[row, labels] < count_weighted.max(axis=1)-1e-10)),
        'feasible_improving_single_transfer_count': int(improvement.sum()),
        'maximum_feasible_single_transfer_gain': max(0., float(best_feasible.max())),
        'transfer_objective': 'assigned_reads' if objective == 'assigned_reads' else 'classification',
        'transfer_is_selection_objective': objective != 'legacy_mixture',
        'source_occupancy_blocked_mutations': int(source_empty.sum()),
        'source_regional_support_blocked_mutations': int(source_unsupported.sum()),
        'destination_constraint_blocked_pairs': int((~permitted).sum()-n),
        'anchor_structurally_occupied': bool(anchor.any()),
        'anchor_mutations': int(anchor.sum()),
        'anchor_assigned_not_best_read_count': int(np.count_nonzero(
            anchor & (assigned < columns.max(axis=1)-1e-10))),
        'anchor_assigned_not_best_count_weighted_count': int(np.count_nonzero(
            anchor & (count_weighted[row, labels] < count_weighted.max(axis=1)-1e-10))),
        'anchor_feasible_improving_single_transfers': int(np.count_nonzero(anchor & improvement)),
        'anchor_assigned_component_probability_min_median_max': [float(anchor_prob.min()),
            float(np.median(anchor_prob)), float(anchor_prob.max())],
        'anchor_biological_support_certified': False,
        'unconditional_MAP_equality_required': False,
        'worst_assignment_records': records,
    }
