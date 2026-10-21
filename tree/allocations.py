"""Bounded posterior allocation summaries at the selected mixture parameters.

The supplied columns contain joint regional reads and assignment validity,
with multiplicity already marginalized. These diagnostics never refit centers,
optimize weights, replace hard labels, or certify biological clonality.
"""
import math

import numpy as np
from scipy.special import logsumexp, xlogy


def allocation_diagnostics(columns, labels, centers, weights, mutation_ids, region_ids):
    """Compare hard, posterior-expected and MAP counts in public label order.

    Only 4096 rows of posterior probabilities are materialized at once. The
    returned evidence has at most 20 mutation records and q-by-q flows, not an
    N-by-q posterior matrix. Exact MAP ties choose the lowest public label;
    the separate not-maximum count ignores log-probability gaps <= 1e-12.
    """
    columns, centers, weights = (np.asarray(value, dtype=float)
                                 for value in (columns, centers, weights))
    labels = np.asarray(labels)
    if columns.ndim != 2 or not all(columns.shape):
        raise ValueError('Allocation columns must be nonempty N by q')
    n, q = columns.shape
    if (labels.shape != (n,) or labels.dtype.kind not in 'iu'
            or not np.array_equal(np.unique(labels), np.arange(q))
            or centers.ndim != 2 or centers.shape[0] != q or centers.shape[1] < 1
            or not np.isfinite(centers).all() or np.any(centers < 0) or np.any(centers > 1)
            or weights.shape != (q,) or not np.isfinite(weights).all() or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)
            or np.isnan(columns).any() or np.isposinf(columns).any()
            or not np.isfinite(columns[np.arange(n), labels]).all()
            or len(mutation_ids) != n or len(region_ids) != centers.shape[1]
            or any(not isinstance(value, str) for value in mutation_ids)
            or any(not isinstance(value, str) for value in region_ids)):
        raise ValueError('Allocation inputs must describe aligned, occupied, finite selected parameters')
    order = np.argsort(-np.linalg.norm(centers, axis=1), kind='stable')
    mapping = np.empty(q, dtype=int)
    mapping[order] = np.arange(q)
    centers, weights = centers[order], weights[order]
    hard = np.bincount(mapping[labels], minlength=q)
    expected, map_counts = np.zeros(q), np.zeros(q, dtype=np.int64)
    flow, map_flow = np.zeros((q, q)), np.zeros((q, q), dtype=np.int64)
    assigned_sums = np.zeros(q)
    worst = []
    disagreement = not_maximum = 0
    entropy_sum = assigned_log_sum = 0.
    log_weights = np.log(weights)
    for start in range(0, n, 4096):
        stop = min(n, start+4096)
        assigned = mapping[labels[start:stop]]
        # Advanced indexing creates a private batch. Remove the common read
        # offset before adding log weights, preserving small allocation terms
        # even when many regions contribute a large negative joint log value.
        joint = columns[start:stop, order]
        joint -= joint.max(axis=1)[:, None]
        joint += log_weights
        log_probability = joint - logsumexp(joint, axis=1)[:, None]
        probability = np.exp(log_probability)
        if not np.allclose(probability.sum(axis=1), 1., atol=5e-12, rtol=0):
            raise ValueError('Posterior allocation probabilities failed normalization')
        rows = np.arange(stop-start)
        calls = log_probability.argmax(axis=1)
        assigned_log = log_probability[rows, assigned]
        assigned_probability = probability[rows, assigned]
        entropy = -xlogy(probability, probability).sum(axis=1)
        expected += probability.sum(axis=0)
        map_counts += np.bincount(calls, minlength=q)
        np.add.at(flow, assigned, probability)
        np.add.at(map_flow, (assigned, calls), 1)
        np.add.at(assigned_sums, assigned, assigned_probability)
        disagreement += int(np.count_nonzero(calls != assigned))
        not_maximum += int(np.count_nonzero(assigned_log < log_probability.max(axis=1)-1e-12))
        entropy_sum += math.fsum(entropy)
        assigned_log_sum += math.fsum(assigned_log)
        indices = np.lexsort((np.asarray(mutation_ids[start:stop]), assigned_log))[:20]
        for row in indices:
            worst.append({
                'mutation_id': mutation_ids[start+row],
                'assigned_cluster_label': int(assigned[row]),
                'posterior_map_cluster_label': int(calls[row]),
                'assigned_cluster_probability': float(assigned_probability[row]),
                'assigned_cluster_log_probability': float(assigned_log[row]),
                'maximum_cluster_probability': float(probability[row].max()),
                'membership_entropy': float(entropy[row]),
                'posterior_expected_cancer_cell_fractions': (probability[row] @ centers).tolist(),
            })
        worst = sorted(worst, key=lambda item: (
            item['assigned_cluster_log_probability'], item['mutation_id']))[:20]
    if (not np.allclose(flow.sum(axis=1), hard, atol=1e-8, rtol=1e-12)
            or not np.allclose(flow.sum(axis=0), expected, atol=1e-8, rtol=1e-12)
            or not np.array_equal(map_flow.sum(axis=1), hard)
            or not np.array_equal(map_flow.sum(axis=0), map_counts)):
        raise ValueError('Posterior allocation flows failed count reconciliation')
    difference = expected - n*weights
    identity_tolerance = n*(1e-8+64*np.finfo(float).eps)+1e-12
    return {
        'allocation_diagnostics_schema_version': 1,
        'scope': 'plug_in_joint_regional_posterior_at_selected_mixture_parameters',
        'cluster_label_domain': 'public_cluster_labels',
        'region_ids': list(region_ids),
        'num_mutations': n, 'num_clusters': q,
        'hard_labels_changed': False,
        'clonality_definition': 'none; cluster zero ranks CCF-vector norm, not biological clonality',
        'mixture_weight_meaning': 'mass among analyzed mutations, not exclusive cancer-cell fraction',
        'posterior_map_tie_policy': 'lowest_public_cluster_label_on_exact_ties',
        'not_maximum_log_probability_tolerance': 1e-12,
        'hard_map_disagreement_num_mutations': disagreement,
        'hard_map_disagreement_fraction': disagreement/n,
        'hard_not_maximum_probability_num_mutations': not_maximum,
        'allocation_total_variation': float(np.abs(hard-expected).sum()/(2*n)),
        'mean_membership_entropy': entropy_sum/n,
        'mean_assigned_log_probability': assigned_log_sum/n,
        'maximum_expected_count_identity_error': float(np.abs(difference).max()),
        'expected_count_identity_tolerance': identity_tolerance,
        'expected_count_identity_passed': bool(np.abs(difference).max() <= identity_tolerance),
        'clusters': [{
            'cluster_label': k, 'cancer_cell_fractions': centers[k].tolist(),
            'hard_num_mutations': int(hard[k]),
            'posterior_expected_num_mutations': float(expected[k]),
            'posterior_map_num_mutations': int(map_counts[k]),
            'fitted_mixture_num_mutations': float(n*weights[k]),
            'mixture_weight': float(weights[k]),
            'mean_assigned_cluster_probability': float(assigned_sums[k]/hard[k]),
        } for k in range(q)],
        'flow_axes': 'rows=hard_public_label, columns=posterior_public_label',
        'posterior_allocation_flow': flow.tolist(),
        'posterior_map_flow': map_flow.tolist(),
        'worst_mutations': worst,
        'worst_mutations_limit': 20,
    }
