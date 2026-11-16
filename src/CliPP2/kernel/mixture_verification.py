"""Independent final checks for soft mixtures, without hard-group assumptions.

No optimizer, runtime emission cache, posterior kernel, or hard refitter is
used here. Positive components may lack MAP members or regional hard-group
coverage; neither condition changes the observed-mixture model.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp

from .joint import joint_log_columns


def _parameters(model, fit):
    centers, weights = np.asarray(fit.centers), np.asarray(fit.weights)
    if (centers.ndim != 2 or centers.shape[1] != model.r or not len(centers)
            or not np.isfinite(centers).all() or np.any((centers < 0) | (centers > 1))):
        raise ValueError("Soft-mixture centers require finite K-by-R CCFs in [0, 1]")
    if (weights.shape != (len(centers),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError("Soft-mixture weights must lie on the probability simplex")
    positive = weights > 0
    q = int(positive.sum())
    if len(np.unique(centers[positive], axis=0)) != q:
        raise ValueError("Exactly duplicate positive mixture centers must be consolidated")
    if not bool(fit.converged) or fit.positive_components != q:
        raise ValueError("Soft-mixture convergence or positive-component count is inconsistent")
    return centers, weights, positive, q


def _independent_columns(model, centers):
    """Reconstruct reads, mask, CN scaling, and support from input authorities."""
    data = model.data
    index = np.ix_(model.active_indices, model.region_indices)
    alt, depth = data.alt_counts[index], data.total_counts[index]
    observed = (np.isfinite(depth) & (depth > 0) & data.count_observed[index]
                & ~data.input_excluded[index])
    if alt.shape != (model.n, model.r) or not observed.any(axis=1).all():
        raise ValueError("Soft-mixture input population has invalid dimensions or no observed reads")
    purity, normal, total = data.purity[index], data.normal_cn[index], data.total_cn[index]
    denominator = (1-purity)*normal+purity*total
    scaling = np.zeros_like(depth, dtype=float)
    if np.any(denominator[observed] <= 0):
        raise ValueError("Observed copy number requires positive total read scaling")
    scaling[observed] = purity[observed]/denominator[observed]
    if model.multiplicity_support is None:
        minimum = np.ones_like(alt, dtype=np.int64)
        maximum = data.major_cn[index]
    else:
        minimum, maximum = model.multiplicity_support.for_data(data)
        minimum, maximum = np.asarray(minimum)[index], np.asarray(maximum)[index]
    return joint_log_columns(alt, depth, scaling, minimum, maximum, observed, centers,
                             max_multiplicity_difference=model.max_multiplicity_difference)


def verify_marginal_mixture(model, fit, *, candidates=None):
    """Verify one final mixture and optionally reconcile its retained score bank.

    The selected state's emissions, posterior matrix, likelihood, BIC and
    weight KKT residuals are independently reconstructed. Bank reconciliation
    checks every retained candidate's parameter count and score arithmetic,
    plus exact selected-state retention; it does not repeat numerical emission
    reconstruction for every search trajectory or certify global optimality.
    """
    if model.fit_multiplicity != "marginal" or model.n < 1 or model.r < 1:
        raise ValueError("Soft-mixture verification requires a nonempty marginal model")
    centers, weights, positive, q = _parameters(model, fit)
    columns = _independent_columns(model, centers)
    if np.isnan(columns).any() or np.isposinf(columns).any():
        raise ValueError("Independent mixture emissions must be finite or negative infinity")
    log_weights = np.full(len(weights), -np.inf)
    log_weights[positive] = np.log(weights[positive])
    joint = columns+log_weights
    rows = logsumexp(joint, axis=1)
    if not np.isfinite(rows).all():
        raise ValueError("Every mutation needs finite mixture probability")
    responsibilities = np.exp(joint-rows[:, None])
    supplied = np.asarray(fit.responsibilities)
    if (supplied.shape != responsibilities.shape or not np.isfinite(supplied).all()
            or np.any(supplied < 0) or np.any(supplied > 1)
            or not np.allclose(supplied, responsibilities, atol=1e-10, rtol=1e-8)
            or not np.allclose(supplied.sum(axis=1), 1., atol=1e-12, rtol=0)
            or np.any(supplied[:, ~positive] != 0)):
        raise ValueError("Independent verification disagrees on mixture responsibilities")
    labels = np.asarray(fit.labels)
    if (labels.dtype.kind not in "iu" or labels.shape != (model.n,)
            or np.any(labels < 0) or np.any(labels >= len(centers))):
        raise ValueError("Independent verification disagrees on mixture MAP labels")
    # Public relabeling may change which index is first in an exact tie; any
    # maximizing component remains a valid MAP label after that permutation.
    if not np.array_equal(responsibilities[np.arange(model.n), labels], responsibilities.max(axis=1)):
        raise ValueError("Independent verification disagrees on mixture MAP labels")
    occupied = int(len(np.unique(labels)))
    if fit.occupied_map_clusters != occupied:
        raise ValueError("Independent verification disagrees on occupied MAP count")
    mixture = math.fsum(rows)
    parameters = q*model.r+q-1
    penalty = parameters*math.log(model.n)
    score = -2*mixture+penalty
    for name, actual, reported in (("log likelihood", mixture, fit.log_likelihood),
                                   ("BIC", score, fit.score)):
        if not np.isfinite(reported) or not np.isclose(actual, reported, atol=1e-7, rtol=1e-10):
            raise ValueError(f"Independent verification disagrees on mixture {name}")
    # Scores are derivatives with respect to simplex weights, divided by N.
    # Include inactive components: an enormous entering derivative must fail,
    # not disappear through an underflowed responsibility or clipped score.
    with np.errstate(over="ignore"):
        weight_score = np.exp(logsumexp(columns-rows[:, None], axis=0)-math.log(model.n))
    weight_gap = model.n*max(0., float(weight_score.max()-1.))
    active_gap = float(weight_score.max()-weight_score[positive].min())
    roundoff = 64*np.finfo(float).eps
    if (not np.isfinite(weight_score).all() or weight_gap > model.n*(1e-8+roundoff)
            or active_gap > 1e-8+roundoff):
        raise ValueError("Independent mixture-weight optimality check failed")
    for name, actual, scale in (("weight_optimality_gap", weight_gap, model.n),
                                ("weight_active_score_gap", active_gap, 1)):
        reported = fit.diagnostics.get(name)
        if reported is not None and (not np.isfinite(reported) or reported < 0
                or not np.isclose(actual, reported, atol=scale*(roundoff+1e-12), rtol=1e-6)):
            raise ValueError(f"Independent verification disagrees on {name}")

    bank_count = None
    if candidates is not None:
        bank = tuple(candidates)
        bank_count, retained = len(bank), False
        for candidate in bank:
            c_centers, c_weights, _, c_q = _parameters(model, candidate)
            c_score = (-2*candidate.log_likelihood
                       +(c_q*model.r+c_q-1)*math.log(model.n))
            if (not np.isfinite(c_score) or not np.isfinite(candidate.score)
                    or not np.isclose(c_score, candidate.score, atol=1e-7, rtol=1e-10)):
                raise ValueError("Retained mixture candidate has inconsistent BIC arithmetic")
            if c_score < score and not np.isclose(c_score, score, atol=1e-7, rtol=1e-10):
                raise ValueError("Selected mixture is not the best complete-bank BIC")
            retained |= (np.array_equal(c_centers, centers) and np.array_equal(c_weights, weights)
                         and candidate.log_likelihood == fit.log_likelihood
                         and candidate.score == fit.score)
        if not retained:
            raise ValueError("Selected mixture state is absent from its retained candidate bank")
    conditional = math.fsum(columns[np.arange(model.n), labels])
    return dict(verification_method="independent_soft_mixture_v1", independently_verified=True,
                fitting_log_likelihood=mixture, mixture_log_likelihood=mixture,
                independent_log_likelihood=mixture, independent_score=score,
                conditional_log_likelihood=conditional, hard_partition_log_likelihood=conditional,
                complexity_penalty=penalty, score=score,
                score_definition="-2*observed_mixture_LL+(K_positive*R+K_positive-1)*log(N)",
                positive_components=q, occupied_map_clusters=occupied,
                weight_optimality_gap=weight_gap, weight_active_score_gap=active_gap,
                num_parameters=parameters, num_mutation_vectors=model.n,
                forbidden_assignment_pairs=0, assignment_validity_log_probability=0.,
                validity_conditioned_mixture_log_likelihood=mixture,
                complete_candidate_bank_reconciled=candidates is not None,
                selected_fitted_state_retained=candidates is not None,
                candidate_count=bank_count, candidate_likelihoods_independently_reconstructed=False,
                global_optimality_proven=False, local_optimality_certified=False,
                joint_mixture_center_optimization=True)
