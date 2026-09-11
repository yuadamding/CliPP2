"""Bounded, truth-free joint-center optimization and coherent dosage starts.

The observed likelihood lives only in objective.py. These multistart methods
retain attained local optima; they do not certify global optimality.
"""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import minimize

from .objective import ObservedModel, observed_terms_numpy, observed_terms_torch, observed_hessian_numpy


def mutation_subset(model: ObservedModel, members: np.ndarray) -> ObservedModel:
    return ObservedModel(
        **{name: getattr(model, name)[members] for name in
           ("alt", "nonalt", "observed", "lower", "upper", "slope", "log_prior", "valid")},
        model_id=model.model_id, coupling=model.coupling, support_policy=model.support_policy,
    )


def dosage_seeds_numpy(model: ObservedModel) -> np.ndarray:
    """M x S x H coherent state seeds; missing counts use the box midpoint.

    No labels, CCF truths, simulator manifests, or realized multiplicities are
    inputs. Invalid padded states are NaN, never alternative biological states.
    """
    total = model.alt + model.nonalt
    active = model.observed & (total > 0)
    mass = model.alt / np.maximum(total, 1)
    midpoint = 0.5 * (model.lower + model.upper)
    seeds = np.where(active[..., None] & (model.slope > 0),
                     mass[..., None] / np.maximum(model.slope, np.finfo(float).tiny),
                     midpoint[..., None])
    seeds = np.clip(seeds, model.lower[..., None], model.upper[..., None])
    return np.where(model.valid, seeds, np.nan)


@dataclass(frozen=True)
class JointMinimum:
    center: np.ndarray
    loss: float
    converged: bool
    starts: int
    evaluations: int
    second_loss_gap: float


def fit_joint_center(model: ObservedModel, *, eps: float, tol: float,
                     max_iter: int, max_modes: int = 24) -> JointMinimum:
    """Fit a full regional vector with deterministic data-derived multistarts.

    Starts and tie order depend on member data only, so repeated immutable
    partitions have the same refit regardless of proposal history/label names.
    All candidate states contribute to every evaluated likelihood.
    """
    if model.coupling != "joint":
        raise ValueError("fit_joint_center requires a joint observed model.")
    if max_iter < 1 or max_modes < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError("Joint refit work limits and tolerance must be positive.")
    lower, upper = model.lower.max(axis=0), model.upper.min(axis=0)
    if np.any(lower > upper):
        raise ValueError("Cluster members have no common feasible CCF box.")
    active = model.observed & ((model.alt + model.nonalt) > 0)
    # A center without any evidence is fixed only for deterministic reporting.
    missing = ~active.any(axis=0)
    lower = np.where(missing, 0.5 * (lower + upper), lower)
    upper = np.where(missing, lower, upper)
    seeds = dosage_seeds_numpy(model)
    starts = [0.5 * (lower + upper), lower, upper]
    for state in range(model.candidate_shape[-1]):
        members = model.valid[:, 0, state]
        if not members.any():
            continue
        for quantile in (0.2, 0.5, 0.8):
            starts.append(np.quantile(seeds[members, :, state], quantile, axis=0))
    unique = []
    for seed in starts:
        seed = np.clip(seed, lower, upper)
        if not any(np.max(np.abs(seed - old)) <= 1e-8 for old in unique):
            unique.append(seed)
    # Score-ranked truncation is deterministic and retains bounded work.
    def evaluate(center):
        nonlocal evaluations
        evaluations += 1
        terms = observed_terms_numpy(model, np.broadcast_to(center, model.shape), eps=eps)
        return float(terms.loss.sum()), terms.gradient.sum(axis=0)

    evaluations = 0
    unique = sorted(unique, key=lambda x: (evaluate(x)[0], tuple(x)))[:max_modes]
    minima = []
    for seed in unique:
        start_loss, _ = evaluate(seed)
        result = minimize(evaluate, seed, jac=True, method="L-BFGS-B",
                          bounds=list(zip(lower, upper)), options={
                              "maxiter": int(max_iter), "maxls": 40,
                              "ftol": 8 * np.finfo(float).eps,
                              "gtol": max(float(tol), 1e-9),
                          })
        center = np.clip(result.x, lower, upper)
        loss, gradient = evaluate(center)
        if not np.isfinite(loss) or loss > start_loss:
            center, loss = seed, start_loss
            _, gradient = evaluate(center)
        projected = center - np.clip(center - gradient, lower, upper)
        resolved = bool(np.max(np.abs(projected)) <= max(10 * tol, 1e-7))
        minima.append((loss, tuple(center), resolved))
    minima.sort(key=lambda item: (item[0], item[1]))
    best = minima[0]
    center, loss = np.asarray(best[1]), best[0]
    # Large read-count sums can trigger L-BFGS-B's relative-loss stop before
    # a small absolute center gradient. Bounded exact-Hessian local polishing
    # resolves that numerical issue without changing modes, boxes or priors.
    for _ in range(min(8, max_iter)):
        loss, gradient = evaluate(center)
        free = (lower < upper) & ~((center <= lower) & (gradient >= 0)) & ~((center >= upper) & (gradient <= 0))
        if not np.any(free) or np.max(np.abs(gradient[free])) <= max(tol, 1e-9):
            break
        hessian = observed_hessian_numpy(model, np.broadcast_to(center, model.shape), eps=eps).sum(axis=0)
        hessian = hessian[np.ix_(free, free)]
        if np.linalg.eigvalsh(hessian).min() <= 0:
            break
        direction = np.zeros_like(center)
        direction[free] = -np.linalg.solve(hessian, gradient[free])
        accepted = False
        for power in range(12):
            trial = np.clip(center + .5**power * direction, lower, upper)
            value, trial_gradient = evaluate(trial)
            rounding = 8 * np.finfo(float).eps * (1 + abs(loss))
            if value <= loss + rounding and np.linalg.norm(trial_gradient[free]) < np.linalg.norm(gradient[free]):
                center, loss, accepted = trial, value, True
                break
        if not accepted:
            break
    loss, gradient = evaluate(center)
    projected = center - np.clip(center - gradient, lower, upper)
    converged = bool(np.max(np.abs(projected)) <= max(10*tol, 1e-7))
    return JointMinimum(center, loss, converged, len(unique), evaluations,
                        max(float(minima[1][0] - loss), 0) if len(minima) > 1 else float("inf"))


def joint_partition_refit(model, labels, *, eps, tol, max_iter):
    from .scalar import PartitionRefitResult

    k, regions = int(labels.max()) + 1, model.shape[1]
    fits = [fit_joint_center(mutation_subset(model, np.flatnonzero(labels == cluster)),
                             eps=eps, tol=tol, max_iter=max_iter) for cluster in range(k)]
    centers = np.stack([fit.center for fit in fits])
    phi = centers[labels]
    # Re-evaluate through the canonical full model, rather than accumulating
    # optimizer-reported approximate/conditional objectives.
    loss = float(observed_terms_numpy(model, phi, eps=eps).loss.sum())
    boundary = active_df = 0
    for cluster in range(k):
        members = labels == cluster
        active = (model.observed[members] & ((model.alt[members] + model.nonalt[members]) > 0)).any(axis=0)
        at_bound = ((centers[cluster] <= model.lower[members].max(axis=0) + 10 * tol)
                    | (centers[cluster] >= model.upper[members].min(axis=0) - 10 * tol))
        boundary += int((active & at_bound).sum())
        active_df += int((active & ~at_bound).sum())
    return PartitionRefitResult(
        phi=phi, cluster_centers=centers, labels=labels.copy(), loglik=-loss,
        fit_loss=loss, n_clusters=k, boundary_count=boundary, active_degrees_of_freedom=active_df,
        finite_candidate_found=bool(np.isfinite(loss) and np.all(np.isfinite(phi))),
        refit_coordinate_count=k * regions,
        refit_finite_coordinate_count=int(np.isfinite(centers).sum()),
        refit_total_grid_points=sum(f.evaluations for f in fits), refit_max_grid_spacing=0.0,
        refit_total_candidate_basins=sum(f.starts for f in fits),
        refit_total_refined_candidates=sum(f.starts for f in fits),
        refit_min_best_second_loss_gap=min(f.second_loss_gap for f in fits),
        loglik_source="fixed_partition_joint_observed_multistart_v1",
        refit_mode="joint_multistart", locally_converged=all(f.converged for f in fits),
    )


@torch.no_grad()
def coherent_start_bank_torch(model, *, eps, steps=64):
    """Optimize each mutation's whole vector, preserving multiple shared modes.

    Acceptance is mutation-wise, never a mixture of region-wise accept/reject
    decisions from different joint trials. Memory is O(M*S*H), not H**S.
    """
    lower, upper = model.lower, model.upper
    active = model.observed & (model.total > 0)
    midpoint = 0.5 * (lower + upper)
    vaf = model.alt / model.total.clamp(min=1)
    starts = []
    for state in range(model.candidate_shape[-1]):
        slope = model.slope[..., state]
        valid = model.valid[..., state]
        current = torch.where(active & valid & (slope > 0),
                              vaf / slope.clamp(min=torch.finfo(slope.dtype).tiny), midpoint)
        current = torch.minimum(torch.maximum(current, lower), upper)
        for _ in range(steps):
            terms = observed_terms_torch(model, current, eps=eps)
            step = terms.gradient / terms.hessian_upper.clamp(min=1e-8)
            original_loss = terms.loss.sum(dim=1, keepdim=True)
            accepted = torch.zeros_like(original_loss, dtype=torch.bool)
            next_phi = current
            for backtrack in range(12):
                trial = torch.minimum(torch.maximum(current - 0.5 ** backtrack * step, lower), upper)
                loss = observed_terms_torch(model, trial, eps=eps).loss.sum(dim=1, keepdim=True)
                take = ~accepted & torch.isfinite(loss) & (loss <= original_loss)
                next_phi = torch.where(take, trial, next_phi)
                accepted |= take
                if bool(accepted.all()):
                    break
            delta = torch.max(torch.abs(next_phi - current))
            current = next_phi
            if float(delta) < 1e-8:
                break
        if not any(torch.allclose(current, old, rtol=0.0, atol=1e-8) for old in starts):
            starts.append(current)
    return tuple(starts)
