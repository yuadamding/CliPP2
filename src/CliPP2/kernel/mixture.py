"""Experimental joint marginal-mixture estimator, separate from partition refits.

This CPU-only research API optimizes centers and weights together. Its positive
mixture components need not have occupied MAP labels, so it deliberately does
not return a production FitResult or use the three-table workflow publisher.
Native marginal emissions/posteriors retain the model's fixed joint support.
"""
from collections import deque
from dataclasses import dataclass
from itertools import zip_longest
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

from .joint import joint_row
from .scoring import WeightConvergenceError, fit_mixture_weights
from ..io.data import freeze_record, readonly_array


class MixtureConvergenceError(RuntimeError):
    """No stationary, nondecreasing mixture fit was obtained from either start."""

    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = freeze_record({} if diagnostics is None else diagnostics)


@dataclass(frozen=True, slots=True, eq=False)
class MixtureFit:
    centers: np.ndarray
    weights: np.ndarray
    responsibilities: np.ndarray
    labels: np.ndarray
    log_likelihood: float
    score: float
    positive_components: int
    occupied_map_clusters: int
    converged: bool
    iterations: int
    diagnostics: dict

    def __post_init__(self):
        for name in ("centers", "weights", "responsibilities", "labels"):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))
        object.__setattr__(self, "diagnostics", freeze_record(self.diagnostics))


@dataclass(frozen=True, slots=True)
class MixtureSearch:
    best: MixtureFit
    candidates: tuple[MixtureFit, ...]
    diagnostics: dict

    def __post_init__(self):
        object.__setattr__(self, "diagnostics", freeze_record(self.diagnostics))


def _maximize_coordinate(alt, ref, slopes, posterior_weights, lower=0., upper=1.):
    """Maximize a concave weighted-binomial M-step, including exact endpoints.

    Inputs are aligned one-dimensional vectors over mutation/multiplicity
    terms. Zero responsibility is discarded before boundary arithmetic.
    """
    alt, ref, slopes, weights = (np.asarray(value, dtype=float) for value in
                                (alt, ref, slopes, posterior_weights))
    if (alt.ndim != 1 or any(value.shape != alt.shape for value in (ref, slopes, weights))
            or any(not np.isfinite(value).all() for value in (alt, ref, slopes, weights))
            or np.any(alt < 0) or np.any(ref < 0) or np.any(slopes <= 0)
            or np.any(weights < 0) or not 0 <= lower <= upper <= 1):
        raise ValueError("Weighted coordinate inputs require finite aligned nonnegative terms")
    used = weights > 0
    alt, ref, slopes, weights = (value[used] for value in (alt, ref, slopes, weights))
    if not len(weights) or lower == upper:
        return float(lower)
    # A positive reference count makes p=1 inadmissible. Components with only
    # ALT reads are constant after the clipped probability reaches one.
    if np.any(ref > 0):
        upper = min(upper, float(np.min(1 / slopes[ref > 0])))
    if lower > upper:
        raise ValueError("Weighted coordinate has no feasible probability domain")

    def derivative(center):
        active = slopes * center <= 1
        a, b, s, w = (value[active] for value in (alt, ref, slopes, weights))
        if center == 0 and np.any(a > 0):
            return np.inf
        if np.any((b > 0) & (s * center >= 1)):
            return -np.inf
        first = float(np.dot(w, a)) / center if center > 0 else 0.
        used_ref = b > 0
        return first - float(np.sum(w[used_ref] * b[used_ref] * s[used_ref]
                                    / (1 - s[used_ref] * center)))

    if derivative(lower) <= 0:
        return float(lower)
    if derivative(upper) >= 0:
        return float(upper)
    for _ in range(64):
        middle = (lower + upper) / 2
        if derivative(middle) > 0:
            lower = middle
        else:
            upper = middle
    return float((lower + upper) / 2)


def _responsibilities(columns, weights):
    log_weights = np.full(len(weights), -np.inf)
    used = weights > 0
    log_weights[used] = np.log(weights[used])
    joint = columns + log_weights
    rows = logsumexp(joint, axis=1)
    if not np.isfinite(rows).all():
        raise ValueError("Mixture centers cannot explain every mutation")
    return float(np.sum(rows)), np.exp(joint - rows[:, None]), rows


def _evaluate(model, centers, weights):
    columns = model.selection_column_set(centers).materialize()
    fitted = fit_mixture_weights(columns, weights)
    ll, posterior, rows = _responsibilities(columns, fitted["cluster_weights"])
    return dict(centers=np.array(centers, copy=True), weights=fitted["cluster_weights"],
                columns=columns, responsibilities=posterior, rows=rows,
                log_likelihood=ll, weight_fit=fitted)


def _raw_derivative(alt, depth, slope, center):
    """Signed log derivative of binomial mass, not of its logarithm."""
    ref, probability = depth - alt, slope * center
    if probability > 1 or depth == 0:
        return 0., -np.inf
    if probability == 0:
        if alt == 0:
            return -1., math.log(depth * slope)
        if alt == 1:
            return 1., math.log(depth * slope)
        return 0., -np.inf
    if probability == 1:
        if ref == 0:
            return 1., math.log(depth * slope)
        if ref == 1:
            return -1., math.log(depth * slope)
        return 0., -np.inf
    derivative = slope * (alt / probability - ref / (1 - probability))
    if derivative == 0:
        return 0., -np.inf
    log_mass = (gammaln(depth + 1) - gammaln(alt + 1) - gammaln(ref + 1)
                + alt * math.log(probability) + ref * math.log1p(-probability))
    return math.copysign(1., derivative), float(log_mass + math.log(abs(derivative)))


def _center_gradient(model, state):
    """Actual mixture derivative, including zero-emission boundary terms.

    Multiplying responsibility by a log-emission derivative loses finite
    derivatives where the emission itself is zero. The exact cavity oracle is
    used only for these endpoint rows; ordinary rows use native derivatives.
    """
    centers, weights = state["centers"], state["weights"]
    gradient = np.zeros_like(centers)
    for k in np.flatnonzero(weights > 0):
        center = centers[k]
        values = np.broadcast_to(center, (model.n, model.r))
        loss, derivative, _ = model.selection_backend.node_terms(values)
        finite = np.isfinite(loss)
        gradient[k] = -np.sum(state["responsibilities"][finite, k, None] * derivative[finite], axis=0)
        for i in np.flatnonzero(~finite):
            row = joint_row(model.alt[i], model.depth[i], model.scaling[i],
                            model.multiplicity_min[i], model.multiplicity_max[i],
                            model.observed[i], center,
                            max_multiplicity_difference=model.max_multiplicity_difference)
            scale = math.log(weights[k]) - row["log_normalizer"] - state["rows"][i]
            for r in np.flatnonzero(model.observed[i]):
                for offset, cavity in enumerate(row["cavity_log_marginal"][r]):
                    multiplicity = int(model.multiplicity_min[i, r]) + offset
                    sign, log_abs = _raw_derivative(model.alt[i, r], model.depth[i, r],
                                                    model.scaling[i, r] * multiplicity, center[r])
                    if sign and np.isfinite(cavity):
                        gradient[k, r] += sign * math.exp(scale + cavity + log_abs)
    if not np.isfinite(gradient).all():
        raise MixtureConvergenceError("Nonfinite mixture derivative")
    projected = gradient.copy()
    projected[(centers <= 0) & (gradient < 0)] = 0.
    projected[(centers >= 1) & (gradient > 0)] = 0.
    return gradient, float(np.max(np.abs(projected)) / (1 + model.n))


def _em_centers(model, state):
    centers = state["centers"].copy()
    for k in np.flatnonzero(state["weights"] > 0):
        posterior = model.selection_backend.joint_diagnostics(
            np.broadcast_to(centers[k], (model.n, model.r)))["probabilities"]
        for r in range(model.r):
            terms = [[], [], [], []]
            for i in np.flatnonzero(model.observed[:, r] & (state["responsibilities"][:, k] > 0)):
                q = np.asarray(posterior[i][r]) * state["responsibilities"][i, k]
                multiplicities = np.arange(model.multiplicity_min[i, r], model.multiplicity_max[i, r]+1)
                terms[0].extend(np.full(len(q), model.alt[i, r]))
                terms[1].extend(np.full(len(q), model.depth[i, r] - model.alt[i, r]))
                terms[2].extend(model.scaling[i, r] * multiplicities)
                terms[3].extend(q)
            if terms[3] and np.sum(terms[3]) > 0:
                centers[k, r] = _maximize_coordinate(*terms)
    return centers


def _rescue(model, state, iterations):
    """Bounded actual-likelihood refinement with fixed weights, then weight MLE."""
    shape, weights = state["centers"].shape, state["weights"]
    best = state

    def objective(vector):
        nonlocal best
        centers = vector.reshape(shape)
        columns = model.selection_column_set(centers).materialize()
        try:
            ll, posterior, rows = _responsibilities(columns, weights)
        except ValueError:
            return np.inf, np.zeros_like(vector)
        trial = dict(centers=centers.copy(), weights=weights, columns=columns,
                     responsibilities=posterior, rows=rows, log_likelihood=ll)
        if ll > best["log_likelihood"]:
            best = trial
        gradient, _ = _center_gradient(model, trial)
        return -ll, -gradient.ravel()

    minimize(objective, state["centers"].ravel(), jac=True, method="L-BFGS-B",
             bounds=[(0., 1.)] * np.prod(shape),
             options={"maxiter": iterations, "ftol": 1e-14, "gtol": 1e-8, "maxls": 30})
    return _evaluate(model, best["centers"], weights)


def _fit_start(model, start, weights, max_iterations, gradient_tolerance):
    """One independent trajectory; numerical failure does not cancel others."""
    trace, converged, residual, rescues, iteration = [], False, np.inf, 0, 0
    state, failure = None, {}
    try:
        try:
            state = _evaluate(model, start, weights)
        except ValueError as exc:
            # An endpoint start may have zero likelihood for every component.
            # Invalid values later in the optimizer are not suppressed.
            return None, dict(converged=False, reason=str(exc),
                              exception=type(exc).__name__, iterations=0)
        trace.append(state["log_likelihood"])
        for iteration in range(1, int(max_iterations) + 1):
            proposed = _evaluate(model, _em_centers(model, state),
                                 np.mean(state["responsibilities"], axis=0))
            change = proposed["log_likelihood"] - state["log_likelihood"]
            if change >= 0:
                state = proposed
            _, residual = _center_gradient(model, state)
            if residual <= gradient_tolerance:
                converged = True
                trace.append(state["log_likelihood"])
                break
            if iteration >= 3 and (iteration % 10 == 0 or change <= 1e-8 * (1 + abs(state["log_likelihood"]))):
                proposed = _rescue(model, state, min(30, int(max_iterations) - iteration + 1))
                rescues += 1
                if proposed["log_likelihood"] >= state["log_likelihood"]:
                    state = proposed
                _, residual = _center_gradient(model, state)
                converged = residual <= gradient_tolerance
            trace.append(state["log_likelihood"])
            if converged:
                break
    except (WeightConvergenceError, MixtureConvergenceError) as exc:
        failure = dict(reason=str(exc), exception=type(exc).__name__,
                       diagnostics=getattr(exc, "diagnostics", {}))
    return state, dict(converged=converged, iterations=iteration, rescues=rescues,
                       projected_gradient_residual=residual if np.isfinite(residual) else None,
                       log_likelihood_trace=tuple(trace), **failure)


def fit_marginal_mixture(model, initial_centers, initial_weights=None, *,
                         max_iterations=2000, gradient_tolerance=1e-5):
    """Fit one experimental mixture order using weighted EM and stationarity.

    Both the supplied centers and a genuinely interior blend are tried. Exact
    zero weights remain valid inactive slots; positive components are never
    removed merely because their MAP assignment is empty. This API does not
    claim global optimality or CUDA support and never hard-refits its labels.
    """
    if model.device != "cpu":
        raise NotImplementedError("Experimental marginal-mixture fitting currently supports CPU only")
    if model.fit_multiplicity != "marginal":
        raise ValueError("Experimental mixture fitting requires marginal multiplicity aggregation")
    centers = np.asarray(initial_centers, dtype=float)
    if (centers.ndim != 2 or centers.shape[1] != model.r or not len(centers)
            or not np.isfinite(centers).all() or np.any((centers < 0) | (centers > 1))):
        raise ValueError("Initial mixture centers must be finite K by R CCFs in [0, 1]")
    if (isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1
            or not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0):
        raise ValueError("Mixture iteration limit and gradient tolerance must be positive")
    weights = (np.full(len(centers), 1 / len(centers)) if initial_weights is None
               else np.asarray(initial_weights, dtype=float))
    if (weights.shape != (len(centers),) or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
        raise ValueError("Initial mixture weights must lie on the probability simplex")
    initial_columns = model.selection_column_set(centers).materialize()
    try:
        initial_ll = _responsibilities(initial_columns, weights)[0]
    except ValueError:
        initial_ll = -np.inf
    runs, accepted = [], []
    for name, start in (("supplied", centers), ("interior", (centers + .5) / 2)):
        state, record = _fit_start(model, start, weights, max_iterations, gradient_tolerance)
        record["start"] = name
        runs.append(record)
        if record["converged"] and state["log_likelihood"] >= initial_ll:
            accepted.append((state, record))
    if not accepted:
        summary = [(run["start"], run["iterations"], run.get("projected_gradient_residual")) for run in runs]
        raise MixtureConvergenceError(f"No stationary nondecreasing marginal-mixture fit: {summary}",
                                      diagnostics=dict(starts=tuple(runs)))
    state, record = max(accepted, key=lambda value: value[0]["log_likelihood"])
    labels = np.argmax(state["responsibilities"], axis=1).astype(np.int64)
    positive = int(np.count_nonzero(state["weights"] > 0))
    ll = state["log_likelihood"]
    # Incumbent trace begins at the caller's actual supplied point, even when
    # the separate interior-start trajectory wins. All start traces are kept.
    trace = [initial_ll] if np.isfinite(initial_ll) else []
    for value in record["log_likelihood_trace"]:
        if not trace or value >= trace[-1]:
            trace.append(value)
    diagnostics = dict(log_likelihood_trace=tuple(trace), starts=tuple(runs),
                       projected_gradient_residual=record["projected_gradient_residual"],
                       weight_optimality_gap=state["weight_fit"]["weight_optimality_gap"],
                       weight_active_score_gap=state["weight_fit"]["weight_active_score_gap"],
                       estimator="experimental_joint_marginal_mixture", device="cpu",
                       hard_refits=0, global_optimality_claimed=False)
    return MixtureFit(state["centers"], state["weights"], state["responsibilities"], labels,
                      ll, -2 * ll + (positive * model.r + positive - 1) * math.log(model.n),
                      positive, int(len(np.unique(labels))), True, record["iterations"], diagnostics)


def _mixture_proposals(model, parent, pilot, max_clusters):
    """Resumable per-parent operator stream; no MAP-empty component deletion."""
    active = np.flatnonzero(parent.weights > 0)
    centers = parent.centers[active]
    weights = parent.weights[active]
    assignment = np.argmax(parent.responsibilities[:, active], axis=1)
    # Missing pilot coordinates affect initialization only, never likelihood
    # observations or fixed multiplicity support.
    points = np.where(np.isfinite(pilot), pilot, centers[assignment])
    births, merges, transfers, relocations = [], [], [], []
    for source in range(len(active)):
        members = np.flatnonzero(assignment == source)
        if len(members) < 2:
            continue
        for region in np.argsort(-np.ptp(points[members], axis=0), kind="stable"):
            ordered = members[np.argsort(points[members, region], kind="stable")]
            for cut in sorted({len(ordered)//2, int(np.argmax(np.diff(points[ordered, region])))+1}):
                left, right = ordered[:cut], ordered[cut:]
                a, b = np.mean(points[left], axis=0), np.mean(points[right], axis=0)
                if np.array_equal(a, b):
                    continue
                split = centers.copy()
                split[source] = a
                split = np.vstack((split, b))
                mass = weights[source] * len(left) / len(members)
                split_weights = np.append(weights, weights[source] - mass)
                split_weights[source] = mass
                births.append((source, split, split_weights))
    for first in range(len(active)):
        for second in range(first + 1, len(active)):
            keep = np.arange(len(active)) != second
            merged = centers.copy()
            mass = weights[first] + weights[second]
            merged[first] = (weights[first] * centers[first] + weights[second] * centers[second]) / mass
            merged_weights = weights.copy()
            merged_weights[first] = mass
            merges.append((merged[keep], merged_weights[keep], "merge"))
            # Atomic relocation can cross an unfavorable lower-K fit.
            for source, split, split_weights in births:
                if source in (first, second):
                    continue
                value, mixed_weights = split.copy(), split_weights.copy()
                value[first], mixed_weights[first] = merged[first], mass
                selected = np.arange(len(value)) != second
                relocations.append((value[selected], mixed_weights[selected], "split_merge"))
    # Transfer seeds can be unfavorable at the old centers; only the coherent
    # optimized mixture, never a frozen-column gain, is scored.
    columns = model.selection_column_set(centers).materialize()
    for source in range(len(active)):
        members = np.flatnonzero(assignment == source)
        if len(members) < 2:
            continue
        for target in range(len(active)):
            if source == target:
                continue
            gain = columns[members, target] - columns[members, source]
            ordered = members[np.argsort(-gain, kind="stable")]
            for count in sorted({1, len(members)//2}):
                moved, remaining = ordered[:count], ordered[count:]
                destination = np.concatenate((np.flatnonzero(assignment == target), moved))
                value = centers.copy()
                value[source] = np.mean(points[remaining], axis=0)
                value[target] = np.mean(points[destination], axis=0)
                transfers.append((value, weights.copy(), "transfer"))
    birth_proposals = ((value, mass, "birth") for _, value, mass in births
                       if len(active) < max_clusters)
    for group in zip_longest(birth_proposals, transfers, merges, relocations):
        yield from (proposal for proposal in group if proposal is not None)


def search_marginal_mixture(model, initial_fits, *, max_clusters, proposal_budget=32,
                            max_iterations=2000, gradient_tolerance=1e-5):
    """Bounded experimental birth/merge/split-merge search over mixture fits.

    Seeds may be center arrays or objects with ``centers`` and ``weights``.
    Proposal centers are pilot means, never conditional hard-partition refits.
    The budget counts numerical proposal attempts, not successful fits. Every
    converged candidate is reconciled with the final winner by mixture BIC.
    """
    if (isinstance(max_clusters, bool) or int(max_clusters) != max_clusters or max_clusters < 1
            or isinstance(proposal_budget, bool) or int(proposal_budget) != proposal_budget
            or proposal_budget < 0):
        raise ValueError("Experimental mixture search requires positive K and nonnegative proposal budget")
    seeds = tuple(initial_fits)
    if not seeds:
        raise ValueError("Experimental mixture search requires at least one center seed")
    bank, failures, seen = [], [], set()
    frontier, scheduled = deque(), set()
    attempted, expanded = 0, 0

    def enqueue(fit):
        key = (fit.centers.shape, fit.centers.tobytes(), fit.weights.tobytes())
        if key not in scheduled:
            scheduled.add(key)
            frontier.append((fit, None))

    def admit(centers, weights, origin):
        centers = np.asarray(centers, dtype=float)
        key = (centers.shape, centers.tobytes(), None if weights is None else np.asarray(weights).tobytes())
        if key in seen:
            return None
        seen.add(key)
        try:
            fit = fit_marginal_mixture(model, centers, weights, max_iterations=max_iterations,
                                       gradient_tolerance=gradient_tolerance)
        except (MixtureConvergenceError, WeightConvergenceError) as exc:
            failures.append(dict(origin=origin, reason=str(exc),
                                 diagnostics=getattr(exc, "diagnostics", {})))
            return None
        bank.append(fit)
        enqueue(fit)
        return fit

    for seed in seeds:
        centers = getattr(seed, "centers", seed)
        if len(centers) > max_clusters:
            raise ValueError("Initial mixture seed exceeds max_clusters")
        admit(centers, getattr(seed, "weights", None), "seed")
    if not bank:
        raise MixtureConvergenceError(f"All experimental mixture seeds failed: {failures}")
    pilot, _ = model.pilots()
    while frontier and attempted < proposal_budget:
        parent, proposals = frontier.popleft()
        if proposals is None:
            proposals = _mixture_proposals(model, parent, pilot, max_clusters)
            expanded += 1
        try:
            centers, weights, origin = next(proposals)
        except StopIteration:
            continue
        attempted += 1
        admit(centers, weights, origin)
        # A child gets a turn before its parent resumes, so a large parent's
        # proposal list cannot consume the entire budget before recurrence.
        frontier.append((parent, proposals))
    best = min(bank, key=lambda fit: fit.score)
    return MixtureSearch(best, tuple(bank), dict(proposal_attempts=attempted,
                         proposal_budget=int(proposal_budget), failures=tuple(failures),
                         proposal_quantum=1, expanded_states=expanded,
                         remaining_frontier_states=len(frontier),
                         candidate_count=len(bank), estimator="experimental_joint_marginal_mixture",
                         occupied_map_count_is_component_count=False,
                         exhaustive_search=False, global_optimality_claimed=False))
