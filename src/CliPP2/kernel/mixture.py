"""Joint marginal-mixture estimation without a tree or hard partition refit.

The CPU optimizer serves production fits and saved-population comparisons.
Positive mixture components need not have occupied MAP labels. Native marginal
emissions/posteriors retain the model's fixed joint multiplicity support.
"""
from collections import deque
from dataclasses import dataclass
from itertools import zip_longest
import math
from time import perf_counter

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


def _record(model, name, amount=1):
    telemetry = model.telemetry
    name = "mixture_" + name
    telemetry[name] = telemetry.get(name, 0) + amount


def canonical_components(centers, weights, *, drop_zero=True):
    """Consolidate exact duplicate vectors only; never merge nearby centers.

    Exact zero slots may be retained in optimizer starts so the weight solver
    can test their entering direction. Final stationary states drop those slots,
    but retain every distinct positive component, including MAP-empty ones.
    """
    centers, weights = np.asarray(centers, float), np.asarray(weights, float)
    unique, inverse = np.unique(centers + 0., axis=0, return_inverse=True)
    mass = np.array([math.fsum(weights[inverse == k]) for k in range(len(unique))])
    used = mass > 0 if drop_zero else np.ones(len(mass), bool)
    return unique[used], mass[used]


def _state_key(centers, weights):
    centers, weights = canonical_components(centers, weights, drop_zero=False)
    return centers.shape, centers.tobytes(), weights.tobytes()


def _packed_terms(model, offsets):
    """Prepare immutable region-major terms once, not per component/iteration."""
    packed = getattr(model, "_mixture_packed_terms", None)
    if packed is not None:
        if not np.array_equal(packed[0], offsets):
            raise ValueError("Mixture posterior packing changed within an immutable model")
        return packed[1]
    started = perf_counter()
    widths = (model.multiplicity_max - model.multiplicity_min + 1).T.ravel()
    expected = np.r_[0, np.cumsum(widths)]
    if not np.array_equal(expected, offsets):
        raise ValueError("Native posterior offsets differ from fixed multiplicity support")
    cells = np.repeat(np.arange(model.n * model.r), widths)
    multiplicities = (model.multiplicity_min.T.ravel()[cells] + np.arange(len(cells))
                      - np.repeat(expected[:-1], widths))
    terms = []
    for r in range(model.r):
        begin, end = int(expected[r*model.n]), int(expected[(r+1)*model.n])
        selected = cells[begin:end] % model.n
        slope = model.scaling[selected, r] * multiplicities[begin:end]
        distinct, inverse = np.unique(slope, return_inverse=True)
        terms.append((slice(begin, end), selected, model.alt[selected, r],
                      model.depth[selected, r]-model.alt[selected, r], slope,
                      model.observed[selected, r], distinct, inverse))
    model._mixture_packed_terms = (np.array(offsets, copy=True), tuple(terms))
    _record(model, "packing_seconds", perf_counter()-started)
    _record(model, "packing_calls")
    return tuple(terms)


def _solve_aggregated(alt_mass, slopes, ref_mass, lower=0., upper=1.):
    """Exact equal-slope aggregation with safeguarded Newton and endpoint checks."""
    used = ref_mass > 0
    slopes, ref_mass = slopes[used], ref_mass[used]
    if lower == upper or alt_mass == 0:
        return float(lower)
    if not len(slopes):
        return float(upper)
    if len(slopes) == 1:
        return float(np.clip(alt_mass/(slopes[0]*(alt_mass+ref_mass[0])), lower, upper))
    coefficients = ref_mass * slopes

    def derivative(x):
        if x == 0:
            return np.inf, -np.inf
        denominator = 1-slopes*x
        if np.any(denominator <= 0):
            return -np.inf, -np.inf
        return (alt_mass/x - np.sum(coefficients/denominator),
                -alt_mass/(x*x) - np.sum(coefficients*slopes/(denominator*denominator)))

    if derivative(lower)[0] <= 0:
        return float(lower)
    if derivative(upper)[0] >= 0:
        return float(upper)
    x = (lower+upper)/2
    for _ in range(80):
        gradient, curvature = derivative(x)
        if gradient > 0:
            lower = x
        elif gradient < 0:
            upper = x
        else:
            return float(x)
        if upper-lower <= 2*np.finfo(float).eps:
            break
        proposed = x-gradient/curvature
        x = proposed if lower < proposed < upper else (lower+upper)/2
    return float((lower+upper)/2)


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
    if np.all(slopes <= 1):
        distinct, inverse = np.unique(slopes, return_inverse=True)
        return _solve_aggregated(float(np.dot(weights, alt)), distinct,
            np.bincount(inverse, weights=weights*ref, minlength=len(distinct)), lower, upper)
    # Preserve the clipped-probability fallback for generic slopes above one;
    # the biological CN model ordinarily uses slopes at most one.
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


def _evaluate(model, centers, weights, *, optimize_weights=True, reuse=None):
    if reuse is not None and np.array_equal(centers, reuse["centers"]):
        step, columns = reuse["e_step"], reuse["columns"]
        _record(model, "emission_reuses")
    else:
        started = perf_counter()
        step = model.selection_backend.mixture_e_step(centers)
        columns = step["log_marginal"]
        _record(model, "posterior_seconds", perf_counter()-started)
        _record(model, "posterior_calls")
    started = perf_counter()
    fitted = (fit_mixture_weights(columns, weights, telemetry=model.telemetry)
              if optimize_weights else None)
    weights = fitted["cluster_weights"] if fitted is not None else np.asarray(weights)/np.sum(weights)
    _record(model, "weight_seconds", perf_counter()-started)
    _record(model, "full_weight_solves" if optimize_weights else "em_weight_updates")
    ll, posterior, rows = _responsibilities(columns, weights)
    return dict(centers=np.array(centers, copy=True), weights=weights,
                columns=columns, responsibilities=posterior, rows=rows,
                log_likelihood=ll, weight_fit=fitted, e_step=step)


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
    started = perf_counter()
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
    _record(model, "gradient_seconds", perf_counter()-started)
    _record(model, "gradient_calls")
    return gradient, float(np.max(np.abs(projected)) / (1 + model.n))


def _em_centers(model, state):
    centers = state["centers"].copy()
    step = state["e_step"]
    terms = _packed_terms(model, step["offsets"])
    for k in np.flatnonzero(state["weights"] > 0):
        for r, (section, rows, alt, ref, slopes, observed, distinct, inverse) in enumerate(terms):
            started = perf_counter()
            tau = state["responsibilities"][rows, k]
            used = observed & (tau > 0)
            posterior = step["probabilities"][k, section]
            if not np.isfinite(posterior[used]).all() or np.any(posterior[used] < 0):
                raise FloatingPointError("Positive-responsibility cells require finite multiplicity posteriors")
            weights = np.zeros(len(rows))
            # Missing and impossible cells have undefined (NaN) posteriors;
            # multiplying them by zero would still contaminate the M-step.
            np.multiply(posterior, tau, out=weights, where=used)
            alt_mass = float(np.dot(weights, alt))
            ref_mass = np.bincount(inverse, weights=weights*ref, minlength=len(distinct))
            _record(model, "mstep_aggregation_seconds", perf_counter()-started)
            if np.any(weights > 0):
                started = perf_counter()
                centers[k, r] = (_solve_aggregated(alt_mass, distinct, ref_mass)
                    if np.all(distinct <= 1) else _maximize_coordinate(alt, ref, slopes, weights))
                _record(model, "root_seconds", perf_counter()-started)
                _record(model, "root_calls")
    return centers


def _rescue(model, state, iterations, gradient_tolerance=1e-5):
    """Bounded actual-likelihood refinement with fixed weights, then weight MLE."""
    started = perf_counter()
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
        return -ll/(1+model.n), -gradient.ravel()/(1+model.n)

    result = minimize(objective, state["centers"].ravel(), jac=True, method="L-BFGS-B",
             bounds=[(0., 1.)] * np.prod(shape),
             options={"maxiter": iterations, "ftol": 1e-14, "gtol": gradient_tolerance, "maxls": 30})
    _record(model, "rescue_seconds", perf_counter()-started)
    _record(model, "rescue_calls")
    _record(model, "rescue_iterations", int(result.nit))
    _record(model, "rescue_function_evaluations", int(result.nfev))
    _record(model, "rescue_gradient_evaluations", int(result.njev))
    _record(model, "rescue_status_"+str(result.status))
    fitted = _evaluate(model, best["centers"], weights, reuse=state)
    fitted["rescue_termination"] = dict(status=int(result.status), message=str(result.message),
        iterations=int(result.nit), function_evaluations=int(result.nfev), gradient_evaluations=int(result.njev))
    return fitted


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
                np.mean(state["responsibilities"], axis=0), optimize_weights=iteration % 10 == 0,
                reuse=state)
            change = proposed["log_likelihood"] - state["log_likelihood"]
            if change >= 0:
                state = proposed
            _, residual = _center_gradient(model, state)
            if residual <= gradient_tolerance:
                qualified = _evaluate(model, state["centers"], state["weights"], reuse=state)
                if qualified["log_likelihood"] >= state["log_likelihood"]:
                    state = qualified
                _, residual = _center_gradient(model, state)
                converged = residual <= gradient_tolerance and state["weight_fit"] is not None
                if converged:
                    trace.append(state["log_likelihood"])
                    break
            if iteration >= 3 and (iteration % 10 == 0 or change <= 1e-8 * (1 + abs(state["log_likelihood"]))):
                proposed = _rescue(model, state, min(30, int(max_iterations) - iteration + 1),
                                   gradient_tolerance)
                rescues += 1
                if proposed["log_likelihood"] >= state["log_likelihood"]:
                    state = proposed
                _, residual = _center_gradient(model, state)
                converged = residual <= gradient_tolerance and state["weight_fit"] is not None
            trace.append(state["log_likelihood"])
            if converged:
                break
    except (WeightConvergenceError, MixtureConvergenceError) as exc:
        failure = dict(reason=str(exc), exception=type(exc).__name__,
                       diagnostics=getattr(exc, "diagnostics", {}))
    return state, dict(converged=converged, iterations=iteration, rescues=rescues,
                       projected_gradient_residual=residual if np.isfinite(residual) else None,
                       log_likelihood_trace=tuple(trace),
                       last_rescue_termination=None if state is None else state.get("rescue_termination"),
                       **failure)


def fit_marginal_mixture_starts(model, initial_centers, initial_weights=None, *,
                               max_iterations=2000, gradient_tolerance=1e-5):
    """Retain every distinct converged start, including different effective K.

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
    runs, accepted, seen = [], [], set()
    for name, start in (("supplied", centers), ("interior", (centers + .5) / 2)):
        start, mass = canonical_components(start, weights, drop_zero=False)
        key = _state_key(start, mass)
        if key in seen:
            _record(model, "duplicate_starts_skipped")
            continue
        seen.add(key)
        _record(model, "start_attempts")
        state, record = _fit_start(model, start, mass, max_iterations, gradient_tolerance)
        record["start"] = name
        runs.append(record)
        # Monotonicity is per trajectory. A different-order start may have lower
        # likelihood and better BIC; the supplied start is not its eligibility gate.
        if record["converged"]:
            accepted.append((state, record))
    if not accepted:
        summary = [(run["start"], run["iterations"], run.get("projected_gradient_residual")) for run in runs]
        raise MixtureConvergenceError(f"No stationary nondecreasing marginal-mixture fit: {summary}",
                                      diagnostics=dict(starts=tuple(runs)))
    qualified, seen = [], set()
    for state, record in accepted:
        try:
            canonical, mass = canonical_components(state["centers"], state["weights"])
            # Recheck the exact representation, without hard refitting, weight
            # flooring or approximate merging. One failed final check must not
            # erase another independently qualified start.
            final = _evaluate(model, canonical, mass, reuse=state)
            while np.any(final["weights"] == 0):
                canonical, mass = canonical_components(final["centers"], final["weights"])
                final = _evaluate(model, canonical, mass)
            _, residual = _center_gradient(model, final)
            if residual > gradient_tolerance:
                raise MixtureConvergenceError("Canonical mixture failed center stationarity")
        except (WeightConvergenceError, MixtureConvergenceError) as exc:
            record.update(converged=False, phase="canonical_qualification", reason=str(exc),
                          exception=type(exc).__name__, diagnostics=getattr(exc, "diagnostics", {}))
            continue
        key = _state_key(final["centers"], final["weights"])
        if key in seen:
            continue
        seen.add(key)
        qualified.append((final, record, residual))
    if not qualified:
        raise MixtureConvergenceError("No canonical marginal-mixture start passed final qualification",
                                      diagnostics=dict(starts=tuple(runs)))
    candidates = []
    for final, record, residual in qualified:
        labels = np.argmax(final["responsibilities"], axis=1).astype(np.int64)
        positive = int(np.count_nonzero(final["weights"] > 0))
        ll = final["log_likelihood"]
        diagnostics = dict(log_likelihood_trace=record["log_likelihood_trace"], starts=tuple(runs),
            selected_start=record["start"], cross_start_likelihood_filter=False,
            projected_gradient_residual=residual, gradient_tolerance=gradient_tolerance,
            weight_optimality_gap=final["weight_fit"]["weight_optimality_gap"],
            weight_active_score_gap=final["weight_fit"]["weight_active_score_gap"],
            estimator="joint_marginal_mixture", device="cpu", exact_duplicate_consolidation=True,
            hard_refits=0, global_optimality_claimed=False)
        candidates.append(MixtureFit(final["centers"], final["weights"], final["responsibilities"], labels,
            ll, -2*ll+(positive*(model.r+1)-1)*math.log(model.n), positive,
            int(len(np.unique(labels))), True, record["iterations"], diagnostics))
    return tuple(candidates)


def fit_marginal_mixture(model, initial_centers, initial_weights=None, **kwargs):
    """Convenience best-BIC fit; the search API retains all converged starts."""
    candidates = fit_marginal_mixture_starts(model, initial_centers, initial_weights, **kwargs)
    return min(candidates, key=lambda fit: (fit.score, _state_key(fit.centers, fit.weights)))


def _mixture_proposals(model, parent, pilot, max_clusters):
    """Genuinely lazy proposals, preserving the original operator/order policy."""
    active = np.flatnonzero(parent.weights > 0)
    centers = parent.centers[active]
    weights = parent.weights[active]
    assignment = np.argmax(parent.responsibilities[:, active], axis=1)
    points = np.where(np.isfinite(pilot), pilot, centers[assignment])
    # Only small split descriptors are retained. K-by-R center/weight arrays
    # are materialized at yield, not for the O(R*K^3) split/merge catalogue.
    descriptors = []
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
                mass = weights[source] * len(left) / len(members)
                descriptors.append((source, a, b, mass))

    def split_arrays(descriptor):
        source, a, b, mass = descriptor
        split = centers.copy()
        split[source] = a
        split = np.vstack((split, b))
        split_weights = np.append(weights, weights[source]-mass)
        split_weights[source] = mass
        return split, split_weights

    def births():
        if len(active) < max_clusters:
            for descriptor in descriptors:
                value, mass = split_arrays(descriptor)
                yield value, mass, "birth"

    def pairs():
        for first in range(len(active)):
            for second in range(first+1, len(active)):
                mass = weights[first]+weights[second]
                center = (weights[first]*centers[first]+weights[second]*centers[second])/mass
                yield first, second, center, mass

    def merges():
        for first, second, center, mass in pairs():
            keep = np.arange(len(active)) != second
            merged, merged_weights = centers.copy(), weights.copy()
            merged[first], merged_weights[first] = center, mass
            yield merged[keep], merged_weights[keep], "merge"

    def relocations():
        for first, second, center, mass in pairs():
            for descriptor in descriptors:
                if descriptor[0] not in (first, second):
                    value, mixed_weights = split_arrays(descriptor)
                    value[first], mixed_weights[first] = center, mass
                    selected = np.arange(len(value)) != second
                    yield value[selected], mixed_weights[selected], "split_merge"

    def transfers():
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
                    yield value, weights.copy(), "transfer"

    for group in zip_longest(births(), transfers(), merges(), relocations()):
        yield from (proposal for proposal in group if proposal is not None)


def _observation_seeds(model, max_clusters, pilot):
    """Deterministic observed-data starts; no tree, partition or accepted fit."""
    observed = np.isfinite(pilot)
    means = np.divide(np.where(observed, pilot, 0.).sum(axis=0), observed.sum(axis=0),
                      out=np.full(model.r, .5), where=observed.any(axis=0))
    points = np.where(observed, pilot, means)
    yield means[None, :]
    if max_clusters >= 2 and len(points) >= 2:
        first = int(np.argmax(np.sum((points-means)**2, axis=1)))
        second = int(np.argmax(np.sum((points-points[first])**2, axis=1)))
        if not np.array_equal(points[first], points[second]):
            yield points[[first, second]]


def search_marginal_mixture(model, initial_fits=None, *, max_clusters, proposal_budget=32,
                            max_iterations=2000, gradient_tolerance=1e-5,
                            progress_callback=None):
    """Bounded experimental birth/merge/split-merge search over mixture fits.

    Seeds may be center arrays or objects with ``centers`` and ``weights``.
    Proposal centers are pilot means, never conditional hard-partition refits.
    The budget counts numerical proposal attempts, not successful fits. Every
    converged candidate is reconciled with the final winner by mixture BIC.
    """
    if model.device != "cpu" or model.fit_multiplicity != "marginal":
        raise ValueError("Joint mixture search requires CPU marginal fitting; no backend or objective fallback")
    if (isinstance(max_clusters, bool) or int(max_clusters) != max_clusters or max_clusters < 1
            or isinstance(proposal_budget, bool) or int(proposal_budget) != proposal_budget
            or proposal_budget < 0):
        raise ValueError("Experimental mixture search requires positive K and nonnegative proposal budget")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    started = perf_counter()
    pilot, _ = model.pilots()
    _record(model, "pilot_seconds", perf_counter()-started)
    direct = initial_fits is None
    seeds = tuple(_observation_seeds(model, max_clusters, pilot) if direct else initial_fits)
    if not seeds:
        raise ValueError("Experimental mixture search requires at least one center seed")
    bank, failures, seen = [], [], set()
    frontier, scheduled = deque(), set()
    attempted, expanded = 0, 0

    def enqueue(fit):
        key = _state_key(fit.centers, fit.weights)
        if key not in scheduled:
            scheduled.add(key)
            frontier.append((fit, None))

    def admit(centers, weights, origin):
        centers = np.asarray(centers, dtype=float)
        weights = np.full(len(centers), 1/len(centers)) if weights is None else np.asarray(weights)
        key = _state_key(centers, weights)
        if key in seen:
            return None
        seen.add(key)
        try:
            fits = fit_marginal_mixture_starts(model, centers, weights, max_iterations=max_iterations,
                                              gradient_tolerance=gradient_tolerance)
        except (MixtureConvergenceError, WeightConvergenceError) as exc:
            failures.append(dict(origin=origin, reason=str(exc),
                                 diagnostics=getattr(exc, "diagnostics", {})))
            return None
        for fit in fits:
            key = _state_key(fit.centers, fit.weights)
            if key not in scheduled:
                bank.append(fit)
                enqueue(fit)
        if progress_callback is not None:
            progress_callback(dict(operation="mixture_search", origin=origin,
                attempted_proposals=attempted, candidates=len(bank),
                best_score=min(fit.score for fit in bank)))
        return fits

    for seed in seeds:
        centers = getattr(seed, "centers", seed)
        if len(centers) > max_clusters:
            raise ValueError("Initial mixture seed exceeds max_clusters")
        admit(centers, getattr(seed, "weights", None), "seed")
    if not bank:
        raise MixtureConvergenceError(f"All experimental mixture seeds failed: {failures}")
    while frontier and attempted < proposal_budget:
        parent, proposals = frontier.popleft()
        if proposals is None:
            proposals = _mixture_proposals(model, parent, pilot, max_clusters)
            expanded += 1
        try:
            started = perf_counter()
            centers, weights, origin = next(proposals)
        except StopIteration:
            continue
        finally:
            _record(model, "proposal_seconds", perf_counter()-started)
        attempted += 1
        admit(centers, weights, origin)
        # A child gets a turn before its parent resumes, so a large parent's
        # proposal list cannot consume the entire budget before recurrence.
        frontier.append((parent, proposals))
    best = min(bank, key=lambda fit: (fit.score, _state_key(fit.centers, fit.weights)))
    return MixtureSearch(best, tuple(bank), dict(proposal_attempts=attempted,
                         proposal_budget=int(proposal_budget), failures=tuple(failures),
                         proposal_quantum=1, expanded_states=expanded,
                         remaining_frontier_states=len(frontier),
                         candidate_count=len(bank), estimator="joint_marginal_mixture",
                         initialization="prepared_observations_no_tree" if direct else "explicit_saved_initializers",
                         all_converged_starts_retained=True,
                         work={name:value for name,value in model.telemetry.items()
                               if name.startswith("mixture_") or name.startswith("weight_")},
                         occupied_map_count_is_component_count=False,
                         exhaustive_search=False, global_optimality_claimed=False))
