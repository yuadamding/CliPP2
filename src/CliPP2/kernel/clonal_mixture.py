"""Role-aware soft mixture on the CPU; no hard-partition fitting or native forest."""
from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import gammaln, logsumexp, xlog1py, xlogy

from ..io.data import TumorData, freeze_record, readonly_array
from .scoring import WeightConvergenceError, fit_mixture_weights


class MixtureConvergenceError(RuntimeError):
    """A candidate could not be qualified for the fitted criterion."""


@dataclass(frozen=True)
class ClonalFit:
    """Active terms: the fixed clonal subset, then the distinct free centers."""

    template: tuple
    centers: np.ndarray
    weights: np.ndarray
    log_likelihood: float
    bic: float
    responsibilities: np.ndarray
    diagnostics: object

    def __post_init__(self):
        for name in ("centers", "weights", "responsibilities"):
            object.__setattr__(self, name, readonly_array(getattr(self, name), dtype=float))
        object.__setattr__(self, "template", tuple(self.template))
        object.__setattr__(self, "diagnostics", freeze_record(self.diagnostics))

    @property
    def biological_count(self):
        return len(self.centers) + bool(self.template)

    @property
    def degrees_of_freedom(self):
        return self.centers.size + len(self.weights) - 1


def _single_copy_coordinate(alt_mass, slopes, ref_mass):
    """Exact scalar optimum by the bracketed monotone binomial score."""
    if not alt_mass:
        return 0.
    active = ref_mass > 0
    slopes, ref_mass = slopes[active], ref_mass[active]
    if not len(slopes):
        return 1.
    if len(slopes) == 1:
        return min(1., float(alt_mass / (slopes[0] * (alt_mass + ref_mass[0]))))
    coefficients = slopes * ref_mass

    def score(center):
        denom = 1. - slopes * center
        if np.any(denom <= 0):
            return -np.inf
        return alt_mass - center * float(np.sum(coefficients / denom))

    if score(1.) >= 0:
        return 1.
    return float(brentq(score, 0., 1., xtol=1e-15))


def _clonal_uniform(model):
    """Log-domain range-one DP by the unique minimum of an admissible vector."""
    mask = model.observed
    major = model.major
    cap = np.where(mask, major, np.iinfo(np.int64).max).min(axis=1)
    partition = np.full(model.n, -np.inf)
    count = np.full_like(partition, -np.inf)
    for anchor in range(1, int(cap.max()) + 1):
        rows = np.flatnonzero(cap >= anchor)
        no_min = np.zeros(len(rows))
        has_min = np.full(len(rows), -np.inf)
        no_min_count = no_min.copy()
        has_min_count = has_min.copy()
        for r in range(model.r):
            used = mask[rows, r]
            if not used.any():
                continue
            idx = rows[used]
            a = model.alt[idx, r]
            b = model.ref[idx, r]
            allow_hi = major[idx, r] > anchor
            slope = model.scaling[idx, r]
            low = np.minimum(1., slope * anchor)
            high = np.minimum(1., slope * (anchor + 1))
            constant = model.constants[idx, r]
            x = constant + xlogy(a, low) + xlog1py(b, -low)
            y = np.where(allow_hi, constant + xlogy(a, high) + xlog1py(b, -high), -np.inf)
            has_min[used] = np.logaddexp(has_min[used] + np.logaddexp(x, y), no_min[used] + x)
            no_min[used] += y
            hi_count = np.where(allow_hi, 0., -np.inf)
            has_min_count[used] = np.logaddexp(
                has_min_count[used] + np.logaddexp(0., hi_count), no_min_count[used])
            no_min_count[used] += hi_count
        partition[rows] = np.logaddexp(partition[rows], has_min)
        count[rows] = np.logaddexp(count[rows], has_min_count)
    return partition - count


class ClonalModel:
    """One prepared population, with fixed clonal columns and single-copy pilots."""

    def __init__(self, data):
        if not isinstance(data, TumorData):
            raise TypeError("ClonalModel requires validated TumorData")
        self.data = data
        mask = data.count_observed & ~data.input_excluded & (data.total_counts > 0)
        ids, regions = data.mutation_ids, data.region_ids
        rows = sorted(np.flatnonzero(mask.any(axis=1)), key=ids.__getitem__)
        cols = sorted(range(len(regions)), key=regions.__getitem__)
        if not rows:
            raise ValueError("No informative positive-depth mutation remains")
        self.active_indices = readonly_array(rows, dtype=int)
        self.region_indices = readonly_array(cols, dtype=int)
        ix = np.ix_(rows, cols)
        self.mutation_ids = tuple(ids[i] for i in rows)
        self.region_ids = tuple(regions[j] for j in cols)
        self.observed = readonly_array(mask[ix])
        self.n, self.r = self.observed.shape
        if not self.observed.any(axis=0).all():
            absent = [r for j, r in enumerate(self.region_ids) if not self.observed[:, j].any()]
            raise ValueError(f"Regions have no informative observations: {absent}")
        self.alt = readonly_array(np.where(self.observed, data.alt_counts[ix], 0.))
        self.depth = readonly_array(np.where(self.observed, data.total_counts[ix], 0.))
        self.ref = readonly_array(self.depth - self.alt)
        self.major = readonly_array(data.major_cn[ix], dtype=np.int64)
        purity = data.purity[ix]
        denominator = (1.-purity)*data.normal_cn[ix] + purity*data.total_cn[ix]
        if not np.isfinite(denominator).all() or np.any(denominator <= 0):
            raise ValueError("Normal-plus-tumor denominator must be finite and positive")
        self.scaling = readonly_array(purity / denominator)
        self.constants = readonly_array(
            gammaln(self.depth + 1.) - gammaln(self.alt + 1.) - gammaln(self.ref + 1.))
        self.log_v = readonly_array(self.regional_columns(np.ones(self.r)).sum(axis=1))
        self.structural_uv_equal = bool(np.all(self.major[self.observed] == 1))
        self.log_u = self.log_v if self.structural_uv_equal else readonly_array(_clonal_uniform(self))
        self._slopes = tuple(np.unique(self.scaling[:, r], return_inverse=True) for r in range(self.r))
        pilots = np.full_like(self.alt, np.nan)
        np.divide(self.alt, self.depth * self.scaling, out=pilots, where=self.observed)
        self.pilots = readonly_array(np.clip(pilots, 0., 1.))
        self.pooled_pilots = readonly_array(
            [self.coordinate_mle(r, np.ones(self.n), fallback=.5) for r in range(self.r)])

    def coordinate_mle(self, region, posterior, fallback):
        slopes, inverse = self._slopes[region]
        alt = float(np.dot(posterior, self.alt[:, region]))
        ref_mass = np.bincount(inverse, weights=posterior*self.ref[:, region], minlength=len(slopes))
        if not alt and not np.any(ref_mass):
            return float(fallback)
        return _single_copy_coordinate(alt, slopes, ref_mass)

    def regional_columns(self, center):
        p = np.minimum(1., self.scaling * center)
        return self.constants + xlogy(self.alt, p) + xlog1py(self.ref, -p)

    def log_columns(self, template, centers):
        columns = [self.log_u if kind == "U" else self.log_v for kind in _template(template)]
        columns.extend(self.regional_columns(center).sum(axis=1) for center in centers)
        if not columns:
            raise ValueError("A mixture needs at least one component")
        return np.column_stack(columns)


def _template(template):
    template = tuple(template)
    if template not in ((), ("U",), ("V",), ("U", "V")):
        raise ValueError("Expected a canonical clonal template (), (U), (V), or (U,V)")
    return template


def _canonical(model, template, centers, weights, *, active_only, promote):
    fixed, free = {}, {}
    for i, kind in enumerate(template):
        if not active_only or weights[i] > 0.:
            key = "V" if model.structural_uv_equal else kind
            fixed[key] = fixed.get(key, 0.) + weights[i]
    for center, w in zip(centers, weights[len(template):]):
        if active_only and not w > 0.:
            continue
        if promote and np.all(center == 1.):
            fixed["V"] = fixed.get("V", 0.) + w
        else:
            key = tuple(center)
            free[key] = free.get(key, 0.) + w
    template = tuple(sorted(fixed))
    # Lexicographic sorting gives stable digest keys across equivalent starts.
    centers = np.asarray(sorted(free)).reshape(-1, model.r)
    weights = np.array([fixed[key] for key in template] + [free[key] for key in sorted(free)])
    return template, centers, weights / weights.sum()


def canonical_components(model, template, centers, weights):
    """Exact, likelihood-preserving role cleanup of an active mixture."""
    return _canonical(model, _template(template), centers, weights, active_only=True, promote=True)


@dataclass
class _Evaluation:
    centers: np.ndarray
    columns: np.ndarray
    weights: np.ndarray
    log_likelihood: float
    rows: np.ndarray
    posterior: np.ndarray
    weight_fit: dict | None


def _evaluate(model, template, centers, weights, *, optimize=True, columns=None):
    if columns is None:
        columns = model.log_columns(template, centers)
    if not np.isfinite(columns).any(axis=1).all():
        raise MixtureConvergenceError("Mixture cannot explain every informative mutation")
    fit = fit_mixture_weights(columns, weights) if optimize else None
    weights = fit["cluster_weights"] if fit is not None else weights / weights.sum()
    with np.errstate(divide="ignore"):
        joint = columns + np.log(weights)
    rows = logsumexp(joint, axis=1)
    if not np.isfinite(rows).all():
        raise MixtureConvergenceError("Active mixture terms leave a mutation unexplained")
    return _Evaluation(centers, columns, weights, float(rows.sum()), rows,
                       np.exp(joint - rows[:, None]), fit)


def center_gradient(model, template, state):
    """Signed raw binomial derivatives, including finite one-sided derivatives at 0/1."""
    grad = np.zeros_like(state.centers)
    alt, ref, a = model.alt, model.ref, model.scaling
    for k, center in enumerate(state.centers):
        weight = state.weights[len(template)+k]
        if not weight:
            continue
        p = np.minimum(1., a*center)
        regional = model.regional_columns(center)
        cavity = np.zeros_like(regional)
        cavity[:, 1:] = np.cumsum(regional[:, :-1], axis=1)
        cavity[:, :-1] += np.cumsum(regional[:, :0:-1], axis=1)[:, ::-1]
        factor = model.constants + cavity + math.log(weight) - state.rows[:, None]
        with np.errstate(divide="ignore"):
            up = factor + np.log(a*alt) + xlogy(np.maximum(0., alt-1.), p) + xlog1py(ref, -p)
            down = factor + np.log(a*ref) + xlogy(alt, p) + xlog1py(np.maximum(0., ref-1.), -p)
        # Saturation is only a safeguarded step/entering-direction signal; an
        # enormous derivative cannot qualify the optimizer as stationary.
        grad[k] = np.sum(np.exp(np.minimum(up, 345.)) - np.exp(np.minimum(down, 345.)), axis=0)
    if not np.isfinite(grad).all():
        raise MixtureConvergenceError("Nonfinite free-center mixture derivative")
    projected = np.where(((state.centers == 0.) & (grad < 0.)) |
                         ((state.centers == 1.) & (grad > 0.)), 0., grad)
    return grad, float(np.max(np.abs(projected), initial=0.) / (1. + model.n))


def _em(model, template, state):
    new = state.centers.copy()
    for k in range(len(new)):
        posterior = state.posterior[:, len(template)+k]
        for r in range(model.r):
            new[k, r] = model.coordinate_mle(r, posterior, state.centers[k, r])
    return new


def _rescue(model, template, state, iterations, tolerance):
    """Bounded ascent of the actual center likelihood, then a full weight solve."""
    best = state
    shape = state.centers.shape

    def objective(vector):
        nonlocal best
        try:
            trial = _evaluate(model, template, vector.reshape(shape), state.weights, optimize=False)
        except MixtureConvergenceError:
            return np.inf, np.zeros_like(vector)
        if trial.log_likelihood >= best.log_likelihood:
            best = trial
        return -trial.log_likelihood / (1.+model.n), -center_gradient(model, template, trial)[0].ravel() / (1.+model.n)

    minimize(objective, state.centers.ravel(), jac=True, method="L-BFGS-B",
             bounds=[(0., 1.)]*state.centers.size,
             options={"maxiter": iterations, "ftol": 1e-15, "gtol": tolerance, "maxls": 32})
    return _evaluate(model, template, best.centers, state.weights, columns=best.columns)


def _refine(model, template, state, maximum, tolerance, trace):
    for iteration in range(maximum + 1):
        gradient, residual = center_gradient(model, template, state)
        # A coordinate infinitesimally inside its active face may have an
        # outward derivative while the objective is already rounded to its
        # boundary value. Evaluate that exact face, not a relaxed KKT gate.
        epsilon = 4.*np.finfo(float).eps
        near_lower = (state.centers > 0.) & (state.centers <= epsilon) & (gradient < 0.)
        near_upper = (state.centers < 1.) & (state.centers >= 1.-epsilon) & (gradient > 0.)
        if near_lower.any() or near_upper.any():
            centers = np.where(near_lower, 0., np.where(near_upper, 1., state.centers))
            trial = _evaluate(model, template, centers, state.weights,
                              optimize=state.weight_fit is not None)
            roundoff = epsilon * (1. + abs(state.log_likelihood))
            if trial.log_likelihood >= state.log_likelihood - roundoff:
                state = trial
                trace.append(state.log_likelihood)
                _, residual = center_gradient(model, template, state)
        if residual <= tolerance:
            if state.weight_fit is None:
                state = _evaluate(model, template, state.centers, state.weights, columns=state.columns)
                trace.append(state.log_likelihood)
                _, residual = center_gradient(model, template, state)
            if residual <= tolerance:
                return state, residual, iteration
        if iteration == maximum:
            break
        if iteration % 8 == 7:
            new = _rescue(model, template, state, min(32, maximum-iteration), tolerance)
        else:
            centers = _em(model, template, state)
            new = _evaluate(model, template, centers, state.posterior.mean(axis=0),
                            optimize=False)
            if new.log_likelihood <= state.log_likelihood + 1e-10:
                new = _rescue(model, template, state, min(32, maximum-iteration), tolerance)
        if new.log_likelihood >= state.log_likelihood:
            state = new
        trace.append(state.log_likelihood)
    raise MixtureConvergenceError(f"Free centers failed projected stationarity: residual={residual:.3g}")


def fit_start(model, template, centers, weights=None, *, max_iterations=2000, gradient_tolerance=1e-5):
    """Fit one seed, keeping all permitted terms for its weight qualification."""
    template = _template(template)
    centers = np.asarray(centers, dtype=float)
    if not centers.size:
        centers = centers.reshape(0, model.r)
    if (centers.ndim != 2 or centers.shape[1] != model.r or not np.isfinite(centers).all()
            or np.any((centers < 0.) | (centers > 1.))):
        raise ValueError("Initial free centers must be finite K by R CCFs in [0, 1]")
    if (type(max_iterations) is not int or max_iterations < 1
            or not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0.):
        raise ValueError("Positive mixture iteration limit and gradient tolerance required")
    k = len(template) + len(centers)
    if k == 0:
        raise ValueError("Cannot fit the empty mixture")
    weights = np.full(k, 1./k) if weights is None else np.asarray(weights, dtype=float)
    if (weights.shape != (k,) or not np.isfinite(weights).all() or np.any(weights < 0.)
            or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0.)):
        raise ValueError("Initial mixture weights must lie on the probability simplex")
    requested_template = template
    template, centers, weights = _canonical(model, template, centers, weights,
                                             active_only=False, promote=False)
    interior_restart = False
    try:
        try:
            state = _evaluate(model, template, centers, weights)
        except MixtureConvergenceError:
            if not len(centers):
                raise
            # A seed at an infeasible face is not a restriction on free CCFs.
            state = _evaluate(model, template, (centers+.5)/2., weights)
            interior_restart = True
        trace = [state.log_likelihood]
        iterations = 0
        while True:
            state, residual, steps = _refine(
                model, template, state, max_iterations-iterations, gradient_tolerance, trace)
            iterations += steps
            canonical = _canonical(model, template, state.centers, state.weights,
                                   active_only=False, promote=True)
            new_template, new_centers, new_weights = canonical
            if new_template == template and np.array_equal(new_centers, state.centers):
                break
            template = new_template
            simplified = _evaluate(model, template, new_centers, new_weights)
            if simplified.log_likelihood < state.log_likelihood - 1e-8:
                raise MixtureConvergenceError("Exact role canonicalization reduced likelihood")
            state = simplified
            trace.append(state.log_likelihood)
    except WeightConvergenceError as exc:
        raise MixtureConvergenceError(str(exc)) from exc
    # Zero weights were allowed to enter up to the final weight MLE. Only then
    # remove them for canonical output and for biological/parameter counting.
    used = state.weights > 0
    active_template = tuple(kind for i, kind in enumerate(template) if used[i])
    active_centers = state.centers[used[len(template):]]
    active_weights = state.weights[used]
    ll = state.log_likelihood
    bic = -2.*ll + (active_centers.size + len(active_weights)-1)*math.log(model.n)
    return ClonalFit(active_template, active_centers, active_weights, ll, bic,
                     state.posterior[:, used], dict(
                         requested_template=requested_template, qualified_template=template,
                         interior_restart=interior_restart, iterations=iterations,
                         log_likelihood_trace=trace, projected_gradient_residual=residual,
                         gradient_tolerance=gradient_tolerance, global_optimality_claimed=False,
                         weight_optimality_gap=state.weight_fit["weight_optimality_gap"],
                         weight_active_score_gap=state.weight_fit["weight_active_score_gap"]))
