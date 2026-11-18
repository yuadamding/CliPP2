"""Deterministic discovery and bounded refinement for the single-copy mixture."""
from collections import Counter, deque
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral

import numpy as np
from scipy.special import logsumexp

from .clonal_mixture import MixtureConvergenceError, fit_start
from .scoring import WeightConvergenceError


@dataclass(frozen=True, slots=True)
class SearchResult:
    best: object
    diagnostics: dict


@dataclass(frozen=True, slots=True)
class _Start:
    template: tuple[str, ...]
    centers: np.ndarray
    weights: np.ndarray | None = None


def _interior(centers):
    # Only starts are nudged. The optimizer sees the full closed [0,1] box.
    # In particular, no seed is converted to fixed V before it can move.
    return np.clip(centers, 1e-4, 1.-1e-4)


class _PilotGeometry:
    def __init__(self, model):
        self.order = np.argsort(np.asarray(model.mutation_ids, dtype=str), kind='stable')
        self.points = np.asarray(model.pilots, dtype=float)[self.order]
        self.observed = np.asarray(model.observed)[self.order]
        self.pooled = np.asarray(model.pooled_pilots, dtype=float)
        if (self.points.shape != (model.n, model.r) or self.pooled.shape != (model.r,)
                or not np.array_equal(np.isfinite(self.points), self.observed)
                or not self.observed.any(axis=1).all() or not np.isfinite(self.pooled).all()):
            raise ValueError('Pilot geometry requires finite observed pilots and a finite pooled fallback')
        self.values = np.where(self.observed, self.points, 0.)
        self.counts = self.observed.sum(axis=1)

    def center(self, rows, fallback=None):
        fallback = self.pooled if fallback is None else fallback
        count = self.observed[rows].sum(axis=0)
        return np.divide(self.values[rows].sum(axis=0), count,
                         out=np.array(fallback, dtype=float, copy=True), where=count > 0)

    def distance(self, center, rows=slice(None)):
        return (np.where(self.observed[rows], (self.values[rows]-center)**2, 0.).sum(axis=1)
                / self.counts[rows])

    def split(self, rows, *, median=False):
        """Split on observed coordinates, assigning missing rows by other coordinates."""
        best = None
        for r in range(self.points.shape[1]):
            seen = rows[self.observed[rows, r]]
            if len(seen) < 2:
                continue
            ordered = seen[np.argsort(self.points[seen, r], kind='stable')]
            gaps = np.diff(self.points[ordered, r])
            gap = float(gaps.max())
            if gap == 0.:
                continue
            cut = len(ordered)//2 if median else int(gaps.argmax())+1
            # In the median variation, do not split identical coordinates if
            # another boundary is available. IDs already break ordering ties.
            if median and gaps[cut-1] == 0.:
                boundaries = np.flatnonzero(gaps > 0)+1
                cut = int(boundaries[np.argmin(abs(boundaries-cut))])
            key = (-gap, -min(cut, len(ordered)-cut), r)
            if best is None or key < best[0]:
                best = key, ordered[:cut], ordered[cut:], r
        if best is None:
            middle = len(rows)//2
            return rows[:middle], rows[middle:]
        _, left, right, r = best
        missing = rows[~self.observed[rows, r]]
        da = self.distance(self.center(left), missing)
        db = self.distance(self.center(right), missing)
        choose_left = (da < db) | ((da == db) & (len(left) <= len(right)))
        return (np.sort(np.r_[left, missing[choose_left]]),
                np.sort(np.r_[right, missing[~choose_left]]))

    def seed_families(self, maximum):
        """Increment the two families without an N by N distance matrix."""
        yield np.empty((0, len(self.pooled))), np.empty((0, len(self.pooled)))
        selected, distances = [], None
        available = np.ones(len(self.points), dtype=bool)
        groups = [np.arange(len(self.points))]
        group_centers = [self.center(groups[0])]
        for k in range(1, maximum+1):
            row = (int(self.distance(self.pooled).argmin()) if distances is None
                   else int(np.argmax(np.where(available, distances, -np.inf))))
            available[row] = False
            point = np.where(self.observed[row], self.values[row], self.pooled)
            selected.append(point)
            dist = self.distance(point)
            distances = dist if distances is None else np.minimum(distances, dist)
            if k > 1:
                j = min((i for i, group in enumerate(groups) if len(group) > 1), key=lambda i: (
                    -float(self.distance(group_centers[i], groups[i]).sum()), -len(groups[i]), int(groups[i][0])))
                left, right = self.split(groups[j])
                groups[j] = left
                groups.append(right)
                group_centers[j] = self.center(left)
                group_centers.append(self.center(right))
            yield _interior(selected), _interior(group_centers)


def _key(template, centers, weights):
    """Permuting free components does not create a distinct starting model."""
    return (tuple(template), tuple(map(float, weights[:len(template)])),
            tuple(sorted((tuple(map(float, row)), float(w))
                         for row, w in zip(centers, weights[len(template):]))))


def _verify_incumbent(model, fit):
    """Refinement may only compete against the same population and objective."""
    joint = model.log_columns(fit.template, fit.centers)+np.log(fit.weights)
    rows = logsumexp(joint, axis=1)
    score = -2.*rows.sum()+(fit.centers.size+len(fit.weights)-1)*np.log(model.n)
    if (fit.responsibilities.shape != joint.shape
            or not np.isclose(score, fit.bic, atol=1e-8, rtol=1e-12)
            or not np.isclose(rows.sum(), fit.log_likelihood, atol=1e-8, rtol=1e-12)
            or not np.allclose(np.exp(joint-rows[:, None]), fit.responsibilities, atol=1e-10, rtol=1e-10)):
        raise ValueError('Stage 1 incumbent does not reproduce on the Stage 2 observations')


def _proposals(model, geometry, parent, family, capacity, templates):
    template, centers, weights = parent.template, parent.centers, parent.weights
    c, q = len(template), len(centers)
    fixed, free = weights[:c], weights[c:]
    can_grow = parent.biological_count < capacity

    def start(new_centers, new_free, new_template=template, new_fixed=fixed):
        return _Start(new_template, np.asarray(new_centers).reshape(-1, model.r), np.r_[new_fixed, new_free])

    if family == 'template':
        for permitted in templates:
            if permitted == template or q+bool(permitted) > capacity or not (q or permitted):
                continue
            retained = {term: weight for term, weight in zip(template, fixed)}
            new_fixed = np.array([retained.get(term, min(.05, 1./(c+q+1))) for term in permitted])
            yield start(centers, free, permitted, new_fixed)
        return
    if family == 'promotion':
        nearest = np.argsort(np.sum((centers-1.)**2, axis=1), kind='stable')
        permitted = tuple(term for term in ('U', 'V') if term in template or term == 'V')
        for i in nearest:
            terms = {term: weight for term, weight in zip(template, fixed)}
            terms['V'] = terms.get('V', 0.)+free[i]
            keep = np.arange(q) != i
            yield start(centers[keep], free[keep], permitted, [terms[term] for term in permitted])
        return

    def merged(first, second):
        total = free[first]+free[second]
        return ((free[first]*centers[first]+free[second]*centers[second])/total, total)

    if family == 'merge':
        for i, j in combinations(range(q), 2):
            value, mass = centers.copy(), free.copy()
            value[i], mass[i] = merged(i, j)
            keep = np.arange(q) != j
            yield start(value[keep], mass[keep])
        return
    if family == 'birth':
        if not can_grow:
            return
        original = model.log_columns(template, centers)+np.log(weights)[None, :]
        log_g = logsumexp(original, axis=1)
        # Use a bounded shortlist; likelihood-ranking every raw mutation
        # vector against every observation would require N squared work.
        depth = model.depth.sum(axis=1)[geometry.order]
        poor = np.argsort(log_g[geometry.order]/depth, kind='stable')[:16]
        pool = np.vstack((geometry.pooled, np.where(geometry.observed[poor], geometry.values[poor], geometry.pooled)))
        pool = _interior(pool)
        _, first = np.unique(pool, return_index=True, axis=0)
        pool = pool[np.sort(first)]
        # log mean_i(S_i/g_i) is monotone in D = sum_i(S_i/g_i - 1)
        # and does not overflow for extreme directional likelihood gains.
        ratio = logsumexp(model.log_columns((), pool)-log_g[:, None], axis=0)-np.log(model.n)
        for i in np.argsort(-ratio, kind='stable'):
            mass = min(.1, 1./(q+c+1))
            yield start(np.vstack((centers, pool[i])), np.r_[(1.-mass)*free, mass],
                        template, (1.-mass)*fixed)
        return

    if (family == 'split' and (not can_grow or not q)) or (family == 'split_merge' and q < 3) or (family == 'transfer' and q < 2):
        return
    responsibility = parent.responsibilities[geometry.order]
    biological = (np.c_[responsibility[:, :c].sum(axis=1), responsibility[:, c:]]
                  if c else responsibility)
    labels = np.argmax(biological, axis=1)-bool(c)
    members = [np.flatnonzero(labels == i) for i in range(q)]
    if family == 'transfer':
        columns = model.log_columns((), centers)[geometry.order]
        for source, rows in enumerate(members):
            if len(rows) < 2:
                continue
            for target in range(q):
                if source == target:
                    continue
                # The source column is finite for all its MAP members.
                advantage = columns[rows, target]-columns[rows, source]
                ordered = rows[np.argsort(-advantage, kind='stable')]
                for count in sorted({1, len(rows)//2}):
                    if not np.isfinite(columns[ordered[count-1], target]):
                        continue
                    value = centers.copy()
                    value[source] = _interior(geometry.center(ordered[count:], centers[source]))
                    value[target] = _interior(geometry.center(np.r_[members[target], ordered[:count]], centers[target]))
                    yield start(value, free)
        return

    for source, rows in enumerate(members):
        if len(rows) < 2:
            continue
        for median in (False, True):
            left, right = geometry.split(rows, median=median)
            a, b = _interior(geometry.center(left, centers[source])), _interior(geometry.center(right, centers[source]))
            if np.array_equal(a, b):
                continue
            value, mass = np.vstack((centers, b)), np.r_[free, free[source]*len(right)/len(rows)]
            value[source] = a
            mass[source] -= mass[-1]
            if family == 'split':
                yield start(value, mass)
                continue
            for i, j in combinations((k for k in range(q) if k != source), 2):
                combined, combined_mass = value.copy(), mass.copy()
                combined[i], combined_mass[i] = merged(i, j)
                keep = np.arange(q+1) != j
                yield start(combined[keep], combined_mass[keep])


def search(model, *, max_clusters, proposal_budget=32, incumbent=None, covered_counts=(),
           progress_callback=None, comparison_seeds=(), max_iterations=2000, gradient_tolerance=1e-5):
    """Cover every biological count before bounded optional refinement.

    The initial search need not have used this capacity: Stage 2 can carry an
    incumbent and verified count coverage, preserving its exact BIC even when
    subsequent attempts fail. Comparison seeds are opt-in external A results,
    not another production estimator. All candidate ordering uses BIC.
    """
    if (isinstance(max_clusters, bool) or not isinstance(max_clusters, Integral) or max_clusters < 1
            or isinstance(proposal_budget, bool) or not isinstance(proposal_budget, Integral) or proposal_budget < 0):
        raise ValueError('Search requires a positive cluster capacity and a nonnegative proposal budget')
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError('progress_callback must be callable or None')
    if min(model.n, model.r) < 1:
        raise ValueError('Search needs at least one informative mutation and region')
    capacity = min(model.n, int(max_clusters))
    covered_counts = tuple(covered_counts)
    if any(isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= capacity for k in covered_counts):
        raise ValueError('Covered counts must be integers within the biological capacity')
    covered = set(map(int, covered_counts))
    if covered and incumbent is None:
        raise ValueError('Covered counts require their Stage 1 incumbent')
    if incumbent is not None and incumbent.biological_count > capacity:
        raise ValueError('Stage 2 biological capacity cannot exclude its Stage 1 incumbent')
    # Use canonical V to represent the structurally identical single-copy pair.
    templates = ((), ('V',)) if model.structural_uv_equal else ((), ('U',), ('V',), ('U', 'V'))
    geometry = _PilotGeometry(model)
    seen, records, attempts, skipped = {}, [], Counter(), Counter()
    bank, accepted = {}, 0
    best = incumbent
    families = ('birth', 'split', 'merge', 'transfer', 'split_merge', 'template', 'promotion')
    agenda = {name: deque() for name in families}

    def fit_key(fit):
        return _key(fit.template, fit.centers, fit.weights)

    def rank(fit):
        return fit.bic, fit_key(fit)

    def account(fit, *, preserve=False):
        nonlocal best, accepted
        if fit.biological_count > capacity or not np.isfinite(fit.bic):
            raise ValueError('Numerically accepted state exceeds the search contract')
        accepted += 1
        if best is None or rank(fit) < rank(best):
            best = fit
        # Compact frontier: keep the best state per structural stratum. The
        # BIC of every qualified fit was compared before retaining the frontier.
        stratum = (fit.biological_count, fit.template)
        if preserve or stratum not in bank or rank(fit) < rank(bank[stratum]):
            bank[stratum] = fit
            return True
        return False

    def emit(record):
        if progress_callback is not None:
            progress_callback(dict(operation='clonal_mixture_search', **record,
                                   best_bic=None if best is None else float(best.bic)))

    def admit(seed, family, phase, *, bio_count=None):
        centers = np.asarray(seed.centers, dtype=float)
        if not centers.size:
            centers = centers.reshape(0, model.r)
        template = tuple(seed.template)
        if (centers.ndim != 2 or centers.shape[1] != model.r
                or len(centers)+bool(template)>capacity or (not len(centers) and not template)):
            raise ValueError('Seed center dimensions or biological capacity are invalid')
        q = len(centers)+len(template)
        weights = np.full(q, 1./q) if seed.weights is None else np.asarray(seed.weights, dtype=float)
        if (weights.shape != (q,) or not np.isfinite(weights).all()
                or np.any(weights < 0.) or not weights.sum() > 0.):
            raise ValueError('Seed weights must be finite and nonnegative with positive total mass')
        weights = weights/weights.sum()
        key = _key(template, centers, weights)
        record = dict(family=family, phase=phase, template=list(template), free_count=len(centers),
                      requested_biological_count=len(centers)+bool(template) if bio_count is None else bio_count)
        if key in seen:
            skipped[family] += 1
            records.append(dict(record, status='duplicate_start', reason='exact_same_template_centers_and_weights'))
            return seen[key], False, None
        attempts[family] += 1
        try:
            fit = fit_start(model, template, centers, weights, max_iterations=max_iterations,
                            gradient_tolerance=gradient_tolerance)
        except (MixtureConvergenceError, WeightConvergenceError) as exc:
            seen[key] = False
            records.append(dict(record, status='failed', reason=str(exc)))
            emit(records[-1])
            return False, True, None
        seen[key] = True
        advancing = account(fit)
        records.append(dict(record, status='accepted', bic=float(fit.bic), active_template=list(fit.template),
                            active_biological_count=fit.biological_count))
        emit(records[-1])
        return True, True, fit if advancing else None

    if incumbent is not None:
        _verify_incumbent(model, incumbent)
        account(incumbent, preserve=True)
        admit(_Start(incumbent.template, incumbent.centers, incumbent.weights), 'incumbent_warm_start', 'warm_start')

    # A free-count start is built once and evaluated under each eligible
    # template. A two-term clonal template uses only one biological slot.
    coverage = {k: [] for k in range(1, capacity+1) if k not in covered}
    for free_count, seeds in enumerate(geometry.seed_families(capacity)):
        for template in templates:
            biological = free_count+bool(template)
            if biological not in coverage:
                continue
            for family, centers in zip(('farthest_point', 'recursive_split'), seeds):
                result, _, _ = admit(_Start(template, centers), family, 'mandatory', bio_count=biological)
                coverage[biological].append(result)
    covered.update(k for k, ok in coverage.items() if ok and all(ok))
    for seed in comparison_seeds:
        cert = _Start(tuple(getattr(seed, 'template', ())), np.asarray(getattr(seed, 'centers', seed)),
                      getattr(seed, 'weights', None))
        if len(cert.centers)+bool(cert.template) > capacity:
            raise ValueError('Comparison seed exceeds the biological capacity')
        admit(cert, 'comparison_seed', 'comparison')
    if best is None:
        failures = [entry for entry in records if entry['status'] == 'failed']
        raise MixtureConvergenceError(f'No numerically qualified mixture start; failures: {failures}')

    def schedule(parent):
        for family, queue in agenda.items():
            queue.append((parent, _proposals(model, geometry, parent, family, capacity, templates)))

    # Round-robin by operator AND distinct parent. Improving offspring can
    # spawn more proposals; overrepresented K or operators cannot monopolize
    # the budget merely by generating more duplicate starts.
    proposal_attempts = 0
    if proposal_budget:
        for fit in sorted(bank.values(), key=rank):
            schedule(fit)
        while proposal_attempts < proposal_budget and any(agenda.values()):
            for family, queue in agenda.items():
                if proposal_attempts == proposal_budget:
                    break
                while queue:
                    parent, iterator = queue.popleft()
                    if bank.get((parent.biological_count, parent.template)) is not parent:
                        continue
                    seed = next(iterator, None)
                    if seed is None:
                        continue
                    queue.append((parent, iterator))
                    _, attempted, result = admit(seed, family, 'optional')
                    if not attempted:  # Exact duplicate: no numerical work spent.
                        continue
                    proposal_attempts += 1
                    if result is not None:
                        schedule(result)
                    break

    if incumbent is not None and best.bic > incumbent.bic:
        raise AssertionError('Stage 2 lost the Stage 1 BIC incumbent')
    return SearchResult(best, dict(requested_capacity=int(max_clusters), biological_capacity=capacity,
        templates=[list(template) for template in templates],
        structural_uv_equivalence=bool(model.structural_uv_equal),
        structurally_equivalent_templates_skipped=[['U'], ['U', 'V']] if model.structural_uv_equal else [],
        mandatory_attempts=attempts['farthest_point']+attempts['recursive_split'],
        covered_counts=sorted(covered), incomplete_counts=sorted(set(range(1, capacity+1))-covered),
        inherited_covered_counts=list(covered_counts), proposal_budget=int(proposal_budget),
        proposal_attempts=proposal_attempts, attempts_by_family=dict(attempts), duplicates_by_family=dict(skipped),
        qualified_attempts=accepted-int(incumbent is not None), frontier_strata=len(bank), attempts=records,
        incumbent_bic=None if incumbent is None else float(incumbent.bic),
        selected_bic=float(best.bic), selected_biological_count=best.biological_count,
        exhaustive_search=False, global_optimality_claimed=False))
