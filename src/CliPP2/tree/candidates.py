"""Shared registry of compact memberships, refit attempts and immutable states."""

from dataclasses import dataclass, field, replace
from time import perf_counter
from types import MappingProxyType

import numpy as np

from ..kernel.model import FitResult
from ..kernel.topology import (LabelOrder, Membership, PreparedPartition,
                               canonical_partition, partition_from_cuts)


def _plain(value):
    if isinstance(value, (dict, MappingProxyType)):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return list(map(_plain, value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value.item() if isinstance(value, np.generic) else value


@dataclass(eq=False)
class Candidate:
    fit: FitResult
    provenance: list = field(default_factory=list)
    state_index: int | None = None

    @property
    def score(self):
        return float(self.fit.score)

    @property
    def cuts(self):
        return self.fit.membership.cuts

    @property
    def membership_kind(self):
        return self.fit.membership.kind

    @property
    def requested_k(self):
        return self.provenance[0].get("requested_k", len(self.fit.centers))

    @property
    def kind(self):
        return self.provenance[0].get("route", "unspecified")

    @property
    def diagnostics(self):
        return {"provenance": _plain(self.provenance), "refit": _plain(self.fit.diagnostics)}


@dataclass(frozen=True, slots=True)
class RefitAttempt:
    """Exact refit identity and outcome, including unresolved cached failures."""

    key: tuple
    outcome: str
    reason: str | None
    state_index: int | None
    fit: FitResult | None = None


def _outcome_kind(fitted):
    if fitted.eligible:
        return "eligible"
    if str(fitted.reason).startswith(("unsupported_cluster_region:", "empty_cluster_region_box:")):
        return "structural_exclusion"
    if (fitted.reason == "zero_mixture_weight"
            and fitted.diagnostics.get("rejection_kind") != "weight_convergence"):
        return "completed_ineligible"
    # An unfamiliar rejection must not become a structural search exclusion.
    return "unresolved_numerical"


def score_key(candidate):
    tree = candidate.membership_kind == "tree"
    return (candidate.score, len(candidate.fit.centers), 0 if tree else 1,
            candidate.cuts if tree else LabelOrder(candidate.fit.membership))


def _canonical_inputs(labels, centers, n, r):
    labels = np.asarray(labels)
    if (labels.shape != (n,) or labels.dtype.kind not in "iuf"
            or not np.isfinite(labels).all() or np.any(labels != np.rint(labels))):
        raise ValueError("Candidate labels must be one finite integer per mutation")
    occupied = np.unique(labels)
    if not np.array_equal(occupied, np.arange(len(occupied))):
        raise ValueError("Candidate labels must be occupied consecutive IDs starting at zero")
    labels = labels.astype(np.int64)
    if centers is None:
        return canonical_partition(labels), None
    centers = np.asarray(centers, dtype=float)
    if centers.shape != (len(occupied), r) or not np.isfinite(centers).all():
        raise ValueError("Supplied centers must be finite occupied q by R rows aligned with labels")
    return canonical_partition(labels, centers)


class CandidateBank:
    """Retain eligible states before advancement; cache only exact attempts."""

    numerical_policy = "regional_conditional_refit_v1"

    def __init__(self, model, tree, *, progress=None, request_progress=None,
                 weight_batch_size=1, device_proposals=False):
        if type(weight_batch_size) is not int or weight_batch_size < 1:
            raise ValueError("weight_batch_size must be a positive integer")
        if type(device_proposals) is not bool:
            raise TypeError("device_proposals must be a boolean")
        self.model, self.tree = model, tree
        self.progress = progress
        self.request_progress = request_progress
        self.candidates, self.general_rejections = [], []
        self.by_cuts, self.by_membership, self.ineligible = {}, {}, {}
        self.best_by_q, self.tree_by_q, self.best_overall = {}, {}, None
        self._attempts, self._states, self._structural = {}, {}, {}
        self._outcomes, self._outcome_records = {}, []
        self.duplicate_routes = self.refit_calls = 0
        self.attempt_cache_hits = self.structural_cache_hits = 0
        self.refit_seconds = 0.0
        self.registration_seconds = 0.0
        self._tree_identity = tree.identity
        self.weight_batch_size = weight_batch_size
        self.device_proposals = device_proposals

    def prepare(self, cuts, labels=None, *, parent=None, fitted=None, columns=None):
        """Prepare one active expansion, retaining unchanged group handles.

        This internal entry point accepts canonical labels. Public consider()
        remains the validating boundary. Partitions are not retained by the
        candidate bank: inactive tree states continue to store only cut IDs.
        """
        if labels is None:
            labels = partition_from_cuts(self.tree, cuts)
        counts = np.bincount(labels)
        reusable = {}
        if parent is not None:
            for index, group in enumerate(parent.groups):
                target = int(labels[group.members[0]])
                if counts[target] == len(group.members) and np.all(labels[group.members] == target):
                    reusable[target] = (group, parent.coordinate_fits[index]
                                        if parent.coordinate_fits else None,
                                        parent.group_fits[index] if parent.group_fits else None)
        prepared, reused, reused_groups = [], [], []
        for cluster in range(int(np.max(labels)) + 1):
            group, previous, previous_group = reusable.get(cluster, (None, None, None))
            prepared.append(self.model.prepare_group(np.flatnonzero(labels == cluster))
                            if group is None else group)
            reused.append(previous)
            reused_groups.append(previous_group)
        groups, coordinate_fits, group_fits = tuple(prepared), tuple(reused), tuple(reused_groups)
        if fitted is not None and fitted.fit_objective_id != self.model.method_identity["fit_objective_id"]:
            raise ValueError("Prepared fits belong to another fitting objective")
        if fitted is not None and fitted.coordinate_fits is not None:
            coordinate_fits = tuple(tuple(tuple(value) for value in group)
                                    for group in fitted.coordinate_fits)
        if fitted is not None and fitted.group_fits is not None:
            group_fits = fitted.group_fits
        if columns is not None and columns.aggregation != "marginal":
            raise ValueError("Prepared parents require marginal selection columns")
        return PreparedPartition(self.model.identity, self._tree_identity,
                                 Membership(self.tree, cuts) if cuts is not None else Membership(explicit_labels=labels),
                                 labels, groups, coordinate_fits,
                                 columns if parent is None else parent.selection_columns, group_fits)

    def consider_cuts(self, cuts, centers, provenance):
        """Trusted canonical cuts: exact cache lookup precedes labels/groups."""
        membership = ("tree", cuts)
        found, candidate = self._cached(membership, centers, provenance)
        if not found:
            candidate = self._consider(self.prepare(cuts), centers, provenance)
        if self.progress is not None:
            self.progress(self, provenance)
        return candidate

    def consider_prepared(self, partition, centers, provenance):
        if (partition.tree_identity != self._tree_identity
                or partition.model_identity != self.model.identity):
            raise ValueError("Prepared partition belongs to another model/tree")
        found, candidate = self._cached(partition.membership.key, centers, provenance)
        if not found:
            candidate = self._consider(partition, centers, provenance)
        if self.progress is not None:
            self.progress(self, provenance)
        return candidate

    def consider(self, cuts, labels, centers, provenance, *, parent=None):
        raw = np.asarray(cuts)
        if raw.ndim != 1 or not np.isfinite(raw).all() or np.any(raw != np.rint(raw)):
            raise ValueError("Tree cuts must be finite integer edge IDs")
        cuts = tuple(sorted(map(int, cuts)))
        if len(set(cuts)) != len(cuts) or any(not 0 <= edge < len(self.tree.edges) for edge in cuts):
            raise ValueError("Tree cuts must be distinct valid edge IDs")
        labels, centers = _canonical_inputs(labels, centers, self.model.n, self.model.r)
        if not np.array_equal(labels, partition_from_cuts(self.tree, cuts)):
            raise ValueError("Candidate labels and frozen-tree cuts disagree")
        found, candidate = self._cached(("tree", cuts), centers, provenance)
        if not found:
            candidate = self._consider(self.prepare(cuts, labels, parent=parent), centers, provenance)
        if self.progress is not None:
            self.progress(self, provenance)
        return candidate

    def consider_membership(self, labels, initial_centers, provenance, *, parent=None):
        labels, centers = _canonical_inputs(labels, initial_centers, self.model.n, self.model.r)
        found, candidate = self._cached(("general", labels.tobytes()), centers, provenance)
        if not found:
            candidate = self._consider(self.prepare(None, labels, parent=parent), centers, provenance)
        if self.progress is not None:
            self.progress(self, provenance)
        return candidate

    def _cached(self, membership, centers, provenance):
        provenance = _plain(provenance)
        if membership in self._structural:
            self.structural_cache_hits += 1
            self._structural[membership]["provenance"].append(provenance)
            return True, None
        attempt = (membership, None if centers is None else centers.tobytes(), self.numerical_policy)
        if attempt in self._attempts:
            self.attempt_cache_hits += 1
            candidate = self._attempts[attempt]
            if candidate is not None:
                candidate.provenance.append(provenance)
                self.duplicate_routes += 1
            return True, candidate
        return False, None

    def _consider(self, partition, centers, provenance):
        provenance = _plain(provenance)
        started = perf_counter()
        self.refit_calls += 1
        try:
            fitted = self.model.refit_prepared(partition, initial_centers=centers,
                progress_callback=None if self.request_progress is None else
                lambda operation: self.request_progress(self, provenance, operation))
        finally:
            self.refit_seconds += perf_counter()-started
        return self._commit(partition, centers, provenance, fitted)

    def _commit(self, partition, centers, provenance, fitted):
        started = perf_counter()
        try:
            return self._commit_result(partition, centers, provenance, fitted)
        finally:
            self.registration_seconds += perf_counter()-started

    def _commit_result(self, partition, centers, provenance, fitted):
        """Publish exactly one completed fit, always in original request order."""
        membership, cuts = partition.membership.key, partition.cuts
        provenance = _plain(provenance)
        attempt = (membership, None if centers is None else centers.tobytes(), self.numerical_policy)
        outcome = _outcome_kind(fitted)
        self._attempts[attempt] = None
        if not fitted.eligible:
            rejection = {"cuts" if cuts is not None else "membership": list(cuts) if cuts is not None else membership,
                         "reason": fitted.reason, "provenance": [provenance]}
            if cuts is None:
                self.general_rejections.append(rejection)
            else:
                self.ineligible[cuts] = rejection
            if outcome == "structural_exclusion":
                self._structural[membership] = rejection
            self._store_outcome(attempt, outcome, fitted.reason, None,
                                fitted if outcome == "completed_ineligible" else None)
            return None
        if fitted.membership is not partition.membership:
            raise ValueError("Conditional refit changed the proposed membership")
        q = len(partition.groups)
        if fitted.centers.shape != (q, self.model.r) or not np.isfinite(fitted.centers).all():
            raise ValueError("Eligible fitted centers must be finite occupied q by R rows")
        weights = fitted.weights
        if (weights.shape != (q,) or not np.isfinite(weights).all()
                or np.any(weights <= 0) or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
            raise ValueError("Eligible fitted weights must be positive normalized occupied rows")
        if not np.isfinite([fitted.conditional_log_likelihood, fitted.mixture_log_likelihood,
                            fitted.complexity_penalty, fitted.score]).all():
            raise ValueError("An eligible candidate has nonfinite likelihood, penalty or score")
        state_key = membership, fitted.centers.tobytes(), weights.tobytes()
        candidate = self._states.get(state_key)
        if candidate is None:
            candidate = Candidate(fitted, [provenance], len(self.candidates))
            self.candidates.append(candidate)
            self._states[state_key] = candidate
        else:
            if any(getattr(candidate.fit, name) != getattr(fitted, name) for name in (
                    "conditional_log_likelihood", "mixture_log_likelihood", "complexity_penalty", "score")):
                raise ValueError("Identical fitted parameters produced inconsistent candidate scores")
            candidate.provenance.append(provenance)
            self.duplicate_routes += 1
        self._attempts[attempt] = candidate
        self._store_outcome(attempt, outcome, fitted.reason, candidate.state_index)
        index, key = (self.by_cuts, cuts) if cuts is not None else (self.by_membership, membership[1])
        previous = index.get(key)
        if previous is None or score_key(candidate) < score_key(previous):
            index[key] = candidate
        q = len(candidate.fit.centers)
        previous = self.best_by_q.get(q)
        if previous is None or score_key(candidate) < score_key(previous):
            self.best_by_q[q] = candidate
        if self.best_overall is None or score_key(candidate) < score_key(self.best_overall):
            self.best_overall = candidate
        if cuts is not None:
            self.tree_by_q.setdefault(q, {})[cuts] = self.by_cuts[cuts]
        return candidate

    def compact_payloads(self):
        """Keep coordinate-reuse payloads only for indexed count winners.

        Preserve every state's exact centers, weights, score and membership for
        independent verification while releasing nonwinner reuse payloads.
        """
        keep = {c.state_index for c in self.best_by_q.values()}
        for candidate in self.candidates:
            if candidate.state_index not in keep and (candidate.fit.coordinate_fits is not None
                                                     or candidate.fit.group_fits is not None):
                candidate.fit = replace(candidate.fit, coordinate_fits=None, group_fits=None)

    def _store_outcome(self, key, outcome, reason, state_index, fitted=None):
        if key in self._outcomes:
            raise ValueError("An exact refit attempt was committed twice")
        self._outcomes[key] = len(self._outcome_records)
        self._outcome_records.append(RefitAttempt(key, outcome, reason, state_index, fitted))

    def consider_independent(self, requests):
        """Optional bounded batching; dependencies must remain outside this call.

        Each request is ``(cuts_or_None, labels, centers, provenance, parent)``.
        Duplicate memberships and existing attempt/structural hits are ordering
        barriers. CPU requests retain ordinary serial consideration.
        """
        def prepared():
            for cuts, labels, centers, provenance, parent in requests:
                labels, centers = _canonical_inputs(labels, centers, self.model.n, self.model.r)
                if cuts is not None:
                    raw = np.asarray(cuts)
                    if raw.ndim != 1 or not np.isfinite(raw).all() or np.any(raw != np.rint(raw)):
                        raise ValueError("Tree cuts must be finite integer edge IDs")
                    cuts = tuple(sorted(map(int, cuts)))
                    if (len(set(cuts)) != len(cuts) or any(not 0 <= edge < len(self.tree.edges) for edge in cuts)
                            or not np.array_equal(labels, partition_from_cuts(self.tree, cuts))):
                        raise ValueError("Candidate labels and frozen-tree cuts disagree")
                yield self.prepare(cuts, labels, parent=parent), centers, provenance
        return self.consider_prepared_independent(prepared())

    def consider_prepared_independent(self, requests):
        """One prepared numerical path, with exact-cache logical order barriers."""

        model = self.model
        enabled = self.weight_batch_size > 1 and model.backend.kind == "cuda"
        if not enabled:
            return [self.consider_prepared(partition, centers, provenance)
                    for partition, centers, provenance in requests]
        results, pending, memberships = [], [], set()
        def flush():
            if not pending:
                return
            started = perf_counter()
            began = set()

            def begin(index):
                if index not in began:
                    self.refit_calls += 1
                    began.add(index)

            def observe(index, event):
                begin(index)
                self.request_progress(self, pending[index][2], event)

            def commit(index, fitted):
                begin(index)
                partition, centers, provenance, output_index = pending[index]
                result = self._commit(partition, centers, provenance, fitted)
                results[output_index] = result
                if self.progress is not None:
                    self.progress(self, provenance)

            work = [(partition, centers, None if self.request_progress is None else
                     lambda event, index=index: observe(index, event))
                    for index, (partition, centers, _, _) in enumerate(pending)]
            try:
                model.refit_prepared_batch(work, max_active=self.weight_batch_size, on_result=commit)
            finally:
                self.refit_seconds += perf_counter()-started
            pending.clear()
            memberships.clear()

        for output_index, item in enumerate(requests):
            results.append(None)
            partition, centers, provenance = item
            if (partition.tree_identity != self._tree_identity
                    or partition.model_identity != model.identity):
                raise ValueError("Prepared partition belongs to another model/tree")
            membership = partition.membership.key
            attempt = (membership, None if centers is None else centers.tobytes(), self.numerical_policy)
            if pending and (membership in memberships or len(pending) >= self.weight_batch_size
                            or membership in self._structural or attempt in self._attempts):
                flush()
            if membership in self._structural or attempt in self._attempts:
                results[output_index] = self.consider_prepared(partition, centers, provenance)
                continue
            pending.append((partition, centers, _plain(provenance), output_index))
            memberships.add(membership)
        flush()
        return results
