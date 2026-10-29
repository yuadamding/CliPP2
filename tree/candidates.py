"""Shared registry of compact memberships, refit attempts and immutable states."""

from dataclasses import dataclass, field
from time import perf_counter
from types import MappingProxyType

import numpy as np

from ..kernel.model import FitResult, RegionalModel
from ..kernel.topology import FrozenTree, Membership, PreparedPartition, canonical_partition, partition_from_cuts


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
    # An unfamiliar rejection cannot certify a required numerical evaluation.
    return "unresolved_numerical"


def score_key(candidate):
    tree = candidate.membership_kind == "tree"
    return (candidate.score, len(candidate.fit.centers), 0 if tree else 1,
            candidate.cuts if tree else tuple(candidate.fit.labels))


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
    normalize_provenance = staticmethod(_plain)

    def __init__(self, model, tree, *, progress=None, request_progress=None,
                 weight_batch_size=1, device_proposals=False, retain_evidence=True):
        if type(weight_batch_size) is not int or weight_batch_size < 1:
            raise ValueError("weight_batch_size must be a positive integer")
        if type(device_proposals) is not bool:
            raise TypeError("device_proposals must be a boolean")
        if type(retain_evidence) is not bool:
            raise TypeError("retain_evidence must be a boolean")
        self.retain_evidence = retain_evidence
        self.model, self.tree = model, tree
        self.progress = progress
        self.request_progress = request_progress
        self.candidates, self.general_rejections, self.attempt_records = [], [], []
        self.by_cuts, self.by_membership, self.ineligible = {}, {}, {}
        self.membership_attempts = {}
        self._attempts, self._states, self._structural = {}, {}, {}
        self._outcomes, self._outcome_records = {}, []
        self.duplicate_routes = self.refit_calls = 0
        self.attempt_cache_hits = self.structural_cache_hits = 0
        self.refit_seconds = 0.0
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
        groups, coordinate_fits = (), ()
        if hasattr(self.model, "prepare_group"):
            counts = np.bincount(labels)
            reusable = {}
            if parent is not None:
                for index, group in enumerate(parent.groups):
                    target = int(labels[group.members[0]])
                    if counts[target] == len(group.members) and np.all(labels[group.members] == target):
                        reusable[target] = (group, parent.coordinate_fits[index]
                                            if parent.coordinate_fits else None)
            prepared, reused = [], []
            for cluster in range(int(np.max(labels)) + 1):
                group, previous = reusable.get(cluster, (None, None))
                prepared.append(self.model.prepare_group(np.flatnonzero(labels == cluster))
                                if group is None else group)
                reused.append(previous)
            groups = tuple(prepared)
            coordinate_fits = tuple(reused)
        if fitted is not None and getattr(fitted, "coordinate_fits", None) is not None:
            coordinate_fits = tuple(tuple(tuple(value) for value in group)
                                    for group in fitted.coordinate_fits)
        return PreparedPartition(getattr(self.model, "identity", ""), self._tree_identity,
                                 Membership(self.tree, cuts) if cuts is not None else Membership(explicit_labels=labels),
                                 labels, groups, coordinate_fits,
                                 columns if parent is None else parent.columns)

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
                or partition.model_identity != getattr(self.model, "identity", "")):
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
        self.membership_attempts[(labels.tobytes(), None if centers is None else centers.tobytes())] = candidate
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
        labels = partition.labels
        provenance = _plain(provenance)
        started = perf_counter()
        self.refit_calls += 1
        try:
            if self.request_progress is not None and type(self.model) is RegionalModel:
                fitted = self.model.refit_prepared(
                    partition, initial_centers=centers,
                    progress_callback=lambda operation: self.request_progress(self, provenance, operation))
            else:
                fitted = (self.model.refit_prepared(partition, initial_centers=centers)
                          if hasattr(self.model, "refit_prepared")
                          else self.model.refit(labels, initial_centers=centers))
        finally:
            self.refit_seconds += perf_counter()-started
        return self._commit(partition, centers, provenance, fitted)

    def _commit(self, partition, centers, provenance, fitted):
        """Publish exactly one completed fit, always in original request order."""
        membership, cuts, labels = partition.membership.key, partition.cuts, partition.labels
        provenance = _plain(provenance)
        attempt = (membership, None if centers is None else centers.tobytes(), self.numerical_policy)
        outcome = _outcome_kind(fitted)
        record = None
        if self.retain_evidence:
            record = {"membership_kind": membership[0],
                      "cuts" if cuts is not None else "labels": list(cuts) if cuts is not None else labels.tolist(),
                      "initial_centers_ccf": None if centers is None else centers.tolist(),
                      "numerical_policy": self.numerical_policy, "provenance": provenance,
                      "eligible": bool(fitted.eligible), "reason": fitted.reason,
                      "outcome": outcome}
            if outcome == "completed_ineligible":
                record["completed_fit"] = candidate_record(Candidate(fitted))
            self.attempt_records.append(record)
        self._attempts[attempt] = None
        if not fitted.eligible:
            rejection = {"cuts" if cuts is not None else "labels": list(cuts) if cuts is not None else labels.tolist(),
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
        # Prepared results share this exact membership; reuse its active labels
        # instead of expanding the same retained tree cuts again for validation.
        fitted_labels = (labels if isinstance(fitted, FitResult)
                         and fitted.membership is partition.membership else fitted.labels)
        canonical, aligned = _canonical_inputs(fitted_labels, fitted.centers, self.model.n, self.model.r)
        if not np.array_equal(canonical, labels):
            raise ValueError("Conditional refit changed the proposed membership")
        order = list(dict.fromkeys(np.asarray(fitted_labels, dtype=int).tolist()))
        weights = np.asarray(fitted.weights, dtype=float)
        if (weights.shape != (len(order),) or not np.isfinite(weights).all()
                or np.any(weights <= 0) or not np.isclose(weights.sum(), 1., atol=1e-12, rtol=0)):
            raise ValueError("Eligible fitted weights must be positive normalized occupied rows")
        if not np.isfinite([fitted.conditional_log_likelihood, fitted.mixture_log_likelihood,
                            fitted.complexity_penalty, fitted.score]).all():
            raise ValueError("An eligible candidate has nonfinite likelihood, penalty or score")
        weights = weights[order]
        state_key = membership, aligned.tobytes(), weights.tobytes()
        candidate = self._states.get(state_key)
        if candidate is None:
            coordinate_fits = getattr(fitted, "coordinate_fits", None)
            if coordinate_fits is not None:
                coordinate_fits = np.asarray(coordinate_fits)[order]
            # Production prepared refits already own the exact compact membership.
            # Keep validating generic/public refits before aligning their row order.
            state = fitted
            if (not isinstance(fitted, FitResult) or fitted.membership is not partition.membership
                    or order != list(range(len(order)))):
                state = FitResult(partition.membership, aligned, weights,
                                  float(fitted.conditional_log_likelihood), float(fitted.mixture_log_likelihood),
                                  float(fitted.complexity_penalty), float(fitted.score), True,
                                  fitted.reason, fitted.diagnostics, coordinate_fits)
            candidate = Candidate(state, [provenance], len(self.candidates))
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
        if record is not None:
            record["state_index"] = candidate.state_index
        index, key = (self.by_cuts, cuts) if cuts is not None else (self.by_membership, labels.tobytes())
        previous = index.get(key)
        if previous is None or score_key(candidate) < score_key(previous):
            index[key] = candidate
        return candidate

    def _store_outcome(self, key, outcome, reason, state_index, fitted=None):
        if key in self._outcomes:
            raise ValueError("An exact refit attempt was committed twice")
        self._outcomes[key] = len(self._outcome_records)
        self._outcome_records.append(RefitAttempt(key, outcome, reason, state_index, fitted))

    def refit_attempt_index(self, labels, initial_centers):
        labels, centers = _canonical_inputs(labels, initial_centers, self.model.n, self.model.r)
        key = (("general", labels.tobytes()),
               None if centers is None else centers.tobytes(), self.numerical_policy)
        if key not in self._outcomes:
            raise ValueError("Required membership refit has no exact outcome record")
        return self._outcomes[key]

    def consider_independent(self, requests):
        """Optional bounded batching; dependencies must remain outside this call.

        Each request is ``(cuts_or_None, labels, centers, provenance, parent)``.
        Duplicate memberships and existing attempt/structural hits are ordering
        barriers. Custom models/overrides retain ordinary serial consideration.
        """
        def serial(item):
            cuts, labels, centers, provenance, parent = item
            return (self.consider_membership(labels, centers, provenance, parent=parent) if cuts is None
                    else self.consider(cuts, labels, centers, provenance, parent=parent))

        model = self.model
        enabled = (self.weight_batch_size > 1 and type(model) is RegionalModel
                   and model.backend.kind == "cuda" and not any(name in vars(model) for name in
                       ("refit", "refit_prepared", "_refit_groups", "_refit_steps", "_weight_result")))
        if not enabled:
            return [serial(item) for item in requests]
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
                if partition.cuts is None:
                    self.membership_attempts[(partition.labels.tobytes(),
                        None if centers is None else centers.tobytes())] = result
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
            cuts, labels, centers, provenance, parent = item
            labels, centers = _canonical_inputs(labels, centers, model.n, model.r)
            if cuts is not None:
                raw = np.asarray(cuts)
                if raw.ndim != 1 or not np.isfinite(raw).all() or np.any(raw != np.rint(raw)):
                    raise ValueError("Tree cuts must be finite integer edge IDs")
                cuts = tuple(sorted(map(int, cuts)))
                if (len(set(cuts)) != len(cuts) or any(not 0 <= edge < len(self.tree.edges) for edge in cuts)
                        or not np.array_equal(labels, partition_from_cuts(self.tree, cuts))):
                    raise ValueError("Candidate labels and frozen-tree cuts disagree")
            membership = ("tree", cuts) if cuts is not None else ("general", labels.tobytes())
            attempt = (membership, None if centers is None else centers.tobytes(), self.numerical_policy)
            if pending and (membership in memberships or len(pending) >= self.weight_batch_size
                            or membership in self._structural or attempt in self._attempts):
                flush()
            if membership in self._structural or attempt in self._attempts:
                results[output_index] = serial(item)
                continue
            pending.append((self.prepare(cuts, labels, parent=parent), centers, _plain(provenance), output_index))
            memberships.add(membership)
        flush()
        return results

    def evidence(self, selected):
        if not self.retain_evidence:
            raise ValueError("Research evidence was not retained for this candidate bank")
        return candidate_evidence(self.tree, self.candidates, selected, self.attempt_records)

    def compact_attempt_links(self):
        """Share immutable membership keys, without retaining research records."""
        return tuple((key[0], candidate is not None,
                      None if candidate is None else candidate.state_index)
                     for key, candidate in self._attempts.items())


def candidate_record(candidate):
    """One explicit research-export record, never a numerical execution step."""
    fit = candidate.fit
    return {
        "membership_kind": candidate.membership_kind,
        "cuts" if candidate.cuts is not None else "labels":
            list(candidate.cuts) if candidate.cuts is not None else fit.labels.tolist(),
        "centers_ccf": fit.centers.tolist(), "weights": fit.weights.tolist(),
        "conditional_log_likelihood": float(fit.conditional_log_likelihood),
        "mixture_log_likelihood": float(fit.mixture_log_likelihood),
        "complexity_penalty": float(fit.complexity_penalty), "score": float(fit.score),
        "eligible": bool(fit.eligible), "reason": fit.reason,
        "diagnostics": _plain(fit.diagnostics), "provenance": _plain(candidate.provenance),
    }


def candidate_evidence(tree, candidates, selected, attempts=()):
    """JSON-ready replay payload; connected memberships never store N labels."""
    states = [candidate_record(candidate) for candidate in candidates]
    selected_index = next((i for i, candidate in enumerate(candidates) if candidate is selected), None)
    if selected_index is None:
        raise ValueError("Selected fitted state is absent from the candidate bank")
    return {"version": 1, "tree": {"identity": tree.identity, "n": tree.n,
            "edges": tree.edges.tolist(), "kind": tree.kind,
            "mutation_ids": list(tree.mutation_ids)},
            "selected_state_index": selected_index, "states": states,
            "refit_attempts": _plain(attempts)}


def _membership_size(tree, key):
    kind, value = key
    if kind == "tree":
        if (not isinstance(value, tuple) or any(type(edge) is not int or
                not 0 <= edge < len(tree.edges) for edge in value)
                or tuple(sorted(set(value))) != value):
            raise ValueError("Candidate bank has invalid tree cuts")
        return len(value)+1
    if kind != "general" or not isinstance(value, bytes) or len(value) != 8*tree.n:
        raise ValueError("Candidate bank has invalid general membership")
    labels = np.frombuffer(value, dtype=np.int64)
    q = len(np.unique(labels))
    if list(dict.fromkeys(labels.tolist())) != list(range(q)):
        raise ValueError("Candidate bank must retain canonical occupied memberships")
    return q


def validate_candidate_bank(tree, candidates, selected, attempt_links=None):
    """Audit retained objects directly; no export, decoding or refit execution."""
    if not isinstance(tree, FrozenTree):
        raise ValueError("Candidate bank requires a validated immutable FrozenTree")
    if not candidates or not any(candidate is selected for candidate in candidates):
        raise ValueError("Selected fitted state is absent from the complete candidate bank")
    memberships, region_count = [], None
    for index, candidate in enumerate(candidates):
        fit, membership = candidate.fit, candidate.fit.membership
        if (not isinstance(membership, Membership) or
                (membership.tree is not None and membership.tree is not tree
                 and membership.tree.identity != tree.identity)):
            raise ValueError("Candidate belongs to a different frozen tree")
        key = membership.key
        q = _membership_size(tree, key)
        centers, weights = np.asarray(fit.centers), np.asarray(fit.weights)
        if (centers.ndim != 2 or centers.shape[0] != q or not centers.shape[1]
                or not np.isfinite(centers).all() or np.any(centers < 0) or np.any(centers > 1)):
            raise ValueError("Candidate bank needs finite q-by-R CCF centers in [0,1]")
        if region_count is None:
            region_count = centers.shape[1]
        elif centers.shape[1] != region_count:
            raise ValueError("Candidate bank mixes different region counts")
        if (weights.shape != (q,) or not np.isfinite(weights).all() or np.any(weights <= 0)
                or not np.isclose(weights.sum(), 1., rtol=0., atol=1e-12)):
            raise ValueError("Candidate bank has invalid occupied weights")
        penalty = float((q*region_count+q-1)*np.log(tree.n))
        score = -2*float(fit.mixture_log_likelihood)+penalty
        if (fit.eligible is not True or not np.isfinite(score)
                or not np.isfinite(fit.conditional_log_likelihood)
                or not np.isclose(penalty, fit.complexity_penalty, atol=1e-7, rtol=1e-10)
                or not np.isclose(score, fit.score, atol=1e-7, rtol=1e-10)):
            raise ValueError("Candidate bank score arithmetic is inconsistent")
        if candidate.state_index is not None and (type(candidate.state_index) is not int
                                                  or candidate.state_index != index):
            raise ValueError("Retained fitted-state index disagrees with candidate order")
        memberships.append(key)
    if attempt_links is not None:
        retained = set()
        for key, eligible, index in attempt_links:
            _membership_size(tree, key)
            if type(eligible) is not bool:
                raise ValueError("Refit attempt eligibility must be explicit")
            if not eligible:
                if index is not None:
                    raise ValueError("A rejected refit attempt cannot reference an eligible state")
                continue
            if type(index) is not int or not 0 <= index < len(candidates):
                raise ValueError("An eligible refit attempt is missing its retained fitted state")
            if key != memberships[index]:
                raise ValueError("Refit attempt and retained fitted-state memberships disagree")
            retained.add(index)
        if retained != set(range(len(candidates))):
            raise ValueError("Candidate bank contains fitted states without an eligible refit attempt")
    if selected is not min(candidates, key=score_key):
        raise ValueError("Selected fit is not the best complete-bank score and tie-break")
    return selected


def evidence_attempt_links(tree, attempts, region_count):
    """Decode optional research records into the same compact audit links."""
    if not isinstance(attempts, (list, tuple)):
        raise ValueError("Candidate evidence refit attempts must be a sequence")
    links = []
    for attempt in attempts:
        key = _evidence_membership(tree, attempt).key
        q = _membership_size(tree, key)
        initial = attempt.get("initial_centers_ccf")
        if initial is not None:
            initial = np.asarray(initial, dtype=float)
            if initial.shape != (q, region_count) or not np.isfinite(initial).all():
                raise ValueError("Refit attempt has invalid initial centers")
        if not attempt.get("eligible", False) and "state_index" in attempt:
            raise ValueError("A rejected refit attempt cannot reference an eligible state")
        links.append((key, attempt.get("eligible"), attempt.get("state_index")))
    return tuple(links)


def _evidence_membership(tree, item):
    if item["membership_kind"] == "tree":
        cuts = tuple(item["cuts"])
        _membership_size(tree, ("tree", cuts))
        return Membership(tree, cuts)
    if item["membership_kind"] != "general":
        raise ValueError("Unknown candidate membership kind")
    return Membership(explicit_labels=np.asarray(item["labels"]))


def replay_candidate_evidence(evidence):
    """Explicit research decode followed by the ordinary retained-object audit."""
    if evidence.get("version") != 1:
        raise ValueError("Unknown candidate-evidence version")
    record = evidence["tree"]
    edges = (np.empty((0, 2), dtype=np.int64) if record["n"] == 1 and record["edges"] == []
             else np.asarray(record["edges"]))
    tree = FrozenTree(edges, record["n"], tuple(record["mutation_ids"]), record["kind"])
    if tree.identity != record["identity"]:
        raise ValueError("Candidate evidence tree identity mismatch")
    candidates = []
    # Only explicit decoding expands tree memberships; reuse them within replay.
    partitions = {}
    for index, item in enumerate(evidence["states"]):
        membership = _evidence_membership(tree, item)
        if membership.cuts is None:
            labels = membership.labels
        elif membership.cuts in partitions:
            labels = partitions[membership.cuts]
        else:
            labels = partition_from_cuts(tree, membership.cuts)
            partitions[membership.cuts] = labels
        centers = np.asarray(item["centers_ccf"], dtype=float)
        if centers.ndim != 2 or not all(centers.shape):
            raise ValueError("Candidate evidence needs nonempty q-by-R centers")
        canonical, aligned = _canonical_inputs(labels, centers, tree.n, centers.shape[1])
        if not np.array_equal(canonical, labels) or not np.array_equal(aligned, centers):
            raise ValueError("Candidate evidence must retain canonical aligned memberships")
        fit = FitResult(membership, centers, np.asarray(item["weights"], dtype=float),
                        item["conditional_log_likelihood"], item["mixture_log_likelihood"],
                        item["complexity_penalty"], item["score"], item["eligible"],
                        item["reason"], item["diagnostics"])
        candidates.append(Candidate(fit, item["provenance"], index))
    selected_index = evidence["selected_state_index"]
    if type(selected_index) is not int or not 0 <= selected_index < len(candidates):
        raise ValueError("Candidate evidence has no selected state")
    attempts = evidence.get("refit_attempts", ())
    links = evidence_attempt_links(tree, attempts, candidates[0].fit.centers.shape[1])
    selected = validate_candidate_bank(tree, candidates, candidates[selected_index], links if attempts else None)
    return selected, tuple(candidates)
