"""Shared registry of compact memberships, refit attempts and immutable states."""

from dataclasses import dataclass, field
from collections import OrderedDict
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
                 weight_batch_size=1, weight_batch_state_bytes=8*1024**2,
                 device_proposals=False):
        if type(weight_batch_size) is not int or weight_batch_size < 1:
            raise ValueError("weight_batch_size must be a positive integer")
        if type(weight_batch_state_bytes) is not int or weight_batch_state_bytes < 0:
            raise ValueError("weight_batch_state_bytes must be a nonnegative integer")
        if type(device_proposals) is not bool:
            raise TypeError("device_proposals must be a boolean")
        self.model, self.tree = model, tree
        self.progress = progress
        self.request_progress = request_progress
        self.candidates, self.general_rejections, self.attempt_records = [], [], []
        self.by_cuts, self.by_membership, self.ineligible = {}, {}, {}
        self.membership_attempts, self.structural_rejections = {}, {}
        self._attempts, self._states, self._structural = {}, {}, {}
        self.duplicate_routes = self.refit_calls = 0
        self.attempt_cache_hits = self.structural_cache_hits = 0
        self.refit_seconds = 0.0
        self._tree_identity = tree.identity
        self.weight_batch_size = weight_batch_size
        self.weight_batch_state_bytes = weight_batch_state_bytes
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
        record = {"membership_kind": membership[0],
                  "cuts" if cuts is not None else "labels": list(cuts) if cuts is not None else labels.tolist(),
                  "initial_centers_ccf": None if centers is None else centers.tolist(),
                  "numerical_policy": self.numerical_policy, "provenance": provenance,
                  "eligible": bool(fitted.eligible), "reason": fitted.reason}
        self.attempt_records.append(record)
        self._attempts[attempt] = None
        if not fitted.eligible:
            rejection = {"cuts" if cuts is not None else "labels": record.get("cuts", record.get("labels")),
                         "reason": fitted.reason, "provenance": [provenance]}
            if cuts is None:
                self.general_rejections.append(rejection)
            else:
                self.ineligible[cuts] = rejection
            if str(fitted.reason).startswith(("unsupported_cluster_region", "empty_cluster_region_box")):
                self._structural[membership] = rejection
                if cuts is None:
                    self.structural_rejections[labels.tobytes()] = rejection
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
        record["state_index"] = candidate.state_index
        index, key = (self.by_cuts, cuts) if cuts is not None else (self.by_membership, labels.tobytes())
        previous = index.get(key)
        if previous is None or score_key(candidate) < score_key(previous):
            index[key] = candidate
        return candidate

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
        reserved = 0

        def flush():
            nonlocal reserved
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
                model.refit_prepared_batch(work, max_active=self.weight_batch_size,
                    state_budget_bytes=self.weight_batch_state_bytes, on_result=commit)
            finally:
                self.refit_seconds += perf_counter()-started
            pending.clear()
            memberships.clear()
            reserved = 0

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
            charge = model.refit_batch_reservation(int(labels.max())+1, progress=self.request_progress is not None)
            if pending and (membership in memberships or len(pending) >= self.weight_batch_size
                            or reserved+charge > self.weight_batch_state_bytes
                            or membership in self._structural or attempt in self._attempts):
                flush()
            if membership in self._structural or attempt in self._attempts:
                results[output_index] = serial(item)
                continue
            if charge > self.weight_batch_state_bytes:
                raise MemoryError("One prepared candidate exceeds the live-state budget")
            pending.append((self.prepare(cuts, labels, parent=parent), centers, _plain(provenance), output_index))
            memberships.add(membership)
            reserved += charge
        flush()
        return results

    def evidence(self, selected):
        return candidate_evidence(self.tree, self.candidates, selected, self.attempt_records)


def candidate_evidence(tree, candidates, selected, attempts=()):
    """JSON-ready replay payload; connected memberships never store N labels."""
    states = []
    for candidate in candidates:
        fit = candidate.fit
        states.append({
            "membership_kind": candidate.membership_kind,
            "cuts" if candidate.cuts is not None else "labels":
                list(candidate.cuts) if candidate.cuts is not None else fit.labels.tolist(),
            "centers_ccf": fit.centers.tolist(), "weights": fit.weights.tolist(),
            "conditional_log_likelihood": float(fit.conditional_log_likelihood),
            "mixture_log_likelihood": float(fit.mixture_log_likelihood),
            "complexity_penalty": float(fit.complexity_penalty), "score": float(fit.score),
            "eligible": bool(fit.eligible), "reason": fit.reason,
            "diagnostics": _plain(fit.diagnostics), "provenance": _plain(candidate.provenance),
        })
    selected_index = next((i for i, candidate in enumerate(candidates) if candidate is selected), None)
    if selected_index is None:
        raise ValueError("Selected fitted state is absent from the candidate bank")
    return {"version": 1, "tree": {"identity": tree.identity, "n": tree.n,
            "edges": tree.edges.tolist(), "kind": tree.kind,
            "mutation_ids": list(tree.mutation_ids)},
            "selected_state_index": selected_index, "states": states,
            "refit_attempts": _plain(attempts)}


def replay_candidate_evidence(evidence):
    """Reconstruct memberships and replay score selection, not refit numerics.

    The persisted centers and weights permit a separate likelihood replay with
    original inputs. This check alone does not verify those inputs/emissions.
    """
    if evidence.get("version") != 1:
        raise ValueError("Unknown candidate-evidence version")
    record = evidence["tree"]
    edges = (np.empty((0, 2), dtype=np.int64) if record["n"] == 1 and record["edges"] == []
             else np.asarray(record["edges"]))
    tree = FrozenTree(edges, record["n"], tuple(record["mutation_ids"]), record["kind"])
    if tree.identity != record["identity"]:
        raise ValueError("Candidate evidence tree identity mismatch")
    candidates, region_count = [], None
    # Replay-local and independent of fitted caches. Many state/attempt records
    # share a compact cut set; reconstruct each membership at most once.
    partitions, partition_bytes = OrderedDict(), 0

    def replay_partition(cuts):
        nonlocal partition_bytes
        if cuts in partitions:
            partitions.move_to_end(cuts)
            return partitions[cuts]
        labels = partition_from_cuts(tree, cuts)
        # Evidence can describe an enormous bank. Replay caching is bounded
        # execution work, not permanent expansion of every compact cut state.
        if labels.nbytes <= 8*1024**2:
            while partition_bytes + labels.nbytes > 8*1024**2:
                _, removed = partitions.popitem(last=False)
                partition_bytes -= removed.nbytes
            partitions[cuts] = labels
            partition_bytes += labels.nbytes
        return labels
    for item in evidence["states"]:
        kind = item["membership_kind"]
        cuts = tuple(item["cuts"]) if kind == "tree" else None
        if kind not in ("tree", "general"):
            raise ValueError("Unknown candidate membership kind")
        if cuts is not None and (any(type(edge) is not int or not 0 <= edge < len(tree.edges)
                                    for edge in cuts) or tuple(sorted(set(cuts))) != cuts):
            raise ValueError("Candidate evidence has invalid tree cuts")
        labels = replay_partition(cuts) if cuts is not None else np.asarray(item["labels"])
        centers = np.asarray(item["centers_ccf"], dtype=float)
        if (centers.ndim != 2 or not all(centers.shape)
                or not np.isfinite(centers).all()
                or np.any(centers < 0) or np.any(centers > 1)):
            raise ValueError("Candidate evidence needs finite q-by-R CCF centers in [0,1]")
        if region_count is None:
            region_count = centers.shape[1]
        elif centers.shape[1] != region_count:
            raise ValueError("Candidate evidence mixes different region counts")
        canonical, aligned = _canonical_inputs(labels, centers, tree.n, centers.shape[1])
        if not np.array_equal(canonical, labels) or not np.array_equal(aligned, centers):
            raise ValueError("Candidate evidence must retain canonical aligned memberships")
        weights = np.asarray(item["weights"], dtype=float)
        if (weights.shape != (len(centers),) or not np.isfinite(weights).all()
                or np.any(weights <= 0) or not np.isclose(weights.sum(), 1., rtol=0., atol=1e-12)):
            raise ValueError("Candidate evidence has invalid occupied weights")
        q, r = centers.shape
        penalty = float((q*r+q-1)*np.log(tree.n))
        score = -2*float(item["mixture_log_likelihood"])+penalty
        if (item["eligible"] is not True or not np.isfinite(score)
                or not np.isfinite(item["conditional_log_likelihood"])
                or not np.isclose(penalty, item["complexity_penalty"], atol=1e-7, rtol=1e-10)
                or not np.isclose(score, item["score"], atol=1e-7, rtol=1e-10)):
            raise ValueError("Candidate evidence score arithmetic is inconsistent")
        membership = (Membership(tree, cuts) if cuts is not None else Membership(explicit_labels=labels))
        fit = FitResult(membership, centers, weights,
                          item["conditional_log_likelihood"], item["mixture_log_likelihood"],
                          item["complexity_penalty"], item["score"], True,
                          item["reason"], item["diagnostics"])
        candidates.append(Candidate(fit, item["provenance"]))
    # Arithmetic-only callers may omit attempts. A saved search ledger that
    # includes them must account for every retained eligible fitted state.
    attempts = evidence.get("refit_attempts", ())
    if not isinstance(attempts, (list, tuple)):
        raise ValueError("Candidate evidence refit attempts must be a sequence")
    retained = set()
    for attempt in attempts:
        if type(attempt.get("eligible")) is not bool:
            raise ValueError("Refit attempt eligibility must be explicit")
        kind = attempt.get("membership_kind")
        if kind == "tree":
            cuts = tuple(attempt["cuts"])
            if (any(type(edge) is not int or not 0 <= edge < len(tree.edges) for edge in cuts)
                    or tuple(sorted(set(cuts))) != cuts):
                raise ValueError("Refit attempt has invalid tree cuts")
            labels = replay_partition(cuts)
        elif kind == "general":
            cuts, labels = None, np.asarray(attempt["labels"])
            canonical, _ = _canonical_inputs(labels, None, tree.n, region_count)
            if not np.array_equal(canonical, labels):
                raise ValueError("Refit attempt must retain canonical memberships")
        else:
            raise ValueError("Unknown refit-attempt membership kind")
        initial = attempt.get("initial_centers_ccf")
        if initial is not None:
            _canonical_inputs(labels, initial, tree.n, region_count)
        if not attempt["eligible"]:
            if "state_index" in attempt:
                raise ValueError("A rejected refit attempt cannot reference an eligible state")
            continue
        index = attempt.get("state_index")
        if type(index) is not int or not 0 <= index < len(candidates):
            raise ValueError("An eligible refit attempt is missing its retained fitted state")
        candidate = candidates[index]
        if (kind != candidate.membership_kind or cuts != candidate.cuts
                or (cuts is None and not np.array_equal(labels, candidate.fit.labels))):
            raise ValueError("Refit attempt and retained fitted-state memberships disagree")
        retained.add(index)
    if attempts and retained != set(range(len(candidates))):
        raise ValueError("Candidate evidence contains fitted states without an eligible refit attempt")
    selected_index = evidence["selected_state_index"]
    if type(selected_index) is not int or not 0 <= selected_index < len(candidates):
        raise ValueError("Candidate evidence has no selected state")
    selected = candidates[selected_index]
    if selected is not min(candidates, key=score_key):
        raise ValueError("Candidate evidence selection is not the complete-bank minimum")
    return selected, tuple(candidates)
