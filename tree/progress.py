"""Optional host-side progress snapshots; no files or GPU synchronization."""
from copy import deepcopy
from time import perf_counter


class ProgressReporter:
    """Observe completed operations, at most once per five seconds or phase.

    Snapshots contain no live fitting objects. Callback failures propagate;
    observation is never a reason to alter search or numerical policy.
    """

    def __init__(self, model, callback):
        if not callable(callback):
            raise TypeError("progress_callback must be callable or None")
        self.model, self.callback = model, callback
        self.started = self.last = perf_counter()
        self.phase = "initialization"
        self.subset_visits = 0
        self.seed_index = self.requested_k = self.occupied_q = None

    def completed_candidate(self, bank, provenance):
        self.emit(bank, provenance=provenance)

    def emit(self, bank=None, *, provenance=None, complete=False):
        now = perf_counter()
        if not complete and now-self.last < 5.0:
            return
        self.last = now
        provenance = {} if provenance is None else provenance
        occupied = provenance.get("occupied_q", self.occupied_q)
        requested = provenance.get("requested_k", self.requested_k)
        if provenance.get("route") == "boundary_refinement":
            occupied = len(provenance["parent_cuts"])+1
            requested = None  # The refinement provenance carries no capacity.
        record = {
            "event": "clipp2_progress", "phase": self.phase,
            "phase_complete": complete, "elapsed_seconds": now-self.started,
            "seed_index": provenance.get("seed_index", self.seed_index),
            "requested_k": requested,
            "occupied_q": occupied,
            "route": provenance.get("route"), "visited_cut_subsets": self.subset_visits,
            "telemetry": dict(getattr(self.model, "telemetry", {})),
        }
        if bank is not None:
            record["candidates"] = {
                "refit_calls": bank.refit_calls, "refit_and_score_seconds": bank.refit_seconds,
                "attempt_cache_hits": bank.attempt_cache_hits,
                "structural_cache_hits": bank.structural_cache_hits,
                "eligible_states": len(bank.candidates),
                "ineligible_tree_partitions": len(bank.ineligible),
                "general_rejections": len(bank.general_rejections),
            }
        backend = getattr(self.model, "backend", None)
        if backend is not None:
            # Native diagnostics copy already-maintained host counters; they do
            # not call cudaStreamSynchronize or add work to optimizer iterations.
            record["native"] = backend.diagnostics()
        self.callback(deepcopy(record))
