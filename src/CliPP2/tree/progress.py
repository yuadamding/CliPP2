"""Optional host-side progress snapshots; no files or GPU synchronization."""
from copy import deepcopy
from time import perf_counter

PROGRESS_SCHEMA_VERSION = 4


class ProgressReporter:
    """Observe phase starts/completions and throttled completed operations.

    Snapshots contain no live fitting objects. Callback failures propagate;
    observation is never a reason to alter search or numerical policy.
    """

    def __init__(self, model, callback):
        if not callable(callback):
            raise TypeError("progress_callback must be callable or None")
        self.model, self.callback = model, callback
        self.started = self.last = perf_counter()
        self.phase_clock = self.started
        self.phase = "initialization"
        self.seed_index = self.requested_k = self.occupied_q = None

    def completed_candidate(self, bank, provenance):
        self.emit(bank, provenance=provenance)

    def start_phase(self, phase, bank=None):
        self.phase = phase
        self.phase_clock = perf_counter()
        self.seed_index = self.requested_k = self.occupied_q = None
        self.emit(bank, started=True)

    def completed_operation(self, bank, provenance, operation):
        self.emit(bank, provenance=provenance, operation=operation)

    def emit(self, bank=None, *, provenance=None, operation=None, complete=False, started=False):
        now = perf_counter()
        if not (complete or started) and now-self.last < 5.0:
            return
        self.last = now
        provenance = {} if provenance is None else provenance
        occupied = provenance.get("occupied_q", self.occupied_q)
        requested = provenance.get("requested_k", self.requested_k)
        if provenance.get("route") == "boundary_refinement":
            occupied = len(provenance["parent_cuts"])+1
            requested = None  # The refinement provenance carries no capacity.
        if operation is not None:
            occupied = operation.get("occupied_q", occupied)
            requested = operation.get("requested_k", requested)
        record = {
            "event": "clipp2_progress", "progress_schema_version": PROGRESS_SCHEMA_VERSION,
            "phase": self.phase,
            "phase_started": started, "phase_complete": complete,
            "elapsed_seconds": now-self.started, "operation": operation,
            "phase_elapsed_seconds": now-self.phase_clock,
            "seed_index": provenance.get("seed_index", self.seed_index),
            "requested_k": requested,
            "occupied_q": occupied,
            "route": provenance.get("route"),
            "telemetry": {} if self.model is None else dict(self.model.telemetry),
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
        if self.model is not None:
            # Native diagnostics copy already-maintained host counters; they do
            # not call cudaStreamSynchronize or add work to optimizer iterations.
            record["native"] = self.model.backend.diagnostics()
        self.callback(deepcopy(record))
