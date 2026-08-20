"""Compatibility facade for the package-owned model-selection search.

The implementation lives in :mod:`CliPP2.model_selection.search`.  Keep the
historical import surface here while callers migrate; no search policy belongs
in the runner layer.
"""

from ..model_selection.search import (
    BICSelectionResult,
    NoCertifiedRawReferenceError,
    NoEligibleModelSelectionCandidatesError,
    _RawStartAttempt,
    _adaptive_stop_certifies_global_optimum,
    _assemble_selection_result,
    _bootstrap_independent_start_specs,
    _candidate_record_representatives,
    _rescore_partition_candidates,
    _select_raw_reference,
    _select_raw_start_attempt,
    _solver_recovery_fit_options,
    select_model,
)

__all__ = [
    "BICSelectionResult",
    "NoCertifiedRawReferenceError",
    "NoEligibleModelSelectionCandidatesError",
    "_RawStartAttempt",
    "_adaptive_stop_certifies_global_optimum",
    "_assemble_selection_result",
    "_bootstrap_independent_start_specs",
    "_candidate_record_representatives",
    "_rescore_partition_candidates",
    "_select_raw_reference",
    "_select_raw_start_attempt",
    "_solver_recovery_fit_options",
    "select_model",
]
