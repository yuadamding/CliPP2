"""Discovery and refinement of the same scientific model."""

from .kernel.clonal_search import search


def discover(model, config, *, progress_callback=None):
    return search(model, max_clusters=config.max_clusters, proposal_budget=config.proposal_budget,
                  progress_callback=progress_callback)


def refine(model, incoming, config, *, progress_callback=None):
    """Keep the incoming solution, even when a numerical search proposal fails."""
    accepted = incoming.best
    if accepted.biological_count > config.stage_two_max_clusters:
        raise ValueError("Stage 2 capacity is smaller than the accepted Stage 1 biological count.")
    result = search(model, max_clusters=config.stage_two_max_clusters,
                    proposal_budget=config.stage_two_proposal_budget, incumbent=accepted,
                    covered_counts=tuple(k for k in incoming.diagnostics["covered_counts"]
                                         if k <= config.stage_two_max_clusters),
                    progress_callback=progress_callback)
    if result.best.bic > accepted.bic:
        raise ValueError("Stage 2 did not retain the accepted Stage 1 criterion.")
    return result
