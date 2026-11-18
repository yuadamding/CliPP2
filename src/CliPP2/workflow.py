"""Freeze preprocessing multiplicities, fit CCFs, and publish assignments."""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from threadpoolctl import threadpool_limits

from .config import FitConfig
from .io.clonal_reporting import BiologicalAssignment, assign_biological, write_results
from .io.tumor_txt import load_tumor_txt
from .kernel.clonal_mixture import ClonalFit, ClonalModel
from .kernel.clonal_search import search


@dataclass(frozen=True, slots=True)
class FitResult:
    """Selected fit and publication details, kept in memory."""

    tumor_id: str
    config: FitConfig
    input_mutations: int
    read_counts: Mapping
    fit: ClonalFit
    assignment: BiologicalAssignment
    search_diagnostics: dict
    output_paths: dict[str, Path]


def run_workflow(input_path, outdir, config=FitConfig(), *, progress_callback=None):
    """Publish the three result tables and return the selected mixture."""
    if not isinstance(config, FitConfig):
        raise TypeError("config must be a FitConfig.")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None.")
    output = Path(outdir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")

    data = load_tumor_txt(input_path, missing_counts=config.missing_counts)
    limits = nullcontext() if config.cpu_threads is None else threadpool_limits(config.cpu_threads, user_api="blas")
    with limits:
        model = ClonalModel(data)
        selected = search(model, max_clusters=config.max_clusters, proposal_budget=config.proposal_budget,
                          progress_callback=progress_callback)
        assignment = assign_biological(model, selected.best)
        paths = write_results(data.tumor_id, assignment, output)
    return FitResult(data.tumor_id, config, len(data.mutation_ids), data.read_count_report,
                     selected.best, assignment, selected.diagnostics, paths)
