"""Core package for multi-region clipp.

Layout:
- `core`: observed-data pairwise fusion and fitting
- `io`: TSV loading and simulation-to-TSV conversion
- `metrics`: simulation-time evaluation helpers
- `runners`: end-to-end fitting, benchmarking, and settings selection
- `sim`: synthetic data generation workflows
"""

from ._version import __version__
from .core import (
    FitOptions,
    FitResult,
    PairwiseFusionGraph,
    TorchRuntime,
    build_complete_adaptive_graph,
    build_complete_uniform_graph,
    compute_exact_observed_data_pilot,
    compute_pooled_observed_data_start,
    compute_scalar_cell_wells,
    compute_scalar_well_start_bank,
    fit_observed_data_pairwise_fusion,
    resolve_pairwise_fusion_graph,
    fit_single_stage_em,
)
from .io import (
    ConversionConfig,
    TumorData,
    PatientData,
    convert_one_tumor,
    convert_simulation_root,
    load_tumor_tsv,
    load_patient_tsv,
)
from .metrics import evaluate_fit_against_simulation
from .runners import (
    MassiveMultiregionBenchmarkConfig,
    ModelSelectionResult,
    TumorRegime,
    PatientRegime,
    RecommendedSettings,
    process_one_file,
    recommend_settings_from_data,
    recommend_settings_from_regime,
    run_cohort_benchmark,
    run_directory,
    run_massive_multiregion_benchmark,
    run_simulation_benchmark,
    run_single_region_cohort_benchmark,
    select_model,
    summarize_patient_regime,
    summarize_tumor_regime,
)
from .sim import SimulationGridConfig, SimulationPackageConfig, generate_and_convert_simulation, run_simulation_grid

__all__ = [
    "ConversionConfig",
    "FitOptions",
    "FitResult",
    "MassiveMultiregionBenchmarkConfig",
    "ModelSelectionResult",
    "PairwiseFusionGraph",
    "TorchRuntime",
    "TumorData",
    "PatientRegime",
    "PatientData",
    "TumorRegime",
    "RecommendedSettings",
    "SimulationGridConfig",
    "SimulationPackageConfig",
    "__version__",
    "build_complete_adaptive_graph",
    "build_complete_uniform_graph",
    "compute_exact_observed_data_pilot",
    "compute_pooled_observed_data_start",
    "compute_scalar_cell_wells",
    "compute_scalar_well_start_bank",
    "convert_one_tumor",
    "convert_simulation_root",
    "evaluate_fit_against_simulation",
    "fit_observed_data_pairwise_fusion",
    "fit_single_stage_em",
    "generate_and_convert_simulation",
    "load_tumor_tsv",
    "load_patient_tsv",
    "resolve_pairwise_fusion_graph",
    "process_one_file",
    "recommend_settings_from_data",
    "recommend_settings_from_regime",
    "run_cohort_benchmark",
    "run_simulation_grid",
    "run_simulation_benchmark",
    "run_massive_multiregion_benchmark",
    "run_directory",
    "run_single_region_cohort_benchmark",
    "select_model",
    "summarize_patient_regime",
    "summarize_tumor_regime",
]
