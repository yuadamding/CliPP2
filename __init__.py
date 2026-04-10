"""Core package for multi-region clipp.

Layout:
- `core`: direct partition search and fitting
- `io`: TSV loading and simulation-to-TSV conversion
- `metrics`: simulation-time evaluation helpers
- `runners`: end-to-end fitting, benchmarking, and settings selection
- `sim`: synthetic data generation workflows
"""

from ._version import __version__
from .core import FitOptions, FitResult, fit_profiled_partition_search, fit_single_stage_em
from .io import (
    ConversionConfig,
    PatientData,
    TumorData,
    convert_one_tumor,
    convert_simulation_root,
    load_patient_tsv,
    load_tumor_tsv,
)
from .metrics import evaluate_fit_against_simulation
from .runners import (
    MassiveMultiregionBenchmarkConfig,
    ModelSelectionResult,
    PatientRegime,
    TumorRegime,
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
    "PatientRegime",
    "PatientData",
    "TumorData",
    "TumorRegime",
    "RecommendedSettings",
    "SimulationGridConfig",
    "SimulationPackageConfig",
    "__version__",
    "convert_one_tumor",
    "convert_simulation_root",
    "evaluate_fit_against_simulation",
    "fit_profiled_partition_search",
    "fit_single_stage_em",
    "generate_and_convert_simulation",
    "load_patient_tsv",
    "load_tumor_tsv",
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
