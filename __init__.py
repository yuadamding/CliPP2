"""Core package for multi-region clipp.

Layout:
- `core`: graph construction and EM/ADMM fitting
- `io`: TSV loading and simulation-to-TSV conversion
- `metrics`: simulation-time evaluation helpers
- `runners`: end-to-end fitting, benchmarking, and settings selection
- `sim`: synthetic data generation workflows
"""

from ._version import __version__
from .core import GraphData, FitOptions, FitResult, build_knn_graph, fit_single_stage_em
from .io import ConversionConfig, PatientData, convert_simulation_root, load_patient_tsv
from .metrics import evaluate_fit_against_simulation
from .runners import (
    MassiveMultiregionBenchmarkConfig,
    ModelSelectionResult,
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
)
from .sim import SimulationGridConfig, SimulationPackageConfig, generate_and_convert_simulation, run_simulation_grid

__all__ = [
    "ConversionConfig",
    "FitOptions",
    "FitResult",
    "GraphData",
    "MassiveMultiregionBenchmarkConfig",
    "ModelSelectionResult",
    "PatientRegime",
    "PatientData",
    "RecommendedSettings",
    "SimulationGridConfig",
    "SimulationPackageConfig",
    "__version__",
    "build_knn_graph",
    "convert_simulation_root",
    "evaluate_fit_against_simulation",
    "fit_single_stage_em",
    "generate_and_convert_simulation",
    "load_patient_tsv",
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
]
