from .model_selection import ModelSelectionResult, select_model
from .benchmark import (
    MassiveMultiregionBenchmarkConfig,
    run_cohort_benchmark,
    run_massive_multiregion_benchmark,
    run_simulation_benchmark,
    run_single_region_cohort_benchmark,
)
from .outputs import (
    cell_output_table,
    cluster_output_table,
    evaluation_to_frame,
    mutation_output_table,
    write_fit_outputs,
)
from .pipeline import process_one_file, run_directory
from .plotting import plot_benchmark_outcomes
from .selection import (
    ADAPTIVE_LAMBDA_GRID_MODES,
    CV_STABILITY_LAMBDA_GRID_MODES,
    FIXED_LAMBDA_GRID_MODES,
    LAMBDA_GRID_MODES,
    LambdaBracket,
    compute_classic_bic,
    compute_extended_bic,
    default_lambda_grid,
    is_adaptive_lambda_grid_mode,
    is_cv_stability_lambda_grid_mode,
)
from .settings import (
    TumorRegime,
    PatientRegime,
    RecommendedSettings,
    recommend_settings_from_data,
    recommend_settings_from_regime,
    summarize_tumor_regime,
    summarize_patient_regime,
)

__all__ = [
    "MassiveMultiregionBenchmarkConfig",
    "ModelSelectionResult",
    "ADAPTIVE_LAMBDA_GRID_MODES",
    "CV_STABILITY_LAMBDA_GRID_MODES",
    "FIXED_LAMBDA_GRID_MODES",
    "LAMBDA_GRID_MODES",
    "LambdaBracket",
    "TumorRegime",
    "PatientRegime",
    "RecommendedSettings",
    "cell_output_table",
    "cluster_output_table",
    "compute_classic_bic",
    "compute_extended_bic",
    "default_lambda_grid",
    "is_adaptive_lambda_grid_mode",
    "is_cv_stability_lambda_grid_mode",
    "evaluation_to_frame",
    "mutation_output_table",
    "process_one_file",
    "plot_benchmark_outcomes",
    "recommend_settings_from_data",
    "recommend_settings_from_regime",
    "run_cohort_benchmark",
    "run_ray_cohort_benchmark",
    "run_directory",
    "run_massive_multiregion_benchmark",
    "run_simulation_benchmark",
    "run_single_region_cohort_benchmark",
    "select_model",
    "summarize_tumor_regime",
    "summarize_patient_regime",
    "write_fit_outputs",
]


def __getattr__(name: str):
    if name == "run_ray_cohort_benchmark":
        from .benchmark_cohort_ray import run_ray_cohort_benchmark

        return run_ray_cohort_benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
