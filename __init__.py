"""CliPP2: certified BIC selection for observed-data pairwise fusion."""

from ._version import __version__
from .core import (
    FitOptions,
    FitResult,
    PairwiseFusionGraph,
    fit_observed_data_pairwise_fusion,
    fit_single_stage_em,
    resolve_pairwise_fusion_graph,
)
from .io.data import PatientData, TumorData, load_patient_tsv, load_tumor_tsv
from .runners import (
    BICSelectionResult,
    LambdaBracket,
    ModelSelectionResult,
    RecommendedSettings,
    SimulationDiagnostics,
    TumorRegime,
    default_lambda_grid,
    process_one_file,
    recommend_settings_from_data,
    run_directory,
    select_model,
    summarize_tumor_regime,
)

__all__ = [
    "BICSelectionResult",
    "FitOptions",
    "FitResult",
    "LambdaBracket",
    "ModelSelectionResult",
    "PairwiseFusionGraph",
    "PatientData",
    "RecommendedSettings",
    "SimulationDiagnostics",
    "TumorData",
    "TumorRegime",
    "__version__",
    "default_lambda_grid",
    "fit_observed_data_pairwise_fusion",
    "fit_single_stage_em",
    "load_patient_tsv",
    "load_tumor_tsv",
    "process_one_file",
    "recommend_settings_from_data",
    "resolve_pairwise_fusion_graph",
    "run_directory",
    "select_model",
    "summarize_tumor_regime",
]
