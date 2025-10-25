"""
Utility modules for neuro-IDE analysis and visualization.
"""

# Try importing SE+ST utilities (requires scanpy)
try:
    from .se_st_utils import (
        preprocess_data_for_se_st,
        create_se_st_batch,
        load_se_st_model,
        predict_perturbation_effects,
        evaluate_cross_cell_type_performance,
        compare_with_baseline,
        save_results,
    )
    _has_se_st = True
except ImportError:
    _has_se_st = False

# Neural ODE analysis utilities (core dependencies only)
from .neural_ode_analysis import (
    analyze_perturbation_dynamics,
    visualize_perturbation_dynamics,
    compare_perturbations,
    export_analysis_data
)

__all__ = [
    # Neural ODE analysis utilities
    "analyze_perturbation_dynamics",
    "visualize_perturbation_dynamics",
    "compare_perturbations",
    "export_analysis_data"
]

# Add SE+ST utilities if available
if _has_se_st:
    __all__.extend([
        "preprocess_data_for_se_st",
        "create_se_st_batch",
        "load_se_st_model",
        "predict_perturbation_effects",
        "evaluate_cross_cell_type_performance",
        "compare_with_baseline",
        "save_results",
    ])
