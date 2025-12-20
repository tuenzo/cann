"""Experimental protocols for serial dependence studies."""

from .single_layer_exp import (
    run_single_layer_experiment,
    run_experiment_with_recording,
    SingleLayerExperimentConfig,
)
from .two_layer_exp import (
    run_two_layer_experiment,
    run_within_trial_experiment,
    run_between_trial_experiment,
    run_isi_sweep,
    run_iti_sweep,
    TwoLayerExperimentConfig,
)
from .bayesian_analysis import (
    run_bayesian_analysis,
    simulate_bayesian_trial,
    decompose_bias,
    BayesianParams,
)
from .supplementary_exp import (
    run_decoder_comparison,
    run_reversed_layer_experiment,
    run_parameter_sensitivity,
    run_heterogeneity_experiment,
    run_all_supplementary,
)

__all__ = [
    # Single layer
    "run_single_layer_experiment",
    "run_experiment_with_recording",
    "SingleLayerExperimentConfig",
    # Two layer
    "run_two_layer_experiment",
    "run_within_trial_experiment",
    "run_between_trial_experiment",
    "run_isi_sweep",
    "run_iti_sweep",
    "TwoLayerExperimentConfig",
    # Bayesian
    "run_bayesian_analysis",
    "simulate_bayesian_trial",
    "decompose_bias",
    "BayesianParams",
    # Supplementary
    "run_decoder_comparison",
    "run_reversed_layer_experiment",
    "run_parameter_sensitivity",
    "run_heterogeneity_experiment",
    "run_all_supplementary",
]

