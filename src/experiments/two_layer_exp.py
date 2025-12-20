"""
Two-Layer CANN Experiments
==========================

Reproduces Figure 3-4 from Zhang et al., NeurIPS 2025:
- Fig 3: Within-trial (repulsion) vs Between-trial (attraction)
- Fig 4: Temporal window analysis (ISI/ITI effects)

Key findings:
- Within-trial: STD in lower layer → repulsion from S1 during S2 perception
- Between-trial: STF in higher layer → attraction to previous trial's S1
"""

from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from ..models.cann import CANNParams
from ..models.two_layer_cann import TwoLayerParams, TwoLayerCANN


@dataclass
class TwoLayerExperimentConfig:
    """Configuration for two-layer experiments."""
    # Network parameters
    N: int = 180
    J0: float = 0.5
    a: float = 30.0
    tau: float = 1.0
    k: float = 0.5
    dt: float = 0.1
    
    # Lower layer STP (STD-dominated)
    low_tau_d: float = 3.0
    low_tau_f: float = 0.3
    
    # Higher layer STP (STF-dominated)
    high_tau_d: float = 0.3
    high_tau_f: float = 5.0
    
    U: float = 0.3
    
    # Inter-layer connections
    J_ff: float = 0.3
    J_fb: float = 0.2
    a_ff: float = 30.0
    a_fb: float = 30.0
    
    # Stimulus parameters
    stim_duration: float = 500.0  # ms
    stim_amplitude: float = 1.0
    stim_width: float = 30.0
    
    # Timing parameters
    isi: float = 1000.0  # ms (within-trial)
    iti: float = 3000.0  # ms (between-trial)
    
    # Experiment parameters
    reference_theta: float = 90.0
    delta_range: Tuple[float, float] = (-90.0, 90.0)
    delta_step: float = 5.0
    n_trials: int = 20
    
    decode_method: str = 'pvm'
    decode_layer: str = 'high'
    
    def to_two_layer_params(self) -> TwoLayerParams:
        """Convert to TwoLayerParams."""
        low_params = CANNParams(
            N=self.N, J0=self.J0, a=self.a, tau=self.tau, k=self.k,
            A=self.stim_amplitude, stim_width=self.stim_width, dt=self.dt,
            tau_d=self.low_tau_d, tau_f=self.low_tau_f, U=self.U,
        )
        high_params = CANNParams(
            N=self.N, J0=self.J0, a=self.a, tau=self.tau, k=self.k,
            A=self.stim_amplitude, stim_width=self.stim_width, dt=self.dt,
            tau_d=self.high_tau_d, tau_f=self.high_tau_f, U=self.U,
        )
        return TwoLayerParams(
            low=low_params, high=high_params,
            J_ff=self.J_ff, J_fb=self.J_fb, a_ff=self.a_ff, a_fb=self.a_fb,
        )


def run_within_trial_experiment(
    config: Optional[TwoLayerExperimentConfig] = None,
    verbose: bool = True,
) -> Dict:
    """Run within-trial serial dependence experiment.
    
    Protocol: S1 → ISI → S2 (same trial)
    Expected: Repulsion from S1 due to STD in lower layer
    
    Args:
        config: Experiment configuration
        verbose: Show progress
        
    Returns:
        Dictionary with results
    """
    if config is None:
        config = TwoLayerExperimentConfig()
    
    params = config.to_two_layer_params()
    model = TwoLayerCANN(params)
    
    deltas = np.arange(
        config.delta_range[0],
        config.delta_range[1] + config.delta_step,
        config.delta_step
    )
    
    all_deltas = []
    all_errors = []
    
    iterator = tqdm(deltas, desc='Within-Trial') if verbose else deltas
    
    for delta in iterator:
        theta_s2 = config.reference_theta
        theta_s1 = theta_s2 + delta
        if theta_s1 >= 180:
            theta_s1 -= 180
        elif theta_s1 < 0:
            theta_s1 += 180
        
        for _ in range(config.n_trials):
            model.reset()
            
            # Present S1
            model.present_stimulus(theta_s1, config.stim_duration)
            
            # ISI
            model.evolve(config.isi)
            
            # Present S2
            model.present_stimulus(theta_s2, config.stim_duration)
            
            # Decode
            perceived = model.decode(config.decode_layer, config.decode_method)
            
            # Error
            error = perceived - theta_s2
            if error > 90:
                error -= 180
            elif error < -90:
                error += 180
            
            all_deltas.append(delta)
            all_errors.append(error)
    
    deltas_arr = np.array(all_deltas)
    errors_arr = np.array(all_errors)
    
    unique_deltas = np.unique(deltas_arr)
    mean_errors = np.array([np.mean(errors_arr[deltas_arr == d]) for d in unique_deltas])
    std_errors = np.array([np.std(errors_arr[deltas_arr == d]) for d in unique_deltas])
    
    return {
        'experiment_type': 'within_trial',
        'delta': unique_deltas,
        'errors': mean_errors,
        'errors_std': std_errors,
        'all_deltas': deltas_arr,
        'all_errors': errors_arr,
        'isi': config.isi,
    }


def run_between_trial_experiment(
    config: Optional[TwoLayerExperimentConfig] = None,
    verbose: bool = True,
) -> Dict:
    """Run between-trial serial dependence experiment.
    
    Protocol: Trial_n-1(S1) → ITI → Trial_n(S1)
    Expected: Attraction to previous S1 due to STF in higher layer
    
    Args:
        config: Experiment configuration
        verbose: Show progress
        
    Returns:
        Dictionary with results
    """
    if config is None:
        config = TwoLayerExperimentConfig()
    
    params = config.to_two_layer_params()
    model = TwoLayerCANN(params)
    
    deltas = np.arange(
        config.delta_range[0],
        config.delta_range[1] + config.delta_step,
        config.delta_step
    )
    
    all_deltas = []
    all_errors = []
    
    iterator = tqdm(deltas, desc='Between-Trial') if verbose else deltas
    
    for delta in iterator:
        theta_curr = config.reference_theta  # Current trial S1
        theta_prev = theta_curr + delta  # Previous trial S1
        if theta_prev >= 180:
            theta_prev -= 180
        elif theta_prev < 0:
            theta_prev += 180
        
        for _ in range(config.n_trials):
            model.reset()
            
            # Previous trial S1
            model.present_stimulus(theta_prev, config.stim_duration)
            
            # ITI
            model.evolve(config.iti)
            
            # Current trial S1
            model.present_stimulus(theta_curr, config.stim_duration)
            
            # Decode
            perceived = model.decode(config.decode_layer, config.decode_method)
            
            # Error
            error = perceived - theta_curr
            if error > 90:
                error -= 180
            elif error < -90:
                error += 180
            
            all_deltas.append(delta)
            all_errors.append(error)
    
    deltas_arr = np.array(all_deltas)
    errors_arr = np.array(all_errors)
    
    unique_deltas = np.unique(deltas_arr)
    mean_errors = np.array([np.mean(errors_arr[deltas_arr == d]) for d in unique_deltas])
    std_errors = np.array([np.std(errors_arr[deltas_arr == d]) for d in unique_deltas])
    
    return {
        'experiment_type': 'between_trial',
        'delta': unique_deltas,
        'errors': mean_errors,
        'errors_std': std_errors,
        'all_deltas': deltas_arr,
        'all_errors': errors_arr,
        'iti': config.iti,
    }


def run_two_layer_experiment(
    config: Optional[TwoLayerExperimentConfig] = None,
    verbose: bool = True,
) -> Dict:
    """Run both within-trial and between-trial experiments.
    
    Args:
        config: Experiment configuration
        verbose: Show progress
        
    Returns:
        Dictionary with both results
    """
    within_results = run_within_trial_experiment(config, verbose)
    between_results = run_between_trial_experiment(config, verbose)
    
    return {
        'within_trial': within_results,
        'between_trial': between_results,
        'config': config,
    }


def run_isi_sweep(
    isi_values: List[float],
    config: Optional[TwoLayerExperimentConfig] = None,
    delta_test: float = 30.0,
    verbose: bool = True,
) -> Dict:
    """Sweep ISI values for within-trial experiment.
    
    Tests how within-trial repulsion decays with ISI.
    
    Args:
        isi_values: ISI values to test (ms)
        config: Configuration
        delta_test: Fixed delta to test
        verbose: Show progress
        
    Returns:
        Dictionary with ISI sweep results
    """
    if config is None:
        config = TwoLayerExperimentConfig()
    
    params = config.to_two_layer_params()
    model = TwoLayerCANN(params)
    
    theta_s2 = config.reference_theta
    theta_s1 = theta_s2 + delta_test
    if theta_s1 >= 180:
        theta_s1 -= 180
    
    mean_biases = []
    std_biases = []
    
    iterator = tqdm(isi_values, desc='ISI Sweep') if verbose else isi_values
    
    for isi in iterator:
        errors = []
        for _ in range(config.n_trials):
            model.reset()
            model.present_stimulus(theta_s1, config.stim_duration)
            model.evolve(isi)
            model.present_stimulus(theta_s2, config.stim_duration)
            
            perceived = model.decode(config.decode_layer)
            error = perceived - theta_s2
            if error > 90:
                error -= 180
            elif error < -90:
                error += 180
            errors.append(error)
        
        mean_biases.append(np.mean(errors))
        std_biases.append(np.std(errors))
    
    return {
        'isi_values': np.array(isi_values),
        'bias': np.array(mean_biases),
        'bias_std': np.array(std_biases),
        'delta': delta_test,
        'experiment_type': 'isi_sweep',
    }


def run_iti_sweep(
    iti_values: List[float],
    config: Optional[TwoLayerExperimentConfig] = None,
    delta_test: float = 30.0,
    verbose: bool = True,
) -> Dict:
    """Sweep ITI values for between-trial experiment.
    
    Tests how between-trial attraction decays with ITI.
    
    Args:
        iti_values: ITI values to test (ms)
        config: Configuration
        delta_test: Fixed delta to test
        verbose: Show progress
        
    Returns:
        Dictionary with ITI sweep results
    """
    if config is None:
        config = TwoLayerExperimentConfig()
    
    params = config.to_two_layer_params()
    model = TwoLayerCANN(params)
    
    theta_curr = config.reference_theta
    theta_prev = theta_curr + delta_test
    if theta_prev >= 180:
        theta_prev -= 180
    
    mean_biases = []
    std_biases = []
    
    iterator = tqdm(iti_values, desc='ITI Sweep') if verbose else iti_values
    
    for iti in iterator:
        errors = []
        for _ in range(config.n_trials):
            model.reset()
            model.present_stimulus(theta_prev, config.stim_duration)
            model.evolve(iti)
            model.present_stimulus(theta_curr, config.stim_duration)
            
            perceived = model.decode(config.decode_layer)
            error = perceived - theta_curr
            if error > 90:
                error -= 180
            elif error < -90:
                error += 180
            errors.append(error)
        
        mean_biases.append(np.mean(errors))
        std_biases.append(np.std(errors))
    
    return {
        'iti_values': np.array(iti_values),
        'bias': np.array(mean_biases),
        'bias_std': np.array(std_biases),
        'delta': delta_test,
        'experiment_type': 'iti_sweep',
    }


def run_temporal_window_analysis(
    isi_values: List[float] = [500, 1000, 2000, 4000],
    iti_values: List[float] = [1000, 2000, 4000, 8000],
    config: Optional[TwoLayerExperimentConfig] = None,
    verbose: bool = True,
) -> Dict:
    """Run comprehensive temporal window analysis.
    
    Reproduces Figure 4 from the paper.
    
    Args:
        isi_values: ISI values to test
        iti_values: ITI values to test
        config: Configuration
        verbose: Show progress
        
    Returns:
        Dictionary with temporal analysis results
    """
    isi_results = run_isi_sweep(isi_values, config, verbose=verbose)
    iti_results = run_iti_sweep(iti_values, config, verbose=verbose)
    
    return {
        'isi_sweep': isi_results,
        'iti_sweep': iti_results,
    }


def run_with_recording(
    config: Optional[TwoLayerExperimentConfig] = None,
    delta: float = 30.0,
) -> Dict:
    """Run experiment with activity recording.
    
    Useful for visualizing layer activities over time.
    
    Args:
        config: Configuration
        delta: Stimulus difference
        
    Returns:
        Dictionary with recorded activities
    """
    if config is None:
        config = TwoLayerExperimentConfig()
    
    params = config.to_two_layer_params()
    model = TwoLayerCANN(params)
    model.reset()
    
    theta_s2 = config.reference_theta
    theta_s1 = theta_s2 + delta
    if theta_s1 >= 180:
        theta_s1 -= 180
    
    # Storage
    times = []
    low_activities = []
    high_activities = []
    
    # S1
    s1_hist = model.present_stimulus(theta_s1, config.stim_duration, record=True)
    times.extend(s1_hist['time'])
    low_activities.extend(s1_hist['low_r'])
    high_activities.extend(s1_hist['high_r'])
    
    # ISI (subsample for efficiency)
    n_isi = int(config.isi / config.dt)
    for i in range(n_isi):
        model.step()
        if i % 10 == 0:
            times.append(model.time)
            low_activities.append(model.state.low.r)
            high_activities.append(model.state.high.r)
    
    # S2
    s2_hist = model.present_stimulus(theta_s2, config.stim_duration, record=True)
    times.extend(s2_hist['time'])
    low_activities.extend(s2_hist['low_r'])
    high_activities.extend(s2_hist['high_r'])
    
    return {
        'time': np.array(times),
        'low_activity': np.array(low_activities),
        'high_activity': np.array(high_activities),
        'theta_s1': theta_s1,
        'theta_s2': theta_s2,
    }

