"""
Single-Layer CANN Experiments
=============================

Reproduces Figure 2 from Zhang et al., NeurIPS 2025:
- Fig 2A-C: STD-dominated CANN → Repulsion effect
- Fig 2D-F: STF-dominated CANN → Attraction effect

Experiment protocol:
1. Present S1 at θ₁ for stim_duration
2. Wait for ISI (inter-stimulus interval)
3. Present S2 at θ₂ = θ₁ + Δθ for stim_duration
4. Decode perceived S2
5. Compute adjustment error
"""

from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from ..models.cann import CANNParams, SingleLayerCANN, create_stimulus
from ..decoding import decode_orientation


@dataclass
class SingleLayerExperimentConfig:
    """Configuration for single-layer experiments."""
    # Network parameters
    N: int = 180
    J0: float = 0.5
    a: float = 30.0
    tau: float = 1.0
    k: float = 0.5
    dt: float = 0.1
    
    # STP parameters
    tau_d: float = 3.0
    tau_f: float = 0.3
    U: float = 0.3
    
    # Stimulus parameters
    stim_duration: float = 500.0  # ms
    stim_amplitude: float = 1.0
    stim_width: float = 30.0  # degrees
    
    # Experiment parameters
    isi: float = 1000.0  # ms
    reference_theta: float = 90.0  # degrees (S1 position)
    delta_range: Tuple[float, float] = (-90.0, 90.0)
    delta_step: float = 5.0
    n_trials: int = 20
    
    # Decoding
    decode_method: str = 'pvm'
    
    def to_cann_params(self) -> CANNParams:
        """Convert to CANNParams."""
        return CANNParams(
            N=self.N, J0=self.J0, a=self.a, tau=self.tau, k=self.k,
            A=self.stim_amplitude, stim_width=self.stim_width, dt=self.dt,
            tau_d=self.tau_d, tau_f=self.tau_f, U=self.U,
        )


def run_single_trial(
    model: SingleLayerCANN,
    theta_s1: float,
    theta_s2: float,
    stim_duration: float,
    isi: float,
    decode_method: str = 'pvm',
    record: bool = False,
) -> Dict:
    """Run a single serial dependence trial.
    
    Args:
        model: CANN model
        theta_s1: S1 orientation (degrees)
        theta_s2: S2 orientation (degrees)
        stim_duration: Stimulus duration (ms)
        isi: Inter-stimulus interval (ms)
        decode_method: Decoding method
        record: Whether to record time course
        
    Returns:
        Dictionary with trial results
    """
    model.reset()
    
    result = {'theta_s1': theta_s1, 'theta_s2': theta_s2}
    
    # Present S1
    s1_history = model.present_stimulus(theta_s1, stim_duration, record=record)
    if record and s1_history:
        result['s1_history'] = s1_history
    
    # ISI period
    model.evolve(isi)
    
    # Present S2
    s2_history = model.present_stimulus(theta_s2, stim_duration, record=record)
    if record and s2_history:
        result['s2_history'] = s2_history
    
    # Decode perceived orientation
    perceived = model.decode(method=decode_method)
    result['perceived'] = perceived
    
    # Compute adjustment error
    error = perceived - theta_s2
    # Wrap to [-90, 90]
    if error > 90:
        error -= 180
    elif error < -90:
        error += 180
    result['error'] = error
    
    # Delta (S1 - S2)
    delta = theta_s1 - theta_s2
    if delta > 90:
        delta -= 180
    elif delta < -90:
        delta += 180
    result['delta'] = delta
    
    return result


def run_single_layer_experiment(
    config: Optional[SingleLayerExperimentConfig] = None,
    stp_type: str = 'std',
    verbose: bool = True,
) -> Dict:
    """Run complete single-layer serial dependence experiment.
    
    Args:
        config: Experiment configuration
        stp_type: 'std' for STD-dominated (repulsion) or 'stf' for STF-dominated (attraction)
        verbose: Show progress bar
        
    Returns:
        Dictionary with all experimental results
    """
    if config is None:
        config = SingleLayerExperimentConfig()
    
    # Set STP parameters based on type
    if stp_type == 'std':
        config.tau_d = 3.0
        config.tau_f = 0.3
    elif stp_type == 'stf':
        config.tau_d = 0.3
        config.tau_f = 5.0
    else:
        raise ValueError(f"Unknown stp_type: {stp_type}")
    
    # Create model
    params = config.to_cann_params()
    model = SingleLayerCANN(params)
    
    # Generate delta values
    deltas = np.arange(
        config.delta_range[0], 
        config.delta_range[1] + config.delta_step,
        config.delta_step
    )
    
    # Storage for results
    all_deltas = []
    all_errors = []
    all_trials = []
    
    # Run experiments
    iterator = tqdm(deltas, desc=f'{stp_type.upper()} Experiment') if verbose else deltas
    
    for delta in iterator:
        theta_s2 = config.reference_theta
        theta_s1 = theta_s2 + delta
        
        # Wrap to [0, 180]
        if theta_s1 >= 180:
            theta_s1 -= 180
        elif theta_s1 < 0:
            theta_s1 += 180
        
        for trial in range(config.n_trials):
            result = run_single_trial(
                model, theta_s1, theta_s2,
                config.stim_duration, config.isi,
                config.decode_method
            )
            all_deltas.append(delta)
            all_errors.append(result['error'])
            all_trials.append(result)
    
    # Aggregate results
    deltas_arr = np.array(all_deltas)
    errors_arr = np.array(all_errors)
    
    # Compute mean error for each delta
    unique_deltas = np.unique(deltas_arr)
    mean_errors = np.array([
        np.mean(errors_arr[deltas_arr == d]) for d in unique_deltas
    ])
    std_errors = np.array([
        np.std(errors_arr[deltas_arr == d]) for d in unique_deltas
    ])
    
    return {
        'config': config,
        'stp_type': stp_type,
        'delta': unique_deltas,
        'errors': mean_errors,
        'errors_std': std_errors,
        'all_deltas': deltas_arr,
        'all_errors': errors_arr,
        'all_trials': all_trials,
        'theta': np.linspace(0, 180, config.N, endpoint=False),
    }


def run_experiment_with_recording(
    config: Optional[SingleLayerExperimentConfig] = None,
    stp_type: str = 'std',
    delta_to_record: float = 30.0,
) -> Dict:
    """Run experiment with neural activity recording for one delta value.
    
    Used for generating activity heatmaps and STP dynamics plots.
    
    Args:
        config: Experiment configuration
        stp_type: 'std' or 'stf'
        delta_to_record: Delta value to record (degrees)
        
    Returns:
        Dictionary with recorded time courses
    """
    if config is None:
        config = SingleLayerExperimentConfig()
    
    if stp_type == 'std':
        config.tau_d = 3.0
        config.tau_f = 0.3
    else:
        config.tau_d = 0.3
        config.tau_f = 5.0
    
    params = config.to_cann_params()
    model = SingleLayerCANN(params)
    model.reset()
    
    theta_s2 = config.reference_theta
    theta_s1 = theta_s2 + delta_to_record
    if theta_s1 >= 180:
        theta_s1 -= 180
    elif theta_s1 < 0:
        theta_s1 += 180
    
    # Record time courses
    all_time = []
    all_activity = []
    all_stp_x = []
    all_stp_u = []
    
    # S1 presentation
    s1_history = model.present_stimulus(theta_s1, config.stim_duration, record=True)
    all_time.extend(s1_history['time'])
    all_activity.extend(s1_history['r'])
    all_stp_x.extend(s1_history['stp_x'])
    all_stp_u.extend(s1_history['stp_u'])
    
    # ISI (record every 10 steps for efficiency)
    n_isi_steps = int(config.isi / config.dt)
    for i in range(n_isi_steps):
        model.step()
        if i % 10 == 0:
            all_time.append(model.time)
            all_activity.append(model.state.r)
            all_stp_x.append(model.state.stp.x)
            all_stp_u.append(model.state.stp.u)
    
    # S2 presentation
    s2_history = model.present_stimulus(theta_s2, config.stim_duration, record=True)
    all_time.extend(s2_history['time'])
    all_activity.extend(s2_history['r'])
    all_stp_x.extend(s2_history['stp_x'])
    all_stp_u.extend(s2_history['stp_u'])
    
    # Find neuron index for stim position
    theta_arr = np.linspace(0, 180, config.N, endpoint=False)
    stim_neuron = np.argmin(np.abs(theta_arr - theta_s1))
    
    return {
        'time': np.array(all_time),
        'activity': np.array(all_activity),
        'stp_x': np.array(all_stp_x),
        'stp_u': np.array(all_stp_u),
        'theta': theta_arr,
        'theta_s1': theta_s1,
        'theta_s2': theta_s2,
        'stim_neuron': stim_neuron,
        'stp_type': stp_type,
    }


def run_isi_sweep(
    isi_values: List[float],
    config: Optional[SingleLayerExperimentConfig] = None,
    stp_type: str = 'std',
    delta_test: float = 30.0,
    verbose: bool = True,
) -> Dict:
    """Run experiment with different ISI values.
    
    Tests how serial dependence decays with ISI.
    
    Args:
        isi_values: List of ISI values to test (ms)
        config: Experiment configuration
        stp_type: 'std' or 'stf'
        delta_test: Delta value to test
        verbose: Show progress
        
    Returns:
        Dictionary with ISI sweep results
    """
    if config is None:
        config = SingleLayerExperimentConfig()
    
    if stp_type == 'std':
        config.tau_d = 3.0
        config.tau_f = 0.3
    else:
        config.tau_d = 0.3
        config.tau_f = 5.0
    
    params = config.to_cann_params()
    model = SingleLayerCANN(params)
    
    theta_s2 = config.reference_theta
    theta_s1 = theta_s2 + delta_test
    if theta_s1 >= 180:
        theta_s1 -= 180
    
    mean_errors = []
    std_errors = []
    
    iterator = tqdm(isi_values, desc='ISI Sweep') if verbose else isi_values
    
    for isi in iterator:
        trial_errors = []
        for _ in range(config.n_trials):
            result = run_single_trial(
                model, theta_s1, theta_s2,
                config.stim_duration, isi, config.decode_method
            )
            trial_errors.append(result['error'])
        
        mean_errors.append(np.mean(trial_errors))
        std_errors.append(np.std(trial_errors))
    
    return {
        'isi_values': np.array(isi_values),
        'bias': np.array(mean_errors),
        'bias_std': np.array(std_errors),
        'stp_type': stp_type,
        'delta': delta_test,
    }


def compare_stp_types(
    config: Optional[SingleLayerExperimentConfig] = None,
    verbose: bool = True,
) -> Dict:
    """Run both STD and STF experiments for comparison.
    
    Args:
        config: Experiment configuration
        verbose: Show progress
        
    Returns:
        Dictionary with both experiment results
    """
    std_results = run_single_layer_experiment(config, 'std', verbose)
    stf_results = run_single_layer_experiment(config, 'stf', verbose)
    
    return {
        'std': std_results,
        'stf': stf_results,
    }

