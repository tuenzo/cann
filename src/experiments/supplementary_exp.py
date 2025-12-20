"""
Supplementary Experiments
=========================

Implements control analyses from the Appendix of Zhang et al., NeurIPS 2025:

- Fig S4: Single-layer control (random cue)
- Fig S5: Decoder comparison (PVM, COM, ML, Peak)
- Fig S6: Reversed STD/STF order
- Fig S7: Parameter sensitivity analysis
- Fig S8: Synaptic heterogeneity

These experiments validate the robustness of the main findings.
"""

from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp

from ..models.cann import CANNParams, SingleLayerCANN
from ..models.two_layer_cann import TwoLayerParams, TwoLayerCANN
from ..decoding import (
    decode_orientation,
    population_vector_decode,
    center_of_mass_decode,
    maximum_likelihood_decode,
    peak_decode,
)


# ==================== Fig S5: Decoder Comparison ====================

def run_decoder_comparison(
    n_trials: int = 20,
    isi: float = 1000.0,
    stp_type: str = 'std',
    verbose: bool = True,
) -> Dict:
    """Compare different decoding methods.
    
    Reproduces Appendix F analysis showing all four decoders
    produce consistent results.
    
    Args:
        n_trials: Trials per condition
        isi: Inter-stimulus interval
        stp_type: 'std' or 'stf'
        verbose: Show progress
        
    Returns:
        Dictionary with results for each decoder
    """
    # Setup
    if stp_type == 'std':
        params = CANNParams(tau_d=3.0, tau_f=0.3)
    else:
        params = CANNParams(tau_d=0.3, tau_f=5.0)
    
    model = SingleLayerCANN(params)
    theta = np.linspace(0, 180, params.N, endpoint=False)
    
    deltas = np.arange(-90, 91, 5)
    reference = 90.0
    stim_duration = 500.0
    
    # Results storage
    methods = ['pvm', 'com', 'ml', 'peak']
    results = {m: {'errors': []} for m in methods}
    
    iterator = tqdm(deltas, desc='Decoder Comparison') if verbose else deltas
    
    for delta in iterator:
        theta_s2 = reference
        theta_s1 = theta_s2 + delta
        if theta_s1 >= 180:
            theta_s1 -= 180
        elif theta_s1 < 0:
            theta_s1 += 180
        
        for _ in range(n_trials):
            model.reset()
            model.present_stimulus(theta_s1, stim_duration)
            model.evolve(isi)
            model.present_stimulus(theta_s2, stim_duration)
            
            # Get activity and decode with each method
            r = model.get_activity()
            
            for method in methods:
                perceived = decode_orientation(r, theta, method)
                error = perceived - theta_s2
                if error > 90:
                    error -= 180
                elif error < -90:
                    error += 180
                results[method]['errors'].append((delta, error))
    
    # Aggregate results
    output = {}
    for method in methods:
        data = results[method]['errors']
        deltas_arr = np.array([d[0] for d in data])
        errors_arr = np.array([d[1] for d in data])
        
        unique_deltas = np.unique(deltas_arr)
        mean_errors = np.array([
            np.mean(errors_arr[deltas_arr == d]) for d in unique_deltas
        ])
        
        output[method] = {
            'delta': unique_deltas,
            'errors': mean_errors,
        }
    
    return output


# ==================== Fig S6: Reversed Layer Order ====================

def run_reversed_layer_experiment(
    n_trials: int = 20,
    isi: float = 1000.0,
    iti: float = 3000.0,
    verbose: bool = True,
) -> Dict:
    """Test network with reversed STD/STF layer order.
    
    Lower layer: STF-dominated
    Higher layer: STD-dominated
    
    Expected: Both within-trial and between-trial show attraction
    (as per Appendix G, Fig S6).
    
    Args:
        n_trials: Trials per condition
        isi: Inter-stimulus interval
        iti: Inter-trial interval
        verbose: Show progress
        
    Returns:
        Dictionary with results
    """
    # Reversed parameters: STF in lower, STD in higher
    low_params = CANNParams(tau_d=0.3, tau_f=5.0)   # STF-dominated
    high_params = CANNParams(tau_d=3.0, tau_f=0.3)  # STD-dominated
    
    params = TwoLayerParams(low=low_params, high=high_params)
    model = TwoLayerCANN(params)
    
    deltas = np.arange(-90, 91, 5)
    reference = 90.0
    stim_duration = 500.0
    
    within_errors = []
    between_errors = []
    
    # Within-trial
    iterator = tqdm(deltas, desc='Reversed: Within-Trial') if verbose else deltas
    for delta in iterator:
        theta_s2 = reference
        theta_s1 = theta_s2 + delta
        if theta_s1 >= 180:
            theta_s1 -= 180
        elif theta_s1 < 0:
            theta_s1 += 180
        
        for _ in range(n_trials):
            model.reset()
            model.present_stimulus(theta_s1, stim_duration)
            model.evolve(isi)
            model.present_stimulus(theta_s2, stim_duration)
            
            perceived = model.decode('high')
            error = perceived - theta_s2
            if error > 90:
                error -= 180
            elif error < -90:
                error += 180
            within_errors.append((delta, error))
    
    # Between-trial
    iterator = tqdm(deltas, desc='Reversed: Between-Trial') if verbose else deltas
    for delta in iterator:
        theta_curr = reference
        theta_prev = theta_curr + delta
        if theta_prev >= 180:
            theta_prev -= 180
        elif theta_prev < 0:
            theta_prev += 180
        
        for _ in range(n_trials):
            model.reset()
            model.present_stimulus(theta_prev, stim_duration)
            model.evolve(iti)
            model.present_stimulus(theta_curr, stim_duration)
            
            perceived = model.decode('high')
            error = perceived - theta_curr
            if error > 90:
                error -= 180
            elif error < -90:
                error += 180
            between_errors.append((delta, error))
    
    # Aggregate
    def aggregate(errors):
        deltas_arr = np.array([e[0] for e in errors])
        errors_arr = np.array([e[1] for e in errors])
        unique_deltas = np.unique(deltas_arr)
        mean_errors = np.array([
            np.mean(errors_arr[deltas_arr == d]) for d in unique_deltas
        ])
        return unique_deltas, mean_errors
    
    within_delta, within_mean = aggregate(within_errors)
    between_delta, between_mean = aggregate(between_errors)
    
    return {
        'within_trial': {'delta': within_delta, 'errors': within_mean},
        'between_trial': {'delta': between_delta, 'errors': between_mean},
        'note': 'Both should show attraction with reversed layer order',
    }


# ==================== Fig S7: Parameter Sensitivity ====================

def run_parameter_sensitivity(
    param_sets: List[Dict],
    n_trials: int = 10,
    verbose: bool = True,
) -> Dict:
    """Test robustness to STP parameter variations.
    
    Varies τ_d and τ_f while maintaining STD or STF dominance.
    
    Args:
        param_sets: List of parameter dictionaries
        n_trials: Trials per condition
        verbose: Show progress
        
    Returns:
        Dictionary with results for each parameter set
    """
    if not param_sets:
        # Default parameter variations
        param_sets = [
            # STD variations
            {'tau_d': 2.0, 'tau_f': 0.1, 'name': 'STD: τd=2, τf=0.1'},
            {'tau_d': 3.0, 'tau_f': 0.3, 'name': 'STD: τd=3, τf=0.3 (default)'},
            {'tau_d': 4.0, 'tau_f': 0.3, 'name': 'STD: τd=4, τf=0.3'},
            # STF variations
            {'tau_d': 0.1, 'tau_f': 4.0, 'name': 'STF: τd=0.1, τf=4'},
            {'tau_d': 0.3, 'tau_f': 5.0, 'name': 'STF: τd=0.3, τf=5 (default)'},
            {'tau_d': 0.3, 'tau_f': 6.0, 'name': 'STF: τd=0.3, τf=6'},
        ]
    
    results = {}
    
    for ps in param_sets:
        name = ps.get('name', f"τd={ps['tau_d']}, τf={ps['tau_f']}")
        print(f"  Testing: {name}") if verbose else None
        
        params = CANNParams(tau_d=ps['tau_d'], tau_f=ps['tau_f'])
        model = SingleLayerCANN(params)
        theta = np.linspace(0, 180, params.N, endpoint=False)
        
        deltas = np.arange(-90, 91, 10)  # Coarser for speed
        reference = 90.0
        errors_list = []
        
        for delta in deltas:
            theta_s2 = reference
            theta_s1 = theta_s2 + delta
            if theta_s1 >= 180:
                theta_s1 -= 180
            elif theta_s1 < 0:
                theta_s1 += 180
            
            for _ in range(n_trials):
                model.reset()
                model.present_stimulus(theta_s1, 500)
                model.evolve(1000)
                model.present_stimulus(theta_s2, 500)
                
                perceived = model.decode()
                error = perceived - theta_s2
                if error > 90:
                    error -= 180
                elif error < -90:
                    error += 180
                errors_list.append((delta, error))
        
        deltas_arr = np.array([e[0] for e in errors_list])
        errors_arr = np.array([e[1] for e in errors_list])
        unique_deltas = np.unique(deltas_arr)
        mean_errors = np.array([
            np.mean(errors_arr[deltas_arr == d]) for d in unique_deltas
        ])
        
        results[name] = {
            'delta': unique_deltas,
            'errors': mean_errors,
            'tau_d': ps['tau_d'],
            'tau_f': ps['tau_f'],
        }
    
    return results


# ==================== Fig S8: Synaptic Heterogeneity ====================

def run_heterogeneity_experiment(
    heterogeneity_level: float = 0.1,
    n_trials: int = 20,
    stp_type: str = 'std',
    verbose: bool = True,
) -> Dict:
    """Test effect of synaptic heterogeneity on model performance.
    
    Adds ±heterogeneity_level random perturbation to STP parameters.
    
    Args:
        heterogeneity_level: Fraction of variation (0.1 = ±10%)
        n_trials: Trials per condition
        stp_type: 'std' or 'stf'
        verbose: Show progress
        
    Returns:
        Dictionary comparing homogeneous vs heterogeneous results
    """
    # Note: Full heterogeneity would require modifying the model
    # to have per-synapse STP parameters. Here we approximate by
    # running multiple trials with slightly different global parameters.
    
    if stp_type == 'std':
        base_tau_d, base_tau_f = 3.0, 0.3
    else:
        base_tau_d, base_tau_f = 0.3, 5.0
    
    deltas = np.arange(-90, 91, 5)
    reference = 90.0
    
    # Homogeneous (baseline)
    params_homo = CANNParams(tau_d=base_tau_d, tau_f=base_tau_f)
    model_homo = SingleLayerCANN(params_homo)
    
    homo_errors = []
    iterator = tqdm(deltas, desc='Homogeneous') if verbose else deltas
    for delta in iterator:
        theta_s2 = reference
        theta_s1 = theta_s2 + delta
        if theta_s1 >= 180:
            theta_s1 -= 180
        elif theta_s1 < 0:
            theta_s1 += 180
        
        for _ in range(n_trials):
            model_homo.reset()
            model_homo.present_stimulus(theta_s1, 500)
            model_homo.evolve(1000)
            model_homo.present_stimulus(theta_s2, 500)
            
            perceived = model_homo.decode()
            error = perceived - theta_s2
            if error > 90:
                error -= 180
            elif error < -90:
                error += 180
            homo_errors.append((delta, error))
    
    # Heterogeneous (multiple parameter samples)
    np.random.seed(42)
    hetero_errors = []
    iterator = tqdm(deltas, desc='Heterogeneous') if verbose else deltas
    for delta in iterator:
        theta_s2 = reference
        theta_s1 = theta_s2 + delta
        if theta_s1 >= 180:
            theta_s1 -= 180
        elif theta_s1 < 0:
            theta_s1 += 180
        
        for _ in range(n_trials):
            # Sample perturbed parameters
            tau_d_pert = base_tau_d * (1 + heterogeneity_level * (2 * np.random.rand() - 1))
            tau_f_pert = base_tau_f * (1 + heterogeneity_level * (2 * np.random.rand() - 1))
            
            params_hetero = CANNParams(tau_d=tau_d_pert, tau_f=tau_f_pert)
            model_hetero = SingleLayerCANN(params_hetero)
            
            model_hetero.present_stimulus(theta_s1, 500)
            model_hetero.evolve(1000)
            model_hetero.present_stimulus(theta_s2, 500)
            
            perceived = model_hetero.decode()
            error = perceived - theta_s2
            if error > 90:
                error -= 180
            elif error < -90:
                error += 180
            hetero_errors.append((delta, error))
    
    # Aggregate
    def aggregate(errors):
        deltas_arr = np.array([e[0] for e in errors])
        errors_arr = np.array([e[1] for e in errors])
        unique_deltas = np.unique(deltas_arr)
        mean_errors = np.array([
            np.mean(errors_arr[deltas_arr == d]) for d in unique_deltas
        ])
        return unique_deltas, mean_errors
    
    homo_delta, homo_mean = aggregate(homo_errors)
    hetero_delta, hetero_mean = aggregate(hetero_errors)
    
    return {
        'homogeneous': {'delta': homo_delta, 'errors': homo_mean},
        'heterogeneous': {'delta': hetero_delta, 'errors': hetero_mean},
        'heterogeneity_level': heterogeneity_level,
        'stp_type': stp_type,
    }


# ==================== Combined Supplementary Analysis ====================

def run_all_supplementary(
    n_trials: int = 10,
    verbose: bool = True,
) -> Dict:
    """Run all supplementary experiments.
    
    Args:
        n_trials: Trials per condition
        verbose: Show progress
        
    Returns:
        Dictionary with all supplementary results
    """
    print("=" * 60)
    print("Running Supplementary Experiments")
    print("=" * 60)
    
    results = {}
    
    print("\n[1/4] Decoder Comparison (Fig S5)...")
    results['decoder_comparison'] = run_decoder_comparison(
        n_trials=n_trials, stp_type='std', verbose=verbose
    )
    
    print("\n[2/4] Reversed Layer Order (Fig S6)...")
    results['reversed_layers'] = run_reversed_layer_experiment(
        n_trials=n_trials, verbose=verbose
    )
    
    print("\n[3/4] Parameter Sensitivity (Fig S7)...")
    results['parameter_sensitivity'] = run_parameter_sensitivity(
        param_sets=[], n_trials=n_trials, verbose=verbose
    )
    
    print("\n[4/4] Synaptic Heterogeneity (Fig S8)...")
    results['heterogeneity_std'] = run_heterogeneity_experiment(
        heterogeneity_level=0.1, stp_type='std', n_trials=n_trials, verbose=verbose
    )
    results['heterogeneity_stf'] = run_heterogeneity_experiment(
        heterogeneity_level=0.1, stp_type='stf', n_trials=n_trials, verbose=verbose
    )
    
    return results

