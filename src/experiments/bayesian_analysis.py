"""
Bayesian Interpretation of Serial Dependence
=============================================

This module implements the Bayesian framework for understanding
serial dependence effects as described in Zhang et al., NeurIPS 2025.

Key concepts:
- Efficient encoding: Prior stimuli recalibrate neural sensitivity (repulsion)
- Bayesian decoding: Prior distribution biases posterior perception (attraction)

The STP mechanism provides a neural implementation of both:
- STD → Efficient encoding → Repulsion
- STF → Prior maintenance → Attraction

Reference: Zhang et al., NeurIPS 2025 (Section 2.4)
"""

from typing import Optional, Tuple, Dict, Callable
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import vonmises
from functools import partial


@dataclass
class BayesianParams:
    """Parameters for Bayesian analysis."""
    # Prior parameters
    prior_mean: float = 90.0       # Prior center (degrees)
    prior_kappa: float = 2.0       # Prior concentration (von Mises)
    
    # Likelihood parameters
    likelihood_kappa: float = 10.0  # Sensory precision
    
    # Encoding shift (from STD)
    encoding_shift_gain: float = 0.1  # How much prior shifts encoding
    
    # Prior update rate (from STF)
    prior_update_rate: float = 0.3    # How fast prior updates


def von_mises_pdf(theta: np.ndarray, mu: float, kappa: float) -> np.ndarray:
    """Von Mises probability density function.
    
    Circular analog of Gaussian distribution.
    
    Args:
        theta: Angles in degrees
        mu: Mean direction in degrees
        kappa: Concentration parameter (higher = more concentrated)
        
    Returns:
        PDF values
    """
    theta_rad = np.deg2rad(theta)
    mu_rad = np.deg2rad(mu)
    return vonmises.pdf(theta_rad, kappa, loc=mu_rad)


def compute_likelihood(
    theta: np.ndarray,
    theta_true: float,
    kappa: float,
    encoding_shift: float = 0.0,
) -> np.ndarray:
    """Compute likelihood function with optional encoding shift.
    
    The likelihood represents sensory evidence for each orientation.
    STD causes an encoding shift away from prior stimuli.
    
    Args:
        theta: Candidate orientations (degrees)
        theta_true: True stimulus orientation
        kappa: Sensory precision
        encoding_shift: Shift due to efficient encoding (degrees)
        
    Returns:
        Likelihood values (unnormalized)
    """
    # Effective stimulus after encoding shift
    theta_encoded = theta_true + encoding_shift
    return von_mises_pdf(theta, theta_encoded, kappa)


def compute_prior(
    theta: np.ndarray,
    prior_mean: float,
    prior_kappa: float,
) -> np.ndarray:
    """Compute prior distribution.
    
    The prior represents expectations based on recent history.
    STF maintains a prior centered on recent stimuli.
    
    Args:
        theta: Candidate orientations (degrees)
        prior_mean: Prior center (previous stimulus)
        prior_kappa: Prior concentration
        
    Returns:
        Prior probability values
    """
    return von_mises_pdf(theta, prior_mean, prior_kappa)


def bayesian_inference(
    theta: np.ndarray,
    likelihood: np.ndarray,
    prior: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Perform Bayesian inference to compute posterior.
    
    posterior ∝ likelihood × prior
    
    Args:
        theta: Candidate orientations
        likelihood: Likelihood values
        prior: Prior values
        
    Returns:
        posterior: Posterior distribution
        map_estimate: Maximum a posteriori estimate
    """
    # Posterior
    posterior = likelihood * prior
    posterior = posterior / (np.sum(posterior) * np.diff(theta)[0])  # Normalize
    
    # MAP estimate
    map_idx = np.argmax(posterior)
    map_estimate = theta[map_idx]
    
    return posterior, map_estimate


def simulate_bayesian_trial(
    theta_prior: float,
    theta_true: float,
    params: BayesianParams,
    theta_range: np.ndarray = None,
) -> Dict:
    """Simulate Bayesian perception for a single trial.
    
    Models how prior stimulus influences perception of current stimulus
    through both encoding (repulsion) and decoding (attraction).
    
    Args:
        theta_prior: Prior stimulus orientation (S1)
        theta_true: Current stimulus orientation (S2)
        params: Bayesian parameters
        theta_range: Range of candidate orientations
        
    Returns:
        Dictionary with simulation results
    """
    if theta_range is None:
        theta_range = np.linspace(0, 180, 181)
    
    # Compute encoding shift (repulsion from prior)
    delta = theta_true - theta_prior
    # Wrap to [-90, 90]
    if delta > 90:
        delta -= 180
    elif delta < -90:
        delta += 180
    
    encoding_shift = -params.encoding_shift_gain * delta  # Repulsive shift
    
    # Likelihood with encoding shift
    likelihood = compute_likelihood(
        theta_range, theta_true, params.likelihood_kappa, encoding_shift
    )
    
    # Prior centered on previous stimulus
    prior = compute_prior(theta_range, theta_prior, params.prior_kappa)
    
    # Bayesian inference
    posterior, perceived = bayesian_inference(theta_range, likelihood, prior)
    
    # Compute bias
    bias = perceived - theta_true
    if bias > 90:
        bias -= 180
    elif bias < -90:
        bias += 180
    
    return {
        'theta_prior': theta_prior,
        'theta_true': theta_true,
        'perceived': perceived,
        'bias': bias,
        'encoding_shift': encoding_shift,
        'theta_range': theta_range,
        'likelihood': likelihood,
        'prior': prior,
        'posterior': posterior,
    }


def run_bayesian_analysis(
    deltas: np.ndarray = None,
    params: BayesianParams = None,
    reference: float = 90.0,
) -> Dict:
    """Run Bayesian analysis across stimulus differences.
    
    Args:
        deltas: S1-S2 differences to test
        params: Bayesian parameters
        reference: Reference orientation for S2
        
    Returns:
        Dictionary with analysis results
    """
    if deltas is None:
        deltas = np.arange(-90, 91, 5)
    if params is None:
        params = BayesianParams()
    
    biases = []
    results = []
    
    for delta in deltas:
        theta_prior = reference + delta
        if theta_prior >= 180:
            theta_prior -= 180
        elif theta_prior < 0:
            theta_prior += 180
        
        result = simulate_bayesian_trial(theta_prior, reference, params)
        biases.append(result['bias'])
        results.append(result)
    
    return {
        'deltas': deltas,
        'biases': np.array(biases),
        'params': params,
        'trials': results,
    }


def decompose_bias(
    theta_prior: float,
    theta_true: float,
    params: BayesianParams,
) -> Dict:
    """Decompose bias into encoding and decoding components.
    
    Total bias = Encoding bias (repulsion) + Decoding bias (attraction)
    
    Args:
        theta_prior: Prior stimulus
        theta_true: Current stimulus
        params: Bayesian parameters
        
    Returns:
        Dictionary with bias decomposition
    """
    theta_range = np.linspace(0, 180, 181)
    
    delta = theta_true - theta_prior
    if delta > 90:
        delta -= 180
    elif delta < -90:
        delta += 180
    
    # === Encoding-only (no prior) ===
    encoding_shift = -params.encoding_shift_gain * delta
    likelihood_shifted = compute_likelihood(
        theta_range, theta_true, params.likelihood_kappa, encoding_shift
    )
    uniform_prior = np.ones_like(theta_range) / len(theta_range)
    _, perceived_encoding = bayesian_inference(
        theta_range, likelihood_shifted, uniform_prior
    )
    encoding_bias = perceived_encoding - theta_true
    
    # === Decoding-only (no encoding shift) ===
    likelihood_unshifted = compute_likelihood(
        theta_range, theta_true, params.likelihood_kappa, 0
    )
    prior = compute_prior(theta_range, theta_prior, params.prior_kappa)
    _, perceived_decoding = bayesian_inference(
        theta_range, likelihood_unshifted, prior
    )
    decoding_bias = perceived_decoding - theta_true
    
    # === Combined ===
    _, perceived_combined = bayesian_inference(
        theta_range, likelihood_shifted, prior
    )
    combined_bias = perceived_combined - theta_true
    
    # Wrap biases
    for b in [encoding_bias, decoding_bias, combined_bias]:
        if b > 90:
            b -= 180
        elif b < -90:
            b += 180
    
    return {
        'encoding_bias': encoding_bias,   # From STD (repulsion)
        'decoding_bias': decoding_bias,   # From STF (attraction)
        'combined_bias': combined_bias,
        'delta': delta,
    }


def compare_bayesian_with_neural(
    neural_deltas: np.ndarray,
    neural_errors: np.ndarray,
    params: BayesianParams = None,
) -> Dict:
    """Compare neural model results with Bayesian predictions.
    
    Args:
        neural_deltas: Delta values from neural model
        neural_errors: Errors from neural model
        params: Bayesian parameters to optimize
        
    Returns:
        Dictionary with comparison results
    """
    if params is None:
        params = BayesianParams()
    
    # Run Bayesian model
    bayesian_results = run_bayesian_analysis(neural_deltas, params)
    
    # Compute correlation
    corr = np.corrcoef(neural_errors, bayesian_results['biases'])[0, 1]
    
    # Compute MSE
    mse = np.mean((neural_errors - bayesian_results['biases'])**2)
    
    return {
        'neural_deltas': neural_deltas,
        'neural_errors': neural_errors,
        'bayesian_biases': bayesian_results['biases'],
        'correlation': corr,
        'mse': mse,
        'params': params,
    }


def optimize_bayesian_params(
    neural_deltas: np.ndarray,
    neural_errors: np.ndarray,
    n_iter: int = 100,
) -> BayesianParams:
    """Optimize Bayesian parameters to match neural model.
    
    Uses simple grid search to find best parameters.
    
    Args:
        neural_deltas: Delta values
        neural_errors: Errors from neural model
        n_iter: Number of optimization iterations
        
    Returns:
        Optimized parameters
    """
    from scipy.optimize import minimize
    
    def loss(params_vec):
        params = BayesianParams(
            prior_kappa=params_vec[0],
            likelihood_kappa=params_vec[1],
            encoding_shift_gain=params_vec[2],
        )
        result = compare_bayesian_with_neural(neural_deltas, neural_errors, params)
        return result['mse']
    
    # Initial guess
    x0 = [2.0, 10.0, 0.1]
    bounds = [(0.1, 10), (1, 50), (0, 0.5)]
    
    result = minimize(loss, x0, bounds=bounds, method='L-BFGS-B')
    
    return BayesianParams(
        prior_kappa=result.x[0],
        likelihood_kappa=result.x[1],
        encoding_shift_gain=result.x[2],
    )


def plot_bayesian_analysis(
    results: Dict,
    ax=None,
) -> None:
    """Plot Bayesian analysis results.
    
    Args:
        results: Results from run_bayesian_analysis
        ax: Matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(results['deltas'], results['biases'], 'o-', 
            color='purple', linewidth=2, label='Bayesian Model')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Δθ (Prior - Current) (°)')
    ax.set_ylabel('Perceptual Bias (°)')
    ax.set_title('Bayesian Model of Serial Dependence')
    ax.legend()
    
    return ax


def plot_bias_decomposition(
    deltas: np.ndarray,
    params: BayesianParams = None,
    ax=None,
) -> None:
    """Plot decomposition of bias into encoding and decoding components.
    
    Args:
        deltas: Stimulus differences to plot
        params: Bayesian parameters
        ax: Matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if params is None:
        params = BayesianParams()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    encoding_biases = []
    decoding_biases = []
    combined_biases = []
    
    reference = 90.0
    
    for delta in deltas:
        theta_prior = reference + delta
        if theta_prior >= 180:
            theta_prior -= 180
        elif theta_prior < 0:
            theta_prior += 180
        
        decomp = decompose_bias(theta_prior, reference, params)
        encoding_biases.append(decomp['encoding_bias'])
        decoding_biases.append(decomp['decoding_bias'])
        combined_biases.append(decomp['combined_bias'])
    
    ax.plot(deltas, encoding_biases, 'r--', linewidth=2, 
            label='Encoding (Repulsion)')
    ax.plot(deltas, decoding_biases, 'b--', linewidth=2,
            label='Decoding (Attraction)')
    ax.plot(deltas, combined_biases, 'k-', linewidth=2.5,
            label='Combined')
    
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.5)
    ax.set_xlabel('Δθ (Prior - Current) (°)')
    ax.set_ylabel('Bias (°)')
    ax.set_title('Decomposition of Serial Dependence Bias')
    ax.legend()
    
    return ax

