"""
Population Decoding Methods
===========================

Implements four decoding methods for reading out orientation from
neural population activity:

1. Population Vector Method (PVM) - weighted circular mean
2. Center of Mass (COM) - equivalent to PVM for symmetric tuning
3. Maximum Likelihood (ML) - Poisson likelihood maximization
4. Peak Decoding - preferred orientation of most active neuron

Reference: Zhang et al., NeurIPS 2025 (Appendix F)
"""

from typing import Literal, Optional
from functools import partial
import jax
import jax.numpy as jnp


@jax.jit
def population_vector_decode(
    r: jnp.ndarray,
    theta: jnp.ndarray,
) -> float:
    """Population Vector Method (PVM) decoding.
    
    Computes the weighted circular mean of population activity:
    
    θ_hat = arg(Σ r_i * exp(2j * θ_i)) / 2
    
    Note: We use 2θ because orientations are in [0, 180°] and the
    circular mean needs to handle the 0°≡180° wraparound.
    
    Args:
        r: Firing rates, shape (N,)
        theta: Preferred orientations in degrees, shape (N,)
        
    Returns:
        Decoded orientation in degrees [0, 180)
    """
    # Convert to radians and double for circular statistics on [0, 180]
    theta_rad = theta * jnp.pi / 180.0 * 2  # [0, 2π]
    
    # Weighted sum in complex plane
    z = jnp.sum(r * jnp.exp(1j * theta_rad))
    
    # Extract angle and convert back
    decoded_rad = jnp.angle(z) / 2  # Back to [−π/2, π/2]
    decoded_deg = decoded_rad * 180.0 / jnp.pi
    
    # Map to [0, 180)
    decoded_deg = jnp.where(decoded_deg < 0, decoded_deg + 180, decoded_deg)
    
    return decoded_deg


@jax.jit
def center_of_mass_decode(
    r: jnp.ndarray,
    theta: jnp.ndarray,
) -> float:
    """Center of Mass (COM) decoding.
    
    θ_hat = arg(Σ r_i * exp(2j * θ_i)) / 2
    
    For symmetric, densely distributed tuning curves, COM and PVM
    yield identical estimates.
    
    Args:
        r: Firing rates, shape (N,)
        theta: Preferred orientations in degrees, shape (N,)
        
    Returns:
        Decoded orientation in degrees [0, 180)
    """
    # Identical to PVM for our model
    return population_vector_decode(r, theta)


def create_tuning_curves(
    theta: jnp.ndarray,
    width: float = 30.0,
    amplitude: float = 1.0,
) -> jnp.ndarray:
    """Create Gaussian tuning curve templates.
    
    Args:
        theta: Preferred orientations, shape (N,)
        width: Tuning width in degrees
        amplitude: Maximum response amplitude
        
    Returns:
        Tuning curve matrix, shape (N, N_stimulus)
        where entry [i, j] is response of neuron i to stimulus j
    """
    N = theta.shape[0]
    
    # Stimulus values (same as neuron preferences for our grid)
    stim = theta[:, None]  # (N, 1)
    pref = theta[None, :]  # (1, N)
    
    # Circular distance
    dx = stim - pref
    dx = jnp.where(dx > 90, dx - 180, dx)
    dx = jnp.where(dx < -90, dx + 180, dx)
    
    # Gaussian tuning curves
    tuning = amplitude * jnp.exp(-dx**2 / (2 * width**2))
    
    return tuning


@partial(jax.jit, static_argnames=['width'])
def maximum_likelihood_decode(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    width: float = 30.0,
) -> float:
    """Maximum Likelihood (ML) decoding.
    
    Assumes Poisson firing with Gaussian tuning curves.
    Finds θ that maximizes the log-likelihood:
    
    θ_ML = argmax_θ Σ [r_i * log(λ_i(θ)) - λ_i(θ)]
    
    where λ_i(θ) is the expected firing rate of neuron i for stimulus θ.
    
    Args:
        r: Observed firing rates, shape (N,)
        theta: Preferred orientations in degrees, shape (N,)
        width: Tuning curve width in degrees
        
    Returns:
        Maximum likelihood estimate in degrees [0, 180)
    """
    N = theta.shape[0]
    
    # Test all possible stimulus values (same resolution as neurons)
    # For each test stimulus, compute log-likelihood
    def log_likelihood(test_theta):
        # Expected firing rates under this stimulus hypothesis
        dx = theta - test_theta
        dx = jnp.where(dx > 90, dx - 180, dx)
        dx = jnp.where(dx < -90, dx + 180, dx)
        
        lambda_expected = jnp.exp(-dx**2 / (2 * width**2))
        lambda_expected = jnp.maximum(lambda_expected, 1e-10)  # Avoid log(0)
        
        # Poisson log-likelihood (ignoring constant terms)
        ll = jnp.sum(r * jnp.log(lambda_expected) - lambda_expected)
        return ll
    
    # Evaluate at all candidate orientations
    ll_values = jax.vmap(log_likelihood)(theta)
    
    # Return orientation with maximum likelihood
    max_idx = jnp.argmax(ll_values)
    return theta[max_idx]


@jax.jit
def peak_decode(
    r: jnp.ndarray,
    theta: jnp.ndarray,
) -> float:
    """Peak decoding - simplest heuristic.
    
    Returns the preferred orientation of the most active neuron:
    
    θ_Peak = θ_{argmax_i r_i}
    
    Args:
        r: Firing rates, shape (N,)
        theta: Preferred orientations in degrees, shape (N,)
        
    Returns:
        Decoded orientation in degrees [0, 180)
    """
    max_idx = jnp.argmax(r)
    return theta[max_idx]


def decode_orientation(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    method: Literal['pvm', 'com', 'ml', 'peak'] = 'pvm',
    **kwargs,
) -> float:
    """Unified interface for orientation decoding.
    
    Args:
        r: Firing rates, shape (N,)
        theta: Preferred orientations in degrees, shape (N,)
        method: Decoding method to use
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Decoded orientation in degrees [0, 180)
        
    Example:
        >>> theta = jnp.linspace(0, 180, 180, endpoint=False)
        >>> r = jnp.exp(-(theta - 45)**2 / (2 * 30**2))  # Peak at 45°
        >>> decode_orientation(r, theta, method='pvm')
        45.0
    """
    methods = {
        'pvm': population_vector_decode,
        'com': center_of_mass_decode,
        'ml': maximum_likelihood_decode,
        'peak': peak_decode,
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Use one of {list(methods.keys())}")
    
    decoder = methods[method]
    
    if method == 'ml':
        width = kwargs.get('width', 30.0)
        return decoder(r, theta, width)
    else:
        return decoder(r, theta)


# Batch decoding for multiple trials
@jax.jit
def batch_pvm_decode(
    r_batch: jnp.ndarray,
    theta: jnp.ndarray,
) -> jnp.ndarray:
    """Batch population vector decoding.
    
    Args:
        r_batch: Firing rates, shape (n_trials, N)
        theta: Preferred orientations, shape (N,)
        
    Returns:
        Decoded orientations, shape (n_trials,)
    """
    return jax.vmap(population_vector_decode, in_axes=(0, None))(r_batch, theta)


@jax.jit
def compute_adjustment_error(
    perceived: float,
    actual: float,
) -> float:
    """Compute adjustment error (perceived - actual) with wraparound.
    
    Positive error = bias toward prior stimulus (attraction)
    Negative error = bias away from prior stimulus (repulsion)
    
    Args:
        perceived: Decoded orientation (degrees)
        actual: True stimulus orientation (degrees)
        
    Returns:
        Error in degrees, wrapped to [-90, 90]
    """
    error = perceived - actual
    
    # Wrap to [-90, 90]
    error = jnp.where(error > 90, error - 180, error)
    error = jnp.where(error < -90, error + 180, error)
    
    return error


@jax.jit
def compute_relative_error(
    perceived: float,
    actual: float,
    prior: float,
) -> float:
    """Compute signed adjustment error relative to prior stimulus.
    
    Sign convention:
    - Positive: Bias toward prior (attraction)
    - Negative: Bias away from prior (repulsion)
    
    Args:
        perceived: Decoded orientation
        actual: True stimulus orientation
        prior: Prior stimulus orientation (S1)
        
    Returns:
        Signed error relative to prior direction
    """
    raw_error = compute_adjustment_error(perceived, actual)
    
    # Determine direction of prior relative to actual
    prior_direction = prior - actual
    prior_direction = jnp.where(prior_direction > 90, prior_direction - 180, prior_direction)
    prior_direction = jnp.where(prior_direction < -90, prior_direction + 180, prior_direction)
    prior_sign = jnp.sign(prior_direction)
    
    # Positive error means bias in direction of prior
    # Need to flip sign if error is away from prior
    error_sign = jnp.sign(raw_error)
    same_direction = error_sign == prior_sign
    
    # Return signed error (positive = toward prior)
    return jnp.where(same_direction, jnp.abs(raw_error), -jnp.abs(raw_error))


# Utility for comparing decoders
def compare_decoders(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    width: float = 30.0,
) -> dict:
    """Compare all four decoding methods.
    
    Args:
        r: Firing rates, shape (N,)
        theta: Preferred orientations, shape (N,)
        width: Tuning width for ML decoder
        
    Returns:
        Dictionary with decoded values for each method
    """
    return {
        'pvm': float(population_vector_decode(r, theta)),
        'com': float(center_of_mass_decode(r, theta)),
        'ml': float(maximum_likelihood_decode(r, theta, width)),
        'peak': float(peak_decode(r, theta)),
    }

