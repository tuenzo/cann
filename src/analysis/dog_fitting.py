"""
Derivative of Gaussian (DoG) Curve Fitting
==========================================

Fits the serial dependence adjustment error curves using a 
Derivative of Gaussian (DoG) function, which captures the
characteristic S-shaped bias pattern.

DoG function:
    f(x) = A * x * exp(-x² / (2σ²))

where:
    A = amplitude (positive for attraction, negative for repulsion)
    σ = width parameter (determines peak location)

Reference: Zhang et al., NeurIPS 2025
"""

from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp
from scipy.optimize import curve_fit
import numpy as np


class DoGParams(NamedTuple):
    """Parameters of fitted DoG curve.
    
    Attributes:
        amplitude: Peak amplitude (positive = attraction, negative = repulsion)
        sigma: Width parameter
        peak_location: Location of maximum/minimum (degrees)
        r_squared: Goodness of fit
    """
    amplitude: float
    sigma: float
    peak_location: float
    r_squared: float


def dog_function(
    x: np.ndarray,
    amplitude: float,
    sigma: float,
) -> np.ndarray:
    """Derivative of Gaussian function.
    
    f(x) = A * x * exp(-x² / (2σ²))
    
    Args:
        x: Input values (stimulus differences in degrees)
        amplitude: Peak amplitude
        sigma: Width parameter
        
    Returns:
        DoG function values
    """
    return amplitude * x * np.exp(-x**2 / (2 * sigma**2))


def dog_function_jax(
    x: jnp.ndarray,
    amplitude: float,
    sigma: float,
) -> jnp.ndarray:
    """JAX version of DoG function for JIT compilation."""
    return amplitude * x * jnp.exp(-x**2 / (2 * sigma**2))


def fit_dog(
    delta: np.ndarray,
    errors: np.ndarray,
    initial_guess: Optional[tuple] = None,
) -> DoGParams:
    """Fit DoG function to adjustment error data.
    
    Args:
        delta: Stimulus differences (S1 - S2) in degrees
        errors: Adjustment errors in degrees
        initial_guess: Optional (amplitude, sigma) starting values
        
    Returns:
        Fitted DoG parameters
    """
    # Convert to numpy if needed
    delta = np.asarray(delta)
    errors = np.asarray(errors)
    
    # Initial guess
    if initial_guess is None:
        # Estimate from data
        max_abs_error = np.max(np.abs(errors))
        # Peak of DoG is at x = sigma, so estimate from where max error occurs
        peak_idx = np.argmax(np.abs(errors))
        sigma_init = np.abs(delta[peak_idx]) if delta[peak_idx] != 0 else 30.0
        amplitude_init = max_abs_error / (sigma_init * np.exp(-0.5))
        # Determine sign based on correlation
        if np.mean(errors * delta) > 0:
            amplitude_init = np.abs(amplitude_init)  # Attraction
        else:
            amplitude_init = -np.abs(amplitude_init)  # Repulsion
        initial_guess = (amplitude_init, sigma_init)
    
    try:
        # Fit with bounds to ensure reasonable values
        popt, _ = curve_fit(
            dog_function,
            delta,
            errors,
            p0=initial_guess,
            bounds=([-10, 5], [10, 90]),  # amplitude, sigma bounds
            maxfev=5000,
        )
        amplitude, sigma = popt
        
        # Compute R-squared
        predicted = dog_function(delta, amplitude, sigma)
        ss_res = np.sum((errors - predicted)**2)
        ss_tot = np.sum((errors - np.mean(errors))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        # Peak location is at x = ±sigma (depending on sign of amplitude)
        peak_location = sigma if amplitude > 0 else -sigma
        
    except RuntimeError:
        # Fitting failed, return default values
        amplitude = 0.0
        sigma = 30.0
        peak_location = 0.0
        r_squared = 0.0
    
    return DoGParams(
        amplitude=amplitude,
        sigma=sigma,
        peak_location=np.abs(sigma),
        r_squared=r_squared,
    )


def compute_serial_bias(
    delta: np.ndarray,
    errors: np.ndarray,
) -> dict:
    """Compute comprehensive serial bias statistics.
    
    Args:
        delta: Stimulus differences (degrees)
        errors: Adjustment errors (degrees)
        
    Returns:
        Dictionary with bias statistics
    """
    # Fit DoG
    dog_params = fit_dog(delta, errors)
    
    # Compute mean bias at different ranges
    mask_small = np.abs(delta) <= 30
    mask_medium = (np.abs(delta) > 30) & (np.abs(delta) <= 60)
    mask_large = np.abs(delta) > 60
    
    mean_bias_small = np.mean(errors[mask_small]) if np.any(mask_small) else 0.0
    mean_bias_medium = np.mean(errors[mask_medium]) if np.any(mask_medium) else 0.0
    mean_bias_large = np.mean(errors[mask_large]) if np.any(mask_large) else 0.0
    
    # Effect type based on DoG amplitude
    if dog_params.amplitude > 0.1:
        effect_type = "attraction"
    elif dog_params.amplitude < -0.1:
        effect_type = "repulsion"
    else:
        effect_type = "neutral"
    
    return {
        'dog_params': dog_params,
        'effect_type': effect_type,
        'mean_bias': np.mean(errors),
        'std_bias': np.std(errors),
        'max_bias': np.max(errors),
        'min_bias': np.min(errors),
        'bias_small_delta': mean_bias_small,
        'bias_medium_delta': mean_bias_medium,
        'bias_large_delta': mean_bias_large,
    }


def generate_dog_curve(
    delta_range: tuple = (-90, 90),
    n_points: int = 181,
    amplitude: float = 1.0,
    sigma: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate smooth DoG curve for plotting.
    
    Args:
        delta_range: (min, max) of delta values
        n_points: Number of points
        amplitude: DoG amplitude
        sigma: DoG width
        
    Returns:
        delta: x values
        dog: y values
    """
    delta = np.linspace(delta_range[0], delta_range[1], n_points)
    dog = dog_function(delta, amplitude, sigma)
    return delta, dog


# Statistical analysis functions
def bootstrap_dog_fit(
    delta: np.ndarray,
    errors: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> dict:
    """Bootstrap confidence intervals for DoG parameters.
    
    Args:
        delta: Stimulus differences
        errors: Adjustment errors
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        
    Returns:
        Dictionary with parameter estimates and CIs
    """
    n = len(delta)
    amplitudes = []
    sigmas = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        delta_boot = delta[idx]
        errors_boot = errors[idx]
        
        try:
            params = fit_dog(delta_boot, errors_boot)
            amplitudes.append(params.amplitude)
            sigmas.append(params.sigma)
        except:
            continue
    
    amplitudes = np.array(amplitudes)
    sigmas = np.array(sigmas)
    
    alpha = (1 - confidence) / 2
    
    return {
        'amplitude_mean': np.mean(amplitudes),
        'amplitude_ci': (np.percentile(amplitudes, alpha * 100),
                        np.percentile(amplitudes, (1 - alpha) * 100)),
        'sigma_mean': np.mean(sigmas),
        'sigma_ci': (np.percentile(sigmas, alpha * 100),
                    np.percentile(sigmas, (1 - alpha) * 100)),
    }


def t_test_bias(
    errors: np.ndarray,
    null_hypothesis: float = 0.0,
) -> dict:
    """One-sample t-test for bias significance.
    
    Args:
        errors: Adjustment errors
        null_hypothesis: Expected mean under null
        
    Returns:
        Dictionary with t-statistic and p-value
    """
    from scipy import stats
    
    t_stat, p_value = stats.ttest_1samp(errors, null_hypothesis)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'mean': np.mean(errors),
        'std': np.std(errors),
        'n': len(errors),
        'significant': p_value < 0.05,
    }


# =============================================================================
# STP Analytical Fitting (for Fig.2B/E per Appendix C)
# =============================================================================

def analytical_stp_x(
    theta: np.ndarray,
    theta_s1: float,
    theta_s2: float,
    A1: float,
    A2: float,
    a: float,
) -> np.ndarray:
    """Analytical form for x(θ) from Appendix C Eq.(15).
    
    x(θ,t) = 1 - A1_x(t) * exp(-(θ-θ_s1)²/2a²) - A2_x(t) * exp(-(θ-θ_s2)²/2a²)
    
    Args:
        theta: Orientation array (degrees)
        theta_s1: First stimulus position (degrees)
        theta_s2: Second stimulus position (degrees)
        A1: Amplitude for S1 depletion
        A2: Amplitude for S2 depletion
        a: Spatial width (degrees)
    
    Returns:
        x(θ) analytical prediction
    """
    # Handle circular distance
    dx1 = theta - theta_s1
    dx1 = np.where(dx1 > 90, dx1 - 180, dx1)
    dx1 = np.where(dx1 < -90, dx1 + 180, dx1)
    
    dx2 = theta - theta_s2
    dx2 = np.where(dx2 > 90, dx2 - 180, dx2)
    dx2 = np.where(dx2 < -90, dx2 + 180, dx2)
    
    return 1.0 - A1 * np.exp(-dx1**2 / (2 * a**2)) - A2 * np.exp(-dx2**2 / (2 * a**2))


def analytical_stp_u(
    theta: np.ndarray,
    theta_s1: float,
    theta_s2: float,
    A1: float,
    A2: float,
    a: float,
    U: float = 0.2,
) -> np.ndarray:
    """Analytical form for u(θ) from Appendix C Eq.(17).
    
    u(θ,t) = U + A1_u(t) * exp(-(θ-θ_s1)²/2a²) + A2_u(t) * exp(-(θ-θ_s2)²/2a²)
    
    Args:
        theta: Orientation array (degrees)
        theta_s1: First stimulus position (degrees)
        theta_s2: Second stimulus position (degrees)
        A1: Amplitude for S1 facilitation
        A2: Amplitude for S2 facilitation
        a: Spatial width (degrees)
        U: Baseline release probability
    
    Returns:
        u(θ) analytical prediction
    """
    # Handle circular distance
    dx1 = theta - theta_s1
    dx1 = np.where(dx1 > 90, dx1 - 180, dx1)
    dx1 = np.where(dx1 < -90, dx1 + 180, dx1)
    
    dx2 = theta - theta_s2
    dx2 = np.where(dx2 > 90, dx2 - 180, dx2)
    dx2 = np.where(dx2 < -90, dx2 + 180, dx2)
    
    return U + A1 * np.exp(-dx1**2 / (2 * a**2)) + A2 * np.exp(-dx2**2 / (2 * a**2))


def fit_analytical_stp(
    theta: np.ndarray,
    stp_data: np.ndarray,
    theta_s1: float,
    theta_s2: float,
    stp_type: str = 'std',
    U: float = 0.5,
) -> dict:
    """Fit analytical STP model to numerical data.
    
    Fits the analytical STP spatial distribution from Appendix C:
    - STD (Eq.15): x(θ) = 1 - A1*exp(-Δθ₁²/2a²) - A2*exp(-Δθ₂²/2a²)
    - STF (Eq.17): u(θ) = U + A1*exp(-Δθ₁²/2a²) + A2*exp(-Δθ₂²/2a²)
    
    Args:
        theta: Orientation array (degrees)
        stp_data: STP variable data (x for STD, u for STF)
        theta_s1: First stimulus position (degrees)
        theta_s2: Second stimulus position (degrees)
        stp_type: 'std' or 'stf'
        U: Baseline release probability (for STF)
    
    Returns:
        Dictionary with fitted parameters and curve:
        - A1: Amplitude for S1
        - A2: Amplitude for S2
        - a: Spatial width (degrees)
        - fitted_curve: Fitted STP distribution
        - r_squared: Goodness of fit
    """
    if stp_type == 'std':
        # Fit x(θ) = 1 - A1*exp(...) - A2*exp(...)
        def model(theta_arr, A1, A2, a):
            return analytical_stp_x(theta_arr, theta_s1, theta_s2, A1, A2, a)
        
        # Initial guess: A1 > A2 (S1 has more depletion due to longer time)
        p0 = [0.3, 0.2, 20.0]
        bounds = ([0, 0, 5], [1, 1, 60])
    else:
        # Fit u(θ) = U + A1*exp(...) + A2*exp(...)
        def model(theta_arr, A1, A2, a):
            return analytical_stp_u(theta_arr, theta_s1, theta_s2, A1, A2, a, U)
        
        # Initial guess
        p0 = [0.3, 0.2, 20.0]
        bounds = ([0, 0, 5], [1, 1, 60])
    
    try:
        popt, _ = curve_fit(model, theta, stp_data, p0=p0, bounds=bounds, maxfev=5000)
        fitted_curve = model(theta, *popt)
        
        # Compute R²
        ss_res = np.sum((stp_data - fitted_curve) ** 2)
        ss_tot = np.sum((stp_data - np.mean(stp_data)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            'A1': popt[0],
            'A2': popt[1],
            'a': popt[2],
            'fitted_curve': fitted_curve,
            'r_squared': r_squared,
        }
    except Exception as e:
        print(f"Warning: STP fitting failed: {e}")
        return {
            'A1': 0,
            'A2': 0,
            'a': 20,
            'fitted_curve': stp_data,
            'r_squared': 0,
        }

