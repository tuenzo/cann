"""
Short-Term Synaptic Plasticity (STP) Module
============================================

Implements STD (Short-Term Depression) and STF (Short-Term Facilitation)
dynamics based on the Tsodyks-Markram model.

Key equations:
    dx/dt = (1-x)/τ_d - u·x·r      (neurotransmitter recovery)
    du/dt = (U-u)/τ_f + U(1-u)r    (release probability)
    
    Effective synaptic strength: J_eff = J · u · x

Reference: Zhang et al., NeurIPS 2025
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from functools import partial


class STPState(NamedTuple):
    """State variables for STP dynamics.
    
    Attributes:
        x: Available neurotransmitter fraction (0-1), shape (N,)
        u: Release probability (0-1), shape (N,)
    """
    x: jnp.ndarray  # 可用神经递质比例
    u: jnp.ndarray  # 释放概率


def create_stp_state(N: int, U: float = 0.3) -> STPState:
    """Initialize STP state variables.
    
    Args:
        N: Number of neurons
        U: Baseline release probability
        
    Returns:
        Initial STP state with x=1 (full recovery) and u=U (baseline)
    """
    return STPState(
        x=jnp.ones(N),      # 完全恢复
        u=jnp.full(N, U),   # 基础释放概率
    )


@partial(jax.jit, static_argnames=['dt'])
def stp_step(
    state: STPState,
    r: jnp.ndarray,
    tau_d: float,
    tau_f: float,
    U: float,
    dt: float,
) -> STPState:
    """Single step update of STP dynamics using Euler method.
    
    Implements the Tsodyks-Markram STP model:
        dx/dt = (1-x)/τ_d - u·x·r
        du/dt = (U-u)/τ_f + U(1-u)r
    
    Args:
        state: Current STP state (x, u)
        r: Presynaptic firing rate, shape (N,)
        tau_d: Depression time constant (seconds)
        tau_f: Facilitation time constant (seconds)
        U: Baseline release probability
        dt: Time step (milliseconds, will be converted to seconds)
        
    Returns:
        Updated STP state
        
    Note:
        - STD-dominated: τ_d >> τ_f (e.g., τ_d=3s, τ_f=0.3s) → repulsion
        - STF-dominated: τ_f >> τ_d (e.g., τ_f=5s, τ_d=0.3s) → attraction
    """
    x, u = state.x, state.u
    dt_s = dt * 0.001  # Convert ms to s
    
    # STD dynamics: neurotransmitter depletion and recovery
    # dx/dt = (1-x)/τ_d - u·x·r
    dx = (1.0 - x) / tau_d - u * x * r
    
    # STF dynamics: release probability facilitation and decay
    # du/dt = (U-u)/τ_f + U(1-u)r
    du = (U - u) / tau_f + U * (1.0 - u) * r
    
    # Euler update
    x_new = x + dx * dt_s
    u_new = u + du * dt_s
    
    # Clip to valid range [0, 1]
    x_new = jnp.clip(x_new, 0.0, 1.0)
    u_new = jnp.clip(u_new, 0.0, 1.0)
    
    return STPState(x=x_new, u=u_new)


@jax.jit
def compute_effective_weight(
    J: jnp.ndarray,
    stp_state: STPState,
) -> jnp.ndarray:
    """Compute effective synaptic weight modulated by STP.
    
    J_eff = J · u · x
    
    Args:
        J: Base synaptic weight matrix, shape (N, N) or (N,)
        stp_state: Current STP state
        
    Returns:
        Effective weight with same shape as J
    """
    # For 1D (per-neuron weights) or 2D (weight matrix)
    if J.ndim == 1:
        return J * stp_state.u * stp_state.x
    else:
        # For weight matrix, modulate by presynaptic neuron's STP
        return J * (stp_state.u * stp_state.x)[None, :]


@partial(jax.jit, static_argnames=['n_steps', 'dt'])
def evolve_stp_only(
    state: STPState,
    r: jnp.ndarray,
    tau_d: float,
    tau_f: float,
    U: float,
    dt: float,
    n_steps: int,
) -> STPState:
    """Evolve STP dynamics for multiple steps with constant firing rate.
    
    Useful for simulating ISI/ITI periods where neural activity is stable.
    
    Args:
        state: Initial STP state
        r: Firing rate (constant during evolution)
        tau_d, tau_f, U: STP parameters
        dt: Time step (ms)
        n_steps: Number of steps
        
    Returns:
        Final STP state
    """
    def step_fn(state, _):
        new_state = stp_step(state, r, tau_d, tau_f, U, dt)
        return new_state, None
    
    final_state, _ = jax.lax.scan(step_fn, state, None, length=n_steps)
    return final_state


def get_stp_time_course(
    state: STPState,
    r: jnp.ndarray,
    tau_d: float,
    tau_f: float,
    U: float,
    dt: float,
    n_steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Record STP dynamics over time for visualization.
    
    Args:
        state: Initial STP state
        r: Firing rate, shape (N,)
        tau_d, tau_f, U: STP parameters
        dt: Time step (ms)
        n_steps: Number of time steps
        
    Returns:
        x_history: shape (n_steps, N)
        u_history: shape (n_steps, N)
    """
    def step_fn(state, _):
        new_state = stp_step(state, r, tau_d, tau_f, U, dt)
        return new_state, (new_state.x, new_state.u)
    
    _, (x_history, u_history) = jax.lax.scan(
        step_fn, state, None, length=n_steps
    )
    return x_history, u_history


# Utility functions for analysis
@jax.jit
def stp_steady_state(
    r: jnp.ndarray,
    tau_d: float,
    tau_f: float,
    U: float,
) -> STPState:
    """Compute analytical steady state of STP for constant firing rate.
    
    At steady state: dx/dt = 0, du/dt = 0
    
    Solving:
        x_ss = 1 / (1 + u_ss * r * τ_d)
        u_ss = U * (1 + r * τ_f) / (1 + U * r * τ_f)
        
    Args:
        r: Constant firing rate
        tau_d, tau_f, U: STP parameters
        
    Returns:
        Steady state STP values
    """
    # First compute steady-state u
    u_ss = U * (1.0 + r * tau_f) / (1.0 + U * r * tau_f)
    
    # Then compute steady-state x
    x_ss = 1.0 / (1.0 + u_ss * r * tau_d)
    
    return STPState(x=x_ss, u=u_ss)


@jax.jit
def stp_efficacy(state: STPState) -> jnp.ndarray:
    """Compute synaptic efficacy u*x.
    
    This is the factor that modulates the effective synaptic strength.
    
    Args:
        state: STP state
        
    Returns:
        Efficacy u*x, shape (N,)
    """
    return state.u * state.x

