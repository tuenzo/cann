"""
Worker module for multiprocessing multi-core CANN experiments
========================================================

每个独立的 worker 进程运行一批 trials。
使用 jax.vmap 实现批量并行（SIMD）。
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Any
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..models.cann import (
    CANNState, CANNParamsNumeric,
    create_stp_state,
)
from ..models.stp import STPState


class TrialConfig:
    """Trial configuration."""
    N: int = 100
    dt: float = 0.1
    s1_duration: float = 200.0
    isi: float = 1000.0
    s2_duration: float = 200.0
    delay: float = 3400.0
    cue_duration: float = 500.0
    alpha_ext: float = 20.0
    a_ext: float = 0.3
    alpha_cue: float = 2.5
    a_cue: float = 0.4


def cann_step_fast(
    state: CANNState,
    I_ext: jnp.ndarray,
    kernel: jnp.ndarray,
    tau: float,
    k: float,
    rho: float,
    dt: float,
    tau_d: float,
    tau_f: float,
    U: float,
) -> CANNState:
    """CANN step with all scalar params for JIT compatibility."""
    u, r, stp = state.u, state.r, state.stp
    
    # STP efficacy: u_stp * x_stp
    efficacy = stp.u * stp.x
    
    # Circular convolution for recurrent input
    N = kernel.shape[0]
    r_eff = r * efficacy
    # Use FFT-based circular convolution
    r_fft = jnp.fft.fft(r_eff)
    k_fft = jnp.fft.fft(kernel)
    recurrent_input = jnp.real(jnp.fft.ifft(r_fft * k_fft))
    
    # Membrane potential dynamics
    du = (-u + rho * recurrent_input + I_ext) / tau
    u_new = u + du * dt
    
    # Firing rate with divisive normalization
    u_pos = jnp.maximum(u_new, 0)
    u_squared = u_pos ** 2
    normalization = 1.0 + k * rho * jnp.sum(u_squared)
    r_new = u_squared / normalization
    
    # STP dynamics
    dt_sec = dt / 1000.0
    
    x_old, u_stp_old = stp.x, stp.u
    
    dx = (1.0 - x_old) / tau_d - u_stp_old * x_old * r
    x_new = x_old + dx * dt_sec
    x_new = jnp.clip(x_new, 0.0, 1.0)
    
    du_stp = (U - u_stp_old) / tau_f + U * (1.0 - u_stp_old) * r
    u_stp_new = u_stp_old + du_stp * dt_sec
    u_stp_new = jnp.clip(u_stp_new, 0.0, 1.0)
    
    stp_new = STPState(x=x_new, u=u_stp_new)
    
    return CANNState(u=u_new, r=r_new, stp=stp_new)


def run_trial_vectorized(
    theta_s1_batch: jnp.ndarray,
    theta_s2_batch: jnp.ndarray,
    kernel: jnp.ndarray,
    params: CANNParamsNumeric,
    trial_config: TrialConfig,
) -> tuple:
    """Run multiple trials in parallel using jax.vmap."""
    N = trial_config.N
    dt = trial_config.dt
    batch_size = theta_s1_batch.shape[0]
    
    # Compute step counts
    n_s1 = int(trial_config.s1_duration / dt)
    n_isi = int(trial_config.isi / dt)
    n_s2 = int(trial_config.s2_duration / dt)
    n_delay = int(trial_config.delay / dt)
    n_cue = int(trial_config.cue_duration / dt)
    
    # Initialize batched states
    initial_states = CANNState(
        u=jnp.zeros((batch_size, N)),
        r=jnp.zeros((batch_size, N)),
        stp=STPState(
            x=jnp.ones((batch_size, N)) * 1.0,
            u=jnp.ones((batch_size, N)) * params.U,
        ),
    )
    
    # Create theta array
    theta = jnp.linspace(-90, 90, N, endpoint=False)
    
    # Create batched stimuli
    def make_stimulus_batch(theta_stim_batch, amplitude, width):
        dx = theta[None, :] - theta_stim_batch[:, None]
        dx = jnp.where(dx > 90, dx - 180, dx)
        dx = jnp.where(dx < -90, dx + 180, dx)
        return amplitude * jnp.exp(-dx**2 / (2 * width**2))
    
    I_s1_batch = make_stimulus_batch(
        theta_s1_batch, trial_config.alpha_ext,
        trial_config.a_ext * 180 / jnp.pi
    )
    I_s2_batch = make_stimulus_batch(
        theta_s2_batch, trial_config.alpha_ext,
        trial_config.a_ext * 180 / jnp.pi
    )
    I_cue = jnp.zeros((batch_size, N)) + trial_config.alpha_cue * jnp.exp(
        -(theta[None, :])**2 / (2 * (trial_config.a_cue * 180 / jnp.pi)**2)
    )
    I_zero = jnp.zeros((batch_size, N))
    
    # Extract scalar parameters
    tau, k, rho = params.tau, params.k, params.rho
    dt_val = params.dt
    tau_d, tau_f, U = params.tau_d, params.tau_f, params.U
    
    # Vectorized phase functions
    def s1_step_batch(states, _):
        new_states = jax.vmap(
            cann_step_fast, in_axes=(0, 0, None, None, None, None, None, None, None, None, None)
        )(states, I_s1_batch, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        return new_states, None
    
    def isi_step_batch(states, _):
        new_states = jax.vmap(
            cann_step_fast, in_axes=(0, 0, None, None, None, None, None, None, None, None, None)
        )(states, I_zero, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        return new_states, None
    
    def s2_step_batch(states, _):
        new_states = jax.vmap(
            cann_step_fast, in_axes=(0, 0, None, None, None, None, None, None, None, None, None)
        )(states, I_s2_batch, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        return new_states, None
    
    def delay_step_batch(states, _):
        new_states = jax.vmap(
            cann_step_fast, in_axes=(0, 0, None, None, None, None, None, None, None, None, None)
        )(states, I_zero, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        return new_states, None
    
    def cue_step_batch(states, _):
        new_states = jax.vmap(
            cann_step_fast, in_axes=(0, 0, None, None, None, None, None, None, None, None, None)
        )(states, I_cue, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        return new_states, None
    
    # Run phases
    state_s1, _ = jax.lax.scan(s1_step_batch, initial_states, None, length=n_s1)
    state_isi, _ = jax.lax.scan(isi_step_batch, state_s1, None, length=n_isi)
    state_s2, _ = jax.lax.scan(s2_step_batch, state_isi, None, length=n_s2)
    state_delay, _ = jax.lax.scan(delay_step_batch, state_s2, None, length=n_delay)
    state_final, _ = jax.lax.scan(cue_step_batch, state_delay, None, length=n_cue)
    
    # Decode
    theta_rad = theta * jnp.pi / 180
    cos_sum = jnp.sum(state_final.r * jnp.cos(2 * theta_rad)[None, :], axis=1)
    sin_sum = jnp.sum(state_final.r * jnp.sin(2 * theta_rad)[None, :], axis=1)
    perceived_rad = jnp.arctan2(sin_sum, cos_sum) / 2
    perceived = perceived_rad * 180 / jnp.pi
    
    # Wrap to [-90, 90)
    perceived = jnp.where(perceived >= 90, perceived - 180, perceived)
    perceived = jnp.where(perceived < -90, perceived + 180, perceived)
    
    # Compute error
    error = perceived - theta_s2_batch
    error = jnp.where(error > 90, error - 180, error)
    error = jnp.where(error < -90, error + 180, error)
    
    return perceived, error


# JIT compile
_run_trial_vectorized_jit = jax.jit(run_trial_vectorized, static_argnums=(3, 4))


def worker_run_batch(args: tuple) -> List[Dict[str, Any]]:
    """
    Worker function for multiprocessing.
    
    Each worker process receives a batch of trials and runs them
    using jax.vmap for SIMD parallelism within the batch.
    
    Args:
        args: (batch_trials, kernel, params, trial_config, batch_size)
        
    Returns:
        List of trial results
    """
    batch_trials, kernel_np, params, trial_config, batch_size = args
    
    # Convert to JAX arrays
    kernel = jnp.array(kernel_np)
    
    # Initialize JAX in this worker process
    # This is important: each worker needs its own JAX backend
    jax.config.update('jax_platforms', 'cpu')
    
    # Process trials in sub-batches
    all_results = []
    
    for i in range(0, len(batch_trials), batch_size):
        end_idx = min(i + batch_size, len(batch_trials))
        sub_batch = batch_trials[i:end_idx]
        
        # Prepare batched inputs
        theta_s1 = jnp.array([t['theta_s1'] for t in sub_batch])
        theta_s2 = jnp.array([t['theta_s2'] for t in sub_batch])
        
        # Run batch with jax.vmap
        perceived_batch, error_batch = _run_trial_vectorized_jit(
            theta_s1, theta_s2, kernel, params, trial_config
        )
        
        # Collect results
        for j, (p, e) in enumerate(zip(perceived_batch, error_batch)):
            all_results.append({
                **sub_batch[j],
                'perceived': float(p),
                'error': float(e),
            })
    
    return all_results


if __name__ == '__main__':
    # For testing worker independently
    print("Worker module loaded successfully")

