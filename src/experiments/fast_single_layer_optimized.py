"""
Fast Single-Layer CANN Experiments (JAX Optimized Version)
==========================================================

统一的单层CANN实验模块，包含：
- jax.vmap 批量并行加速（用于大规模实验）
- 单次试验记录功能（用于可视化）

优化策略：
- jax.vmap: 向量化 trials 层（批量并行，利用 CPU SIMD）
- jax.lax.scan: 向量化时间演化

预期速度提升: 200-300x（相比原始 Python 循环版本）

注意：此模块合并了原 fast_single_layer.py 的功能。
"""

from typing import Optional, Dict, Tuple, NamedTuple, List
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import time

from ..models.cann import (
    CANNState,
    create_gaussian_kernel, create_stp_state,
)
from ..models.stp import STPState
from ..decoding import decode_orientation


class TrialConfig(NamedTuple):
    """Trial configuration for fast simulation.
    
    All parameters from Table 1 and paper requirements.
    """
    N: int = 100
    dt: float = 0.1
    s1_duration: float = 200.0
    isi: float = 1000.0
    s2_duration: float = 200.0
    delay: float = 3400.0
    cue_duration: float = 500.0
    # Stimulus parameters (External Input, Table 1)
    alpha_sti: float = 20.0        # Strength of external stimulus
    a_sti: float = 0.3               # Spatial scale of external stimulus (radians)
    mu_sti: float = 0.5               # Noise strength of external stimulus
    # Cue parameters (External Input, Table 1)
    alpha_cue: float = 2.5        # Strength of external cue
    a_cue: float = 0.4               # Spatial scale of external cue (radians)
    mu_cue: float = 1.0               # Noise strength of external cue


class CANNParamsNumeric(NamedTuple):
    """Numeric-only CANN parameters for JIT compilation.
    
    Note: These are default values (STD-dominated).
    For STF-dominated, these will be overridden in run_fast_experiment_optimized.
    """
    N: int = 100
    J0: float = 0.13                  # STD: 0.13, STF: 0.09
    a: float = 0.5                     # STD: 0.5, STF: 0.15 (radians)
    tau: float = 10.0                  # Time constant (ms), 0.01s = 10ms
    k: float = 0.0018                  # STD: 0.0018, STF: 0.0095
    rho: float = 1.0
    dt: float = 0.1                     # Time step (ms)
    tau_d: float = 3.0                  # STD: 3.0, STF: 0.3 (s)
    tau_f: float = 0.3                  # STD: 0.3, STF: 5.0 (s)
    U: float = 0.5                      # STD: 0.5, STF: 0.2


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
    
    # STP dynamics (Tsodyks-Markram model)
    # dx/dt = (1-x)/τ_d - u*x*r
    # du/dt = (U-u)/τ_f + U(1-u)*r
    dt_sec = dt / 1000.0  # Convert ms to seconds
    
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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run multiple trials in parallel using jax.vmap.
    
    Args:
        theta_s1_batch: Array of first stimulus orientations [batch_size]
        theta_s2_batch: Array of second stimulus orientations [batch_size]
        kernel: Connection kernel
        params: Numeric CANN parameters
        trial_config: Trial timing configuration
        
    Returns:
        (perceived_angles, errors) both of shape [batch_size]
    """
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
            x=jnp.ones((batch_size, N)) * 1.0,  # Initial x = 1
            u=jnp.ones((batch_size, N)) * params.U,  # Initial u = U
        ),
    )
    
    # Create theta array
    theta = jnp.linspace(-90, 90, N, endpoint=False)
    
    # Create batched stimuli function
    def make_stimulus_batch(theta_stim_batch, amplitude, width):
        # theta_stim_batch: [batch_size]
        # theta: [N]
        # Output: [batch_size, N]
        dx = theta[None, :] - theta_stim_batch[:, None]
        dx = jnp.where(dx > 90, dx - 180, dx)
        dx = jnp.where(dx < -90, dx + 180, dx)
        return amplitude * jnp.exp(-dx**2 / (2 * width**2))
    
    I_s1_batch = make_stimulus_batch(
        theta_s1_batch, trial_config.alpha_sti,
        trial_config.a_sti * 180 / jnp.pi
    )
    I_s2_batch = make_stimulus_batch(
        theta_s2_batch, trial_config.alpha_sti,
        trial_config.a_sti * 180 / jnp.pi
    )
    I_cue = jnp.zeros((batch_size, N)) + trial_config.alpha_cue * jnp.exp(
        -(theta[None, :])**2 / (2 * (trial_config.a_cue * 180 / jnp.pi)**2)
    )
    I_zero = jnp.zeros((batch_size, N))
    
    # Extract scalar parameters
    tau, k, rho = params.tau, params.k, params.rho
    dt_val = params.dt
    tau_d, tau_f, U = params.tau_d, params.tau_f, params.U
    
    # Vectorized phase functions using jax.vmap
    def s1_step_batch(states, _):
        # states: CANNState with batched arrays [batch_size, N]
        new_states = jax.vmap(
            cann_step_fast, in_axes=(0, 0, None, None, None, None, None, None, None, None)
        )(states, I_s1_batch, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        return new_states, None
    
    def isi_step_batch(states, _):
        new_states = jax.vmap(
            cann_step_fast, in_axes=(0, 0, None, None, None, None, None, None, None, None)
        )(states, I_zero, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        return new_states, None
    
    def s2_step_batch(states, _):
        new_states = jax.vmap(
            cann_step_fast, in_axes=(0, 0, None, None, None, None, None, None, None, None)
        )(states, I_s2_batch, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        return new_states, None
    
    def delay_step_batch(states, _):
        new_states = jax.vmap(
            cann_step_fast, in_axes=(0, 0, None, None, None, None, None, None, None, None)
        )(states, I_zero, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        return new_states, None
    
    def cue_step_batch(states, _):
        new_states = jax.vmap(
            cann_step_fast, in_axes=(0, 0, None, None, None, None, None, None, None, None)
        )(states, I_cue, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        return new_states, None
    
    # Run phases
    state_s1, _ = jax.lax.scan(s1_step_batch, initial_states, None, length=n_s1)
    state_isi, _ = jax.lax.scan(isi_step_batch, state_s1, None, length=n_isi)
    state_s2, _ = jax.lax.scan(s2_step_batch, state_isi, None, length=n_s2)
    state_delay, _ = jax.lax.scan(delay_step_batch, state_s2, None, length=n_delay)
    state_final, _ = jax.lax.scan(cue_step_batch, state_delay, None, length=n_cue)
    
    # Decode: population vector method
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


# JIT compile the vectorized trial function
_run_trial_vectorized_jit = jax.jit(run_trial_vectorized, static_argnums=(3, 4))


def run_fast_experiment_optimized(
    stp_type: str = 'std',
    n_runs: int = 20,
    n_trials_per_run: int = 100,
    delta_range: Tuple[float, float] = (-90.0, 90.0),
    delta_step: float = 1.0,
    isi: float = 1000.0,
    seed: int = 42,
    verbose: bool = True,
    batch_size: int = 10,
) -> Dict:
    """Run fast single-layer experiment with jax.vmap optimization.
    
    Optimization strategy:
    - jax.vmap: Vectorize trials within a batch (SIMD parallelism)
    - Process trials in batches to balance memory usage and speed
    
    Args:
        stp_type: 'std' for STD-dominated, 'stf' for STF-dominated
        n_runs: Number of simulation runs
        n_trials_per_run: Trials per run
        delta_range: (min, max) delta values in degrees
        delta_step: Step size for delta
        isi: Inter-stimulus interval in ms
        seed: Random seed
        verbose: Show progress bar
        batch_size: Number of trials per batch for jax.vmap (tunable)
        
    Returns:
        Dictionary with experiment results
    """
    start_time = time.time()
    
    # Set parameters based on STP type (Table 1)
    if stp_type == 'std':
        # STD-dominated (Fig. 2A-C): Repulsion effect
        params = CANNParamsNumeric(
            N=100, J0=0.13, a=0.5, tau=10.0, k=0.0018, rho=1.0, dt=0.1,
            tau_d=3.0, tau_f=0.3, U=0.5,
        )
    else:  # stf
        # STF-dominated (Fig. 2D-F): Attraction effect
        params = CANNParamsNumeric(
            N=100, J0=0.09, a=0.15, tau=10.0, k=0.0095, rho=1.0, dt=0.1,
            tau_d=0.3, tau_f=5.0, U=0.2,
        )
    
    trial_config = TrialConfig(
        N=params.N, dt=params.dt,
        isi=isi,
    )
    
    # Create kernel (centered theta range)
    kernel = create_gaussian_kernel(
        params.N, params.a, params.J0, 'centered'
    )
    
    # Generate delta values
    deltas = np.arange(delta_range[0], delta_range[1] + delta_step, delta_step)
    
    # Warm up JIT (first call compiles)
    if verbose:
        print(f"JIT 编译中...")
    _ = _run_trial_vectorized_jit(
        jnp.array([0.0]), jnp.array([0.0]), kernel, params, trial_config
    )
    
    if verbose:
        print(f"使用 jax.vmap 批量并行 (batch_size={batch_size})")
        print(f"开始运行 {n_runs} runs × {n_trials_per_run} trials...")
    
    # Prepare all trials
    np.random.seed(seed)
    total_trials = n_runs * n_trials_per_run
    
    all_theta_s1 = []
    all_theta_s2 = []
    all_delta = []
    all_run_ids = []
    all_trial_ids = []
    
    for run_id in range(n_runs):
        for trial_id in range(n_trials_per_run):
            delta = np.random.choice(deltas)
            theta_s2 = 0.0  # Reference
            theta_s1 = theta_s2 + delta
            
            # Wrap
            if theta_s1 >= 90:
                theta_s1 -= 180
            elif theta_s1 < -90:
                theta_s1 += 180
            
            all_theta_s1.append(theta_s1)
            all_theta_s2.append(theta_s2)
            all_delta.append(delta)
            all_run_ids.append(run_id)
            all_trial_ids.append(trial_id)
    
    # Convert to arrays
    all_theta_s1 = np.array(all_theta_s1)
    all_theta_s2 = np.array(all_theta_s2)
    
    # Process in batches
    all_trials = []
    
    iterator = range(0, total_trials, batch_size)
    if verbose:
        iterator = tqdm(iterator, desc='Processing batches')
    
    for i in iterator:
        end_idx = min(i + batch_size, total_trials)
        
        theta_s1_batch = jnp.array(all_theta_s1[i:end_idx])
        theta_s2_batch = jnp.array(all_theta_s2[i:end_idx])
        
        # Run batch
        perceived_batch, error_batch = _run_trial_vectorized_jit(
            theta_s1_batch, theta_s2_batch, kernel, params, trial_config
        )
        
        # Collect results
        for j in range(len(perceived_batch)):
            idx = i + j
            all_trials.append({
                'run_id': all_run_ids[idx],
                'trial_id': all_trial_ids[idx],
                'theta_s1': float(all_theta_s1[idx]),
                'theta_s2': float(all_theta_s2[idx]),
                'delta': float(all_delta[idx]),
                'perceived': float(perceived_batch[j]),
                'error': float(error_batch[j]),
            })
    
    elapsed = time.time() - start_time
    
    # Convert to arrays
    trials_df = {k: np.array([t[k] for t in all_trials]) for k in all_trials[0].keys()}
    
    # Compute binned statistics
    unique_deltas = np.unique(trials_df['delta'])
    mean_errors = []
    se_errors = []
    
    for d in unique_deltas:
        mask = np.abs(trials_df['delta'] - d) < 0.5
        run_means = []
        for run_id in range(n_runs):
            run_mask = mask & (trials_df['run_id'] == run_id)
            if np.any(run_mask):
                run_means.append(np.mean(trials_df['error'][run_mask]))
        
        run_means = np.array(run_means)
        mean_errors.append(np.mean(run_means))
        se_errors.append(np.std(run_means) / np.sqrt(len(run_means)) if len(run_means) > 0 else 0)
    
    curve_binned = {
        'delta': np.array(unique_deltas),
        'mean_error': np.array(mean_errors),
        'se_error': np.array(se_errors),
    }
    
    # DoG fitting
    from ..analysis.dog_fitting import fit_dog
    dog_params = fit_dog(curve_binned['delta'], curve_binned['mean_error'])
    
    return {
        'stp_type': stp_type,
        'n_runs': n_runs,
        'n_trials_per_run': n_trials_per_run,
        'elapsed_time': elapsed,
        'trials_df': trials_df,
        'curve_binned': curve_binned,
        'dog_fit': {
            'amplitude': dog_params.amplitude,
            'sigma': dog_params.sigma,
            'r_squared': dog_params.r_squared,
        },
    }


# =============================================================================
# Single Trial with Recording (for visualization)
# =============================================================================

def run_trial_with_recording(
    theta_s1: float,
    theta_s2: float,
    kernel: jnp.ndarray,
    params: CANNParamsNumeric,
    trial_config: TrialConfig,
) -> Dict:
    """Run a single trial and record neural activity for visualization.
    
    This function is used to generate Fig.2A/D (activity heatmaps) and
    Fig.2B/E (STP dynamics).
    
    Args:
        theta_s1: First stimulus orientation (degrees)
        theta_s2: Second stimulus orientation (degrees)
        kernel: Connection kernel
        params: Numeric CANN parameters
        trial_config: Trial timing configuration
        
    Returns:
        Dictionary with timeseries and results
    """
    N = trial_config.N
    dt = trial_config.dt
    
    # Compute step counts
    n_s1 = int(trial_config.s1_duration / dt)
    n_isi = int(trial_config.isi / dt)
    n_s2 = int(trial_config.s2_duration / dt)
    n_delay = int(trial_config.delay / dt)
    n_cue = int(trial_config.cue_duration / dt)
    
    # Initialize state
    state = CANNState(
        u=jnp.zeros(N),
        r=jnp.zeros(N),
        stp=STPState(
            x=jnp.ones(N) * 1.0,
            u=jnp.ones(N) * params.U,
        ),
    )
    
    # Create theta array
    theta = jnp.linspace(-90, 90, N, endpoint=False)
    
    # Create stimuli
    def make_stimulus(theta_stim, amplitude, width):
        dx = theta - theta_stim
        dx = jnp.where(dx > 90, dx - 180, dx)
        dx = jnp.where(dx < -90, dx + 180, dx)
        return amplitude * jnp.exp(-dx**2 / (2 * width**2))
    
    I_s1 = make_stimulus(theta_s1, trial_config.alpha_sti,
                         trial_config.a_sti * 180 / jnp.pi)
    I_s2 = make_stimulus(theta_s2, trial_config.alpha_sti,
                         trial_config.a_sti * 180 / jnp.pi)
    I_cue = make_stimulus(0.0, trial_config.alpha_cue,
                          trial_config.a_cue * 180 / jnp.pi)
    I_zero = jnp.zeros(N)
    
    # Extract scalar parameters
    tau, k, rho = params.tau, params.k, params.rho
    dt_val = params.dt
    tau_d, tau_f, U = params.tau_d, params.tau_f, params.U
    
    # Recording lists
    all_time = []
    all_r = []
    all_stp_x = []
    all_stp_u = []
    
    t = 0.0
    
    # Phase 1: S1 presentation
    for _ in range(n_s1):
        state = cann_step_fast(state, I_s1, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        t += dt
        all_time.append(t)
        all_r.append(np.array(state.r))
        all_stp_x.append(np.array(state.stp.x))
        all_stp_u.append(np.array(state.stp.u))
    
    # Phase 2: ISI
    for i in range(n_isi):
        state = cann_step_fast(state, I_zero, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        t += dt
        all_time.append(t)
        all_r.append(np.array(state.r))
        all_stp_x.append(np.array(state.stp.x))
        all_stp_u.append(np.array(state.stp.u))
    
    # Phase 3: S2 presentation
    for _ in range(n_s2):
        state = cann_step_fast(state, I_s2, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        t += dt
        all_time.append(t)
        all_r.append(np.array(state.r))
        all_stp_x.append(np.array(state.stp.x))
        all_stp_u.append(np.array(state.stp.u))
    
    # Phase 4: Delay
    for i in range(n_delay):
        state = cann_step_fast(state, I_zero, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        t += dt
        all_time.append(t)
        all_r.append(np.array(state.r))
        all_stp_x.append(np.array(state.stp.x))
        all_stp_u.append(np.array(state.stp.u))
    
    # Phase 5: Cue - collect activity for decoding
    cue_activity = []
    for _ in range(n_cue):
        state = cann_step_fast(state, I_cue, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        t += dt
        cue_activity.append(np.array(state.r))
        all_time.append(t)
        all_r.append(np.array(state.r))
        all_stp_x.append(np.array(state.stp.x))
        all_stp_u.append(np.array(state.stp.u))
    
    # Decode: average activity during cue period
    mean_activity = np.mean(cue_activity, axis=0)
    
    # Population vector decode
    theta_np = np.array(theta)
    theta_rad = theta_np * np.pi / 180
    cos_sum = np.sum(mean_activity * np.cos(2 * theta_rad))
    sin_sum = np.sum(mean_activity * np.sin(2 * theta_rad))
    perceived_rad = np.arctan2(sin_sum, cos_sum) / 2
    perceived = perceived_rad * 180 / np.pi
    
    # Wrap to [-90, 90)
    if perceived >= 90:
        perceived -= 180
    elif perceived < -90:
        perceived += 180
    
    # Compute error
    error = perceived - theta_s2
    if error > 90:
        error -= 180
    elif error < -90:
        error += 180
    
    # Find neuron index closest to S1
    s1_neuron = int(np.argmin(np.abs(theta_np - theta_s1)))
    s2_neuron = int(np.argmin(np.abs(theta_np - theta_s2)))
    
    return {
        'timeseries': {
            'time': np.array(all_time),
            'r': np.array(all_r),
            'stp_x': np.array(all_stp_x),
            'stp_u': np.array(all_stp_u),
        },
        'theta': theta_np,
        'theta_s1': theta_s1,
        'theta_s2': theta_s2,
        's1_neuron': s1_neuron,
        's2_neuron': s2_neuron,
        'perceived': perceived,
        'error': error,
        'delta': theta_s1 - theta_s2,
    }


def run_fast_experiment_with_recording(
    stp_type: str = 'std',
    delta_to_record: float = -30.0,
    isi: float = 1000.0,
) -> Dict:
    """Run a single trial with recording for visualization.
    
    This is the main entry point for generating visualization data
    (Fig.2A/D activity heatmaps, Fig.2B/E STP dynamics).
    
    Args:
        stp_type: 'std' for STD-dominated, 'stf' for STF-dominated
        delta_to_record: Delta value for the recorded trial (degrees)
        isi: Inter-stimulus interval in ms
        
    Returns:
        Dictionary with timeseries and results
    """
    # Set parameters based on STP type (Table 1)
    if stp_type == 'std':
        params = CANNParamsNumeric(
            N=100, J0=0.13, a=0.5, tau=10.0, k=0.0018, rho=1.0, dt=0.1,
            tau_d=3.0, tau_f=0.3, U=0.5,
        )
    else:  # stf
        params = CANNParamsNumeric(
            N=100, J0=0.09, a=0.15, tau=10.0, k=0.0095, rho=1.0, dt=0.1,
            tau_d=0.3, tau_f=5.0, U=0.2,
        )
    
    trial_config = TrialConfig(
        N=params.N, dt=params.dt,
        isi=isi,
    )
    
    # Create kernel
    kernel = create_gaussian_kernel(
        params.N, params.a, params.J0, 'centered'
    )
    
    # Stimulus orientations
    theta_s2 = 0.0  # Reference
    theta_s1 = theta_s2 + delta_to_record
    
    # Wrap
    if theta_s1 >= 90:
        theta_s1 -= 180
    elif theta_s1 < -90:
        theta_s1 += 180
    
    return run_trial_with_recording(
        theta_s1, theta_s2, kernel, params, trial_config
    )
