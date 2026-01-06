"""
Fast Single-Layer CANN Experiments using JAX vectorization
============================================================

使用 jax.lax.scan 实现向量化时间演化，大幅提升计算速度。
"""

from typing import Optional, Dict, Tuple, NamedTuple
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from ..models.cann import (
    CANNParams, CANNState,
    create_gaussian_kernel, create_stp_state,
    cann_step, create_stimulus,
)
from ..models.stp import STPState
from ..decoding import decode_orientation


class TrialConfig(NamedTuple):
    """Trial configuration for fast simulation."""
    N: int = 100
    dt: float = 0.1
    s1_duration: float = 200.0
    isi: float = 1000.0
    s2_duration: float = 200.0
    delay: float = 3400.0
    cue_duration: float = 500.0
    # Stimulus parameters
    alpha_ext: float = 20.0
    a_ext: float = 0.3  # radians
    # Cue parameters
    alpha_cue: float = 2.5
    a_cue: float = 0.4  # radians


@partial(jax.jit, static_argnums=(2, 3))
def run_phase(state: CANNState, I_ext: jnp.ndarray, n_steps: int, 
              kernel: jnp.ndarray, params: CANNParams) -> CANNState:
    """Run a single phase using jax.lax.fori_loop.
    
    Args:
        state: Initial CANN state
        I_ext: External input (constant for the phase)
        n_steps: Number of time steps
        kernel: Connection kernel
        params: CANN parameters
        
    Returns:
        Final state after n_steps
    """
    def body_fn(i, state):
        return cann_step(state, I_ext, kernel, params)
    
    return jax.lax.fori_loop(0, n_steps, body_fn, state)


class CANNParamsNumeric(NamedTuple):
    """Numeric-only CANN parameters for JIT compilation."""
    N: int = 100
    J0: float = 0.5
    a: float = 30.0
    tau: float = 1.0
    k: float = 0.5
    rho: float = 1.0
    dt: float = 0.1
    tau_d: float = 3.0
    tau_f: float = 0.3
    U: float = 0.3


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


def run_trial_fast(
    theta_s1: float,
    theta_s2: float,
    kernel: jnp.ndarray,
    params: CANNParamsNumeric,
    trial_config: TrialConfig,
) -> Tuple[float, float]:
    """Run a single trial using JIT-compiled phases.
    
    This is much faster than the Python-loop version because
    each phase is compiled as a single XLA operation.
    
    Args:
        theta_s1: First stimulus orientation (degrees)
        theta_s2: Second stimulus orientation (degrees)
        kernel: Connection kernel
        params: Numeric CANN parameters
        trial_config: Trial timing configuration
        
    Returns:
        (perceived_angle, error)
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
        stp=create_stp_state(N, params.U),
    )
    
    # Create theta array
    theta = jnp.linspace(-90, 90, N, endpoint=False)
    
    # Create stimuli
    def make_stimulus(theta_stim, amplitude, width):
        dx = theta - theta_stim
        dx = jnp.where(dx > 90, dx - 180, dx)
        dx = jnp.where(dx < -90, dx + 180, dx)
        return amplitude * jnp.exp(-dx**2 / (2 * width**2))
    
    I_s1 = make_stimulus(theta_s1, trial_config.alpha_ext, 
                         trial_config.a_ext * 180 / jnp.pi)
    I_s2 = make_stimulus(theta_s2, trial_config.alpha_ext,
                         trial_config.a_ext * 180 / jnp.pi)
    I_cue = make_stimulus(0.0, trial_config.alpha_cue,
                          trial_config.a_cue * 180 / jnp.pi)  # Cue at reference
    I_zero = jnp.zeros(N)
    
    # Extract scalar parameters
    tau, k, rho = params.tau, params.k, params.rho
    dt_val = params.dt
    tau_d, tau_f, U = params.tau_d, params.tau_f, params.U
    
    # Phase 1: S1 presentation
    def s1_step(state, _):
        return cann_step_fast(state, I_s1, kernel, tau, k, rho, dt_val, tau_d, tau_f, U), None
    state, _ = jax.lax.scan(s1_step, state, None, length=n_s1)
    
    # Phase 2: ISI
    def isi_step(state, _):
        return cann_step_fast(state, I_zero, kernel, tau, k, rho, dt_val, tau_d, tau_f, U), None
    state, _ = jax.lax.scan(isi_step, state, None, length=n_isi)
    
    # Phase 3: S2 presentation
    def s2_step(state, _):
        return cann_step_fast(state, I_s2, kernel, tau, k, rho, dt_val, tau_d, tau_f, U), None
    state, _ = jax.lax.scan(s2_step, state, None, length=n_s2)
    
    # Phase 4: Delay
    def delay_step(state, _):
        return cann_step_fast(state, I_zero, kernel, tau, k, rho, dt_val, tau_d, tau_f, U), None
    state, _ = jax.lax.scan(delay_step, state, None, length=n_delay)
    
    # Phase 5: Cue - collect activity for decoding
    def cue_step(state, _):
        new_state = cann_step_fast(state, I_cue, kernel, tau, k, rho, dt_val, tau_d, tau_f, U)
        return new_state, new_state.r
    state, cue_activity = jax.lax.scan(cue_step, state, None, length=n_cue)
    
    # Decode: average activity during cue period
    mean_activity = jnp.mean(cue_activity, axis=0)
    
    # Population vector decode
    theta_rad = theta * jnp.pi / 180
    cos_sum = jnp.sum(mean_activity * jnp.cos(2 * theta_rad))
    sin_sum = jnp.sum(mean_activity * jnp.sin(2 * theta_rad))
    perceived_rad = jnp.arctan2(sin_sum, cos_sum) / 2
    perceived = perceived_rad * 180 / jnp.pi
    
    # Wrap to [-90, 90)
    perceived = jnp.where(perceived >= 90, perceived - 180, perceived)
    perceived = jnp.where(perceived < -90, perceived + 180, perceived)
    
    # Compute error
    error = perceived - theta_s2
    error = jnp.where(error > 90, error - 180, error)
    error = jnp.where(error < -90, error + 180, error)
    
    return perceived, error


# JIT compile the trial function
_run_trial_jit = jax.jit(run_trial_fast, static_argnums=(4,))


def run_fast_experiment(
    stp_type: str = 'std',
    n_runs: int = 20,
    n_trials_per_run: int = 100,
    delta_range: Tuple[float, float] = (-90.0, 90.0),
    delta_step: float = 1.0,
    isi: float = 1000.0,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """Run fast single-layer experiment using JIT compilation.
    
    Args:
        stp_type: 'std' for STD-dominated, 'stf' for STF-dominated
        n_runs: Number of simulation runs
        n_trials_per_run: Trials per run
        delta_range: (min, max) delta values in degrees
        delta_step: Step size for delta
        isi: Inter-stimulus interval in ms
        seed: Random seed
        verbose: Show progress bar
        
    Returns:
        Dictionary with experiment results
    """
    import time
    start_time = time.time()
    
    # Set parameters based on STP type (Table 1)
    if stp_type == 'std':
        params = CANNParamsNumeric(
            N=100, J0=0.13, a=0.5, tau=10.0, k=0.0018, rho=1.0, dt=0.1,
            tau_d=3.0, tau_f=0.3, U=0.5,
        )
    else:  # stf
        params = CANNParamsNumeric(
            N=100, J0=0.06, a=0.4, tau=10.0, k=0.005, rho=1.0, dt=0.1,
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
    _ = _run_trial_jit(0.0, 0.0, kernel, params, trial_config)
    
    if verbose:
        print(f"开始运行 {n_runs} runs × {n_trials_per_run} trials...")
    
    # Run experiment
    all_trials = []
    np.random.seed(seed)
    
    iterator = range(n_runs)
    if verbose:
        iterator = tqdm(iterator, desc=f'{stp_type.upper()} Runs')
    
    for run_id in iterator:
        for trial_id in range(n_trials_per_run):
            # Random delta
            delta = np.random.choice(deltas)
            
            # Stimulus orientations
            theta_s2 = 0.0  # Reference
            theta_s1 = theta_s2 + delta
            
            # Wrap
            if theta_s1 >= 90:
                theta_s1 -= 180
            elif theta_s1 < -90:
                theta_s1 += 180
            
            # Run trial
            perceived, error = _run_trial_jit(
                theta_s1, theta_s2, kernel, params, trial_config
            )
            
            all_trials.append({
                'run_id': run_id,
                'trial_id': trial_id,
                'theta_s1': float(theta_s1),
                'theta_s2': float(theta_s2),
                'delta': float(delta),
                'perceived': float(perceived),
                'error': float(error),
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


def run_trial_with_recording(
    theta_s1: float,
    theta_s2: float,
    kernel: jnp.ndarray,
    params: CANNParamsNumeric,
    trial_config: TrialConfig,
) -> Dict:
    """Run a single trial and record neural activity for visualization.
    
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
        stp=create_stp_state(N, params.U),
    )
    
    # Create theta array
    theta = jnp.linspace(-90, 90, N, endpoint=False)
    
    # Create stimuli
    def make_stimulus(theta_stim, amplitude, width):
        dx = theta - theta_stim
        dx = jnp.where(dx > 90, dx - 180, dx)
        dx = jnp.where(dx < -90, dx + 180, dx)
        return amplitude * jnp.exp(-dx**2 / (2 * width**2))
    
    I_s1 = make_stimulus(theta_s1, trial_config.alpha_ext, 
                         trial_config.a_ext * 180 / jnp.pi)
    I_s2 = make_stimulus(theta_s2, trial_config.alpha_ext,
                         trial_config.a_ext * 180 / jnp.pi)
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
        if i % 10 == 0:  # Record every 10 steps
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
    
    Args:
        stp_type: 'std' for STD-dominated, 'stf' for STF-dominated
        delta_to_record: Delta value for the recorded trial
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
            N=100, J0=0.06, a=0.4, tau=10.0, k=0.005, rho=1.0, dt=0.1,
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

