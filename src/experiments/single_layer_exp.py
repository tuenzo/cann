"""
Single-Layer CANN Experiments (Fig.2 Reproduction)
===================================================

Reproduces Figure 2 from Zhang et al., NeurIPS 2025:
- Fig 2A-C: STD-dominated CANN → Repulsion effect
- Fig 2D-F: STF-dominated CANN → Attraction effect

Experiment protocol (Post-cueing paradigm):
1. Present S1 at θ₁ for 200ms
2. Wait for ISI (inter-stimulus interval, default 1000ms)
3. Present S2 at θ₂ = θ₁ + Δθ for 200ms
4. Delay period (3400ms, no stimulus)
5. Present Cue to recall S2 for 500ms
6. ITI (inter-trial interval, 1000ms)
7. Decode perceived S2 during cue period
8. Compute adjustment error

Reference: Zhang et al., NeurIPS 2025, Appendix B (p.22)
"""

from typing import Optional, Tuple, Dict, List, NamedTuple
from dataclasses import dataclass, field
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from ..models.cann import CANNParams, SingleLayerCANN, create_stimulus
from ..decoding import decode_orientation


# =============================================================================
# Trial Timeline Structure (Checklist #12)
# =============================================================================

class TrialTimeline(NamedTuple):
    """Trial timeline structure matching paper specification.
    
    Reference: Appendix B "Post-cueing paradigm" (p.22)
    """
    s1_duration: float = 200.0       # S1 presentation (ms)
    isi: float = 1000.0              # Inter-stimulus interval (ms)
    s2_duration: float = 200.0       # S2 presentation (ms)
    delay: float = 3400.0            # Delay period (ms)
    cue_duration: float = 500.0      # Cue period (ms)
    iti: float = 1000.0              # Inter-trial interval (ms)
    
    def get_phase_times(self) -> Dict[str, Tuple[float, float]]:
        """Get start and end times for each phase."""
        t = 0.0
        phases = {}
        
        phases['S1'] = (t, t + self.s1_duration)
        t += self.s1_duration
        
        phases['ISI'] = (t, t + self.isi)
        t += self.isi
        
        phases['S2'] = (t, t + self.s2_duration)
        t += self.s2_duration
        
        phases['Delay'] = (t, t + self.delay)
        t += self.delay
        
        phases['Cue'] = (t, t + self.cue_duration)
        t += self.cue_duration
        
        phases['ITI'] = (t, t + self.iti)
        t += self.iti
        
        phases['total'] = (0.0, t)
        return phases
    
    def total_duration(self) -> float:
        """Total trial duration in ms."""
        return (self.s1_duration + self.isi + self.s2_duration + 
                self.delay + self.cue_duration + self.iti)


# =============================================================================
# Parameter Configurations (Checklist #15, #16, #17)
# =============================================================================

@dataclass
class STDConfig:
    """STD-dominated single-layer CANN parameters (Table 1, p.23).
    
    For reproducing Fig.2A-C (repulsion effect).
    """
    # Network parameters
    N: int = 100                      # Number of neurons (#8)
    J0: float = 0.13                  # Connection strength (#15)
    a: float = 0.5                    # Gaussian tuning width (radians) (#15)
    tau: float = 10.0                 # Time constant (ms), 0.01s = 10ms (#17)
    k: float = 0.0018                 # Divisive normalization strength (#15)
    rho: float = 1.0                  # Neural density
    dt: float = 0.1                   # Time step (ms)
    
    # STP parameters (#15)
    tau_d: float = 3.0                # STD time constant (s)
    tau_f: float = 0.3                # STF time constant (s)
    U: float = 0.5                    # Baseline release probability
    
    # Noise parameters (#3, #4)
    mu_J: float = 0.01                # Connection noise strength
    mu_b: float = 0.5                 # Background noise strength
    
    # Stimulus parameters (#11)
    alpha_sti: float = 20.0           # Stimulus amplitude
    a_sti: float = 0.3                # Stimulus width (radians)
    mu_sti: float = 0.5               # Stimulus noise
    
    # Cue parameters (#10, #11)
    alpha_cue: float = 2.5            # Cue amplitude (weaker than stimulus)
    a_cue: float = 0.4                # Cue width (wider than stimulus)
    mu_cue: float = 1.0               # Cue noise (stronger than stimulus)


@dataclass
class STFConfig:
    """STF-dominated single-layer CANN parameters (Table 1, p.23).
    
    For reproducing Fig.2D-F (attraction effect).
    """
    # Network parameters
    N: int = 100                      # Number of neurons (#8)
    J0: float = 0.09                  # Connection strength (#16)
    a: float = 0.15                   # Gaussian tuning width (radians) (#16)
    tau: float = 10.0                 # Time constant (ms), 0.01s = 10ms (#17)
    k: float = 0.0095                 # Divisive normalization strength (#16)
    rho: float = 1.0                  # Neural density
    dt: float = 0.1                   # Time step (ms)
    
    # STP parameters (#16)
    tau_d: float = 0.3                # STD time constant (s)
    tau_f: float = 5.0                # STF time constant (s)
    U: float = 0.2                    # Baseline release probability
    
    # Noise parameters (#3, #4)
    mu_J: float = 0.01                # Connection noise strength
    mu_b: float = 0.5                 # Background noise strength
    
    # Stimulus parameters (#11)
    alpha_sti: float = 20.0           # Stimulus amplitude
    a_sti: float = 0.3                # Stimulus width (radians)
    mu_sti: float = 0.5               # Stimulus noise
    
    # Cue parameters (#10, #11)
    alpha_cue: float = 2.5            # Cue amplitude
    a_cue: float = 0.4                # Cue width
    mu_cue: float = 1.0               # Cue noise


@dataclass
class SingleLayerExperimentConfig:
    """Complete configuration for single-layer experiments.
    
    Combines network parameters, timeline, and experiment settings.
    """
    # Network type
    stp_type: str = 'std'             # 'std' or 'stf'
    
    # Timeline (#12)
    timeline: TrialTimeline = field(default_factory=TrialTimeline)
    
    # Experiment parameters (#18, #22)
    reference_theta: float = 0.0      # S2 position (degrees), θ_s2=0° (#13)
    delta_range: Tuple[float, float] = (-90.0, 90.0)  # ΔS range
    delta_step: float = 1.0           # Step size = 1° (#18)
    n_runs: int = 20                  # Number of simulation runs (#22)
    n_trials_per_run: int = 100       # Trials per run (#22)
    
    # Decoding (#21)
    decode_method: str = 'pvm'
    
    # Random seed for reproducibility
    seed: int = 42
    
    def get_network_config(self):
        """Get appropriate network config based on stp_type."""
        if self.stp_type == 'std':
            return STDConfig()
        elif self.stp_type == 'stf':
            return STFConfig()
        else:
            raise ValueError(f"Unknown stp_type: {self.stp_type}")
    
    def to_cann_params(self) -> CANNParams:
        """Convert to CANNParams for model initialization."""
        net_config = self.get_network_config()
        return CANNParams(
            N=net_config.N,
            J0=net_config.J0,
            a=net_config.a * 180 / np.pi,  # Convert radians to degrees
            tau=net_config.tau,
            k=net_config.k,
            rho=net_config.rho,
            A=net_config.alpha_sti,
            stim_width=net_config.a_sti * 180 / np.pi,
            dt=net_config.dt,
            tau_d=net_config.tau_d,
            tau_f=net_config.tau_f,
            U=net_config.U,
        )


# =============================================================================
# Stimulus Creation with Noise (Checklist #9, #10, #11)
# =============================================================================

def create_stimulus_with_noise(
    theta_stim: float,
    N: int,
    amplitude: float,
    width: float,
    noise_strength: float,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Create Gaussian stimulus with additive noise.
    
    I_ext = α * exp(-(θ - θ_stim)² / (2 * a²)) + μ * ξ
    
    Reference: Sec.3 (p.4), Eq. for I_ext
    
    Args:
        theta_stim: Stimulus orientation in degrees
        N: Number of neurons
        amplitude: α_ext (stimulus/cue amplitude)
        width: a_ext (stimulus/cue width in degrees)
        noise_strength: μ_ext (noise strength)
        key: JAX random key
        
    Returns:
        External input array of shape (N,)
    """
    # Neural positions in degrees (-90 to 90)
    theta = jnp.linspace(-90, 90, N, endpoint=False)
    
    # Circular distance
    dx = theta - theta_stim
    dx = jnp.where(dx > 90, dx - 180, dx)
    dx = jnp.where(dx < -90, dx + 180, dx)
    
    # Gaussian signal
    signal = amplitude * jnp.exp(-dx**2 / (2 * width**2))
    
    # Additive noise
    noise = noise_strength * jax.random.normal(key, shape=(N,))
    
    return signal + noise


def create_background_noise(
    N: int,
    mu_b: float,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Create background noise for neural dynamics.
    
    Reference: Appendix A Eq.(5) - μ_b * ξ_b term
    
    Args:
        N: Number of neurons
        mu_b: Background noise strength
        key: JAX random key
        
    Returns:
        Background noise array of shape (N,)
    """
    return mu_b * jax.random.normal(key, shape=(N,))


# =============================================================================
# Connection Noise (Checklist #3)
# =============================================================================

def create_noisy_kernel(
    kernel: jnp.ndarray,
    mu_J: float,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    """Add participant-level noise to connection kernel.
    
    J̃ = J * (1 + μ_J * ξ_J)
    
    Reference: Appendix A "Model Details" (p.22), Table 1 "μJ" (p.23)
    
    Args:
        kernel: Base connection kernel
        mu_J: Connection noise strength (default 0.01)
        key: JAX random key
        
    Returns:
        Noisy kernel
    """
    xi_J = jax.random.normal(key, shape=kernel.shape)
    return kernel * (1.0 + mu_J * xi_J)


# =============================================================================
# Single Trial Execution (Checklist #12, #21)
# =============================================================================

def run_single_trial(
    model: SingleLayerCANN,
    theta_s1: float,
    theta_s2: float,
    config: SingleLayerExperimentConfig,
    key: jax.random.PRNGKey,
    record: bool = False,
) -> Dict:
    """Run a single serial dependence trial with full timeline.
    
    Timeline: S1 → ISI → S2 → Delay → Cue → ITI
    
    Reference: Appendix B "Post-cueing paradigm" (p.22)
    
    Args:
        model: CANN model
        theta_s1: S1 orientation (degrees)
        theta_s2: S2 orientation (degrees)
        config: Experiment configuration
        key: JAX random key
        record: Whether to record time course
        
    Returns:
        Dictionary with trial results
    """
    model.reset()
    net_config = config.get_network_config()
    timeline = config.timeline
    N = net_config.N
    dt = net_config.dt
    
    result = {
        'theta_s1': theta_s1,
        'theta_s2': theta_s2,
    }
    
    # Split random keys
    keys = jax.random.split(key, 10)
    key_idx = 0
    
    # Recording storage
    if record:
        all_time = []
        all_r = []
        all_stp_x = []
        all_stp_u = []
        all_decoded = []
    
    # Phase 1: S1 presentation (200ms)
    I_s1 = create_stimulus_with_noise(
        theta_s1, N, net_config.alpha_sti, 
        net_config.a_sti * 180 / np.pi,  # Convert to degrees
        net_config.mu_sti, keys[key_idx]
    )
    key_idx += 1
    
    n_steps_s1 = int(timeline.s1_duration / dt)
    for _ in range(n_steps_s1):
        # Add background noise
        bg_noise = create_background_noise(N, net_config.mu_b, keys[key_idx])
        keys = jax.random.split(keys[key_idx], 2)
        key_idx = 0
        
        model.step(I_s1 + bg_noise)
        
        if record:
            all_time.append(model.time)
            all_r.append(model.state.r.copy())
            all_stp_x.append(model.state.stp.x.copy())
            all_stp_u.append(model.state.stp.u.copy())
            all_decoded.append(model.decode(config.decode_method))
    
    # Phase 2: ISI (inter-stimulus interval, default 1000ms)
    n_steps_isi = int(timeline.isi / dt)
    for i in range(n_steps_isi):
        bg_noise = create_background_noise(N, net_config.mu_b, keys[key_idx])
        keys = jax.random.split(keys[key_idx], 2)
        key_idx = 0
        
        model.step(bg_noise)  # Only background noise
        
        if record and i % 10 == 0:  # Record every 10 steps for efficiency
            all_time.append(model.time)
            all_r.append(model.state.r.copy())
            all_stp_x.append(model.state.stp.x.copy())
            all_stp_u.append(model.state.stp.u.copy())
            all_decoded.append(model.decode(config.decode_method))
    
    # Phase 3: S2 presentation (200ms)
    I_s2 = create_stimulus_with_noise(
        theta_s2, N, net_config.alpha_sti,
        net_config.a_sti * 180 / np.pi,
        net_config.mu_sti, keys[key_idx]
    )
    key_idx += 1
    
    n_steps_s2 = int(timeline.s2_duration / dt)
    for _ in range(n_steps_s2):
        bg_noise = create_background_noise(N, net_config.mu_b, keys[key_idx])
        keys = jax.random.split(keys[key_idx], 2)
        key_idx = 0
        
        model.step(I_s2 + bg_noise)
        
        if record:
            all_time.append(model.time)
            all_r.append(model.state.r.copy())
            all_stp_x.append(model.state.stp.x.copy())
            all_stp_u.append(model.state.stp.u.copy())
            all_decoded.append(model.decode(config.decode_method))
    
    # Phase 4: Delay period (3400ms)
    n_steps_delay = int(timeline.delay / dt)
    for i in range(n_steps_delay):
        bg_noise = create_background_noise(N, net_config.mu_b, keys[key_idx])
        keys = jax.random.split(keys[key_idx], 2)
        key_idx = 0
        
        model.step(bg_noise)
        
        if record and i % 100 == 0:  # Sparse recording during delay
            all_time.append(model.time)
            all_r.append(model.state.r.copy())
            all_stp_x.append(model.state.stp.x.copy())
            all_stp_u.append(model.state.stp.u.copy())
            all_decoded.append(model.decode(config.decode_method))
    
    # Phase 5: Cue period (500ms) - recall S2
    # Cue is weaker and noisier than stimulus (#10, #11)
    I_cue = create_stimulus_with_noise(
        theta_s2, N, net_config.alpha_cue,  # Weaker amplitude
        net_config.a_cue * 180 / np.pi,     # Wider width
        net_config.mu_cue, keys[key_idx]    # More noise
    )
    key_idx += 1
    
    # Storage for cue-period activity (for averaging, #21)
    cue_activity = []
    
    n_steps_cue = int(timeline.cue_duration / dt)
    for _ in range(n_steps_cue):
        bg_noise = create_background_noise(N, net_config.mu_b, keys[key_idx])
        keys = jax.random.split(keys[key_idx], 2)
        key_idx = 0
        
        model.step(I_cue + bg_noise)
        cue_activity.append(model.state.r.copy())
        
        if record:
            all_time.append(model.time)
            all_r.append(model.state.r.copy())
            all_stp_x.append(model.state.stp.x.copy())
            all_stp_u.append(model.state.stp.u.copy())
            all_decoded.append(model.decode(config.decode_method))
    
    # Decode using cue-period averaged activity (#21)
    cue_activity_mean = jnp.mean(jnp.array(cue_activity), axis=0)
    theta_arr = jnp.linspace(-90, 90, N, endpoint=False)
    perceived = decode_orientation(cue_activity_mean, theta_arr, config.decode_method)
    result['perceived'] = float(perceived)
    
    # Phase 6: ITI (1000ms) - not used for decoding
    # Skip simulation for efficiency unless recording
    if record:
        n_steps_iti = int(timeline.iti / dt)
        for i in range(n_steps_iti):
            if i % 100 == 0:
                model.step(jnp.zeros(N))
                all_time.append(model.time)
                all_r.append(model.state.r.copy())
                all_stp_x.append(model.state.stp.x.copy())
                all_stp_u.append(model.state.stp.u.copy())
                all_decoded.append(model.decode(config.decode_method))
    
    # Compute adjustment error (#20)
    error = result['perceived'] - theta_s2
    # Wrap to [-90, 90]
    if error > 90:
        error -= 180
    elif error < -90:
        error += 180
    result['error'] = error
    
    # Delta (S1 - S2) (#19)
    delta = theta_s1 - theta_s2
    if delta > 90:
        delta -= 180
    elif delta < -90:
        delta += 180
    result['delta'] = delta
    
    # Add recording data
    if record:
        result['timeseries'] = {
            'time': np.array(all_time),
            'r': np.array(all_r),
            'stp_x': np.array(all_stp_x),
            'stp_u': np.array(all_stp_u),
            'decoded': np.array(all_decoded),
        }
        result['cue_activity_mean'] = np.array(cue_activity_mean)
    
    return result


# =============================================================================
# Full Experiment with Multiple Runs (Checklist #22)
# =============================================================================

def run_single_layer_experiment(
    config: Optional[SingleLayerExperimentConfig] = None,
    stp_type: str = 'std',
    verbose: bool = True,
) -> Dict:
    """Run complete single-layer serial dependence experiment.
    
    Runs 20 simulation runs × 100 trials each for statistical validity.
    
    Reference: Sec.3.1 (p.5), Appendix B (p.22)
    
    Args:
        config: Experiment configuration
        stp_type: 'std' for STD-dominated (repulsion) or 'stf' for STF-dominated (attraction)
        verbose: Show progress bar
        
    Returns:
        Dictionary with all experimental results including:
        - trials_df: All trial-level results
        - curve_binned: Binned mean errors with SE
        - dog_fit: DoG fitting results
    """
    if config is None:
        config = SingleLayerExperimentConfig()
    config.stp_type = stp_type
    
    # Initialize random key
    key = jax.random.PRNGKey(config.seed)
    
    # Get network config
    net_config = config.get_network_config()
    
    # Generate delta values (#18: step = 1°)
    deltas = np.arange(
        config.delta_range[0],
        config.delta_range[1] + config.delta_step,
        config.delta_step
    )
    
    # Storage for all trials
    all_trials = []
    
    # Run 20 simulation runs (#22)
    for run_id in tqdm(range(config.n_runs), desc=f'{stp_type.upper()} Runs', disable=not verbose):
        # Create new model with run-specific connection noise (#3)
        key, subkey = jax.random.split(key)
        params = config.to_cann_params()
        model = SingleLayerCANN(params)
        
        # Add connection noise to kernel
        model.kernel = create_noisy_kernel(model.kernel, net_config.mu_J, subkey)
        
        # Run 100 trials per run (#22)
        for trial_id in range(config.n_trials_per_run):
            # Sample random delta
            delta = np.random.choice(deltas)
            
            # Compute stimulus orientations
            theta_s2 = config.reference_theta
            theta_s1 = theta_s2 + delta
            
            # Wrap to [-90, 90)
            if theta_s1 >= 90:
                theta_s1 -= 180
            elif theta_s1 < -90:
                theta_s1 += 180
            
            # Run trial
            key, subkey = jax.random.split(key)
            result = run_single_trial(model, theta_s1, theta_s2, config, subkey, record=False)
            
            # Store results
            all_trials.append({
                'run_id': run_id,
                'trial_id': trial_id,
                'theta_s1': theta_s1,
                'theta_s2': theta_s2,
                'delta': result['delta'],
                'perceived': result['perceived'],
                'error': result['error'],
                'isi': config.timeline.isi,
            })
    
    # Convert to arrays for analysis
    trials_df = {k: np.array([t[k] for t in all_trials]) for k in all_trials[0].keys()}
    
    # Compute binned statistics with standard error across runs
    unique_deltas = np.unique(trials_df['delta'])
    mean_errors = []
    se_errors = []
    
    for d in unique_deltas:
        # Get errors for this delta across all runs
        mask = np.abs(trials_df['delta'] - d) < 0.5
        errors_at_delta = trials_df['error'][mask]
        
        # Compute per-run means
        run_means = []
        for run_id in range(config.n_runs):
            run_mask = mask & (trials_df['run_id'] == run_id)
            if np.any(run_mask):
                run_means.append(np.mean(trials_df['error'][run_mask]))
        
        run_means = np.array(run_means)
        mean_errors.append(np.mean(run_means))
        se_errors.append(np.std(run_means) / np.sqrt(len(run_means)))  # Standard error
    
    curve_binned = {
        'delta': np.array(unique_deltas),
        'mean_error': np.array(mean_errors),
        'se_error': np.array(se_errors),
    }
    
    # DoG fitting (#23)
    from ..analysis.dog_fitting import fit_dog, compute_serial_bias
    dog_params = fit_dog(curve_binned['delta'], curve_binned['mean_error'])
    bias_stats = compute_serial_bias(curve_binned['delta'], curve_binned['mean_error'])
    
    return {
        'config': config,
        'stp_type': stp_type,
        'timeline': config.timeline.get_phase_times(),
        'trials_df': trials_df,
        'curve_binned': curve_binned,
        'dog_fit': {
            'amplitude': dog_params.amplitude,
            'sigma': dog_params.sigma,
            'peak_location': dog_params.peak_location,
            'r_squared': dog_params.r_squared,
        },
        'bias_stats': bias_stats,
        'theta': np.linspace(-90, 90, net_config.N, endpoint=False),
    }


# =============================================================================
# Experiment with Recording (Checklist #24, #25)
# =============================================================================

def run_experiment_with_recording(
    config: Optional[SingleLayerExperimentConfig] = None,
    stp_type: str = 'std',
    delta_to_record: float = -30.0,  # θ_s1 = -30°, θ_s2 = 0° (#13)
) -> Dict:
    """Run experiment with neural activity recording for visualization.
    
    Used for generating Fig.2A/D (activity heatmaps) and Fig.2B/E (bias curves).
    
    Reference: Fig.2 caption (p.4-5), Example: θ_s1=-30°, θ_s2=0°
    
    Args:
        config: Experiment configuration
        stp_type: 'std' or 'stf'
        delta_to_record: Delta value to record (degrees)
        
    Returns:
        Dictionary with recorded time courses for visualization
    """
    if config is None:
        config = SingleLayerExperimentConfig()
    config.stp_type = stp_type
    
    # Initialize
    key = jax.random.PRNGKey(config.seed)
    net_config = config.get_network_config()
    params = config.to_cann_params()
    model = SingleLayerCANN(params)
    
    # Add connection noise
    key, subkey = jax.random.split(key)
    model.kernel = create_noisy_kernel(model.kernel, net_config.mu_J, subkey)
    
    # Set stimulus orientations (#13: θ_s1=-30°, θ_s2=0°)
    theta_s2 = config.reference_theta  # 0°
    theta_s1 = theta_s2 + delta_to_record  # -30°
    
    # Run trial with recording
    key, subkey = jax.random.split(key)
    result = run_single_trial(model, theta_s1, theta_s2, config, subkey, record=True)
    
    # Get theta array
    theta_arr = np.linspace(-90, 90, net_config.N, endpoint=False)
    
    # Find neuron indices for stimulus positions
    s1_neuron = np.argmin(np.abs(theta_arr - theta_s1))
    s2_neuron = np.argmin(np.abs(theta_arr - theta_s2))
    
    # Extract final activity pattern for Fig.2B/E (#25)
    cue_activity = result.get('cue_activity_mean', model.state.r)
    
    return {
        'timeseries': result['timeseries'],
        'theta': theta_arr,
        'theta_s1': theta_s1,
        'theta_s2': theta_s2,
        's1_neuron': s1_neuron,
        's2_neuron': s2_neuron,
        'stp_type': stp_type,
        'perceived': result['perceived'],
        'error': result['error'],
        'cue_activity': cue_activity,
        'final_stp_x': result['timeseries']['stp_x'][-1] if 'timeseries' in result else None,
        'final_stp_u': result['timeseries']['stp_u'][-1] if 'timeseries' in result else None,
        'timeline': config.timeline.get_phase_times(),
    }


def generate_fig2b_data(
    config: Optional[SingleLayerExperimentConfig] = None,
    stp_type: str = 'std',
) -> Dict:
    """Generate data specifically for Fig.2B/E visualization.
    
    Returns retrieved response bump and STP asymmetry (#25).
    
    Args:
        config: Experiment configuration
        stp_type: 'std' or 'stf'
        
    Returns:
        Dictionary with bump and STP distribution data
    """
    result = run_experiment_with_recording(config, stp_type, delta_to_record=-30.0)
    
    theta = result['theta']
    
    # Response bump (blue curve in Fig.2B/E top)
    response_bump = result['cue_activity']
    
    # STP distribution (bottom panel)
    # For STD: use x (depression variable)
    # For STF: use u (facilitation variable)
    if stp_type == 'std':
        stp_distribution = result['final_stp_x']
        stp_label = 'x (Depression)'
    else:
        stp_distribution = result['final_stp_u']
        stp_label = 'u (Facilitation)'
    
    return {
        'theta': theta,
        'response_bump': response_bump,
        'stp_distribution': stp_distribution,
        'stp_label': stp_label,
        'theta_s1': result['theta_s1'],
        'theta_s2': result['theta_s2'],
        'perceived': result['perceived'],
        'stp_type': stp_type,
    }


# =============================================================================
# ISI Sweep (Checklist #14)
# =============================================================================

def run_isi_sweep(
    isi_values: List[float],
    config: Optional[SingleLayerExperimentConfig] = None,
    stp_type: str = 'std',
    verbose: bool = True,
) -> Dict:
    """Run experiment with different ISI values.
    
    Tests how serial dependence decays with ISI (Fig.2C/F).
    
    Reference: Fig.2 caption "under different ISI conditions" (p.5)
    
    Args:
        isi_values: List of ISI values to test (ms)
        config: Experiment configuration
        stp_type: 'std' or 'stf'
        verbose: Show progress
        
    Returns:
        Dictionary with ISI sweep results including DoG amplitude for each ISI
    """
    if config is None:
        config = SingleLayerExperimentConfig()
    config.stp_type = stp_type
    
    results_by_isi = {}
    
    for isi in tqdm(isi_values, desc='ISI Sweep', disable=not verbose):
        # Update ISI in timeline
        config.timeline = TrialTimeline(isi=isi)
        
        # Run full experiment for this ISI
        result = run_single_layer_experiment(config, stp_type, verbose=False)
        
        results_by_isi[isi] = {
            'curve_binned': result['curve_binned'],
            'dog_fit': result['dog_fit'],
        }
    
    # Extract DoG amplitude vs ISI
    isi_arr = np.array(list(results_by_isi.keys()))
    amplitude_arr = np.array([results_by_isi[isi]['dog_fit']['amplitude'] for isi in isi_arr])
    
    return {
        'isi_values': isi_arr,
        'dog_amplitude': amplitude_arr,
        'results_by_isi': results_by_isi,
        'stp_type': stp_type,
    }


# =============================================================================
# Comparison Function
# =============================================================================

def compare_stp_types(
    config: Optional[SingleLayerExperimentConfig] = None,
    verbose: bool = True,
) -> Dict:
    """Run both STD and STF experiments for comparison.
    
    Reproduces both Fig.2A-C (STD) and Fig.2D-F (STF).
    
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


# =============================================================================
# Validation Functions (for testing)
# =============================================================================

def validate_config(config: SingleLayerExperimentConfig) -> Dict[str, bool]:
    """Validate configuration against paper requirements.
    
    Returns checklist validation results.
    """
    net_config = config.get_network_config()
    timeline = config.timeline
    
    checks = {}
    
    # #8: N=100
    checks['N_is_100'] = (net_config.N == 100)
    
    # #12: Timeline
    checks['s1_duration_200ms'] = (timeline.s1_duration == 200.0)
    checks['s2_duration_200ms'] = (timeline.s2_duration == 200.0)
    checks['isi_1000ms'] = (timeline.isi == 1000.0)
    checks['delay_3400ms'] = (timeline.delay == 3400.0)
    checks['cue_500ms'] = (timeline.cue_duration == 500.0)
    checks['iti_1000ms'] = (timeline.iti == 1000.0)
    
    # #18: Delta step = 1°
    checks['delta_step_1deg'] = (config.delta_step == 1.0)
    
    # #22: 20 runs × 100 trials
    checks['n_runs_20'] = (config.n_runs == 20)
    checks['n_trials_100'] = (config.n_trials_per_run == 100)
    
    # #15/#16: STP parameters
    if config.stp_type == 'std':
        checks['J0_correct'] = (net_config.J0 == 0.13)
        checks['a_correct'] = (net_config.a == 0.5)
        checks['k_correct'] = (net_config.k == 0.0018)
        checks['tau_d_correct'] = (net_config.tau_d == 3.0)
        checks['tau_f_correct'] = (net_config.tau_f == 0.3)
        checks['U_correct'] = (net_config.U == 0.5)
    elif config.stp_type == 'stf':
        checks['J0_correct'] = (net_config.J0 == 0.09)
        checks['a_correct'] = (net_config.a == 0.15)
        checks['k_correct'] = (net_config.k == 0.0095)
        checks['tau_d_correct'] = (net_config.tau_d == 0.3)
        checks['tau_f_correct'] = (net_config.tau_f == 5.0)
        checks['U_correct'] = (net_config.U == 0.2)
    
    # #17: τ = 0.01s = 10ms
    checks['tau_10ms'] = (net_config.tau == 10.0)
    
    # #3: Connection noise
    checks['mu_J_0.01'] = (net_config.mu_J == 0.01)
    
    # #11: Input parameters
    checks['alpha_sti_20'] = (net_config.alpha_sti == 20.0)
    checks['a_sti_0.3'] = (net_config.a_sti == 0.3)
    checks['alpha_cue_2.5'] = (net_config.alpha_cue == 2.5)
    checks['a_cue_0.4'] = (net_config.a_cue == 0.4)
    
    return checks


def print_validation_report(config: SingleLayerExperimentConfig):
    """Print validation report."""
    checks = validate_config(config)
    
    print(f"\n{'='*60}")
    print(f"Validation Report - {config.stp_type.upper()} Configuration")
    print(f"{'='*60}")
    
    passed = sum(checks.values())
    total = len(checks)
    
    for check_name, passed_check in checks.items():
        status = "✅" if passed_check else "❌"
        print(f"  {status} {check_name}")
    
    print(f"{'='*60}")
    print(f"Passed: {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"{'='*60}\n")
