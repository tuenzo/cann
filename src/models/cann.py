"""
Single-Layer Continuous Attractor Neural Network (CANN) Model
=============================================================

Implements a ring attractor network for orientation representation
with divisive normalization and short-term synaptic plasticity.

Key equations:
    τ du(x,t)/dt = -u(x,t) + ρ∫dx' J(x,x')r(x',t) + I_ext(x,t)
    r(x,t) = u(x,t)² / (1 + kρ∫u(x',t)²dx')  [divisive normalization]
    J(x,x') = J₀/(√(2π)a) exp(-(x-x')²/(2a²)) [Gaussian connectivity]

Reference: Zhang et al., NeurIPS 2025
"""

from typing import NamedTuple, Optional
from functools import partial
import jax
import jax.numpy as jnp
from jax.scipy.signal import fftconvolve

from .stp import STPState, create_stp_state, stp_step, stp_efficacy


class CANNParams(NamedTuple):
    """Parameters for single-layer CANN.
    
    Attributes:
        N: Number of neurons
        J0: Recurrent connection strength
        a: Gaussian tuning width (degrees)
        tau: Membrane time constant (ms)
        k: Divisive normalization strength
        rho: Neural density
        A: Stimulus amplitude
        stim_width: Stimulus width (degrees)
        dt: Time step (ms)
        tau_d: STD time constant (s)
        tau_f: STF time constant (s)
        U: Baseline release probability
    """
    N: int = 180
    J0: float = 0.5
    a: float = 30.0
    tau: float = 1.0
    k: float = 0.5
    rho: float = 1.0
    A: float = 1.0
    stim_width: float = 30.0
    dt: float = 0.1
    tau_d: float = 3.0
    tau_f: float = 0.3
    U: float = 0.3


class CANNState(NamedTuple):
    """State variables for CANN.
    
    Attributes:
        u: Membrane potential, shape (N,)
        r: Firing rate, shape (N,)
        stp: STP state (x, u)
    """
    u: jnp.ndarray    # 膜电位
    r: jnp.ndarray    # 发放率
    stp: STPState     # STP状态


def create_gaussian_kernel(N: int, a: float, J0: float) -> jnp.ndarray:
    """Create Gaussian recurrent connectivity kernel.
    
    J(x,x') = J₀/(√(2π)a) exp(-(x-x')²/(2a²))
    
    Args:
        N: Number of neurons (covering 180°)
        a: Tuning width in degrees
        J0: Connection strength
        
    Returns:
        Kernel array of shape (N,) for circular convolution
    """
    # Neural positions in degrees (0 to 180)
    theta = jnp.linspace(0, 180, N, endpoint=False)
    
    # Distance from center, accounting for circular topology
    dx = theta - 90  # Center at 90 degrees
    
    # Gaussian kernel
    kernel = J0 / (jnp.sqrt(2 * jnp.pi) * a) * jnp.exp(-dx**2 / (2 * a**2))
    
    # Shift to center kernel at position 0 for convolution
    kernel = jnp.roll(kernel, -N // 2)
    
    return kernel


def create_stimulus(
    theta_stim: float,
    N: int,
    A: float = 1.0,
    width: float = 30.0,
) -> jnp.ndarray:
    """Create Gaussian stimulus input at given orientation.
    
    I_ext(x) = A * exp(-(x - θ_stim)² / (2 * width²))
    
    Args:
        theta_stim: Stimulus orientation in degrees (0-180)
        N: Number of neurons
        A: Stimulus amplitude
        width: Stimulus width in degrees
        
    Returns:
        External input array of shape (N,)
    """
    theta = jnp.linspace(0, 180, N, endpoint=False)
    
    # Circular distance (handle wrap-around at 0/180)
    dx = theta - theta_stim
    # Map to [-90, 90] range for proper circular distance
    dx = jnp.where(dx > 90, dx - 180, dx)
    dx = jnp.where(dx < -90, dx + 180, dx)
    
    return A * jnp.exp(-dx**2 / (2 * width**2))


@jax.jit
def divisive_normalization(
    u: jnp.ndarray,
    k: float,
    rho: float = 1.0,
) -> jnp.ndarray:
    """Apply divisive normalization to compute firing rate.
    
    r = u² / (1 + kρ∫u²dx')
    
    For discrete case with N neurons covering 180°:
    r = u² / (1 + k * rho * (180/N) * Σu²)
    
    Args:
        u: Membrane potential, shape (N,)
        k: Normalization strength
        rho: Neural density
        
    Returns:
        Firing rate, shape (N,)
    """
    N = u.shape[0]
    dx = 180.0 / N  # Spatial resolution
    
    # Only consider positive part for rate computation
    u_pos = jnp.maximum(u, 0.0)
    
    # Global normalization factor
    norm_factor = 1.0 + k * rho * dx * jnp.sum(u_pos**2)
    
    return u_pos**2 / norm_factor


@jax.jit
def circular_convolve(x: jnp.ndarray, kernel: jnp.ndarray) -> jnp.ndarray:
    """Efficient circular convolution using FFT.
    
    The orientation space is circular (0° ≡ 180°), so we need
    circular boundary conditions.
    
    Args:
        x: Input signal, shape (N,)
        kernel: Convolution kernel, shape (N,)
        
    Returns:
        Circular convolution result, shape (N,)
    """
    # FFT-based circular convolution
    N = x.shape[0]
    result = jnp.real(jnp.fft.ifft(jnp.fft.fft(x) * jnp.fft.fft(kernel)))
    return result


def cann_step(
    state: CANNState,
    I_ext: jnp.ndarray,
    kernel: jnp.ndarray,
    params: CANNParams,
) -> CANNState:
    """Single time step update of CANN dynamics.
    
    Updates membrane potential, firing rate, and STP state.
    
    Args:
        state: Current CANN state
        I_ext: External input, shape (N,)
        kernel: Recurrent connectivity kernel
        params: Network parameters
        
    Returns:
        Updated CANN state
    """
    u, r, stp = state.u, state.r, state.stp
    
    # Compute STP-modulated effective connectivity
    efficacy = stp_efficacy(stp)  # u_stp * x_stp
    
    # Recurrent input with STP modulation
    # J_eff(x,x') = J(x,x') * u(x') * x(x')
    recurrent_input = circular_convolve(r * efficacy, kernel)
    
    # Membrane potential dynamics: τ du/dt = -u + recurrent + I_ext
    du = (-u + params.rho * recurrent_input + I_ext) / params.tau
    u_new = u + du * params.dt
    
    # Firing rate with divisive normalization
    r_new = divisive_normalization(u_new, params.k, params.rho)
    
    # Update STP state
    stp_new = stp_step(stp, r, params.tau_d, params.tau_f, params.U, params.dt)
    
    return CANNState(u=u_new, r=r_new, stp=stp_new)


class SingleLayerCANN:
    """Single-layer CANN model with STP.
    
    This class provides a convenient interface for simulating the CANN
    with various experimental protocols.
    
    Example:
        >>> params = CANNParams(tau_d=3.0, tau_f=0.3)  # STD-dominated
        >>> model = SingleLayerCANN(params)
        >>> model.reset()
        >>> model.present_stimulus(45.0, duration=500)  # Present at 45°
        >>> decoded = model.decode()  # Get perceived orientation
    """
    
    def __init__(self, params: Optional[CANNParams] = None):
        """Initialize CANN model.
        
        Args:
            params: Network parameters, uses defaults if None
        """
        self.params = params or CANNParams()
        self.kernel = create_gaussian_kernel(
            self.params.N, self.params.a, self.params.J0
        )
        self.theta = jnp.linspace(0, 180, self.params.N, endpoint=False)
        self.reset()
        
        # JIT compile the step function
        self._step = jax.jit(lambda state, I_ext: cann_step(
            state, I_ext, self.kernel, self.params
        ))
    
    def reset(self):
        """Reset network to initial state."""
        N = self.params.N
        self.state = CANNState(
            u=jnp.zeros(N),
            r=jnp.zeros(N),
            stp=create_stp_state(N, self.params.U),
        )
        self.time = 0.0
        self._history = []
    
    def step(self, I_ext: Optional[jnp.ndarray] = None):
        """Advance simulation by one time step.
        
        Args:
            I_ext: External input. If None, uses zero input.
        """
        if I_ext is None:
            I_ext = jnp.zeros(self.params.N)
        
        self.state = self._step(self.state, I_ext)
        self.time += self.params.dt
    
    def evolve(self, duration: float, I_ext: Optional[jnp.ndarray] = None):
        """Evolve network for specified duration.
        
        Args:
            duration: Time to evolve in milliseconds
            I_ext: Constant external input during evolution
        """
        n_steps = int(duration / self.params.dt)
        if I_ext is None:
            I_ext = jnp.zeros(self.params.N)
        
        for _ in range(n_steps):
            self.state = self._step(self.state, I_ext)
        
        self.time += duration
    
    def present_stimulus(
        self,
        theta_stim: float,
        duration: float,
        record: bool = False,
    ) -> Optional[dict]:
        """Present a stimulus at given orientation.
        
        Args:
            theta_stim: Stimulus orientation in degrees
            duration: Presentation duration in ms
            record: Whether to record state history
            
        Returns:
            If record=True, returns dict with time course data
        """
        I_ext = create_stimulus(
            theta_stim, self.params.N, self.params.A, self.params.stim_width
        )
        
        n_steps = int(duration / self.params.dt)
        
        if record:
            history = {
                'time': [],
                'u': [],
                'r': [],
                'stp_x': [],
                'stp_u': [],
            }
            
            for _ in range(n_steps):
                self.state = self._step(self.state, I_ext)
                self.time += self.params.dt
                
                history['time'].append(self.time)
                history['u'].append(self.state.u)
                history['r'].append(self.state.r)
                history['stp_x'].append(self.state.stp.x)
                history['stp_u'].append(self.state.stp.u)
            
            # Convert to arrays
            for key in history:
                history[key] = jnp.array(history[key])
            
            return history
        else:
            self.evolve(duration, I_ext)
            return None
    
    def get_activity(self) -> jnp.ndarray:
        """Get current firing rate activity."""
        return self.state.r
    
    def get_stp_state(self) -> STPState:
        """Get current STP state."""
        return self.state.stp
    
    def decode(self, method: str = 'pvm') -> float:
        """Decode perceived orientation from population activity.
        
        Args:
            method: Decoding method ('pvm', 'com', 'ml', 'peak')
            
        Returns:
            Decoded orientation in degrees
        """
        from ..decoding import decode_orientation
        return decode_orientation(self.state.r, self.theta, method)


# Vectorized functions for batch processing
@partial(jax.jit, static_argnames=['n_steps'])
def run_cann_trial(
    initial_state: CANNState,
    stimulus: jnp.ndarray,
    kernel: jnp.ndarray,
    params: CANNParams,
    n_steps: int,
) -> tuple[CANNState, jnp.ndarray]:
    """Run a single trial with constant stimulus.
    
    Args:
        initial_state: Starting state
        stimulus: External input (constant)
        kernel: Connectivity kernel
        params: Network parameters
        n_steps: Number of time steps
        
    Returns:
        final_state: State after simulation
        r_history: Firing rate history (n_steps, N)
    """
    def step_fn(state, _):
        new_state = cann_step(state, stimulus, kernel, params)
        return new_state, new_state.r
    
    final_state, r_history = jax.lax.scan(
        step_fn, initial_state, None, length=n_steps
    )
    return final_state, r_history


def run_serial_dependence_trial(
    model: SingleLayerCANN,
    theta_s1: float,
    theta_s2: float,
    isi: float,
    stim_duration: float = 500.0,
) -> tuple[float, float]:
    """Run a single serial dependence trial.
    
    Protocol:
    1. Present S1 for stim_duration
    2. Wait for ISI (inter-stimulus interval)
    3. Present S2 for stim_duration
    4. Decode perceived S2
    5. Compute adjustment error
    
    Args:
        model: CANN model instance
        theta_s1: First stimulus orientation (degrees)
        theta_s2: Second stimulus orientation (degrees)
        isi: Inter-stimulus interval (ms)
        stim_duration: Stimulus presentation time (ms)
        
    Returns:
        perceived: Decoded orientation of S2
        error: Adjustment error (perceived - actual)
    """
    # Present S1
    model.present_stimulus(theta_s1, stim_duration)
    
    # ISI period (no stimulus)
    model.evolve(isi)
    
    # Present S2
    model.present_stimulus(theta_s2, stim_duration)
    
    # Decode perceived orientation
    perceived = model.decode()
    
    # Compute error (positive = attraction toward S1, negative = repulsion)
    error = perceived - theta_s2
    
    # Wrap to [-90, 90]
    if error > 90:
        error -= 180
    elif error < -90:
        error += 180
    
    return perceived, error

