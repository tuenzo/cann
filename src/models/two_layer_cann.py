"""
Two-Layer Continuous Attractor Neural Network Model
====================================================

Implements a hierarchical two-layer CANN with:
- Lower layer: STD-dominated (sensory cortex, e.g., V1) → repulsion
- Higher layer: STF-dominated (prefrontal cortex) → attraction

The layers are connected by:
- Feedforward connections: lower → higher (J_ff)
- Feedback connections: higher → lower (J_fb)

Reference: Zhang et al., NeurIPS 2025
"""

from typing import NamedTuple, Optional
from functools import partial
import jax
import jax.numpy as jnp

from .stp import STPState, create_stp_state, stp_step, stp_efficacy
from .cann import (
    CANNParams, CANNState, 
    create_gaussian_kernel, create_stimulus,
    divisive_normalization, circular_convolve,
)


class TwoLayerParams(NamedTuple):
    """Parameters for two-layer CANN.
    
    Attributes:
        low: Lower layer parameters (STD-dominated)
        high: Higher layer parameters (STF-dominated)
        J_ff: Feedforward connection strength (lower → higher)
        J_fb: Feedback connection strength (higher → lower)
        a_ff: Feedforward connection width (degrees)
        a_fb: Feedback connection width (degrees)
    """
    low: CANNParams = CANNParams(tau_d=3.0, tau_f=0.3)   # STD-dominated
    high: CANNParams = CANNParams(tau_d=0.3, tau_f=5.0)  # STF-dominated
    J_ff: float = 0.3   # 前馈连接强度
    J_fb: float = 0.2   # 反馈连接强度
    a_ff: float = 30.0  # 前馈连接宽度
    a_fb: float = 30.0  # 反馈连接宽度


class TwoLayerState(NamedTuple):
    """State variables for two-layer CANN.
    
    Attributes:
        low: Lower layer state
        high: Higher layer state
    """
    low: CANNState
    high: CANNState


def create_inter_layer_kernel(
    N: int, 
    a: float, 
    J0: float,
) -> jnp.ndarray:
    """Create Gaussian inter-layer connectivity kernel.
    
    Args:
        N: Number of neurons per layer
        a: Connection width in degrees
        J0: Connection strength
        
    Returns:
        Kernel for inter-layer connections, shape (N,)
    """
    return create_gaussian_kernel(N, a, J0)


def two_layer_step(
    state: TwoLayerState,
    I_ext: jnp.ndarray,
    kernel_low: jnp.ndarray,
    kernel_high: jnp.ndarray,
    kernel_ff: jnp.ndarray,
    kernel_fb: jnp.ndarray,
    params: TwoLayerParams,
) -> TwoLayerState:
    """Single time step update of two-layer CANN.
    
    Update order:
    1. Compute feedforward input to higher layer
    2. Update higher layer
    3. Compute feedback input to lower layer  
    4. Update lower layer
    
    Args:
        state: Current two-layer state
        I_ext: External input to lower layer, shape (N,)
        kernel_low: Lower layer recurrent kernel
        kernel_high: Higher layer recurrent kernel
        kernel_ff: Feedforward connection kernel
        kernel_fb: Feedback connection kernel
        params: Network parameters
        
    Returns:
        Updated two-layer state
    """
    low_state, high_state = state.low, state.high
    
    # === Lower layer (STD-dominated) ===
    low_u, low_r, low_stp = low_state.u, low_state.r, low_state.stp
    
    # Feedback from higher layer
    feedback = circular_convolve(high_state.r, kernel_fb)
    
    # STP-modulated recurrent input
    low_efficacy = stp_efficacy(low_stp)
    low_recurrent = circular_convolve(low_r * low_efficacy, kernel_low)
    
    # Membrane potential dynamics
    low_du = (-low_u + params.low.rho * low_recurrent + I_ext + feedback) / params.low.tau
    low_u_new = low_u + low_du * params.low.dt
    
    # Firing rate
    low_r_new = divisive_normalization(low_u_new, params.low.k, params.low.rho)
    
    # STP update
    low_stp_new = stp_step(
        low_stp, low_r, 
        params.low.tau_d, params.low.tau_f, params.low.U, params.low.dt
    )
    
    low_state_new = CANNState(u=low_u_new, r=low_r_new, stp=low_stp_new)
    
    # === Higher layer (STF-dominated) ===
    high_u, high_r, high_stp = high_state.u, high_state.r, high_state.stp
    
    # Feedforward from lower layer (use NEW lower layer activity)
    feedforward = circular_convolve(low_r_new, kernel_ff)
    
    # STP-modulated recurrent input
    high_efficacy = stp_efficacy(high_stp)
    high_recurrent = circular_convolve(high_r * high_efficacy, kernel_high)
    
    # Membrane potential dynamics
    high_du = (-high_u + params.high.rho * high_recurrent + feedforward) / params.high.tau
    high_u_new = high_u + high_du * params.high.dt
    
    # Firing rate
    high_r_new = divisive_normalization(high_u_new, params.high.k, params.high.rho)
    
    # STP update
    high_stp_new = stp_step(
        high_stp, high_r,
        params.high.tau_d, params.high.tau_f, params.high.U, params.high.dt
    )
    
    high_state_new = CANNState(u=high_u_new, r=high_r_new, stp=high_stp_new)
    
    return TwoLayerState(low=low_state_new, high=high_state_new)


class TwoLayerCANN:
    """Two-layer CANN model for serial dependence.
    
    This model captures:
    - Within-trial repulsion: STD in lower layer
    - Between-trial attraction: STF in higher layer
    
    Example:
        >>> params = TwoLayerParams()
        >>> model = TwoLayerCANN(params)
        >>> model.reset()
        >>> model.present_stimulus(45.0, duration=500)  # S1
        >>> model.evolve(1000)  # ISI
        >>> model.present_stimulus(60.0, duration=500)  # S2
        >>> perceived = model.decode('high')  # Decode from higher layer
    """
    
    def __init__(self, params: Optional[TwoLayerParams] = None):
        """Initialize two-layer CANN.
        
        Args:
            params: Network parameters, uses defaults if None
        """
        self.params = params or TwoLayerParams()
        N = self.params.low.N
        
        # Create connectivity kernels
        self.kernel_low = create_gaussian_kernel(
            N, self.params.low.a, self.params.low.J0
        )
        self.kernel_high = create_gaussian_kernel(
            N, self.params.high.a, self.params.high.J0
        )
        self.kernel_ff = create_inter_layer_kernel(
            N, self.params.a_ff, self.params.J_ff
        )
        self.kernel_fb = create_inter_layer_kernel(
            N, self.params.a_fb, self.params.J_fb
        )
        
        self.theta = jnp.linspace(0, 180, N, endpoint=False)
        self.reset()
        
        # JIT compile step function
        self._step = jax.jit(
            lambda state, I_ext: two_layer_step(
                state, I_ext,
                self.kernel_low, self.kernel_high,
                self.kernel_ff, self.kernel_fb,
                self.params
            )
        )
    
    def reset(self):
        """Reset network to initial state."""
        N = self.params.low.N
        
        low_state = CANNState(
            u=jnp.zeros(N),
            r=jnp.zeros(N),
            stp=create_stp_state(N, self.params.low.U),
        )
        high_state = CANNState(
            u=jnp.zeros(N),
            r=jnp.zeros(N),
            stp=create_stp_state(N, self.params.high.U),
        )
        
        self.state = TwoLayerState(low=low_state, high=high_state)
        self.time = 0.0
    
    def step(self, I_ext: Optional[jnp.ndarray] = None):
        """Advance simulation by one time step.
        
        Args:
            I_ext: External input to lower layer
        """
        if I_ext is None:
            I_ext = jnp.zeros(self.params.low.N)
        
        self.state = self._step(self.state, I_ext)
        self.time += self.params.low.dt
    
    def evolve(self, duration: float, I_ext: Optional[jnp.ndarray] = None):
        """Evolve network for specified duration.
        
        Args:
            duration: Time in milliseconds
            I_ext: Constant external input
        """
        n_steps = int(duration / self.params.low.dt)
        if I_ext is None:
            I_ext = jnp.zeros(self.params.low.N)
        
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
            theta_stim, self.params.low.N, 
            self.params.low.A, self.params.low.stim_width
        )
        
        n_steps = int(duration / self.params.low.dt)
        
        if record:
            history = {
                'time': [],
                'low_r': [], 'high_r': [],
                'low_stp_x': [], 'low_stp_u': [],
                'high_stp_x': [], 'high_stp_u': [],
            }
            
            for _ in range(n_steps):
                self.state = self._step(self.state, I_ext)
                self.time += self.params.low.dt
                
                history['time'].append(self.time)
                history['low_r'].append(self.state.low.r)
                history['high_r'].append(self.state.high.r)
                history['low_stp_x'].append(self.state.low.stp.x)
                history['low_stp_u'].append(self.state.low.stp.u)
                history['high_stp_x'].append(self.state.high.stp.x)
                history['high_stp_u'].append(self.state.high.stp.u)
            
            for key in history:
                history[key] = jnp.array(history[key])
            
            return history
        else:
            self.evolve(duration, I_ext)
            return None
    
    def get_activity(self, layer: str = 'high') -> jnp.ndarray:
        """Get firing rate activity from specified layer.
        
        Args:
            layer: 'low' or 'high'
            
        Returns:
            Firing rate array
        """
        if layer == 'low':
            return self.state.low.r
        else:
            return self.state.high.r
    
    def decode(self, layer: str = 'high', method: str = 'pvm') -> float:
        """Decode perceived orientation from population activity.
        
        Args:
            layer: 'low' for sensory, 'high' for perceptual
            method: Decoding method ('pvm', 'com', 'ml', 'peak')
            
        Returns:
            Decoded orientation in degrees
        """
        from ..decoding import decode_orientation
        r = self.get_activity(layer)
        return decode_orientation(r, self.theta, method)


def run_two_layer_trial(
    model: TwoLayerCANN,
    theta_s1: float,
    theta_s2: float,
    isi: float,
    stim_duration: float = 500.0,
    decode_layer: str = 'high',
) -> tuple[float, float]:
    """Run a single serial dependence trial with two-layer model.
    
    Args:
        model: Two-layer CANN model
        theta_s1: First stimulus orientation
        theta_s2: Second stimulus orientation
        isi: Inter-stimulus interval (ms)
        stim_duration: Stimulus duration (ms)
        decode_layer: Layer to decode from
        
    Returns:
        perceived: Decoded orientation
        error: Adjustment error
    """
    # Present S1
    model.present_stimulus(theta_s1, stim_duration)
    
    # ISI
    model.evolve(isi)
    
    # Present S2
    model.present_stimulus(theta_s2, stim_duration)
    
    # Decode
    perceived = model.decode(decode_layer)
    
    # Compute error
    error = perceived - theta_s2
    if error > 90:
        error -= 180
    elif error < -90:
        error += 180
    
    return perceived, error


def run_within_trial_experiment(
    model: TwoLayerCANN,
    theta_s1: float,
    theta_s2: float,
    isi: float,
    stim_duration: float = 500.0,
) -> tuple[float, float]:
    """Run within-trial serial dependence experiment.
    
    Within-trial: Effect of S1 on perception of S2 within the same trial.
    Expected: Repulsion (negative bias) due to STD in lower layer.
    
    Args:
        model: Two-layer model
        theta_s1, theta_s2: Stimulus orientations
        isi: Inter-stimulus interval
        stim_duration: Stimulus duration
        
    Returns:
        perceived: Decoded S2
        error: Within-trial adjustment error
    """
    model.reset()
    return run_two_layer_trial(model, theta_s1, theta_s2, isi, stim_duration, 'high')


def run_between_trial_experiment(
    model: TwoLayerCANN,
    theta_s1_prev: float,
    theta_s1_curr: float,
    iti: float,
    stim_duration: float = 500.0,
) -> tuple[float, float]:
    """Run between-trial serial dependence experiment.
    
    Between-trial: Effect of S1 from previous trial on S1 of current trial.
    Expected: Attraction (positive bias) due to STF in higher layer.
    
    Args:
        model: Two-layer model (should NOT be reset between trials)
        theta_s1_prev: S1 from previous trial
        theta_s1_curr: S1 from current trial (to be perceived)
        iti: Inter-trial interval
        stim_duration: Stimulus duration
        
    Returns:
        perceived: Decoded current S1
        error: Between-trial adjustment error
    """
    # Previous trial S1
    model.present_stimulus(theta_s1_prev, stim_duration)
    
    # ITI
    model.evolve(iti)
    
    # Current trial S1
    model.present_stimulus(theta_s1_curr, stim_duration)
    
    # Decode
    perceived = model.decode('high')
    
    # Error
    error = perceived - theta_s1_curr
    if error > 90:
        error -= 180
    elif error < -90:
        error += 180
    
    return perceived, error

