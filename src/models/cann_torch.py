"""
PyTorch implementation of Continuous Attractor Neural Network (CANN)
====================================================================

GPU-accelerated CANN model using PyTorch for Windows CUDA support.
支持单样本和批量处理两种模式。

所有时间常数单位统一为毫秒(ms)。
"""

import torch
import torch.nn.functional as F
from typing import NamedTuple, Optional, Tuple, Dict
from dataclasses import dataclass
import math
import numpy as np


# ============ Configuration ============

@dataclass
class CANNConfig:
    """CANN model configuration.
    
    时间单位说明：
    - tau: 神经电流时间常数 (ms)，Table 1: 0.01s = 10ms
    - tau_d: 突触抑制时间常数 (秒)，Table 1: STD=3s, STF=0.3s
    - tau_f: 突触促进时间常数 (秒)，Table 1: STD=0.3s, STF=5s
    - dt: 积分时间步长 (ms)
    
    注：STP时间常数使用秒，神经电流时间常数使用毫秒
    """
    N: int = 100            # Number of neurons
    J0: float = 0.5         # Connection strength
    a: float = 0.5          # Connection width (radians)
    tau: float = 10.0       # Membrane time constant (ms)
    k: float = 0.5          # Divisive normalization strength
    rho: float = 1.0        # Neural density
    dt: float = 0.1         # Time step (ms)
    tau_d: float = 3.0      # STD recovery time constant (秒!)
    tau_f: float = 0.3      # STF facilitation time constant (秒!)
    U: float = 0.45         # Baseline release probability
    # 噪声参数 (Table 1)
    mu_J: float = 0.01      # Connection noise strength
    mu_b: float = 0.5       # Background noise strength
    
    @classmethod
    def std_dominated(cls):
        """STD-dominated: repulsion effect (Table 1).
        
        Table 1 参数:
        - J0=0.13, a=0.5 rad, k=0.0018, τ=10ms
        - τ_d=3s, τ_f=0.3s, U=0.5 (STP时间常数为秒)
        - µJ=0.01, µb=0.5 (噪声参数)
        """
        return cls(N=100, J0=0.13, a=0.5, tau=10.0, k=0.0018, rho=1.0, dt=0.1,
                   tau_d=3.0, tau_f=0.3, U=0.5, mu_J=0.01, mu_b=0.5)
    
    @classmethod
    def stf_dominated(cls):
        """STF-dominated: attraction effect (Table 1).
        
        Table 1 参数:
        - J0=0.09, a=0.15 rad, k=0.0095, τ=10ms
        - τ_d=0.3s, τ_f=5s, U=0.2 (STP时间常数为秒)
        - µJ=0.01, µb=0.5 (噪声参数)
        """
        return cls(N=100, J0=0.09, a=0.15, tau=10.0, k=0.0095, rho=1.0, dt=0.1,
                   tau_d=0.3, tau_f=5.0, U=0.2, mu_J=0.01, mu_b=0.5)


@dataclass
class TrialConfig:
    """Trial timing configuration (单位: ms)."""
    s1_duration: float = 200.0    # ms
    isi: float = 1000.0           # ms
    s2_duration: float = 200.0    # ms  
    delay: float = 3400.0         # ms
    cue_duration: float = 500.0   # ms
    # Stimulus parameters (Table 1)
    alpha_ext: float = 20.0       # Stimulus amplitude
    a_ext: float = 0.3            # Stimulus width (radians)
    mu_sti: float = 0.5           # Stimulus noise strength
    # Cue parameters (Table 1)
    alpha_cue: float = 2.5        # Cue amplitude
    a_cue: float = 0.4            # Cue width (radians)
    mu_cue: float = 1.0           # Cue noise strength


# ============ State Types ============

class CANNState(NamedTuple):
    """Single-sample CANN state."""
    u: torch.Tensor      # Membrane potential (N,)
    r: torch.Tensor      # Firing rate (N,)
    stp_x: torch.Tensor  # STP depression variable (N,)
    stp_u: torch.Tensor  # STP facilitation variable (N,)


class BatchCANNState(NamedTuple):
    """Batch CANN state."""
    u: torch.Tensor      # (batch, N)
    r: torch.Tensor      # (batch, N)
    stp_x: torch.Tensor  # (batch, N)
    stp_u: torch.Tensor  # (batch, N)


# ============ Single-Sample Model ============

class CANN(torch.nn.Module):
    """PyTorch CANN model with STP dynamics (single sample)."""
    
    def __init__(self, config: CANNConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        step = 180.0 / config.N
        self.theta = torch.arange(-90, 90, step, device=self.device, dtype=torch.float32)[:config.N]
        
        self.kernel = self._create_kernel()
        self.kernel_fft = torch.fft.fft(self.kernel)
    
    def _create_kernel(self) -> torch.Tensor:
        """Create Gaussian connection kernel."""
        N = self.config.N
        J0 = self.config.J0
        a = self.config.a
        
        theta = torch.linspace(-90, 90, N, device=self.device, dtype=torch.float32)
        theta = theta - theta[N // 2]
        theta_rad = theta * math.pi / 180.0
        kernel = J0 * torch.exp(-theta_rad**2 / (2 * a**2))
        kernel = kernel / kernel.sum() * N
        return kernel
    
    def init_state(self) -> CANNState:
        """Initialize CANN state."""
        N = self.config.N
        U = self.config.U
        
        return CANNState(
            u=torch.zeros(N, device=self.device, dtype=torch.float32),
            r=torch.zeros(N, device=self.device, dtype=torch.float32),
            stp_x=torch.ones(N, device=self.device, dtype=torch.float32),
            stp_u=torch.full((N,), U, device=self.device, dtype=torch.float32),
        )
    
    def create_stimulus(self, theta_stim: float, amplitude: float = 20.0, 
                       width: float = 0.3) -> torch.Tensor:
        """Create Gaussian stimulus."""
        dx = self.theta - theta_stim
        dx = torch.where(dx > 90, dx - 180, dx)
        dx = torch.where(dx < -90, dx + 180, dx)
        width_deg = width * 180 / math.pi
        return amplitude * torch.exp(-dx**2 / (2 * width_deg**2))
    
    def step(self, state: CANNState, I_ext: torch.Tensor) -> CANNState:
        """Perform one simulation step."""
        cfg = self.config
        u, r, stp_x, stp_u = state
        
        efficacy = stp_u * stp_x
        r_eff = r * efficacy
        
        r_fft = torch.fft.fft(r_eff)
        recurrent_input = torch.fft.ifft(r_fft * self.kernel_fft).real
        
        du = (-u + cfg.rho * recurrent_input + I_ext) / cfg.tau
        u_new = u + du * cfg.dt
        
        u_pos = torch.clamp(u_new, min=0)
        u_squared = u_pos ** 2
        normalization = 1.0 + cfg.k * cfg.rho * u_squared.sum()
        r_new = u_squared / normalization
        
        # STP dynamics - tau_d/tau_f 是秒，dt 是毫秒，需要转换
        dt_sec = cfg.dt / 1000.0  # 毫秒转秒
        dx = (1.0 - stp_x) / cfg.tau_d - stp_u * stp_x * r
        stp_x_new = torch.clamp(stp_x + dx * dt_sec, 0.0, 1.0)
        
        du_stp = (cfg.U - stp_u) / cfg.tau_f + cfg.U * (1.0 - stp_u) * r
        stp_u_new = torch.clamp(stp_u + du_stp * dt_sec, 0.0, 1.0)
        
        return CANNState(u=u_new, r=r_new, stp_x=stp_x_new, stp_u=stp_u_new)
    
    def run_phase(self, state: CANNState, I_ext: torch.Tensor, n_steps: int) -> CANNState:
        """Run multiple simulation steps."""
        for _ in range(n_steps):
            state = self.step(state, I_ext)
        return state
    
    def decode_orientation(self, r: torch.Tensor) -> float:
        """Decode orientation from population activity."""
        if r.dim() == 2:
            r = r.mean(dim=0)
        
        theta_rad = self.theta * math.pi / 180
        cos_sum = (r * torch.cos(2 * theta_rad)).sum()
        sin_sum = (r * torch.sin(2 * theta_rad)).sum()
        
        perceived_rad = torch.atan2(sin_sum, cos_sum) / 2
        perceived = perceived_rad * 180 / math.pi
        
        if perceived >= 90:
            perceived -= 180
        elif perceived < -90:
            perceived += 180
        
        return perceived.item()


# ============ Batch-Parallel Model ============

class BatchCANN:
    """Batch-parallel CANN model for GPU acceleration."""
    
    def __init__(self, config: CANNConfig, batch_size: int, device: torch.device):
        self.config = config
        self.batch_size = batch_size
        self.device = device
        
        step = 180.0 / config.N
        self.theta = torch.arange(-90, 90, step, device=device, dtype=torch.float32)[:config.N]
        
        self.kernel = self._create_kernel()
        self.kernel_fft = torch.fft.fft(self.kernel)
    
    def _create_kernel(self) -> torch.Tensor:
        """Create Gaussian connection kernel."""
        N = self.config.N
        J0 = self.config.J0
        a = self.config.a
        
        theta = torch.linspace(-90, 90, N, device=self.device)
        theta = theta - theta[N // 2]
        theta_rad = theta * math.pi / 180.0
        kernel = J0 * torch.exp(-theta_rad**2 / (2 * a**2))
        kernel = kernel / kernel.sum() * N
        return kernel
    
    def init_state(self) -> BatchCANNState:
        """Initialize batch CANN state."""
        N = self.config.N
        B = self.batch_size
        U = self.config.U
        
        return BatchCANNState(
            u=torch.zeros(B, N, device=self.device),
            r=torch.zeros(B, N, device=self.device),
            stp_x=torch.ones(B, N, device=self.device),
            stp_u=torch.full((B, N), U, device=self.device),
        )
    
    def create_batch_stimulus(self, theta_stim: torch.Tensor, amplitude: float = 20.0,
                               width: float = 0.3, noise_strength: float = 0.0) -> torch.Tensor:
        """Create batch stimuli with optional noise.
        
        Args:
            theta_stim: (batch,) stimulus orientations in degrees
            amplitude: Stimulus amplitude (α_ext or α_cue)
            width: Stimulus width in radians (a_ext or a_cue)
            noise_strength: Noise strength (µ_sti or µ_cue)
            
        Returns:
            (batch, N) stimulus array
        """
        dx = self.theta.unsqueeze(0) - theta_stim.unsqueeze(1)
        dx = torch.where(dx > 90, dx - 180, dx)
        dx = torch.where(dx < -90, dx + 180, dx)
        width_deg = width * 180 / math.pi
        stimulus = amplitude * torch.exp(-dx**2 / (2 * width_deg**2))
        
        # 添加刺激噪声 µ_sti 或 µ_cue
        if noise_strength > 0:
            noise = noise_strength * torch.randn_like(stimulus)
            stimulus = stimulus + noise
        
        return stimulus
    
    def step(self, state: BatchCANNState, I_ext: torch.Tensor) -> BatchCANNState:
        """Perform one simulation step for batch."""
        cfg = self.config
        u, r, stp_x, stp_u = state
        
        efficacy = stp_u * stp_x
        r_eff = r * efficacy
        
        r_fft = torch.fft.fft(r_eff, dim=1)
        recurrent = torch.fft.ifft(r_fft * self.kernel_fft.unsqueeze(0), dim=1).real
        
        # 添加神经元交互噪声 µJ
        if cfg.mu_J > 0:
            noise_J = cfg.mu_J * torch.randn_like(recurrent)
            recurrent = recurrent + noise_J
        
        # 添加背景噪声 µb
        background_noise = torch.zeros_like(I_ext)
        if cfg.mu_b > 0:
            background_noise = cfg.mu_b * torch.randn_like(I_ext)
        
        du = (-u + cfg.rho * recurrent + I_ext + background_noise) / cfg.tau
        u_new = u + du * cfg.dt
        
        u_pos = torch.clamp(u_new, min=0)
        u_sq = u_pos ** 2
        norm = 1.0 + cfg.k * cfg.rho * u_sq.sum(dim=1, keepdim=True)
        r_new = u_sq / norm
        
        # STP dynamics - tau_d/tau_f 是秒，dt 是毫秒，需要转换
        dt_sec = cfg.dt / 1000.0  # 毫秒转秒
        dx = (1.0 - stp_x) / cfg.tau_d - stp_u * stp_x * r
        stp_x_new = torch.clamp(stp_x + dx * dt_sec, 0.0, 1.0)
        
        du_stp = (cfg.U - stp_u) / cfg.tau_f + cfg.U * (1.0 - stp_u) * r
        stp_u_new = torch.clamp(stp_u + du_stp * dt_sec, 0.0, 1.0)
        
        return BatchCANNState(u=u_new, r=r_new, stp_x=stp_x_new, stp_u=stp_u_new)
    
    def run_phase(self, state: BatchCANNState, I_ext: torch.Tensor, n_steps: int) -> BatchCANNState:
        """Run phase with batch processing."""
        with torch.no_grad():
            for _ in range(n_steps):
                state = self.step(state, I_ext)
        return state
    
    def run_phase_record(self, state: BatchCANNState, I_ext: torch.Tensor, 
                          n_steps: int) -> Tuple[BatchCANNState, torch.Tensor]:
        """Run phase and record activity for decoding."""
        activity = []
        for _ in range(n_steps):
            state = self.step(state, I_ext)
            activity.append(state.r.clone())
        return state, torch.stack(activity, dim=1)
    
    def decode_orientation(self, r: torch.Tensor) -> torch.Tensor:
        """Decode from (batch, N) or (batch, T, N) activity."""
        if r.dim() == 3:
            r = r.mean(dim=1)
        
        theta_rad = self.theta * math.pi / 180
        cos_sum = (r * torch.cos(2 * theta_rad)).sum(dim=1)
        sin_sum = (r * torch.sin(2 * theta_rad)).sum(dim=1)
        
        perceived_rad = torch.atan2(sin_sum, cos_sum) / 2
        perceived = perceived_rad * 180 / math.pi
        
        perceived = torch.where(perceived >= 90, perceived - 180, perceived)
        perceived = torch.where(perceived < -90, perceived + 180, perceived)
        
        return perceived


# ============ Utility Functions ============

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def print_device_info():
    """Print device information."""
    print(f"\nPyTorch 配置:")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"  使用 CPU 后端")
