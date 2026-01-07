"""
Fast Single-Layer CANN Experiments using PyTorch GPU
=====================================================

使用 PyTorch 实现 GPU 加速的单层 CANN 实验。
支持批量并行和单试次录制两种模式。
"""

import numpy as np
import torch
import time
from typing import Dict, Tuple, Optional
from pathlib import Path

from ..models.cann_torch import (
    CANNConfig,
    TrialConfig,
    BatchCANN,
    BatchCANNState,
)


# ============ Batch Experiment Functions ============

def run_batch_trials(
    model: BatchCANN,
    theta_s1: torch.Tensor,
    theta_s2: torch.Tensor,
    trial_config: TrialConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run batch of trials, return (perceived, error).
    
    Args:
        model: BatchCANN model instance
        theta_s1: First stimulus orientations (batch,)
        theta_s2: Second stimulus orientations (batch,)
        trial_config: Trial timing configuration
        
    Returns:
        (perceived, error) tensors
    """
    cfg = model.config
    dt = cfg.dt
    
    n_s1 = int(trial_config.s1_duration / dt)
    n_isi = int(trial_config.isi / dt)
    n_s2 = int(trial_config.s2_duration / dt)
    n_delay = int(trial_config.delay / dt)
    n_cue = int(trial_config.cue_duration / dt)
    
    # Create stimuli with noise (Table 1: µ_sti=0.5, µ_cue=1.0)
    I_s1 = model.create_batch_stimulus(
        theta_s1, trial_config.alpha_ext, trial_config.a_ext, trial_config.mu_sti
    )
    I_s2 = model.create_batch_stimulus(
        theta_s2, trial_config.alpha_ext, trial_config.a_ext, trial_config.mu_sti
    )
    I_cue = model.create_batch_stimulus(
        torch.zeros(model.batch_size, device=model.device),
        trial_config.alpha_cue, trial_config.a_cue, trial_config.mu_cue
    )
    I_zero = torch.zeros(model.batch_size, cfg.N, device=model.device)
    
    # Initialize
    state = model.init_state()
    
    # Run phases
    state = model.run_phase(state, I_s1, n_s1)
    state = model.run_phase(state, I_zero, n_isi)
    state = model.run_phase(state, I_s2, n_s2)
    state = model.run_phase(state, I_zero, n_delay)
    state, cue_activity = model.run_phase_record(state, I_cue, n_cue)
    
    # Decode
    perceived = model.decode_orientation(cue_activity)
    
    # Compute error
    error = perceived - theta_s2
    error = torch.where(error > 90, error - 180, error)
    error = torch.where(error < -90, error + 180, error)
    
    return perceived, error


def run_experiment(
    stp_type: str = 'std',
    n_runs: int = 20,
    n_trials_per_run: int = 100,
    delta_range: Tuple[float, float] = (-90.0, 90.0),
    delta_step: float = 1.0,
    isi: float = 1000.0,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict:
    """Run complete experiment with batch processing.
    
    Args:
        stp_type: 'std' for STD-dominated, 'stf' for STF-dominated
        n_runs: Number of simulation runs
        n_trials_per_run: Trials per run
        delta_range: (min, max) delta values in degrees
        delta_step: Step size for delta
        isi: Inter-stimulus interval in ms
        seed: Random seed
        device: PyTorch device (auto-detect if None)
        verbose: Show progress info
        
    Returns:
        Dictionary with experiment results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_time = time.time()
    
    config = CANNConfig.std_dominated() if stp_type == 'std' else CANNConfig.stf_dominated()
    trial_config = TrialConfig(isi=isi)
    
    deltas = np.arange(delta_range[0], delta_range[1] + delta_step, delta_step)
    total_trials = n_runs * n_trials_per_run
    
    if verbose:
        print(f"\n{stp_type.upper()} 实验:")
        print(f"  设备: {device}")
        print(f"  总试次: {total_trials}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    all_deltas = np.random.choice(deltas, size=total_trials)
    all_theta_s2 = np.zeros(total_trials)
    all_theta_s1 = all_theta_s2 + all_deltas
    
    all_theta_s1 = np.where(all_theta_s1 >= 90, all_theta_s1 - 180, all_theta_s1)
    all_theta_s1 = np.where(all_theta_s1 < -90, all_theta_s1 + 180, all_theta_s1)
    
    theta_s1_batch = torch.tensor(all_theta_s1, device=device, dtype=torch.float32)
    theta_s2_batch = torch.tensor(all_theta_s2, device=device, dtype=torch.float32)
    
    model = BatchCANN(config, batch_size=total_trials, device=device)
    
    if verbose:
        print(f"  运行批量模拟...")
    
    perceived, errors = run_batch_trials(model, theta_s1_batch, theta_s2_batch, trial_config)
    
    perceived_np = perceived.cpu().numpy()
    errors_np = errors.cpu().numpy()
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"  完成! 耗时: {elapsed:.1f}s ({total_trials/elapsed:.0f} trials/s)")
    
    run_ids = np.repeat(np.arange(n_runs), n_trials_per_run)
    trial_ids = np.tile(np.arange(n_trials_per_run), n_runs)
    
    trials_df = {
        'run_id': run_ids,
        'trial_id': trial_ids,
        'theta_s1': all_theta_s1,
        'theta_s2': all_theta_s2,
        'delta': all_deltas,
        'perceived': perceived_np,
        'error': errors_np,
    }
    
    unique_deltas = np.unique(all_deltas)
    mean_errors = []
    se_errors = []
    
    for d in unique_deltas:
        mask = np.abs(all_deltas - d) < 0.5
        run_means = []
        for run_id in range(n_runs):
            run_mask = mask & (run_ids == run_id)
            if np.any(run_mask):
                run_means.append(np.mean(errors_np[run_mask]))
        
        run_means = np.array(run_means)
        mean_errors.append(np.mean(run_means))
        se_errors.append(np.std(run_means) / np.sqrt(len(run_means)) if len(run_means) > 0 else 0)
    
    curve_binned = {
        'delta': np.array(unique_deltas),
        'mean_error': np.array(mean_errors),
        'se_error': np.array(se_errors),
    }
    
    return {
        'stp_type': stp_type,
        'n_runs': n_runs,
        'n_trials_per_run': n_trials_per_run,
        'elapsed_time': elapsed,
        'trials_df': trials_df,
        'curve_binned': curve_binned,
    }


# ============ Single Trial Recording ============

def run_single_trial_with_recording(
    stp_type: str = 'std',
    delta: float = -30.0,
    isi: float = 1000.0,
    device: Optional[torch.device] = None,
) -> Dict:
    """Run a single trial with neural activity recording.
    
    Args:
        stp_type: 'std' for STD-dominated, 'stf' for STF-dominated
        delta: Stimulus difference in degrees
        isi: Inter-stimulus interval in ms
        device: PyTorch device (auto-detect if None)
        
    Returns:
        Dictionary with timeseries and results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = CANNConfig.std_dominated() if stp_type == 'std' else CANNConfig.stf_dominated()
    trial_config = TrialConfig(isi=isi)
    
    theta_s2 = 0.0
    theta_s1 = theta_s2 + delta
    if theta_s1 >= 90:
        theta_s1 -= 180
    elif theta_s1 < -90:
        theta_s1 += 180
    
    model = BatchCANN(config, batch_size=1, device=device)
    
    theta_s1_t = torch.tensor([theta_s1], device=device, dtype=torch.float32)
    theta_s2_t = torch.tensor([theta_s2], device=device, dtype=torch.float32)
    
    dt = config.dt
    n_s1 = int(trial_config.s1_duration / dt)
    n_isi = int(trial_config.isi / dt)
    n_s2 = int(trial_config.s2_duration / dt)
    n_delay = int(trial_config.delay / dt)
    n_cue = int(trial_config.cue_duration / dt)
    
    I_s1 = model.create_batch_stimulus(theta_s1_t, trial_config.alpha_ext, trial_config.a_ext)
    I_s2 = model.create_batch_stimulus(theta_s2_t, trial_config.alpha_ext, trial_config.a_ext)
    I_cue = model.create_batch_stimulus(
        torch.zeros(1, device=device), trial_config.alpha_cue, trial_config.a_cue
    )
    I_zero = torch.zeros(1, config.N, device=device)
    
    all_time = []
    all_r = []
    all_stp_x = []
    all_stp_u = []
    
    state = model.init_state()
    t = 0.0
    
    # Phase 1: S1
    for _ in range(n_s1):
        state = model.step(state, I_s1)
        t += dt
        all_time.append(t)
        all_r.append(state.r[0].cpu().numpy().copy())
        all_stp_x.append(state.stp_x[0].cpu().numpy().copy())
        all_stp_u.append(state.stp_u[0].cpu().numpy().copy())
    
    # Phase 2: ISI (sparse)
    for i in range(n_isi):
        state = model.step(state, I_zero)
        t += dt
        if i % 10 == 0:
            all_time.append(t)
            all_r.append(state.r[0].cpu().numpy().copy())
            all_stp_x.append(state.stp_x[0].cpu().numpy().copy())
            all_stp_u.append(state.stp_u[0].cpu().numpy().copy())
    
    # Phase 3: S2
    for _ in range(n_s2):
        state = model.step(state, I_s2)
        t += dt
        all_time.append(t)
        all_r.append(state.r[0].cpu().numpy().copy())
        all_stp_x.append(state.stp_x[0].cpu().numpy().copy())
        all_stp_u.append(state.stp_u[0].cpu().numpy().copy())
    
    # Phase 4: Delay (sparse)
    for i in range(n_delay):
        state = model.step(state, I_zero)
        t += dt
        if i % 100 == 0:
            all_time.append(t)
            all_r.append(state.r[0].cpu().numpy().copy())
            all_stp_x.append(state.stp_x[0].cpu().numpy().copy())
            all_stp_u.append(state.stp_u[0].cpu().numpy().copy())
    
    # Phase 5: Cue
    cue_activity = []
    for _ in range(n_cue):
        state = model.step(state, I_cue)
        t += dt
        cue_activity.append(state.r[0].cpu().numpy().copy())
        all_time.append(t)
        all_r.append(state.r[0].cpu().numpy().copy())
        all_stp_x.append(state.stp_x[0].cpu().numpy().copy())
        all_stp_u.append(state.stp_u[0].cpu().numpy().copy())
    
    # Decode
    mean_activity = np.mean(cue_activity, axis=0)
    theta_np = model.theta.cpu().numpy()
    theta_rad = theta_np * np.pi / 180
    cos_sum = np.sum(mean_activity * np.cos(2 * theta_rad))
    sin_sum = np.sum(mean_activity * np.sin(2 * theta_rad))
    perceived_rad = np.arctan2(sin_sum, cos_sum) / 2
    perceived = perceived_rad * 180 / np.pi
    
    if perceived >= 90:
        perceived -= 180
    elif perceived < -90:
        perceived += 180
    
    error = perceived - theta_s2
    if error > 90:
        error -= 180
    elif error < -90:
        error += 180
    
    s1_neuron = int(np.argmin(np.abs(theta_np - theta_s1)))
    
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
        'perceived': perceived,
        'error': error,
        'delta': delta,
    }
