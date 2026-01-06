"""
Multi-process experiment runner for CANN simulations
=================================================

使用 multiprocessing.Pool 实现多核并行加速。
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from pathlib import Path
import sys

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .worker_multi_process import worker_run_batch, TrialConfig
from ..models.cann import create_gaussian_kernel
from ..analysis.dog_fitting import fit_dog


def create_params(stp_type: str) -> dict:
    """Create CANN parameters based on STP type (Table 1)."""
    if stp_type == 'std':
        return {
            'N': 100,
            'J0': 0.13,
            'a': 0.5,
            'tau': 10.0,
            'k': 0.0018,
            'rho': 1.0,
            'dt': 0.1,
            'tau_d': 280.0,
            'tau_f': 20.0,
            'U': 0.45,
        }
    else:  # stf
        return {
            'N': 100,
            'J0': 0.06,
            'a': 0.4,
            'tau': 10.0,
            'k': 0.005,
            'rho': 1.0,
            'dt': 0.1,
            'tau_d': 20.0,
            'tau_f': 620.0,
            'U': 0.08,
        }


def prepare_trials(
    n_runs: int,
    n_trials_per_run: int,
    delta_range: Tuple[float, float],
    delta_step: float,
    seed: int,
) -> List[Dict]:
    """Prepare all trial data."""
    np.random.seed(seed)
    
    deltas = np.arange(delta_range[0], delta_range[1] + delta_step, delta_step)
    all_trials = []
    
    for run_id in range(n_runs):
        for trial_id in range(n_trials_per_run):
            delta = np.random.choice(deltas)
            theta_s2 = 0.0  # Reference
            theta_s1 = theta_s2 + delta
            
            # Wrap to [-90, 90)
            if theta_s1 >= 90:
                theta_s1 -= 180
            elif theta_s1 < -90:
                theta_s1 += 180
            
            all_trials.append({
                'run_id': run_id,
                'trial_id': trial_id,
                'theta_s1': theta_s1,
                'theta_s2': theta_s2,
                'delta': delta,
            })
    
    return all_trials


def split_trials(all_trials: List[Dict], n_batches: int) -> List[List[Dict]]:
    """Split trials into batches for multiprocessing."""
    trials_per_batch = len(all_trials) // n_batches
    batches = []
    
    for i in range(n_batches):
        start_idx = i * trials_per_batch
        end_idx = start_idx + trials_per_batch if i < n_batches - 1 else len(all_trials)
        batches.append(all_trials[start_idx:end_idx])
    
    return batches


def run_experiment_multi_process(
    stp_type: str = 'std',
    n_runs: int = 20,
    n_trials_per_run: int = 100,
    delta_range: Tuple[float, float] = (-90.0, 90.0),
    delta_step: float = 1.0,
    isi: float = 1000.0,
    seed: int = 42,
    verbose: bool = True,
    n_workers: Optional[int] = None,
    batch_size_per_worker: int = 10,
) -> Dict:
    """Run fast single-layer experiment with multi-process optimization.
    
    Optimization strategy:
    - multiprocessing: Parallelize across CPU cores (process-level)
    - jax.vmap: Vectorize trials within each worker (SIMD-level)
    
    Args:
        stp_type: 'std' for STD-dominated, 'stf' for STF-dominated
        n_runs: Number of simulation runs
        n_trials_per_run: Trials per run
        delta_range: (min, max) delta values in degrees
        delta_step: Step size for delta
        isi: Inter-stimulus interval in ms
        seed: Random seed
        verbose: Show progress bar
        n_workers: Number of worker processes (default: all CPU cores)
        batch_size_per_worker: Batch size for jax.vmap in each worker
        
    Returns:
        Dictionary with experiment results
    """
    start_time = time.time()
    
    # Set parameters
    params_dict = create_params(stp_type)
    from ..models.cann import CANNParamsNumeric
    params = CANNParamsNumeric(**params_dict)
    
    trial_config = TrialConfig(
        N=params.N,
        dt=params.dt,
        isi=isi,
    )
    
    # Create kernel
    kernel_np = create_gaussian_kernel(
        params.N, params.a, params.J0, 'centered'
    )
    
    # Setup multiprocessing
    n_workers = n_workers or cpu_count()
    if verbose:
        print(f"JIT 编译中...")
        print(f"使用 {n_workers} 个 CPU 核心（多进程并行）")
        print(f"每个 Worker 的 batch_size: {batch_size_per_worker} (jax.vmap)")
        print(f"开始运行 {n_runs} runs × {n_trials_per_run} trials...")
    
    # Prepare trials
    total_trials = n_runs * n_trials_per_run
    all_trials = prepare_trials(
        n_runs, n_trials_per_run, delta_range, delta_step, seed
    )
    
    # Split trials into batches for each worker
    batches = split_trials(all_trials, n_workers)
    
    if verbose:
        print(f"  总 trials: {total_trials}")
        print(f"  每个 Worker 处理: {[len(b) for b in batches]} trials")
    
    # Run with multiprocessing
    from tqdm import tqdm
    all_results = []
    
    worker_args = [
        (batch, kernel_np, params, trial_config, batch_size_per_worker)
        for batch in batches
    ]
    
    with Pool(processes=n_workers) as pool:
        if verbose:
            results_iter = tqdm(
                pool.imap(worker_run_batch, worker_args),
                total=len(batches),
                desc=f'{stp_type.upper()} Workers',
            )
        else:
            results_iter = pool.imap(worker_run_batch, worker_args)
        
        for batch_results in results_iter:
            all_results.extend(batch_results)
    
    elapsed = time.time() - start_time
    
    # Convert to arrays
    trials_df = {k: np.array([t[k] for t in all_results]) for k in all_results[0].keys()}
    
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
        'optimization': {
            'n_workers': n_workers,
            'batch_size_per_worker': batch_size_per_worker,
            'total_cores': cpu_count(),
        },
    }


if __name__ == '__main__':
    # Test the multi-process runner
    import argparse
    
    parser = argparse.ArgumentParser(description='Test multi-process runner')
    parser.add_argument('--n_runs', type=int, default=2, help='Number of runs')
    parser.add_argument('--n_trials', type=int, default=10, help='Trials per run')
    parser.add_argument('--n_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size per worker')
    
    args = parser.parse_args()
    
    result = run_experiment_multi_process(
        stp_type='std',
        n_runs=args.n_runs,
        n_trials_per_run=args.n_trials,
        verbose=True,
        n_workers=args.n_workers,
        batch_size_per_worker=args.batch_size,
    )
    
    print(f"\n✅ 测试完成！")
    print(f"   耗时: {result['elapsed_time']:.1f} 秒")
    print(f"   Trials: {len(result['trials_df']['error'])}")
    print(f"   平均速度: {len(result['trials_df']['error'])/result['elapsed_time']:.1f} trials/秒")

