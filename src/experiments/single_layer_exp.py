"""
Single-Layer CANN Experiments (Lightweight Wrapper)
=================================================

这是一个轻量级包装器，直接使用 fast_single_layer_optimized 的优化实现。
所有优化逻辑都集中在 fast_single_layer_optimized.py 中。

主要功能：
- run_single_layer_experiment: 运行完整实验（使用 jax.vmap 优化）
- run_experiment_with_recording: 运行带记录的实验
- 提供向后兼容的别名

注意：所有优化细节请参见 fast_single_layer_optimized.py
"""

from typing import Dict, Tuple, List
import numpy as np

# 从优化版本导入（这是唯一的实际实现）
from .fast_single_layer_optimized import (
    run_fast_experiment_optimized,
    run_fast_experiment_with_recording,
)


# 向后兼容性：保留旧的导入别名
run_experiment_single = run_fast_experiment_optimized
run_experiment = run_fast_experiment_optimized
run_fast_experiment_optimized = run_fast_experiment_optimized


# 导出数据类（用于向后兼容）
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    """Single-layer experiment result."""
    trials_df: Dict
    curve_binned: Dict
    dog_fit: Dict
    elapsed_time: float
    stp_type: str
    n_runs: int
    n_trials_per_run: int


# 便利函数别名
def run_single_layer_experiment(
    stp_type: str = 'std',
    n_runs: int = 20,
    n_trials_per_run: int = 100,
    delta_range: Tuple[float, float] = (-90.0, 90.0),
    delta_step: float = 1.0,
    isi: float = 1000.0,
    seed: int = 42,
    verbose: bool = True,
    batch_size: int = 50,
) -> ExperimentResult:
    """Run single-layer CANN experiment (optimized version).
    
    Wrapper for fast_single_layer_optimized.run_fast_experiment_optimized.
    
    Args:
        stp_type: 'std' for STD-dominated, 'stf' for STF-dominated
        n_runs: Number of simulation runs
        n_trials_per_run: Trials per run
        delta_range: (min, max) delta values in degrees
        delta_step: Step size for delta
        isi: Inter-stimulus interval in ms
        seed: Random seed
        verbose: Show progress bar
        batch_size: Batch size for jax.vmap optimization (default: 50)
        
    Returns:
        ExperimentResult with trials data, curve fit, and timing
        
    Example:
        >>> result = run_single_layer_experiment(
        ...     stp_type='std', n_runs=2, n_trials_per_run=10
        ... )
        >>> print(f"STD 实验完成！耗时: {result.elapsed_time:.1f} 秒")
    """
    result = run_fast_experiment_optimized(
        stp_type=stp_type,
        n_runs=n_runs,
        n_trials_per_run=n_trials_per_run,
        delta_range=delta_range,
        delta_step=delta_step,
        isi=isi,
        seed=seed,
        verbose=verbose,
        batch_size=batch_size,
    )
    
    return ExperimentResult(
        trials_df=result['trials_df'],
        curve_binned=result['curve_binned'],
        dog_fit=result['dog_fit'],
        elapsed_time=result['elapsed_time'],
        stp_type=result['stp_type'],
        n_runs=result['n_runs'],
        n_trials_per_run=result['n_trials_per_run'],
    )


def run_experiment_with_recording(
    stp_type: str = 'std',
    delta_to_record: float = -30.0,
    isi: float = 1000.0,
) -> Dict:
    """Run a single trial with recording for visualization.
    
    Wrapper for fast_single_layer_optimized.run_fast_experiment_with_recording.
    
    Args:
        stp_type: 'std' or 'stf'
        delta_to_record: Delta value for the recorded trial (degrees)
        isi: Inter-stimulus interval in ms
        
    Returns:
        Dictionary with timeseries and results
        
    Example:
        >>> recording = run_experiment_with_recording(
        ...     stp_type='std', delta_to_record=-30.0
        ... )
        >>> print(f"神经活动记录在: {recording['timeseries']['r'].shape}")
    """
    result = run_fast_experiment_with_recording(
        stp_type=stp_type,
        delta_to_record=delta_to_record,
        isi=isi,
    )
    
    return result


if __name__ == '__main__':
    print("Single-Layer CANN Experiments (Lightweight Wrapper)")
    print("=" * 60)
    print("注意：此模块只是包装器，所有优化逻辑在 fast_single_layer_optimized.py 中")
    print("=" * 60)
    print()
    
    # 简单测试
    print("运行快速测试...")
    result = run_single_layer_experiment(
        stp_type='std',
        n_runs=2,
        n_trials_per_run=10,
        batch_size=5,
        verbose=True,
    )
    
    print(f"完成！耗时: {result.elapsed_time:.1f} 秒")
    print(f"Trials: {len(result.trials_df['error'])}")
