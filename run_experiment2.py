#!/usr/bin/env python3
"""
运行实验2：单层 CANN 实验（JAX 向量化加速版本）
================================================

使用 jax.lax.scan 实现向量化时间演化，大幅提升计算速度。
预期速度提升: 20-50x

用法：
    python run_experiment2.py [--quick] [--n_runs N] [--n_trials N]
    
示例：
    python run_experiment2.py --quick          # 快速测试（~10秒）
    python run_experiment2.py                  # 完整实验（~16分钟）
    python run_experiment2.py --n_runs 5       # 自定义 runs
"""

import sys
import argparse
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description='JAX 向量化加速 CANN 实验')
    parser.add_argument('--n_runs', type=int, default=20,
                        help='Simulation runs 数量（默认 20）')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='每个 run 的 trials 数量（默认 100）')
    parser.add_argument('--delta_step', type=float, default=1.0,
                        help='Delta 步长（度，默认 1.0）')
    parser.add_argument('--isi', type=float, default=1000.0,
                        help='ISI（毫秒，默认 1000）')
    parser.add_argument('--quick', action='store_true',
                        help='快速测试模式（2 runs × 10 trials）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()
    
    # 快速测试模式
    if args.quick:
        args.n_runs = 2
        args.n_trials = 10
        args.delta_step = 10.0
    
    print("=" * 70)
    print("实验2：单层 CANN 实验（JAX 向量化加速版本）")
    print("=" * 70)
    
    # 导入实验模块
    from src.experiments.fast_single_layer import run_fast_experiment
    
    total_trials = 2 * args.n_runs * args.n_trials
    print(f"\n配置:")
    print(f"  Runs: {args.n_runs}")
    print(f"  Trials/Run: {args.n_trials}")
    print(f"  总 Trials: {total_trials}")
    print(f"  Delta 步长: {args.delta_step}°")
    print(f"  ISI: {args.isi} ms")
    
    total_start = time.time()
    
    # 运行 STD 实验
    print("\n" + "=" * 70)
    print("[1/2] STD 实验（排斥效应）")
    print("=" * 70)
    
    std_results = run_fast_experiment(
        stp_type='std',
        n_runs=args.n_runs,
        n_trials_per_run=args.n_trials,
        delta_step=args.delta_step,
        isi=args.isi,
        seed=args.seed,
        verbose=True,
    )
    
    print(f"\n✅ STD 实验完成！")
    print(f"   Delta 范围: [{std_results['curve_binned']['delta'].min():.1f}°, "
          f"{std_results['curve_binned']['delta'].max():.1f}°]")
    print(f"   平均误差范围: [{std_results['curve_binned']['mean_error'].min():.2f}°, "
          f"{std_results['curve_binned']['mean_error'].max():.2f}°]")
    print(f"   DoG 幅度: {std_results['dog_fit']['amplitude']:.2f}°")
    print(f"   DoG σ: {std_results['dog_fit']['sigma']:.2f}°")
    print(f"   耗时: {std_results['elapsed_time']:.1f} 秒")
    
    # 运行 STF 实验
    print("\n" + "=" * 70)
    print("[2/2] STF 实验（吸引效应）")
    print("=" * 70)
    
    stf_results = run_fast_experiment(
        stp_type='stf',
        n_runs=args.n_runs,
        n_trials_per_run=args.n_trials,
        delta_step=args.delta_step,
        isi=args.isi,
        seed=args.seed + 10000,
        verbose=True,
    )
    
    print(f"\n✅ STF 实验完成！")
    print(f"   Delta 范围: [{stf_results['curve_binned']['delta'].min():.1f}°, "
          f"{stf_results['curve_binned']['delta'].max():.1f}°]")
    print(f"   平均误差范围: [{stf_results['curve_binned']['mean_error'].min():.2f}°, "
          f"{stf_results['curve_binned']['mean_error'].max():.2f}°]")
    print(f"   DoG 幅度: {stf_results['dog_fit']['amplitude']:.2f}°")
    print(f"   DoG σ: {stf_results['dog_fit']['sigma']:.2f}°")
    print(f"   耗时: {stf_results['elapsed_time']:.1f} 秒")
    
    # 总结
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("✅ 所有实验完成！")
    print("=" * 70)
    print(f"  总耗时: {total_time:.1f} 秒")
    print(f"  总 Trials: {total_trials}")
    print(f"  平均速度: {total_trials/total_time:.1f} trials/秒")
    
    # 性能估计
    if args.quick:
        # 估算完整实验时间
        full_trials = 2 * 20 * 100
        estimated_full_time = full_trials / (total_trials / total_time)
        print(f"\n  预计完整实验 (20 runs × 100 trials) 耗时: {estimated_full_time/60:.1f} 分钟")
        print("\n注意：这是快速测试模式。")
        print("完整实验请运行: python run_experiment2.py")
    
    return std_results, stf_results


if __name__ == '__main__':
    main()

