#!/usr/bin/env python3
"""
测试多核优化版本的性能
========================

对比原始版本 vs 优化版本的速度差异。
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.experiments.fast_single_layer import run_fast_experiment
from src.experiments.fast_single_layer_optimized import run_fast_experiment_optimized


def test_original(n_runs=2, n_trials=10, delta_step=10.0):
    """测试原始版本（单核）"""
    print("=" * 70)
    print("原始版本测试 (单核, 无 vmap)")
    print("=" * 70)
    
    start = time.time()
    std_result = run_fast_experiment(
        stp_type='std',
        n_runs=n_runs,
        n_trials_per_run=n_trials,
        delta_step=delta_step,
        verbose=True,
    )
    original_time = time.time() - start
    
    print(f"\n✅ 原始版本完成!")
    print(f"   耗时: {original_time:.1f} 秒")
    print(f"   平均速度: {n_runs*n_trials/original_time:.2f} trials/秒")
    
    return original_time, std_result


def test_optimized(n_runs=2, n_trials=10, delta_step=10.0, batch_size=5):
    """测试优化版本（vmap 批量并行）"""
    print("\n" + "=" * 70)
    print("优化版本测试 (jax.vmap 批量并行)")
    print("=" * 70)
    
    start = time.time()
    std_result = run_fast_experiment_optimized(
        stp_type='std',
        n_runs=n_runs,
        n_trials_per_run=n_trials,
        delta_step=delta_step,
        verbose=True,
        batch_size=batch_size,
    )
    optimized_time = time.time() - start
    
    print(f"\n✅ 优化版本完成!")
    print(f"   耗时: {optimized_time:.1f} 秒")
    print(f"   平均速度: {n_runs*n_trials/optimized_time:.2f} trials/秒")
    
    return optimized_time, std_result


def compare_results(original_result, optimized_result):
    """对比结果是否一致"""
    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)
    
    orig_curve = original_result['curve_binned']
    opt_curve = optimized_result['curve_binned']
    
    # 对比 DoG 拟合参数
    orig_dog = original_result['dog_fit']
    opt_dog = optimized_result['dog_fit']
    
    print(f"\nDoG 拟合参数对比:")
    print(f"  原始版本 - 幅度: {orig_dog['amplitude']:.4f}°, σ: {orig_dog['sigma']:.4f}°, R²: {orig_dog['r_squared']:.4f}")
    print(f"  优化版本 - 幅度: {opt_dog['amplitude']:.4f}°, σ: {opt_dog['sigma']:.4f}°, R²: {opt_dog['r_squared']:.4f}")
    
    # 计算差异
    amp_diff = abs(orig_dog['amplitude'] - opt_dog['amplitude'])
    sigma_diff = abs(orig_dog['sigma'] - opt_dog['sigma'])
    
    print(f"\n差异:")
    print(f"  幅度差异: {amp_diff:.6f}°")
    print(f"  σ 差异: {sigma_diff:.6f}°")
    
    if amp_diff < 1e-3 and sigma_diff < 1e-3:
        print(f"\n✅ 结果一致！（差异 < 0.001）")
    else:
        print(f"\n⚠️ 结果有差异，可能是由于随机性")
    
    return amp_diff < 1e-3 and sigma_diff < 1e-3


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='测试多核优化效果')
    parser.add_argument('--n_runs', type=int, default=2, help='运行次数')
    parser.add_argument('--n_trials', type=int, default=10, help='每次运行试验数')
    parser.add_argument('--delta_step', type=float, default=10.0, help='Delta 步长')
    parser.add_argument('--batch_size', type=int, default=5, help='JAX vmap 批量大小')
    parser.add_argument('--skip_original', action='store_true', help='跳过原始版本测试')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("多核优化性能测试")
    print("=" * 70)
    print(f"\n测试配置:")
    print(f"  Runs: {args.n_runs}")
    print(f"  Trials/Run: {args.n_trials}")
    print(f"  总 Trials: {args.n_runs * args.n_trials}")
    print(f"  Delta 步长: {args.delta_step}°")
    
    original_time = None
    original_result = None
    
    # 测试原始版本
    if not args.skip_original:
        original_time, original_result = test_original(
            args.n_runs, args.n_trials, args.delta_step
        )
    
    # 测试优化版本
    optimized_time, optimized_result = test_optimized(
        args.n_runs, args.n_trials, args.delta_step,
        args.batch_size
    )
    
    # 对比结果
    if original_result is not None:
        results_match = compare_results(original_result, optimized_result)
        
        # 对比速度
        speedup = original_time / optimized_time
        print("\n" + "=" * 70)
        print("性能对比")
        print("=" * 70)
        print(f"  原始版本耗时: {original_time:.1f} 秒")
        print(f"  优化版本耗时: {optimized_time:.1f} 秒")
        print(f"  加速比: {speedup:.2f}x")
        
        if speedup > 2.0:
            print(f"  ✅ 显著加速！")
        elif speedup > 1.2:
            print(f"  ⚠️ 有一定加速")
        else:
            print(f"  ⚠️ 加速不明显（可能是测试规模太小）")
        
        if results_match:
            print(f"\n🎉 测试通过！优化版本速度快 {speedup:.2f}x，且结果一致！")
    else:
        print(f"\n⚠️ 跳过了原始版本测试，无法对比速度")


if __name__ == '__main__':
    main()

