#!/usr/bin/env python3
"""
多核并行测试验证脚本
===================

验证 multiprocessing 实现是否真正使用了多个 CPU 核心。
测试 4 个关键指标：
1. 进程数量（是否启动了多个 worker 进程）
2. 系统 CPU 总使用率（是否接近 100%）
3. 加速比（多核 vs 单核）
4. 核心利用率（是否使用了多个核心）
"""

import sys
import time
import argparse
from pathlib import Path
import subprocess
import psutil

sys.path.insert(0, str(Path(__file__).parent))

from src.experiments.run_experiment_multi_process import run_experiment_multi_process
from src.experiments.fast_single_layer_optimized import run_fast_experiment_optimized


def check_process_count(process_name="python"):
    """指标 1: 检查进程数量"""
    procs = [p for p in psutil.process_iter() 
              if process_name in p.name() and 'run_fig2' in ' '.join(p.cmdline())]
    
    return {
        'process_count': len(procs),
        'pids': [p.pid for p in procs],
    }


def check_system_cpu():
    """指标 2: 检查系统 CPU 使用率"""
    cpu_percent = psutil.cpu_percent(interval=1.0)
    cpu_count = psutil.cpu_count(logical=True)
    
    # 计算使用的核心数（近似）
    used_cores = cpu_percent * cpu_count / 100.0
    
    return {
        'system_cpu_percent': cpu_percent,
        'cpu_count': cpu_count,
        'used_cores': used_cores,
    }


def run_and_monitor_single():
    """运行单核版本并监控"""
    print("\n" + "=" * 70)
    print("[1/2] 运行单核版本...")
    print("=" * 70)
    
    start_time = time.time()
    
    # 启动进程
    cmd = [
        sys.executable, 'scripts/run_fig2.py',
        '--n_runs', '2',
        '--n_trials', '10',
        '--batch_size', '10',
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 监控 CPU 使用
    max_cpu_usage = 0
    max_process_count = 0
    
    while proc.poll() is None:
        time.sleep(0.5)
        
        # 检查进程数
        proc_info = check_process_count()
        max_process_count = max(max_process_count, proc_info['process_count'])
        
        # 检查 CPU 使用
        cpu_info = check_system_cpu()
        max_cpu_usage = max(max_cpu_usage, cpu_info['system_cpu_percent'])
    
    elapsed_time = time.time() - start_time
    
    return {
        'elapsed_time': elapsed_time,
        'max_process_count': max_process_count,
        'max_cpu_usage': max_cpu_usage,
        'exit_code': proc.returncode,
    }


def run_and_monitor_multi(n_workers=12):
    """运行多核版本并监控"""
    print("\n" + "=" * 70)
    print("[2/2] 运行多核版本...")
    print("=" * 70)
    
    start_time = time.time()
    
    # 启动进程
    cmd = [
        sys.executable, 'scripts/run_fig2.py',
        '--n_runs', '2',
        '--n_trials', '10',
        '--batch_size', '10',
        '--use_multiprocess',
        '--n_workers', str(n_workers),
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 监控 CPU 使用
    max_cpu_usage = 0
    max_process_count = 0
    
    while proc.poll() is None:
        time.sleep(0.5)
        
        # 检查进程数
        proc_info = check_process_count()
        max_process_count = max(max_process_count, proc_info['process_count'])
        
        # 检查 CPU 使用
        cpu_info = check_system_cpu()
        max_cpu_usage = max(max_cpu_usage, cpu_info['system_cpu_percent'])
    
    elapsed_time = time.time() - start_time
    
    return {
        'elapsed_time': elapsed_time,
        'max_process_count': max_process_count,
        'max_cpu_usage': max_cpu_usage,
        'exit_code': proc.returncode,
        'n_workers': n_workers,
    }


def print_test_results(single_results, multi_results):
    """输出测试结果"""
    print("\n" + "=" * 70)
    print("多核并行测试结果")
    print("=" * 70)
    
    # ========== 指标 1: 进程数量 ==========
    print("\n[✅/❌] 指标 1: 进程数量")
    print(f"  期望: {multi_results['n_workers']} 个进程")
    print(f"  实际: {single_results['max_process_count']} 个进程（单核）")
    print(f"  实际: {multi_results['max_process_count']} 个进程（多核）")
    
    if multi_results['max_process_count'] >= multi_results['n_workers'] * 0.9:
        print(f"  结果: ✅ PASS（接近 {multi_results['n_workers']} 个进程）")
    else:
        print(f"  结果: ❌ FAIL（只有 {multi_results['max_process_count']} 个进程）")
    
    # ========== 指标 2: 系统 CPU 总使用率 ==========
    print("\n[✅/❌] 指标 2: 系统 CPU 总使用率")
    print(f"  期望: > 70%")
    print(f"  实际: {single_results['max_cpu_usage']:.1f}%（单核）")
    print(f"  实际: {multi_results['max_cpu_usage']:.1f}%（多核）")
    
    if multi_results['max_cpu_usage'] > 70:
        print(f"  结果: ✅ PASS（CPU 使用率高）")
    else:
        print(f"  结果: ❌ FAIL（CPU 使用率 {multi_results['max_cpu_usage']:.1f}% 太低）")
    
    # ========== 指标 3: 加速比 ==========
    print("\n[✅/❌] 指标 3: 加速比")
    speedup = single_results['elapsed_time'] / multi_results['elapsed_time']
    print(f"  期望: > 5x")
    print(f"  单核耗时: {single_results['elapsed_time']:.1f} 秒")
    print(f"  多核耗时: {multi_results['elapsed_time']:.1f} 秒")
    print(f"  加速比: {speedup:.2f}x")
    
    if speedup > 5:
        print(f"  结果: ✅ PASS（显著加速）")
    elif speedup > 1.5:
        print(f"  结果: ⚠️  PARTIAL（有一定加速）")
    else:
        print(f"  结果: ❌ FAIL（无明显加速）")
    
    # ========== 指标 4: 核心利用率 ==========
    cpu_count = psutil.cpu_count(logical=True)
    used_cores_single = single_results['max_cpu_usage'] * cpu_count / 100.0
    used_cores_multi = multi_results['max_cpu_usage'] * cpu_count / 100.0
    
    print("\n[✅/❌] 指标 4: 核心利用率")
    print(f"  期望: > 80%（使用 > {cpu_count * 0.8:.0f} 个核心）")
    print(f"  实际: {used_cores_single:.1f} 个核心（单核，{single_results['max_cpu_usage']:.1f}%）")
    print(f"  实际: {used_cores_multi:.1f} 个核心（多核，{multi_results['max_cpu_usage']:.1f}%）")
    
    if used_cores_multi > cpu_count * 0.8:
        print(f"  结果: ✅ PASS（接近满载）")
    else:
        print(f"  结果: ❌ FAIL（核心使用率 {used_cores_multi:.1f}/{cpu_count} 太低）")
    
    # ========== 总结 ==========
    print("\n" + "=" * 70)
    
    # 统计通过数
    pass_count = 0
    
    # 指标 1
    if multi_results['max_process_count'] >= multi_results['n_workers'] * 0.9:
        pass_count += 1
        print("指标 1: ✅ PASS")
    else:
        print("指标 1: ❌ FAIL")
    
    # 指标 2
    if multi_results['max_cpu_usage'] > 70:
        pass_count += 1
        print("指标 2: ✅ PASS")
    else:
        print("指标 2: ❌ FAIL")
    
    # 指标 3
    if speedup > 5:
        pass_count += 1
        print("指标 3: ✅ PASS")
    else:
        print("指标 3: ❌ FAIL")
    
    # 指标 4
    if used_cores_multi > cpu_count * 0.8:
        pass_count += 1
        print("指标 4: ✅ PASS")
    else:
        print("指标 4: ❌ FAIL")
    
    # 最终结果
    print("\n" + "=" * 70)
    if pass_count == 4:
        print("最终结果: ✅ 多核并行实现成功！")
        print(f"加速比: {speedup:.2f}x")
    elif pass_count >= 2:
        print("最终结果: ⚠️  部分成功")
        print(f"通过 {pass_count}/4 个指标")
    else:
        print("最终结果: ❌ 多核并行实现失败")
    
    print("=" * 70)
    
    return pass_count == 4


def main():
    parser = argparse.ArgumentParser(description='Test multi-core parallel implementation')
    parser.add_argument('--n_workers', type=int, default=12,
                        help='Number of worker processes (default: 12)')
    parser.add_argument('--skip_single', action='store_true',
                        help='Skip single-core test')
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("多核并行验证测试")
    print("=" * 70)
    print(f"\n测试配置:")
    print(f"  Worker 进程数: {args.n_workers}")
    print(f"  测试规模: 2 runs × 10 trials = 20 trials")
    print(f"  Batch size: 10 (jax.vmap)")
    print(f"  CPU 核心数: {psutil.cpu_count(logical=True)}")
    
    # 运行单核测试
    single_results = None
    if not args.skip_single:
        single_results = run_and_monitor_single()
    
    # 运行多核测试
    multi_results = run_and_monitor_multi(n_workers=args.n_workers)
    
    # 输出结果
    if single_results:
        print_test_results(single_results, multi_results)
    else:
        print("\n单核测试已跳过。")
        print(f"\n多核测试结果:")
        print(f"  进程数: {multi_results['max_process_count']}")
        print(f"  最大 CPU 使用: {multi_results['max_cpu_usage']:.1f}%")
        print(f"  耗时: {multi_results['elapsed_time']:.1f} 秒")


if __name__ == '__main__':
    main()

