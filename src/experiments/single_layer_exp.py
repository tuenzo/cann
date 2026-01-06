"""
Single-Layer CANN Experiments (JAX Optimized Wrapper)
===================================================

轻量级包装器，直接使用 fast_single_layer_optimized 的优化实现（jax.vmap 批量并行）。

使用方式：
    from src.experiments.single_layer_exp import run_single_layer_experiment

预期性能：267x 加速（相比原始 Python 循环版本）
"""

# 直接从优化版本导入
from .fast_single_layer_optimized import (
    run_fast_experiment_optimized as run_single_layer_experiment,
    run_fast_experiment_with_recording as run_single_layer_recording,
)


if __name__ == '__main__':
    print("Single-Layer CANN Experiments (JAX Optimized Wrapper)")
    print("=" * 60)
    print()
    print("此模块是 fast_single_layer_optimized.py 的轻量级包装器")
    print("所有优化都通过 jax.vmap 实现批量并行")
    print("=" * 60)
    print()
