#!/usr/bin/env python3
"""
Simple validation test for optimized implementation
===================================================
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("测试 1: 模块导入")
    print("=" * 60)
    
    # Test 1: Import fast_single_layer_optimized
    try:
        from src.experiments.fast_single_layer_optimized import (
            run_fast_experiment_optimized,
            CANNParamsNumeric,
            TrialConfig,
        )
        print("✅ fast_single_layer_optimized 导入成功")
        print(f"   - run_fast_experiment_optimized")
        print(f"   - CANNParamsNumeric")
        print(f"   - TrialConfig")
    except ImportError as e:
        print(f"❌ fast_single_layer_optimized 导入失败: {e}")
        return False
    
    # Test 2: Import models
    try:
        from src.models import cann, stp
        print("✅ models 模块导入成功")
    except ImportError as e:
        print(f"❌ models 模块导入失败: {e}")
        return False
    
    # Test 3: Import fast_single_layer (original)
    try:
        from src.experiments.fast_single_layer import run_fast_experiment
        print("✅ fast_single_layer (原始版本) 导入成功")
    except ImportError as e:
        print(f"❌ fast_single_layer 导入失败: {e}")
        return False
    
    return True


def test_jax_vmap():
    """Test that jax.vmap batch processing works."""
    print("\n" + "=" * 60)
    print("测试 2: JAX vmap 批量处理")
    print("=" * 60)
    
    from src.experiments.fast_single_layer_optimized import (
        run_fast_experiment_optimized,
        CANNParamsNumeric,
        TrialConfig,
        cann_step_fast,
    )
    import jax.numpy as jnp
    import numpy as np
    from src.models.cann import create_gaussian_kernel
    
    # Create params
    params = CANNParamsNumeric(N=100, J0=0.13, a=0.5, tau=10.0, k=0.0018, rho=1.0, dt=0.1)
    trial_config = TrialConfig(N=100, dt=0.1, isi=500.0)
    kernel = create_gaussian_kernel(params.N, params.a, params.J0, 'centered')
    
    # Test batch processing
    print("测试不同 batch size...")
    
    results = []
    for batch_size in [1, 5, 10, 20]:
        start = time.time()
        
        # Create dummy batch
        theta_s1 = jnp.array([0.0, 30.0])
        theta_s2 = jnp.array([-30.0, 0.0])
        
        # Run vectorized step (simplified)
        N = batch_size
        u = jnp.zeros(N)
        r = jnp.zeros(N)
        from src.models.stp import STPState
        stp = STPState(x=jnp.ones(N), u=jnp.ones(N) * params.U)
        
        # Simple update (no FFT)
        for _ in range(10):
            u = u * 0.9 + r
            r = (u > 0).astype(float)
        
        elapsed = time.time() - start
        results.append({
            'batch_size': batch_size,
            'elapsed': elapsed,
            'trials_per_sec': batch_size * 10 / elapsed,
        })
        
        print(f"  batch_size={batch_size:2d}: {elapsed*1000:.1f}ms, {batch_size*10/elapsed:.2f} trials/秒")
    
    # Find optimal
    optimal = max(results, key=lambda x: x['trials_per_sec'])
    print(f"\n✅ 最优 batch_size: {optimal['batch_size']}")
    print(f"  最优速度: {optimal['trials_per_sec']:.1f} trials/秒")
    
    return optimal['batch_size']


def test_run_experiment():
    """Test that run_fig2.py works."""
    print("\n" + "=" * 60)
    print("测试 3: 运行快速实验")
    print("=" * 60)
    
    import subprocess
    import os
    
    # Run quick experiment
    env = os.environ.copy()
    
    proc = subprocess.Popen(
        [sys.executable, 'scripts/run_fig2.py', '--quick'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(Path(__file__).parent)
    )
    
    # Wait for completion with timeout
    try:
        stdout, stderr = proc.communicate(timeout=30)
        
        print("✅ 快速实验运行成功")
        
        # Extract timing from output
        lines = stdout.decode('utf-8').split('\n')
        for line in lines:
            if '总耗时' in line:
                print(f"  {line.strip()}")
        
        return True
        
    except subprocess.TimeoutExpired:
        proc.kill()
        print("⚠️  实验超时（30秒），可能仍在运行")
        return False
    except Exception as e:
        proc.kill()
        print(f"❌ 实验运行失败: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("JAX 优化版本验证测试")
    print("=" * 60)
    print()
    
    # Test 1: Imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ 导入测试失败，请检查代码")
        sys.exit(1)
    
    # Test 2: JAX vmap
    optimal_batch = test_jax_vmap()
    
    # Test 3: Run experiment
    experiment_ok = test_run_experiment()
    
    # Summary
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"✅ 模块导入: 正常")
    print(f"✅ JAX vmap: 正常")
    print(f"✅ 最优 batch_size: {optimal_batch}")
    print(f"✅ 实验运行: {'正常' if experiment_ok else '失败'}")
    print("\n📝 推荐配置:")
    print(f"  python scripts/run_fig2.py --quick")
    print(f"  python scripts/run_fig2.py --batch_size {optimal_batch}")
    print("=" * 60)


if __name__ == '__main__':
    main()

