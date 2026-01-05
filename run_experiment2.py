#!/usr/bin/env python3
"""
运行实验2：单层 CANN 实验（快速测试版本）
==========================================

这是一个快速测试版本，使用较少的 trials 来验证代码是否正常工作。
完整实验请使用 scripts/run_fig2.py
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.experiments.single_layer_exp import (
    SingleLayerExperimentConfig,
    run_single_layer_experiment,
    run_experiment_with_recording,
    print_validation_report,
)

def main():
    print("=" * 70)
    print("实验2：单层 CANN 实验（快速测试版本）")
    print("=" * 70)
    
    # 验证配置
    print("\n[步骤 1/4] 验证配置...")
    config_std = SingleLayerExperimentConfig(stp_type='std')
    print_validation_report(config_std)
    
    # 快速测试配置（减少 trials 以加快速度）
    print("\n[步骤 2/4] 运行快速测试（STD，2 runs × 10 trials）...")
    config_test = SingleLayerExperimentConfig(
        stp_type='std',
        n_runs=2,              # 减少到 2 runs（完整版是 20）
        n_trials_per_run=10,  # 减少到 10 trials（完整版是 100）
        delta_step=10.0,      # 增大步长以加快速度
    )
    
    try:
        std_results = run_single_layer_experiment(
            config_test, 
            stp_type='std', 
            verbose=True
        )
        
        print("\n✅ STD 实验完成！")
        print(f"   Delta 范围: [{std_results['curve_binned']['delta'].min():.1f}°, "
              f"{std_results['curve_binned']['delta'].max():.1f}°]")
        print(f"   平均误差范围: [{std_results['curve_binned']['mean_error'].min():.2f}°, "
              f"{std_results['curve_binned']['mean_error'].max():.2f}°]")
        print(f"   DoG 幅度: {std_results['dog_fit']['amplitude']:.2f}°")
        print(f"   DoG σ: {std_results['dog_fit']['sigma']:.2f}°")
        
    except Exception as e:
        print(f"\n❌ STD 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 记录单次试验（用于可视化）
    print("\n[步骤 3/4] 记录单次试验（用于可视化）...")
    try:
        recording = run_experiment_with_recording(
            stp_type='std',
            delta_to_record=-30.0  # θ_s1=-30°, θ_s2=0°
        )
        
        print("✅ 记录完成！")
        print(f"   时间序列长度: {len(recording['timeseries']['time'])} 点")
        print(f"   感知朝向: {recording['perceived']:.2f}°")
        print(f"   误差: {recording['error']:.2f}°")
        
    except Exception as e:
        print(f"\n❌ 记录失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 快速 STF 测试
    print("\n[步骤 4/4] 运行快速测试（STF，2 runs × 10 trials）...")
    config_stf = SingleLayerExperimentConfig(
        stp_type='stf',
        n_runs=2,
        n_trials_per_run=10,
        delta_step=10.0,
    )
    
    try:
        stf_results = run_single_layer_experiment(
            config_stf,
            stp_type='stf',
            verbose=True
        )
        
        print("\n✅ STF 实验完成！")
        print(f"   Delta 范围: [{stf_results['curve_binned']['delta'].min():.1f}°, "
              f"{stf_results['curve_binned']['delta'].max():.1f}°]")
        print(f"   平均误差范围: [{stf_results['curve_binned']['mean_error'].min():.2f}°, "
              f"{stf_results['curve_binned']['mean_error'].max():.2f}°]")
        print(f"   DoG 幅度: {stf_results['dog_fit']['amplitude']:.2f}°")
        print(f"   DoG σ: {stf_results['dog_fit']['sigma']:.2f}°")
        
    except Exception as e:
        print(f"\n❌ STF 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("✅ 所有实验完成！")
    print("=" * 70)
    print("\n注意：这是快速测试版本。")
    print("完整实验请使用: python scripts/run_fig2.py")
    print("完整实验配置: 20 runs × 100 trials，delta_step=1.0°")


if __name__ == '__main__':
    main()

