#!/usr/bin/env python3
"""
Reproduce Figure 2: Single-Layer CANN Experiments (JAX 加速版)
===============================================================

使用 JAX 向量化加速，比原版快 20-50x。

Generates:
- Fig 2A-C: STD-dominated CANN (repulsion effect)
- Fig 2D-F: STF-dominated CANN (attraction effect)

Usage:
    python scripts/run_fig2.py [--output_dir results/fig2] [--quick]
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.experiments.fast_single_layer import (
    run_fast_experiment_with_recording,
)
from src.experiments.fast_single_layer_optimized import (
    run_fast_experiment_optimized as run_fast_experiment,
)
from src.visualization.plots import (
    setup_figure_style,
    plot_fig2_single_layer,
    plot_neural_activity,
    plot_stp_dynamics,
    plot_adjustment_error,
)
from src.analysis.dog_fitting import fit_dog, compute_serial_bias


def main():
    parser = argparse.ArgumentParser(description='Reproduce Figure 2 (Fast Version)')
    parser.add_argument('--output_dir', type=str, default='results/fig2',
                        help='Output directory for figures')
    parser.add_argument('--n_runs', type=int, default=20,
                        help='Number of simulation runs')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of trials per run')
    parser.add_argument('--delta_step', type=float, default=1.0,
                        help='Delta step size (degrees)')
    parser.add_argument('--isi', type=float, default=1000.0,
                        help='Inter-stimulus interval (ms)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test mode (2 runs × 10 trials)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size for jax.vmap optimization (default: 10)')
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        args.n_runs = 2
        args.n_trials = 10
        args.delta_step = 10.0
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_figure_style()
    
    total_trials = 2 * args.n_runs * args.n_trials
    
    print("=" * 60)
    print("Figure 2: Single-Layer CANN Experiments (JAX 加速版)")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  Runs: {args.n_runs}")
    print(f"  Trials/Run: {args.n_trials}")
    print(f"  总 Trials: {total_trials}")
    print(f"  Delta 步长: {args.delta_step}°")
    print(f"  ISI: {args.isi} ms")
    print(f"  Batch size: {args.batch_size} (jax.vmap)")
    print(f"  输出目录: {output_dir}")
    
    total_start = time.time()
    
    # ========== STD-dominated Experiment (Fig 2A-C) ==========
    print("\n" + "=" * 60)
    print("[1/4] Running STD-dominated experiment (repulsion)...")
    print("=" * 60)
    
    std_results = run_fast_experiment(
        stp_type='std',
        n_runs=args.n_runs,
        n_trials_per_run=args.n_trials,
        delta_step=args.delta_step,
        isi=args.isi,
        seed=args.seed,
        verbose=True,
        batch_size=args.batch_size,
    )
    
    print(f"\n✅ STD 实验完成！耗时: {std_results['elapsed_time']:.1f} 秒")
    
    # ========== STD Recording ==========
    print("\n[2/4] Recording STD neural activity...")
    std_recording = run_fast_experiment_with_recording(
        stp_type='std',
        delta_to_record=-30.0,  # θ_s1=-30°, θ_s2=0° per paper
        isi=args.isi,
    )
    print("✅ STD 记录完成！")
    
    # ========== STF-dominated Experiment (Fig 2D-F) ==========
    print("\n" + "=" * 60)
    print("[3/4] Running STF-dominated experiment (attraction)...")
    print("=" * 60)
    
    stf_results = run_fast_experiment(
        stp_type='stf',
        n_runs=args.n_runs,
        n_trials_per_run=args.n_trials,
        delta_step=args.delta_step,
        isi=args.isi,
        seed=args.seed + 10000,
        verbose=True,
        batch_size=args.batch_size,
    )
    
    print(f"\n✅ STF 实验完成！耗时: {stf_results['elapsed_time']:.1f} 秒")
    
    # ========== STF Recording ==========
    print("\n[4/4] Recording STF neural activity...")
    stf_recording = run_fast_experiment_with_recording(
        stp_type='stf',
        delta_to_record=-30.0,  # θ_s1=-30°, θ_s2=0° per paper
        isi=args.isi,
    )
    print("✅ STF 记录完成！")
    
    # ========== Analysis ==========
    print("\n" + "=" * 60)
    print("Analysis Results")
    print("=" * 60)
    
    # STD results
    std_delta = std_results['trials_df']['delta']
    std_errors = std_results['trials_df']['error']
    std_bias = compute_serial_bias(std_delta, std_errors)
    print(f"\nSTD-dominated (Fig 2A-C):")
    print(f"  Effect type: {std_bias['effect_type']}")
    print(f"  DoG amplitude: {std_bias['dog_params'].amplitude:.2f}°")
    print(f"  DoG peak location: {std_bias['dog_params'].peak_location:.1f}°")
    print(f"  R²: {std_bias['dog_params'].r_squared:.3f}")
    
    # STF results
    stf_delta = stf_results['trials_df']['delta']
    stf_errors = stf_results['trials_df']['error']
    stf_bias = compute_serial_bias(stf_delta, stf_errors)
    print(f"\nSTF-dominated (Fig 2D-F):")
    print(f"  Effect type: {stf_bias['effect_type']}")
    print(f"  DoG amplitude: {stf_bias['dog_params'].amplitude:.2f}°")
    print(f"  DoG peak location: {stf_bias['dog_params'].peak_location:.1f}°")
    print(f"  R²: {stf_bias['dog_params'].r_squared:.3f}")
    
    # ========== Generate Figures ==========
    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)
    
    # Combine data for plotting
    std_plot_data = {
        'time': std_recording['timeseries']['time'],
        'activity': std_recording['timeseries']['r'],
        'stp_x': std_recording['timeseries']['stp_x'],
        'stp_u': std_recording['timeseries']['stp_u'],
        'theta': std_recording['theta'],
        'stim_neuron': std_recording['s1_neuron'],
        'delta': std_results['curve_binned']['delta'],
        'errors': std_results['curve_binned']['mean_error'],
    }
    
    stf_plot_data = {
        'time': stf_recording['timeseries']['time'],
        'activity': stf_recording['timeseries']['r'],
        'stp_x': stf_recording['timeseries']['stp_x'],
        'stp_u': stf_recording['timeseries']['stp_u'],
        'theta': stf_recording['theta'],
        'stim_neuron': stf_recording['s1_neuron'],
        'delta': stf_results['curve_binned']['delta'],
        'errors': stf_results['curve_binned']['mean_error'],
    }
    
    # Main figure
    fig = plot_fig2_single_layer(
        std_plot_data, stf_plot_data,
        save_path=output_dir / 'fig2_complete.png'
    )
    print(f"  Saved: {output_dir / 'fig2_complete.png'}")
    
    # Individual panels
    # STD neural activity
    fig_a, ax_a = plt.subplots(figsize=(8, 4))
    plot_neural_activity(
        std_recording['timeseries']['time'], std_recording['timeseries']['r'],
        std_recording['theta'], ax=ax_a, title='Fig 2A: STD Neural Activity'
    )
    fig_a.savefig(output_dir / 'fig2a_std_activity.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig2a_std_activity.png'}")
    plt.close(fig_a)
    
    # STD STP dynamics
    fig_b, ax_b = plt.subplots(figsize=(8, 3))
    plot_stp_dynamics(
        std_recording['timeseries']['time'], 
        std_recording['timeseries']['stp_x'], 
        std_recording['timeseries']['stp_u'],
        std_recording['s1_neuron'], ax=ax_b, title='Fig 2B: STD Dynamics'
    )
    fig_b.savefig(output_dir / 'fig2b_std_stp.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig2b_std_stp.png'}")
    plt.close(fig_b)
    
    # STD adjustment error
    fig_c, ax_c = plt.subplots(figsize=(6, 4))
    plot_adjustment_error(
        std_results['curve_binned']['delta'], 
        std_results['curve_binned']['mean_error'],
        ax=ax_c, title='Fig 2C: STD Adjustment Error', color='#E74C3C'
    )
    fig_c.savefig(output_dir / 'fig2c_std_error.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig2c_std_error.png'}")
    plt.close(fig_c)
    
    # STF panels
    fig_d, ax_d = plt.subplots(figsize=(8, 4))
    plot_neural_activity(
        stf_recording['timeseries']['time'], stf_recording['timeseries']['r'],
        stf_recording['theta'], ax=ax_d, title='Fig 2D: STF Neural Activity'
    )
    fig_d.savefig(output_dir / 'fig2d_stf_activity.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig2d_stf_activity.png'}")
    plt.close(fig_d)
    
    fig_e, ax_e = plt.subplots(figsize=(8, 3))
    plot_stp_dynamics(
        stf_recording['timeseries']['time'], 
        stf_recording['timeseries']['stp_x'], 
        stf_recording['timeseries']['stp_u'],
        stf_recording['s1_neuron'], ax=ax_e, title='Fig 2E: STF Dynamics'
    )
    fig_e.savefig(output_dir / 'fig2e_stf_stp.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig2e_stf_stp.png'}")
    plt.close(fig_e)
    
    fig_f, ax_f = plt.subplots(figsize=(6, 4))
    plot_adjustment_error(
        stf_results['curve_binned']['delta'], 
        stf_results['curve_binned']['mean_error'],
        ax=ax_f, title='Fig 2F: STF Adjustment Error', color='#3498DB'
    )
    fig_f.savefig(output_dir / 'fig2f_stf_error.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig2f_stf_error.png'}")
    plt.close(fig_f)
    
    # Save numerical results
    np.savez(
        output_dir / 'fig2_data.npz',
        std_delta=std_results['curve_binned']['delta'],
        std_errors=std_results['curve_binned']['mean_error'],
        std_errors_se=std_results['curve_binned']['se_error'],
        stf_delta=stf_results['curve_binned']['delta'],
        stf_errors=stf_results['curve_binned']['mean_error'],
        stf_errors_se=stf_results['curve_binned']['se_error'],
    )
    print(f"  Saved: {output_dir / 'fig2_data.npz'}")
    
    # Summary
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("Figure 2 reproduction complete!")
    print("=" * 60)
    print(f"  总耗时: {total_time:.1f} 秒")
    print(f"  总 Trials: {total_trials}")
    print(f"  平均速度: {total_trials/total_time:.1f} trials/秒")
    print(f"  输出目录: {output_dir}")
    
    if args.quick:
        # Estimate full experiment time
        full_trials = 2 * 20 * 100
        estimated_full_time = full_trials / (total_trials / total_time)
        print(f"\n  预计完整实验 (20 runs × 100 trials) 耗时: {estimated_full_time/60:.1f} 分钟")
        print("\n注意：这是快速测试模式。")
        print("完整实验请运行: python scripts/run_fig2.py")


if __name__ == '__main__':
    main()
