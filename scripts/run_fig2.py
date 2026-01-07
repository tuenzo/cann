#!/usr/bin/env python3
"""
Figure 2: Complete Single-Layer CANN Experiments (PyTorch GPU)
================================================================

PyTorch GPU 批量并行版本，完整复现论文 Figure 2。

Generates:
- Fig 2A-C: STD-dominated CANN (repulsion effect)
- Fig 2D-F: STF-dominated CANN (attraction effect)

Usage:
    python scripts/run_fig2.py [--quick] [--n_runs 20] [--n_trials 100]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import time
import argparse
from pathlib import Path

# 从 src.experiments 导入实验函数
from src.experiments.fast_single_layer import (
    run_experiment,
    run_single_trial_with_recording,
)

# 从 src.visualization 导入绘图函数
from src.visualization.plots import (
    plot_fig2_single_layer,
    setup_figure_style,
)

# 从 src.analysis 导入分析函数
from src.analysis.dog_fitting import fit_dog


# ============ Figure Generation ============

def generate_fig2(std_results, stf_results, std_recording, stf_recording, save_path=None):
    """Generate complete Figure 2 using visualization module."""
    # 准备数据格式以匹配 plot_fig2_single_layer 接口
    std_data = {
        'time': std_recording['timeseries']['time'],
        'activity': std_recording['timeseries']['r'],
        'theta': std_recording['theta'],
        'stp_x': std_recording['timeseries']['stp_x'],
        'stp_u': std_recording['timeseries']['stp_u'],
        'stim_neuron': std_recording['s1_neuron'],
        'delta': std_results['curve_binned']['delta'],
        'errors': std_results['curve_binned']['mean_error'],
    }
    
    stf_data = {
        'time': stf_recording['timeseries']['time'],
        'activity': stf_recording['timeseries']['r'],
        'theta': stf_recording['theta'],
        'stp_x': stf_recording['timeseries']['stp_x'],
        'stp_u': stf_recording['timeseries']['stp_u'],
        'stim_neuron': stf_recording['s1_neuron'],
        'delta': stf_results['curve_binned']['delta'],
        'errors': stf_results['curve_binned']['mean_error'],
    }
    
    fig = plot_fig2_single_layer(std_data, stf_data, save_path=save_path)
    return fig


# ============ Main ============

def main():
    parser = argparse.ArgumentParser(description='Figure 2 - PyTorch GPU')
    parser.add_argument('--output_dir', type=str, default='results/fig2')
    parser.add_argument('--n_runs', type=int, default=20)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--delta_step', type=float, default=1.0)
    parser.add_argument('--isi', type=float, default=1000.0)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    if args.quick:
        args.n_runs = 2
        args.n_trials = 10
        args.delta_step = 10.0
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 60)
    print("Figure 2: Single-Layer CANN (PyTorch GPU)")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Runs: {args.n_runs}")
    print(f"  Trials/Run: {args.n_trials}")
    print(f"  Delta step: {args.delta_step} deg")
    print(f"  ISI: {args.isi} ms")
    print(f"  Output: {output_dir}")
    
    total_start = time.time()
    
    # STD Experiment
    print("\n" + "=" * 60)
    print("[1/4] STD-dominated experiment (repulsion)...")
    print("=" * 60)
    
    std_results = run_experiment(
        stp_type='std',
        n_runs=args.n_runs,
        n_trials_per_run=args.n_trials,
        delta_step=args.delta_step,
        isi=args.isi,
        seed=args.seed,
        device=device,
    )
    
    print("\n[2/4] Recording STD neural activity...")
    std_recording = run_single_trial_with_recording(
        stp_type='std', delta=-30.0, isi=args.isi, device=device
    )
    print("  Done!")
    
    # STF Experiment
    print("\n" + "=" * 60)
    print("[3/4] STF-dominated experiment (attraction)...")
    print("=" * 60)
    
    stf_results = run_experiment(
        stp_type='stf',
        n_runs=args.n_runs,
        n_trials_per_run=args.n_trials,
        delta_step=args.delta_step,
        isi=args.isi,
        seed=args.seed + 10000,
        device=device,
    )
    
    print("\n[4/4] Recording STF neural activity...")
    stf_recording = run_single_trial_with_recording(
        stp_type='stf', delta=-30.0, isi=args.isi, device=device
    )
    print("  Done!")
    
    # Analysis
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)
    
    # 使用 src.analysis.dog_fitting 的 fit_dog
    std_dog = fit_dog(std_results['curve_binned']['delta'], 
                      std_results['curve_binned']['mean_error'])
    stf_dog = fit_dog(stf_results['curve_binned']['delta'],
                      stf_results['curve_binned']['mean_error'])
    
    # DoGParams 有属性: amplitude, sigma, r_squared, peak_location
    std_effect = 'repulsion' if std_dog.amplitude > 0 else 'attraction'
    stf_effect = 'repulsion' if stf_dog.amplitude > 0 else 'attraction'
    
    print(f"\nSTD-dominated (Fig 2A-C):")
    print(f"  Effect: {std_effect}")
    print(f"  DoG amplitude: {std_dog.amplitude:.2f} deg")
    print(f"  DoG peak: {std_dog.peak_location:.1f} deg")
    print(f"  R2: {std_dog.r_squared:.3f}")
    
    print(f"\nSTF-dominated (Fig 2D-F):")
    print(f"  Effect: {stf_effect}")
    print(f"  DoG amplitude: {stf_dog.amplitude:.2f} deg")
    print(f"  DoG peak: {stf_dog.peak_location:.1f} deg")
    print(f"  R2: {stf_dog.r_squared:.3f}")
    
    # Generate Figures using visualization module
    print("\n" + "=" * 60)
    print("Generating figures")
    print("=" * 60)
    
    setup_figure_style()
    generate_fig2(
        std_results, stf_results, std_recording, stf_recording,
        save_path=str(output_dir / 'fig2_complete.png')
    )
    print(f"  Saved: {output_dir / 'fig2_complete.png'}")
    
    # Save data
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
    total_trials = 2 * args.n_runs * args.n_trials
    
    print("\n" + "=" * 60)
    print("Figure 2 Complete!")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Total time: {total_time:.1f} s")
    print(f"  Total trials: {total_trials}")
    print(f"  Speed: {total_trials/total_time:.1f} trials/s")
    print(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
