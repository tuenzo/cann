#!/usr/bin/env python3
"""
Reproduce Figure 2: Single-Layer CANN Experiments
==================================================

Generates:
- Fig 2A-C: STD-dominated CANN (repulsion effect)
- Fig 2D-F: STF-dominated CANN (attraction effect)

Usage:
    python scripts/run_fig2.py [--output_dir results/fig2]
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.experiments.single_layer_exp import (
    SingleLayerExperimentConfig,
    TrialTimeline,
    run_single_layer_experiment,
    run_experiment_with_recording,
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
    parser = argparse.ArgumentParser(description='Reproduce Figure 2')
    parser.add_argument('--output_dir', type=str, default='results/fig2',
                        help='Output directory for figures')
    parser.add_argument('--n_runs', type=int, default=20,
                        help='Number of simulation runs')
    parser.add_argument('--n_trials_per_run', type=int, default=100,
                        help='Number of trials per run')
    parser.add_argument('--isi', type=float, default=1000.0,
                        help='Inter-stimulus interval (ms)')
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_figure_style()
    
    print("=" * 60)
    print("Figure 2: Single-Layer CANN Experiments")
    print("=" * 60)
    
    # Configuration
    timeline = TrialTimeline(isi=args.isi)
    config = SingleLayerExperimentConfig(
        n_runs=args.n_runs,
        n_trials_per_run=args.n_trials_per_run,
        timeline=timeline,
    )
    
    # ========== STD-dominated Experiment (Fig 2A-C) ==========
    print("\n[1/4] Running STD-dominated experiment (repulsion)...")
    std_results = run_single_layer_experiment(config, stp_type='std', verbose=True)
    
    print("\n[2/4] Recording STD neural activity...")
    std_recording = run_experiment_with_recording(
        config=config, 
        stp_type='std', 
        delta_to_record=-30.0  # θ_s1=-30°, θ_s2=0° per paper
    )
    
    # ========== STF-dominated Experiment (Fig 2D-F) ==========
    print("\n[3/4] Running STF-dominated experiment (attraction)...")
    stf_results = run_single_layer_experiment(config, stp_type='stf', verbose=True)
    
    print("\n[4/4] Recording STF neural activity...")
    stf_recording = run_experiment_with_recording(
        config=config, 
        stp_type='stf', 
        delta_to_record=-30.0  # θ_s1=-30°, θ_s2=0° per paper
    )
    
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
    
    print("\n" + "=" * 60)
    print("Figure 2 reproduction complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

