#!/usr/bin/env python3
"""
Reproduce Figure 3: Two-Layer CANN Experiments
===============================================

Generates:
- Fig 3A-B: Model architecture and layer activities
- Fig 3C: Within-trial serial dependence (repulsion)
- Fig 3D: Between-trial serial dependence (attraction)

Usage:
    python scripts/run_fig3.py [--output_dir results/fig3]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.experiments.two_layer_exp import (
    TwoLayerExperimentConfig,
    run_within_trial_experiment,
    run_between_trial_experiment,
    run_with_recording,
)
from src.visualization.plots import (
    setup_figure_style,
    plot_fig3_two_layer,
    plot_adjustment_error,
)
from src.analysis.dog_fitting import compute_serial_bias


def main():
    parser = argparse.ArgumentParser(description='Reproduce Figure 3')
    parser.add_argument('--output_dir', type=str, default='results/fig3',
                        help='Output directory')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of trials per condition')
    parser.add_argument('--isi', type=float, default=1000.0,
                        help='Inter-stimulus interval (ms)')
    parser.add_argument('--iti', type=float, default=3000.0,
                        help='Inter-trial interval (ms)')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_figure_style()
    
    print("=" * 60)
    print("Figure 3: Two-Layer CANN Experiments")
    print("=" * 60)
    
    config = TwoLayerExperimentConfig(
        n_trials=args.n_trials,
        isi=args.isi,
        iti=args.iti,
    )
    
    # ========== Within-Trial Experiment ==========
    print("\n[1/3] Running within-trial experiment (repulsion)...")
    within_results = run_within_trial_experiment(config, verbose=True)
    
    # ========== Between-Trial Experiment ==========
    print("\n[2/3] Running between-trial experiment (attraction)...")
    between_results = run_between_trial_experiment(config, verbose=True)
    
    # ========== Record Activity ==========
    print("\n[3/3] Recording layer activities...")
    recording = run_with_recording(config, delta=30.0)
    
    # ========== Analysis ==========
    print("\n" + "=" * 60)
    print("Analysis Results")
    print("=" * 60)
    
    within_bias = compute_serial_bias(within_results['delta'], within_results['errors'])
    print(f"\nWithin-Trial (Fig 3C):")
    print(f"  Effect type: {within_bias['effect_type']}")
    print(f"  DoG amplitude: {within_bias['dog_params'].amplitude:.2f}°")
    print(f"  DoG peak: {within_bias['dog_params'].peak_location:.1f}°")
    print(f"  R²: {within_bias['dog_params'].r_squared:.3f}")
    
    between_bias = compute_serial_bias(between_results['delta'], between_results['errors'])
    print(f"\nBetween-Trial (Fig 3D):")
    print(f"  Effect type: {between_bias['effect_type']}")
    print(f"  DoG amplitude: {between_bias['dog_params'].amplitude:.2f}°")
    print(f"  DoG peak: {between_bias['dog_params'].peak_location:.1f}°")
    print(f"  R²: {between_bias['dog_params'].r_squared:.3f}")
    
    # ========== Generate Figures ==========
    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)
    
    within_plot = {
        'delta': within_results['delta'],
        'errors': within_results['errors'],
        'low_activity': recording['low_activity'],
        'high_activity': recording['high_activity'],
    }
    
    between_plot = {
        'delta': between_results['delta'],
        'errors': between_results['errors'],
    }
    
    # Main figure
    fig = plot_fig3_two_layer(
        within_plot, between_plot,
        save_path=output_dir / 'fig3_complete.png'
    )
    print(f"  Saved: {output_dir / 'fig3_complete.png'}")
    
    # Individual panels
    fig_c, ax_c = plt.subplots(figsize=(6, 4))
    plot_adjustment_error(
        within_results['delta'], within_results['errors'],
        ax=ax_c, title='Fig 3C: Within-Trial Bias (Repulsion)',
        color='#E74C3C'
    )
    fig_c.savefig(output_dir / 'fig3c_within_trial.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig3c_within_trial.png'}")
    plt.close(fig_c)
    
    fig_d, ax_d = plt.subplots(figsize=(6, 4))
    plot_adjustment_error(
        between_results['delta'], between_results['errors'],
        ax=ax_d, title='Fig 3D: Between-Trial Bias (Attraction)',
        color='#3498DB'
    )
    fig_d.savefig(output_dir / 'fig3d_between_trial.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig3d_between_trial.png'}")
    plt.close(fig_d)
    
    # Layer activity comparison
    fig_ab, ax_ab = plt.subplots(figsize=(8, 4))
    t = np.arange(len(recording['low_activity']))
    ax_ab.plot(t, np.max(recording['low_activity'], axis=1),
               'b-', label='Lower Layer (V1)', linewidth=2)
    ax_ab.plot(t, np.max(recording['high_activity'], axis=1),
               'r-', label='Higher Layer (PFC)', linewidth=2)
    ax_ab.set_xlabel('Time Step')
    ax_ab.set_ylabel('Peak Activity')
    ax_ab.set_title('Fig 3B: Layer Activities')
    ax_ab.legend()
    fig_ab.savefig(output_dir / 'fig3b_layer_activity.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig3b_layer_activity.png'}")
    plt.close(fig_ab)
    
    # Save data
    np.savez(
        output_dir / 'fig3_data.npz',
        within_delta=within_results['delta'],
        within_errors=within_results['errors'],
        between_delta=between_results['delta'],
        between_errors=between_results['errors'],
    )
    print(f"  Saved: {output_dir / 'fig3_data.npz'}")
    
    print("\n" + "=" * 60)
    print("Figure 3 reproduction complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

