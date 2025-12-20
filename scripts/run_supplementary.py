#!/usr/bin/env python3
"""
Reproduce Supplementary Figures
================================

Generates:
- Fig S5: Decoder comparison (PVM, COM, ML, Peak)
- Fig S6: Reversed layer order experiment
- Fig S7: Parameter sensitivity analysis
- Fig S8: Synaptic heterogeneity analysis

Usage:
    python scripts/run_supplementary.py [--output_dir results/supplementary]
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.experiments.supplementary_exp import (
    run_decoder_comparison,
    run_reversed_layer_experiment,
    run_parameter_sensitivity,
    run_heterogeneity_experiment,
    run_all_supplementary,
)
from src.visualization.plots import setup_figure_style, plot_adjustment_error
from src.analysis.dog_fitting import fit_dog


def main():
    parser = argparse.ArgumentParser(description='Reproduce Supplementary Figures')
    parser.add_argument('--output_dir', type=str, default='results/supplementary',
                        help='Output directory')
    parser.add_argument('--n_trials', type=int, default=10,
                        help='Number of trials per condition')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_figure_style()
    
    print("=" * 60)
    print("Supplementary Experiments")
    print("=" * 60)
    
    # ========== Fig S5: Decoder Comparison ==========
    print("\n[1/4] Running Decoder Comparison (Fig S5)...")
    decoder_results = run_decoder_comparison(n_trials=args.n_trials, verbose=True)
    
    fig_s5, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = {'pvm': '#3498DB', 'com': '#2ECC71', 'ml': '#E74C3C', 'peak': '#9B59B6'}
    titles = {'pvm': 'A. Population Vector', 'com': 'B. Center of Mass',
              'ml': 'C. Maximum Likelihood', 'peak': 'D. Peak Decoding'}
    
    for ax, (method, data) in zip(axes.flatten(), decoder_results.items()):
        plot_adjustment_error(
            data['delta'], data['errors'],
            ax=ax, title=titles[method], color=colors[method]
        )
    
    plt.tight_layout()
    fig_s5.savefig(output_dir / 'fig_s5_decoder_comparison.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig_s5_decoder_comparison.png'}")
    plt.close(fig_s5)
    
    # ========== Fig S6: Reversed Layer Order ==========
    print("\n[2/4] Running Reversed Layer Order (Fig S6)...")
    reversed_results = run_reversed_layer_experiment(n_trials=args.n_trials, verbose=True)
    
    fig_s6, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    plot_adjustment_error(
        reversed_results['within_trial']['delta'],
        reversed_results['within_trial']['errors'],
        ax=axes[0], title='A. Within-Trial (Reversed)',
        color='#3498DB'
    )
    axes[0].set_title('A. Within-Trial\n(STF-lower, STD-higher → Attraction)')
    
    plot_adjustment_error(
        reversed_results['between_trial']['delta'],
        reversed_results['between_trial']['errors'],
        ax=axes[1], title='B. Between-Trial (Reversed)',
        color='#3498DB'
    )
    axes[1].set_title('B. Between-Trial\n(STF-lower, STD-higher → Attraction)')
    
    plt.tight_layout()
    fig_s6.savefig(output_dir / 'fig_s6_reversed_layers.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig_s6_reversed_layers.png'}")
    plt.close(fig_s6)
    
    # ========== Fig S7: Parameter Sensitivity ==========
    print("\n[3/4] Running Parameter Sensitivity (Fig S7)...")
    sensitivity_results = run_parameter_sensitivity(
        param_sets=[], n_trials=args.n_trials, verbose=True
    )
    
    fig_s7, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # STD variations
    ax_std = axes[0]
    for name, data in sensitivity_results.items():
        if 'STD' in name:
            ax_std.plot(data['delta'], data['errors'], 'o-', 
                       label=f"τd={data['tau_d']}, τf={data['tau_f']}")
    ax_std.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax_std.set_xlabel('Δθ (°)')
    ax_std.set_ylabel('Adjustment Error (°)')
    ax_std.set_title('A. STD Parameter Variations')
    ax_std.legend()
    
    # STF variations
    ax_stf = axes[1]
    for name, data in sensitivity_results.items():
        if 'STF' in name:
            ax_stf.plot(data['delta'], data['errors'], 's-',
                       label=f"τd={data['tau_d']}, τf={data['tau_f']}")
    ax_stf.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax_stf.set_xlabel('Δθ (°)')
    ax_stf.set_ylabel('Adjustment Error (°)')
    ax_stf.set_title('B. STF Parameter Variations')
    ax_stf.legend()
    
    plt.tight_layout()
    fig_s7.savefig(output_dir / 'fig_s7_parameter_sensitivity.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig_s7_parameter_sensitivity.png'}")
    plt.close(fig_s7)
    
    # ========== Fig S8: Synaptic Heterogeneity ==========
    print("\n[4/4] Running Synaptic Heterogeneity (Fig S8)...")
    hetero_std = run_heterogeneity_experiment(
        heterogeneity_level=0.1, stp_type='std', n_trials=args.n_trials, verbose=True
    )
    hetero_stf = run_heterogeneity_experiment(
        heterogeneity_level=0.1, stp_type='stf', n_trials=args.n_trials, verbose=True
    )
    
    fig_s8, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # STD heterogeneity
    axes[0].plot(hetero_std['homogeneous']['delta'], hetero_std['homogeneous']['errors'],
                 'o-', color='#E74C3C', label='Uniform', linewidth=2)
    axes[0].plot(hetero_std['heterogeneous']['delta'], hetero_std['heterogeneous']['errors'],
                 's--', color='#E74C3C', alpha=0.7, label='±10% Variation', linewidth=2)
    axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    axes[0].set_xlabel('Δθ (°)')
    axes[0].set_ylabel('Adjustment Error (°)')
    axes[0].set_title('A. STD Heterogeneity')
    axes[0].legend()
    
    # STF heterogeneity
    axes[1].plot(hetero_stf['homogeneous']['delta'], hetero_stf['homogeneous']['errors'],
                 'o-', color='#3498DB', label='Uniform', linewidth=2)
    axes[1].plot(hetero_stf['heterogeneous']['delta'], hetero_stf['heterogeneous']['errors'],
                 's--', color='#3498DB', alpha=0.7, label='±10% Variation', linewidth=2)
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    axes[1].set_xlabel('Δθ (°)')
    axes[1].set_ylabel('Adjustment Error (°)')
    axes[1].set_title('B. STF Heterogeneity')
    axes[1].legend()
    
    plt.tight_layout()
    fig_s8.savefig(output_dir / 'fig_s8_heterogeneity.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig_s8_heterogeneity.png'}")
    plt.close(fig_s8)
    
    # Save all data
    np.savez(
        output_dir / 'supplementary_data.npz',
        decoder_results=decoder_results,
        reversed_results=reversed_results,
        sensitivity_results=sensitivity_results,
        hetero_std=hetero_std,
        hetero_stf=hetero_stf,
    )
    print(f"  Saved: {output_dir / 'supplementary_data.npz'}")
    
    print("\n" + "=" * 60)
    print("Supplementary experiments complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

