#!/usr/bin/env python3
"""
Reproduce Figure 4: Temporal Window Analysis
=============================================

Generates:
- Fig 4A: ISI effect on within-trial repulsion
- Fig 4B: ITI effect on between-trial attraction

Shows how serial dependence decays with temporal separation.

Usage:
    python scripts/run_fig4.py [--output_dir results/fig4]
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
    run_isi_sweep,
    run_iti_sweep,
)
from src.visualization.plots import setup_figure_style, plot_fig4_temporal


def main():
    parser = argparse.ArgumentParser(description='Reproduce Figure 4')
    parser.add_argument('--output_dir', type=str, default='results/fig4',
                        help='Output directory')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of trials per condition')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_figure_style()
    
    print("=" * 60)
    print("Figure 4: Temporal Window Analysis")
    print("=" * 60)
    
    config = TwoLayerExperimentConfig(n_trials=args.n_trials)
    
    # ISI values to test (in ms)
    isi_values = [250, 500, 1000, 2000, 4000, 8000]
    
    # ITI values to test (in ms)
    iti_values = [500, 1000, 2000, 4000, 8000, 16000]
    
    # ========== ISI Sweep (Within-Trial) ==========
    print("\n[1/2] Running ISI sweep (within-trial repulsion decay)...")
    isi_results = run_isi_sweep(isi_values, config, delta_test=30.0, verbose=True)
    
    # ========== ITI Sweep (Between-Trial) ==========
    print("\n[2/2] Running ITI sweep (between-trial attraction decay)...")
    iti_results = run_iti_sweep(iti_values, config, delta_test=30.0, verbose=True)
    
    # ========== Analysis ==========
    print("\n" + "=" * 60)
    print("Analysis Results")
    print("=" * 60)
    
    print("\nISI Effect on Within-Trial Repulsion:")
    for isi, bias in zip(isi_results['isi_values'], isi_results['bias']):
        print(f"  ISI={isi:5.0f}ms: bias={bias:+.2f}°")
    
    print("\nITI Effect on Between-Trial Attraction:")
    for iti, bias in zip(iti_results['iti_values'], iti_results['bias']):
        print(f"  ITI={iti:5.0f}ms: bias={bias:+.2f}°")
    
    # ========== Generate Figures ==========
    print("\n" + "=" * 60)
    print("Generating Figures")
    print("=" * 60)
    
    # Main figure
    fig = plot_fig4_temporal(
        isi_results, iti_results,
        save_path=output_dir / 'fig4_complete.png'
    )
    print(f"  Saved: {output_dir / 'fig4_complete.png'}")
    
    # Individual panels with error bars
    fig_a, ax_a = plt.subplots(figsize=(6, 4))
    ax_a.errorbar(
        isi_results['isi_values'], isi_results['bias'],
        yerr=isi_results['bias_std'],
        fmt='o-', color='#E74C3C', linewidth=2, markersize=8,
        capsize=4, capthick=1.5
    )
    ax_a.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax_a.set_xlabel('ISI (ms)')
    ax_a.set_ylabel('Within-Trial Bias (°)')
    ax_a.set_title('Fig 4A: ISI Effect on Repulsion')
    ax_a.set_xscale('log')
    fig_a.savefig(output_dir / 'fig4a_isi_sweep.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig4a_isi_sweep.png'}")
    plt.close(fig_a)
    
    fig_b, ax_b = plt.subplots(figsize=(6, 4))
    ax_b.errorbar(
        iti_results['iti_values'], iti_results['bias'],
        yerr=iti_results['bias_std'],
        fmt='s-', color='#3498DB', linewidth=2, markersize=8,
        capsize=4, capthick=1.5
    )
    ax_b.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax_b.set_xlabel('ITI (ms)')
    ax_b.set_ylabel('Between-Trial Bias (°)')
    ax_b.set_title('Fig 4B: ITI Effect on Attraction')
    ax_b.set_xscale('log')
    fig_b.savefig(output_dir / 'fig4b_iti_sweep.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig4b_iti_sweep.png'}")
    plt.close(fig_b)
    
    # Combined log-log plot
    fig_combined, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # ISI
    axes[0].plot(isi_results['isi_values'], np.abs(isi_results['bias']),
                 'o-', color='#E74C3C', linewidth=2, markersize=8)
    axes[0].set_xlabel('ISI (ms)')
    axes[0].set_ylabel('|Repulsion Bias| (°)')
    axes[0].set_xscale('log')
    axes[0].set_title('A. Repulsion Decay with ISI')
    axes[0].grid(True, alpha=0.3)
    
    # ITI
    axes[1].plot(iti_results['iti_values'], np.abs(iti_results['bias']),
                 's-', color='#3498DB', linewidth=2, markersize=8)
    axes[1].set_xlabel('ITI (ms)')
    axes[1].set_ylabel('|Attraction Bias| (°)')
    axes[1].set_xscale('log')
    axes[1].set_title('B. Attraction Decay with ITI')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_combined.savefig(output_dir / 'fig4_decay_analysis.png', bbox_inches='tight')
    print(f"  Saved: {output_dir / 'fig4_decay_analysis.png'}")
    plt.close(fig_combined)
    
    # Save data
    np.savez(
        output_dir / 'fig4_data.npz',
        isi_values=isi_results['isi_values'],
        isi_bias=isi_results['bias'],
        isi_bias_std=isi_results['bias_std'],
        iti_values=iti_results['iti_values'],
        iti_bias=iti_results['bias'],
        iti_bias_std=iti_results['bias_std'],
    )
    print(f"  Saved: {output_dir / 'fig4_data.npz'}")
    
    print("\n" + "=" * 60)
    print("Figure 4 reproduction complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

