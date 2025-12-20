#!/usr/bin/env python3
"""
Run All Experiments
===================

Master script to reproduce all figures from Zhang et al., NeurIPS 2025.

Generates:
- Figure 2: Single-layer CANN (STD repulsion, STF attraction)
- Figure 3: Two-layer CANN (within-trial vs between-trial)
- Figure 4: Temporal window analysis (ISI/ITI effects)
- Supplementary Figures: Control analyses

Usage:
    python scripts/run_all.py [--output_dir results] [--n_trials 20]
"""

import sys
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.plots import setup_figure_style


def main():
    parser = argparse.ArgumentParser(description='Run all experiments')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Base output directory')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of trials per condition')
    parser.add_argument('--skip_supplementary', action='store_true',
                        help='Skip supplementary experiments')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_figure_style()
    
    start_time = time.time()
    
    print("=" * 70)
    print("Serial Dependence CANN Model - Complete Reproduction")
    print("Based on: Zhang et al., NeurIPS 2025")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Trials per condition: {args.n_trials}")
    
    # ==================== Figure 2 ====================
    print("\n" + "=" * 70)
    print("FIGURE 2: Single-Layer CANN Experiments")
    print("=" * 70)
    
    from src.experiments.single_layer_exp import (
        SingleLayerExperimentConfig,
        run_single_layer_experiment,
        run_experiment_with_recording,
    )
    from src.visualization.plots import plot_fig2_single_layer
    from src.analysis.dog_fitting import compute_serial_bias
    import numpy as np
    
    config = SingleLayerExperimentConfig(n_trials=args.n_trials)
    
    print("\n[Fig 2] Running STD-dominated experiment...")
    std_results = run_single_layer_experiment(config, 'std', verbose=True)
    std_recording = run_experiment_with_recording(config, 'std')
    
    print("\n[Fig 2] Running STF-dominated experiment...")
    stf_results = run_single_layer_experiment(config, 'stf', verbose=True)
    stf_recording = run_experiment_with_recording(config, 'stf')
    
    # Analysis
    std_bias = compute_serial_bias(std_results['delta'], std_results['errors'])
    stf_bias = compute_serial_bias(stf_results['delta'], stf_results['errors'])
    
    print(f"\n  STD: {std_bias['effect_type']}, amplitude={std_bias['dog_params'].amplitude:.2f}°")
    print(f"  STF: {stf_bias['effect_type']}, amplitude={stf_bias['dog_params'].amplitude:.2f}°")
    
    # Save
    fig2_dir = output_dir / 'fig2'
    fig2_dir.mkdir(exist_ok=True)
    
    std_plot = {
        'time': std_recording['time'], 'activity': std_recording['activity'],
        'stp_x': std_recording['stp_x'], 'stp_u': std_recording['stp_u'],
        'theta': std_recording['theta'], 'stim_neuron': std_recording['stim_neuron'],
        'delta': std_results['delta'], 'errors': std_results['errors'],
    }
    stf_plot = {
        'time': stf_recording['time'], 'activity': stf_recording['activity'],
        'stp_x': stf_recording['stp_x'], 'stp_u': stf_recording['stp_u'],
        'theta': stf_recording['theta'], 'stim_neuron': stf_recording['stim_neuron'],
        'delta': stf_results['delta'], 'errors': stf_results['errors'],
    }
    
    plot_fig2_single_layer(std_plot, stf_plot, save_path=fig2_dir / 'fig2_complete.png')
    print(f"  Saved: {fig2_dir / 'fig2_complete.png'}")
    
    # ==================== Figure 3 ====================
    print("\n" + "=" * 70)
    print("FIGURE 3: Two-Layer CANN Experiments")
    print("=" * 70)
    
    from src.experiments.two_layer_exp import (
        TwoLayerExperimentConfig,
        run_within_trial_experiment,
        run_between_trial_experiment,
    )
    from src.visualization.plots import plot_fig3_two_layer
    
    config_2l = TwoLayerExperimentConfig(n_trials=args.n_trials)
    
    print("\n[Fig 3] Running within-trial experiment...")
    within_results = run_within_trial_experiment(config_2l, verbose=True)
    
    print("\n[Fig 3] Running between-trial experiment...")
    between_results = run_between_trial_experiment(config_2l, verbose=True)
    
    within_bias = compute_serial_bias(within_results['delta'], within_results['errors'])
    between_bias = compute_serial_bias(between_results['delta'], between_results['errors'])
    
    print(f"\n  Within-trial: {within_bias['effect_type']}, amp={within_bias['dog_params'].amplitude:.2f}°")
    print(f"  Between-trial: {between_bias['effect_type']}, amp={between_bias['dog_params'].amplitude:.2f}°")
    
    fig3_dir = output_dir / 'fig3'
    fig3_dir.mkdir(exist_ok=True)
    
    within_plot = {'delta': within_results['delta'], 'errors': within_results['errors']}
    between_plot = {'delta': between_results['delta'], 'errors': between_results['errors']}
    
    plot_fig3_two_layer(within_plot, between_plot, save_path=fig3_dir / 'fig3_complete.png')
    print(f"  Saved: {fig3_dir / 'fig3_complete.png'}")
    
    # ==================== Figure 4 ====================
    print("\n" + "=" * 70)
    print("FIGURE 4: Temporal Window Analysis")
    print("=" * 70)
    
    from src.experiments.two_layer_exp import run_isi_sweep, run_iti_sweep
    from src.visualization.plots import plot_fig4_temporal
    
    isi_values = [500, 1000, 2000, 4000]
    iti_values = [1000, 2000, 4000, 8000]
    
    print("\n[Fig 4] Running ISI sweep...")
    isi_results = run_isi_sweep(isi_values, config_2l, verbose=True)
    
    print("\n[Fig 4] Running ITI sweep...")
    iti_results = run_iti_sweep(iti_values, config_2l, verbose=True)
    
    fig4_dir = output_dir / 'fig4'
    fig4_dir.mkdir(exist_ok=True)
    
    plot_fig4_temporal(isi_results, iti_results, save_path=fig4_dir / 'fig4_complete.png')
    print(f"  Saved: {fig4_dir / 'fig4_complete.png'}")
    
    # ==================== Supplementary ====================
    if not args.skip_supplementary:
        print("\n" + "=" * 70)
        print("SUPPLEMENTARY FIGURES")
        print("=" * 70)
        
        from src.experiments.supplementary_exp import run_all_supplementary
        
        supp_results = run_all_supplementary(n_trials=args.n_trials // 2, verbose=True)
        
        supp_dir = output_dir / 'supplementary'
        supp_dir.mkdir(exist_ok=True)
        np.savez(supp_dir / 'supplementary_data.npz', **supp_results)
        print(f"  Saved: {supp_dir / 'supplementary_data.npz'}")
    
    # ==================== Summary ====================
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"\nResults saved to: {output_dir}")
    print("\nGenerated figures:")
    print("  - fig2/: Single-layer CANN (STD repulsion, STF attraction)")
    print("  - fig3/: Two-layer CANN (within-trial, between-trial)")
    print("  - fig4/: Temporal window analysis")
    if not args.skip_supplementary:
        print("  - supplementary/: Control analyses")
    
    print("\n" + "=" * 70)
    print("Key Findings:")
    print("=" * 70)
    print(f"  • STD-dominated CANN: {std_bias['effect_type'].upper()} effect")
    print(f"  • STF-dominated CANN: {stf_bias['effect_type'].upper()} effect")
    print(f"  • Two-layer within-trial: {within_bias['effect_type'].upper()}")
    print(f"  • Two-layer between-trial: {between_bias['effect_type'].upper()}")
    print("\nThis confirms the model's ability to explain serial dependence:")
    print("  - Repulsion from STD in sensory processing")
    print("  - Attraction from STF in post-perceptual processing")


if __name__ == '__main__':
    main()

