#!/usr/bin/env python3
"""
Regenerate Figure 2 with New Layout
====================================

This script regenerates Figure 2 using saved complete data from fig2_complete_data.npz.
No need to re-run experiments - just loads data and generates new figure.

The new layout has A and D each containing two vertically stacked subplots:
- A1/D1: Neural activity heatmap
- A2/D2: STP variable heatmap (x for STD, u for STF)

Usage:
    python scripts/regenerate_fig2.py [--output_dir results/fig2]
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.plots import (
    setup_figure_style,
    plot_fig2_single_layer,
    plot_fig2_panel_BE,  # New function for correct Fig.2B/E
)


def main():
    parser = argparse.ArgumentParser(description='Regenerate Figure 2 with new layout')
    parser.add_argument('--output_dir', type=str, default='results/fig2',
                        help='Output directory for figures')
    parser.add_argument('--output_name', type=str, default='fig2_complete.png',
                        help='Output filename')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    setup_figure_style()
    
    print("=" * 60)
    print("Regenerate Figure 2 with New Layout")
    print("=" * 60)
    
    # Try to load complete data first
    complete_data_path = output_dir / 'fig2_complete_data.npz'
    simple_data_path = output_dir / 'fig2_data.npz'
    
    if complete_data_path.exists():
        print(f"\n[1/2] Loading complete data from {complete_data_path}...")
        data = np.load(complete_data_path)
        
        # Prepare STD plot data
        std_plot_data = {
            'time': data['std_time'],
            'activity': data['std_activity'],
            'stp_x': data['std_stp_x'],
            'stp_u': data['std_stp_u'],
            'theta': data['std_theta'],
            'stim_neuron': int(data['std_stim_neuron']),
            'delta': data['std_delta'],
            'errors': data['std_errors'],
        }
        
        # Prepare STF plot data
        stf_plot_data = {
            'time': data['stf_time'],
            'activity': data['stf_activity'],
            'stp_x': data['stf_stp_x'],
            'stp_u': data['stf_stp_u'],
            'theta': data['stf_theta'],
            'stim_neuron': int(data['stf_stim_neuron']),
            'delta': data['stf_delta'],
            'errors': data['stf_errors'],
        }
        
        print("  ✅ Complete data loaded successfully!")
        
    elif simple_data_path.exists():
        print(f"\n⚠️  Complete data not found: {complete_data_path}")
        print(f"    Only error curve data available: {simple_data_path}")
        print("\n    Please re-run the experiment to save complete data:")
        print("    python scripts/run_fig2.py")
        return
    else:
        print(f"\n❌ No data files found in {output_dir}")
        print("    Please run the experiment first:")
        print("    python scripts/run_fig2.py")
        return
    
    # Reconstruct recording format for plot_fig2_panel_BE
    # The recording dict needs: timeseries, theta, theta_s1, theta_s2
    # Per paper: θ_s1=-30°, θ_s2=0° (delta = -30°)
    theta_arr = data['std_theta']
    s1_neuron = int(data['std_stim_neuron'])
    s2_neuron = 90  # θ_s2=0° corresponds to neuron 90 in 180-neuron network
    
    # Convert neuron indices to theta values
    theta_s1 = theta_arr[s1_neuron] if s1_neuron < len(theta_arr) else -30.0
    theta_s2 = theta_arr[s2_neuron] if s2_neuron < len(theta_arr) else 0.0
    
    std_recording = {
        'timeseries': {
            'time': data['std_time'],
            'r': data['std_activity'],
            'stp_x': data['std_stp_x'],
            'stp_u': data['std_stp_u'],
        },
        'theta': theta_arr,
        's1_neuron': s1_neuron,
        's2_neuron': s2_neuron,
        'theta_s1': theta_s1,
        'theta_s2': theta_s2,
    }
    
    stf_recording = {
        'timeseries': {
            'time': data['stf_time'],
            'r': data['stf_activity'],
            'stp_x': data['stf_stp_x'],
            'stp_u': data['stf_stp_u'],
        },
        'theta': data['stf_theta'],
        's1_neuron': int(data['stf_stim_neuron']),
        's2_neuron': s2_neuron,
        'theta_s1': theta_s1,
        'theta_s2': theta_s2,
    }
    
    # Generate figure with new layout
    print("\n[2/4] Generating Main Figure with New Layout...")
    
    output_path = output_dir / args.output_name
    fig = plot_fig2_single_layer(
        std_plot_data, stf_plot_data,
        figsize=(16, 12),
        save_path=output_path
    )
    print(f"  ✅ Saved: {output_path}")
    plt.close(fig)
    
    # Generate correct Fig.2B panel (STD: Response + STP spatial)
    print("\n[3/4] Generating Fig.2B panel (STD: Response + x(θ) spatial)...")
    fig_b, (ax_b_top, ax_b_bottom) = plot_fig2_panel_BE(
        std_recording,
        stp_type='std',
        title_prefix='B',
        figsize=(8, 6),
        save_path=output_dir / 'fig2b_std_stp.png'
    )
    print(f"  ✅ Saved: {output_dir / 'fig2b_std_stp.png'}")
    plt.close(fig_b)
    
    # Generate correct Fig.2E panel (STF: Response + STP spatial)
    print("\n[4/4] Generating Fig.2E panel (STF: Response + u(θ) spatial)...")
    fig_e, (ax_e_top, ax_e_bottom) = plot_fig2_panel_BE(
        stf_recording,
        stp_type='stf',
        title_prefix='E',
        figsize=(8, 6),
        save_path=output_dir / 'fig2e_stf_stp.png'
    )
    print(f"  ✅ Saved: {output_dir / 'fig2e_stf_stp.png'}")
    plt.close(fig_e)
    
    print("\n" + "=" * 60)
    print("Figure 2 regeneration complete!")
    print("=" * 60)
    print(f"  Output directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  - {args.output_name}: Main complete figure")
    print(f"  - fig2b_std_stp.png: Fig.2B (Response + x(θ) spatial)")
    print(f"  - fig2e_stf_stp.png: Fig.2E (Response + u(θ) spatial)")
    print("\nFig.2B/E layout (per paper requirements):")
    print("  Top: Neural response during cue period")
    print("  Bottom: STP spatial distribution at delay end")
    print("    - STD (Fig.2B): x(θ) showing depletion near θ_s1")
    print("    - STF (Fig.2E): u(θ) showing facilitation near θ_s1")


if __name__ == '__main__':
    main()
