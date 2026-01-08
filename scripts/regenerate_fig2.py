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
)


def main():
    parser = argparse.ArgumentParser(description='Regenerate Figure 2 with new layout')
    parser.add_argument('--output_dir', type=str, default='results/fig2',
                        help='Output directory for figures')
    parser.add_argument('--output_name', type=str, default='fig2_new_layout.png',
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
    
    # Generate figure with new layout
    print("\n[2/2] Generating Figure with New Layout...")
    
    output_path = output_dir / args.output_name
    fig = plot_fig2_single_layer(
        std_plot_data, stf_plot_data,
        figsize=(16, 12),
        save_path=output_path
    )
    print(f"  ✅ Saved: {output_path}")
    
    plt.close(fig)
    
    print("\n" + "=" * 60)
    print("Figure 2 regeneration complete!")
    print("=" * 60)
    print(f"  Output: {output_path}")
    print("\nNew layout:")
    print("  [0,0] A: A1=Neural Activity + A2=STP x heatmap")
    print("  [0,1] B: STP dynamics (single neuron)")
    print("  [0,2] C: STD adjustment error curve")
    print("  [1,0] D: D1=Neural Activity + D2=STP u heatmap")
    print("  [1,1] E: STP dynamics (single neuron)")
    print("  [1,2] F: STF adjustment error curve")


if __name__ == '__main__':
    main()
