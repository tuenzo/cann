"""
Visualization Module
====================

Comprehensive plotting functions for reproducing all figures from
Zhang et al., NeurIPS 2025.

Includes:
- Neural activity heatmaps
- STP dynamics plots
- Adjustment error curves with DoG fits
- Serial dependence analysis plots
"""

from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from ..analysis.dog_fitting import fit_dog, dog_function


def setup_figure_style():
    """Set up publication-quality figure style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_neural_activity(
    time: np.ndarray,
    activity: np.ndarray,
    theta: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Neural Activity",
    cmap: str = 'hot',
    vmax: Optional[float] = None,
) -> plt.Axes:
    """Plot neural activity heatmap over time.
    
    Args:
        time: Time points (ms), shape (T,)
        activity: Firing rates, shape (T, N)
        theta: Preferred orientations (degrees), shape (N,)
        ax: Matplotlib axes
        title: Plot title
        cmap: Colormap name
        vmax: Maximum value for colormap
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    
    if vmax is None:
        vmax = np.max(activity)
    
    im = ax.imshow(
        activity.T,
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], theta[0], theta[-1]],
        cmap=cmap,
        vmin=0,
        vmax=vmax,
    )
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Preferred Orientation (°)')
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Firing Rate')
    
    return ax


def plot_stp_dynamics(
    time: np.ndarray,
    stp_x: np.ndarray,
    stp_u: np.ndarray,
    neuron_idx: int,
    ax: Optional[plt.Axes] = None,
    title: str = "STP Dynamics",
) -> plt.Axes:
    """Plot STP variables over time for a single neuron.
    
    Args:
        time: Time points (ms)
        stp_x: Neurotransmitter availability, shape (T, N)
        stp_u: Release probability, shape (T, N)
        neuron_idx: Index of neuron to plot
        ax: Matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))
    
    ax.plot(time, stp_x[:, neuron_idx], 'b-', label='x (availability)', linewidth=2)
    ax.plot(time, stp_u[:, neuron_idx], 'r-', label='u (release prob.)', linewidth=2)
    ax.plot(time, stp_x[:, neuron_idx] * stp_u[:, neuron_idx], 
            'g--', label='u·x (efficacy)', linewidth=2)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('STP Variable')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.set_ylim(0, 1.1)
    
    return ax


def plot_adjustment_error(
    delta: np.ndarray,
    errors: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Adjustment Error",
    color: str = 'blue',
    fit_dog: bool = True,
    show_stats: bool = True,
) -> plt.Axes:
    """Plot adjustment error curve with DoG fit.
    
    Args:
        delta: Stimulus differences (S1 - S2) in degrees
        errors: Adjustment errors in degrees
        ax: Matplotlib axes
        title: Plot title
        color: Plot color
        fit_dog: Whether to fit and plot DoG curve
        show_stats: Whether to show statistics
        
    Returns:
        Matplotlib axes
    """
    from ..analysis.dog_fitting import fit_dog as do_fit, dog_function
    
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    
    # Scatter plot of data
    ax.scatter(delta, errors, c=color, alpha=0.6, s=30, label='Data')
    
    # Fit and plot DoG
    if fit_dog and len(delta) > 2:
        params = do_fit(delta, errors)
        
        # Generate smooth curve
        delta_smooth = np.linspace(delta.min(), delta.max(), 100)
        dog_curve = dog_function(delta_smooth, params.amplitude, params.sigma)
        
        ax.plot(delta_smooth, dog_curve, color=color, linewidth=2, 
                label=f'DoG fit (R²={params.r_squared:.2f})')
        
        if show_stats:
            # Add annotation
            effect = "Attraction" if params.amplitude > 0 else "Repulsion"
            ax.annotate(
                f'{effect}\nAmp: {params.amplitude:.2f}°\nPeak: {params.peak_location:.1f}°',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top',
                fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
    
    # Reference line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    
    ax.set_xlabel('Δθ (S1 - S2) (°)')
    ax.set_ylabel('Adjustment Error (°)')
    ax.set_title(title)
    ax.legend(loc='best')
    
    return ax


def plot_serial_dependence(
    within_delta: np.ndarray,
    within_errors: np.ndarray,
    between_delta: np.ndarray,
    between_errors: np.ndarray,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot within-trial and between-trial serial dependence.
    
    Args:
        within_delta: Within-trial stimulus differences
        within_errors: Within-trial errors
        between_delta: Between-trial stimulus differences  
        between_errors: Between-trial errors
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Within-trial (repulsion expected)
    plot_adjustment_error(
        within_delta, within_errors,
        ax=axes[0],
        title='Within-Trial Serial Dependence',
        color='#E74C3C',  # Red for repulsion
    )
    
    # Between-trial (attraction expected)
    plot_adjustment_error(
        between_delta, between_errors,
        ax=axes[1],
        title='Between-Trial Serial Dependence',
        color='#3498DB',  # Blue for attraction
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_temporal_window(
    isi_values: np.ndarray,
    iti_values: np.ndarray,
    bias_matrix: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Temporal Window of Serial Dependence",
) -> plt.Axes:
    """Plot bias as a function of ISI and ITI.
    
    Args:
        isi_values: ISI values (ms)
        iti_values: ITI values (ms)
        bias_matrix: Bias values, shape (len(isi), len(iti))
        ax: Matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    # Create custom diverging colormap
    cmap = plt.cm.RdBu_r
    
    im = ax.imshow(
        bias_matrix,
        aspect='auto',
        origin='lower',
        extent=[iti_values[0], iti_values[-1], isi_values[0], isi_values[-1]],
        cmap=cmap,
        vmin=-np.max(np.abs(bias_matrix)),
        vmax=np.max(np.abs(bias_matrix)),
    )
    
    ax.set_xlabel('ITI (ms)')
    ax.set_ylabel('ISI (ms)')
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Serial Bias (°)')
    
    return ax


def plot_fig2_single_layer(
    std_results: dict,
    stf_results: dict,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Reproduce Figure 2: Single-layer CANN experiments.
    
    Args:
        std_results: Results from STD-dominated experiment
        stf_results: Results from STF-dominated experiment
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: STD-dominated (A-C)
    # A: Neural activity
    ax1 = fig.add_subplot(gs[0, 0])
    if 'activity' in std_results:
        plot_neural_activity(
            std_results['time'], std_results['activity'], 
            std_results['theta'], ax=ax1, title='A. STD Neural Activity'
        )
    
    # B: STP dynamics
    ax2 = fig.add_subplot(gs[0, 1])
    if 'stp_x' in std_results:
        plot_stp_dynamics(
            std_results['time'], std_results['stp_x'], std_results['stp_u'],
            neuron_idx=std_results.get('stim_neuron', 45),
            ax=ax2, title='B. STD Dynamics'
        )
    
    # C: Adjustment error
    ax3 = fig.add_subplot(gs[0, 2])
    plot_adjustment_error(
        std_results['delta'], std_results['errors'],
        ax=ax3, title='C. STD Adjustment Error (Repulsion)',
        color='#E74C3C'
    )
    
    # Row 2: STF-dominated (D-F)
    # D: Neural activity
    ax4 = fig.add_subplot(gs[1, 0])
    if 'activity' in stf_results:
        plot_neural_activity(
            stf_results['time'], stf_results['activity'],
            stf_results['theta'], ax=ax4, title='D. STF Neural Activity'
        )
    
    # E: STP dynamics
    ax5 = fig.add_subplot(gs[1, 1])
    if 'stp_x' in stf_results:
        plot_stp_dynamics(
            stf_results['time'], stf_results['stp_x'], stf_results['stp_u'],
            neuron_idx=stf_results.get('stim_neuron', 45),
            ax=ax5, title='E. STF Dynamics'
        )
    
    # F: Adjustment error
    ax6 = fig.add_subplot(gs[1, 2])
    plot_adjustment_error(
        stf_results['delta'], stf_results['errors'],
        ax=ax6, title='F. STF Adjustment Error (Attraction)',
        color='#3498DB'
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_fig3_two_layer(
    within_trial_results: dict,
    between_trial_results: dict,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Reproduce Figure 3: Two-layer CANN experiments.
    
    Args:
        within_trial_results: Within-trial serial dependence results
        between_trial_results: Between-trial serial dependence results
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # A: Two-layer architecture schematic (placeholder)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.5, 'Two-Layer CANN\nArchitecture', 
             ha='center', va='center', fontsize=14,
             transform=ax1.transAxes)
    ax1.set_title('A. Model Architecture')
    ax1.axis('off')
    
    # B: Layer activities
    ax2 = fig.add_subplot(gs[0, 1])
    if 'low_activity' in within_trial_results and 'high_activity' in within_trial_results:
        t = np.arange(len(within_trial_results['low_activity']))
        ax2.plot(t, np.max(within_trial_results['low_activity'], axis=1), 
                 'b-', label='Low Layer', linewidth=2)
        ax2.plot(t, np.max(within_trial_results['high_activity'], axis=1),
                 'r-', label='High Layer', linewidth=2)
        ax2.set_xlabel('Time (a.u.)')
        ax2.set_ylabel('Peak Activity')
        ax2.legend()
    ax2.set_title('B. Layer Activities')
    
    # C: Within-trial (repulsion)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_adjustment_error(
        within_trial_results['delta'], within_trial_results['errors'],
        ax=ax3, title='C. Within-Trial Bias (Repulsion)',
        color='#E74C3C'
    )
    
    # D: Between-trial (attraction)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_adjustment_error(
        between_trial_results['delta'], between_trial_results['errors'],
        ax=ax4, title='D. Between-Trial Bias (Attraction)',
        color='#3498DB'
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_fig4_temporal(
    isi_bias: dict,
    iti_bias: dict,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Reproduce Figure 4: Temporal window analysis.
    
    Args:
        isi_bias: Dict with 'isi_values' and 'bias' arrays
        iti_bias: Dict with 'iti_values' and 'bias' arrays
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # A: ISI effect on within-trial bias
    ax1 = axes[0]
    ax1.plot(isi_bias['isi_values'], isi_bias['bias'], 'o-', 
             color='#E74C3C', linewidth=2, markersize=8)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('ISI (ms)')
    ax1.set_ylabel('Within-Trial Bias (°)')
    ax1.set_title('A. ISI Effect on Repulsion')
    
    # B: ITI effect on between-trial bias
    ax2 = axes[1]
    ax2.plot(iti_bias['iti_values'], iti_bias['bias'], 's-',
             color='#3498DB', linewidth=2, markersize=8)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax2.set_xlabel('ITI (ms)')
    ax2.set_ylabel('Between-Trial Bias (°)')
    ax2.set_title('B. ITI Effect on Attraction')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def plot_decoder_comparison(
    delta: np.ndarray,
    errors_dict: dict,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Compare different decoding methods (Appendix F).
    
    Args:
        delta: Stimulus differences
        errors_dict: Dict mapping decoder name to errors array
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    colors = {'pvm': '#3498DB', 'com': '#2ECC71', 'ml': '#E74C3C', 'peak': '#9B59B6'}
    titles = {'pvm': 'Population Vector', 'com': 'Center of Mass',
              'ml': 'Maximum Likelihood', 'peak': 'Peak Decoding'}
    
    for ax, (name, errors) in zip(axes, errors_dict.items()):
        plot_adjustment_error(
            delta, errors, ax=ax,
            title=titles.get(name, name),
            color=colors.get(name, 'blue')
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def create_summary_figure(
    results: dict,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create comprehensive summary figure.
    
    Args:
        results: Dictionary with all experimental results
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # Title
    fig.suptitle('Serial Dependence: Neural Correlates of Repulsion and Attraction',
                 fontsize=14, fontweight='bold')
    
    # Add subplots based on available results
    # ... (customize based on results structure)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig



def plot_stp_all_neurons(
    time: np.ndarray,
    stp_var: np.ndarray,
    theta: np.ndarray,
    var_name: str = 'x (availability)',
    ax: Optional[plt.Axes] = None,
    title: str = "STP Dynamics (All Neurons)",
    cmap: str = 'viridis',
) -> plt.Axes:
    """Plot STP variable for ALL neurons over time (heatmap).
    
    Args:
        time: Time points (ms), shape (T,)
        stp_var: STP variable array, shape (T, N)
        theta: Preferred orientations (degrees), shape (N,)
        var_name: Variable name for title and colorbar
        ax: Matplotlib axes
        title: Plot title
        cmap: Colormap name
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    
    im = ax.imshow(
        stp_var.T,
        aspect='auto',
        origin='lower',
        extent=[time[0], time[-1], theta[0], theta[-1]],
        cmap=cmap,
        vmin=np.min(stp_var),
        vmax=np.max(stp_var),
    )
    
    plt.colorbar(im, ax=ax, label=var_name)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Preferred Orientation (°)')
    ax.set_title(title)
    
    return ax


def plot_fig2_single_layer_v2(
    std_results: dict,
    stf_results: dict,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Reproduce Figure 2: Single-layer CANN experiments (with STP heatmaps).
    
    Args:
        std_results: Results from STD-dominated experiment
        stf_results: Results from STF-dominated experiment
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)
    
    # Row 1: STD-dominated (A-C)
    # A: Neural activity
    ax1 = fig.add_subplot(gs[0, 0])
    if 'activity' in std_results:
        plot_neural_activity(
            std_results['time'], std_results['activity'], 
            std_results['theta'], ax=ax1, title='A. STD Neural Activity'
        )
    
    # A2: STP x (all neurons)
    ax1b = fig.add_subplot(gs[0, 1])
    if 'stp_x' in std_results:
        plot_stp_all_neurons(
            std_results['time'], std_results['stp_x'],
            std_results['theta'], var_name='x (availability)',
            ax=ax1b, title='A2. STD STP x (All Neurons)', cmap='hot'
        )
    
    # B: STP dynamics (single neuron)
    ax2 = fig.add_subplot(gs[0, 2])
    if 'stp_x' in std_results:
        plot_stp_dynamics(
            std_results['time'], std_results['stp_x'], std_results['stp_u'],
            neuron_idx=std_results.get('stim_neuron', 45),
            ax=ax2, title='B. STD Dynamics'
        )
    
    # C: Adjustment error
    ax3 = fig.add_subplot(gs[0, 2])
    plot_adjustment_error(
        std_results['delta'], std_results['errors'],
        ax=ax3, title='C. STD Adjustment Error (Repulsion)',
        color='#E74C3C'
    )
    
    # Row 2: STF-dominated (D-F)
    # D: Neural activity
    ax4 = fig.add_subplot(gs[1, 0])
    if 'activity' in stf_results:
        plot_neural_activity(
            stf_results['time'], stf_results['activity'],
            stf_results['theta'], ax=ax4, title='D. STF Neural Activity'
        )
    
    # D2: STP u (all neurons)
    ax4b = fig.add_subplot(gs[1, 1])
    if 'stp_u' in stf_results:
        plot_stp_all_neurons(
            stf_results['time'], stf_results['stp_u'],
            stf_results['theta'], var_name='u (release prob.)',
            ax=ax4b, title='D2. STF STP u (All Neurons)', cmap='cool'
        )
    
    # E: STP dynamics (single neuron)
    ax5 = fig.add_subplot(gs[1, 1])
    if 'stp_u' in stf_results:
        plot_stp_dynamics(
            stf_results['time'], stf_results['stp_x'], stf_results['stp_u'],
            neuron_idx=stf_results.get('stim_neuron', 45),
            ax=ax5, title='E. STF Dynamics'
        )
    
    # F: Adjustment error
    ax6 = fig.add_subplot(gs[1, 2])
    plot_adjustment_error(
        stf_results['delta'], stf_results['errors'],
        ax=ax6, title='F. STF Adjustment Error (Attraction)',
        color='#3498DB'
    )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig
