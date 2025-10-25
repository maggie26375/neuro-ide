"""
Analysis and Visualization Tools for Neural ODE Perturbation Dynamics.

This module provides tools to analyze and visualize the dynamics of perturbation
responses in the Neural ODE model, including trajectory analysis and velocity fields.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import seaborn as sns


def analyze_perturbation_dynamics(
    model,
    initial_states: torch.Tensor,
    perturbation_emb: torch.Tensor,
    time_points: Optional[torch.Tensor] = None,
    num_time_points: int = 50,
    device: str = 'cpu'
) -> Dict[str, Union[torch.Tensor, np.ndarray, Dict]]:
    """
    Analyze the dynamics of perturbation responses.

    Args:
        model: NeuralODEPerturbationModel instance
        initial_states: Initial states [batch_size, state_dim]
        perturbation_emb: Perturbation embeddings [batch_size, pert_dim]
        time_points: Optional custom time points
        num_time_points: Number of time points to sample
        device: Device to run on

    Returns:
        analysis_results: Dictionary containing:
            - trajectory: Full state trajectory [num_time_points, batch_size, state_dim]
            - velocity_field: Velocity at each time point
            - time_points: Time points used
            - perturbation_response: Response magnitude over time
            - state_statistics: Mean, std, min, max at each time point
            - convergence_metrics: Metrics about trajectory convergence
    """
    model.eval()
    model = model.to(device)
    initial_states = initial_states.to(device)
    perturbation_emb = perturbation_emb.to(device)

    with torch.no_grad():
        # 1. Set time points if needed (model has internal time points)
        if time_points is not None:
            # Save original time points
            original_time_points = model.time_points
            model.time_points = time_points.to(device)
        else:
            # If not provided, create them
            if not hasattr(model, 'time_points') or model.time_points.shape[0] != num_time_points:
                model.time_points = torch.linspace(0, 1, num_time_points).to(device)

        # 2. Get full trajectory
        trajectory = model(
            initial_states,
            perturbation_emb,
            return_trajectory=True
        )  # [num_time_points, batch_size, state_dim]

        # Get actual time points used
        time_points = model.time_points

        # Restore original time points if we changed them
        if 'original_time_points' in locals():
            model.time_points = original_time_points

        # 3. Compute velocity field at each time point
        velocity_field = []
        for t_idx, t in enumerate(time_points):
            states_at_t = trajectory[t_idx]  # [batch_size, state_dim]

            # Use ODEWrapper to compute velocity
            from models.neural_ode_perturbation import NeuralODEPerturbationModel
            if hasattr(model, 'ode_func'):
                velocity = model.ode_func(t, states_at_t, perturbation_emb)
                velocity_field.append(velocity)

        velocity_field = torch.stack(velocity_field)  # [num_time_points, batch_size, state_dim]

        # 4. Compute perturbation response magnitude
        # Distance from initial state
        response_magnitude = torch.norm(
            trajectory - initial_states.unsqueeze(0),
            dim=-1
        )  # [num_time_points, batch_size]

        # Velocity magnitude
        velocity_magnitude = torch.norm(velocity_field, dim=-1)  # [num_time_points, batch_size]

        # 5. Compute state statistics at each time point
        state_statistics = {
            'mean': trajectory.mean(dim=1),  # [num_time_points, state_dim]
            'std': trajectory.std(dim=1),
            'min': trajectory.min(dim=1)[0],
            'max': trajectory.max(dim=1)[0]
        }

        # 6. Convergence metrics
        # Rate of change in final 20% of trajectory
        final_portion = int(num_time_points * 0.8)
        final_velocities = velocity_magnitude[final_portion:]

        convergence_metrics = {
            'final_velocity_mean': final_velocities.mean().item(),
            'final_velocity_std': final_velocities.std().item(),
            'trajectory_length': response_magnitude[-1].mean().item(),
            'is_converging': (final_velocities.mean() < velocity_magnitude.mean()).item(),
            'convergence_rate': (velocity_magnitude[0].mean() / (final_velocities.mean() + 1e-8)).item()
        }

        # 7. Dimensionality reduction metrics (PCA-like analysis)
        # Flatten trajectory for analysis
        traj_flat = trajectory.reshape(num_time_points, -1)  # [num_time_points, batch_size * state_dim]

        # Compute trajectory variance captured by principal directions
        if traj_flat.shape[0] > 1:
            centered = traj_flat - traj_flat.mean(dim=0, keepdim=True)
            cov = (centered.T @ centered) / (traj_flat.shape[0] - 1)
            eigenvalues = torch.linalg.eigvalsh(cov)
            total_variance = eigenvalues.sum()
            explained_variance_ratio = eigenvalues / (total_variance + 1e-8)

            convergence_metrics['intrinsic_dimensionality'] = (
                (explained_variance_ratio > 0.01).sum().item()
            )

    # Convert to numpy for easier visualization
    analysis_results = {
        'trajectory': trajectory.cpu().numpy(),
        'velocity_field': velocity_field.cpu().numpy(),
        'time_points': time_points.cpu().numpy(),
        'response_magnitude': response_magnitude.cpu().numpy(),
        'velocity_magnitude': velocity_magnitude.cpu().numpy(),
        'state_statistics': {
            k: v.cpu().numpy() for k, v in state_statistics.items()
        },
        'convergence_metrics': convergence_metrics,
        'initial_states': initial_states.cpu().numpy(),
        'perturbation_emb': perturbation_emb.cpu().numpy()
    }

    return analysis_results


def visualize_perturbation_dynamics(
    analysis_results: Dict,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (20, 12),
    sample_trajectories: int = 5,
    pca_components: Optional[int] = 2,
    show_velocity_field: bool = True,
    dpi: int = 100
) -> plt.Figure:
    """
    Visualize perturbation dynamics with multiple panels.

    Args:
        analysis_results: Results from analyze_perturbation_dynamics()
        save_path: Optional path to save the figure
        figsize: Figure size (width, height)
        sample_trajectories: Number of individual trajectories to plot
        pca_components: Number of PCA components for dimensionality reduction
        show_velocity_field: Whether to show velocity field visualization
        dpi: DPI for saved figure

    Returns:
        fig: Matplotlib figure object
    """
    # Extract data
    trajectory = analysis_results['trajectory']  # [T, B, D]
    velocity_field = analysis_results['velocity_field']
    time_points = analysis_results['time_points']
    response_magnitude = analysis_results['response_magnitude']
    velocity_magnitude = analysis_results['velocity_magnitude']
    state_stats = analysis_results['state_statistics']
    convergence = analysis_results['convergence_metrics']

    num_time_points, batch_size, state_dim = trajectory.shape

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # ============================================================
    # Panel 1: Response Magnitude Over Time
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])

    # Plot individual trajectories (sample)
    indices = np.linspace(0, batch_size - 1, min(sample_trajectories, batch_size), dtype=int)
    for idx in indices:
        ax1.plot(time_points, response_magnitude[:, idx], alpha=0.3, linewidth=1)

    # Plot mean and std
    mean_response = response_magnitude.mean(axis=1)
    std_response = response_magnitude.std(axis=1)
    ax1.plot(time_points, mean_response, 'r-', linewidth=2, label='Mean')
    ax1.fill_between(
        time_points,
        mean_response - std_response,
        mean_response + std_response,
        alpha=0.2, color='r', label='±1 std'
    )

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Response Magnitude')
    ax1.set_title('Perturbation Response Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ============================================================
    # Panel 2: Velocity Magnitude Over Time
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])

    mean_velocity = velocity_magnitude.mean(axis=1)
    std_velocity = velocity_magnitude.std(axis=1)

    ax2.plot(time_points, mean_velocity, 'b-', linewidth=2, label='Mean velocity')
    ax2.fill_between(
        time_points,
        mean_velocity - std_velocity,
        mean_velocity + std_velocity,
        alpha=0.2, color='b', label='±1 std'
    )

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Velocity Magnitude')
    ax2.set_title('System Velocity Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ============================================================
    # Panel 3: Convergence Metrics
    # ============================================================
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    metrics_text = "Convergence Metrics:\n\n"
    metrics_text += f"Final Velocity (mean): {convergence['final_velocity_mean']:.4f}\n"
    metrics_text += f"Final Velocity (std): {convergence['final_velocity_std']:.4f}\n"
    metrics_text += f"Trajectory Length: {convergence['trajectory_length']:.4f}\n"
    metrics_text += f"Is Converging: {convergence['is_converging']}\n"
    metrics_text += f"Convergence Rate: {convergence['convergence_rate']:.2f}x\n"
    if 'intrinsic_dimensionality' in convergence:
        metrics_text += f"Intrinsic Dim: {convergence['intrinsic_dimensionality']}\n"

    ax3.text(0.1, 0.9, metrics_text, transform=ax3.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ============================================================
    # Panel 4: State Statistics Over Time
    # ============================================================
    ax4 = fig.add_subplot(gs[1, 0])

    # Plot first few dimensions of mean state
    num_dims_to_plot = min(5, state_dim)
    for dim in range(num_dims_to_plot):
        ax4.plot(time_points, state_stats['mean'][:, dim],
                label=f'Dim {dim}', alpha=0.7)

    ax4.set_xlabel('Time')
    ax4.set_ylabel('State Value (mean)')
    ax4.set_title(f'Mean State Evolution (first {num_dims_to_plot} dims)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ============================================================
    # Panel 5: State Variance Over Time
    # ============================================================
    ax5 = fig.add_subplot(gs[1, 1])

    # Plot variance across batch
    for dim in range(num_dims_to_plot):
        ax5.plot(time_points, state_stats['std'][:, dim]**2,
                label=f'Dim {dim}', alpha=0.7)

    ax5.set_xlabel('Time')
    ax5.set_ylabel('State Variance')
    ax5.set_title(f'State Variance Over Time (first {num_dims_to_plot} dims)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # ============================================================
    # Panel 6: PCA Projection of Trajectory
    # ============================================================
    if pca_components == 2:
        ax6 = fig.add_subplot(gs[1, 2])

        # Perform PCA on trajectory
        traj_reshaped = trajectory.reshape(num_time_points, batch_size * state_dim)

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        traj_pca = pca.fit_transform(traj_reshaped)

        # Color by time
        scatter = ax6.scatter(traj_pca[:, 0], traj_pca[:, 1],
                            c=time_points, cmap='viridis',
                            s=50, alpha=0.6)

        # Draw trajectory path
        ax6.plot(traj_pca[:, 0], traj_pca[:, 1], 'k-', alpha=0.3, linewidth=1)

        # Mark start and end
        ax6.scatter(traj_pca[0, 0], traj_pca[0, 1],
                   c='green', s=200, marker='o', label='Start',
                   edgecolors='black', linewidths=2)
        ax6.scatter(traj_pca[-1, 0], traj_pca[-1, 1],
                   c='red', s=200, marker='s', label='End',
                   edgecolors='black', linewidths=2)

        ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax6.set_title('Trajectory in PCA Space')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax6)
        cbar.set_label('Time')

    # ============================================================
    # Panel 7: Individual Trajectory Samples (First 3 dims)
    # ============================================================
    ax7 = fig.add_subplot(gs[2, 0], projection='3d')

    if state_dim >= 3:
        # Plot a few sample trajectories in 3D
        for idx in indices:
            ax7.plot(trajectory[:, idx, 0],
                    trajectory[:, idx, 1],
                    trajectory[:, idx, 2],
                    alpha=0.6, linewidth=1.5)

        # Mark start and end points
        ax7.scatter(trajectory[0, indices, 0],
                   trajectory[0, indices, 1],
                   trajectory[0, indices, 2],
                   c='green', s=100, marker='o', label='Start')
        ax7.scatter(trajectory[-1, indices, 0],
                   trajectory[-1, indices, 1],
                   trajectory[-1, indices, 2],
                   c='red', s=100, marker='s', label='End')

        ax7.set_xlabel('State Dim 0')
        ax7.set_ylabel('State Dim 1')
        ax7.set_zlabel('State Dim 2')
        ax7.set_title('3D Trajectory Visualization')
        ax7.legend()

    # ============================================================
    # Panel 8: Velocity Field Heatmap
    # ============================================================
    if show_velocity_field:
        ax8 = fig.add_subplot(gs[2, 1])

        # Show velocity field over time and dimensions (subsample)
        num_dims_to_show = min(20, state_dim)
        velocity_subset = velocity_magnitude[:, :min(10, batch_size)]

        im = ax8.imshow(velocity_subset.T, aspect='auto', cmap='hot',
                       interpolation='nearest')
        ax8.set_xlabel('Time Step')
        ax8.set_ylabel('Sample Index')
        ax8.set_title('Velocity Magnitude Heatmap')
        plt.colorbar(im, ax=ax8, label='Velocity')

    # ============================================================
    # Panel 9: Phase Portrait (2D projection)
    # ============================================================
    ax9 = fig.add_subplot(gs[2, 2])

    # Plot state vs velocity for first dimension
    for idx in indices:
        ax9.plot(trajectory[:, idx, 0], velocity_field[:, idx, 0],
                alpha=0.6, linewidth=1.5)

    # Add arrows to show direction
    for idx in indices[:2]:  # Just a couple for clarity
        subsample_step = max(1, num_time_points // 10)
        subsampled_traj = trajectory[::subsample_step, idx, 0]
        subsampled_vel = velocity_field[::subsample_step, idx, 0]

        if len(subsampled_traj) > 1:
            # Use differences to show direction
            ax9.quiver(subsampled_traj[:-1],
                      subsampled_vel[:-1],
                      subsampled_traj[1:] - subsampled_traj[:-1],
                      subsampled_vel[1:] - subsampled_vel[:-1],
                      scale_units='xy', angles='xy', scale=1, alpha=0.3, width=0.003)

    ax9.set_xlabel('State (Dim 0)')
    ax9.set_ylabel('Velocity (Dim 0)')
    ax9.set_title('Phase Portrait (Dim 0)')
    ax9.grid(True, alpha=0.3)

    # Overall title
    fig.suptitle('Neural ODE Perturbation Dynamics Analysis',
                fontsize=16, fontweight='bold')

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def compare_perturbations(
    model,
    initial_states: torch.Tensor,
    perturbations: List[torch.Tensor],
    perturbation_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 10)
) -> plt.Figure:
    """
    Compare dynamics across multiple perturbations.

    Args:
        model: NeuralODEPerturbationModel instance
        initial_states: Initial states [batch_size, state_dim]
        perturbations: List of perturbation embeddings to compare
        perturbation_names: Optional names for each perturbation
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        fig: Matplotlib figure
    """
    num_perturbations = len(perturbations)

    if perturbation_names is None:
        perturbation_names = [f'Perturbation {i+1}' for i in range(num_perturbations)]

    # Analyze each perturbation
    all_results = []
    for pert in perturbations:
        results = analyze_perturbation_dynamics(model, initial_states, pert)
        all_results.append(results)

    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Panel 1: Response magnitude comparison
    ax = axes[0, 0]
    for i, (results, name) in enumerate(zip(all_results, perturbation_names)):
        mean_response = results['response_magnitude'].mean(axis=1)
        ax.plot(results['time_points'], mean_response, label=name, linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Response Magnitude')
    ax.set_title('Response Magnitude Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Velocity magnitude comparison
    ax = axes[0, 1]
    for i, (results, name) in enumerate(zip(all_results, perturbation_names)):
        mean_velocity = results['velocity_magnitude'].mean(axis=1)
        ax.plot(results['time_points'], mean_velocity, label=name, linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Velocity Magnitude')
    ax.set_title('Velocity Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Convergence metrics comparison
    ax = axes[1, 0]
    metrics = ['final_velocity_mean', 'trajectory_length', 'convergence_rate']
    x_pos = np.arange(len(metrics))
    width = 0.8 / num_perturbations

    for i, (results, name) in enumerate(zip(all_results, perturbation_names)):
        values = [results['convergence_metrics'][m] for m in metrics]
        ax.bar(x_pos + i * width, values, width, label=name, alpha=0.7)

    ax.set_xticks(x_pos + width * (num_perturbations - 1) / 2)
    ax.set_xticklabels(['Final Velocity', 'Trajectory Length', 'Conv. Rate'])
    ax.set_ylabel('Value')
    ax.set_title('Convergence Metrics Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Final state distribution comparison
    ax = axes[1, 1]
    for i, (results, name) in enumerate(zip(all_results, perturbation_names)):
        final_states = results['trajectory'][-1, :, 0]  # First dimension
        ax.hist(final_states, bins=30, alpha=0.5, label=name, density=True)
    ax.set_xlabel('State Value (Dim 0)')
    ax.set_ylabel('Density')
    ax.set_title('Final State Distribution (Dim 0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Comparison figure saved to {save_path}")

    return fig


def export_analysis_data(
    analysis_results: Dict,
    output_dir: Union[str, Path],
    prefix: str = 'analysis'
):
    """
    Export analysis data to files.

    Args:
        analysis_results: Results from analyze_perturbation_dynamics()
        output_dir: Directory to save files
        prefix: Prefix for output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trajectory
    np.save(output_dir / f'{prefix}_trajectory.npy', analysis_results['trajectory'])

    # Save velocity field
    np.save(output_dir / f'{prefix}_velocity_field.npy', analysis_results['velocity_field'])

    # Save time points
    np.save(output_dir / f'{prefix}_time_points.npy', analysis_results['time_points'])

    # Save metrics as text
    with open(output_dir / f'{prefix}_metrics.txt', 'w') as f:
        f.write("Convergence Metrics:\n")
        f.write("=" * 50 + "\n")
        for key, value in analysis_results['convergence_metrics'].items():
            f.write(f"{key}: {value}\n")

    print(f"Analysis data exported to {output_dir}")
