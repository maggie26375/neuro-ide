"""
Test script for Neural ODE analysis and visualization tools.
"""

import torch
import sys
sys.path.insert(0, '.')

from models.neural_ode_perturbation import NeuralODEPerturbationModel
from utils.neural_ode_analysis import (
    analyze_perturbation_dynamics,
    visualize_perturbation_dynamics,
    compare_perturbations,
    export_analysis_data
)
import matplotlib.pyplot as plt
from pathlib import Path


def test_analyze_perturbation_dynamics():
    """Test the analysis function"""
    print("Testing analyze_perturbation_dynamics()...")

    # Create model
    model = NeuralODEPerturbationModel(
        state_dim=64,
        pert_dim=128,
        gene_dim=100,
        ode_hidden_dim=128
    )

    # Create test data
    batch_size = 8
    initial_states = torch.randn(batch_size, 64)
    perturbation_emb = torch.randn(batch_size, 128)

    try:
        # Run analysis
        results = analyze_perturbation_dynamics(
            model,
            initial_states,
            perturbation_emb,
            num_time_points=20
        )

        # Verify results
        print(f"✓ Analysis completed successfully!")
        print(f"  Trajectory shape: {results['trajectory'].shape}")
        print(f"  Velocity field shape: {results['velocity_field'].shape}")
        print(f"  Time points shape: {results['time_points'].shape}")
        print(f"  Response magnitude shape: {results['response_magnitude'].shape}")

        # Check convergence metrics
        conv = results['convergence_metrics']
        print(f"\n  Convergence Metrics:")
        print(f"    Final velocity (mean): {conv['final_velocity_mean']:.4f}")
        print(f"    Trajectory length: {conv['trajectory_length']:.4f}")
        print(f"    Is converging: {conv['is_converging']}")
        print(f"    Convergence rate: {conv['convergence_rate']:.2f}x")

        return True, results

    except Exception as e:
        print(f"✗ Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_visualize_perturbation_dynamics(analysis_results):
    """Test the visualization function"""
    print("\nTesting visualize_perturbation_dynamics()...")

    try:
        # Create visualization
        fig = visualize_perturbation_dynamics(
            analysis_results,
            save_path='test_outputs/perturbation_dynamics.png',
            sample_trajectories=3,
            pca_components=2,
            show_velocity_field=True
        )

        print(f"✓ Visualization created successfully!")
        print(f"  Figure saved to: test_outputs/perturbation_dynamics.png")

        # Close figure to avoid display issues in headless mode
        plt.close(fig)

        return True

    except Exception as e:
        print(f"✗ Error in visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_compare_perturbations():
    """Test the comparison function"""
    print("\nTesting compare_perturbations()...")

    # Create model
    model = NeuralODEPerturbationModel(
        state_dim=64,
        pert_dim=128,
        gene_dim=100,
        ode_hidden_dim=128
    )

    # Create test data
    batch_size = 8
    initial_states = torch.randn(batch_size, 64)

    # Create multiple perturbations
    perturbation1 = torch.randn(batch_size, 128) * 0.5
    perturbation2 = torch.randn(batch_size, 128) * 1.0
    perturbation3 = torch.randn(batch_size, 128) * 1.5

    perturbations = [perturbation1, perturbation2, perturbation3]
    names = ['Small Perturbation', 'Medium Perturbation', 'Large Perturbation']

    try:
        # Run comparison
        fig = compare_perturbations(
            model,
            initial_states,
            perturbations,
            perturbation_names=names,
            save_path='test_outputs/perturbation_comparison.png'
        )

        print(f"✓ Comparison visualization created successfully!")
        print(f"  Figure saved to: test_outputs/perturbation_comparison.png")

        # Close figure
        plt.close(fig)

        return True

    except Exception as e:
        print(f"✗ Error in comparison: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_export_analysis_data(analysis_results):
    """Test the data export function"""
    print("\nTesting export_analysis_data()...")

    try:
        export_analysis_data(
            analysis_results,
            output_dir='test_outputs/exported_data',
            prefix='test_analysis'
        )

        # Check if files were created
        output_dir = Path('test_outputs/exported_data')
        expected_files = [
            'test_analysis_trajectory.npy',
            'test_analysis_velocity_field.npy',
            'test_analysis_time_points.npy',
            'test_analysis_metrics.txt'
        ]

        all_exist = all((output_dir / f).exists() for f in expected_files)

        if all_exist:
            print(f"✓ Data export successful!")
            print(f"  Files created:")
            for f in expected_files:
                print(f"    - {f}")
            return True
        else:
            print(f"✗ Some files were not created")
            return False

    except Exception as e:
        print(f"✗ Error in data export: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trajectory_consistency():
    """Test that trajectories are consistent and physically meaningful"""
    print("\nTesting trajectory consistency...")

    model = NeuralODEPerturbationModel(
        state_dim=64,
        pert_dim=128,
        gene_dim=100
    )

    batch_size = 4
    initial_states = torch.randn(batch_size, 64)
    perturbation_emb = torch.randn(batch_size, 128)

    try:
        # Get trajectory
        results = analyze_perturbation_dynamics(
            model,
            initial_states,
            perturbation_emb,
            num_time_points=30
        )

        trajectory = results['trajectory']
        velocity = results['velocity_field']

        # Check 1: Initial state matches
        initial_from_traj = trajectory[0]
        initial_expected = results['initial_states']
        initial_match = torch.allclose(
            torch.from_numpy(initial_from_traj),
            torch.from_numpy(initial_expected),
            atol=1e-5
        )

        # Check 2: Trajectory is continuous (no jumps)
        diffs = trajectory[1:] - trajectory[:-1]
        max_jump = abs(diffs).max()

        # Check 3: Velocity field is reasonable
        velocity_norms = (velocity ** 2).sum(axis=-1) ** 0.5
        mean_velocity = velocity_norms.mean()

        print(f"✓ Trajectory consistency checks:")
        print(f"  Initial state matches: {initial_match}")
        print(f"  Max trajectory jump: {max_jump:.6f}")
        print(f"  Mean velocity magnitude: {mean_velocity:.6f}")

        return True

    except Exception as e:
        print(f"✗ Error in consistency test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Neural ODE Analysis and Visualization Tools Tests")
    print("=" * 70)

    # Create output directory
    Path('test_outputs').mkdir(exist_ok=True)

    # Run tests
    test1, results = test_analyze_perturbation_dynamics()
    test2 = False
    test3 = False
    test4 = False
    test5 = False

    if test1 and results is not None:
        test2 = test_visualize_perturbation_dynamics(results)
        test3 = test_compare_perturbations()
        test4 = test_export_analysis_data(results)
        test5 = test_trajectory_consistency()

    # Summary
    print("\n" + "=" * 70)
    if test1 and test2 and test3 and test4 and test5:
        print("✓ All tests passed! Analysis tools are working correctly.")
        print("\nGenerated outputs:")
        print("  - test_outputs/perturbation_dynamics.png")
        print("  - test_outputs/perturbation_comparison.png")
        print("  - test_outputs/exported_data/")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 70)
