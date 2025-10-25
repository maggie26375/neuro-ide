"""
Simple test script to verify Neural ODE functionality after bug fixes.
"""

import torch
import sys
sys.path.insert(0, '.')

from models.neural_ode_perturbation import NeuralODEPerturbationModel

def test_neural_ode_forward():
    """Test basic forward pass of Neural ODE model"""
    print("Testing Neural ODE forward pass...")

    # Create a small test model
    model = NeuralODEPerturbationModel(
        state_dim=64,
        pert_dim=128,
        gene_dim=100,
        ode_hidden_dim=32,
        ode_layers=2,
        num_time_points=5
    )

    # Create dummy inputs
    batch_size = 4
    initial_states = torch.randn(batch_size, 64)
    perturbation_emb = torch.randn(batch_size, 128)

    try:
        # Test forward pass
        predictions = model(initial_states, perturbation_emb, return_trajectory=False)
        print(f"✓ Forward pass successful! Output shape: {predictions.shape}")

        # Test trajectory return
        trajectory = model(initial_states, perturbation_emb, return_trajectory=True)
        print(f"✓ Trajectory generation successful! Shape: {trajectory.shape}")

        # Test backward pass (ensure gradients work with adjoint method)
        loss = predictions.sum()
        loss.backward()
        print("✓ Backward pass successful! Gradients computed.")

        return True

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ode_func():
    """Test the ODE function directly"""
    print("\nTesting ODE function...")

    from models.neural_ode_perturbation import PerturbationODEFunc

    ode_func = PerturbationODEFunc(
        state_dim=64,
        pert_dim=128,
        hidden_dim=32,
        num_layers=2
    )

    batch_size = 4
    x = torch.randn(batch_size, 64)
    pert_emb = torch.randn(batch_size, 128)
    t = torch.tensor(0.5)

    try:
        velocity = ode_func(t, x, pert_emb)
        print(f"✓ ODE function call successful! Velocity shape: {velocity.shape}")
        return True
    except Exception as e:
        print(f"✗ Error in ODE function: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Neural ODE Bug Fix Verification Tests")
    print("=" * 60)

    test1 = test_ode_func()
    test2 = test_neural_ode_forward()

    print("\n" + "=" * 60)
    if test1 and test2:
        print("✓ All tests passed! Neural ODE is working correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60)
