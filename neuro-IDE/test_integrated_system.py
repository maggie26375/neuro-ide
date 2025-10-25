"""
Test script for integrated perception system.
"""

import torch
import sys
sys.path.insert(0, '.')

from models.integrated_perception_system import IntegratedPerceptionSystem

def test_integrated_system():
    """Test Integrated Perception System"""
    print("Testing Integrated Perception System...")

    system = IntegratedPerceptionSystem(
        input_dim=512,
        feature_dim=128,
        num_features=10,
        max_dim=512,
        min_dim=64,
        hidden_dim=256,
        num_time_steps=10,
        intervention_dim=128
    )

    batch_size = 4
    seq_len = 10

    # 准备输入
    x = torch.randn(batch_size, 512)
    temporal_sequence = torch.randn(batch_size, seq_len, 512)
    available_features = [torch.randn(batch_size, 128) for _ in range(10)]

    try:
        coordinated_output, system_info = system(
            x, temporal_sequence, available_features
        )

        print(f"✓ Integrated System forward pass successful!")
        print(f"  Input shape: {x.shape}")
        print(f"  Temporal sequence shape: {temporal_sequence.shape}")
        print(f"  Coordinated output shape: {coordinated_output.shape}")
        print(f"  Perception info keys: {list(system_info['perception'].keys())}")
        print(f"  Control info keys: {list(system_info['control'].keys())}")
        print(f"  Adaptation info keys: {list(system_info['adaptation'].keys())}")

        return True
    except Exception as e:
        print(f"✗ Error in Integrated System: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_system_loss():
    """Test system loss computation"""
    print("\nTesting System Loss Computation...")

    system = IntegratedPerceptionSystem(
        input_dim=512,
        feature_dim=128,
        num_features=10
    )

    batch_size = 4
    seq_len = 10

    x = torch.randn(batch_size, 512)
    temporal_sequence = torch.randn(batch_size, seq_len, 512)
    available_features = [torch.randn(batch_size, 128) for _ in range(10)]
    target = torch.randn(batch_size, 512)

    try:
        output, system_info = system(x, temporal_sequence, available_features)
        loss = system.compute_system_loss(output, target, system_info)

        print(f"✓ System loss computation successful!")
        print(f"  Loss value: {loss.item():.4f}")

        # Test backward
        loss.backward()
        print(f"✓ Backward pass successful!")

        return True
    except Exception as e:
        print(f"✗ Error in system loss: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Integrated Perception System Tests")
    print("=" * 60)

    test1 = test_integrated_system()
    test2 = test_system_loss()

    print("\n" + "=" * 60)
    if test1 and test2:
        print("✓ All tests passed! Integrated system is working correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60)
