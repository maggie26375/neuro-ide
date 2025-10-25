"""
Test script for perception layer bug fixes.
"""

import torch
import sys
sys.path.insert(0, '.')

from models.active_perception import ActivePerceptionLayer
from models.temporal_control import TemporalControlLayer
from models.adaptive_representation import AdaptiveRepresentationNetwork

def test_active_perception():
    """Test Active Perception Layer"""
    print("Testing Active Perception Layer...")

    layer = ActivePerceptionLayer(
        input_dim=512,
        feature_dim=128,
        num_features=10,
        attention_dim=64,
        sampling_budget=5
    )

    batch_size = 4
    x = torch.randn(batch_size, 512)
    available_features = [torch.randn(batch_size, 128) for _ in range(10)]

    try:
        enhanced_x, perception_info = layer(x, available_features)
        print(f"✓ Active Perception forward pass successful!")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {enhanced_x.shape}")
        print(f"  Uncertainty shape: {perception_info['uncertainty'].shape}")
        print(f"  Selected features shape: {perception_info['selected_features'].shape}")
        return True
    except Exception as e:
        print(f"✗ Error in Active Perception Layer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_temporal_control():
    """Test Temporal Control Layer"""
    print("\nTesting Temporal Control Layer...")

    layer = TemporalControlLayer(
        input_dim=512,
        hidden_dim=256,
        num_time_steps=10,
        intervention_dim=128
    )

    batch_size = 4
    seq_len = 10
    temporal_sequence = torch.randn(batch_size, seq_len, 512)
    current_state = torch.randn(batch_size, 512)

    try:
        next_state, control_info = layer(temporal_sequence, current_state)
        print(f"✓ Temporal Control forward pass successful!")
        print(f"  Input sequence shape: {temporal_sequence.shape}")
        print(f"  Current state shape: {current_state.shape}")
        print(f"  Next state shape: {next_state.shape}")
        print(f"  Intervention prob shape: {control_info['intervention_prob'].shape}")
        print(f"  Intervention strength shape: {control_info['intervention_strength'].shape}")
        return True
    except Exception as e:
        print(f"✗ Error in Temporal Control Layer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_representation():
    """Test Adaptive Representation Network"""
    print("\nTesting Adaptive Representation Network...")

    network = AdaptiveRepresentationNetwork(
        input_dim=512,
        max_dim=512,
        min_dim=64,
        num_compression_levels=4
    )

    batch_size = 4
    x = torch.randn(batch_size, 512)

    try:
        adaptive_repr, adaptation_info = network(x)
        print(f"✓ Adaptive Representation forward pass successful!")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {adaptive_repr.shape}")
        print(f"  Target dim: {adaptation_info['target_dim']}")
        print(f"  Efficiency ratio: {adaptation_info['efficiency_ratio']}")
        print(f"  Quality score shape: {adaptation_info['quality_score'].shape}")
        return True
    except Exception as e:
        print(f"✗ Error in Adaptive Representation Network: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backward_pass():
    """Test backward pass for all layers"""
    print("\nTesting backward pass...")

    # Test Active Perception
    layer1 = ActivePerceptionLayer(input_dim=512, feature_dim=128, num_features=10, attention_dim=64)
    x1 = torch.randn(2, 512, requires_grad=True)
    features1 = [torch.randn(2, 128) for _ in range(10)]
    enhanced_x1, _ = layer1(x1, features1)
    loss1 = enhanced_x1.sum()
    loss1.backward()

    # Test Temporal Control
    layer2 = TemporalControlLayer(input_dim=512, hidden_dim=256)
    sequence = torch.randn(2, 10, 512, requires_grad=True)
    state = torch.randn(2, 512, requires_grad=True)
    next_state, _ = layer2(sequence, state)
    loss2 = next_state.sum()
    loss2.backward()

    # Test Adaptive Representation
    layer3 = AdaptiveRepresentationNetwork(input_dim=512, max_dim=512)
    x3 = torch.randn(2, 512, requires_grad=True)
    adaptive_repr, _ = layer3(x3)
    loss3 = adaptive_repr.sum()
    loss3.backward()

    print("✓ All backward passes successful! Gradients computed.")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Perception Layers Bug Fix Verification Tests")
    print("=" * 60)

    test1 = test_active_perception()
    test2 = test_temporal_control()
    test3 = test_adaptive_representation()
    test4 = test_backward_pass()

    print("\n" + "=" * 60)
    if test1 and test2 and test3 and test4:
        print("✓ All tests passed! Perception layers are working correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60)
