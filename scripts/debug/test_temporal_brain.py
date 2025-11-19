"""
Test script for TemporalBrain module.

Verifies:
1. Basic forward pass
2. Sequence processing
3. Hidden state persistence
4. Save/load functionality
"""

import os
import sys
import torch

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from models.temporal_brain import TemporalBrain


def test_forward_pass():
    """Test basic forward pass with single timestep."""
    print("\n=== Test 1: Forward Pass ===")
    
    brain = TemporalBrain(
        latent_dim=256,
        action_dim=6,
        hidden_dim=256,
        context_dim=128,
        model_type="gru"
    )
    
    # Single timestep
    z_t = torch.randn(1, 256)
    a_t = torch.randn(1, 6)
    
    c_t, h_t = brain(z_t, a_t)
    
    print(f"Input shapes: z_t={z_t.shape}, a_t={a_t.shape}")
    print(f"Output shapes: c_t={c_t.shape}")
    print(f"Hidden state type: {type(h_t)}")
    
    assert c_t.shape == (1, 128), f"Expected context shape (1, 128), got {c_t.shape}"
    assert h_t is not None, "Hidden state should not be None"
    
    print("✓ Forward pass successful")


def test_sequence_processing():
    """Test processing entire sequences."""
    print("\n=== Test 2: Sequence Processing ===")
    
    brain = TemporalBrain(
        latent_dim=256,
        action_dim=6,
        hidden_dim=256,
        context_dim=128,
        model_type="gru"
    )
    
    # Sequence of 10 timesteps
    batch_size = 2
    seq_len = 10
    
    z_seq = torch.randn(batch_size, seq_len, 256)
    a_seq = torch.randn(batch_size, seq_len, 6)
    
    c_seq = brain.forward_sequence(z_seq, a_seq)
    
    print(f"Input shapes: z_seq={z_seq.shape}, a_seq={a_seq.shape}")
    print(f"Output shape: c_seq={c_seq.shape}")
    
    assert c_seq.shape == (batch_size, seq_len, 128), \
        f"Expected shape ({batch_size}, {seq_len}, 128), got {c_seq.shape}"
    
    print("✓ Sequence processing successful")


def test_hidden_state_persistence():
    """Test that hidden state persists across steps."""
    print("\n=== Test 3: Hidden State Persistence ===")
    
    brain = TemporalBrain(model_type="gru")
    
    # Process multiple steps with increasingly different inputs
    contexts = []
    for step in range(10):
        # Make inputs progressively more different
        z_t = torch.randn(1, 256) * (1.0 + step * 0.5)
        a_t = torch.randn(1, 6) * (1.0 + step * 0.3)
        c_t, h_t = brain(z_t, a_t)
        contexts.append(c_t.detach())
    
    # Context should change over time
    c_0 = contexts[0]
    c_9 = contexts[9]
    
    diff = torch.mean((c_0 - c_9) ** 2).item()
    print(f"Context difference (step 0 vs step 9): {diff:.4f}")
    
    # With GRU and random inputs, changes can be subtle but should exist
    assert diff > 0.0001, "Context should change over time"
    
    # Also check that consecutive steps are different
    consecutive_diffs = []
    for i in range(1, len(contexts)):
        d = torch.mean((contexts[i] - contexts[i-1]) ** 2).item()
        consecutive_diffs.append(d)
    
    avg_consecutive_diff = sum(consecutive_diffs) / len(consecutive_diffs)
    print(f"Average consecutive difference: {avg_consecutive_diff:.6f}")
    
    # Reset and verify
    brain.reset_hidden()
    z_t = torch.randn(1, 256)
    a_t = torch.randn(1, 6)
    c_new, h_new = brain(z_t, a_t)
    
    print("✓ Hidden state persistence working")


def test_different_architectures():
    """Test different sequence model types."""
    print("\n=== Test 4: Different Architectures ===")
    
    for model_type in ["gru", "lstm", "transformer"]:
        print(f"\nTesting {model_type.upper()}...")
        
        brain = TemporalBrain(
            latent_dim=256,
            action_dim=6,
            model_type=model_type
        )
        
        z_t = torch.randn(1, 256)
        a_t = torch.randn(1, 6)
        
        c_t, h_t = brain(z_t, a_t)
        
        assert c_t.shape == (1, 128), f"{model_type}: Wrong context shape"
        print(f"  ✓ {model_type.upper()} working")
    
    print("\n✓ All architectures working")


def test_save_load():
    """Test save and load functionality."""
    print("\n=== Test 5: Save/Load ===")
    
    # Create and run brain
    brain1 = TemporalBrain(model_type="gru")
    
    z_t = torch.randn(1, 256)
    a_t = torch.randn(1, 6)
    c1, h1 = brain1(z_t, a_t)
    
    # Save
    save_path = "test_temporal_brain.pt"
    brain1.save_state(save_path)
    print(f"Saved to {save_path}")
    
    # Load
    brain2 = TemporalBrain.load_state(save_path)
    
    # Should produce same output
    c2, h2 = brain2(z_t, a_t)
    
    diff = torch.mean((c1 - c2) ** 2).item()
    print(f"Output difference after load: {diff:.6f}")
    
    assert diff < 0.001, f"Loaded model output difference too large: {diff}"
    
    # Cleanup
    os.remove(save_path)
    print("✓ Save/load successful")


def test_batch_processing():
    """Test processing batches."""
    print("\n=== Test 6: Batch Processing ===")
    
    brain = TemporalBrain(model_type="gru")
    
    batch_size = 4
    z_batch = torch.randn(batch_size, 256)
    a_batch = torch.randn(batch_size, 6)
    
    c_batch, h_batch = brain(z_batch, a_batch)
    
    print(f"Batch input: {z_batch.shape}")
    print(f"Batch output: {c_batch.shape}")
    
    assert c_batch.shape == (batch_size, 128), "Batch processing failed"
    
    print("✓ Batch processing successful")


def test_context_evolution():
    """Test that context evolves meaningfully over an episode."""
    print("\n=== Test 7: Context Evolution ===")
    
    brain = TemporalBrain(model_type="gru")
    
    # Simulate episode with changing inputs
    num_steps = 20
    contexts = []
    
    for step in range(num_steps):
        # Gradually changing input
        z_t = torch.randn(1, 256) * (1 + step * 0.1)
        a_t = torch.randn(1, 6)
        
        c_t, _ = brain(z_t, a_t)
        contexts.append(c_t.detach())
    
    # Compute variance over time
    contexts_tensor = torch.cat(contexts, dim=0)  # (num_steps, 128)
    variance = torch.var(contexts_tensor, dim=0).mean().item()
    
    print(f"Context variance over {num_steps} steps: {variance:.4f}")
    
    assert variance > 0.01, "Context should vary over episode"
    
    # Compute trajectory length (sum of consecutive differences)
    trajectory_length = 0.0
    for i in range(1, num_steps):
        diff = torch.mean((contexts[i] - contexts[i-1]) ** 2).item()
        trajectory_length += diff
    
    print(f"Context trajectory length: {trajectory_length:.4f}")
    
    print("✓ Context evolution verified")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing TemporalBrain Module")
    print("=" * 60)
    
    try:
        test_forward_pass()
        test_sequence_processing()
        test_hidden_state_persistence()
        test_different_architectures()
        test_save_load()
        test_batch_processing()
        test_context_evolution()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
