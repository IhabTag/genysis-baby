"""
Test integration of TemporalBrain and EpisodicRetrieval with CuriousAgent.

Verifies:
1. Agent can be created with temporal brain
2. Temporal context is generated during action selection
3. Episodic retrieval works
4. State persistence works
"""

import os
import sys
import torch
import numpy as np

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

from models.world_model import WorldModel
from models.temporal_brain import TemporalBrain
from agent.curious_agent import CuriousAgent


def test_agent_with_temporal_brain():
    """Test creating agent with temporal brain."""
    print("\n=== Test 1: Agent with Temporal Brain ===")
    
    # Create world model
    world_model = WorldModel(img_size=128, latent_dim=256, proj_dim=64)
    
    # Create temporal brain
    temporal_brain = TemporalBrain(
        latent_dim=256,
        action_dim=6,
        hidden_dim=256,
        context_dim=128,
        model_type="gru"
    )
    
    # Create agent with temporal brain
    agent = CuriousAgent(
        world_model=world_model,
        device="cpu",
        temporal_brain=temporal_brain,
        use_episodic_retrieval=True,
        n_candidates=4  # Fewer for faster testing
    )
    
    print(f"Agent created successfully")
    print(f"  Temporal brain: {agent.temporal_brain is not None}")
    print(f"  Episodic retrieval: {agent.episodic_retrieval is not None}")
    print(f"  Temporal weight: {agent.temporal_weight}")
    
    assert agent.temporal_brain is not None
    assert agent.episodic_retrieval is not None
    
    print("✓ Agent creation successful")


def test_action_selection_with_context():
    """Test that action selection generates temporal context."""
    print("\n=== Test 2: Action Selection with Context ===")
    
    world_model = WorldModel(img_size=128, latent_dim=256, proj_dim=64)
    temporal_brain = TemporalBrain(latent_dim=256, action_dim=6)
    
    agent = CuriousAgent(
        world_model=world_model,
        temporal_brain=temporal_brain,
        use_episodic_retrieval=True,
        n_candidates=3,
        fast_mode=True  # Fast mode for testing
    )
    
    # Create dummy frame
    frame = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
    
    # First action (no context yet)
    action1, score1 = agent.select_action(frame)
    print(f"Action 1: {action1['type']}, score: {score1:.4f}")
    print(f"  Context after step 1: {agent.last_context is not None}")
    
    # Second action (should have context)
    frame2 = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
    action2, score2 = agent.select_action(frame2)
    print(f"Action 2: {action2['type']}, score: {score2:.4f}")
    print(f"  Context after step 2: {agent.last_context is not None}")
    
    # Third action (context should evolve)
    frame3 = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
    action3, score3 = agent.select_action(frame3)
    print(f"Action 3: {action3['type']}, score: {score3:.4f}")
    
    assert agent.last_context is not None, "Context should be generated"
    assert agent.last_z is not None, "Last latent should be tracked"
    assert agent.last_action_vec is not None, "Last action should be tracked"
    
    print("✓ Temporal context generation working")


def test_episodic_retrieval_integration():
    """Test that episodic retrieval works during action selection."""
    print("\n=== Test 3: Episodic Retrieval Integration ===")
    
    world_model = WorldModel(img_size=128, latent_dim=256, proj_dim=64)
    temporal_brain = TemporalBrain(latent_dim=256, action_dim=6)
    
    agent = CuriousAgent(
        world_model=world_model,
        temporal_brain=temporal_brain,
        use_episodic_retrieval=True,
        n_candidates=3
    )
    
    # Add some states to memory
    for i in range(10):
        frame = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
        agent.select_action(frame)
    
    print(f"Episodic memory size: {agent.memory.size}")
    
    # Check retrieval stats
    if agent.episodic_retrieval:
        stats = agent.episodic_retrieval.get_retrieval_stats()
        print(f"Retrieval stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    assert agent.memory.size >= 10, "Memory should have states"
    
    print("✓ Episodic retrieval integration working")


def test_state_persistence():
    """Test saving and loading agent with temporal brain."""
    print("\n=== Test 4: State Persistence ===")
    
    world_model = WorldModel(img_size=128, latent_dim=256, proj_dim=64)
    temporal_brain = TemporalBrain(latent_dim=256, action_dim=6)
    
    agent1 = CuriousAgent(
        world_model=world_model,
        temporal_brain=temporal_brain,
        use_episodic_retrieval=True,
        n_candidates=3
    )
    
    # Run a few steps
    for i in range(5):
        frame = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
        agent1.select_action(frame)
    
    # Save state
    state_dir = "test_agent_state"
    agent1.save_state(state_dir)
    print(f"Saved state to {state_dir}")
    
    # Create new agent and load
    agent2 = CuriousAgent(
        world_model=world_model,
        temporal_brain=TemporalBrain(latent_dim=256, action_dim=6),
        use_episodic_retrieval=True
    )
    
    agent2.load_state(state_dir)
    print(f"Loaded state from {state_dir}")
    
    # Verify memory was loaded
    print(f"  Memory size after load: {agent2.memory.size}")
    assert agent2.memory.size == agent1.memory.size
    
    # Cleanup
    import shutil
    if os.path.exists(state_dir):
        shutil.rmtree(state_dir)
    
    print("✓ State persistence working")


def test_without_temporal_brain():
    """Test that agent still works without temporal brain (backward compatibility)."""
    print("\n=== Test 5: Backward Compatibility (No Temporal Brain) ===")
    
    world_model = WorldModel(img_size=128, latent_dim=256, proj_dim=64)
    
    # Create agent WITHOUT temporal brain
    agent = CuriousAgent(
        world_model=world_model,
        temporal_brain=None,  # No temporal brain
        use_episodic_retrieval=False,  # No retrieval either
        n_candidates=3
    )
    
    # Should still work
    frame = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
    action, score = agent.select_action(frame)
    
    print(f"Action: {action['type']}, score: {score:.4f}")
    print(f"  Temporal brain: {agent.temporal_brain}")
    print(f"  Episodic retrieval: {agent.episodic_retrieval}")
    
    assert agent.temporal_brain is None
    assert agent.episodic_retrieval is None
    
    print("✓ Backward compatibility maintained")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing CuriousAgent + Temporal Brain Integration")
    print("=" * 60)
    
    try:
        test_agent_with_temporal_brain()
        test_action_selection_with_context()
        test_episodic_retrieval_integration()
        test_state_persistence()
        test_without_temporal_brain()
        
        print("\n" + "=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED")
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
