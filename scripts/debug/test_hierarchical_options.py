"""
Test Phase 5: Hierarchical Options

Tests:
1. Option policy creation and forward pass
2. Option library management
3. Hierarchical agent creation
4. Option selection and execution
5. State persistence
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
from agent.option_policy import OptionPolicy, OptionLibrary, create_default_options
from agent.hierarchical_agent import HierarchicalAgent


def test_option_policy():
    """Test option policy creation and forward pass."""
    print("\n=== Test 1: Option Policy ===")
    
    policy = OptionPolicy(
        option_name="scroll_down",
        latent_dim=256,
        context_dim=128,
        action_dim=6
    )
    
    # Test forward pass
    z_t = torch.randn(1, 256)
    c_t = torch.randn(1, 128)
    
    action_params, terminate_prob = policy(z_t, c_t)
    
    print(f"Action params shape: {action_params.shape}")
    print(f"Terminate prob: {terminate_prob.item():.3f}")
    
    assert action_params.shape == (1, 6)
    assert 0 <= terminate_prob.item() <= 1
    
    # Test action decoding
    action = policy.decode_action(action_params[0])
    print(f"Decoded action: {action}")
    
    assert 'type' in action
    assert action['type'] == 'SCROLL'
    
    print("✓ Option policy working")


def test_option_library():
    """Test option library management."""
    print("\n=== Test 2: Option Library ===")
    
    library = OptionLibrary(device="cpu")
    
    # Add some options
    for opt_name in ["scroll_down", "scroll_up", "move_mouse"]:
        policy = OptionPolicy(option_name=opt_name)
        library.add_option(opt_name, policy, metadata={'test': True})
    
    print(f"Library size: {len(library)}")
    print(f"Options: {library.list_options()}")
    
    assert len(library) == 3
    assert "scroll_down" in library
    
    # Test get
    policy = library.get_option("scroll_down")
    assert policy is not None
    assert policy.option_name == "scroll_down"
    
    # Test metadata
    meta = library.get_metadata("scroll_down")
    assert meta['test'] == True
    
    print("✓ Option library working")


def test_save_load_library():
    """Test saving and loading option library."""
    print("\n=== Test 3: Save/Load Library ===")
    
    # Create and save
    library1 = OptionLibrary()
    for opt_name in ["scroll_down", "move_mouse"]:
        policy = OptionPolicy(option_name=opt_name)
        library1.add_option(opt_name, policy)
    
    save_path = "test_library.pt"
    library1.save(save_path)
    
    # Load
    library2 = OptionLibrary()
    library2.load(save_path)
    
    print(f"Loaded {len(library2)} options")
    assert len(library2) == len(library1)
    assert set(library2.list_options()) == set(library1.list_options())
    
    # Cleanup
    os.remove(save_path)
    
    print("✓ Save/load working")


def test_default_options():
    """Test creating default option library."""
    print("\n=== Test 4: Default Options ===")
    
    library = create_default_options()
    
    print(f"Default library has {len(library)} options")
    print(f"Options: {library.list_options()}")
    
    assert len(library) > 5
    assert "scroll_down" in library
    assert "scroll_up" in library
    assert "move_mouse" in library
    
    print("✓ Default options created")


def test_hierarchical_agent_creation():
    """Test creating hierarchical agent."""
    print("\n=== Test 5: Hierarchical Agent Creation ===")
    
    world_model = WorldModel(img_size=128, latent_dim=256, proj_dim=64)
    option_library = create_default_options()
    
    agent = HierarchicalAgent(
        world_model=world_model,
        option_library=option_library,
        device="cpu"
    )
    
    print(f"Agent created")
    print(f"  Options available: {len(agent.option_library)}")
    print(f"  Current option: {agent.current_option}")
    print(f"  Memory size: {agent.memory.size}")
    
    assert agent.current_option is None
    assert agent.options_executed == 0
    
    print("✓ Hierarchical agent created")


def test_option_selection_and_execution():
    """Test option selection and execution."""
    print("\n=== Test 6: Option Selection & Execution ===")
    
    world_model = WorldModel(img_size=128, latent_dim=256, proj_dim=64)
    option_library = create_default_options()
    
    agent = HierarchicalAgent(
        world_model=world_model,
        option_library=option_library,
        epsilon=0.0,  # No random selection
        device="cpu"
    )
    
    # Execute several steps
    for i in range(10):
        frame = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
        action, info = agent.step(frame)
        
        option_name = info['option'] if info['option'] else 'None'
        print(f"Step {i}: {action['type']:12s} | "
              f"Option: {option_name:15s} | "
              f"Option step: {info['option_step']}")
        
        assert 'type' in action
        assert info['option'] is not None
    
    stats = agent.get_statistics()
    print(f"\nStatistics:")
    print(f"  Options executed: {stats['options_executed']}")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Avg option length: {stats['avg_option_length']:.1f}")
    print(f"  Option counts: {stats['option_counts']}")
    
    assert stats['options_executed'] >= 1
    assert stats['total_steps'] == 10
    
    print("✓ Option selection and execution working")


def test_hierarchical_with_temporal_brain():
    """Test hierarchical agent with temporal brain."""
    print("\n=== Test 7: Hierarchical + Temporal Brain ===")
    
    world_model = WorldModel(img_size=128, latent_dim=256, proj_dim=64)
    temporal_brain = TemporalBrain(latent_dim=256, action_dim=6)
    option_library = create_default_options()
    
    agent = HierarchicalAgent(
        world_model=world_model,
        option_library=option_library,
        temporal_brain=temporal_brain,
        device="cpu"
    )
    
    # Execute steps
    for i in range(5):
        frame = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
        action, info = agent.step(frame)
        
        print(f"Step {i}: {action['type']} (option: {info['option']})")
    
    assert agent.last_z is not None
    assert agent.last_action_vec is not None
    
    print("✓ Temporal brain integration working")


def test_hierarchical_reset():
    """Test agent reset."""
    print("\n=== Test 8: Agent Reset ===")
    
    world_model = WorldModel(img_size=128, latent_dim=256, proj_dim=64)
    option_library = create_default_options()
    
    agent = HierarchicalAgent(
        world_model=world_model,
        option_library=option_library
    )
    
    # Execute some steps
    for i in range(3):
        frame = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
        agent.step(frame)
    
    # Reset
    agent.reset()
    
    print("Agent reset")
    print(f"  Current option: {agent.current_option}")
    print(f"  Option step: {agent.option_step}")
    
    assert agent.current_option is None
    assert agent.option_step == 0
    assert agent.last_z is None
    
    print("✓ Reset working")


def test_state_persistence():
    """Test saving and loading agent state."""
    print("\n=== Test 9: State Persistence ===")
    
    world_model = WorldModel(img_size=128, latent_dim=256, proj_dim=64)
    option_library = create_default_options()
    
    agent1 = HierarchicalAgent(
        world_model=world_model,
        option_library=option_library
    )
    
    # Execute steps
    for i in range(5):
        frame = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
        agent1.step(frame)
    
    # Save
    state_dir = "test_hierarchical_state"
    agent1.save_state(state_dir)
    
    # Load
    agent2 = HierarchicalAgent(
        world_model=world_model,
        option_library=option_library
    )
    agent2.load_state(state_dir)
    
    print(f"Memory size after load: {agent2.memory.size}")
    assert agent2.memory.size == agent1.memory.size
    
    # Cleanup
    import shutil
    if os.path.exists(state_dir):
        shutil.rmtree(state_dir)
    
    print("✓ State persistence working")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Phase 5: Hierarchical Options")
    print("=" * 60)
    
    try:
        test_option_policy()
        test_option_library()
        test_save_load_library()
        test_default_options()
        test_hierarchical_agent_creation()
        test_option_selection_and_execution()
        test_hierarchical_with_temporal_brain()
        test_hierarchical_reset()
        test_state_persistence()
        
        print("\n" + "=" * 60)
        print("✅ ALL PHASE 5 TESTS PASSED")
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
