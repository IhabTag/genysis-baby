"""
Train option policies via behavioral cloning.

Loads discovered options and trains micro-policies to imitate
the behavior observed in those segments.

Training method: Supervised learning
- Input: latent states from segments
- Target: actions taken in those segments
- Loss: MSE for actions + BCE for termination
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.world_model import WorldModel
from models.utils.preprocessing import preprocess_frame, encode_action
from agent.option_policy import OptionPolicy, OptionLibrary
import cv2


def load_episode_segment(
    episode_dir: str,
    start: int,
    end: int,
    world_model: WorldModel,
    device: str = "cpu"
) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    """
    Load latent states and actions from an episode segment.
    
    Args:
        episode_dir: Path to episode directory
        start, end: Segment boundaries
        world_model: WorldModel for encoding frames
        device: Device for tensors
    
    Returns:
        states: List of latent states z_t
        actions: List of action vectors
    """
    # Load metadata
    meta_path = os.path.join(episode_dir, "metadata.json")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    steps = metadata.get('steps', [])
    
    states = []
    actions = []
    
    world_model.eval()
    
    with torch.no_grad():
        for i in range(start, min(end, len(steps))):
            # Load frame
            frame_path = os.path.join(episode_dir, f"frame_{i:06d}.png")
            if not os.path.exists(frame_path):
                continue
            
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Encode to latent
            img_np = preprocess_frame(frame)
            img_t = torch.tensor(img_np, dtype=torch.float32, device=device).unsqueeze(0)
            z_t = world_model.encode(img_t)
            
            states.append(z_t.squeeze(0))
            
            # Get action
            step = steps[i]
            action = step.get('action', {'type': 'NOOP'})
            
            # Encode action
            # Get screen dimensions from frame
            h, w = frame.shape[:2]
            act_vec = encode_action(action, screen_width=w, screen_height=h)
            
            actions.append(act_vec)
    
    return states, actions


def train_option_policy(
    option_name: str,
    option_instances: List[Dict],
    world_model: WorldModel,
    num_epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    device: str = "cpu"
) -> OptionPolicy:
    """
    Train a micro-policy for a specific option type via behavioral cloning.
    
    Args:
        option_name: Name of option to train
        option_instances: List of option instances (from discovery)
        world_model: WorldModel for encoding frames
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device
    
    Returns:
        Trained OptionPolicy
    """
    print(f"\n{'='*60}")
    print(f"Training option: {option_name}")
    print(f"Instances: {len(option_instances)}")
    print(f"{'='*60}")
    
    # Create policy
    policy = OptionPolicy(
        option_name=option_name,
        latent_dim=256,
        context_dim=128,
        action_dim=6,
        use_context=False  # Don't use context for now
    ).to(device)
    
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    # Load all training data
    all_states = []
    all_actions = []
    all_is_last = []
    
    print("Loading training data...")
    for instance in tqdm(option_instances[:100]):  # Limit to 100 instances for speed
        episode_dir = instance['episode']
        start = instance['start']
        end = instance['end']
        
        try:
            states, actions = load_episode_segment(
                episode_dir, start, end, world_model, device
            )
            
            if len(states) < 2:
                continue
            
            all_states.extend(states)
            all_actions.extend(actions)
            
            # Mark last step of each instance
            is_last = [False] * len(states)
            is_last[-1] = True
            all_is_last.extend(is_last)
            
        except Exception as e:
            print(f"Error loading {episode_dir}: {e}")
            continue
    
    if len(all_states) == 0:
        print(f"No training data for {option_name}")
        return policy
    
    print(f"Loaded {len(all_states)} training samples")
    
    # Convert to tensors
    states_tensor = torch.stack(all_states)  # (N, latent_dim)
    actions_tensor = torch.tensor(
        np.array(all_actions), dtype=torch.float32, device=device
    )  # (N, action_dim)
    is_last_tensor = torch.tensor(
        all_is_last, dtype=torch.float32, device=device
    ).unsqueeze(1)  # (N, 1)
    
    # Training loop
    num_samples = len(states_tensor)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        policy.train()
        
        # Shuffle data
        indices = torch.randperm(num_samples)
        
        epoch_loss = 0.0
        epoch_action_loss = 0.0
        epoch_term_loss = 0.0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            batch_indices = indices[start_idx:end_idx]
            
            batch_states = states_tensor[batch_indices]
            batch_actions = actions_tensor[batch_indices]
            batch_is_last = is_last_tensor[batch_indices]
            
            # Forward pass
            pred_actions, pred_term = policy(batch_states)
            
            # Action loss (MSE)
            action_loss = nn.functional.mse_loss(pred_actions, batch_actions)
            
            # Termination loss (BCE)
            term_loss = nn.functional.binary_cross_entropy(pred_term, batch_is_last)
            
            # Combined loss
            total_loss = action_loss + 0.1 * term_loss
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_action_loss += action_loss.item()
            epoch_term_loss += term_loss.item()
        
        # Average losses
        epoch_loss /= num_batches
        epoch_action_loss /= num_batches
        epoch_term_loss /= num_batches
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}/{num_epochs}: "
                  f"loss={epoch_loss:.4f} "
                  f"(action={epoch_action_loss:.4f}, term={epoch_term_loss:.4f})")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
    
    policy.eval()
    print(f"Training complete. Best loss: {best_loss:.4f}")
    
    return policy


def main():
    """
    Main training script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train option policies")
    parser.add_argument('--options', type=str, required=True, 
                       help='Path to discovered options JSON')
    parser.add_argument('--world-model', type=str, default='checkpoints/world_model_contrastive.pt',
                       help='Path to world model checkpoint')
    parser.add_argument('--output', type=str, default='checkpoints/option_library.pt',
                       help='Output path for trained option library')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs per option')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--min-instances', type=int, default=5, 
                       help='Minimum instances required to train option')
    
    args = parser.parse_args()
    
    # Load discovered options
    print(f"Loading discovered options from {args.options}...")
    with open(args.options, 'r') as f:
        discovered_options = json.load(f)
    
    print(f"Found {len(discovered_options)} option types")
    
    # Load world model
    print(f"Loading world model from {args.world_model}...")
    world_model = WorldModel(img_size=128, latent_dim=256, proj_dim=64)
    
    if os.path.exists(args.world_model):
        state_dict = torch.load(args.world_model, map_location=args.device)
        world_model.load_state_dict(state_dict, strict=False)
        print("World model loaded")
    else:
        print("Warning: World model not found, using random weights")
    
    world_model = world_model.to(args.device)
    world_model.eval()
    
    # Create option library
    library = OptionLibrary(device=args.device)
    
    # Train each option type
    for option_name, instances in discovered_options.items():
        if len(instances) < args.min_instances:
            print(f"\nSkipping {option_name}: only {len(instances)} instances "
                  f"(min: {args.min_instances})")
            continue
        
        try:
            policy = train_option_policy(
                option_name=option_name,
                option_instances=instances,
                world_model=world_model,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=args.device
            )
            
            # Add to library
            library.add_option(
                option_name,
                policy,
                metadata={
                    'trained': True,
                    'instances': len(instances),
                    'epochs': args.epochs
                }
            )
            
        except Exception as e:
            print(f"Error training {option_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save library
    print(f"\nSaving option library to {args.output}...")
    library.save(args.output)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Trained {len(library)} option policies")
    print(f"Saved to {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
