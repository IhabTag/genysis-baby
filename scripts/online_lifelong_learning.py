import os
import sys
import json
import torch
import numpy as np
import argparse

# ------------------------------------------------------------
# Ensure project root is importable
# ------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from env.computer_env import ComputerEnv
from agent.curious_agent import CuriousAgent
from agent.hierarchical_agent import HierarchicalAgent
from models.world_model import WorldModel
from models.temporal_brain import TemporalBrain
from models.utils.preprocessing import preprocess_frame, encode_action
from agent.random_agent import random_action
from agent.option_policy import OptionLibrary, create_default_options


# ------------------------------------------------------------
# Observation Normalizer (Option A)
# ------------------------------------------------------------
def unwrap_obs(obs):
    """
    Safely extract an image array (H,W,3) from an observation.
    Compatible with:
      - obs = ndarray
      - obs = {"image": ndarray, ...}
      - obs = {"obs": ndarray, "t": ...}
    """
    if isinstance(obs, dict):
        if "image" in obs:
            return obs["image"]

        for v in obs.values():
            if isinstance(v, np.ndarray) and v.ndim == 3:
                return v

        return None

    if isinstance(obs, np.ndarray):
        return obs

    return None


# ------------------------------------------------------
# Training step
# ------------------------------------------------------
def train_step(world_model, optimizer, frame_t, frame_next, act_vec, device):
    """
    Computes world model reconstruction + latent prediction loss.
    """
    world_model.train()

    img_t = torch.tensor(
        preprocess_frame(frame_t), dtype=torch.float32, device=device
    ).unsqueeze(0)

    img_next = torch.tensor(
        preprocess_frame(frame_next), dtype=torch.float32, device=device
    ).unsqueeze(0)

    act_vec_t = torch.tensor(
        act_vec, dtype=torch.float32, device=device
    ).unsqueeze(0)

    pred_img, z_t = world_model(img_t, act_vec_t)

    recon_loss = torch.mean((pred_img - img_next) ** 2)

    # latent consistency
    z_next_pred = world_model.predict_latent(z_t, act_vec_t)
    z_next_true = world_model.encode(img_next)
    latent_loss = torch.mean((z_next_pred - z_next_true) ** 2)

    total = recon_loss + 0.1 * latent_loss

    optimizer.zero_grad()
    total.backward()
    optimizer.step()

    return total.item()


# ------------------------------------------------------
# Age persistence helpers
# ------------------------------------------------------
def load_age(state_dir: str):
    """
    Load total_episodes and total_steps from age.json if present.
    Returns (total_episodes, total_steps).
    """
    path = os.path.join(state_dir, "age.json")
    if not os.path.exists(path):
        return 0, 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        total_episodes = int(data.get("total_episodes", 0))
        total_steps = int(data.get("total_steps", 0))
        return total_episodes, total_steps
    except Exception as e:
        print(f"[age] Warning: failed to load age.json: {e}")
        return 0, 0


def save_age(state_dir: str, total_episodes: int, total_steps: int):
    """
    Save total_episodes and total_steps to age.json.
    Also stores a derived estimated cognitive age in months.
    """
    path = os.path.join(state_dir, "age.json")
    os.makedirs(state_dir, exist_ok=True)

    # Heuristic: every 50k steps ≈ 1 human cognitive month
    estimated_months = total_steps / 50_000.0
    data = {
        "total_episodes": total_episodes,
        "total_steps": total_steps,
        "estimated_cognitive_age_months": estimated_months,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def pretty_age(total_steps: int):
    """
    Make a small human-readable age estimate from total_steps.
    """
    months = total_steps / 50_000.0
    years = months / 12.0
    return years, months


# ------------------------------------------------------
# Main loop (persistent lifelong learning)
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Online lifelong learning for GENYSIS-BABY")
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--use-temporal', action='store_true', default=True, help='Enable temporal brain (default: True)')
    parser.add_argument('--no-temporal', action='store_false', dest='use_temporal', help='Disable temporal brain')
    parser.add_argument('--use-hierarchical', action='store_true', default=True, help='Enable hierarchical agent (default: True)')
    parser.add_argument('--no-hierarchical', action='store_false', dest='use_hierarchical', help='Disable hierarchical agent')
    parser.add_argument('--option-library', type=str, default=None, 
                       help='Path to trained option library (for hierarchical mode)')
    parser.add_argument('--episodes', type=int, default=999999, help='Number of episodes to run')
    parser.add_argument('--fast-mode', action='store_true', default=True, help='Fast mode (skip OCR on predictions)')
    
    args = parser.parse_args()
    
    device = args.device
    print(f"Starting online lifelong learning on device={device}")
    print(f"  Temporal brain: {args.use_temporal}")
    print(f"  Hierarchical agent: {args.use_hierarchical}")
    print(f"  Fast mode: {args.fast_mode}")

    CKPT_PATH = "checkpoints/world_model_contrastive.pt"
    STATE_DIR = "state"

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)

    # Load world model
    world_model = WorldModel(action_dim=7)
    if os.path.exists(CKPT_PATH):
        print("Loading checkpoint:", CKPT_PATH)
        sd = torch.load(CKPT_PATH, map_location=device)
        
        # Check for dimension mismatch in dynamics model
        if 'dynamics.0.weight' in sd:
            saved_weight = sd['dynamics.0.weight']
            current_weight = world_model.dynamics[0].weight
            
            if saved_weight.shape != current_weight.shape:
                print(f"Warning: Shape mismatch in dynamics model ({saved_weight.shape} vs {current_weight.shape}). Adapting weights...")
                
                # Create new weight tensor with current shape
                new_weight = current_weight.data.clone()
                
                # Assuming latent_dim=256 is first part of input
                latent_dim = 256
                
                # Copy latent part (0:256)
                new_weight[:, :latent_dim] = saved_weight[:, :latent_dim]
                
                # Copy old action part (256:256+6)
                # Old shape was [256, 262], so action part is 256:262
                # New shape is [256, 263]
                old_action_dim = saved_weight.shape[1] - latent_dim
                new_weight[:, latent_dim:latent_dim+old_action_dim] = saved_weight[:, latent_dim:]
                
                # Zero out the new action dimension(s)
                new_weight[:, latent_dim+old_action_dim:].zero_()
                
                # Update state dict
                sd['dynamics.0.weight'] = new_weight
                
        world_model.load_state_dict(sd, strict=False)
    else:
        print("No checkpoint found, starting from scratch.")

    world_model = world_model.to(device)
    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-4)

    # Create environment
    env = ComputerEnv(width=1024, height=768, max_steps=150)

    # Create temporal brain if requested
    temporal_brain = None
    if args.use_temporal:
        print("Creating temporal brain (GRU)...")
        temporal_brain = TemporalBrain(
            latent_dim=256,
            action_dim=7,
            hidden_dim=256,
            context_dim=128,
            model_type="gru"
        )
        temporal_brain = temporal_brain.to(device)
        temporal_brain.eval()
        
        # Try to load temporal brain state
        temporal_path = os.path.join(STATE_DIR, "temporal_brain.pt")
        if os.path.exists(temporal_path):
            checkpoint = torch.load(temporal_path, map_location=device)
            sd = checkpoint['model_state_dict']
            
            # Handle weight mismatch for RNN input layer
            if 'rnn.weight_ih_l0' in sd:
                saved_weight = sd['rnn.weight_ih_l0']
                current_weight = temporal_brain.rnn.weight_ih_l0
                
                if saved_weight.shape != current_weight.shape:
                    print(f"Warning: Shape mismatch in TemporalBrain ({saved_weight.shape} vs {current_weight.shape}). Adapting weights...")
                    
                    # Create new weight tensor
                    new_weight = current_weight.data.clone()
                    
                    # Input is [latent_dim, action_dim]
                    # RNN weight_ih is [3*hidden, input_dim] (for GRU)
                    latent_dim = 256
                    old_action_dim = saved_weight.shape[1] - latent_dim
                    
                    # Copy latent part
                    new_weight[:, :latent_dim] = saved_weight[:, :latent_dim]
                    
                    # Copy old action part
                    new_weight[:, latent_dim:latent_dim+old_action_dim] = saved_weight[:, latent_dim:]
                    
                    # Zero new action part
                    new_weight[:, latent_dim+old_action_dim:].zero_()
                    
                    sd['rnn.weight_ih_l0'] = new_weight
            
            temporal_brain.load_state_dict(sd, strict=False)
            # Reset hidden state to avoid shape mismatch there too
            temporal_brain.reset_hidden() 
            print(f"Loaded temporal brain from {temporal_path}")

    # Create agent
    if args.use_hierarchical:
        # Load or create option library
        if args.option_library and os.path.exists(args.option_library):
            print(f"Loading option library from {args.option_library}...")
            option_library = OptionLibrary(device=device)
            option_library.load(args.option_library)
        else:
            print("Creating default option library...")
            option_library = create_default_options(device=device)
        
        print(f"Creating hierarchical agent with {len(option_library)} options...")
        agent = HierarchicalAgent(
            world_model=world_model,
            option_library=option_library,
            temporal_brain=temporal_brain,
            device=device,
            max_option_steps=30,  # Shorter options for more variety
            epsilon=0.2,  # More random exploration
            curiosity_weight=2.0,  # Higher curiosity
            novelty_weight=1.5,  # Higher novelty seeking
            diversity_bonus=0.5,  # Encourage trying different options
            exploration_boost=1.5  # Boost early exploration
        )
        
        # Load agent state
        agent.load_state(STATE_DIR)
        print("Loaded hierarchical agent state from", STATE_DIR)
        
    else:
        # Standard curious agent
        print("Creating standard curious agent...")
        agent = CuriousAgent(
            world_model=world_model,
            action_generator=lambda: random_action(width=1024, height=768),
            n_candidates=4,
            device=device,
            max_memory=1500,
            epsilon=0.05,
            proj_dim=64,
            fast_mode=args.fast_mode,
            mem_sample_size=512,
            temporal_brain=temporal_brain,
            use_episodic_retrieval=args.use_temporal
        )
        
        # Load persistent agent state
        agent.load_state(STATE_DIR)
        print("Loaded agent state from", STATE_DIR)

    # Load age
    total_episodes, total_steps = load_age(STATE_DIR)
    years, months = pretty_age(total_steps)
    print(
        f"[age] Loaded age.json → episodes={total_episodes}, "
        f"steps={total_steps}, ~{months:.2f} months (~{years:.2f} years)"
    )

    num_episodes = args.episodes

    for ep in range(total_episodes + 1, total_episodes + 1 + num_episodes):
        print(f"\n=== EPISODE {ep} ===")

        obs = env.reset(meta={"episode": ep})
        frame = unwrap_obs(obs)

        if frame is None:
            raise RuntimeError("Reset returned no valid frame")

        done = False
        step = 0

        while not done:
            # Agent decides action
            if args.use_hierarchical:
                action, info = agent.step(frame, screen_width=1024, screen_height=768)
                curiosity = 0.0  # Hierarchical agent doesn't return curiosity score
                
                # Print option info occasionally
                if step % 1 == 0 and info.get('option'):
                    print(f"  [Hierarchical] Option: {info['option']}, step: {info['option_step']}")
            else:
                action, curiosity = agent.select_action(frame)

            # Encode action for world model
            act_vec = encode_action(
                action, screen_width=1024, screen_height=768
            )

            # Environment step
            next_obs, reward, done, info = env.step(action)
            next_frame = unwrap_obs(next_obs)

            if next_frame is None:
                print("Warning: next_frame=None, skipping this transition.")
                continue

            # Train world model
            loss = train_step(
                world_model, optimizer,
                frame, next_frame,
                act_vec, device
            )

            # Update memory (only for standard agent)
            if not args.use_hierarchical:
                agent.remember_state(frame)

            # Advance
            frame = next_frame
            step += 1
            total_steps += 1  # global age counter

            if step % 1 == 0:
                print(
                    f"[ep {ep}] step={step} loss={loss:.4f} curiosity={curiosity:.4f} "
                    f"(global_steps={total_steps})"
                )

        # Episode finished
        total_episodes += 1

        # Save model + agent brain
        torch.save(world_model.state_dict(), CKPT_PATH)
        agent.save_state(STATE_DIR)
        
        # Save temporal brain if used
        if temporal_brain is not None:
            temporal_path = os.path.join(STATE_DIR, "temporal_brain.pt")
            temporal_brain.save_state(temporal_path)

        # Save age
        save_age(STATE_DIR, total_episodes, total_steps)
        years, months = pretty_age(total_steps)

        print(
            f"Episode {ep} finished. "
            f"Total episodes={total_episodes}, total steps={total_steps}, "
            f"estimated cognitive age ≈ {months:.2f} months (~{years:.2f} years). "
            f"State + age saved.\n"
        )


if __name__ == "__main__":
    main()
