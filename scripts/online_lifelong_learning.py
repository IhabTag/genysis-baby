import os
import sys
import time
import torch
import numpy as np

# ------------------------------------------------------------
# Ensure project root is importable
# ------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from env.computer_env import ComputerEnv
from agent.curious_agent import CuriousAgent
from models.world_model import WorldModel
from models.utils.preprocessing import preprocess_frame, encode_action
from agent.random_agent import random_action


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
        # Common key
        if "image" in obs:
            return obs["image"]

        # Fallback: auto-detect the first 3D ndarray
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
# Main loop
# ------------------------------------------------------
def main():
    device = "cpu"
    print(f"Starting online lifelong learning on device={device} ...")

    # Load world model
    ckpt_path = "checkpoints/world_model_contrastive.pt"
    print("Loading checkpoint:", ckpt_path)
    world_model = WorldModel()
    world_model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    world_model = world_model.to(device)

    optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-4)

    # Create environment
    env = ComputerEnv(width=1024, height=768, max_steps=150)

    # Create agent
    agent = CuriousAgent(
        world_model=world_model,
        action_generator=lambda: random_action(width=1024, height=768),
        n_candidates=6,
        device=device,
        max_memory=1500,
        epsilon=0.05,
    )

    num_episodes = 999999

    for ep in range(1, num_episodes + 1):
        print(f"\n=== EPISODE {ep} ===")

        obs = env.reset(meta={"episode": ep})
        frame = unwrap_obs(obs)

        if frame is None:
            raise RuntimeError("Reset returned no valid frame")

        done = False
        step = 0

        while not done:
            # Agent decides action
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

            # Update memory
            agent.remember_state(frame)

            # Advance
            frame = next_frame
            step += 1

            if step % 10 == 0:
                print(
                    f"[ep {ep}] step={step} loss={loss:.4f} curiosity={curiosity:.4f}"
                )

        print(f"Episode {ep} finished.\n")


if __name__ == "__main__":
    main()
