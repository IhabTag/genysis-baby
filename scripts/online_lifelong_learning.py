import os
import sys
import json
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
    device = "cpu"
    print(f"Starting online lifelong learning on device={device} ...")

    CKPT_PATH = "checkpoints/world_model_contrastive.pt"
    STATE_DIR = "state"

    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)

    # Load world model
    world_model = WorldModel()
    if os.path.exists(CKPT_PATH):
        print("Loading checkpoint:", CKPT_PATH)
        sd = torch.load(CKPT_PATH, map_location=device)
        world_model.load_state_dict(sd, strict=False)
    else:
        print("No checkpoint found, starting from scratch.")

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
        proj_dim=64,
    )

    # Load persistent agent state (episodic + text + goals + meta)
    agent.load_state(STATE_DIR)
    print("Loaded agent state from", STATE_DIR)

    # Load age
    total_episodes, total_steps = load_age(STATE_DIR)
    years, months = pretty_age(total_steps)
    print(
        f"[age] Loaded age.json → episodes={total_episodes}, "
        f"steps={total_steps}, ~{months:.2f} months (~{years:.2f} years)"
    )

    num_episodes = 999999

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
            total_steps += 1  # ✅ global age counter

            if step % 10 == 0:
                print(
                    f"[ep {ep}] step={step} loss={loss:.4f} curiosity={curiosity:.4f} "
                    f"(global_steps={total_steps})"
                )

        # Episode finished
        total_episodes += 1

        # Save model + agent brain
        torch.save(world_model.state_dict(), CKPT_PATH)
        agent.save_state(STATE_DIR)

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
