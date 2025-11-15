import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from env.computer_env import ComputerEnv
from agent.curious_agent import CuriousAgent
from agent.random_agent import random_action
from memory.replay_buffer import ReplayBuffer

from models.world_model import WorldModel
from models.utils.preprocessing import preprocess_frame, encode_action


# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BUFFER_SIZE = 10000
BATCH_SIZE = 32
TRAIN_EVERY = 50        # train every N environment steps
WARMUP_STEPS = 200      # start training after N steps in buffer
MAX_TOTAL_STEPS = 5000  # total environment interactions
LR = 1e-4

CKPT_PATH = "checkpoints/world_model_contrastive.pt"
CKPT_OUT = "checkpoints/world_model_online.pt"


def make_agent(world_model, env_width: int, env_height: int) -> CuriousAgent:
    agent = CuriousAgent(
        world_model=world_model,
        action_generator=lambda: random_action(width=env_width, height=env_height),
        device=DEVICE,
    )
    return agent


def train_step(
    world_model: WorldModel,
    buffer: ReplayBuffer,
    optimizer: optim.Optimizer,
    screen_width: int,
    screen_height: int,
) -> float:
    """
    One gradient step on a minibatch sampled from the replay buffer.

    Loss = recon_loss(predicted_frame vs next_frame)
         + latent_loss(predicted_latent vs encoded_latent_next)
    """
    world_model.train()

    obs_batch, actions_batch, next_obs_batch = buffer.sample(BATCH_SIZE)

    # Convert to tensors
    obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=DEVICE)       # (B,3,H,W)
    next_obs_t = torch.tensor(next_obs_batch, dtype=torch.float32, device=DEVICE)

    # Encode actions
    act_vecs = []
    for a in actions_batch:
        v = encode_action(a, screen_width=screen_width, screen_height=screen_height)
        act_vecs.append(v)
    act_vecs = np.stack(act_vecs, axis=0)  # (B, A)
    act_t = torch.tensor(act_vecs, dtype=torch.float32, device=DEVICE)

    # Forward
    z_t = world_model.encode(obs_t)                    # (B, latent_dim)
    z_next_true = world_model.encode(next_obs_t)       # (B, latent_dim)

    z_next_pred = world_model.predict_latent(z_t, act_t)
    pred_imgs, _ = world_model(obs_t, act_t)           # (B,3,H,W) predicted next frame

    recon_loss = torch.mean((pred_imgs - next_obs_t) ** 2)
    latent_loss = torch.mean((z_next_pred - z_next_true) ** 2)

    loss = recon_loss + 0.5 * latent_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=5.0)
    optimizer.step()

    world_model.eval()
    return loss.item()


def main():
    # --------------------------------------------------------
    # 1) Environment
    # --------------------------------------------------------
    env = ComputerEnv()
    obs = env.reset()
    frame = obs["image"]
    H, W, _ = frame.shape

    # --------------------------------------------------------
    # 2) World model + optimizer
    # --------------------------------------------------------
    wm = WorldModel().to(DEVICE)

    if os.path.exists(CKPT_PATH):
        print(f"Loading checkpoint: {CKPT_PATH}")
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        wm.load_state_dict(ckpt, strict=False)
    else:
        print(f"WARNING: No checkpoint found at {CKPT_PATH}, starting from scratch.")

    wm.eval()
    optimizer = optim.Adam(wm.parameters(), lr=LR)

    # --------------------------------------------------------
    # 3) Curious agent using this world model
    # --------------------------------------------------------
    agent = make_agent(wm, env_width=W, env_height=H)

    # --------------------------------------------------------
    # 4) Replay buffer
    # --------------------------------------------------------
    buffer = ReplayBuffer(max_size=BUFFER_SIZE)

    total_steps = 0
    episode_idx = 0

    print(f"Starting online lifelong learning on device={DEVICE} ...")

    while total_steps < MAX_TOTAL_STEPS:
        episode_idx += 1
        obs = env.reset()
        done = False

        print(f"\n=== EPISODE {episode_idx} ===")

        while not done and total_steps < MAX_TOTAL_STEPS:
            frame = obs["image"]  # (H,W,3) uint8

            # 1) Agent picks action (no grad)
            action, curiosity_score = agent.select_action(frame)

            # 2) Step in env
            next_obs, _, done, _ = env.step(action)
            next_frame = next_obs["image"]

            # 3) Store in replay buffer (preprocessed)
            obs_proc = preprocess_frame(frame)         # (3,128,128)
            next_obs_proc = preprocess_frame(next_frame)

            buffer.add(obs_proc, action, next_obs_proc)

            total_steps += 1

            # 4) Online training
            if len(buffer) >= WARMUP_STEPS and total_steps % TRAIN_EVERY == 0:
                loss = train_step(
                    wm,
                    buffer,
                    optimizer,
                    screen_width=W,
                    screen_height=H,
                )
                print(
                    f"[step {total_steps}] train_step loss={loss:.6f}, "
                    f"buffer_size={len(buffer)}, curiosity={curiosity_score:.4f}"
                )

            obs = next_obs

        # Optionally save intermediate checkpoints
        if episode_idx % 5 == 0:
            print(f"Saving intermediate checkpoint to {CKPT_OUT}")
            torch.save(wm.state_dict(), CKPT_OUT)

    print(f"\nTraining finished. Saving final checkpoint to {CKPT_OUT}")
    torch.save(wm.state_dict(), CKPT_OUT)


if __name__ == "__main__":
    main()
