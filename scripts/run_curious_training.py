import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from env.computer_env import ComputerEnv
from agent.curious_agent import CuriousAgent
from agent.random_agent import random_action
from models.world_model import WorldModel
from models.utils.preprocessing import ExperienceDataset
from models.utils.attention import SaliencyAttention
from scripts.train_world_model_contrastive import ContrastiveWorldModelLoss


CKPT_PATH = os.path.join(ROOT, "checkpoints", "world_model_contrastive.pt")


def save_episode_transitions(base_path, ep_idx, transitions):
    ep_path = os.path.join(base_path, f"ep_{ep_idx:06d}")
    os.makedirs(ep_path, exist_ok=True)

    for i, tr in enumerate(transitions):
        np.savez_compressed(
            os.path.join(ep_path, f"step_{i:06d}.npz"),
            obs_t=tr["obs_t"],
            obs_next=tr["obs_next"],
            action=tr["action"],
        )

    print(f"[run_curious_training] Saved episode {ep_idx} → {ep_path}")


def online_train_step(
    model,
    dataset_path,
    batch_size=8,
    lr=1e-4,
    epochs=1,
    device="cpu",
):
    ds = ExperienceDataset(dataset_path, img_size=128)
    if len(ds) == 0:
        print("[online_train_step] No samples found; skipping update.")
        return model

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = ContrastiveWorldModelLoss(
        lambda_recon=1.0,
        lambda_latent=0.1,
        lambda_contrastive=0.5,
        lambda_attn=0.3,
        temperature=0.1,
    )
    att_module = SaliencyAttention()

    print(f"[online_train_step] Training on {len(ds)} samples...")

    for epoch in range(epochs):
        total_loss = 0.0

        for img_t, act_vec, img_next in loader:
            img_t = img_t.to(device)
            act_vec = act_vec.to(device)
            img_next = img_next.to(device)

            z_t = model.encode(img_t)
            z_next_true = model.encode(img_next)

            p_t = model.project(z_t)
            p_next = model.project(z_next_true)

            pred_img_next, z_next_pred = model(img_t, act_vec)

            # Build attention maps for img_next
            B = img_next.size(0)
            att_maps = []
            for i in range(B):
                frame = img_next[i].permute(1, 2, 0).detach().cpu().numpy()
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                att_map, _ = att_module.compute(frame, prev_frame=None)
                att_maps.append(att_map)

            att_maps = np.stack(att_maps, axis=0)
            att_maps_t = torch.tensor(att_maps, dtype=torch.float32, device=device).unsqueeze(1)

            loss, _, _, _, _ = criterion(
                pred_img_next,
                img_next,
                z_next_pred,
                z_next_true,
                p_t,
                p_next,
                att_map=att_maps_t,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[online_train_step] Epoch {epoch+1}/{epochs} — loss={avg_loss:.6f}")

    return model


def main(num_episodes=3, max_steps=60, online_epochs=1):
    env = ComputerEnv()

    # Load or init world model
    model = WorldModel(action_dim=5, img_size=128)
    if os.path.exists(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
        print(f"Loaded checkpoint: {CKPT_PATH}")
    else:
        print("No checkpoint found, starting with randomly initialized model.")

    agent = CuriousAgent(
        world_model=model,
        action_generator=lambda: random_action(width=env.width, height=env.height),
        n_candidates=10,
        device="cpu",
        max_memory=1000,
        local_weight=1.0,
        novelty_weight=0.5,
        attention_weight=0.4,
        epsilon=0.1,
    )

    online_dir = os.path.join(ROOT, "datasets", "online_experience")
    os.makedirs(online_dir, exist_ok=True)

    for ep in range(1, num_episodes + 1):
        print(f"\n=== Online Episode {ep} ===")

        obs = env.reset(meta={"episode": ep})
        agent.remember_state(obs)
        time.sleep(0.1)

        transitions = []

        for step in range(max_steps):
            action, score = agent.select_action(obs)
            print(f"[EP {ep} STEP {step}] {action} | curiosity={score:.4f}")

            obs_t = obs
            obs_next, _, done, _ = env.step(action)

            transitions.append(
                {
                    "obs_t": np.asarray(obs_t, dtype=np.uint8),
                    "obs_next": np.asarray(obs_next, dtype=np.uint8),
                    "action": action,
                }
            )

            agent.remember_state(obs_next)
            obs = obs_next
            time.sleep(0.05)

            if done:
                break

        save_episode_transitions(online_dir, ep, transitions)

        print("Running online world-model update...")
        model = online_train_step(
            model,
            online_dir,
            epochs=online_epochs,
            device="cpu",
        )
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"Updated model saved to {CKPT_PATH}")

        agent.world_model = model
        agent.world_model.eval()

    print("\n[run_curious_training] Finished all episodes.")


if __name__ == "__main__":
    main()
