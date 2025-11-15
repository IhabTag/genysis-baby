import os
import sys
import time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from env.computer_env import ComputerEnv
from agent.random_agent import random_action


def save_transition(path, step_id, obs_t, action, obs_next):
    """
    Save a single transition to compressed npz:
      - obs_t:    (H,W,3) uint8
      - obs_next: (H,W,3) uint8
      - action:   dict
    """
    os.makedirs(path, exist_ok=True)

    obs_t_arr = np.asarray(obs_t, dtype=np.uint8).copy()
    obs_next_arr = np.asarray(obs_next, dtype=np.uint8).copy()

    np.savez_compressed(
        os.path.join(path, f"step_{step_id:06d}.npz"),
        obs_t=obs_t_arr,
        obs_next=obs_next_arr,
        action=action,
    )


def collect_episodes(num_episodes: int = 5, max_steps: int = 80):
    env = ComputerEnv()
    base_path = os.path.join(ROOT, "datasets", "experience")
    os.makedirs(base_path, exist_ok=True)

    for ep in range(1, num_episodes + 1):
        print(f"\n=== EPISODE {ep} ===")

        ep_path = os.path.join(base_path, f"ep_{ep:06d}")
        os.makedirs(ep_path, exist_ok=True)

        obs = env.reset(meta={"episode": ep})
        time.sleep(0.1)

        for step in range(max_steps):
            action = random_action(width=env.width, height=env.height)

            obs_t = obs
            obs_next, _, done, _ = env.step(action)

            save_transition(ep_path, step, obs_t, action, obs_next)

            obs = obs_next
            time.sleep(0.05)

            if done:
                break

        print(f"Episode {ep} saved → {ep_path}")

    print("\n[collect_random_dataset] Collection complete.")


if __name__ == "__main__":
    collect_episodes(num_episodes=5, max_steps=80)
