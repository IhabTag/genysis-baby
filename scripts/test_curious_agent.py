import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from env.computer_env import ComputerEnv
from agent.curious_agent import CuriousAgent
from agent.random_agent import random_action
from models.world_model import WorldModel

CKPT_PATH = os.path.join(ROOT, "checkpoints", "world_model_contrastive.pt")


def main():
    env = ComputerEnv(max_steps=40)

    model = WorldModel(action_dim=5, img_size=128)
    if os.path.exists(CKPT_PATH):
        model.load_state_dict(
            __import__("torch").load(CKPT_PATH, map_location="cpu")
        )
        print(f"Loaded world model from {CKPT_PATH}")
    else:
        print("WARNING: No checkpoint found, using random weights.")

    agent = CuriousAgent(
        world_model=model,
        action_generator=lambda: random_action(width=env.width, height=env.height),
        n_candidates=8,
        device="cpu",
        max_memory=1000,
        local_weight=1.0,
        novelty_weight=0.5,
        attention_weight=0.4,
        epsilon=0.1,
    )

    obs = env.reset(meta={"mode": "test_curious_agent"})
    agent.remember_state(obs)
    time.sleep(0.1)

    for step in range(env.max_steps):
        action, score = agent.select_action(obs)
        print(f"[STEP {step}] {action} | curiosity={score:.4f}")

        obs_next, _, done, _ = env.step(action)
        agent.remember_state(obs_next)

        obs = obs_next
        time.sleep(0.05)

        if done:
            break

    print("[test_curious_agent] Done.")


if __name__ == "__main__":
    main()
