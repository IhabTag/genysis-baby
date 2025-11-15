import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch
from env.computer_env import ComputerEnv
from agent.curious_agent import CuriousAgent
from agent.text_actions import TextActionAgent
from agent.random_agent import random_action
from models.world_model import WorldModel


def extract_frame(obs):
    """
    Supports both:
      A) obs = {"image": ...}
      B) obs = raw numpy image
    """
    if isinstance(obs, dict) and "image" in obs:
        return obs["image"]
    return obs  # assume raw image array


def main():
    # -------------------------------------------------------
    # 1) Environment
    # -------------------------------------------------------
    env = ComputerEnv()
    obs = env.reset()
    frame = extract_frame(obs)

    # -------------------------------------------------------
    # 2) Load world model
    # -------------------------------------------------------
    wm = WorldModel()
    ckpt_path = "checkpoints/world_model_contrastive.pt"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    wm.load_state_dict(ckpt, strict=False)
    wm.eval()

    # -------------------------------------------------------
    # 3) Create CuriousAgent
    # -------------------------------------------------------
    agent = CuriousAgent(
        world_model=wm,
        action_generator=lambda: random_action(width=1024, height=768),
        device="cpu",
    )

    # -------------------------------------------------------
    # 4) Attach semantic action module
    # -------------------------------------------------------
    taa = TextActionAgent(
        executor=env.executor,
        interpreter=agent.screen_interpreter,
        memory=agent.text_memory,
    )
    agent.attach_text_action_agent(taa)

    # -------------------------------------------------------
    # 5) Visible text
    # -------------------------------------------------------
    print("\n=== Visible Text Elements ===")
    visible = agent.get_visible_text(frame)
    for e in visible:
        print(f"Text='{e['text']}', center={e['center']}, role={e['role']}")

    # -------------------------------------------------------
    # 6) Semantic click test
    # -------------------------------------------------------
    target = input("\nEnter text to click: ").strip()
    print(f"[Test] Trying to click '{target}'...")
    agent.semantic_find_and_click(frame, target)

    time.sleep(0.2)

    # -------------------------------------------------------
    # 7) Semantic typing test
    # -------------------------------------------------------
    target2 = input("\nEnter target text to type into: ").strip()
    content = input("Enter content to type: ")

    agent.semantic_type_into(frame, target2, content)

    print("\nDone.")


if __name__ == "__main__":
    main()
