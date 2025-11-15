import os
import sys
import time
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from env.computer_env import ComputerEnv
from agent.curious_agent import CuriousAgent
from agent.text_actions import TextActionAgent
from agent.instruction_agent import InstructionAgent
from agent.random_agent import random_action
from models.world_model import WorldModel


def main():
    # -------------------------------------------------------
    # 1) Environment & Executor
    # -------------------------------------------------------
    env = ComputerEnv()
    obs = env.reset()          # ✅ obs is a dict: {"image": img, "t": step}
    frame = obs["image"]       # FIXED: this now always works

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
    # 4) Attach TextActionAgent
    # -------------------------------------------------------
    taa = TextActionAgent(
        executor=env.executor,
        interpreter=agent.screen_interpreter,
        memory=agent.text_memory,
    )
    agent.attach_text_action_agent(taa)

    # -------------------------------------------------------
    # 5) Print visible text
    # -------------------------------------------------------
    print("\n=== Visible text ===")
    elems = agent.get_visible_text(frame)
    for e in elems:
        print(f"Text='{e['text']}' | center={e['center']} | role={e['role']}")

    # -------------------------------------------------------
    # 6) Instruction agent
    # -------------------------------------------------------
    IA = InstructionAgent(
        curious_agent=agent,
        text_action_agent=taa,
        env=env,
    )

    # -------------------------------------------------------
    # 7) Command loop
    # -------------------------------------------------------
    while True:
        cmd = input("\nEnter instruction (or 'exit'): ").strip()
        if cmd.lower() == "exit":
            break

        # Get fresh frame before executing
        obs = env._observe()
        frame = obs["image"]

        results = IA.execute(cmd)

        print("\nExecution results:")
        for r in results:
            print(r)

        time.sleep(0.5)


if __name__ == "__main__":
    main()
