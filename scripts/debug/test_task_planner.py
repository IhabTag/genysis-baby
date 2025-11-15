import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch

from env.computer_env import ComputerEnv
from agent.curious_agent import CuriousAgent
from agent.text_actions import TextActionAgent
from agent.instruction_agent import InstructionAgent
from agent.task_planner import TaskPlanner
from agent.random_agent import random_action

from models.world_model import WorldModel


def main():
    env = ComputerEnv()
    obs = env.reset()
    frame = obs["image"]

    wm = WorldModel()
    ckpt = torch.load("checkpoints/world_model_contrastive.pt", map_location="cpu")
    wm.load_state_dict(ckpt, strict=False)
    wm.eval()

    ca = CuriousAgent(
        world_model=wm,
        action_generator=lambda: random_action(width=1024, height=768),
        device="cpu",
    )

    taa = TextActionAgent(
        executor=env.executor,
        interpreter=ca.screen_interpreter,
        memory=ca.text_memory,
    )
    ca.attach_text_action_agent(taa)

    IA = InstructionAgent(
        curious_agent=ca,
        text_action_agent=taa,
        env=env
    )

    planner = TaskPlanner(IA, ca, taa, env)

    # Try multi-step commands
    while True:
        cmd = input("\nEnter multi-step task (or 'exit'): ").strip()
        if cmd == "exit":
            break

        planner.execute(cmd)


if __name__ == "__main__":
    main()
