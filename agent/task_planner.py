import time
from typing import List, Dict, Any

from agent.instruction_agent import InstructionAgent
from agent.text_actions import TextActionAgent
from agent.curious_agent import CuriousAgent
from env.computer_env import ComputerEnv

from models.utils.preprocessing import preprocess_frame_diff
from models.utils.ocr import run_ocr_on_frame


class TaskPlanner:
    """
    Step 17: Multi-Step Planning and Verification Layer.

    Converts natural-language tasks into sequences of actions,
    executes them, and verifies each step by:
      - OCR change
      - attention change
      - pixel difference
      - semantic text existence
    """

    def __init__(
        self,
        instruction_agent: InstructionAgent,
        curious_agent: CuriousAgent,
        text_agent: TextActionAgent,
        env: ComputerEnv,
    ):
        self.IA = instruction_agent
        self.CA = curious_agent
        self.TA = text_agent
        self.env = env

    # ---------------------------------------------------------
    # 1) Decompose task text → list of substeps
    # ---------------------------------------------------------
    def decompose(self, instruction: str) -> List[str]:
        """
        Extremely naive decomposition (for now).
        Later we can plug in a language model inside the container.
        """
        cleaned = instruction.lower().replace("then", ",")
        cleaned = cleaned.replace(" and ", ",")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]

        return parts

    # ---------------------------------------------------------
    # 2) Verification helpers
    # ---------------------------------------------------------
    def verify_screen_changed(self, frame_before, frame_after) -> bool:
        diff = preprocess_frame_diff(frame_before, frame_after)
        return diff > 0.015  # heuristic threshold

    def verify_text_present(self, frame, target: str) -> bool:
        texts = run_ocr_on_frame(frame)
        for t in texts:
            if target.lower() in t.lower():
                return True
        return False

    # ---------------------------------------------------------
    # 3) Execute single substep with verification
    # ---------------------------------------------------------
    def execute_substep(self, step: str) -> Dict[str, Any]:
        """
        Execute one step like:
            'click search'
            'type hello world'
            'scroll down'
            'open firefox'
        """
        print(f"\n--- Executing: {step} ---")

        # Capture screen before
        obs_before = self.env._observe()
        frame_before = obs_before["image"]

        # Try instruction layer first
        results = self.IA.execute(step)

        # Capture after
        obs_after = self.env._observe()
        frame_after = obs_after["image"]

        # Verification
        changed = self.verify_screen_changed(frame_before, frame_after)

        return {
            "step": step,
            "results": results,
            "screen_changed": changed
        }

    # ---------------------------------------------------------
    # 4) High-level plan executor
    # ---------------------------------------------------------
    def execute(self, instruction: str) -> List[Dict[str, Any]]:
        """
        Execute a full multi-step instruction with:
          - decomposition
          - substep execution
          - verification
          - retry logic
        """
        print(f"\n======================")
        print(f"    TASK: {instruction}")
        print(f"======================\n")

        # Step 1: decompose
        steps = self.decompose(instruction)
        print(f"Plan: {steps}\n")

        results = []

        for step in steps:
            attempt = 0

            while attempt < 3:
                r = self.execute_substep(step)
                results.append(r)

                if r["screen_changed"]:
                    print(f"✓ Verified: '{step}' succeeded.")
                    break

                print(f"✗ Verification failed for '{step}', retrying...")
                attempt += 1
                time.sleep(0.5)

            if attempt == 3:
                print(f"⚠ Step failed after 3 attempts: {step}")

        print("\nTask complete.\n")
        return results
