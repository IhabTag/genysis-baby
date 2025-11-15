import time
from typing import Any, Dict, Tuple

import numpy as np

from .obs import ScreenCapturer
from .actions import ActionExecutor
from .logging import EpisodeLogger


class ComputerEnv:
    """
    Minimal gym-like environment for the AGI baby's world.

    Observations are now dicts:

        {
            "image": np.ndarray (H,W,3) RGB uint8,
            "t": step_index
        }

    This matches all Phase-3 agents, instruction agents,
    text-action agents, curiosity modules, etc.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 1024,
        max_steps: int = 200,
        log_dir: str = "logs",
        action_delay: float = 0.02,
    ):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.action_delay = action_delay

        # Sensorimotor components
        self.capturer = ScreenCapturer(width, height)
        self.executor = ActionExecutor(width, height)

        # Logging
        self.logger = EpisodeLogger(log_dir)

        self.t = 0
        self.active = False

    # --------------------------------------------------------
    # Internal: capture observation
    # --------------------------------------------------------
    def _observe(self) -> Dict[str, Any]:
        """
        Return a dict with:
          - image: RGB frame
          - t: current timestep
        """
        img = self.capturer.capture()
        return {
            "image": img,
            "t": self.t
        }

    # --------------------------------------------------------
    # Public API: reset / step
    # --------------------------------------------------------
    def reset(self, meta: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Start a new episode and return the first observation dict.
        """
        if meta is None:
            meta = {}

        self.logger.start_episode(meta)
        self.t = 0
        self.active = True

        obs = self._observe()
        return obs

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute `action`, capture next observation, and return:
          (obs_next_dict, reward, done, info)
        """
        if not self.active:
            raise RuntimeError("Environment not active. Call reset() before step().")

        # 1. Execute action
        self.executor.execute(action)
        if self.action_delay > 0:
            time.sleep(self.action_delay)

        # 2. Increment step counter
        self.t += 1

        # 3. Capture new observation dict
        obs_next = self._observe()

        # 4. Reward + termination
        reward = 0.0
        done = self.t >= self.max_steps
        info: Dict[str, Any] = {}

        # 5. Log step
        self.logger.log_step(obs_next, action, reward, done, info)

        # 6. If episode ended, close logger
        if done:
            self.logger.end_episode()
            self.active = False

        return obs_next, reward, done, info
