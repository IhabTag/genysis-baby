import time
from typing import Any, Dict, Tuple

import numpy as np

from .obs import ScreenCapturer
from .actions import ActionExecutor
from .logging import EpisodeLogger


class ComputerEnv:
    """
    A minimal gym-like environment representing the AGI baby's world:

      - Observation: RGB desktop frame (H, W, 3) as uint8
      - Action: dict interpreted by ActionExecutor
      - Reward: currently always 0.0 (curiosity is handled outside)
      - Done: episode length exceeded (max_steps)

    This is intentionally simple and "dumb".
    All intelligence is in the models, curiosity, and agents.
    """

    def __init__(
        self,
        width: int = 1024,
        height: int = 768,
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
    def _observe(self) -> np.ndarray:
        img = self.capturer.capture()
        return img

    # --------------------------------------------------------
    # Public API: reset / step
    # --------------------------------------------------------
    def reset(self, meta: Dict[str, Any] = None) -> np.ndarray:
        """
        Start a new episode and return initial observation (RGB image).
        """
        if meta is None:
            meta = {}

        self.logger.start_episode(meta)
        self.t = 0
        self.active = True

        obs = self._observe()
        return obs

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute `action`, capture next observation, and return:
          (obs_next, reward, done, info)

        - `reward` is always 0.0 here.
          Curiosity/intrinsic reward is computed in the agent/memory modules.
        """
        if not self.active:
            raise RuntimeError("Environment not active. Call reset() before step().")

        # 1. Execute action
        self.executor.execute(action)
        if self.action_delay > 0:
            time.sleep(self.action_delay)

        # 2. Increment step counter
        self.t += 1

        # 3. Capture new observation
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
