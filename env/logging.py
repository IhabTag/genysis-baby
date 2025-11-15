import os
import json
import time
from typing import Any, Dict, Optional


class EpisodeLogger:
    """
    Minimal logger for episodes.

    It does NOT store full images (to avoid huge logs), only metadata:
      - step index
      - reward
      - done flag
      - action dict
      - obs_shape
      - optional info
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.current_episode_dir: Optional[str] = None
        self.current_log_path: Optional[str] = None
        self.current_file = None
        self.step_idx = 0

    def start_episode(self, meta: Dict[str, Any]) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.current_episode_dir = os.path.join(self.log_dir, f"ep_{ts}")
        os.makedirs(self.current_episode_dir, exist_ok=True)

        self.current_log_path = os.path.join(self.current_episode_dir, "episode.jsonl")
        self.current_file = open(self.current_log_path, "w", encoding="utf-8")
        self.step_idx = 0

        header = {
            "type": "meta",
            "timestamp": ts,
            "meta": meta or {},
        }
        self.current_file.write(json.dumps(header) + "\n")
        self.current_file.flush()

    def log_step(
        self,
        obs,
        action: Dict[str, Any],
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        """
        Log a single step. `obs` is expected to be a numpy array (H, W, 3) but we only store shape.
        """
        if self.current_file is None:
            return

        obs_shape = getattr(obs, "shape", None)

        rec = {
            "type": "step",
            "step": self.step_idx,
            "obs_shape": obs_shape,
            "action": action,
            "reward": float(reward),
            "done": bool(done),
            "info": info or {},
        }
        self.current_file.write(json.dumps(rec) + "\n")
        self.current_file.flush()

        self.step_idx += 1

    def end_episode(self) -> None:
        if self.current_file is not None:
            footer = {"type": "end", "step": self.step_idx}
            self.current_file.write(json.dumps(footer) + "\n")
            self.current_file.flush()
            self.current_file.close()

        self.current_file = None
        self.current_episode_dir = None
        self.current_log_path = None
        self.step_idx = 0
