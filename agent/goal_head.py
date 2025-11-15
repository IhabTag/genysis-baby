import numpy as np
from typing import Dict, Any, List
import hashlib
import os
import json


class GoalCuriosityHead:
    """
    A goal-discovery and goal-curiosity module.

    The agent receives intrinsic reward when it causes:
      - New windows or UI layouts
      - New OCR content
      - New application-like states
      - Structural changes in the screen

    Each "goal state" is summarized into a signature.
    Novelty of that signature drives goal curiosity.

    Now with save/load for persistence.
    """

    def __init__(self, max_goals: int = 5000):
        # A dictionary of goal signatures the agent has achieved
        self.known_goals: Dict[str, float] = {}
        self.max_goals = max_goals

    # -------------------------------------------------------------
    # Utility: compute hash fingerprint for any structure
    # -------------------------------------------------------------
    def _hash(self, obj: Any) -> str:
        h = hashlib.sha1(str(obj).encode()).hexdigest()
        return h

    # -------------------------------------------------------------
    # Extract structured features from screen interpretation
    # -------------------------------------------------------------
    def extract_features(
        self,
        ocr_tokens: List[str],
        layout_vec: np.ndarray,
        num_windows: int,
        cursor_mode: str,
        screen_elems: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Returns a structured feature dict representing "goal state".
        """
        return {
            "ocr": tuple(sorted(ocr_tokens)),
            "layout": tuple(np.round(layout_vec, 3).tolist()),
            "windows": num_windows,
            "cursor": cursor_mode,
            "elems": len(screen_elems),
        }

    # -------------------------------------------------------------
    # Compute goal novelty: how new is this state?
    # -------------------------------------------------------------
    def compute_goal_curiosity(self, features: Dict[str, Any]) -> float:
        """
        Computes novelty of the goal state based on its hash signature.
        High if never seen, decays as repeated.
        """

        sig = self._hash(features)

        if sig not in self.known_goals:
            novelty = 1.0  # maximum novelty
            if len(self.known_goals) < self.max_goals:
                self.known_goals[sig] = 1.0
        else:
            # Decrease novelty every time it reoccurs
            self.known_goals[sig] *= 0.9
            novelty = self.known_goals[sig]

        return float(novelty)

    # -------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------
    def to_dict(self) -> Dict[str, float]:
        return dict(self.known_goals)

    def from_dict(self, data: Dict[str, float]):
        if data is None:
            self.known_goals = {}
        else:
            self.known_goals = {str(k): float(v) for k, v in data.items()}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.from_dict(data)
