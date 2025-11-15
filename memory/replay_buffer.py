import numpy as np
from typing import List, Dict, Any, Tuple


class ReplayBuffer:
    """
    Generic replay buffer for world model training.

    Stores:
      - obs: preprocessed image (3,H,W) float32
      - action: dict (raw action, will be encoded later)
      - next_obs: preprocessed image (3,H,W) float32
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size

        self.obs_buf: List[np.ndarray] = []
        self.next_obs_buf: List[np.ndarray] = []
        self.action_buf: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.obs_buf)

    def add(self, obs: np.ndarray, action: Dict[str, Any], next_obs: np.ndarray) -> None:
        """
        obs, next_obs: preprocessed images (3,H,W) float32 in [0,1]
        action: raw action dict
        """
        if len(self.obs_buf) >= self.max_size:
            # FIFO
            self.obs_buf.pop(0)
            self.next_obs_buf.pop(0)
            self.action_buf.pop(0)

        self.obs_buf.append(obs)
        self.next_obs_buf.append(next_obs)
        self.action_buf.append(action)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray]:
        assert len(self.obs_buf) >= batch_size, "Not enough samples in buffer"

        idxs = np.random.choice(len(self.obs_buf), size=batch_size, replace=False)

        obs_batch = np.stack([self.obs_buf[i] for i in idxs], axis=0)       # (B,3,H,W)
        next_obs_batch = np.stack([self.next_obs_buf[i] for i in idxs], axis=0)
        actions_batch = [self.action_buf[i] for i in idxs]

        return obs_batch, actions_batch, next_obs_batch
