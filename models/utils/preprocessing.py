import os
import glob
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


# ------------------------------------------------------------
# Frame preprocessing
# ------------------------------------------------------------
def preprocess_frame(frame: np.ndarray, size: int = 128) -> np.ndarray:
    """
    Convert HxWx3 RGB uint8 frame to:
      - resized (size, size)
      - float32 in [0,1]
      - CHW format

    Returns: np.ndarray of shape (3, size, size), dtype float32
    """
    assert frame.ndim == 3 and frame.shape[2] == 3, "Expected RGB frame HxWx3"

    # Resize
    img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0  # [0,1]
    img = np.transpose(img, (2, 0, 1))    # HWC -> CHW
    return img


# ------------------------------------------------------------
# Action encoding
# ------------------------------------------------------------
ACTION_TYPES = [
    "MOVE_MOUSE",
    "LEFT_CLICK",
    "RIGHT_CLICK",
    "SCROLL",
    "TYPE_TEXT",
]


def encode_action(action: Dict[str, Any], screen_width: int = 1024, screen_height: int = 768) -> np.ndarray:
    """
    Encode an action dict into a fixed-size vector of length 5.

    We use:
      - type_id_norm: scalar in [0,1] (action type index / (N-1))
      - x_norm: x / screen_width  (0 if not applicable)
      - y_norm: y / screen_height (0 if not applicable)
      - scroll_norm: scroll amount / 100, clipped to [-1,1]
      - text_len_norm: len(text) / 20, clipped to [0,1]

    This is simple but enough for the world model to correlate actions with effects.
    """
    a_type = action.get("type", "NOOP")
    if a_type in ACTION_TYPES:
        type_idx = ACTION_TYPES.index(a_type)
    else:
        type_idx = 0

    if len(ACTION_TYPES) > 1:
        type_id_norm = type_idx / float(len(ACTION_TYPES) - 1)
    else:
        type_id_norm = 0.0

    # Coordinates (if move)
    x = float(action.get("x", 0.0))
    y = float(action.get("y", 0.0))
    x_norm = np.clip(x / max(screen_width, 1), 0.0, 1.0)
    y_norm = np.clip(y / max(screen_height, 1), 0.0, 1.0)

    # Scroll
    scroll = float(action.get("amount", 0.0))
    scroll_norm = np.clip(scroll / 100.0, -1.0, 1.0)

    # Text length
    text = action.get("text", "")
    text_len_norm = np.clip(len(str(text)) / 20.0, 0.0, 1.0)

    vec = np.array(
        [type_id_norm, x_norm, y_norm, scroll_norm, text_len_norm],
        dtype=np.float32,
    )
    return vec


# ------------------------------------------------------------
# Experience dataset (for training)
# ------------------------------------------------------------
class ExperienceDataset(Dataset):
    """
    Loads experience transitions saved as compressed npz files.

    Each file is expected to contain:
      - obs_t:   HxWx3 RGB uint8
      - obs_next: HxWx3 RGB uint8
      - action: Python dict

    The directory structure is:
      root/
        ep_000001/
          step_000000.npz
          step_000001.npz
        ep_000002/
          ...

    Returns tuples:
      (img_t, act_vec, img_next)

      img_t:    torch.FloatTensor (3, size, size) in [0,1]
      act_vec:  torch.FloatTensor (5,)
      img_next: torch.FloatTensor (3, size, size) in [0,1]
    """

    def __init__(self, root_dir: str, img_size: int = 128):
        self.root_dir = root_dir
        self.img_size = img_size

        pattern = os.path.join(self.root_dir, "ep_*", "step_*.npz")
        self.files: List[str] = sorted(glob.glob(pattern))

        print(f"[ExperienceDataset] Found {len(self.files)} npz files under {self.root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)

        obs_t = data["obs_t"]  # H,W,3
        obs_next = data["obs_next"]
        action = data["action"].item() if isinstance(data["action"], np.ndarray) else data["action"]

        # Preprocess frames
        img_t = preprocess_frame(obs_t, size=self.img_size)
        img_next = preprocess_frame(obs_next, size=self.img_size)

        # Encode action
        act_vec = encode_action(action)

        # Convert to torch tensors
        img_t_t = torch.from_numpy(img_t)           # (3, H, W)
        img_next_t = torch.from_numpy(img_next)     # (3, H, W)
        act_vec_t = torch.from_numpy(act_vec)       # (5,)

        return img_t_t, act_vec_t, img_next_t
