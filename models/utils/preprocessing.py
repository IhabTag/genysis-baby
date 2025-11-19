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
    if frame is None:
        return np.zeros((3, size, size), dtype=np.float32)
        
    assert frame.ndim == 3 and frame.shape[2] == 3, "Expected RGB frame HxWx3"

    # Resize
    img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0  # [0,1]
    img = np.transpose(img, (2, 0, 1))    # HWC -> CHW
    return img


def preprocess_frame_diff(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """
    Compute normalized pixel difference between two RGB frames.

    Returns:
        float in [0, 1], higher = bigger change.
    """
    if frame_a is None or frame_b is None:
        return 1.0

    if frame_a.shape != frame_b.shape:
        return 1.0

    # Convert to grayscale for difference metric
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_RGB2GRAY)

    diff = cv2.absdiff(gray_a, gray_b)
    diff_norm = np.mean(diff) / 255.0  # normalize

    return float(diff_norm)


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
    Encode an action dict into a fixed-size vector of length 7.

    Vector components:
    0: MOVE_MOUSE flag (1.0 if move, 0.0 otherwise)
    1: x coordinate (normalized 0-1)
    2: y coordinate (normalized 0-1)
    3: LEFT_CLICK flag (1.0 if click, 0.0 otherwise)
    4: RIGHT_CLICK flag (1.0 if click, 0.0 otherwise)
    5: SCROLL amount (normalized -1 to 1)
    6: TYPE_TEXT flag (1.0 if type, 0.0 otherwise)

    This sparse encoding helps the world model distinguish distinct action types.
    """
    vec = np.zeros(7, dtype=np.float32)
    
    a_type = action.get("type", "NOOP")
    
    if a_type == "MOVE_MOUSE":
        vec[0] = 1.0
        x = float(action.get("x", 0.0))
        y = float(action.get("y", 0.0))
        vec[1] = np.clip(x / max(screen_width, 1), 0.0, 1.0)
        vec[2] = np.clip(y / max(screen_height, 1), 0.0, 1.0)
        
    elif a_type == "LEFT_CLICK":
        vec[3] = 1.0
        
    elif a_type == "RIGHT_CLICK":
        vec[4] = 1.0
        
    elif a_type == "SCROLL":
        amount = float(action.get("amount", 0.0))
        # Normalize scroll: assuming typical range is -100 to 100 per step, but can be larger
        # We clip to [-1, 1] to keep inputs stable
        vec[5] = np.clip(amount / 100.0, -1.0, 1.0)
        
    elif a_type == "TYPE_TEXT":
        vec[6] = 1.0
        # Could encode text length here if needed, but binary flag is a good start
        
    return vec


# ------------------------------------------------------------
# Experience dataset (for training)
# ------------------------------------------------------------
class ExperienceDataset(Dataset):
    """
    Loads experience transitions saved as compressed npz files.
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
        act_vec_t = torch.from_numpy(act_vec)       # (7,)

        return img_t_t, act_vec_t, img_next_t
