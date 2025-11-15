import os
import numpy as np
import torch
from typing import Optional


class EpisodicBuffer:
    """
    Optimized episodic memory storing projection vectors (p_t).
    Uses a fixed-size circular NumPy array for fast append & retrieval.

    Now with save/load support for persistent lifelong memory,
    and optional sampling to keep per-step cost small.
    """

    def __init__(
        self,
        max_size: int = 2000,
        device: str = "cpu",
        proj_dim: int = 64,
    ):
        self.max_size = max_size
        self.device = device
        self.proj_dim = proj_dim

        # Preallocate memory: (max_size, proj_dim)
        self.buffer = np.zeros((max_size, proj_dim), dtype=np.float32)

        self.ptr = 0          # write pointer
        self.size = 0         # number of items currently stored

    # --------------------------------------------------------------
    def add(self, vec: np.ndarray):
        """
        vec: (proj_dim,) numpy float32
        """
        assert vec.shape[-1] == self.proj_dim, (
            f"Expected projection dim {self.proj_dim}, got {vec.shape[-1]}"
        )

        self.buffer[self.ptr] = vec
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # --------------------------------------------------------------
    def get_memory_tensor(self, sample_size: Optional[int] = None):
        """
        Returns Tensor (N, proj_dim) on the correct device.

        If sample_size is provided and size > sample_size,
        returns a random subset of that size to keep computation cheap.
        """
        if self.size == 0:
            return None

        if sample_size is not None and self.size > sample_size:
            idx = np.random.choice(self.size, size=sample_size, replace=False)
            mem_np = self.buffer[idx]
        else:
            mem_np = self.buffer[:self.size]  # view, no copy

        mem_t = torch.from_numpy(mem_np).to(self.device)
        return mem_t

    # --------------------------------------------------------------
    # Persistence
    # --------------------------------------------------------------
    def save(self, path: str):
        """
        Save episodic buffer to disk (npz).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            buffer=self.buffer,
            ptr=self.ptr,
            size=self.size,
            proj_dim=self.proj_dim,
            max_size=self.max_size,
        )

    def load(self, path: str):
        """
        Load episodic buffer from disk if file exists.
        """
        if not os.path.exists(path):
            return

        data = np.load(path)
        buf = data["buffer"]
        ptr = int(data["ptr"])
        size = int(data["size"])
        proj_dim = int(data.get("proj_dim", buf.shape[1]))
        max_size = int(data.get("max_size", buf.shape[0]))

        # Adjust to current config if necessary
        if proj_dim != self.proj_dim:
            print(
                f"[EpisodicBuffer] Warning: proj_dim mismatch "
                f"(file={proj_dim}, current={self.proj_dim}). Skipping load."
            )
            return

        n = min(size, self.max_size, buf.shape[0])

        self.buffer[:n] = buf[:n]
        self.size = n
        self.ptr = ptr % self.max_size
