import numpy as np
import torch


class EpisodicBuffer:
    """
    Optimized episodic memory storing projection vectors (p_t).
    Uses a fixed-size circular NumPy array for fast append & retrieval.

    No slow Python list → tensor conversions.
    """

    def __init__(self, max_size: int = 2000, device: str = "cpu", proj_dim: int = 64):
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
    def get_memory_tensor(self):
        """
        Returns Tensor (size, proj_dim) on the correct device.
        """
        if self.size == 0:
            return None

        mem_np = self.buffer[:self.size]  # view, no copy
        mem_t = torch.from_numpy(mem_np).to(self.device)
        return mem_t
