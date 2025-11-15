import torch
from typing import Optional


class EpisodicBuffer:
    """
    Minimal episodic memory storing projection vectors p_t.

    Supports:
        - append(p_t)
        - get_memory_tensor() -> (N, proj_dim)
        - bounded capacity with FIFO removal
    """

    def __init__(self, max_size: int = 1000, device: str = "cpu"):
        self.max_size = max_size
        self.device = device
        self.memory = []  # list of numpy or torch vectors

    def add(self, p_vec):
        """
        p_vec: torch.Tensor or numpy array of shape (proj_dim,)
        """
        if isinstance(p_vec, torch.Tensor):
            p = p_vec.detach().cpu().numpy()
        else:
            p = p_vec

        self.memory.append(p)
        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def get_memory_tensor(self) -> Optional[torch.Tensor]:
        """
        Returns: torch.Tensor (N, proj_dim) on CPU.
        """
        if len(self.memory) == 0:
            return None
        arr = torch.tensor(self.memory, dtype=torch.float32)
        return arr.to(self.device)
