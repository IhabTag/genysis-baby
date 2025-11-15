import torch
import torch.nn as nn


class LatentDynamics(nn.Module):
    """
    Latent forward dynamics model:

      (z_t, a_t) -> z_{t+1}

    where:
      - z_t: (B, latent_dim)
      - a_t: (B, action_dim)
    """

    def __init__(self, latent_dim: int = 256, action_dim: int = 5, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, action_vec: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_t, action_vec], dim=-1)
        z_next = self.net(x)
        return z_next
