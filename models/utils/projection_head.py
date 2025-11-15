import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    Simple 2-layer MLP that maps latent z -> projection p.
    Often used in contrastive learning (SimCLR-style).
    """

    def __init__(self, latent_dim: int = 256, proj_dim: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        p = self.net(z)
        # L2 normalization along feature dim
        p = F.normalize(p, dim=-1)
        return p
