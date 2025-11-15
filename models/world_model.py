import torch
import torch.nn as nn

from models.utils.encoder_blocks import ConvEncoder, ConvDecoder
from models.utils.dynamics import LatentDynamics
from models.utils.projection_head import ProjectionHead


class WorldModel(nn.Module):
    """
    World model mapping:
      - image_t + action_t → predicted image_{t+1} and latent z_{t+1}

    Components:
      - ConvEncoder: image -> latent z
      - LatentDynamics: (z_t, a_t) -> z_{t+1}
      - ConvDecoder: z_{t+1} -> image_{t+1}
      - ProjectionHead: z -> p (normalized projection for contrastive loss)
    """

    def __init__(
        self,
        action_dim: int = 5,
        img_size: int = 128,
        latent_dim: int = 256,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim

        self.encoder = ConvEncoder(img_size=img_size, latent_dim=latent_dim)
        self.dynamics = LatentDynamics(latent_dim=latent_dim, action_dim=action_dim)
        self.decoder = ConvDecoder(img_size=img_size, latent_dim=latent_dim)
        self.proj_head = ProjectionHead(latent_dim=latent_dim, proj_dim=proj_dim)

    # --------------------------------------------------------
    # Encoding / projection
    # --------------------------------------------------------
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B, 3, H, W), float in [0,1]
        returns: z: (B, latent_dim)
        """
        return self.encoder(img)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim)
        returns: p: (B, proj_dim), L2-normalized
        """
        return self.proj_head(z)

    # --------------------------------------------------------
    # Dynamics
    # --------------------------------------------------------
    def predict_latent(self, z_t: torch.Tensor, action_vec: torch.Tensor) -> torch.Tensor:
        """
        z_t: (B, latent_dim)
        action_vec: (B, action_dim)
        returns: z_next_pred: (B, latent_dim)
        """
        return self.dynamics(z_t, action_vec)

    # --------------------------------------------------------
    # Full forward pass
    # --------------------------------------------------------
    def forward(self, img_t: torch.Tensor, action_vec: torch.Tensor):
        """
        img_t: (B, 3, H, W)
        action_vec: (B, action_dim)

        Returns:
          pred_img_next: (B, 3, H, W) in [0,1]
          z_next_pred:   (B, latent_dim)
        """
        z_t = self.encode(img_t)
        z_next_pred = self.predict_latent(z_t, action_vec)
        pred_img_next = self.decoder(z_next_pred)
        return pred_img_next, z_next_pred
