import torch
import torch.nn as nn

from models.utils.encoder_blocks import ConvEncoder, ConvDecoder


class WorldModel(nn.Module):
    """
    Unified world model:
      - encoder → latent z
      - projection head → p
      - dynamics f(z, a) → z_next
      - decoder → predicted frame
    """

    def __init__(self, img_size=128, latent_dim=256, proj_dim=64, action_dim=6):
        super().__init__()

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Encoder + decoder
        self.encoder = ConvEncoder(img_size=img_size, latent_dim=latent_dim)
        self.decoder = ConvDecoder(img_size=img_size, latent_dim=latent_dim)

        # Projection head (contrastive)
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, proj_dim),
        )

        # Dynamics model f(z,a)
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    # ---------------------------------------------------------
    #  LOADING (handles old keys)
    # ---------------------------------------------------------
    def load_state_dict(self, state_dict, strict=False):
        new_sd = {}
        for k, v in state_dict.items():
            nk = k
            if k.startswith("encoder.conv"):
                nk = k.replace("encoder.conv", "encoder.net")
            if k.startswith("decoder.deconv"):
                nk = k.replace("decoder.deconv", "decoder.net")
            new_sd[nk] = v
        super().load_state_dict(new_sd, strict=strict)

    # ---------------------------------------------------------
    #  FORWARD FUNCTIONS (NO @torch.no_grad HERE)
    # ---------------------------------------------------------
    def encode(self, img_t):
        return self.encoder(img_t)

    def project(self, z):
        return self.projector(z)

    def predict_latent(self, z_t, action_vec):
        inp = torch.cat([z_t, action_vec], dim=1)
        return self.dynamics(inp)

    def forward(self, img_t, action_vec):
        """
        Return:
            predicted_frame, z_t
        """
        z_t = self.encode(img_t)
        z_next = self.predict_latent(z_t, action_vec)
        img_next = self.decoder(z_next)
        return img_next, z_t
