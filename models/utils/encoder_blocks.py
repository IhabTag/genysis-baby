import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  ENCODER
# ============================================================
class Encoder(nn.Module):
    """
    Simple CNN encoder for world model.
    Input: (B,3,H,W)
    Output: (B, latent_dim)
    """

    def __init__(self, img_size=128, latent_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 64x64
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(inplace=True),
        )

        self.latent_dim = latent_dim
        self.fc = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        h = self.net(x)  # (B,256,8,8)
        h = h.reshape(h.size(0), -1)  # FIX: reshape instead of view
        z = self.fc(h)               # (B,latent_dim)
        return z


# Alias for backward compatibility with world_model.py
class ConvEncoder(Encoder):
    pass


# ============================================================
#  DECODER
# ============================================================
class Decoder(nn.Module):
    """
    Mirrors the encoder architecture:
    Input: (B, latent_dim)
    Output: (B,3,H,W) in [0,1]
    """

    def __init__(self, img_size=128, latent_dim=256):
        super().__init__()

        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 32x32
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 64x64
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # 128x128
            nn.Sigmoid(),  # Output in [0,1]
        )

    def forward(self, z):
        h = self.fc(z)                        # (B, 256*8*8)
        h = h.reshape(-1, 256, 8, 8)          # (B,256,8,8)
        img = self.net(h)                     # (B,3,128,128)
        return img


# Alias for compatibility
class ConvDecoder(Decoder):
    pass
