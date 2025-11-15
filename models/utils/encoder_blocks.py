import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """
    Simple CNN encoder:
      (B,3,H,W) -> (B,latent_dim)

    For img_size=128, we progressively downsample to 8x8.
    """

    def __init__(self, img_size: int = 128, latent_dim: int = 256):
        super().__init__()

        self.conv = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # For img_size=128, after 4 strided convs: 128 -> 8
        final_spatial = img_size // 16
        conv_out_dim = 256 * final_spatial * final_spatial

        self.fc = nn.Linear(conv_out_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return z


class ConvDecoder(nn.Module):
    """
    CNN decoder:
      (B,latent_dim) -> (B,3,H,W) in [0,1]
    Mirrors ConvEncoder with transposed convs.
    """

    def __init__(self, img_size: int = 128, latent_dim: int = 256):
        super().__init__()

        final_spatial = img_size // 16  # 8 for 128x128
        self.fc = nn.Linear(latent_dim, 256 * final_spatial * final_spatial)

        self.deconv = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # outputs in [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = self.fc(z)
        # reshape to (B, 256, H', W')
        # H' = W' = img_size // 16
        spatial = int((h.size(1) // 256) ** 0.5)
        h = h.view(B, 256, spatial, spatial)
        x_recon = self.deconv(h)
        return x_recon
