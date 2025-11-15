import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.world_model import WorldModel
from models.utils.preprocessing import ExperienceDataset
from models.utils.attention import SaliencyAttention


# ------------------------------------------------------------
# Base world model loss (recon + latent)
# ------------------------------------------------------------
class WorldModelLoss(nn.Module):
    def __init__(self, lambda_latent: float = 0.1):
        super().__init__()
        self.lambda_latent = lambda_latent

    def forward(self, pred_img, target_img, z_next_pred, z_next_true):
        """
        pred_img, target_img: (B,3,H,W) in [0,1]
        z_next_pred, z_next_true: (B,latent_dim)
        """
        loss_recon = F.mse_loss(pred_img, target_img)
        loss_latent = F.mse_loss(z_next_pred, z_next_true)
        total = loss_recon + self.lambda_latent * loss_latent
        return total, loss_recon, loss_latent


# ------------------------------------------------------------
# Contrastive loss (InfoNCE-like)
# ------------------------------------------------------------
class TemporalContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, p_t, p_next):
        """
        p_t, p_next: (B, D) L2-normalized projections
        """
        B, D = p_t.shape
        logits = torch.matmul(p_t, p_next.T) / self.temperature
        labels = torch.arange(B, device=p_t.device)

        loss_i2j = F.cross_entropy(logits, labels)
        loss_j2i = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_i2j + loss_j2i)


# ------------------------------------------------------------
# Combined loss with attention weighting
# ------------------------------------------------------------
class ContrastiveWorldModelLoss(nn.Module):
    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_latent: float = 0.1,
        lambda_contrastive: float = 0.5,
        lambda_attn: float = 0.3,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.base_loss = WorldModelLoss(lambda_latent=lambda_latent)
        self.contrastive = TemporalContrastiveLoss(temperature=temperature)
        self.lambda_recon = lambda_recon
        self.lambda_latent = lambda_latent
        self.lambda_contrastive = lambda_contrastive
        self.lambda_attn = lambda_attn

    def forward(
        self,
        pred_img,
        target_img,
        z_next_pred,
        z_next_true,
        p_t,
        p_next,
        att_map=None,
    ):
        # Base recon + latent
        base_total, loss_recon, loss_latent = self.base_loss(
            pred_img, target_img, z_next_pred, z_next_true
        )

        loss_contrast = self.contrastive(p_t, p_next)

        loss_att = 0.0
        if att_map is not None:
            if att_map.dim() == 3:
                att_map = att_map.unsqueeze(1)  # (B,1,H,W)
            per_pixel = (pred_img - target_img) ** 2  # (B,3,H,W)
            att_w = att_map.expand_as(per_pixel)
            loss_att = (per_pixel * att_w).mean()

        total = (
            self.lambda_recon * loss_recon
            + self.lambda_latent * loss_latent
            + self.lambda_contrastive * loss_contrast
            + self.lambda_attn * loss_att
        )

        return total, loss_recon, loss_latent, loss_contrast, loss_att


# ------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------
IMG_SIZE = 128
ACTION_DIM = 5
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 10
DEVICE = "cpu"


def train():
    dataset_root = os.path.join(ROOT, "datasets", "experience")
    print(f"\n=== Loading Dataset from {dataset_root} ===")
    ds = ExperienceDataset(dataset_root, img_size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print(f"Dataset size: {len(ds)} samples")
    print(f"Batches per epoch: {len(loader)}")

    model = WorldModel(action_dim=ACTION_DIM, img_size=IMG_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = ContrastiveWorldModelLoss(
        lambda_recon=1.0,
        lambda_latent=0.1,
        lambda_contrastive=0.5,
        lambda_attn=0.3,
        temperature=0.1,
    )

    att_module = SaliencyAttention()

    ckpt_path = os.path.join(ROOT, "checkpoints", "world_model_contrastive.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    print("\n=== Training (Recon + Latent + Contrastive + Attention) ===")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for img_t, act_vec, img_next in loader:
            img_t = img_t.to(DEVICE)
            act_vec = act_vec.to(DEVICE)
            img_next = img_next.to(DEVICE)

            # Encode
            z_t = model.encode(img_t)
            z_next_true = model.encode(img_next)

            p_t = model.project(z_t)
            p_next = model.project(z_next_true)

            # Predict
            pred_img_next, z_next_pred = model(img_t, act_vec)

            # Build attention maps on target images
            B = img_next.size(0)
            att_maps = []
            for i in range(B):
                frame = img_next[i].permute(1, 2, 0).detach().cpu().numpy()
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                att_map, _ = att_module.compute(frame, prev_frame=None)
                att_maps.append(att_map)

            att_maps = np.stack(att_maps, axis=0)  # (B,H,W)
            att_maps_t = torch.tensor(att_maps, dtype=torch.float32, device=DEVICE).unsqueeze(1)

            # Loss
            loss, loss_recon, loss_latent, loss_contrast, loss_att = criterion(
                pred_img_next,
                img_next,
                z_next_pred,
                z_next_true,
                p_t,
                p_next,
                att_map=att_maps_t,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch}/{EPOCHS}] total={avg_loss:.6f}")

        torch.save(model.state_dict(), ckpt_path)
        print(f"  checkpoint saved → {ckpt_path}")

    print("\n[train_world_model_contrastive] Training complete.")


if __name__ == "__main__":
    train()
