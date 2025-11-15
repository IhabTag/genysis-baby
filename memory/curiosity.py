import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CuriosityModule(nn.Module):
    """
    Computes intrinsic reward for exploration as:

      curiosity = w_latent * ||p_pred - p_true||^2
                  + w_novelty * novelty_score
                  + w_att * ||A_pred - A_t||^2

    where:
      - p_pred  = projection(model.predict_latent)
      - p_true  = projection(model.encode(obs_next))
      - novelty = distance to nearest memory embedding (projection space)
      - A_pred  = attention map for predicted frame
      - A_t     = attention map for current frame
    """

    def __init__(
        self,
        world_model,
        att_module=None,
        device="cpu",
        w_latent: float = 1.0,
        w_novelty: float = 0.5,
        w_attention: float = 0.3,
    ):
        super().__init__()
        self.world_model = world_model
        self.att_module = att_module
        self.device = device

        self.w_latent = w_latent
        self.w_novelty = w_novelty
        self.w_attention = w_attention

    # ---------------------------------------------------------
    # Low-level curiosity: latent prediction error
    # ---------------------------------------------------------
    @torch.no_grad()
    def latent_curiosity(self, img_t, act_vec, img_next):
        """
        L_latent = || p_pred - p_true ||^2
        """
        img_t = img_t.unsqueeze(0).to(self.device)
        img_next = img_next.unsqueeze(0).to(self.device)
        act_vec = act_vec.unsqueeze(0).to(self.device)

        # True embedding
        z_true = self.world_model.encode(img_next)
        p_true = self.world_model.project(z_true)

        # Predicted embedding
        z_next_pred = self.world_model.predict_latent(
            self.world_model.encode(img_t), act_vec
        )
        p_pred = self.world_model.project(z_next_pred)

        # Curiosity score = MSE
        err = F.mse_loss(p_pred, p_true, reduction="mean").item()
        return err

    # ---------------------------------------------------------
    # Novelty vs memory: min L2 distance to episodic embeddings
    # ---------------------------------------------------------
    @torch.no_grad()
    def novelty_curiosity(self, p_next_pred, memory_tensor: Optional[torch.Tensor]):
        """
        p_next_pred: (1, proj_dim)
        memory_tensor: (N, proj_dim)
        Returns a scalar novelty score.
        """
        if memory_tensor is None or memory_tensor.size(0) == 0:
            return 0.0

        diff = memory_tensor - p_next_pred
        dists = torch.mean(diff ** 2, dim=1)  # (N,)
        novelty = torch.min(dists).item()
        return novelty

    # ---------------------------------------------------------
    # Attention change curiosity: ||A_pred - A_true||^2
    # ---------------------------------------------------------
    @torch.no_grad()
    def attention_curiosity(self, A_pred, A_true):
        """
        A_pred, A_true: (H, W) float32 in [0,1]
        """
        if A_pred is None or A_true is None:
            return 0.0

        if A_pred.shape != A_true.shape:
            # Should never happen after resizing in agent; fallback
            min_h = min(A_pred.shape[0], A_true.shape[0])
            min_w = min(A_pred.shape[1], A_true.shape[1])
            A_pred = A_pred[:min_h, :min_w]
            A_true = A_true[:min_h, :min_w]

        diff = (A_pred - A_true) ** 2
        return float(diff.mean())

    # ---------------------------------------------------------
    # Combined curiosity formula
    # ---------------------------------------------------------
    @torch.no_grad()
    def compute_intrinsic_reward(
        self,
        img_t,
        act_vec,
        img_next,
        p_next_pred,
        A_pred,
        A_true,
        memory_tensor=None,
    ):
        lat_err = self.latent_curiosity(img_t, act_vec, img_next)
        nov = self.novelty_curiosity(p_next_pred, memory_tensor)
        att_err = self.attention_curiosity(A_pred, A_true)

        total = (
            self.w_latent * lat_err
            + self.w_novelty * nov
            + self.w_attention * att_err
        )
        return total
