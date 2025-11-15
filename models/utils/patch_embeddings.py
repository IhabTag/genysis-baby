import numpy as np
import torch
from typing import Dict, Any, List

from models.utils.attention import SaliencyAttention
from models.utils.preprocessing import preprocess_frame
from models.world_model import WorldModel


class PatchAttentionEmbedder:
    """
    High-level helper that:
      1. Runs saliency-based attention on a full frame
      2. Extracts top-K salient patches
      3. Encodes patches with the world model encoder
      4. Projects them into contrastive embedding space

    This is the first step toward:
      - text-region detection
      - object-centric world models
      - proto-reading of UI elements
    """

    def __init__(
        self,
        world_model: WorldModel,
        device: str = "cpu",
        img_size: int = 128,
        patch_size: int = 128,
        max_patches: int = 5,
    ):
        self.world_model = world_model.to(device)
        self.world_model.eval()

        self.device = device
        self.img_size = img_size
        self.patch_size = patch_size
        self.max_patches = max_patches

        self.att_module = SaliencyAttention()

    @torch.no_grad()
    def compute(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        frame: RGB uint8 (H,W,3)

        Returns dict:
          {
            "att_map": (H,W) float32 in [0,1],
            "regions": [ {bbox, score, center}, ... ],
            "patches": list of RGB uint8 (patch_size, patch_size, 3),
            "embeddings": np.ndarray (N, D)  # patch-level proj vectors
          }
        """
        assert frame.ndim == 3 and frame.shape[2] == 3, "Expected RGB frame HxWx3"

        # 1. Compute attention map + regions
        att_map, regions = self.att_module.compute(frame, prev_frame=None)

        # 2. Extract patches
        patches, infos = self.att_module.extract_patches(
            frame,
            att_map,
            regions,
            patch_size=self.patch_size,
            max_patches=self.max_patches,
        )

        if len(patches) == 0:
            return {
                "att_map": att_map,
                "regions": [],
                "patches": [],
                "embeddings": np.zeros((0, self.world_model.proj_head.net[-1].out_features), dtype=np.float32),
            }

        # 3. Preprocess patches to (B,3,H,W) tensors
        patch_tensors = []
        for p in patches:
            p_chw = preprocess_frame(p, size=self.img_size)  # (3,H,W)
            patch_tensors.append(p_chw)

        patch_batch = np.stack(patch_tensors, axis=0)  # (N,3,H,W)
        patch_batch_t = torch.tensor(
            patch_batch, dtype=torch.float32, device=self.device
        )

        # 4. Encode + project
        z = self.world_model.encode(patch_batch_t)    # (N,latent_dim)
        p = self.world_model.project(z)               # (N,proj_dim)

        embeddings_np = p.cpu().numpy()

        return {
            "att_map": att_map,
            "regions": infos,
            "patches": patches,
            "embeddings": embeddings_np,
        }
