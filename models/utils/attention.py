import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional


class SaliencyAttention:
    """
    Simple visual attention module combining:
      - static saliency (edges / gradients)
      - optional motion saliency (frame difference)

    Given:
      - current frame (RGB, HxWx3, uint8)
      - optional previous frame (RGB)
    Returns:
      - attention_map: (H, W), float32 in [0,1]
      - regions: list of top-K salient regions (proto-objects)
    """

    def __init__(
        self,
        blur_ksize: int = 7,
        motion_weight: float = 0.5,
        static_weight: float = 1.0,
        num_regions: int = 5,
        min_region_size: int = 32,
    ):
        self.blur_ksize = blur_ksize
        self.motion_weight = motion_weight
        self.static_weight = static_weight
        self.num_regions = num_regions
        self.min_region_size = min_region_size

    # --------------------------------------------------------
    # Main entry
    # --------------------------------------------------------
    def compute(
        self,
        frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        frame: RGB uint8 (H, W, 3)
        prev_frame: RGB uint8 or None

        Returns:
          attention_map: (H, W), float32 in [0,1]
          regions: list of dicts with keys:
                   { "y1", "y2", "x1", "x2", "score" }
        """
        assert frame.ndim == 3 and frame.shape[2] == 3, "Expected RGB frame HxWx3"
        h, w, _ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # --- Static saliency: gradients / edges ---
        static_map = self._compute_static_saliency(gray)

        # --- Motion saliency (if previous frame given) ---
        if prev_frame is not None and prev_frame.shape == frame.shape:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            motion_map = self._compute_motion_saliency(gray, prev_gray)
        else:
            motion_map = np.zeros_like(static_map, dtype=np.float32)

        # Weighted combination
        att = (
            self.static_weight * static_map
            + self.motion_weight * motion_map
        )

        # Normalize to [0,1]
        att = att - att.min()
        if att.max() > 0:
            att = att / att.max()
        else:
            att = np.zeros_like(att, dtype=np.float32)

        regions = self._extract_regions(att, h, w)
        return att.astype(np.float32), regions

    # --------------------------------------------------------
    # Static saliency: Sobel gradient magnitude
    # --------------------------------------------------------
    def _compute_static_saliency(self, gray: np.ndarray) -> np.ndarray:
        gray_blur = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        gx = cv2.Sobel(gray_blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_blur, cv2.CV_32F, 0, 1, ksize=3)

        mag = cv2.magnitude(gx, gy)

        mag = mag - mag.min()
        if mag.max() > 0:
            mag = mag / mag.max()
        else:
            mag = np.zeros_like(mag, dtype=np.float32)

        return mag.astype(np.float32)

    # --------------------------------------------------------
    # Motion saliency: absolute frame difference
    # --------------------------------------------------------
    def _compute_motion_saliency(self, gray: np.ndarray, prev_gray: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(gray, prev_gray)
        diff = cv2.GaussianBlur(diff, (self.blur_ksize, self.blur_ksize), 0)

        diff = diff.astype(np.float32)
        diff = diff - diff.min()
        if diff.max() > 0:
            diff = diff / diff.max()
        else:
            diff = np.zeros_like(diff, dtype=np.float32)

        return diff

    # --------------------------------------------------------
    # Extract top-K salient regions using threshold + contours
    # --------------------------------------------------------
    def _extract_regions(self, att: np.ndarray, h: int, w: int) -> List[Dict]:
        """
        att: (H, W) in [0,1]
        """
        thresh_val = 0.6
        mask = (att >= thresh_val).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < self.min_region_size and ch < self.min_region_size:
                continue

            region_att = att[y:y + ch, x:x + cw]
            score = float(region_att.max()) if region_att.size > 0 else 0.0

            regions.append({
                "y1": int(y),
                "y2": int(y + ch),
                "x1": int(x),
                "x2": int(x + cw),
                "score": score,
            })

        regions.sort(key=lambda r: r["score"], reverse=True)
        return regions[: self.num_regions]

    # --------------------------------------------------------
    # NEW: Extract patches from top salient regions
    # --------------------------------------------------------
    def extract_patches(
        self,
        frame: np.ndarray,
        att_map: np.ndarray,
        regions: List[Dict],
        patch_size: int = 128,
        max_patches: int = 5,
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Given:
          - frame: RGB uint8 (H,W,3)
          - att_map: (H,W) float32 in [0,1]
          - regions: list of {x1,x2,y1,y2,score}
        Returns:
          - patches: list of RGB uint8 (patch_size, patch_size, 3)
          - infos:   list of dicts:
                     {
                       "bbox": (y1, y2, x1, x2),
                       "score": float,
                       "center": (cy, cx)
                     }
        """
        h, w, _ = frame.shape
        patches: List[np.ndarray] = []
        infos: List[Dict] = []

        for r in regions[:max_patches]:
            y1, y2 = r["y1"], r["y2"]
            x1, x2 = r["x1"], r["x2"]
            score = r.get("score", 0.0)

            # Compute center of the region
            cy = (y1 + y2) // 2
            cx = (x1 + x2) // 2

            # Define a square patch around the center
            half = max((y2 - y1), (x2 - x1)) // 2
            half = max(half, patch_size // 4)  # avoid too tiny patches

            top = max(cy - half, 0)
            bottom = min(cy + half, h)
            left = max(cx - half, 0)
            right = min(cx + half, w)

            region_patch = frame[top:bottom, left:right, :]
            if region_patch.size == 0:
                continue

            # Resize to (patch_size, patch_size)
            patch_resized = cv2.resize(
                region_patch,
                (patch_size, patch_size),
                interpolation=cv2.INTER_AREA,
            )

            patches.append(patch_resized)
            infos.append(
                {
                    "bbox": (int(top), int(bottom), int(left), int(right)),
                    "score": float(score),
                    "center": (int(cy), int(cx)),
                }
            )

        return patches, infos
