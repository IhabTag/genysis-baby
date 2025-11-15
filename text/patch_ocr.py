import numpy as np
from typing import Dict, Any

from models.utils.attention import SaliencyAttention
from text.ocr import OCRReader


class PatchOCR:
    """
    High-level OCR pipeline:

      1. Compute attention map on full frame
      2. Extract top-K salient patches
      3. Run OCR on each patch
      4. Return:
         - attention map
         - region metadata (bbox, score, center)
         - text per patch

    This is the "proto-reading" module: the baby doesn't understand
    the text semantically yet, but it can *see* and *transcribe* it.
    """

    def __init__(
        self,
        lang: str = "eng",
        patch_size: int = 128,
        max_patches: int = 5,
    ):
        self.att = SaliencyAttention()
        self.ocr = OCRReader(lang=lang)
        self.patch_size = patch_size
        self.max_patches = max_patches

    def compute(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        frame: RGB uint8 (H,W,3)

        Returns a dict:
          {
            "att_map": (H,W) float32,
            "regions": [ {bbox, score, center}, ... ],
            "results": [
              {
                "bbox": (y1,y2,x1,x2),
                "center": (cy,cx),
                "score": float,
                "text": str,
              },
              ...
            ]
          }
        """
        assert frame.ndim == 3 and frame.shape[2] == 3, "Expected RGB frame HxWx3"

        # 1. Attention map + regions
        att_map, regions = self.att.compute(frame, prev_frame=None)

        # 2. Extract patches
        patches, infos = self.att.extract_patches(
            frame,
            att_map,
            regions,
            patch_size=self.patch_size,
            max_patches=self.max_patches,
        )

        results = []
        for patch, info in zip(patches, infos):
            text = self.ocr.read_patch(patch)
            res = {
                "bbox": info["bbox"],
                "center": info["center"],
                "score": info["score"],
                "text": text,
            }
            results.append(res)

        return {
            "att_map": att_map,
            "regions": infos,
            "results": results,
        }
