import numpy as np
from typing import Dict, List
from text.patch_ocr import PatchOCR


class ScreenInterpreter:
    """
    Turns OCR patches + embeddings into
    structured screen elements.
    """

    def __init__(self, patch_size=128, max_patches=5):
        self.ocr_engine = PatchOCR(
            lang="eng",
            patch_size=patch_size,
            max_patches=max_patches,
        )

    def interpret(self, frame: np.ndarray) -> List[Dict]:
        """
        Returns a list of objects:
           {text, bbox, center, score, role_guess}
        """
        out = self.ocr_engine.compute(frame)
        results = []

        for r in out["results"]:
            text = r["text"].strip()

            # crude heuristic role guess
            role = self._infer_role(text)

            results.append(
                {
                    "text": text,
                    "bbox": r["bbox"],
                    "center": r["center"],
                    "score": r["score"],
                    "role": role,
                }
            )

        return results

    def _infer_role(self, text: str) -> str:
        """
        Heuristics to classify GUI text:
        """
        t = text.lower()

        if not t:
            return "unknown"

        if "search" in t:
            return "search_box"

        if any(btn in t for btn in ["ok", "cancel", "close", "apply"]):
            return "button"

        if len(t) < 3:
            return "short_label"

        return "label"
