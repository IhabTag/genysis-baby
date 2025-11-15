import os
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from env.obs import ScreenCapturer
from text.patch_ocr import PatchOCR


def main():
    capturer = ScreenCapturer(width=1024, height=768)
    frame = capturer.capture()
    print(f"Captured frame: {frame.shape}, dtype={frame.dtype}")

    ocr_engine = PatchOCR(
        lang="eng",
        patch_size=128,
        max_patches=5,
    )

    result = ocr_engine.compute(frame)

    att_map = result["att_map"]
    regions = result["regions"]
    ocr_results = result["results"]

    print("\n=== OCR proto-reading results ===")
    if not ocr_results:
        print("No salient text regions detected.")
    else:
        for i, r in enumerate(ocr_results):
            y1, y2, x1, x2 = r["bbox"]
            text = r["text"]
            score = r["score"]
            center = r["center"]
            print(f"[Patch {i}] score={score:.3f}, bbox={r['bbox']}, center={center}")
            print(f"  text: {repr(text)}")

    # Visualization
    try:
        import cv2
    except ImportError:
        cv2 = None

    if cv2 is not None:
        overlay = frame.copy()
        for r in ocr_results:
            y1, y2, x1, x2 = r["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.title("Frame")
        plt.imshow(frame)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Attention map")
        plt.imshow(att_map, cmap="viridis")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Frame + OCR regions")
        plt.imshow(overlay)
        plt.axis("off")

        plt.tight_layout()
        plt.show()
    else:
        print("\nOpenCV not installed; skipping visualization.")


if __name__ == "__main__":
    main()
