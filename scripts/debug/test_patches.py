import os
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from env.obs import ScreenCapturer
from models.world_model import WorldModel
from models.utils.patch_embeddings import PatchAttentionEmbedder


def main():
    capturer = ScreenCapturer(width=1024, height=768)
    frame = capturer.capture()
    print(f"Captured frame: {frame.shape}, dtype={frame.dtype}")

    # Load or init world model (no need for trained weights for shape test)
    model = WorldModel(action_dim=5, img_size=128)
    embedder = PatchAttentionEmbedder(
        world_model=model,
        device="cpu",
        img_size=128,
        patch_size=128,
        max_patches=5,
    )

    result = embedder.compute(frame)

    att_map = result["att_map"]
    regions = result["regions"]
    patches = result["patches"]
    embeddings = result["embeddings"]

    print(f"Attention map shape: {att_map.shape}")
    print(f"Number of regions: {len(regions)}")
    print(f"Number of patches: {len(patches)}")
    print(f"Embeddings shape: {embeddings.shape}")

    # Quick visualization: frame + att_map + first few patches
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.title("Frame")
    plt.imshow(frame)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Attention map")
    plt.imshow(att_map, cmap="viridis")
    plt.axis("off")

    # Overlay regions on frame
    overlay = frame.copy()
    for r in regions:
        y1, y2, x1, x2 = r["bbox"]
        cv2 = __import__("cv2")
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    plt.subplot(2, 3, 3)
    plt.title("Frame + regions")
    plt.imshow(overlay)
    plt.axis("off")

    # Show up to 3 patches
    for i in range(min(3, len(patches))):
        plt.subplot(2, 3, 4 + i)
        plt.title(f"Patch {i}")
        plt.imshow(patches[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
