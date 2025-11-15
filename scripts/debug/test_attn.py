import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from env.obs import ScreenCapturer
from models.utils.attention import SaliencyAttention


def main():
    capturer = ScreenCapturer(width=640, height=480)
    frame = capturer.capture()
    print(f"Captured frame: {frame.shape}, dtype={frame.dtype}")

    att_module = SaliencyAttention()
    att_map, regions = att_module.compute(frame, prev_frame=None)

    print(f"Attention map shape: {att_map.shape}, min={att_map.min():.4f}, max={att_map.max():.4f}")
    print("Top regions:")
    for r in regions:
        print(f"  {r}")


if __name__ == "__main__":
    main()
