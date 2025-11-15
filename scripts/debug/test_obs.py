import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from env.obs import ScreenCapturer


def main():
    capturer = ScreenCapturer(width=1024, height=768)
    frame = capturer.capture()
    print(f"Captured frame shape: {frame.shape}, dtype: {frame.dtype}")


if __name__ == "__main__":
    main()
