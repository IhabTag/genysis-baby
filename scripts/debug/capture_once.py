import os
import sys
import cv2

# ---------------------------------------------------------
# Add project root to sys.path
# ---------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from env.computer_env import ComputerEnv


def main():
    env = ComputerEnv()
    obs = env._observe()              # fresh reading
    frame = obs["image"]              # RGB frame

    print("Captured:", frame.shape, frame.dtype)

    out_path = "capture.png"
    cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
