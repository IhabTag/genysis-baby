import os
import sys
import glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


def main():
    dataset_root = os.path.join(ROOT, "datasets", "experience")
    pattern = os.path.join(dataset_root, "ep_*", "step_*.npz")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No npz files found under {dataset_root}")
        return

    path = files[0]
    print(f"Inspecting: {path}")
    data = np.load(path, allow_pickle=True)

    obs_t = data["obs_t"]
    obs_next = data["obs_next"]
    action = data["action"]

    print(f"obs_t shape: {obs_t.shape}, dtype: {obs_t.dtype}")
    print(f"obs_next shape: {obs_next.shape}, dtype: {obs_next.dtype}")
    print(f"action type: {type(action)}, value: {action}")


if __name__ == "__main__":
    main()
