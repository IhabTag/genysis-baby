import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.world_model import WorldModel
from models.utils.preprocessing import ExperienceDataset


CKPT_PATH = os.path.join(ROOT, "checkpoints", "world_model_contrastive.pt")


def main():
    device = "cpu"

    print(f"Loading dataset from datasets/experience ...")
    ds = ExperienceDataset(os.path.join(ROOT, "datasets", "experience"), img_size=128)

    if len(ds) == 0:
        print("No data found in datasets/experience. Run collect_random_dataset.py first.")
        return

    img_t, act_vec, img_next = ds[0]

    img_t_b = img_t.unsqueeze(0).to(device)
    act_vec_b = act_vec.unsqueeze(0).to(device)

    print(f"Loading world model from: {CKPT_PATH}")
    model = WorldModel(action_dim=5, img_size=128).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        pred_img_next, _ = model(img_t_b, act_vec_b)

    # Convert to numpy HWC
    img_t_np = np.transpose(img_t.numpy(), (1, 2, 0))
    img_next_np = np.transpose(img_next.numpy(), (1, 2, 0))
    pred_np = np.transpose(pred_img_next.squeeze(0).cpu().numpy(), (1, 2, 0))

    img_t_np = np.clip(img_t_np, 0, 1)
    img_next_np = np.clip(img_next_np, 0, 1)
    pred_np = np.clip(pred_np, 0, 1)

    # Plot
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("img_t")
    plt.imshow(img_t_np)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("img_next (true)")
    plt.imshow(img_next_np)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("pred_next")
    plt.imshow(pred_np)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
