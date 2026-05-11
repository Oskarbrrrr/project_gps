import os

import matplotlib.pyplot as plt
import numpy as np

from src.dataset import MultimodalDataset


def visualize_bev_sequence(scenario_name="scenario32", sample_idx=10, mode="train"):
    print(f"Loading {scenario_name} / {mode} ...")
    dataset = MultimodalDataset(mode=mode, scenario_name=scenario_name)

    print(f"Reading sample {sample_idx} ...")
    _, _, lidars, _, target, _ = dataset[sample_idx]
    lidar_np = lidars.squeeze(1).numpy()

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(
        f"LiDAR BEV Sequence ({scenario_name}, {mode}, sample={sample_idx}, target={int(target)})",
        fontsize=16,
    )

    for t in range(5):
        ax = axes[t]
        ax.imshow(lidar_np[t], cmap="viridis", origin="lower")
        ax.set_title(f"Frame {t + 1}\nnonzero={int(np.count_nonzero(lidar_np[t]))}")
        ax.axis("off")

    plt.tight_layout()

    os.makedirs("logs", exist_ok=True)
    out_path = f"logs/lidar_bev_vis_{scenario_name}_{mode}_sample_{sample_idx}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    visualize_bev_sequence("scenario34", sample_idx=50, mode="train")
    visualize_bev_sequence("scenario34", sample_idx=100, mode="train")
