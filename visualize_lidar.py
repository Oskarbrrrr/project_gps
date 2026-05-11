import os

import matplotlib.pyplot as plt
import numpy as np

from src.dataset import MultimodalDataset


def _ensure_logs_dir():
    os.makedirs("logs", exist_ok=True)


def _save_figure(fig, out_path):
    _ensure_logs_dir()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {out_path}")


def visualize_bev_sequence(scenario_name="scenario32", sample_idx=10, mode="train"):
    print(f"Loading {scenario_name} / {mode} ...")
    dataset = MultimodalDataset(mode=mode, scenario_name=scenario_name)
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
    _save_figure(fig, f"logs/lidar_bev_vis_{scenario_name}_{mode}_sample_{sample_idx}.png")


def visualize_paper_style_triptych(
    scenario_name="scenario34",
    sample_idx=50,
    mode="train",
    frame_idx=1,
):
    print(f"Rendering paper-style view for {scenario_name} / {mode} / sample {sample_idx} ...")
    dataset = MultimodalDataset(mode=mode, scenario_name=scenario_name)
    _, _, _, _, target, _ = dataset[sample_idx]
    debug = dataset.get_lidar_debug_sequence(sample_idx)

    names = ["BEV", "Virtual Point Generation", "BEV+Generation"]
    tensors = [debug["base"], debug["virtual"], debug["combined"]]

    fig, axes = plt.subplots(3, 1, figsize=(5, 12))
    fig.suptitle(
        f"{scenario_name} {mode} sample={sample_idx} frame={frame_idx + 1} target={int(target)}",
        fontsize=16,
    )

    for ax, name, tensor in zip(axes, names, tensors):
        image = tensor[frame_idx]
        ax.imshow(image, cmap="viridis", origin="lower")
        ax.set_title(f"{name}\nnonzero={int(np.count_nonzero(image))}")
        ax.axis("off")

    plt.tight_layout()
    out_path = f"logs/lidar_paper_style_{scenario_name}_{mode}_sample_{sample_idx}_frame_{frame_idx + 1}.png"
    _save_figure(fig, out_path)


def visualize_debug_layers(
    scenario_name="scenario34",
    sample_idx=50,
    mode="train",
    frame_idx=1,
):
    print(f"Rendering debug layers for {scenario_name} / {mode} / sample {sample_idx} ...")
    dataset = MultimodalDataset(mode=mode, scenario_name=scenario_name)
    _, _, _, _, target, _ = dataset[sample_idx]
    debug = dataset.get_lidar_debug_sequence(sample_idx)

    names = [
        ("base", "Base BEV"),
        ("coarse_current", "Coarse Current Count"),
        ("coarse_prev", "Coarse Prev Count"),
        ("coarse_diff", "Coarse Positive Diff"),
        ("raw_motion", "Raw Motion Cue"),
        ("region_mask", "Region Mask"),
        ("virtual", "Virtual Points"),
        ("combined", "Combined BEV"),
    ]

    fig, axes = plt.subplots(len(names), 1, figsize=(5, 28))
    fig.suptitle(
        f"LiDAR debug layers: {scenario_name} {mode} sample={sample_idx} frame={frame_idx + 1} target={int(target)}",
        fontsize=16,
    )

    for ax, (key, title) in zip(axes, names):
        image = debug[key][frame_idx]
        ax.imshow(image, cmap="viridis", origin="lower")
        ax.set_title(f"{title}\nnonzero={int(np.count_nonzero(image))}")
        ax.axis("off")

    plt.tight_layout()
    out_path = f"logs/lidar_debug_layers_{scenario_name}_{mode}_sample_{sample_idx}_frame_{frame_idx + 1}.png"
    _save_figure(fig, out_path)


if __name__ == "__main__":
    visualize_bev_sequence("scenario34", sample_idx=50, mode="train")
    visualize_bev_sequence("scenario34", sample_idx=100, mode="train")
    visualize_paper_style_triptych("scenario34", sample_idx=50, mode="train", frame_idx=1)
    visualize_paper_style_triptych("scenario34", sample_idx=100, mode="train", frame_idx=1)
    visualize_debug_layers("scenario34", sample_idx=50, mode="train", frame_idx=1)
    visualize_debug_layers("scenario34", sample_idx=100, mode="train", frame_idx=1)
