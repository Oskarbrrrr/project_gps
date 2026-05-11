import os

import matplotlib.pyplot as plt
import numpy as np

from src.dataset import MultimodalDataset


def _ensure_logs_dir():
    os.makedirs("logs", exist_ok=True)


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
    _ensure_logs_dir()
    out_path = f"logs/lidar_bev_vis_{scenario_name}_{mode}_sample_{sample_idx}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {out_path}")


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

    frame_names = ["BEV", "Virtual Point Generation", "BEV+Generation"]
    frame_tensors = [debug["base"], debug["virtual"], debug["combined"]]

    fig, axes = plt.subplots(3, 1, figsize=(5, 12))
    fig.suptitle(
        f"{scenario_name} {mode} sample={sample_idx} frame={frame_idx + 1} target={int(target)}",
        fontsize=16,
    )

    for row_idx, (title, tensor) in enumerate(zip(frame_names, frame_tensors)):
        ax = axes[row_idx]
        image = tensor[frame_idx]
        ax.imshow(image, cmap="viridis", origin="lower")
        ax.set_title(f"{title}\nnonzero={int(np.count_nonzero(image))}")
        ax.axis("off")

    plt.tight_layout()
    _ensure_logs_dir()
    out_path = (
        f"logs/lidar_paper_style_{scenario_name}_{mode}_sample_{sample_idx}_frame_{frame_idx + 1}.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {out_path}")


def visualize_orientation_variants(
    scenario_name="scenario34",
    sample_idx=50,
    mode="train",
    frame_idx=0,
):
    print(f"Rendering orientation variants for {scenario_name} / {mode} / sample {sample_idx} ...")
    variants = [
        ("original", dict(lidar_swap_xy=False, lidar_flip_x=False, lidar_flip_y=False)),
        ("swap_xy", dict(lidar_swap_xy=True, lidar_flip_x=False, lidar_flip_y=False)),
        ("swap+flip_x", dict(lidar_swap_xy=True, lidar_flip_x=True, lidar_flip_y=False)),
        ("swap+flip_y", dict(lidar_swap_xy=True, lidar_flip_x=False, lidar_flip_y=True)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax, (title, kwargs) in zip(axes, variants):
        dataset = MultimodalDataset(mode=mode, scenario_name=scenario_name, **kwargs)
        debug = dataset.get_lidar_debug_sequence(sample_idx)
        image = debug["base"][frame_idx]
        ax.imshow(image, cmap="viridis", origin="lower")
        ax.set_title(f"{title}\nnonzero={int(np.count_nonzero(image))}")
        ax.axis("off")

    fig.suptitle(
        f"Orientation variants: {scenario_name} {mode} sample={sample_idx} frame={frame_idx + 1}",
        fontsize=16,
    )
    plt.tight_layout()
    _ensure_logs_dir()
    out_path = (
        f"logs/lidar_orientation_variants_{scenario_name}_{mode}_sample_{sample_idx}_frame_{frame_idx + 1}.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    visualize_bev_sequence("scenario34", sample_idx=50, mode="train")
    visualize_bev_sequence("scenario34", sample_idx=100, mode="train")
    visualize_paper_style_triptych("scenario34", sample_idx=50, mode="train", frame_idx=1)
    visualize_paper_style_triptych("scenario34", sample_idx=100, mode="train", frame_idx=1)
    visualize_orientation_variants("scenario34", sample_idx=50, mode="train", frame_idx=0)
    visualize_orientation_variants("scenario34", sample_idx=100, mode="train", frame_idx=0)
