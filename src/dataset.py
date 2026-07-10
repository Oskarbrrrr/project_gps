import os

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultimodalDataset(Dataset):
    def __init__(
        self,
        mode="train",
        data_root="./Data/Multi_Modal",
        split_root="./Data/splits",
        scenario_name="scenario32",
        csv_path=None,
        image_subdir="camera_data",
        gps_stats=None,
        lidar_representation="binary",
        lidar_grid_size=256,
        lidar_x_range=(-30.0, 30.0),
        lidar_y_range=(-30.0, 30.0),
        use_virtual_points=True,
        lidar_swap_xy=False,
        lidar_flip_x=False,
        lidar_flip_y=False,
        lidar_motion_grid_size=64,
        lidar_motion_history_dilation=1,
        lidar_motion_region_expand=2,
        lidar_virtual_seed_expand=1,
        lidar_virtual_expand=1,
        lidar_motion_min_cells=2,
        lidar_motion_max_cells=48,
        lidar_motion_count_threshold=2.0,
        lidar_virtual_jitter_radius_m=0.1,
        missing_enabled=False,
        return_missing_masks=False,
        missing_frame_prob=0.0,
        missing_burst_prob=0.0,
        missing_burst_min=2,
        missing_burst_max=3,
        missing_modality_prob=0.0,
        missing_modality_min=1,
        missing_modality_max=2,
        missing_modalities=None,
        missing_seed=42,
        image_aug=False,
        gps_feature_mode="bemamba",
        gps_future_steps=3,
    ):
        self.data_dir = data_root
        self.mode = mode
        self.image_subdir = image_subdir
        self.image_aug = bool(image_aug)
        self.gps_stats = gps_stats
        self.gps_feature_mode = str(gps_feature_mode)
        if self.gps_feature_mode not in ("bemamba", "physical_kinematic"):
            raise ValueError(f"Unsupported gps_feature_mode: {self.gps_feature_mode}")
        self.gps_future_steps = int(gps_future_steps)
        if self.gps_future_steps < 0:
            raise ValueError("gps_future_steps must be >= 0")
        self.lidar_representation = lidar_representation
        self.lidar_grid_size = int(lidar_grid_size)
        self.lidar_x_range = tuple(lidar_x_range)
        self.lidar_y_range = tuple(lidar_y_range)
        self.use_virtual_points = use_virtual_points
        self.lidar_swap_xy = lidar_swap_xy
        self.lidar_flip_x = lidar_flip_x
        self.lidar_flip_y = lidar_flip_y

        self.lidar_motion_grid_size = int(lidar_motion_grid_size)
        self.lidar_motion_history_dilation = int(lidar_motion_history_dilation)
        self.lidar_motion_region_expand = int(lidar_motion_region_expand)
        self.lidar_virtual_seed_expand = int(lidar_virtual_seed_expand)
        self.lidar_virtual_expand = int(lidar_virtual_expand)
        self.lidar_motion_min_cells = int(lidar_motion_min_cells)
        self.lidar_motion_max_cells = int(lidar_motion_max_cells)
        self.lidar_motion_count_threshold = float(lidar_motion_count_threshold)
        self.lidar_virtual_jitter_radius_m = float(lidar_virtual_jitter_radius_m)

        self.missing_enabled = bool(missing_enabled)
        self.return_missing_masks = bool(return_missing_masks)
        self.missing_frame_prob = self._clip_probability(missing_frame_prob)
        self.missing_burst_prob = self._clip_probability(missing_burst_prob)
        self.missing_burst_min = max(1, int(missing_burst_min))
        self.missing_burst_max = max(self.missing_burst_min, int(missing_burst_max))
        self.missing_modality_prob = self._clip_probability(missing_modality_prob)
        self.missing_modality_min = max(1, int(missing_modality_min))
        self.missing_modality_max = max(self.missing_modality_min, int(missing_modality_max))
        self.missing_seed = int(missing_seed)
        self.missing_epoch = 0
        self.missing_lengths = {
            "img": 5,
            "radar": 5,
            "lidar": 5,
            "gps": 2,
        }
        self.missing_modalities = self._normalize_missing_modalities(missing_modalities)
        self.missing_frame_modalities = tuple(
            m for m in self.missing_modalities if m != "gps"
        )

        if csv_path is None:
            csv_path = os.path.join(split_root, f"{scenario_name}_{mode}.csv")
        self.df = pd.read_csv(csv_path)

        if self.mode == "train" and self.image_aug:
            image_transforms = [
                transforms.Resize((256, 256)),
                transforms.ColorJitter(
                    brightness=0.06,
                    contrast=0.06,
                    saturation=0.05,
                    hue=0.01,
                ),
            ]
        else:
            image_transforms = [transforms.Resize((256, 256))]

        image_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.img_transform = transforms.Compose(image_transforms)

        self._init_gps_normalization()

    def _clip_probability(self, value):
        return float(np.clip(float(value), 0.0, 1.0))

    def set_missing_epoch(self, epoch):
        if self.mode in ("val", "test"):
            return
        self.missing_epoch = int(epoch)

    def _normalize_missing_modalities(self, missing_modalities):
        if missing_modalities is None:
            return tuple(self.missing_lengths.keys())

        alias = {
            "imgs": "img",
            "image": "img",
            "images": "img",
            "camera": "img",
            "cam": "img",
            "rgb": "img",
            "radars": "radar",
            "lidars": "lidar",
        }
        if isinstance(missing_modalities, str):
            if missing_modalities.lower() == "all":
                return tuple(self.missing_lengths.keys())
            items = [
                item.strip()
                for item in missing_modalities.replace(";", ",").split(",")
                if item.strip()
            ]
        else:
            items = list(missing_modalities)

        normalized = []
        for item in items:
            name = alias.get(str(item).lower(), str(item).lower())
            if name not in self.missing_lengths:
                raise ValueError(f"Unsupported missing modality: {item}")
            if name not in normalized:
                normalized.append(name)
        return tuple(normalized)

    def _missing_rng(self, idx):
        epoch_offset = self.missing_epoch * max(1, len(self.df))
        return np.random.default_rng(self.missing_seed + epoch_offset + int(idx))

    def _build_missing_masks(self, idx):
        masks = {
            name: torch.ones(length, dtype=torch.float32)
            for name, length in self.missing_lengths.items()
        }
        if not self.missing_enabled:
            return masks

        rng = self._missing_rng(idx)
        for name in self.missing_frame_modalities:
            length = self.missing_lengths[name]
            if self.missing_frame_prob > 0:
                frame_missing = rng.random(length) < self.missing_frame_prob
                masks[name][torch.from_numpy(frame_missing)] = 0.0

            if self.missing_burst_prob > 0 and rng.random() < self.missing_burst_prob:
                burst_min = min(self.missing_burst_min, length)
                burst_max = min(self.missing_burst_max, length)
                burst_len = int(rng.integers(burst_min, burst_max + 1))
                start = int(rng.integers(0, length - burst_len + 1))
                masks[name][start : start + burst_len] = 0.0

        if self.missing_modality_prob > 0 and rng.random() < self.missing_modality_prob:
            candidates = list(self.missing_modalities)
            if candidates:
                max_count = min(self.missing_modality_max, len(candidates))
                min_count = min(self.missing_modality_min, max_count)
                count = int(rng.integers(min_count, max_count + 1))
                for name in rng.choice(candidates, size=count, replace=False):
                    masks[str(name)].zero_()

        return masks

    def _apply_sequence_mask(self, sequence, mask):
        mask = mask.to(dtype=sequence.dtype, device=sequence.device)
        view_shape = (mask.shape[0],) + (1,) * (sequence.dim() - 1)
        return sequence * mask.view(view_shape)

    def _init_gps_normalization(self):
        if self.gps_stats is not None:
            self.min_dx = float(self.gps_stats["min_dx"])
            self.max_dx = float(self.gps_stats["max_dx"])
            self.min_dy = float(self.gps_stats["min_dy"])
            self.max_dy = float(self.gps_stats["max_dy"])
            if self.gps_feature_mode == "physical_kinematic":
                required = (
                    "mean_dx", "std_dx", "mean_dy", "std_dy", "mean_r", "std_r",
                    "mean_vx", "std_vx", "mean_vy", "std_vy", "mean_speed", "std_speed",
                )
                missing = [key for key in required if key not in self.gps_stats]
                if missing:
                    raise ValueError(
                        "physical_kinematic GPS requires train-derived statistics: "
                        + ", ".join(missing)
                    )
                for key in required:
                    setattr(self, key, float(self.gps_stats[key]))
            return

        all_dx, all_dy, all_r = [], [], []
        all_vx, all_vy, all_speed = [], [], []
        for idx in range(len(self.df)):
            bs_lat, bs_lon = self._read_gps_raw(self.df.iloc[idx]["unit1_loc"])
            ue1_lat, ue1_lon = self._read_gps_raw(self.df.iloc[idx]["unit2_loc_1"])
            ue2_lat, ue2_lon = self._read_gps_raw(self.df.iloc[idx]["unit2_loc_2"])
            if bs_lat == 0.0:
                continue
            dx1, dy1 = self._relative_xy_m(bs_lat, bs_lon, ue1_lat, ue1_lon)
            dx2, dy2 = self._relative_xy_m(bs_lat, bs_lon, ue2_lat, ue2_lon)
            # Keep legacy min/max based on loc_1 for the default BeMamba mode.
            all_dx.append(dx1)
            all_dy.append(dy1)
            if self.gps_feature_mode == "physical_kinematic":
                all_dx.append(dx2)
                all_dy.append(dy2)
                all_r.extend((np.hypot(dx1, dy1), np.hypot(dx2, dy2)))
                vx, vy = dx2 - dx1, dy2 - dy1
                all_vx.append(vx)
                all_vy.append(vy)
                all_speed.append(np.hypot(vx, vy))
        if all_dx:
            self.min_dx, self.max_dx = np.min(all_dx), np.max(all_dx)
            self.min_dy, self.max_dy = np.min(all_dy), np.max(all_dy)
        else:
            self.min_dx, self.max_dx, self.min_dy, self.max_dy = 0, 1, 0, 1
        if self.gps_feature_mode == "physical_kinematic":
            self.mean_dx, self.std_dx = self._mean_std(all_dx)
            self.mean_dy, self.std_dy = self._mean_std(all_dy)
            self.mean_r, self.std_r = self._mean_std(all_r)
            self.mean_vx, self.std_vx = self._mean_std(all_vx)
            self.mean_vy, self.std_vy = self._mean_std(all_vy)
            self.mean_speed, self.std_speed = self._mean_std(all_speed)

    def get_gps_stats(self):
        stats = {
            "min_dx": float(self.min_dx),
            "max_dx": float(self.max_dx),
            "min_dy": float(self.min_dy),
            "max_dy": float(self.max_dy),
        }
        if self.gps_feature_mode == "physical_kinematic":
            for key in (
                "mean_dx", "std_dx", "mean_dy", "std_dy", "mean_r", "std_r",
                "mean_vx", "std_vx", "mean_vy", "std_vy", "mean_speed", "std_speed",
            ):
                stats[key] = float(getattr(self, key))
        return stats

    @staticmethod
    def _mean_std(values):
        if not values:
            return 0.0, 1.0
        mean = float(np.mean(values))
        std = float(np.std(values))
        return mean, max(std, 1e-6)

    @staticmethod
    def _relative_xy_m(bs_lat, bs_lon, ue_lat, ue_lon):
        dx = (ue_lon - bs_lon) * 111320 * np.cos(np.radians(bs_lat))
        dy = (ue_lat - bs_lat) * 111320
        return float(dx), float(dy)

    @staticmethod
    def _zscore(value, mean, std):
        return (float(value) - float(mean)) / max(float(std), 1e-6)

    def _read_gps_raw(self, rel_path):
        try:
            with open(os.path.join(self.data_dir, rel_path), "r") as f:
                lines = f.readlines()
                return float(lines[0].strip()), float(lines[1].strip())
        except Exception:
            return 0.0, 0.0

    def _calc_gps_bemamba_eq1(self, bs_lat, bs_lon, ue_lat, ue_lon):
        dx, dy = self._relative_xy_m(bs_lat, bs_lon, ue_lat, ue_lon)
        dx_norm = (dx - self.min_dx) / (self.max_dx - self.min_dx + 1e-8)
        dy_norm = (dy - self.min_dy) / (self.max_dy - self.min_dy + 1e-8)
        dist = np.sqrt(dx_norm**2 + dy_norm**2)
        angle = np.arctan2(dy_norm, dx_norm)
        return [dist, angle]

    def _calc_gps_physical_kinematic(self, bs_lat, bs_lon, ue1_lat, ue1_lon, ue2_lat, ue2_lon):
        x1, y1 = self._relative_xy_m(bs_lat, bs_lon, ue1_lat, ue1_lon)
        x2, y2 = self._relative_xy_m(bs_lat, bs_lon, ue2_lat, ue2_lon)
        vx, vy = x2 - x1, y2 - y1
        speed = np.hypot(vx, vy)
        heading = np.arctan2(vy, vx) if speed > 1e-8 else 0.0

        future_x = x2 + self.gps_future_steps * vx
        future_y = y2 + self.gps_future_steps * vy
        future_r = np.hypot(future_x, future_y)
        future_theta = np.arctan2(future_y, future_x)
        shared = [
            self._zscore(vx, self.mean_vx, self.std_vx),
            self._zscore(vy, self.mean_vy, self.std_vy),
            self._zscore(speed, self.mean_speed, self.std_speed),
            np.sin(heading),
            np.cos(heading),
            self._zscore(future_x, self.mean_dx, self.std_dx),
            self._zscore(future_y, self.mean_dy, self.std_dy),
            self._zscore(future_r, self.mean_r, self.std_r),
            np.sin(future_theta),
            np.cos(future_theta),
        ]

        features = []
        for x, y in ((x1, y1), (x2, y2)):
            radius = np.hypot(x, y)
            theta = np.arctan2(y, x)
            local = [
                self._zscore(x, self.mean_dx, self.std_dx),
                self._zscore(y, self.mean_dy, self.std_dy),
                self._zscore(radius, self.mean_r, self.std_r),
                np.sin(theta),
                np.cos(theta),
            ]
            features.append(local + shared)
        return features

    def _read_power(self, rel_path):
        try:
            with open(os.path.join(self.data_dir, rel_path), "r") as f:
                return [float(x) for x in f.read().split()]
        except Exception:
            return [0.0] * 64

    def _apply_bev_orientation(self, bev):
        if self.lidar_swap_xy:
            bev = bev.T
        if self.lidar_flip_x:
            bev = np.flip(bev, axis=0)
        if self.lidar_flip_y:
            bev = np.flip(bev, axis=1)
        return np.ascontiguousarray(bev)

    def _binary_dilate(self, mask, radius):
        if radius <= 0:
            return mask.astype(bool)

        mask = mask.astype(bool)
        dilated = np.zeros_like(mask, dtype=bool)
        h, w = mask.shape
        for dx in range(-radius, radius + 1):
            x_src_start = max(0, -dx)
            x_src_end = min(h, h - dx)
            x_dst_start = max(0, dx)
            x_dst_end = min(h, h + dx)
            for dy in range(-radius, radius + 1):
                y_src_start = max(0, -dy)
                y_src_end = min(w, w - dy)
                y_dst_start = max(0, dy)
                y_dst_end = min(w, w + dy)
                dilated[x_dst_start:x_dst_end, y_dst_start:y_dst_end] |= mask[
                    x_src_start:x_src_end, y_src_start:y_src_end
                ]
        return dilated

    def _connected_components(self, mask):
        mask = mask.astype(bool)
        visited = np.zeros_like(mask, dtype=bool)
        h, w = mask.shape
        components = []

        for x in range(h):
            for y in range(w):
                if not mask[x, y] or visited[x, y]:
                    continue

                stack = [(x, y)]
                visited[x, y] = True
                coords = []
                min_x = max_x = x
                min_y = max_y = y

                while stack:
                    cx, cy = stack.pop()
                    coords.append((cx, cy))
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)

                    for nx in range(max(0, cx - 1), min(h, cx + 2)):
                        for ny in range(max(0, cy - 1), min(w, cy + 2)):
                            if mask[nx, ny] and not visited[nx, ny]:
                                visited[nx, ny] = True
                                stack.append((nx, ny))

                components.append(
                    {
                        "coords": coords,
                        "size": len(coords),
                        "bbox": (min_x, max_x, min_y, max_y),
                    }
                )
        return components

    def _filter_motion_coarse(self, coarse_mask):
        filtered = np.zeros_like(coarse_mask, dtype=bool)
        for component in self._connected_components(coarse_mask):
            size = component["size"]
            if size < self.lidar_motion_min_cells:
                continue
            if size > self.lidar_motion_max_cells:
                continue
            for x, y in component["coords"]:
                filtered[x, y] = True
        return filtered

    def _read_lidar_xy(self, rel_path):
        try:
            full_path = os.path.join(self.data_dir, rel_path)
            if not os.path.exists(full_path):
                return np.zeros((0, 2), dtype=np.float32)

            pcd = o3d.io.read_point_cloud(full_path)
            points = np.asarray(pcd.points)
            if len(points) == 0:
                return np.zeros((0, 2), dtype=np.float32)

            x, y = points[:, 0], points[:, 1]
            x_min, x_max = self.lidar_x_range
            y_min, y_max = self.lidar_y_range
            mask = (
                (x >= x_min)
                & (x <= x_max)
                & (y >= y_min)
                & (y <= y_max)
            )
            if not np.any(mask):
                return np.zeros((0, 2), dtype=np.float32)

            return np.stack([x[mask], y[mask]], axis=1).astype(np.float32)
        except Exception:
            return np.zeros((0, 2), dtype=np.float32)

    def _points_to_bev(self, points, grid_size=None):
        if grid_size is None:
            grid_size = self.lidar_grid_size

        bev = np.zeros((grid_size, grid_size), dtype=np.float32)
        if points is None or len(points) == 0:
            return bev

        x, y = points[:, 0], points[:, 1]
        x_min, x_max = self.lidar_x_range
        y_min, y_max = self.lidar_y_range
        x_idx = ((x - x_min) / (x_max - x_min + 1e-6) * (grid_size - 1)).astype(np.int32)
        y_idx = ((y - y_min) / (y_max - y_min + 1e-6) * (grid_size - 1)).astype(np.int32)
        bev[x_idx, y_idx] = 1.0

        if grid_size == self.lidar_grid_size:
            return self._apply_bev_orientation(bev)
        return bev

    def _points_to_count_map(self, points, grid_size):
        count_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        if points is None or len(points) == 0:
            return count_map

        x, y = points[:, 0], points[:, 1]
        x_min, x_max = self.lidar_x_range
        y_min, y_max = self.lidar_y_range
        x_idx = ((x - x_min) / (x_max - x_min + 1e-6) * (grid_size - 1)).astype(np.int32)
        y_idx = ((y - y_min) / (y_max - y_min + 1e-6) * (grid_size - 1)).astype(np.int32)
        np.add.at(count_map, (x_idx, y_idx), 1.0)
        return count_map

    def _normalize_count_map(self, count_map):
        count_map = np.asarray(count_map, dtype=np.float32)
        if count_map.size == 0:
            return count_map
        count_map = np.log1p(np.clip(count_map, a_min=0.0, a_max=None))
        max_value = float(np.max(count_map))
        if max_value <= 1e-8:
            return np.zeros_like(count_map, dtype=np.float32)
        return (count_map / max_value).astype(np.float32)

    def _points_to_indices(self, points, grid_size):
        if points is None or len(points) == 0:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        x, y = points[:, 0], points[:, 1]
        x_min, x_max = self.lidar_x_range
        y_min, y_max = self.lidar_y_range
        x_idx = ((x - x_min) / (x_max - x_min + 1e-6) * (grid_size - 1)).astype(np.int32)
        y_idx = ((y - y_min) / (y_max - y_min + 1e-6) * (grid_size - 1)).astype(np.int32)
        x_idx = np.clip(x_idx, 0, grid_size - 1)
        y_idx = np.clip(y_idx, 0, grid_size - 1)
        return x_idx, y_idx

    def _generate_virtual_points_from_motion(self, current_points, motion_mask):
        if current_points is None or len(current_points) == 0 or (not np.any(motion_mask)):
            return np.zeros((0, 2), dtype=np.float32)

        x_idx, y_idx = self._points_to_indices(current_points, self.lidar_grid_size)
        motion_points = current_points[motion_mask[x_idx, y_idx]]
        if len(motion_points) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        if self.mode == "train" and self.lidar_virtual_jitter_radius_m > 0:
            angles = np.random.uniform(0.0, 2.0 * np.pi, size=len(motion_points))
            radii = np.sqrt(np.random.uniform(0.0, 1.0, size=len(motion_points))) * self.lidar_virtual_jitter_radius_m
            offsets = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
            virtual_points = motion_points + offsets.astype(np.float32)
        else:
            # Keep evaluation deterministic: preserve one-to-one virtual points without random perturbation.
            virtual_points = motion_points.astype(np.float32, copy=True)

        x_min, x_max = self.lidar_x_range
        y_min, y_max = self.lidar_y_range
        valid = (
            (virtual_points[:, 0] >= x_min)
            & (virtual_points[:, 0] <= x_max)
            & (virtual_points[:, 1] >= y_min)
            & (virtual_points[:, 1] <= y_max)
        )
        return virtual_points[valid].astype(np.float32)

    def _upsample_mask(self, coarse_mask, target_size):
        coarse_h, coarse_w = coarse_mask.shape
        scale_h = max(1, target_size // coarse_h)
        scale_w = max(1, target_size // coarse_w)
        upsampled = np.kron(coarse_mask.astype(np.uint8), np.ones((scale_h, scale_w), dtype=np.uint8))
        return upsampled[:target_size, :target_size].astype(bool)

    def _build_lidar_layers(self, current_points, prev_points):
        base_bev = self._points_to_bev(current_points, self.lidar_grid_size)

        if (not self.use_virtual_points) or prev_points is None or len(prev_points) == 0:
            zeros = np.zeros_like(base_bev, dtype=np.float32)
            zeros_coarse = np.zeros((self.lidar_motion_grid_size, self.lidar_motion_grid_size), dtype=np.float32)
            base_count = self._normalize_count_map(self._points_to_count_map(current_points, self.lidar_grid_size))
            return {
                "base": base_bev,
                "base_count": base_count,
                "coarse_current": zeros_coarse,
                "coarse_prev": zeros_coarse,
                "coarse_diff": zeros_coarse,
                "raw_motion": zeros,
                "region_mask": zeros,
                "virtual": zeros,
                "virtual_count": zeros,
                "combined": base_bev.copy(),
                "combined_count": base_count.copy(),
            }

        prev_bev = self._points_to_bev(prev_points, self.lidar_grid_size)
        base_count_raw = self._points_to_count_map(current_points, self.lidar_grid_size)
        current_count = self._points_to_count_map(current_points, self.lidar_motion_grid_size)
        prev_count = self._points_to_count_map(prev_points, self.lidar_motion_grid_size)
        count_diff = current_count - prev_count
        raw_motion = (base_bev > 0.5) & (prev_bev <= 0.5)
        region_mask = self._binary_dilate(raw_motion, self.lidar_motion_region_expand)
        motion_seed = self._binary_dilate(raw_motion, self.lidar_virtual_seed_expand)
        virtual_points = self._generate_virtual_points_from_motion(current_points, motion_seed)
        virtual_mask = self._points_to_bev(virtual_points, self.lidar_grid_size) > 0.5
        virtual_count_raw = self._points_to_count_map(virtual_points, self.lidar_grid_size)
        if self.lidar_virtual_expand > 0:
            virtual_mask = self._binary_dilate(virtual_mask, self.lidar_virtual_expand)

        virtual_only = virtual_mask.astype(bool)
        base_mask = base_bev > 0.5
        combined = np.maximum(base_mask.astype(np.float32), virtual_mask.astype(np.float32))
        combined_count = self._normalize_count_map(base_count_raw + virtual_count_raw)

        return {
            "base": base_bev.astype(np.float32),
            "base_count": self._normalize_count_map(base_count_raw),
            "coarse_current": current_count.astype(np.float32),
            "coarse_prev": prev_count.astype(np.float32),
            "coarse_diff": np.clip(count_diff, a_min=0.0, a_max=None).astype(np.float32),
            "raw_motion": raw_motion.astype(np.float32),
            "region_mask": region_mask.astype(np.float32),
            "virtual": virtual_only.astype(np.float32),
            "virtual_count": self._normalize_count_map(virtual_count_raw),
            "combined": combined.astype(np.float32),
            "combined_count": combined_count.astype(np.float32),
        }

    def _resize_tensor(self, tensor, size=(256, 256)):
        return F.interpolate(
            tensor.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def get_lidar_debug_sequence(self, idx):
        row = self.df.iloc[idx]
        debug = {
            "base": [],
            "base_count": [],
            "coarse_current": [],
            "coarse_prev": [],
            "coarse_diff": [],
            "raw_motion": [],
            "region_mask": [],
            "virtual": [],
            "virtual_count": [],
            "combined": [],
            "combined_count": [],
        }
        prev_points = None

        for t in range(1, 6):
            current_points = self._read_lidar_xy(row[f"unit1_lidar_{t}"])
            layers = self._build_lidar_layers(current_points, prev_points)
            for key in debug:
                debug[key].append(layers[key])
            prev_points = current_points.copy()

        return {key: np.stack(value) for key, value in debug.items()}

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, rel_path):
        normalized = rel_path.replace("\\", "/")
        return normalized.replace("/camera_data/", f"/{self.image_subdir}/")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgs, radars, lidars = [], [], []
        prev_points = None

        for t in range(1, 6):
            img_rel_path = self._resolve_image_path(row[f"unit1_rgb_{t}"])
            img_path = os.path.join(self.data_dir, img_rel_path)
            imgs.append(self.img_transform(Image.open(img_path).convert("RGB")))

            ang_path = os.path.join(
                self.data_dir,
                row[f"unit1_radar_{t}"].replace("/radar_data/", "/radar_data_ang/"),
            )
            vel_path = os.path.join(
                self.data_dir,
                row[f"unit1_radar_{t}"].replace("/radar_data/", "/radar_data_vel/"),
            )
            try:
                ang_arr = np.nan_to_num(np.load(ang_path))
                vel_arr = np.nan_to_num(np.load(vel_path))
                radar_tensor = torch.tensor(
                    np.stack([ang_arr, vel_arr], axis=0),
                    dtype=torch.float32,
                )
            except Exception:
                radar_tensor = torch.zeros((2, 256, 256), dtype=torch.float32)
            radars.append(self._resize_tensor(radar_tensor))

            current_points = self._read_lidar_xy(row[f"unit1_lidar_{t}"])
            layers = self._build_lidar_layers(current_points, prev_points)
            prev_points = current_points.copy()
            lidar_key = "combined_count" if self.lidar_representation == "count" else "combined"
            lidars.append(self._resize_tensor(torch.tensor(layers[lidar_key]).unsqueeze(0)))

        bs_lat, bs_lon = self._read_gps_raw(row["unit1_loc"])
        u1_lat, u1_lon = self._read_gps_raw(row["unit2_loc_1"])
        u2_lat, u2_lon = self._read_gps_raw(row["unit2_loc_2"])
        if self.gps_feature_mode == "physical_kinematic":
            gps_features = self._calc_gps_physical_kinematic(
                bs_lat, bs_lon, u1_lat, u1_lon, u2_lat, u2_lon
            )
        else:
            gps_start = self._calc_gps_bemamba_eq1(bs_lat, bs_lon, u1_lat, u1_lon)
            gps_end = self._calc_gps_bemamba_eq1(bs_lat, bs_lon, u2_lat, u2_lon)
            gps_features = [gps_start, gps_end]
        gps = torch.tensor(gps_features, dtype=torch.float32)

        target = torch.tensor(int(row["unit1_beam"]) - 1, dtype=torch.long)
        power_vec = torch.tensor(self._read_power(row["unit1_pwr_60ghz"]), dtype=torch.float32)

        imgs = torch.stack(imgs)
        radars = torch.stack(radars)
        lidars = torch.stack(lidars)
        missing_masks = self._build_missing_masks(idx)

        if self.missing_enabled:
            imgs = self._apply_sequence_mask(imgs, missing_masks["img"])
            radars = self._apply_sequence_mask(radars, missing_masks["radar"])
            lidars = self._apply_sequence_mask(lidars, missing_masks["lidar"])
            gps = self._apply_sequence_mask(gps, missing_masks["gps"])

        base_output = (imgs, radars, lidars, gps, target, power_vec)
        if not self.return_missing_masks:
            return base_output

        return base_output + (
            missing_masks["img"],
            missing_masks["radar"],
            missing_masks["lidar"],
            missing_masks["gps"],
        )


if __name__ == "__main__":
    MODE = "train"
    SCENARIO = "scenario32"

    # ── Part 1: basic shapes (no missing) ──────────────────────────────
    print("=" * 70)
    print("Part 1: Basic dataset (no missing) — verify shapes")
    print("=" * 70)
    ds_basic = MultimodalDataset(mode=MODE, scenario_name=SCENARIO)
    imgs, radars, lidars, gps, target, power_vec = ds_basic[0]
    print(f"  imgs   : {tuple(imgs.shape)}")
    print(f"  radars : {tuple(radars.shape)}")
    print(f"  lidars : {tuple(lidars.shape)}")
    print(f"  gps    : {tuple(gps.shape)}")
    print(f"  target : {target.item()}  (beam class, 0-63)")
    print(f"  power  : {tuple(power_vec.shape)}")

    # ── Part 2: missing enabled, show masks ────────────────────────────
    print("\n" + "=" * 70)
    print("Part 2: Missing-aware dataset — visualize masks")
    print("=" * 70)
    ds_miss = MultimodalDataset(
        mode=MODE,
        scenario_name=SCENARIO,
        missing_enabled=True,
        return_missing_masks=True,
        missing_frame_prob=0.3,
        missing_burst_prob=0.3,
        missing_burst_min=2,
        missing_burst_max=3,
        missing_modality_prob=0.2,
        missing_modalities="img,radar,lidar",
        missing_seed=42,
    )

    def visualize_mask(name, mask, length):
        """Print a frame-level mask as a visual bar."""
        bar = "".join("█" if mask[i] > 0.5 else "·" for i in range(length))
        ratio = mask.sum().item() / length * 100
        values = " ".join(f"{mask[i]:.0f}" for i in range(length))
        print(f"  {name:>6s}  [{values}]  {bar}  avail={ratio:.0f}%")

    print("\nSamples with missing enabled (seed=42, epoch=0):\n")
    for idx in range(6):
        data = ds_miss[idx]
        imgs, radars, lidars, gps, target, power_vec = data[:6]
        img_m, rad_m, lid_m, gps_m = data[6:]

        print(f"  ── sample {idx}  |  target_beam = {target.item():2d} ──")
        visualize_mask("img", img_m, 5)
        visualize_mask("radar", rad_m, 5)
        visualize_mask("lidar", lid_m, 5)
        visualize_mask("gps", gps_m, 2)
        print()

    # ── Part 3: epoch stability demo ───────────────────────────────────
    print("=" * 70)
    print("Part 3: Epoch reproducibility — same idx, same mask within epoch")
    print("=" * 70)
    a = ds_miss[0]
    b = ds_miss[0]
    img_m_a = a[6]
    img_m_b = b[6]
    same = torch.equal(img_m_a, img_m_b)
    print(f"  ds_miss[0] called twice → masks identical? {same}")
    print(f"  img_mask: [{', '.join(f'{img_m_a[i]:.0f}' for i in range(5))}]")

    ds_miss.set_missing_epoch(1)
    data_ep1 = ds_miss[0]
    img_m_ep1 = data_ep1[6]
    same_ep = torch.equal(img_m_a, img_m_ep1)
    print(f"\n  After set_missing_epoch(1), same idx[0] → masks identical? {same_ep}")
    print(f"  epoch=0 img_mask: [{', '.join(f'{img_m_a[i]:.0f}' for i in range(5))}]")
    print(f"  epoch=1 img_mask: [{', '.join(f'{img_m_ep1[i]:.0f}' for i in range(5))}]")
    print("\n(epoch changes → mask pattern changes, but stays deterministic)")

    print("\n" + "=" * 70)
    print("Done. Run on AutoDL with:  python src/dataset.py")
    print("=" * 70)
