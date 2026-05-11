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
        lidar_grid_size=256,
        lidar_x_range=(-30.0, 30.0),
        lidar_y_range=(-30.0, 30.0),
        use_virtual_points=True,
        lidar_swap_xy=False,
        lidar_flip_x=False,
        lidar_flip_y=False,
        virtual_temporal_tolerance=2,
        virtual_min_component_size=4,
        virtual_max_component_size=256,
        virtual_min_fill_ratio=0.08,
        virtual_max_bbox_size=24,
        virtual_dilation_radius=1,
    ):
        self.data_dir = data_root
        self.lidar_grid_size = int(lidar_grid_size)
        self.lidar_x_range = tuple(lidar_x_range)
        self.lidar_y_range = tuple(lidar_y_range)
        self.use_virtual_points = use_virtual_points
        self.lidar_swap_xy = lidar_swap_xy
        self.lidar_flip_x = lidar_flip_x
        self.lidar_flip_y = lidar_flip_y
        self.virtual_temporal_tolerance = int(virtual_temporal_tolerance)
        self.virtual_min_component_size = int(virtual_min_component_size)
        self.virtual_max_component_size = int(virtual_max_component_size)
        self.virtual_min_fill_ratio = float(virtual_min_fill_ratio)
        self.virtual_max_bbox_size = int(virtual_max_bbox_size)
        self.virtual_dilation_radius = int(virtual_dilation_radius)

        if csv_path is None:
            csv_path = os.path.join(split_root, f"{scenario_name}_{mode}.csv")
        self.df = pd.read_csv(csv_path)

        self.img_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self._init_gps_normalization()

    def _init_gps_normalization(self):
        all_dx, all_dy = [], []
        for idx in range(len(self.df)):
            bs_lat, bs_lon = self._read_gps_raw(self.df.iloc[idx]["unit1_loc"])
            ue_lat, ue_lon = self._read_gps_raw(self.df.iloc[idx]["unit2_loc_1"])
            if bs_lat != 0.0:
                dx = (ue_lon - bs_lon) * 111320 * np.cos(np.radians(bs_lat))
                dy = (ue_lat - bs_lat) * 111320
                all_dx.append(dx)
                all_dy.append(dy)
        if all_dx:
            self.min_dx, self.max_dx = np.min(all_dx), np.max(all_dx)
            self.min_dy, self.max_dy = np.min(all_dy), np.max(all_dy)
        else:
            self.min_dx, self.max_dx, self.min_dy, self.max_dy = 0, 1, 0, 1

    def _read_gps_raw(self, rel_path):
        try:
            with open(os.path.join(self.data_dir, rel_path), "r") as f:
                lines = f.readlines()
                return float(lines[0].strip()), float(lines[1].strip())
        except Exception:
            return 0.0, 0.0

    def _calc_gps_bemamba_eq1(self, bs_lat, bs_lon, ue_lat, ue_lon):
        dx = (ue_lon - bs_lon) * 111320 * np.cos(np.radians(bs_lat))
        dy = (ue_lat - bs_lat) * 111320
        dx_norm = (dx - self.min_dx) / (self.max_dx - self.min_dx + 1e-8)
        dy_norm = (dy - self.min_dy) / (self.max_dy - self.min_dy + 1e-8)
        dist = np.sqrt(dx_norm**2 + dy_norm**2)
        angle = np.arctan2(dy_norm, dx_norm)
        return [dist, angle]

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

    def _binary_dilate(self, bev, radius):
        if radius <= 0:
            return bev.astype(bool)

        bev = bev.astype(bool)
        dilated = np.zeros_like(bev, dtype=bool)
        h, w = bev.shape
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
                dilated[x_dst_start:x_dst_end, y_dst_start:y_dst_end] |= bev[
                    x_src_start:x_src_end, y_src_start:y_src_end
                ]
        return dilated

    def _connected_components(self, bev):
        bev = bev.astype(bool)
        visited = np.zeros_like(bev, dtype=bool)
        h, w = bev.shape
        components = []

        for x in range(h):
            for y in range(w):
                if not bev[x, y] or visited[x, y]:
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
                            if bev[nx, ny] and not visited[nx, ny]:
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

    def _filter_motion_components(self, candidate_mask):
        filtered = np.zeros_like(candidate_mask, dtype=bool)

        for component in self._connected_components(candidate_mask):
            size = component["size"]
            min_x, max_x, min_y, max_y = component["bbox"]
            bbox_h = max_x - min_x + 1
            bbox_w = max_y - min_y + 1
            bbox_area = bbox_h * bbox_w
            fill_ratio = size / max(bbox_area, 1)

            if size < self.virtual_min_component_size:
                continue
            if size > self.virtual_max_component_size:
                continue
            if bbox_h > self.virtual_max_bbox_size or bbox_w > self.virtual_max_bbox_size:
                continue
            if fill_ratio < self.virtual_min_fill_ratio:
                continue

            for x, y in component["coords"]:
                filtered[x, y] = True

        return filtered

    def _ply_to_base_bev(self, rel_path):
        grid_size = self.lidar_grid_size
        bev = np.zeros((grid_size, grid_size), dtype=np.float32)
        try:
            full_path = os.path.join(self.data_dir, rel_path)
            if not os.path.exists(full_path):
                return bev

            pcd = o3d.io.read_point_cloud(full_path)
            points = np.asarray(pcd.points)
            if len(points) == 0:
                return bev

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
                return bev

            x = x[mask]
            y = y[mask]
            x_idx = (
                (x - x_min) / (x_max - x_min + 1e-6) * (grid_size - 1)
            ).astype(np.int32)
            y_idx = (
                (y - y_min) / (y_max - y_min + 1e-6) * (grid_size - 1)
            ).astype(np.int32)
            bev[x_idx, y_idx] = 1.0
        except Exception:
            pass
        return self._apply_bev_orientation(bev)

    def _generate_virtual_points(self, current_bev, prev_bev):
        current_mask = current_bev > 0.5
        prev_mask = prev_bev > 0.5

        # Suppress pseudo-motion caused by rasterization jitter on static structures.
        prev_tolerated = self._binary_dilate(prev_mask, self.virtual_temporal_tolerance)
        candidate_motion = current_mask & (~prev_tolerated)

        # Keep only compact local regions that look more like moving targets.
        filtered_motion = self._filter_motion_components(candidate_motion)
        virtual_mask = self._binary_dilate(filtered_motion, self.virtual_dilation_radius)

        return np.maximum(current_mask.astype(np.float32), virtual_mask.astype(np.float32))

    def _resize_tensor(self, tensor, size=(256, 256)):
        return F.interpolate(
            tensor.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    def get_lidar_debug_sequence(self, idx):
        row = self.df.iloc[idx]
        base_bevs = []
        virtual_bevs = []
        combined_bevs = []
        prev_base_bev = None

        for t in range(1, 6):
            base_bev = self._ply_to_base_bev(row[f"unit1_lidar_{t}"])
            if self.use_virtual_points and prev_base_bev is not None:
                combined_bev = self._generate_virtual_points(base_bev.copy(), prev_base_bev)
                virtual_bev = np.clip(combined_bev - base_bev, 0.0, 1.0)
            else:
                combined_bev = base_bev.copy()
                virtual_bev = np.zeros_like(base_bev)

            base_bevs.append(base_bev)
            virtual_bevs.append(virtual_bev)
            combined_bevs.append(combined_bev)
            prev_base_bev = base_bev.copy()

        return {
            "base": np.stack(base_bevs),
            "virtual": np.stack(virtual_bevs),
            "combined": np.stack(combined_bevs),
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgs, radars, lidars = [], [], []
        prev_bev = None

        for t in range(1, 6):
            img_path = os.path.join(self.data_dir, row[f"unit1_rgb_{t}"])
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

            current_bev = self._ply_to_base_bev(row[f"unit1_lidar_{t}"])
            if self.use_virtual_points and prev_bev is not None:
                current_bev = self._generate_virtual_points(current_bev, prev_bev)
            prev_bev = current_bev.copy()
            lidars.append(self._resize_tensor(torch.tensor(current_bev).unsqueeze(0)))

        bs_lat, bs_lon = self._read_gps_raw(row["unit1_loc"])
        u1_lat, u1_lon = self._read_gps_raw(row["unit2_loc_1"])
        u2_lat, u2_lon = self._read_gps_raw(row["unit2_loc_2"])
        gps_start = self._calc_gps_bemamba_eq1(bs_lat, bs_lon, u1_lat, u1_lon)
        gps_end = self._calc_gps_bemamba_eq1(bs_lat, bs_lon, u2_lat, u2_lon)
        gps = torch.tensor([gps_start, gps_end], dtype=torch.float32)

        target = torch.tensor(int(row["unit1_beam"]) - 1, dtype=torch.long)
        power_vec = torch.tensor(self._read_power(row["unit1_pwr_60ghz"]), dtype=torch.float32)

        return torch.stack(imgs), torch.stack(radars), torch.stack(lidars), gps, target, power_vec


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print(">>> Instantiating MultimodalDataset...")
    dataset = MultimodalDataset(mode="train", scenario_name="scenario32")
    print(f">>> Dataset size: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    imgs, radars, lidars, gps, targets, power_vec = next(iter(dataloader))

    print("\n" + "=" * 50)
    print(f"1. RGB shape:      {imgs.shape}")
    print(f"2. Radar shape:    {radars.shape}")
    print(f"3. LiDAR BEV:      {lidars.shape}")
    print(f"4. GPS shape:      {gps.shape}")
    print(f"5. Target shape:   {targets.shape}")
    print(f"6. Power shape:    {power_vec.shape}")
    print("=" * 50 + "\n")
