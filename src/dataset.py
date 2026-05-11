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
        lidar_motion_grid_size=64,
        lidar_motion_history_dilation=1,
        lidar_motion_region_expand=6,
        lidar_virtual_expand=2,
        lidar_motion_min_cells=2,
        lidar_motion_max_cells=48,
    ):
        self.data_dir = data_root
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
        self.lidar_virtual_expand = int(lidar_virtual_expand)
        self.lidar_motion_min_cells = int(lidar_motion_min_cells)
        self.lidar_motion_max_cells = int(lidar_motion_max_cells)

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
            return {
                "base": base_bev,
                "raw_motion": zeros,
                "region_mask": zeros,
                "virtual": zeros,
                "combined": base_bev.copy(),
            }

        current_coarse = self._points_to_bev(current_points, self.lidar_motion_grid_size) > 0.5
        prev_coarse = self._points_to_bev(prev_points, self.lidar_motion_grid_size) > 0.5
        prev_coarse_dilated = self._binary_dilate(prev_coarse, self.lidar_motion_history_dilation)
        raw_motion_coarse = current_coarse & (~prev_coarse_dilated)
        filtered_motion_coarse = self._filter_motion_coarse(raw_motion_coarse)
        motion_coarse = filtered_motion_coarse if np.any(filtered_motion_coarse) else raw_motion_coarse

        raw_motion = self._upsample_mask(motion_coarse, self.lidar_grid_size)
        region_mask = self._binary_dilate(raw_motion, self.lidar_motion_region_expand)

        base_mask = base_bev > 0.5
        virtual_seed = base_mask & region_mask
        virtual_mask = self._binary_dilate(virtual_seed, self.lidar_virtual_expand)

        # Keep only the enhancement structure, not the whole base map.
        virtual_only = virtual_mask & (~base_mask)
        combined = np.maximum(base_mask.astype(np.float32), virtual_mask.astype(np.float32))

        return {
            "base": base_bev.astype(np.float32),
            "raw_motion": raw_motion.astype(np.float32),
            "region_mask": region_mask.astype(np.float32),
            "virtual": virtual_only.astype(np.float32),
            "combined": combined.astype(np.float32),
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
            "raw_motion": [],
            "region_mask": [],
            "virtual": [],
            "combined": [],
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgs, radars, lidars = [], [], []
        prev_points = None

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

            current_points = self._read_lidar_xy(row[f"unit1_lidar_{t}"])
            layers = self._build_lidar_layers(current_points, prev_points)
            prev_points = current_points.copy()
            lidars.append(self._resize_tensor(torch.tensor(layers["combined"]).unsqueeze(0)))

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
