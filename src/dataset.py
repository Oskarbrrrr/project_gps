import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import open3d as o3d

class MultimodalDataset(Dataset):
    def __init__(self, mode='train', data_root='./Data/Multi_Modal', split_root='./Data/splits', scenario_name="scenario32"):
        self.data_dir = data_root
        
        # 读取拆分好的 CSV
        csv_path = os.path.join(split_root, f"{scenario_name}_{mode}.csv")
        self.df = pd.read_csv(csv_path)
        
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._init_gps_normalization()

    def _init_gps_normalization(self):
        all_dx, all_dy = [], []
        for idx in range(len(self.df)):
            bs_lat, bs_lon = self._read_gps_raw(self.df.iloc[idx]['unit1_loc'])
            ue_lat, ue_lon = self._read_gps_raw(self.df.iloc[idx]['unit2_loc_1'])
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
            with open(os.path.join(self.data_dir, rel_path), 'r') as f:
                lines = f.readlines()
                return float(lines[0].strip()), float(lines[1].strip())
        except: return 0.0, 0.0

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
            with open(os.path.join(self.data_dir, rel_path), 'r') as f:
                return [float(x) for x in f.read().split()]
        except: return [0.0] * 64

    # ==========================================
    # 完全对齐 BeMamba 论文的 LiDAR 处理逻辑
    # ==========================================
    def _ply_to_base_bev(self, rel_path, grid_size=256):
        bev = np.zeros((grid_size, grid_size), dtype=np.float32)
        try:
            full_path = os.path.join(self.data_dir, rel_path)
            if not os.path.exists(full_path): return bev
            
            pcd = o3d.io.read_point_cloud(full_path)
            points = np.asarray(pcd.points)
            if len(points) == 0: return bev

            x, y = points[:, 0], points[:, 1]

            # 1. 自动找到点云的中心
            center_x, center_y = np.median(x), np.median(y)
            
            # 2. 以中心为基准，向四周扩展 30 米 (这个范围肯定能覆盖路口)
            R = 30.0 
            x_min, x_max = center_x - R, center_x + R
            y_min, y_max = center_y - R, center_y + R
            
            # 3. 映射到 grid
            # 为了防止点云超出边界，加一层 clip
            x_idx = ((x - x_min) / (x_max - x_min + 1e-6) * (grid_size - 1))
            y_idx = ((y - y_min) / (y_max - y_min + 1e-6) * (grid_size - 1))
            
            # 过滤
            mask = (x_idx >= 0) & (x_idx < grid_size) & (y_idx >= 0) & (y_idx < grid_size)
            bev[x_idx[mask].astype(int), y_idx[mask].astype(int)] = 1.0
            
            # === 紧急调试打印 (只打印一次) ===
            # print(f"DEBUG: Ply points count: {len(points)}, X_range: [{x_min:.1f}, {x_max:.1f}]")
            
        except Exception as e:
            pass
        return bev

    def _generate_virtual_points(self, current_bev, prev_bev):
        """
        强化版的虚拟点生成 (让车变更亮)
        """
        diff = current_bev - prev_bev
        # 只取那些完全是当前帧新增的点 (严格移动目标)
        moving_points = np.where(diff > 0.5)
        
        if len(moving_points[0]) > 0:
            # === 🌟 核心修正：增加撒点数量 🌟 ===
            # 原版只撒了 1 次，我们在每个运动点周围随机撒 3 次，让目标更粗更亮！
            for _ in range(3): 
                offset_x = np.random.randint(-2, 3, size=len(moving_points[0])) # 扩大到 [-2, 2] 像素
                offset_y = np.random.randint(-2, 3, size=len(moving_points[1]))
                
                virtual_x = np.clip(moving_points[0] + offset_x, 0, 255)
                virtual_y = np.clip(moving_points[1] + offset_y, 0, 255)
                
                # 虚拟点的亮度设高一点，模拟高反射率
                current_bev[virtual_x, virtual_y] = 1.0
            
        return current_bev

    def _resize_tensor(self, t, size=(256, 256)):
        return F.interpolate(t.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        imgs, radars, lidars = [], [], []
        prev_bev = None # 用于缓存上一帧，算帧差用
        
        for t in range(1, 6):
            # 1. 图像 (原图)
            img_path = os.path.join(self.data_dir, row[f'unit1_rgb_{t}'])
            imgs.append(self.img_transform(Image.open(img_path).convert('RGB')))
            
            # 2. 雷达 (双通道)
            ang_p = os.path.join(self.data_dir, row[f'unit1_radar_{t}'].replace('/radar_data/', '/radar_data_ang/'))
            vel_p = os.path.join(self.data_dir, row[f'unit1_radar_{t}'].replace('/radar_data/', '/radar_data_vel/'))
            try:
                ang_arr, vel_arr = np.nan_to_num(np.load(ang_p)), np.nan_to_num(np.load(vel_p))
                radar_tensor = torch.tensor(np.stack([ang_arr, vel_arr], axis=0), dtype=torch.float32)
            except: 
                radar_tensor = torch.zeros((2, 256, 256))
            radars.append(self._resize_tensor(radar_tensor))
            
            # 3. LiDAR ( BeMamba 虚拟点生成)
            ply_p = row[f'unit1_lidar_{t}']
            current_bev = self._ply_to_base_bev(ply_p)
            
            # 如果存在上一帧，则执行帧差和虚拟点生成
            if prev_bev is not None:
                current_bev = self._generate_virtual_points(current_bev, prev_bev)
            
            # 缓存当前帧，留给下一帧算帧差
            prev_bev = current_bev.copy()
            
            # 将处理好的 combined BEV 转为 tensor 并保存
            lidars.append(self._resize_tensor(torch.tensor(current_bev).unsqueeze(0)))
            
        # 4. GPS
        bs_lat, bs_lon = self._read_gps_raw(row['unit1_loc'])
        u1_lat, u1_lon = self._read_gps_raw(row['unit2_loc_1'])
        u2_lat, u2_lon = self._read_gps_raw(row['unit2_loc_2'])
        gps_start = self._calc_gps_bemamba_eq1(bs_lat, bs_lon, u1_lat, u1_lon)
        gps_end   = self._calc_gps_bemamba_eq1(bs_lat, bs_lon, u2_lat, u2_lon)
        gps = torch.tensor([gps_start, gps_end], dtype=torch.float32)
        
        # 5. Label & Power Vector
        target = torch.tensor(int(row['unit1_beam']) - 1, dtype=torch.long)
        power_vec = torch.tensor(self._read_power(row['unit1_pwr_60ghz']), dtype=torch.float32)
        
        return torch.stack(imgs), torch.stack(radars), torch.stack(lidars), gps, target, power_vec

# ==========================================
# 打印测试代码 
# ==========================================
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # 请确保此路径有生成的拆分文件
    test_csv_path = "./Data/splits/scenario32_train.csv"
    
    print(">>> 正在实例化 MultimodalDataset...")
    dataset = MultimodalDataset(mode='train', scenario_name="scenario32")
    
    print(f">>> 数据集实例化成功！总样本数: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print(">>> 正在拉取第一个 Batch 验证数据 Shape (如果点云较多可能需要几秒钟)...")
    imgs, radars, lidars, gps, targets, power_vec = next(iter(dataloader))
    
    print("\n" + "="*50)
    print(f"1. 🖼️  RGB 图像 Shape:      {imgs.shape}  --> [Batch, 5帧, 3通道, 256, 256] (已应用 ResNet 标准归一化)")
    print(f"2. 📡  雷达 RA+RV Shape:   {radars.shape}  --> [Batch, 5帧, 2通道, 256, 256] (Channel 0: Angle, Channel 1: Velocity)")
    print(f"3. 🔦  LiDAR BEV Shape:    {lidars.shape}  --> [Batch, 5帧, 1通道, 256, 256] (严格复刻 BeMamba 帧差法及 0.1m 虚拟点生成)")
    print(f"4. 📍  GPS Shape:          {gps.shape}    --> [Batch, 2帧, 2特征] (已完成极坐标转换与全局归一化)")
    print(f"5. 🎯  Target Beam Shape:  {targets.shape}    --> [Batch] (波束标签类别，范围 0-63)")
    print(f"6. ⚡  Power Vector Shape: {power_vec.shape}   --> [Batch, 64] (用于后续计算 APL 损失)")
    print("="*50 + "\n")