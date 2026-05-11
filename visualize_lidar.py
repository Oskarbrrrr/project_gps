import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 导入我们刚刚写好的 Dataset
from src.dataset import MultimodalDataset

def visualize_bev_sequence(scenario_name="scenario32", sample_idx=10):
    print(f"正在加载 {scenario_name} 数据集...")
    # 实例化数据集
    dataset = MultimodalDataset(mode='train', scenario_name=scenario_name)
    
    print(f"正在抽取第 {sample_idx} 个样本...")
    # 获取特定索引的一个样本
    imgs, radars, lidars, gps, target, power_vec = dataset[sample_idx]
    
    # lidars 的 shape 是 [5帧, 1通道, 256, 256]
    # 我们把它转为 numpy 方便用 matplotlib 画图
    lidar_np = lidars.squeeze(1).numpy() # 变成 [5, 256, 256]
    
    # 创建一个 1 行 5 列的大图
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f'LiDAR BEV Sequence with Virtual Points (Scenario: {scenario_name}, Sample: {sample_idx})', fontsize=16)
    
    for t in range(5):
        ax = axes[t]
        # 使用深色背景 (viridis 伪彩色)，1.0 (车/物体) 会显示为亮黄色
        ax.imshow(lidar_np[t], cmap='viridis', origin='lower')
        ax.set_title(f'Frame {t+1}')
        ax.axis('off')
        
    plt.tight_layout()
    
    # 保存图片
    os.makedirs("logs", exist_ok=True)
    out_path = f"logs/lidar_bev_vis_sample_{sample_idx}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化完成！图片已保存至: {out_path}")

if __name__ == "__main__":
    # 你可以修改 sample_idx 来多看几个不同的样本
    visualize_bev_sequence("scenario34", sample_idx=50)
    visualize_bev_sequence("scenario34", sample_idx=100)