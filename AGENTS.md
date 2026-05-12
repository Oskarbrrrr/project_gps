# AGENTS

## 1. 项目定位

这是一个用于复现论文 `BeMamba: Efficient Multimodal Sensing-Aided Beamforming via State Space Model` 的本地代码仓库。

这个仓库通常只保存代码，不包含：

- 数据集本体
- 训练生成的 checkpoint
- AutoDL 上的大量日志与实验产物

这些内容默认都在 AutoDL 上。

当前最重要的代码文件：

- [train.py](/D:/code/project_gps/train.py)
- [src/dataset.py](/D:/code/project_gps/src/dataset.py)
- [src/model.py](/D:/code/project_gps/src/model.py)
- [src/utils.py](/D:/code/project_gps/src/utils.py)
- [visualize_lidar.py](/D:/code/project_gps/visualize_lidar.py)
- [AUTODL_WORKFLOW.md](/D:/code/project_gps/AUTODL_WORKFLOW.md)

## 2. 当前代码状态

### 2.1 数据输入

`MultimodalDataset` 当前返回：

- `imgs`: `[5, 3, 256, 256]`
- `radars`: `[5, 2, 256, 256]`
- `lidars`: `[5, 1, 256, 256]`
- `gps`: `[2, 2]`
- `target`: 单个 beam 类别，范围 `[0, 63]`
- `power_vec`: `[64]`

也就是说，每个样本包含：

- 5 帧图像
- 5 帧雷达
- 5 帧 LiDAR
- 起点/终点 GPS

### 2.2 LiDAR 预处理

LiDAR 已经不是最早那种随机撒点版了，目前是“质量优先”的近似复现实现。

当前流程：

1. 把点云投影到固定范围的 BEV：
   - `x ∈ [-30m, 30m]`
   - `y ∈ [-30m, 30m]`
2. 用前后帧的低分辨率点数图做 motion cue
3. 过滤掉不合理的小块或大块运动区域
4. 在高分辨率密度图上做局部增强，生成 virtual points
5. 最终 LiDAR 输入为：
   - `combined = max(base_bev, virtual_points)`

要注意：

- 这版 LiDAR 预处理已经比较接近论文风格
- 但它仍然是“高质量近似实现”
- 不能声称它一定和论文作者的原始未公开代码逐行一致

### 2.3 模型结构

`src/model.py` 现在已经重构成更清楚的 BeMamba 复现风格。

当前模型主线：

1. 每个模态先用 CNN backbone 做编码
2. 每个模态再做时序 Mamba 建模
3. GPS 起点/终点投影成 token
4. 用三种模态顺序做跨模态融合
5. 每种顺序各自走一套 bidirectional Mamba fusion
6. 三路结果取平均，最后做分类

当前 backbone 设定：

- Image: `ResNet34`
- Radar: `ResNet18`，输入 2 通道
- LiDAR: `ResNet18`，输入 1 通道

当前设计要点：

- 先做模态内时序建模
- 再做模态间序列融合
- 空间特征先池化成固定 patch grid，默认 `6 x 6`

### 2.4 训练入口

`train.py` 已经从“写死流程”改成了配置化实验入口。

当前具备：

- 支持命令行参数
- 支持指定单场景或多场景
- 固定随机种子
- `AdamW`
- warmup + cosine 学习率调度
- AMP 混合精度
- gradient clipping
- early stopping
- 每次实验保存独立输出目录
- 保存日志和配置快照

示例：

```bash
python train.py --scenarios scenario34 --epochs 80 --batch-size 8
```

## 3. 当前哪些部分比较对齐，哪些仍然是近似

### 3.1 相对已经比较对齐的部分

- 多模态输入形式
- 5 帧时序结构
- Radar 双通道读法
- LiDAR 固定范围 BEV + motion-focused enhancement
- Mamba 用于时序和跨模态融合的总体思路

### 3.2 还不能说完全对齐的部分

- LiDAR virtual point generation 的精确细节
- GPS 归一化是否与论文完全一致
- 训练超参数是否与论文完全一致
- 数据划分是否与论文官方 split 完全一致

所以当前仓库状态更适合叫：

**“高质量复现版本”**

而不是：

**“论文原始实现逐行复刻版本”**

## 4. 本地与 AutoDL 的分工

### 本地负责

- 阅读和修改代码
- 做静态检查
- 维护 git 历史
- 看从 AutoDL 带回来的图和日志

### AutoDL 负责

- 挂载或存放数据集
- 跑训练
- 跑可视化
- 保存 checkpoint
- 保存实验日志

## 5. 推荐工作流

每次尽量按这个闭环走：

1. 在本地改代码
2. 本地 `git commit`
3. 本地 `git push`
4. AutoDL 上同步代码
5. AutoDL 上运行训练或可视化
6. 把结果带回本地分析

不推荐：

- 本地改一部分，AutoDL 再手改一部分但不提交
- AutoDL 上长期保留未提交改动
- 不清楚当前结果到底是哪一版代码跑出来的

前面已经出现过版本混乱，所以后续要尽量避免。

## 6. 运行与排错提醒

### 6.1 LiDAR 排查

只要怀疑 LiDAR 不对，优先跑：

- [visualize_lidar.py](/D:/code/project_gps/visualize_lidar.py)

重点看：

- Base BEV
- Raw Motion Cue
- Region Mask
- Virtual Points
- Combined BEV

### 6.2 训练最小冒烟

当模型或训练逻辑刚改完时，先不要直接跑完整实验。

先用这种最小配置检查链路：

```bash
python train.py --scenarios scenario34 --epochs 1 --batch-size 2 --num-workers 2 --d-model 128 --temporal-layers 1 --fusion-layers 1
```

如果这条能跑通，说明下面这些都基本没坏：

- dataset
- dataloader
- model forward
- loss
- backward
- validation
- checkpoint 保存

### 6.3 AutoDL git 问题

如果 AutoDL 上：

- `git fetch`
- `git pull`

经常卡住或 TLS 出错，不要立刻怀疑代码。先怀疑容器到 GitHub 的网络。

如果 AutoDL 工作区有脏文件，拉代码前先处理干净。

## 7. 当前实现的关键结论

到 `2026-05-12` 为止，仓库里已经完成了三件重要事：

1. LiDAR 预处理已经重构成高质量近似复现版本
2. `src/model.py` 已重构成更清楚的双阶段 BeMamba 风格架构
3. `train.py` 已重构成配置化实验入口

## 8. 后续协作时默认记住

- 当前正式运行环境是 AutoDL
- 当前本地仓库路径是 `D:\code\project_gps`
- 用户当前优先级是“高质量复现”，不优先省算力
- 4090D 可用于跑更重的版本
- 如果只是验证代码链路，优先先跑单场景、单 epoch 冒烟

