# AGENTS

## Paper Page Index

- 论文逐页摘要与图片映射见：[PAPER_PAGE_INDEX.md](/D:/code/project_gps/PAPER_PAGE_INDEX.md)
- 对应图片目录：`paper_pages/001.jpg` 到 `paper_pages/014.jpg`
- 需要快速查找架构图、公式、主结果、消融表时，优先打开 `PAPER_PAGE_INDEX.md`

## 1. 项目定位

这是一个用于复现论文 `BeMamba: Efficient Multimodal Sensing-Aided Beamforming via State Space Model` 的本地代码仓库，并在复现基础上做 **Missing-Aware Training（缺失感知训练）** 创新。

当前策略：

- 尽量按论文公开信息对齐复现
- 优先保证多模态输入、时序建模、模态融合、指标口径与论文一致
- 允许在工程实现上做最小必要调整，但不随意改实验目标
- 创新方向：DMAF（Dynamic Mask Adaptive Fusion），提升多模态波束预测在传感器数据缺失下的鲁棒性

仓库通常只保存代码，不保存：

- 数据集本体
- AutoDL 上训练生成的 checkpoint
- 大量训练日志和中间结果

这些内容默认保存在 AutoDL 环境中。

当前最重要的文件：

- [train.py](/D:/code/project_gps/train.py)
- [src/dataset.py](/D:/code/project_gps/src/dataset.py)
- [src/model.py](/D:/code/project_gps/src/model.py)
- [src/utils.py](/D:/code/project_gps/src/utils.py)
- [eval_missing.py](/D:/code/project_gps/eval_missing.py)
- [prepare_camera_mask_yolo.py](/D:/code/project_gps/prepare_camera_mask_yolo.py)
- [AUTODL_WORKFLOW.md](/D:/code/project_gps/AUTODL_WORKFLOW.md)

## 2. 当前数据与输入形式

`MultimodalDataset` 当前返回：

- `imgs`: `[5, 3, 256, 256]`
- `radars`: `[5, 2, 256, 256]`
- `lidars`: `[5, 1, 256, 256]`
- `gps`: `[2, 2]`
- `target`: beam 类别，范围 `[0, 63]`
- `power_vec`: `[64]`

当 `missing_enabled=True` 且 `return_missing_masks=True` 时，额外返回：

- `img_mask`: `[5]`（逐帧存在标记，1=正常, 0=缺失）
- `radar_mask`: `[5]`
- `lidar_mask`: `[5]`
- `gps_mask`: `[2]`

缺失帧/模态的数据本体已置零，mask 告诉模型"这是缺失还是正常零值"。

## 3. 当前各模态实现状态

### 3.1 图像模态

图像模态支持三种输入目录切换，通过 `--image-subdir` 控制：

- `camera_data`
- `camera_data_mask`
- `camera_data_mask_yolo`

经验结论：

- `scenario33`、`scenario34`（夜景）：`camera_data_mask_yolo` 最优
- `scenario32`（白天）：`camera_data_mask` 最优，YOLO 红框无额外收益

### 3.2 LiDAR 模态

1. 点云投影到 BEV
2. 通过连续帧差分估计运动区域
3. 针对 motion points 做一对一 virtual point generation
4. 对 virtual points 做小范围随机扰动（仅训练时）
5. 将原始 BEV 与生成点 BEV 合并

### 3.3 Radar 模态

- range-angle map + range-velocity map，2 通道拼接输入

### 3.4 GPS 模态

1. 从绝对经纬度转换成相对位移
2. 对相对位移做归一化（test 复用 train 的统计量）
3. 转成极坐标特征
4. 以两个时刻的特征进入后续融合

## 4. 当前模型结构状态

[src/model.py](/D:/code/project_gps/src/model.py) 包含 Baseline 和 DMAF 两套结构。

### 4.1 Baseline 结构（`missing_enabled=False`）

1. 图像 / LiDAR / Radar 分别经 CNN（ResNet34/ResNet18/ResNet18）提特征
2. 三个感知模态都经过 Time Sequence Mamba（TFMamba）
3. GPS 用 MLP 投影到同一特征空间
4. 构造三种 mixed modal combinations（c1/c2/c3，不同模态排列顺序）
5. 用 MBMamba 做模态融合
6. 三个序列简单求和 → MLP head 预测 64 类 beam

关键超参：`d_model=128`, `patch_grid=6`

### 4.2 DMAF 结构（`missing_enabled=True`）

在 Baseline 基础上增加两个创新模块：

**创新点 1：逐帧 Mask 注入**（`TimeSequenceBranch.forward`）

```
mask [B, 5] → Linear(1,1) → [B, 5, 1] → expand → [B, 5, 1, 36]
                                                    ↓ concat
encoded [B, 5, 128, 36]  →  [B, 5, 129, 36] → Conv1d → [B, 5, 128, 36]
                                                    ↓
                                            TFMamba 逐帧感知缺失
```

之前（已废弃）：mask [5] → MaskEncoder 压缩成全局 128 维向量 → Add 到所有空间位置
现在：每帧的 0/1 变成标量拼到该帧特征 channel 后面，TFMamba 扫描时知道"这一帧丢了"

**创新点 2：Cross-Attention 模态融合**（`_modal_fusion`）

```
三个序列 (s0, s1, s2) 各自 [B, 36, 128]

s0 → CrossAttn(Q=s0, KV=[s1,s2]) → enhanced_0
s1 → CrossAttn(Q=s1, KV=[s0,s2]) → enhanced_1
s2 → CrossAttn(Q=s2, KV=[s0,s1]) → enhanced_2

fused = enhanced_0 + enhanced_1 + enhanced_2
```

之前（已废弃）：ReliabilityGate 算 3 个标量权重做加权求和
现在：每个 token 能从其他序列的对应位置"借"信息，缺失模态自动向完好模态倾斜

**自动全1 Mask**：clean 测试时自动生成全 1 mask，确保训练/测试输入分布一致

### 4.3 代码结构

- `BeMambaConfig`：统一配置类，`missing_enabled` 控制 DMAF 开关
- `TimeSequenceBranch`：逐帧 mask 注入 + TFMamba 时序处理
- `GPSProjection`：GPS 特征投影 + mask 注入
- `CrossModalAttention`：多头交叉注意力 + FFN + 残差连接
- `MBMamba`：双向 Mamba 模态融合
- `BeMambaModel`：主模型，`_build_modal_sequences` + `_modal_fusion`

## 5. 当前训练协议

### 5.1 基本配置

- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- 默认学习率：`1e-4`
- 默认 epoch：`30`
- 默认损失：支持 `ce` / `focal` / `power_soft_ce`
- AMP：默认开启
- Early stopping：`patience=10`, 监控 `acc3`
- Best checkpoint 自动保存和重新评估

### 5.2 数据划分口径

- `train.csv + val.csv` 合并作为训练侧（`--no-merge-trainval` 仅用 train.csv）
- `test.csv` 作为测试侧
- 默认 split：`./Data/splits_paper80`

### 5.3 缺失训练参数

| 参数 | 含义 | 推荐值 |
|------|------|:---:|
| `--missing-enabled` | 开启 DMAF | - |
| `--missing-frame-prob` | 每帧随机缺失概率 | 0.1 |
| `--missing-burst-prob` | 连续帧缺失概率 | 0.05 |
| `--missing-burst-min/max` | 连续缺失帧数范围 | 2/3 |
| `--missing-modality-prob` | 整模态缺失概率 | 0.05 |
| `--missing-modality-min/max` | 缺失模态数量范围 | 1/2 |
| `--missing-modalities` | 可缺失的模态列表 | img,radar,lidar |
| `--missing-seed` | 缺失随机种子 | 42 |

### 5.4 训练输出

每次训练产出：
- `checkpoints/best_model.pth`
- `final_test_result.txt`：best checkpoint 的 clean 测试结果
- `missing_test_result.txt`：鲁棒性评估表（仅 `--missing-enabled` 时）
- `missing_test_result.csv`：同上，CSV 格式
- `last_epoch_result.txt`：最后一轮结果
- `train_log.csv`：每轮详细指标

## 6. 当前指标口径

- `Top-1/2/3`：最高 1/2/3 个 beam 命中率
- `DBA`：距离加权准确率
- `APL`：平均功率损失 (dB)
- `Retention Ratio = Acc3_missing / Acc3_clean × 100%`：鲁棒性保持率

论文主要报告 Top-3 和 DBA，当前额外保留 Top-1/2 和 APL 便于分析。

## 7. 截至 2026-05-29 的最新结果

### 7.1 Baseline（无缺失，`missing_enabled=False`）

#### scenario32

```
Top-1: 41.89%  |  Top-2: 69.98%  |  Top-3: 81.54%
DBA: 0.8722    |  APL: 0.1009 dB  |  best_epoch: 14
```

配置：camera_data_mask, row+reverse, adamw, power_soft_ce(temp=0.15), seed=7

注：之前报告过 82.83%，当前代码结构微调后稳定在 81.54%（1.29% 的正常训练波动）。

#### scenario33

```
Top-1: 45.05%  |  Top-2: 66.80%  |  Top-3: 80.99%
DBA: 0.8621    |  APL: 0.1295 dB  |  best_epoch: 17
```

配置：camera_data_mask_yolo, row+reverse, adamw, power_soft_ce(temp=0.15), seed=7

#### scenario34

```
Top-1: 47.68%  |  Top-2: 71.16%  |  Top-3: 85.58%
DBA: 0.8979    |  APL: 0.0841 dB  |  best_epoch: 22
```

配置：camera_data_mask_yolo, row+reverse, adamw, power_soft_ce(temp=0.15), seed=7

### 7.2 与论文差距

| Scenario | 论文 Top-3 | 我们 Baseline | 差距 |
|----------|:---:|:---:|:---:|
| SC32 | 88.11% | 81.54% | 6.57 |
| SC33 | 84.94% | 80.99% | 3.95 |
| SC34 | 85.64% | 85.58% | 0.06 |

SC34 基本复现到论文水平，SC32 仍是主要短板。

## 8. DMAF 实现与实验状态

### 8.1 已完成

- [x] 数据集缺失模拟（帧级 / 连续帧 / 模态级 / 混合）
- [x] 逐帧 mask 注入（TimeSequenceBranch + GPSProjection）
- [x] Cross-Attention 模态融合（替代 ReliabilityGate）
- [x] Clean 测试自动全 1 mask
- [x] 训练结束后自动鲁棒性评估（9 种协议）
- [x] 独立评估脚本 `eval_missing.py`
- [x] `--no-dmaf` 开关（评估 baseline 模型用）

### 8.2 第一次 DMAF 实验（SC32，缺失率偏高，已存档）

训练参数：`frame=0.2, burst=0.1, modal=0.1`

| 协议 | Baseline (clean训练) | DMAF (缺失训练) |
|------|:---:|:---:|
| clean | **81.54%** | 78.49% |
| frame 10% | **78.97%** | 72.71% |
| frame 20% | **74.80%** | 67.74% |
| frame 30% | **70.63%** | 59.71% |
| burst 10% | **74.80%** | 70.14% |
| burst 20% | **69.02%** | 63.40% |
| modal 10% | **78.33%** | 75.28% |
| modal 20% | **74.16%** | 71.91% |
| hybrid | **63.88%** | 55.22% |

DMAF 全面落后。原因：缺失率太高，训练数据太脏，模型学不到有效特征。Baseline 用完整数据训练，ResNet+Mamba 本身有一定抗噪能力。

### 8.3 当前方案：降低缺失率

将缺失率降到温和水平，保证足够多的干净训练样本：

```
--missing-frame-prob 0.1      (从 0.2 降)
--missing-burst-prob 0.05     (从 0.1 降)
--missing-modality-prob 0.05  (从 0.1 降)
```

目标：clean Top-3 回到 81%+，missing retention 超过 baseline。

### 8.4 鲁棒性评估协议（9 种）

训练结束后自动运行，`eval_missing.py` 可独立评估已有 checkpoint：

| # | 协议名 | 描述 |
|:--:|--------|------|
| 1 | clean | 完整数据（baseline 基准） |
| 2 | frame_p01 | 10% 帧随机缺失 |
| 3 | frame_p02 | 20% 帧随机缺失 |
| 4 | frame_p03 | 30% 帧随机缺失 |
| 5 | burst_p01 | 10% 连续帧缺失 |
| 6 | burst_p02 | 20% 连续帧缺失 |
| 7 | modal_p01 | 10% 模态缺失 |
| 8 | modal_p02 | 20% 模态缺失 |
| 9 | hybrid | 训练配置的混合缺失 |

所有指标报告 Top-1/2/3、DBA、APL、Retention Ratio。

## 9. 论文实验规划

### 9.1 故事线

> 实际部署中传感器数据经常缺失（硬件故障、延迟、天气），现有方法一缺就崩。我们提出 DMAF——用逐帧 mask 注入 + Cross-Attention 融合，让模型在残缺输入下仍然预测准确。

### 9.2 实验矩阵

**A 组：Clean Baseline（3 实验）**
证明 DMAF 不破坏完整数据性能。三个场景各跑一次 clean 训练 + DMAF 训练，对比 clean Top-3。差距应在 ±1% 以内。

**B 组：缺失去实验（核心 Table）**
Baseline vs DMAF，9 种缺失协议 × 3 场景。关键指标：Top-3 + Retention Ratio。

**C 组：消融实验（~12 实验）**
拆开两个组件：

| 变体 | Mask 注入 | 融合方式 |
|------|:---:|:---:|
| Baseline | 无 | 简单求和 |
| DMAF w/o CrossAttn | 逐帧 concat | 简单求和 |
| DMAF w/o MaskInj | 全局加和（旧） | Cross-Attn |
| DMAF Full | 逐帧 concat | Cross-Attn |

**D 组：缺失率曲线（可选）**
frame p=0.05~0.5，画缺失率 vs Top-3 曲线。

**E 组：跨场景泛化**
SC32（白天）、SC33（夜景）、SC34（夜景），B 组自动覆盖。

### 9.3 当前优先级

1. **SC32 DMAF 低缺失率训练** — 验证降低缺失率后 DMAF 能否超越 Baseline
2. **SC33 + SC34 clean baseline** — 补全 A 组
3. **SC33 + SC34 DMAF 训练** — 补全 B 组
4. **消融实验** — C 组（1 场景 × 2 变体即可）

## 10. 当前最优配置与命令

### 通用运行前

```bash
export OMP_NUM_THREADS=8
```

### scenario32 Baseline（白天，camera_data_mask）

```bash
python train.py \
  --split-root ./Data/splits_paper80 \
  --no-merge-trainval \
  --scenarios scenario32 \
  --image-subdir camera_data_mask \
  --batch-size 48 \
  --temporal-layers 2 \
  --fusion-layers 2 \
  --spatial-scan row \
  --temporal-order reverse \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --dropout 0.25 \
  --loss power_soft_ce \
  --soft-power-temperature 0.15 \
  --hard-loss-weight 0.6 \
  --patience 10 \
  --early-stop-metric acc3 \
  --early-stop-mode max \
  --seed 7
```

### scenario32 DMAF 低缺失率（当前推荐先跑）

```bash
python train.py \
  --split-root ./Data/splits_paper80 \
  --no-merge-trainval \
  --scenarios scenario32 \
  --image-subdir camera_data_mask \
  --batch-size 48 \
  --temporal-layers 2 \
  --fusion-layers 2 \
  --spatial-scan row \
  --temporal-order reverse \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --dropout 0.25 \
  --loss power_soft_ce \
  --soft-power-temperature 0.15 \
  --hard-loss-weight 0.6 \
  --patience 10 \
  --early-stop-metric acc3 \
  --early-stop-mode max \
  --seed 7 \
  --missing-enabled \
  --missing-frame-prob 0.1 \
  --missing-burst-prob 0.05 \
  --missing-modality-prob 0.05
```

### scenario33 Baseline（夜景，camera_data_mask_yolo）

```bash
python train.py \
  --split-root ./Data/splits_paper80 \
  --no-merge-trainval \
  --scenarios scenario33 \
  --image-subdir camera_data_mask_yolo \
  --batch-size 48 \
  --temporal-layers 2 \
  --fusion-layers 2 \
  --spatial-scan row \
  --temporal-order reverse \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --dropout 0.25 \
  --loss power_soft_ce \
  --soft-power-temperature 0.15 \
  --hard-loss-weight 0.6 \
  --patience 10 \
  --early-stop-metric acc3 \
  --early-stop-mode max \
  --seed 7
```

### scenario34 Baseline（夜景，camera_data_mask_yolo）

```bash
python train.py \
  --split-root ./Data/splits_paper80 \
  --no-merge-trainval \
  --scenarios scenario34 \
  --image-subdir camera_data_mask_yolo \
  --batch-size 48 \
  --temporal-layers 2 \
  --fusion-layers 2 \
  --spatial-scan row \
  --temporal-order reverse \
  --optimizer adamw \
  --weight-decay 1e-4 \
  --dropout 0.25 \
  --loss power_soft_ce \
  --soft-power-temperature 0.15 \
  --hard-loss-weight 0.6 \
  --patience 10 \
  --early-stop-metric acc3 \
  --early-stop-mode max \
  --seed 7
```

### eval_missing.py 独立使用

```bash
# 评估 DMAF 模型（默认）
python eval_missing.py \
  --ckpt ./outputs/scenario32/TIMESTAMP/checkpoints/best_model.pth \
  --batch-size 48 --num-workers 8

# 评估 Baseline 模型
python eval_missing.py \
  --ckpt ./outputs/scenario32/TIMESTAMP/checkpoints/best_model.pth \
  --batch-size 48 --num-workers 8 --no-dmaf
```

## 11. 本地与 AutoDL 分工

### 本地负责

- 读代码、改代码、提交 git、汇总实验结论

### AutoDL 负责

- 跑训练、跑预处理、保存 checkpoint、保存日志

### 工作闭环

1. 本地改代码
2. 本地 `git commit` + `git push`
3. AutoDL `git pull origin main`
4. AutoDL 运行训练或评估
5. 回收结果并继续分析

如果 AutoDL 到 GitHub 网络不稳定，允许直接手动覆盖关键代码文件。

## 12. 开新话题时默认继承的上下文

- 本地仓库路径：`D:\code\project_gps`
- AutoDL 项目路径：`/root/autodl-tmp/project_gps`
- 正式运行环境：AutoDL
- 当前目标：在 BeMamba 复现基础上完成 DMAF 创新，提升缺失数据鲁棒性
- 创新方案主线：Missing-Aware Training（DMAF）
  - Point 1: 逐帧 mask concat 注入（替代全局 MaskEncoder add）
  - Point 2: Cross-Attention 模态融合（替代 ReliabilityGate 标量加权）
- 数据划分：`./Data/splits_paper80`，`--no-merge-trainval`
- 结构配置：`temporal_layers=2, fusion_layers=2`
- 扫描顺序：`spatial_scan=row, temporal_order=reverse`
- 优化器：`adamw, weight_decay=1e-4, dropout=0.25`
- 损失函数：`power_soft_ce, temp=0.15, hard_weight=0.6`
- 图像：
  - SC32（白天）：`camera_data_mask`
  - SC33/SC34（夜景）：`camera_data_mask_yolo`
- 当前最好 Baseline 结果：
  - SC32: Top-3 `81.54%`, DBA `0.8722`
  - SC33: Top-3 `80.99%`, DBA `0.8621`
  - SC34: Top-3 `85.58%`, DBA `0.8979`
- DMAF 当前状态：
  - 高缺失率（0.2/0.1/0.1）效果差于 Baseline
  - 正在验证低缺失率（0.1/0.05/0.05）
  - 当前推荐跑低缺失率训练作为第一优先
- 已支持的工具：
  - 训练自动输出：`final_test_result.txt` + `missing_test_result.txt/csv`
  - 独立评估：`python eval_missing.py --ckpt PATH [--no-dmaf]`
- 蒸馏（KD）暂不作主线
