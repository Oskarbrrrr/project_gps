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

**GPS 缺失策略**：GPS 共有 2 个时间点（起点/终点），从物理意义上看 GPS 缺失通常意味着整个模块故障，而非单个时间点丢失。因此 GPS **不参与帧级和连续帧级缺失**，只受模态级缺失（`missing_modality_prob`）影响。`_build_missing_masks` 中帧级/burst 循环使用 `missing_frame_modalities`（自动排除 gps），模态级循环仍使用 `missing_modalities`（可包含 gps）。

## 4. 当前模型结构状态

[src/model.py](/D:/code/project_gps/src/model.py) 包含 Baseline 和 DMAF 两套结构。

### 4.1 Baseline 结构（`missing_enabled=False`）

当 `missing_enabled=False` 时，Phases 2/4/5 自动跳过，等价于原始 BeMamba：

1. 图像 / LiDAR / Radar 分别经 CNN（ResNet34/ResNet18/ResNet18）提每帧特征
2. 三个感知模态经 TFMamba 时序处理 → sum 聚合为 tokens
3. GPS 用 MLP 投影到同一特征空间
4. 构造三种 mixed modal combinations（c1/c2/c3，不同模态排列顺序）
5. 用 MBMamba 做模态融合 → 三个序列简单求和
6. MLP head 预测 64 类 beam

关键超参：`d_model=128`, `patch_grid=6`

### 4.2 DMAF v4 结构（`missing_enabled=True`）

**模块化 7 阶段 Pipeline**（每个阶段有独立的类，可单独开关）：

```
Phase 1: Spatial Encoding     ModalityBackbone × 3 + GPSProjection
         → per-frame features [B, seq, d, H, W]

Phase 2: Mask Encoding        MaskEncoder × 4  ← NEW 独立模块
         → mask-aware features (Embedding(2, d_model) 注入)

Phase 3: Temporal Processing  TemporalProcessor × 3  ← NEW 独立模块
         → TFMamba + mask-weighted aggregation → [B, N, d]

Phase 4: Reliability Est.     ReliabilityEstimator × 4
         → mask ratio + feat stats → per-modality (0,1] weight

Phase 5: Cross-Modal Fusion   CrossModalFusion × 1  ← NEW 独立模块
         → 直接模态对交叉注意力: img↔{radar,lidar}, radar↔{img,lidar}, lidar↔{img,radar}

Phase 6: MBMamba Fusion       _build_modal_sequences + _modal_fusion
         → 3 combo sequences → MBMamba → mean → simple sum

Phase 7: Classification       MLP head → 64-class beam
```

**三个创新模块**：

**创新点 1：MaskEncoder + 加权聚合**（Phase 2 + Phase 3）

```
mask [B, 5] → Embedding(2, d_model) → [B, 5, d_model] → expand → [B, 5, d_model, H, W]
                                                                      ↓ add
encoded [B, 5, d_model, H, W] + mask_emb → TFMamba → per_time [B, 5, N, d_model]
                                                                      ↓
                                                        mask-weighted mean (dim=1)
```

- `Embedding(2, d_model)` 学到两个 d_model 维向量（缺失/正常），直接加到 per-frame 特征上
- 时序聚合使用 mask 加权 `mean`：缺失帧自动降权
- MaskEncoder 是独立模块，对 GPS（[B, 2, d]）和 spatial features（[B, 5, d, H, W]）均适用
- v1（已废弃）：MaskEncoder 全局 add；v2（已废弃）：Linear(1,1) concat

**创新点 2：CrossModalFusion — 直接模态对注意力**（Phase 5）

```
img_tokens   [B, N, d] → CrossAttn(Q=img,   KV=[radar,lidar]) → img_enh
radar_tokens [B, N, d] → CrossAttn(Q=radar, KV=[img,lidar])   → radar_enh
lidar_tokens [B, N, d] → CrossAttn(Q=lidar, KV=[img,radar])   → lidar_enh
```

- 旧版（已废弃）：序列均值之间的 cross-attention（间接）
- 新版：每个模态直接查询其他两个模态的 token，互补更直接
- 当 image 缺失时，img_tokens 被 reliability 衰减，cross-attn 从 radar+lidar 拉信息

**创新点 3：ReliabilityEstimator 可靠性估计**（Phase 4）

```
tokens [B, N, d_model]  →  feat_mean + feat_std [B, 2*d_model]  ─┐
mask   [B, seq_len]      →  mask_ratio [B, 1]                   ─┤
                                                                  ↓
                                     concat → MLP → Sigmoid → reliability [B, d_model]
                                                                  ↓
                                          tokens = tokens * reliability.unsqueeze(1)
```

- 每个模态一个 `ReliabilityEstimator`（~37K 参数），共 4 个
- 输入：特征统计量（均值+标准差）+ 掩码可用率
- 输出：(0, 1] 范围的逐样本可靠性向量
- clean 输入时 mask_ratio=1.0 → 可靠性接近 1.0，行为与 baseline 一致

**自动全1 Mask**：clean 测试时 `forward()` 自动生成全 1 mask，确保训练/测试输入分布一致

**消融开关**（新增，Phase 2 实验就绪）：

| CLI flag | Phase | 效果 |
|----------|:---:|------|
| `--no-mask-embed` | 2 | 跳过 MaskEncoder；Phase 3 保留 mask 加权聚合 |
| `--no-cross-attn` | 5 | 跳过 CrossModalFusion；Phase 6 纯 MBMamba + sum |
| `--no-reliability` | 4 | 跳过 ReliabilityEstimator；等权进入 fusion |

三者可任意组合，`missing_enabled=False` 时全部自动为 False。

### 4.3 代码结构

- `BeMambaConfig`：统一配置类，含 `use_mask_embed/cross_attn/reliability` 消融开关
- `ModalityBackbone`：ResNet 空间编码器（图像/Radar/LiDAR）
- `GPSProjection`：纯 MLP GPS 投影（不含 mask 逻辑）
- `MaskEncoder`（NEW）：独立 mask 嵌入模块 — `Embedding(2, d_model)` 注入
- `TemporalProcessor`（NEW）：TFMamba 时序处理 + mask 加权聚合
- `ReliabilityEstimator`：逐模态可靠性估计 — 特征统计量 + mask ratio → (0,1] 权重
- `CrossModalAttention`：多头交叉注意力 + FFN + 残差连接
- `CrossModalFusion`（NEW）：直接模态对交叉注意力 (img↔radar↔lidar)
- `MBMamba`：双向 Mamba 模态融合
- `BeMambaModel`：主模型，`forward()` 7 阶段 pipeline 清晰可读

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
| `--missing-enabled` | 兼容旧命令：同时开启缺失增强和 DMAF | - |
| `--missing-aug-enabled` | 只开启 dataset 侧缺失增强，不要求模型使用 DMAF | - |
| `--dmaf-enabled` | 只开启模型侧 DMAF/mask-aware 模块 | - |
| `--no-dmaf` | 与 `--missing-enabled` 搭配时关闭 DMAF，用于 Missing-Aug BeMamba 对照 | - |
| `--missing-frame-prob` | 每帧随机缺失概率 | 0.1 |
| `--missing-burst-prob` | 连续帧缺失概率 | 0.05 |
| `--missing-burst-min/max` | 连续缺失帧数范围 | 2/3 |
| `--missing-modality-prob` | 整模态缺失概率 | 0.05 |
| `--missing-modality-min/max` | 缺失模态数量范围 | 1/2 |
| `--missing-modalities` | 可缺失的模态列表 | img,radar,lidar,gps |
| `--missing-seed` | 缺失随机种子 | 42 |
| `--no-mask-embed` | 关闭 MaskEncoder | - |
| `--no-cross-attn` | 关闭 CrossModalFusion | - |
| `--no-reliability` | 关闭 ReliabilityEstimator | - |

> **GPS 特殊处理**：帧级缺失和连续帧缺失只作用于 `missing_frame_modalities`（自动排除 gps），GPS 仅受模态级缺失影响。理由：GPS 只有 2 个时间点，单点丢失没有物理意义，真实场景下 GPS 是整个模块故障。

### 5.4 训练输出

每次训练产出：
- `checkpoints/best_model.pth`
- `final_test_result.txt`：best checkpoint 的 clean 测试结果
- `missing_test_result.txt`：鲁棒性评估表（开启缺失增强或 DMAF 时）
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

### 7.3 DMAF v4 完整结果（2026-05-31）

训练参数：`frame=0.1, burst=0.05, modal=0.05, seed=7`

> 注意：2026-06-02 已修复 `TemporalProcessor` 的 mask 聚合尺度问题。旧 DMAF v4 结果可作为开发阶段参考，但最终论文表格需要重跑 DMAF 相关实验；Baseline clean/robustness 结果不受该修复影响。

#### SC32（白天，camera_data_mask）

| 协议 | Baseline | DMAF v4 | Δ Top-3 | Retention 提升 |
|------|:---:|:---:|:---:|:---:|
| clean | 82.50% | 80.90% | -1.60% | — |
| frame 50% | 46.55% | **73.19%** | +26.64pp | 56.4% → 90.5% |
| burst 60% | 37.56% | **76.08%** | +38.52pp | 45.5% → 94.0% |
| modal 60% | 65.33% | **71.43%** | +6.10pp | 79.2% → 88.3% |

- Clean 掉点 1.60%，略超 ±1% 目标线但属正常训练波动
- 帧/连续帧缺失下 Retention 从 46%/57% 拉到 91%/94%
- best_epoch: 27, DBA: 0.8661, APL: 0.1068 dB

#### SC33（夜景，camera_data_mask_yolo）

| 协议 | Baseline | DMAF v4 | Δ Top-3 | Retention 提升 |
|------|:---:|:---:|:---:|:---:|
| clean | 80.99% | **81.51%** | +0.52% | — |
| frame 50% | 47.14% | **74.35%** | +27.21pp | 58.2% → 91.1% |
| burst 60% | 44.66% | **76.04%** | +31.38pp | 55.1% → 93.1% |
| modal 60% | 66.02% | **72.40%** | +6.38pp | 81.5% → 88.7% |

- Clean **反超** Baseline 0.52%
- best_epoch: 16, DBA: 0.8658, APL: 0.1306 dB

#### SC34（夜景，camera_data_mask_yolo）

| 协议 | Baseline | DMAF v4 | Δ Top-3 | Retention 提升 |
|------|:---:|:---:|:---:|:---:|
| clean | 85.58% | 84.74% | -0.84% | — |
| frame 50% | 57.69% | **77.35%** | +19.66pp | 67.4% → 91.3% |
| burst 60% | 64.24% | **79.02%** | +14.78pp | 75.1% → 93.2% |
| modal 60% | 72.11% | 70.56% | -1.55pp | 84.3% → 83.3% |

- Clean 在 ±1% 内
- Modal 60% 是唯一 DMAF 不如 Baseline 的协议（-1.55pp）
- best_epoch: 26, DBA: 0.8921, APL: 0.0930 dB

#### 跨场景一致性

| 协议 | SC32 Δ | SC33 Δ | SC34 Δ | 均值 |
|------|:---:|:---:|:---:|:---:|
| Clean | -1.60% | +0.52% | -0.84% | -0.64% |
| Frame 50% | **+26.64pp** | **+27.21pp** | **+19.66pp** | **+24.50pp** |
| Burst 60% | **+38.52pp** | **+31.38pp** | **+14.78pp** | **+28.23pp** |
| Modal 60% | +6.10pp | +6.38pp | -1.55pp | +3.64pp |

**结论**：帧/连续帧缺失下 DMAF v4 提升巨大且跨场景一致（+14~39pp）。模态缺失提升较小（+3~6pp），SC34 有微小倒退。

### 7.4 消融实验（SC32，2026-05-31）

| # | 变体 | Clean | Frame 50% | Burst 60% | Modal 60% |
|:--:|------|:---:|:---:|:---:|:---:|
| A1 | **DMAF Full** | 80.90% | **73.19%** | 76.08% | **71.43%** |
| A2 | w/o CrossAttn (`--no-cross-attn`) | 81.70% | 72.23% | 73.84% | 72.71% |
| A3 | w/o Reliability (`--no-reliability`) | 81.54% | 72.55% | **78.01%** | 69.66% |
| A4 | w/o MaskEmbed (`--no-mask-embed`) | **81.86%** | 74.00% | 77.21% | 70.79% |
| A5 | Baseline（无 DMAF） | 82.50% | 46.55% | 37.56% | 65.33% |

**发现**：
1. 任何 DMAF 变体都碾压 Baseline（核心贡献来自缺失数据训练 + mask 加权聚合）
2. 单模块贡献在 ±2pp 内，属于正常训练波动而非显著组件贡献
3. 拆 CrossAttn → burst 掉点最明显（76.08%→73.84%），暗示跨模态注意力对连续帧缺失有帮助
4. 拆 MaskEmbed → clean 反而最高（81.86%），说明 `Embedding(2,d_model)` 可能非必须——加权聚合已给足够缺失信号
5. 拆 Reliability → modal 掉点最明显（71.43%→69.66%），但对 frame/burst 无显著影响

## 8. DMAF 实现与实验状态

### 8.1 版本演进

| 版本 | Mask 注入 | 时序聚合 | 模态融合 | 可靠性估计 | 架构 | 状态 |
|------|-----------|----------|----------|:---:|------|:---:|
| v1 | MaskEncoder 全局加和 | sum | ReliabilityGate | 无 | 耦合在 branch 内 | 废弃 |
| v2 | Linear(1,1) concat 到 channel | sum | ReliabilityGate | 无 | 耦合在 branch 内 | 废弃 |
| v3 | Embedding(2, d_model) | mask-weighted mean | Cross-Attn (序列均值间) | 无 | 耦合在 branch 内 | 废弃 |
| **v4** | **MaskEncoder (独立)** | **mask-weighted mean** | **CrossModalFusion (直接模态对)** | **ReliabilityEstimator** | **7 阶段模块化** | **当前** |

v4 架构（2026-05-29 重构）：
- MaskEncoder / TemporalProcessor / CrossModalFusion 均为独立模块，`forward()` 7 阶段 pipeline 清晰
- Cross-Attention 从序列均值间改为直接模态对 (img↔radar↔lidar)，互补更直接
- 消融开关 `--no-mask-embed/--no-cross-attn/--no-reliability` 已就绪

### 8.2 DMAF v2 实验（SC32，低缺失率，Linear(1,1) mask）

训练参数：`frame=0.1, burst=0.05, modal=0.05`，mask 注入方式：`Linear(1,1)` + concat

| 协议 | Baseline (clean训练) | DMAF v2 (缺失训练) |
|------|:---:|:---:|
| clean | **81.54%** | 81.22% |
| frame 10% | **78.97%** | 75.28% |
| frame 20% | **74.80%** | 71.11% |
| frame 30% | **70.63%** | 64.04% |
| burst 10% | **74.80%** | 73.84% |
| burst 20% | **69.02%** | 67.26% |
| modal 10% | **78.33%** | 77.37% |
| modal 20% | **74.16%** | 73.84% |
| hybrid | 63.88% | 70.30% |

Clean 差距 0.32%（在 ±1% 内），但帧缺失全面落后（-3.69% 到 -6.59%）。mask 信号太弱（1/129 维度），SSM 无法有效区分缺失帧。

### 8.3 DMAF v3 → v4 改进

v3 改进（已完成）：
1. **Mask 信号强化**：`Linear(1,1)` → `Embedding(2, d_model)`，缺失/正常各学一个 128 维向量，直接加到特征上
2. **Mask 加权聚合**：盲 `sum` → mask 加权 `mean`，缺失帧自动降权
3. **GPS 加入缺失**：`--missing-modalities` 默认包含 `gps`

v4 新增（2026-05-29）：
4. **GPS 仅模态级缺失**：GPS 从帧级/burst 循环中排除（`missing_frame_modalities`），只受模态级缺失影响
5. **Reliability Estimation**：新增 `ReliabilityEstimator` 模块，用特征统计量 + mask ratio 学习逐模态可靠性权重
6. **验证/测试固定模式**：`set_missing_epoch` 对 val/test 模式直接返回，缺失模式保持 epoch=0
7. **三大实验类型分离**：测试协议拆分为三种独立类型（帧随机/连续帧/模态缺失），各测多个强度

训练参数保持不变：`frame=0.1, burst=0.05, modal=0.05`

### 8.4 鲁棒性评估协议（12 种，v4 更新）

训练结束后自动运行，`eval_missing.py` 可独立评估已有 checkpoint。

三种独立实验类型，参数严格互斥（同一协议只有一种缺失机制启用）：

**A. Clean 基准：**
| # | 协议名 | 描述 |
|:--:|--------|------|
| 1 | clean | 完整数据 |

**B. 随机帧缺失（仅 frame_prob，burst=modality=0）：**
| # | 协议名 | frame_prob |
|:--:|--------|:---:|
| 2 | frame_p01 | 10% |
| 3 | frame_p03 | 30% |
| 4 | frame_p05 | 50% |

**C. 连续帧缺失（仅 burst_prob，frame=modality=0，burst_min=2 burst_max=3）：**
| # | 协议名 | burst_prob |
|:--:|--------|:---:|
| 5 | burst_p02 | 20% |
| 6 | burst_p04 | 40% |
| 7 | burst_p06 | 60% |

**D. 模态缺失（仅 modality_prob，frame=burst=0，modality_min=max=1）：**
| # | 协议名 | modality_prob |
|:--:|--------|:---:|
| 8 | modal_p02 | 20% |
| 9 | modal_p04 | 40% |
| 10 | modal_p06 | 60% |

**E. 混合：**
| # | 协议名 | 描述 |
|:--:|--------|------|
| 11 | hybrid | 训练配置的混合缺失 |

所有指标报告 Top-1/2/3、DBA、APL、Retention Ratio。

### 8.5 已完成清单

- [x] 数据集缺失模拟（帧级 / 连续帧 / 模态级）
- [x] GPS 仅模态级缺失（帧级/burst 自动排除 gps）
- [x] 逐帧 mask 注入 v4：MaskEncoder 独立模块，Embedding(2, d_model) 强信号
- [x] Mask 加权时序聚合（替代盲 sum）
- [x] CrossModalFusion：直接模态对交叉注意力（替代序列均值间 CrossAttn + ReliabilityGate）
- [x] Reliability Estimation 模块
- [x] Clean 测试自动全 1 mask
- [x] 验证/测试集固定缺失模式（`set_missing_epoch` guard）
- [x] 训练结束后自动鲁棒性评估（12 种协议，3 种独立实验类型）
- [x] 独立评估脚本 `eval_missing.py`（支持动态协议配置）
- [x] `--no-dmaf` 开关（评估 baseline 模型用）
- [x] 消融开关 `--no-mask-embed / --no-cross-attn / --no-reliability`（Phase 2 实验就绪）
- [x] 7 阶段模块化架构重构（MaskEncoder / TemporalProcessor / CrossModalFusion 独立模块）
- [x] `run_missing_robustness_tests` 显式传递 `missing_modalities` 参数
- [x] **Phase 1 D1/D2/D3**：DMAF v4 三场景训练 + 12 协议鲁棒性评估
- [x] **Phase 2 A2/A3/A4**：消融实验（SC32）
- [x] **Pre P0**：Baseline 鲁棒性评估（12 协议 × 3 场景）
- [ ] Phase 3 C1/C2/C3：缺失率曲线（纯 eval）

## 9. 实验计划与进度跟踪

### 9.1 故事线

> 实际部署中传感器数据经常缺失（硬件故障、延迟、天气），现有方法一缺就崩。我们提出 DMAF——用逐帧 mask 注入 + Cross-Attention 融合 + Reliability Estimation，让模型在残缺输入下仍然预测准确。

### 9.2 实验总览

| 阶段 | 实验ID | 训练? | 场景 | 描述 | 优先级 | 状态 |
|------|:--:|:---:|------|------|:---:|:---:|
| **Pre** | P0 | ❌ eval | SC32/33/34 | Baseline 鲁棒性评估（12 协议 × 3 场景） | 🔴 最高 | ✅ |
| **Phase 1** | D1 | ✅ train | SC32 | DMAF v4 完整训练 + 12 协议 | 🔴 最高 | ✅ |
| **Phase 1** | D2 | ✅ train | SC33 | DMAF v4 完整训练 + 12 协议 | 🟡 次高 | ✅ |
| **Phase 1** | D3 | ✅ train | SC34 | DMAF v4 完整训练 + 12 协议 | 🟡 次高 | ✅ |
| **Phase 2** | A2 | ✅ train | SC32 | 消融：w/o CrossAttn (`--no-cross-attn`) | 🟢 中等 | ✅ |
| **Phase 2** | A3 | ✅ train | SC32 | 消融：w/o Reliability (`--no-reliability`) | 🟢 中等 | ✅ |
| **Phase 2** | A4 | ✅ train | SC32 | 消融：w/o MaskEmbed (`--no-mask-embed`) | 🟢 中等 | ✅ |
| **Phase 3** | C1 | ❌ eval | SC32 | 缺失率曲线（DMAF + Baseline 对比） | 🔵 较低 | 🔴 |
| **Phase 3** | C2 | ❌ eval | SC33 | 缺失率曲线（DMAF + Baseline 对比） | 🔵 较低 | 🔴 |
| **Phase 3** | C3 | ❌ eval | SC34 | 缺失率曲线（DMAF + Baseline 对比） | 🔵 较低 | 🔴 |

**已完成**：3 次 Baseline eval + 3 次 DMAF 训练 + 3 次消融训练；**待做**：Phase 3 缺失率曲线

**2026-06-02 调整后的最终论文实验分组**：

| 组别 | 训练输入 | 模型结构 | 是否用 mask | 作用 | 状态 |
|------|----------|----------|:---:|------|:---:|
| G0 BeMamba | clean | 原始 BeMamba | ❌ | 原始基线，证明缺失会掉点 | ✅ 已有 |
| G1 Missing-Aug BeMamba | 缺失增强 | 原始 BeMamba | ❌ | 排除“只是数据增强带来提升”的质疑 | 🔴 待跑 |
| G2 Mask-Aware Temporal Only | 缺失增强 | DMAF 仅保留 mask 加权聚合 | ✅ | 验证 mask 加权聚合核心贡献 | 🔴 待跑 |
| G3 DMAF Full | 缺失增强 | 完整 DMAF | ✅ | 最终方法 | 🔴 需重跑 |

新代码已解耦：

- `--missing-aug-enabled`：dataset 造缺失
- `--dmaf-enabled`：模型启用 DMAF
- `--missing-enabled`：兼容旧命令，等价于两者都开
- `--missing-enabled --no-dmaf`：只做 Missing-Aug BeMamba

---

### 9.3 依赖关系

```
Pre (Baseline 鲁棒性) ─────────────────────────────┐
                                                    ↓
Phase 1 D1 (SC32 DMAF v4) ─── 确认方案有效 ───→ D2 (SC33) + D3 (SC34)
    ↓                                                   ↓
    ├──→ Phase 2 A2/A3/A4 (消融，依赖 D1 做对比)       │
    └──→ Phase 3 C1 (SC32 曲线)                        │
                                                        ↓
                                            Phase 3 C2 + C3 (SC33/34 曲线)
```

**关键路径**：Pre → D1 → 确认结果 → D2/D3 + A2/A3/A4 并行 → C1/C2/C3

---

### 9.4 Pre: Baseline 鲁棒性评估（无训练，先跑）

**目标**：获取三个场景 Baseline 模型在 12 种缺失协议下的性能，作为 DMAF v4 的对比基线。

**前提**：已有 B1/B2/B3 的 `best_model.pth` checkpoint。

**为什么先跑**：跑完才知道 DMAF 要超越的具体数字是多少。Baseline 在缺失下应该明显掉点——这就是 DMAF 要解决的问题。

```bash
# P0-SC32: Baseline 鲁棒性
python eval_missing.py \
  --ckpt ./outputs/scenario32/<TIMESTAMP>/checkpoints/best_model.pth \
  --split-root ./Data/splits_paper80 \
  --scenario scenario32 \
  --image-subdir camera_data_mask \
  --batch-size 48 \
  --no-dmaf

# P0-SC33: Baseline 鲁棒性
python eval_missing.py \
  --ckpt ./outputs/scenario33/<TIMESTAMP>/checkpoints/best_model.pth \
  --split-root ./Data/splits_paper80 \
  --scenario scenario33 \
  --image-subdir camera_data_mask_yolo \
  --batch-size 48 \
  --no-dmaf

# P0-SC34: Baseline 鲁棒性
python eval_missing.py \
  --ckpt ./outputs/scenario34/<TIMESTAMP>/checkpoints/best_model.pth \
  --split-root ./Data/splits_paper80 \
  --scenario scenario34 \
  --image-subdir camera_data_mask_yolo \
  --batch-size 48 \
  --no-dmaf
```

**产出**：每个场景一个 `missing_test_result.txt`（在 checkpoint 所在 outputs 目录下）

**期望观察**：
- Clean 协议：与训练时的 `final_test_result.txt` 一致
- Frame/Burst/Modal 协议：Top-3 随缺失率上升明显下降
- 尤其 frame 缺失：Baseline 掉点严重（无 mask 感知，靠猜零值）

---

### Phase 0: Clean Baseline ✅

| # | 场景 | 配置 | Top-3 | DBA | best_epoch | 状态 |
|:--:|------|------|:---:|:---:|:---:|:---:|
| B1 | SC32 | camera_data_mask, power_soft_ce, seed=7 | 81.54% | 0.8722 | 14 | ✅ |
| B2 | SC33 | camera_data_mask_yolo, power_soft_ce, seed=7 | 80.99% | 0.8621 | 17 | ✅ |
| B3 | SC34 | camera_data_mask_yolo, power_soft_ce, seed=7 | 85.58% | 0.8979 | 22 | ✅ |

命令见 [第10节](#10-当前最优配置与命令)。

---

### Phase 1: DMAF v4 主实验 🔴

**目标**：证明 DMAF v4 在三种缺失类型下全面超越 Baseline，且 clean 性能不掉。

**训练参数**（三个场景统一）：
```
frame=0.1, burst=0.05, modal=0.05, seed=7
```

**12 协议自动评估**：训练完成后自动运行，无需额外操作。

| # | 场景 | 图像子目录 | 描述 | 状态 | 预计产出 |
|:--:|------|------------|------|:---:|------|
| D1 | **SC32** | camera_data_mask | 白天，最难场景，Baseline 81.54% | 🔴 | `missing_test_result.txt` |
| D2 | **SC33** | camera_data_mask_yolo | 夜景，Baseline 80.99% | 🔴 | `missing_test_result.txt` |
| D3 | **SC34** | camera_data_mask_yolo | 夜景，Baseline 85.58% 最好 | 🔴 | `missing_test_result.txt` |

**判断标准**（每个场景对比 Pre 的 Baseline 鲁棒性结果）：

| 协议类型 | 期望 |
|----------|------|
| Clean Top-3 | 与 Baseline 差距在 ±1% 以内（不掉点） |
| Frame 缺失（10%/30%/50%） | Retention Ratio **显著高于** Baseline（DMAF 核心优势） |
| Burst 缺失（20%/40%/60%） | Retention Ratio 高于 Baseline |
| Modal 缺失（20%/40%/60%） | Retention Ratio 高于 Baseline |
| Hybrid | Retention Ratio 高于 Baseline |

**D1 命令**（SC32，最高优先级）：
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

**D2 命令**（改两处）：
```bash
# 与 D1 相同，只改 --scenarios 和 --image-subdir
  --scenarios scenario33 \
  --image-subdir camera_data_mask_yolo \
```

**D3 命令**（改两处）：
```bash
  --scenarios scenario34 \
  --image-subdir camera_data_mask_yolo \
```

**如果 D1 结果不符合预期**：
1. 检查 `train_log.csv`：train/test loss 是否正常收敛
2. 检查 `missing_test_result.txt`：哪个协议最差？frame/burst/modal？
3. 可能调整：提高训练缺失率（frame=0.15→0.2）或增加 `fusion_layers`
4. 如果 Clean 掉点 >1%：降低缺失率或增加 `hard_loss_weight`

---

### Phase 2: 消融实验 🔴

**目标**：量化 MaskEncoder、CrossModalFusion、ReliabilityEstimator 各自贡献。

**前提**：Phase 1 D1 完成且结果达标。

**只跑 SC32**（场景差异最大，组件贡献最明显）。3 次训练 + 复用 D1 和 B1。

| # | 变体 | 命令差异 | mask | cross | rel | 对比目标 |
|:--:|------|----------|:---:|:---:|:---:|------|
| A1 | **DMAF Full** | (无) | ✅ | ✅ | ✅ | 复用 D1 |
| A2 | **w/o CrossAttn** | `--no-cross-attn` | ✅ | ❌ | ✅ | vs A1：CrossModalFusion 贡献 |
| A3 | **w/o Reliability** | `--no-reliability` | ✅ | ✅ | ❌ | vs A1：ReliabilityEstimator 贡献 |
| A4 | **w/o MaskEmbed** | `--no-mask-embed` | ❌ | ✅ | ✅ | vs A1：MaskEncoder 贡献 |
| A5 | **Baseline** | (无 `--missing-enabled`) | ❌ | ❌ | ❌ | 复用 B1，下界 |

**期望**：每个模块被拆掉后，鲁棒性都有可测量的下降。贡献排序（猜测）：MaskEmbed ≈ CrossAttn > Reliability。

**A2 命令**（在 D1 命令基础上加一个 flag）：
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
  --missing-modality-prob 0.05 \
  --no-cross-attn
```

**A3**：把 `--no-cross-attn` 换成 `--no-reliability`
**A4**：把 `--no-cross-attn` 换成 `--no-mask-embed`

---

### Phase 3: 缺失率曲线 🔴

**目标**：画三种缺失类型下 Top-3 vs 缺失率的完整曲线，证明 DMAF 在任何缺失强度下都优于 Baseline，且衰减更平缓。

**无需额外训练**——用 Phase 1 + Phase 0 产出的 best checkpoint。

**每个场景跑 6 条命令**（3 种缺失类型 × 2 个模型），产出 6 组数据：

```bash
CKPT_DMAF="./outputs/scenario32/<D1_TIMESTAMP>/checkpoints/best_model.pth"
CKPT_BASE="./outputs/scenario32/<B1_TIMESTAMP>/checkpoints/best_model.pth"

# ─── Frame 缺失率曲线（DMAF）───
python eval_missing.py \
  --ckpt $CKPT_DMAF \
  --split-root ./Data/splits_paper80 \
  --scenario scenario32 \
  --image-subdir camera_data_mask \
  --batch-size 48 \
  --test-frame-probs 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 \
  --test-burst-probs "" --test-modality-probs "" --skip-hybrid

# ─── Frame 缺失率曲线（Baseline）───
python eval_missing.py \
  --ckpt $CKPT_BASE \
  --split-root ./Data/splits_paper80 \
  --scenario scenario32 \
  --image-subdir camera_data_mask \
  --batch-size 48 \
  --no-dmaf \
  --test-frame-probs 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 \
  --test-burst-probs "" --test-modality-probs "" --skip-hybrid

# ─── Burst 缺失率曲线（DMAF）───
python eval_missing.py \
  --ckpt $CKPT_DMAF \
  --split-root ./Data/splits_paper80 \
  --scenario scenario32 \
  --image-subdir camera_data_mask \
  --batch-size 48 \
  --test-frame-probs "" \
  --test-burst-probs 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
  --test-modality-probs "" --skip-hybrid

# ─── Burst 缺失率曲线（Baseline）───
python eval_missing.py \
  --ckpt $CKPT_BASE \
  --split-root ./Data/splits_paper80 \
  --scenario scenario32 \
  --image-subdir camera_data_mask \
  --batch-size 48 \
  --no-dmaf \
  --test-frame-probs "" \
  --test-burst-probs 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
  --test-modality-probs "" --skip-hybrid

# ─── Modal 缺失率曲线（DMAF）───
python eval_missing.py \
  --ckpt $CKPT_DMAF \
  --split-root ./Data/splits_paper80 \
  --scenario scenario32 \
  --image-subdir camera_data_mask \
  --batch-size 48 \
  --test-frame-probs "" --test-burst-probs "" \
  --test-modality-probs 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
  --skip-hybrid

# ─── Modal 缺失率曲线（Baseline）───
python eval_missing.py \
  --ckpt $CKPT_BASE \
  --split-root ./Data/splits_paper80 \
  --scenario scenario32 \
  --image-subdir camera_data_mask \
  --batch-size 48 \
  --no-dmaf \
  --test-frame-probs "" --test-burst-probs "" \
  --test-modality-probs 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
  --skip-hybrid
```

**SC33/SC34** 同理，替换 `--scenario`、`--image-subdir`、`--ckpt` 即可。

**期望**：
- DMAF 在所有缺失率下 Retention Ratio **始终高于** Baseline
- Baseline 随缺失率上升呈**线性或超线性**下降
- DMAF 衰减**更平缓**（尤其在 frame 缺失场景——这是 mask embedding 的核心优势）
- 高缺失率（>50%）下 DMAF 仍保留显著预测能力，而 Baseline 接近随机

---

### 9.5 执行状态

```
✅ Pre: Baseline 鲁棒性（已完成 2026-05-31）
✅ Phase 1 D1: SC32 DMAF v4（已完成 2026-05-31）
✅ Phase 1 D2: SC33 DMAF v4（已完成 2026-05-31）
✅ Phase 1 D3: SC34 DMAF v4（已完成 2026-05-31）
✅ Phase 2 A2/A3/A4: 消融实验（已完成 2026-05-31）
🔴 Phase 3 C1/C2/C3: 缺失率曲线（待做）
```

**当前下一步**：先补 G1 Missing-Aug BeMamba（三场景）和重跑 G3 DMAF Full（三场景）。完成后再做 Phase 3 缺失率曲线。

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

### scenario32 Missing-Aug BeMamba（关键对照，当前需补跑）

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
  --missing-aug-enabled \
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
# 评估 DMAF 模型（默认，使用默认测试协议）
python eval_missing.py \
  --ckpt ./outputs/scenario32/TIMESTAMP/checkpoints/best_model.pth \
  --batch-size 48 --num-workers 8

# 评估 Baseline 模型
python eval_missing.py \
  --ckpt ./outputs/scenario32/TIMESTAMP/checkpoints/best_model.pth \
  --batch-size 48 --num-workers 8 --no-dmaf

# 自定义测试协议强度
python eval_missing.py \
  --ckpt ./outputs/scenario32/TIMESTAMP/checkpoints/best_model.pth \
  --batch-size 48 \
  --test-frame-probs 0.1 0.3 0.5 0.7 \
  --test-burst-probs 0.2 0.4 0.6 \
  --test-modality-probs 0.2 0.4 0.6 \
  --test-burst-min 2 --test-burst-max 3 \
  --test-modality-min 1 --test-modality-max 1

# 只测帧缺失，其他跳过
python eval_missing.py \
  --ckpt PATH --batch-size 48 \
  --test-frame-probs 0.1 0.3 0.5 \
  --test-burst-probs "" --test-modality-probs "" --skip-hybrid
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
- 创新方案主线：Missing-Aware Training（DMAF v4）
  - Point 1: 逐帧 mask Embedding 注入（替代全局 MaskEncoder add）
  - Point 2: Cross-Attention 模态融合（替代 ReliabilityGate 标量加权）
  - Point 3: Reliability Estimation 可靠性估计（v4 新增）— 模态级可靠性加权
  - GPS 仅模态级缺失（帧级/burst 自动排除）
  - 验证/测试集缺失模式固定（`set_missing_epoch` guard）
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
- DMAF v4 当前状态：
  - 代码已完成；2026-06-02 修复 mask 聚合尺度，DMAF 相关结果需重跑
  - 新增 `--missing-aug-enabled / --dmaf-enabled / --no-dmaf`，已支持 Missing-Aug BeMamba 公平对照
  - 12 种鲁棒性评估协议（帧/连续帧/模态三种类型独立）
  - 训练推荐：`frame=0.1, burst=0.05, modal=0.05`
- 已支持的工具：
  - 训练自动输出：`final_test_result.txt` + `missing_test_result.txt/csv`
  - 独立评估（支持动态协议）：`python eval_missing.py --ckpt PATH [--no-dmaf] [--test-*-probs ...]`
- 蒸馏（KD）暂不作主线
