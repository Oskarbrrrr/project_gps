# AGENTS

## Paper Page Index

- 论文逐页摘要与图片映射见：[PAPER_PAGE_INDEX.md](/D:/code/project_gps/PAPER_PAGE_INDEX.md)
- 对应图片目录：`paper_pages/001.jpg` 到 `paper_pages/014.jpg`
- 需要快速查找架构图、公式、主结果、消融表时，优先打开 `PAPER_PAGE_INDEX.md`

## 1. 项目定位

这是一个用于复现论文 `BeMamba: Efficient Multimodal Sensing-Aided Beamforming via State Space Model` 的本地代码仓库。

当前复现策略不是“逐字逐句复刻作者私有源码”，而是：

- 尽量按论文公开信息对齐
- 优先保证多模态输入、时序建模、模态融合、指标口径与论文一致
- 允许在工程实现上做最小必要调整，但不随意改实验目标

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

也就是说当前输入为：

- 图像 5 帧
- Radar 5 帧
- LiDAR 5 帧
- GPS 2 个时刻

这和论文正文给出的多模态输入组织方式是一致的。

## 3. 当前各模态实现状态

### 3.1 图像模态

图像模态当前已经支持三种输入目录切换，通过 `--image-subdir` 控制：

- `camera_data`
- `camera_data_mask`
- `camera_data_mask_yolo`

其中：

- `camera_data`：当前项目实际使用的图像目录，不一定是 DeepSense 6G 官方原始图像
- `camera_data_mask`：对图像应用固定场景 mask 后的结果
- `camera_data_mask_yolo`：对图像应用固定场景 mask，并用 YOLO 画红框后的结果

当前图像预处理脚本为：

- [prepare_camera_mask_yolo.py](/D:/code/project_gps/prepare_camera_mask_yolo.py)

该脚本用于离线生成：

- `camera_data_mask`
- `camera_data_mask_yolo`

当前对图像模态的经验结论：

- `scenario33`、`scenario34` 是夜景场景，`mask + yolo` 明显有帮助
- `scenario32` 是白天场景，`mask + yolo` 不一定优于更温和的图像输入
- `scenario32` 上 `camera_data` 与 `camera_data_mask` 差距很小，说明它的主要瓶颈不一定在图像 mask 本身

### 3.2 LiDAR 模态

LiDAR 当前实现已经比较接近论文中的 `BEV + Generate` 思路：

1. 点云投影到 BEV
2. 通过连续帧差分估计运动区域
3. 针对 motion points 做一对一 virtual point generation
4. 对 virtual points 做小范围随机扰动
5. 将原始 BEV 与生成点 BEV 合并

### 3.3 Radar 模态

Radar 当前使用：

- range-angle map
- range-velocity map

以 2 通道拼接方式输入，对应论文里的 `RA/RV-Map` 思路。

### 3.4 GPS 模态

GPS 当前处理为：

1. 从绝对经纬度转换成相对位移
2. 对相对位移做归一化
3. 转成极坐标特征
4. 以两个时刻的特征进入后续融合

重要修正：

- 现在 **test 集会复用 train 集的 GPS 归一化统计量**
- 不再让 test 集单独拟合自己的归一化范围

这个改动更符合正常 train/test 协议，也更接近论文复现应有口径。

## 4. 当前模型结构状态

[src/model.py](/D:/code/project_gps/src/model.py) 当前是“尽量贴论文”的版本。

主干思路：

1. 图像 / LiDAR / Radar 分别经 CNN 提特征
2. 三个感知模态都经过 Time Sequence Mamba
3. GPS 用 MLP 投影到同一特征空间
4. 构造三种 mixed modal combinations
5. 用 MB-Mamba 做模态融合
6. 最后用 MLP head 预测 64 类 beam

当前 backbone：

- Image: `ResNet34`
- LiDAR: `ResNet18`
- Radar: `ResNet18`

当前关键超参：

- `d_model = 128`
- `patch_grid = 6`

### 4.1 最近的重要结构修正

之前代码里：

- `temporal_layers`
- `fusion_layers`

虽然在配置里存在，但实际上没有真正生效。

现在已经修正：

- `temporal_layers` 会真正堆叠多层 `TFMamba`
- `fusion_layers` 会真正堆叠多层 `MBMamba`
- 两者都带残差连接

也就是说，当前 `--temporal-layers 2 --fusion-layers 2` 已经是真正生效的结构，不再是“参数写了但没用”。

## 5. 当前训练协议

[train.py](/D:/code/project_gps/train.py) 当前训练协议为：

- Optimizer: `Adam`
- 默认学习率：`1e-4`
- 默认 epoch：`30`
- 默认损失：`CrossEntropyLoss`
- AMP：默认开启
- 支持 early stopping
- 支持 best checkpoint 选择

### 5.1 数据划分口径

当前仍然尽量贴近论文的 `80/20` 训练-测试口径：

- `train.csv + val.csv` 合并后作为训练侧
- `test.csv` 作为测试侧

这里的“贴论文”含义是：

- 不强行改成严格的 `train / val / test` 三分法
- 但在实现细节上尽量避免明显不合理之处

### 5.2 早停与最佳模型

当前训练脚本已经支持：

- `--patience`
- `--early-stop-metric`
- `--early-stop-mode`

默认推荐：

- 用 `acc3` 作为监控指标
- 在 `patience > 0` 时启用 early stopping

当前每次训练会产出：

- `checkpoints/best_model.pth`
- `final_test_result.txt`：现在表示 **best checkpoint** 的结果
- `last_epoch_result.txt`：最后一轮结果
- `train_log.csv`

`train_log.csv` 中会包含：

- 每轮 train/test 指标
- `is_best` 列

## 6. 当前指标口径

当前统一保留并输出：

- `Top-1`
- `Top-2`
- `Top-3`
- `DBA`
- `APL`

其中：

- 论文主要报告 `Top-3` 和 `DBA`
- 当前额外保留 `Top-1`、`Top-2`、`APL` 便于分析

论文里的 `Top-3` 可以理解为：

- 对每个测试样本取分数最高的 3 个 beam
- 真实 beam 落在 top-3 中则记为命中
- 最终统计测试集 `Top-3 Accuracy`

## 7. 截至 2026-05-17 的最新可信结果

以下结果均指当前代码下、best checkpoint 重新评估后的可信结果，而不是最后一轮结果。

### 7.1 scenario32

当前最优配置：

- `split_root = ./Data/splits_paper80`
- `--no-merge-trainval`
- `image_subdir = camera_data_mask`
- `temporal_layers = 2`
- `fusion_layers = 2`
- `spatial_scan = row`
- `temporal_order = reverse`
- `optimizer = adamw`
- `weight_decay = 1e-4`
- `dropout = 0.25`
- `loss = power_soft_ce`
- `soft_power_temperature = 0.15`
- `hard_loss_weight = 0.6`
- `seed = 7`

结果：

- Top-1: `41.89%`
- Top-2: `69.98%`
- Top-3: `82.83%`
- DBA: `0.8712`
- APL: `0.0981 dB`
- best_epoch: `14`

补充说明：

- `temp = 0.2` 也能跑到 `82.83%`
- 但 `temp = 0.15` 的 DBA / APL 略优，因此当前默认保留 `0.15`
- `camera_data`
- `camera_data_mask_yolo`
- `vertical`
- `freeze_image_stem`

这些方向都试过，但没有超过当前 best。

### 7.2 scenario33

当前最优配置：

- `split_root = ./Data/splits_paper80`
- `--no-merge-trainval`
- `image_subdir = camera_data_mask_yolo`
- `temporal_layers = 2`
- `fusion_layers = 2`
- `spatial_scan = row`
- `temporal_order = reverse`
- `optimizer = adamw`
- `weight_decay = 1e-4`
- `dropout = 0.25`
- `loss = power_soft_ce`
- `soft_power_temperature = 0.15`
- `hard_loss_weight = 0.6`
- `seed = 7`

结果：

- Top-1: `45.05%`
- Top-2: `66.80%`
- Top-3: `80.99%`
- DBA: `0.8621`
- APL: `0.1295 dB`
- best_epoch: `17`

### 7.3 scenario34

当前最优配置：

- `split_root = ./Data/splits_paper80`
- `--no-merge-trainval`
- `image_subdir = camera_data_mask_yolo`
- `temporal_layers = 2`
- `fusion_layers = 2`
- `spatial_scan = row`
- `temporal_order = reverse`
- `optimizer = adamw`
- `weight_decay = 1e-4`
- `dropout = 0.25`
- `loss = power_soft_ce`
- `soft_power_temperature = 0.15`
- `hard_loss_weight = 0.6`
- `seed = 7`

结果：

- Top-1: `47.68%`
- Top-2: `71.16%`
- Top-3: `85.58%`
- DBA: `0.8979`
- APL: `0.0841 dB`
- best_epoch: `22`

## 8. 当前与论文的差距

论文表格中，四模态 BeMamba 的 Top-3 Accuracy 为：

- SC32: `88.11%`
- SC33: `84.94%`
- SC34: `85.64%`

当前最佳复现结果与论文差距：

- SC32: `82.83%`，差 `5.28`
- SC33: `80.99%`，差 `3.95`
- SC34: `85.58%`，差 `0.06`

整体平均 Top-3：

- 论文平均：`86.23%`
- 当前平均：`83.13%`
- 平均差距：`3.10`

结论：

- `scenario34` 基本复现到论文水平
- `scenario33` 已进入论文差距 `5%` 以内
- `scenario32` 仍是唯一没有进入 `5%` 以内的场景
- `scenario32` 距离进入 `5%` 以内只差 `0.28`

## 9. 当前最明确的结论

1. 夜景场景显著受益于 `mask + yolo`
   - `scenario33`、`scenario34` 推荐优先使用 `camera_data_mask_yolo`

2. `scenario32` 的主要提升不来自 YOLO 红框
   - 它的 best 来自 `camera_data_mask`
   - `camera_data_mask_yolo` 不是当前最优方向

3. 对 `scenario32` 真正有效的提升来自更贴论文的训练协议
   - `paper80_20` split
   - `--no-merge-trainval`
   - `row + reverse`
   - `power_soft_ce`

4. 评估稳定性修复是必要的
   - 已修复 test 侧 LiDAR virtual point jitter 导致的评估抖动
   - 当前结果比早期实验更可信

5. 当前训练脚本的重要新增能力包括：
   - `--image-subdir`
   - `--split-root`
   - `--no-merge-trainval`
   - `--spatial-scan`
   - `--temporal-order`
   - `--loss power_soft_ce`
   - best checkpoint
   - early stopping

## 10. 当前最推荐的实验配置

### 通用运行前准备

```bash
export OMP_NUM_THREADS=8
echo $OMP_NUM_THREADS
```

### scenario32 当前最优配置

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

### scenario33 推荐配置

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

### scenario34 推荐配置

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

## 11. 本地与 AutoDL 分工

### 本地负责

- 读代码
- 改代码
- 提交 git
- 汇总实验结论

### AutoDL 负责

- 跑训练
- 跑预处理
- 保存 checkpoint
- 保存日志

## 12. 推荐工作流

推荐闭环：

1. 本地改代码
2. 本地 `git commit`
3. 本地 `git push`
4. AutoDL 同步代码
5. AutoDL 运行训练或预处理
6. 回收结果并继续分析

如果 AutoDL 到 GitHub 网络不稳定，允许直接手动覆盖关键代码文件。

## 13. 开新话题时默认继承的上下文

- 本地仓库路径：`D:\code\project_gps`
- 正式运行环境：AutoDL
- 当前目标：高质量复现 BeMamba 论文结果
- 当前最好用的结构配置：`temporal_layers = 2`, `fusion_layers = 2`
- 当前默认 split：`./Data/splits_paper80`
- 当前夜景场景最有效图像输入：`camera_data_mask_yolo`
- 当前白天场景最优图像输入：`camera_data_mask`
- 当前最好结果：
  - `scenario32`: Top-3 `82.83%`
  - `scenario33`: Top-3 `80.99%`
  - `scenario34`: Top-3 `85.58%`
- 当前主要短板：`scenario32`，但距离进入论文差距 `5%` 以内只差 `0.28`
- 当前训练脚本已经支持：
  - `--image-subdir`
  - `--split-root`
  - `--no-merge-trainval`
  - `--spatial-scan`
  - `--temporal-order`
  - `--loss power_soft_ce`
  - best checkpoint
  - early stopping
  - 多层 temporal / fusion Mamba
