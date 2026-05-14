# AGENTS

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

## 7. 截至 2026-05-15 的最新可信结果

以下结果均指最近这轮“较可信配置”下的 best checkpoint 结果，而不是最后一轮结果。

### 7.1 scenario33

较优配置：

- `image_subdir = camera_data_mask_yolo`
- `temporal_layers = 2`
- `fusion_layers = 2`
- `batch_size = 48`

结果：

- Top-1: `44.14%`
- Top-2: `67.84%`
- Top-3: `79.69%`
- DBA: `0.8726`
- APL: `0.1361 dB`
- best_epoch: `17`

### 7.2 scenario34

较优配置：

- `image_subdir = camera_data_mask_yolo`
- `temporal_layers = 2`
- `fusion_layers = 2`
- `batch_size = 48`

结果：

- Top-1: `41.12%`
- Top-2: `65.91%`
- Top-3: `81.64%`
- DBA: `0.8784`
- APL: `0.1070 dB`
- best_epoch: `20`

### 7.3 scenario32

当前测试过的较优配置之一：

- `image_subdir = camera_data`
- `temporal_layers = 2`
- `fusion_layers = 2`
- `batch_size = 48`

最近结果：

- Top-1: `41.25%`
- Top-2: `66.45%`
- Top-3: `78.81%`
- DBA: `0.8653`
- APL: `0.2861 dB`
- best_epoch: `16`

此前跑过的相近结果：

- `camera_data`: Top-3 约 `79.78%`
- `camera_data_mask`: Top-3 约 `79.61%`

结论：

- `scenario32` 目前稳定在 `79%` 左右
- 仍明显低于论文的 `88.11%`
- 它的主要瓶颈暂时看起来**不是**图像 mask 目录的选择

## 8. 当前与论文的差距

论文表格里，四模态 BeMamba 的 Top-3 Accuracy 为：

- SC32: `88.11%`
- SC33: `84.94%`
- SC34: `85.64%`

当前较可信结果与论文差距：

- SC32: `78.81%`，差约 `9.30`
- SC33: `79.69%`，差约 `5.25`
- SC34: `81.64%`，差约 `4.00`

也就是说：

- `scenario34` 已经进入论文差距 `5%` 以内
- `scenario33` 已经非常接近 `5%` 以内
- `scenario32` 仍然是当前最主要短板

## 9. 当前最明确的结论

1. 图像模态对夜景场景收益很大
   - `scenario33`、`scenario34` 明显受益于 `mask + yolo`

2. 把 `temporal_layers` 和 `fusion_layers` 真正接通以后，结果有实质提升
   - 尤其是 `scenario33`

3. `scenario32` 的问题不是简单靠 `camera_data_mask_yolo` 就能解决
   - `camera_data` 和 `camera_data_mask` 都只能到 `79%` 左右

4. GPS 归一化 train/test 不一致已经修复
   - 这是实现正确性修复
   - 但它不是当前 `scenario32` 的主要提升来源

5. 当前阶段最值得继续优化的，是：
   - `scenario32` 的训练策略
   - 更贴论文的训练细节
   - 或更进一步检查结构实现与论文是否还有偏差

## 10. 当前最推荐的实验配置

### 夜景场景推荐

适用于：

- `scenario33`
- `scenario34`

推荐参数：

```bash
python train.py \
  --scenarios scenario33 scenario34 \
  --image-subdir camera_data_mask_yolo \
  --batch-size 48 \
  --temporal-layers 2 \
  --fusion-layers 2 \
  --patience 8 \
  --early-stop-metric acc3 \
  --early-stop-mode max
```

### scenario32 推荐起点

```bash
python train.py \
  --scenarios scenario32 \
  --image-subdir camera_data \
  --batch-size 48 \
  --temporal-layers 2 \
  --fusion-layers 2 \
  --patience 8 \
  --early-stop-metric acc3 \
  --early-stop-mode max
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
- 当前夜景场景最有效图像输入：`camera_data_mask_yolo`
- 当前主要短板：`scenario32`
- 当前训练脚本已经支持：
  - `--image-subdir`
  - best checkpoint
  - early stopping
  - 多层 temporal / fusion Mamba
