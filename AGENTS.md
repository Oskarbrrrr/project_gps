# 项目工作备忘（AGENTS）

## 1. 项目概览

这个仓库是一个针对论文 **BeMamba: Efficient Multimodal Sensing-Aided Beamforming via State Space Model** 的复现项目代码仓库。当前仓库内只有代码，不包含：

- 数据集本体
- 训练日志
- `checkpoints/`
- `logs/`
- 依赖清单文件（如 `requirements.txt`）

仓库结构很精简，核心逻辑基本都集中在少数几个文件中：

- `train.py`：训练与验证主入口
- `src/dataset.py`：多模态数据读取与预处理
- `src/model.py`：BeMamba 模型结构
- `src/utils.py`：评价指标
- `src/data_split.py`：训练/验证/测试划分脚本
- `src/plot_results.py`：训练曲线绘图
- `visualize_lidar.py`：LiDAR BEV 可视化脚本


## 2. 当前代码的整体流程

### 2.1 训练流程

`train.py` 里定义了 `run_scenario(scenario_name)`，会按场景分别训练：

- `scenario32`
- `scenario33`
- `scenario34`

每个场景流程大致是：

1. 读取 `Data/splits/{scenario}_train.csv`
2. 根据训练集标签统计 beam 类别频率，构造 `AlphaFocalLoss`
3. 构建训练/验证/测试集
4. 初始化 `BeMambaModel`
5. 训练 100 个 epoch
6. 根据验证集 loss 保存最优模型
7. 最后加载最佳权重，在测试集上计算：
   - Top-1 Acc
   - Top-3 Acc
   - DBA
   - APL

训练输出会写到：

- `checkpoints/best_{scenario}.pth`
- `logs/{scenario}_train_log.csv`
- `logs/{scenario}_final_test_result.txt`


### 2.2 模型输入模态

每个样本包含 4 个模态：

- Image
- Radar
- LiDAR
- GPS

并且每个样本按时间序列读取 **5 帧**。


## 3. 数据集读取逻辑

核心文件：`src/dataset.py`

### 3.1 CSV 依赖字段

当前代码假定 CSV 里至少存在这些列：

- `unit1_rgb_1` ~ `unit1_rgb_5`
- `unit1_radar_1` ~ `unit1_radar_5`
- `unit1_lidar_1` ~ `unit1_lidar_5`
- `unit1_loc`
- `unit2_loc_1`
- `unit2_loc_2`
- `unit1_beam`
- `unit1_pwr_60ghz`

说明：

- `unit1` 看起来是基站或主车载端
- `unit2` 看起来是 UE 或目标端
- 标签 `unit1_beam` 最终被转成 `0~63` 的 64 分类


### 3.2 Image 预处理

Image 当前使用的是**原始 RGB 图像**，没有做论文里可能存在的额外几何变换。代码逻辑为：

1. 读取图片
2. resize 到 `256 x 256`
3. 转 tensor
4. 按 ImageNet 均值方差归一化

这和你描述的“Image 我只用原本的图像”一致。


### 3.3 Radar 预处理

Radar 当前不是直接读单个文件，而是把一个路径拆成两个派生路径：

- `radar_data_ang`
- `radar_data_vel`

然后把两者 stack 成一个 2 通道输入：

- channel 0：angle
- channel 1：velocity

如果读取失败，会退化成全零张量 `2 x 256 x 256`。


### 3.4 LiDAR 预处理

LiDAR 是当前最值得重点排查的部分。

#### 当前实现逻辑

`_ply_to_base_bev()` 的处理步骤：

1. 使用 `open3d` 读取 `.ply` 点云
2. 取点云的 `x, y`
3. 用 `x, y` 的**中位数**作为中心
4. 以这个中心为原点，取一个半径 `R = 30m` 的正方形区域
5. 映射到 `256 x 256` 的 BEV 网格
6. 对落在网格内的位置赋值 1

随后在 `__getitem__()` 中，对连续 5 帧做：

1. 当前帧先生成基础 BEV
2. 若存在上一帧，则执行帧差
3. 对“当前新增点”做随机扩散，生成所谓 virtual points
4. 最终得到 `1 x 256 x 256` 的 LiDAR 输入

#### 当前 virtual point 逻辑

`_generate_virtual_points()` 里：

- `diff = current_bev - prev_bev`
- 只保留 `diff > 0.5` 的新增点
- 对每个新增点随机偏移，重复撒点 3 次
- 偏移范围大致是 `[-2, 2]` 像素

这更像是“启发式增强”，未必与论文完全严格一致。

#### LiDAR 部分需要重点怀疑的点

1. **BEV 中心是按每一帧点云中位数自动估计的**
   - 这会导致不同帧之间坐标基准不稳定
   - 如果论文要求使用固定车体坐标系、传感器坐标系或全局坐标系，这里可能已经偏掉

2. **BEV 范围固定为中心周围 `30m`**
   - 这是代码作者手工设定的超参数
   - 不确定是否与论文完全一致

3. **网格赋值时使用的是 `bev[x_idx, y_idx] = 1.0`**
   - 需要确认坐标轴是否应该交换
   - 需要确认是否应该翻转某个轴
   - 很多 BEV 看起来“不对”的问题，本质上就是 `x/y` 轴映射、原点方向、上下翻转、左右翻转不一致

4. **virtual points 是随机生成的**
   - 这会引入不稳定性
   - 如果实验可复现性很重要，后面要考虑固定随机种子，或者改成与论文一致的确定性策略

5. **当前 `visualize_lidar.py` 正适合用来检查这部分**
   - 后续排查 BEV 是否转歪、是否稀疏异常、是否中心漂移，优先看这个脚本


### 3.5 GPS 预处理

GPS 从文本文件中读取经纬度，然后：

1. 以 `unit1_loc` 为基站位置
2. 读取 `unit2_loc_1` 和 `unit2_loc_2`
3. 转成平面位移 `dx, dy`
4. 在当前 split 内做 min-max 归一化
5. 最终构造 `[dist, angle]`

最后每个样本会生成两组 GPS 特征：

- `gps_start`
- `gps_end`

最终 `gps` tensor 形状是 `2 x 2`。


## 4. 模型结构理解

核心文件：`src/model.py`

### 4.1 各模态 backbone

- Image：`ResNet34` 前半段
- Radar：`ResNet18`，首层改成 2 通道输入
- LiDAR：`ResNet18`，首层改成 1 通道输入
- GPS：两层 MLP，输出 128 维

### 4.2 时序建模

每个模态的 5 帧特征会先经过各自的 `Mamba`：

- `tsm_img`
- `tsm_rad`
- `tsm_lid`

### 4.3 多模态融合

融合方式不是简单拼接一次，而是构造了 3 种模态顺序组合，再分别过 `MB_Mamba_Block`：

- `gps_start + img + lid + rad + gps_end`
- `gps_start + lid + rad + img + gps_end`
- `gps_start + rad + img + lid + gps_end`

最后把 3 个输出平均，再过分类头输出 64 类 beam。


## 5. 当前代码里已经能看到的问题

下面这些是“通读代码就能直接看到”的问题，后续优先确认。

### 5.1 `train.py` 与 `MultimodalDataset` 的调用方式不匹配

这是当前最明显的结构性问题之一。

`src/dataset.py` 里构造函数定义是：

- `MultimodalDataset(mode='train', data_root='./Data/Multi_Modal', split_root='./Data/splits', scenario_name='scenario32')`

但 `train.py` 里写的是：

- `train_ds = MultimodalDataset(train_csv_path)`
- `val_ds = MultimodalDataset(f"Data/splits/{scenario_name}_val.csv")`
- `test_ds = MultimodalDataset(f"Data/splits/{scenario_name}_test.csv")`

也就是说，`train.py` 把一个 CSV 路径字符串当成了 `mode` 传入，而不是按 `mode/scenario_name` 方式传参。

如果当前仓库就是实际运行版本，那么这里理论上会直接导致数据集初始化行为异常，除非远程机器上的代码不是这一份，或者你本地这份和正在跑的版本已经不一致。

这件事后面必须核对。


### 5.2 文件里存在中文注释乱码

不少文件中的中文注释在当前环境下显示为乱码，说明文件编码可能混乱，常见原因包括：

- UTF-8 / GBK 混用
- Windows 和 Linux 之间来回编辑
- 编辑器默认编码不同

这不一定影响运行，但会影响维护和排错。


### 5.3 缺少显式依赖清单

从代码看，至少依赖：

- `torch`
- `torchvision`
- `pandas`
- `numpy`
- `Pillow`
- `open3d`
- `matplotlib`
- `scikit-learn`
- `mamba_ssm`

但仓库中没有依赖说明文件。后面如果要在 AutoDL 上稳定复现，最好补：

- `requirements.txt`
或
- `environment.yml`


### 5.4 数据路径假设比较强

当前代码默认数据目录是：

- `./Data/Multi_Modal`
- `./Data/splits`

这意味着：

- 远程机和本地机的目录结构需要尽量一致
- CSV 中存的路径需要和 `data_root` 的拼接逻辑匹配

只要远程和本地目录层次一变，很多地方就会直接报错。


## 6. 我目前对你这个项目状态的判断

基于你给的信息和当前代码，我的理解是：

1. 你在复现论文
2. 除了 Image，其他三个模态预处理你认为已经基本对齐论文
3. 现在最怀疑的是 **LiDAR 转 BEV** 这一步
4. 当前仓库是从 AutoDL 远程机 `git clone` 下来的纯代码副本
5. 本地没有数据、没有日志、没有训练结果
6. 你希望建立一个方便的“本地改代码 -> 上传到远程跑 -> 把结果带回本地继续改”的工作方式

这个判断和当前仓库状态是吻合的。


## 7. 建议的工作模式（本地 + AutoDL）

这是后续建议采用的默认协作方式。

### 7.1 分工建议

本地（当前这个仓库）负责：

- 阅读代码
- 改代码
- 做静态排查
- 写可视化和调试脚本
- 管理 git 版本

AutoDL 远程机负责：

- 放数据集
- 安装依赖
- 正式训练
- 保存日志、模型、可视化结果

### 7.2 推荐闭环

每次都走这个闭环：

1. 在本地改代码
2. git 提交
3. push 到远程仓库
4. 在 AutoDL 上 `git pull`
5. 在 AutoDL 上跑训练/可视化
6. 把日志、图、报错、关键结果同步回本地
7. 再继续改

### 7.3 不建议的方式

尽量不要：

- 本地改一部分，远程再手工改一部分
- 远程直接改代码但不提交
- 用压缩包反复覆盖代码目录

因为这样非常容易把“到底哪份代码跑出来的结果”搞乱。


## 8. 后续排查优先级建议

下一步建议按这个顺序推进：

1. 先修正并核对 `train.py` 和 `dataset.py` 的接口是否一致
2. 再检查 LiDAR BEV 的坐标映射是否正确
3. 再通过 `visualize_lidar.py` 导出样本图，人工看：
   - 是否方向颠倒
   - 是否中心漂移
   - 是否过稀或过密
   - 连续帧差是否合理
4. 再考虑 virtual points 是否真的符合论文
5. 最后再回头看训练效果和指标


## 9. 我和后续协作时应优先记住的事实

- 当前仓库路径：`D:\code\project_gps`
- 论文文件路径：`C:\Users\Oskar\Desktop\BeMamba_Efficient_Multimodal_Sensing-Aided_Beamforming_via_State_Space_Model.pdf`
- 当前本地仓库不包含数据和日志
- 用户当前最关注的是 **LiDAR 转鸟瞰图（BEV）是否有问题**
- Image 模态当前明确是“只用原图”
- 远程 AutoDL 才是正式运行环境
- 用户希望我顺带建立一套简单稳定的 git/远程协作工作流


## 10. 下一步最自然的动作

后续继续推进时，优先做这几件事：

1. 对照论文检查 `src/dataset.py` 中 LiDAR BEV 的实现
2. 修正训练入口与数据集接口不一致的问题
3. 建立一份适合 AutoDL 的运行说明
4. 帮用户把 git 的“本地提交 -> 远程拉取 -> 跑完同步结果”流程固定下来

## 11. 2026-05-11 当前已完成的本地修正

这部分记录“已经动过的代码”，避免后面重复怀疑。

### 11.1 已修正训练入口和数据集接口不匹配

`train.py` 现在已经改成按下面的方式构造数据集：

- `MultimodalDataset(mode='train', scenario_name=scenario_name)`
- `MultimodalDataset(mode='val', scenario_name=scenario_name)`
- `MultimodalDataset(mode='test', scenario_name=scenario_name)`

不再把 CSV 路径字符串误传给 `mode` 参数。

### 11.2 已重写 `src/dataset.py`，保留原功能但修正了 LiDAR BEV 的核心参考系问题

当前版本的关键变化：

1. `MultimodalDataset` 现在支持显式传 `csv_path`
2. LiDAR BEV 使用固定范围投影：
   - `lidar_x_range=(-30.0, 30.0)`
   - `lidar_y_range=(-30.0, 30.0)`
3. 不再对每一帧点云先用中位数自动重心居中后再投影

### 11.3 为什么这个修正很重要

之前的实现是“每一帧单独找中位数中心，再映射成 BEV”。这会带来一个很大的隐患：

- 即使环境本身是静态的
- 只要点云分布略有变化
- 当前帧和上一帧的 BEV 也会因为参考系变化而错位

这样一来，后面的：

- 帧差
- moving points 检测
- virtual points 生成

都会被污染，模型看到的是“人为制造的运动”，而不是真实运动。

### 11.4 已增强 LiDAR 可视化脚本

`visualize_lidar.py` 现在会：

1. 支持显式指定 `mode`
2. 输出文件名里带上 `scenario`、`mode`、`sample_idx`
3. 在每一帧标题里显示非零像素数量

这有助于直接判断：

- BEV 是否过稀
- 连续帧是否突然爆亮
- virtual points 是否把图撒得太满

### 11.5 新增了远程协作文档

已新增文件：

- `AUTODL_WORKFLOW.md`

里面记录了：

- 本地如何提交和 push
- AutoDL 如何 pull 和运行
- 日志结果怎么带回本地
- 哪些 git 使用习惯最稳妥
