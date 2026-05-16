# BeMamba Paper Page Index

本文对应图片目录：`paper_pages/001.jpg` 到 `paper_pages/014.jpg`，按页码递增一一对应。

## 用途

- 新开话题时，快速定位论文某一页的大致内容
- 将论文结构、关键图表、关键公式和仓库复现重点对应起来
- 避免反复手动翻图

## 页码索引

### 第 1 页 - 背景、动机与摘要

- 对应图片：`paper_pages/001.jpg`
- 主要内容：
  - 论文标题、作者信息、摘要、关键词
  - 引言中的动机：高移动性 V2I 场景下，传统 beam training 有 selection delay，多模态感知辅助方法有 computation delay
- 关键图表：
  - 图 1：传统波束训练 vs 多模态感知辅助波束预测
- 复现关注点：
  - 明确论文主要目标不是只提精度，而是同时兼顾低时延与可部署性

### 第 2 页 - 核心思路与相关工作

- 对应图片：`paper_pages/002.jpg`
- 主要内容：
  - 用状态空间模型 Mamba 替代 Transformer 做多模态融合
  - 强调线性复杂度 `O(N)` 的优势
  - 相关工作：感知辅助 beamforming、多模态高效融合
- 关键图表：
  - 图 2：Transformer 融合与 Mamba 融合结构对比
- 复现关注点：
  - 当前复现里 `TFMamba` 和 `MBMamba` 的实现，应优先对齐这里的效率动机

### 第 3 页 - 论文贡献与系统模型

- 对应图片：`paper_pages/003.jpg`
- 主要内容：
  - 4 点主要贡献
  - 系统设计开始：beamforming 数学建模与优化目标
  - 多模态编码问题：不同模态存在异构性
- 复现关注点：
  - 训练目标、输入定义、任务形式应与这里保持一致

### 第 4 页 - 整体架构与图像/LiDAR 预处理

- 对应图片：`paper_pages/004.jpg`
- 主要内容：
  - BeMamba 整体框架
  - RGB 图像的 Mask + Enhance
  - LiDAR 的 BEV 转换与 virtual point generation
- 关键图表：
  - 图 3：整体框架图
  - 图 4：图像处理示例
  - 图 5：LiDAR 处理示例
- 复现关注点：
  - 图像路径切换：`camera_data` / `camera_data_mask` / `camera_data_mask_yolo`
  - LiDAR 当前实现已较接近论文的 `BEV + Generate`

### 第 5 页 - Radar/GPS 预处理与 CNN 提特征

- 对应图片：`paper_pages/005.jpg`
- 主要内容：
  - Radar：range-angle map 与 range-velocity map
  - GPS：相对位置归一化与坐标转换
  - CNN 初步特征提取，包含 ResNet34/ResNet18 设计
- 关键图表/公式：
  - 图 6：Radar 处理示例
  - 公式 1：GPS 坐标转换
  - 公式 2：CNN 下采样与卷积
- 复现关注点：
  - 当前 Radar 两通道输入与论文思路一致
  - GPS 归一化必须遵守 train 统计量复用于 test 的协议

### 第 6 页 - 时序 Mamba 设计

- 对应图片：`paper_pages/006.jpg`
- 主要内容：
  - CNN 特征维度说明
  - SSM / Mamba 基础数学
  - Time Sequence Mamba 设计
- 关键图表/公式：
  - 图 7：Time Sequence Mamba 内部结构
  - 公式 7-9：SSM 连续与离散化公式
- 复现关注点：
  - `TFMamba` 是否与论文的时序聚合逻辑一致
  - `temporal_layers` 是否真正多层堆叠并带残差

### 第 7 页 - 时序聚合公式与模态 Mamba 设计

- 对应图片：`paper_pages/007.jpg`
- 主要内容：
  - 时序 Mamba 聚合公式
  - Modal Sequence Mamba 设计
  - 图像、LiDAR、Radar、GPS 的 mixed combinations 与排序方式
- 关键图表/公式：
  - 图 8：Modal Sequence Mamba 内部结构
  - 公式 12：三种 mixed modal combinations
- 复现关注点：
  - 当前 mixed modal combinations 的构造方式要优先对齐这一页
  - `fusion_layers` 是否真正多层堆叠并带残差

### 第 8 页 - 预测头、实验设置与评估指标

- 对应图片：`paper_pages/008.jpg`
- 主要内容：
  - Modal Sequence Mamba 的双向聚合与 prediction head
  - 数据集 DeepSense 6G
  - 评估指标：Top-K Accuracy 与 DBA
- 关键图表/公式：
  - 表 I：四个场景和样本量统计
  - 公式 15-16：Top-K 与 DBA
- 复现关注点：
  - 当前统一输出 `Top-1/2/3`、`DBA`、`APL`
  - 论文核心对齐指标仍以 `Top-3` 和 `DBA` 为主

### 第 9 页 - 基线对比与效率结果

- 对应图片：`paper_pages/009.jpg`
- 主要内容：
  - 基线模型说明，如 TII-Transfuser
  - BeMamba 在性能和效率上的优势
  - 计算量降低 77.88%，推理速度提升 4.56 倍
- 关键图表：
  - 表 II：不同模态组合下的主结果
  - 图 9：参数量、显存、FLOPs、推理速度对比
- 复现关注点：
  - 需要区分“论文主结果”和“当前仓库最可信结果”
  - 未来若补效率测试，可优先回看这一页

### 第 10 页 - 消融分析一：预处理与 CNN/池化

- 对应图片：`paper_pages/010.jpg`
- 主要内容：
  - 模态预处理的有效性分析
  - CNN 层数与 pooling size 的影响
- 关键图表：
  - 图 10：预测结果可视化
  - 表 III：去除预处理前后的性能对比
- 复现关注点：
  - 图像预处理收益在不同场景不一致，尤其要关注 `scenario32` 与夜景场景差异

### 第 11 页 - 消融分析二：时间顺序与模态组合

- 对应图片：`paper_pages/011.jpg`
- 主要内容：
  - 时间顺序对 Mamba 编码的影响
  - 模态组合方式对性能的影响
  - 模态空间对齐时线性变换的必要性
- 关键图表：
  - 图 11、图 12：CNN 层数和池化大小影响
  - 图 13：时间顺序策略对比
  - 图 14：随机/单一/混合组合策略对比
- 复现关注点：
  - 若继续追 `scenario32`，这页是训练策略和组合策略排查重点

### 第 12 页 - 整体消融、可视化与结论

- 对应图片：`paper_pages/012.jpg`
- 主要内容：
  - t-SNE 特征可视化
  - 模块级整体消融
  - 参数量与 FLOPs 拆解
  - 结论
- 关键图表：
  - 图 15：t-SNE
  - 图 16：特征热力图
  - 表 IV：CNN / Temporal Mamba / Modal Mamba 消融
  - 表 V：Params 与 FLOPs 拆解
- 复现关注点：
  - 表 IV 可用于对照当前结构修正后是否得到趋势一致的收益
  - 表 V 可用于后续补复杂度统计

### 第 13 页 - 参考文献与作者简介一

- 对应图片：`paper_pages/013.jpg`
- 主要内容：
  - 参考文献 [1]-[36]
  - 前三位作者简介
- 复现关注点：
  - 若要追溯相关方法来源，可从这里反查引用

### 第 14 页 - 作者简介二

- 对应图片：`paper_pages/014.jpg`
- 主要内容：
  - 后三位作者简介
- 复现关注点：
  - 无核心技术细节，可忽略

## 快速检索建议

- 找模型整体架构图：第 4 页，`paper_pages/004.jpg`
- 找图像 / LiDAR 预处理：第 4 页，`paper_pages/004.jpg`
- 找 Radar / GPS 处理：第 5 页，`paper_pages/005.jpg`
- 找时序 Mamba：第 6 页，`paper_pages/006.jpg`
- 找模态 Mamba 与 mixed combinations：第 7 页，`paper_pages/007.jpg`
- 找评估指标定义：第 8 页，`paper_pages/008.jpg`
- 找论文主结果：第 9 页，`paper_pages/009.jpg`
- 找预处理消融：第 10 页，`paper_pages/010.jpg`
- 找时间顺序 / 组合策略消融：第 11 页，`paper_pages/011.jpg`
- 找模块消融与复杂度拆解：第 12 页，`paper_pages/012.jpg`

## 与当前仓库最相关的页

- `src/model.py` 重点对应：第 4、6、7、8、12 页
- `src/dataset.py` 重点对应：第 4、5、8 页
- `prepare_camera_mask_yolo.py` 重点对应：第 4、10 页
- `train.py` 重点对应：第 8、9、10、11、12 页
