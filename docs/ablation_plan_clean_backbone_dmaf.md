# 干净骨干与动态掩码自适应融合消融实验设计

本文档记录当前论文实验主线和消融设计。目标是把“完整输入性能提升”和“缺失输入鲁棒性提升”拆清楚，避免把多个增强模块堆在一起后无法说明到底哪个模块起作用。

## 1. 论文主线

最终论文按两项主创新组织：

1. **干净骨干精炼**：在原 BeMamba 完整输入框架上，通过受控消融筛出真正提升完整输入 Top-3 Accuracy 的模块，形成最终干净骨干。
2. **动态掩码自适应融合**：在最终干净骨干上加入缺失增强和缺失感知模块，使模型具备缺失输入鲁棒性。

论文命名上不把“clean 冲分模块”作为第三个最终方法。实现上最终模型可以是“干净骨干 + 鲁棒模块”，但论文叙事上应表述为：

```text
最终干净骨干 + 动态掩码自适应融合
```

方法图建议画成两层结构图：

- 上层：干净骨干如何形成完整输入预测能力。
- 下层：动态掩码自适应融合如何赋予缺失输入鲁棒性。

## 2. Baseline-0 定义

为回答“究竟是哪一个模块起作用”，消融起点改为真正的最小结构：

```text
ResNet/GPS projection + Temporal Mamba + 单顺序 MBMamba + 普通 MLP
```

`single-order` 只保留论文 Eq. 11 对应的第一种模态排列。代码在单顺序输出上乘以 3，使其名义尺度与原三顺序求和一致，避免把“顺序多样性”和“特征幅值变化”混在一起。`three-order MBMamba` 作为 Baseline-0 后的第一个新增模块。

MaskEncoder、ReliabilityEstimator 和 CrossModalAttention 不放进 clean 链。它们依赖缺失 mask，必须在固定缺失增强的 DMAF 链中验证；如果把它们与 clean heads 串成一条链，无法区分数据缺失增强、显式 mask 和 clean 分类头各自的贡献。

## 3. 严格论文口径

干净骨干精炼主消融使用严格论文口径：

```text
Data/splits_paper80_val
--selection-split val
--no-merge-trainval
```

含义：

- 验证集用于选择 checkpoint。
- 测试集只做最终报告。
- `splits_paper80` 的 test-select 结果只能作为探索或历史参考，不进入论文主消融决策。

主指标只使用 **Top-3 Accuracy**。Top-1、Top-2、DBA、APL 只做辅助分析，不能替代 Top-3 决定模块是否进入最终干净骨干。

## 4. 干净骨干精炼消融

### 4.1 消融原则

干净骨干精炼采用 **受控干净消融版本**，不直接复用历史 `clean_plus_v*` 版本作为论文消融。

原因是历史版本存在模块耦合，例如：

- `clean_plus` 不只包含顺序融合门控，还可能包含预测头、空间混合、clean cross-attention。
- `clean_plus_v5` 继承了前面多个模块后再加入波束查询细化头。
- `clean_plus_v10/v14` 又叠加了候选重排、邻域细化和模态特征丢弃。

论文消融需要每一行只改变一个明确功能组。

### 4.2 受控消融入口

后续建议新增训练入口：

```text
--clean-ablation-stage
```

建议取值：

| stage | 中文含义 |
|---|---|
| `base` | Baseline-0：单顺序 MBMamba + MLP |
| `three_order` | 改为三顺序 MBMamba 简单求和 |
| `order_gate` | 顺序融合增强 |
| `attn_head` | 预测头增强 |
| `branch` | 分支监督增强 |
| `beam_query` | 波束查询增强 |
| `ordinal` | 波束序数先验 |
| `neighbor` | 波束邻域增强 |
| `rerank` | 候选重排增强 |
| `modality_dropout` | 训练期模态正则 |

该入口应与历史 `--model-variant clean_plus_v*` 分离，避免论文消融和历史探索版本混用。

### 4.3 前向加入顺序

干净骨干精炼主表采用前向加入消融，但不是失败模块强制累积。

预设测试顺序：

| 顺序 | 功能组 | 主要模块 |
|---:|---|---|
| 0 | Baseline-0 | ResNet/GPS projection + Temporal Mamba + 单顺序 MBMamba + MLP |
| 1 | 三顺序融合 | Three-order MBMamba + simple sum |
| 2 | 顺序融合增强 | OrderFusionGate |
| 3 | 预测头增强 | AttentivePredictionHead |
| 4 | 分支融合增强 | Branch Ensemble；主链默认不额外启用 auxiliary branch loss |
| 5 | 波束查询增强 | Beam Query Refinement；仍只用固定主损失 |
| 6 | 波束序数增强 | Beam Ordinal Prior |
| 7 | 波束邻域增强 | Beam Neighborhood Head |
| 8 | 候选重排增强 | Candidate Reranker；仍只用固定主损失 |
| 9 | 训练期模态正则 | Modality Feature Dropout |

这张表是严格的累计链，相邻两行只增加一个模块。`beam_query` 和 `rerank` 阶段默认不自动打开额外 ranking/rerank loss，避免把“头部结构”和“训练监督”混成一个变量；如果论文还要比较辅助损失，应另开 loss-only 表。由于历史结果表明 Ordinal Prior、Candidate Reranker 等可能为负，论文还需要补最终模型的 leave-one-out；不能只用累计链声称模块具有独立贡献。

### 4.4 模块去留规则

采用 **Top-3 去留规则**：

- 开发阶段只根据固定 val 的 Top-3 判断，不读取或重复使用 SC32 final test。
- 如果 Top-3 持平，但 Top-1/Top-2/DBA/APL 更好，只能作为辅助优势，不能单独决定入选。
- 如果 Top-3 下降，即使辅助指标变好，也不进入最终干净骨干。
- 小幅提升必须在 seeds `11/7/42` 的 paired val 结果中方向稳定，才进入跨场景候选。

累计链负责回答“在固定前序结构下，新模块的条件增量是多少”。模块去留则由 paired val、跨场景验证和最终模型 leave-one-out 共同决定。累计链中即使某一阶段为负，也保留后续行以完整呈现模块交互，但不能把后续结果解释为该失败模块的独立收益。

### 4.5 固定控制变量

干净骨干精炼主消融固定以下条件：

- 场景：SC32 只做冻结后的 val 诊断；新的结构选择以 SC33/SC34 paired val 为主。
- 评估口径：严格论文口径。
- 骨干容量：固定 `backbone_stage=3`。
- 主损失：固定同一个主分类损失，例如 `power_soft_ce`。
- 辅助损失：clean 模块主链全部固定为 0。Top-k ranking loss、candidate rerank loss 如需验证，单独做 loss-only 对照。
- 不把 v15 curriculum 放入主消融，因为当前实验结论为负结果。
- 所有开发消融都加 `--skip-final-test`。只有配置预先声明且跨场景 val 支持后，才各运行一次 final test。

### 4.6 最终移除验证

最终干净骨干确定后，做入选模块移除验证：

```text
最终干净骨干
最终干净骨干 - 入选模块 A
最终干净骨干 - 入选模块 B
最终干净骨干 - 入选模块 C
```

只移除最终真正入选的模块，不移除前向阶段已经失败的模块。

## 5. 鲁棒模块消融

### 5.1 单基座设计

鲁棒模块消融采用 **单基座鲁棒消融**：

```text
所有缺失增强和缺失感知模块都接在同一个最终干净骨干上比较
```

不再额外做“最小 BeMamba 基线 + 鲁棒模块”的双基座消融。最终干净骨干本身就是相对原论文的完整输入创新，再在这个基础上加入鲁棒机制。

### 5.2 鲁棒消融起点

鲁棒前向消融从 **缺失增强鲁棒基线** 开始：

```text
最终干净骨干 + 训练期缺失模拟，但不使用显式缺失感知融合模块
```

这样可以分清：

- 缺失增强本身带来的鲁棒性收益。
- 显式缺失感知模块在缺失增强基础上的额外收益。

### 5.3 鲁棒模块顺序

鲁棒前向消融建议顺序：

| 顺序 | 中文名 | 含义 |
|---:|---|---|
| 0 | 缺失增强鲁棒基线 | 最终干净骨干 + 缺失增强，不使用显式 mask 模块 |
| 1 | 掩码时序聚合 | 使用 mask 对帧级时序特征做加权聚合 |
| 2 | 掩码编码器 | 注入缺失/存在状态 embedding |
| 3 | 可靠性估计器 | 根据 mask ratio 和特征统计调整模态权重 |
| 4 | 跨模态注意力 | 让可用模态补偿缺失模态 |

注意：代码层面要区分“掩码时序聚合”和“掩码编码器”。当前实现中，`--no-mask-embed` 只关闭掩码编码器，但只要 `missing_enabled=True`，mask 仍可能传入 `TemporalProcessor` 做加权聚合。论文表述中必须拆开这两个概念。

现在使用独立入口 `--dmaf-ablation-stage`，相邻阶段固定为：

```text
missing_aug -> mask_pool -> mask_embed -> reliability -> cross_attn
```

- `missing_aug`：数据侧制造缺失，模型不接收 mask。
- `mask_pool`：新增 mask-weighted temporal aggregation。
- `mask_embed`：再新增 MaskEncoder embedding。
- `reliability`：再新增 ReliabilityEstimator。
- `cross_attn`：最后新增 CrossModalFusion。

该入口禁止和 `--missing-enabled`、`--dmaf-enabled`、`--no-mask-*`、`--no-reliability`、`--no-cross-attn` 混用，避免配置看似单变量、实际偷偷改变多个模块。

### 5.4 固定缺失增强策略

鲁棒消融所有行必须固定同一套训练期缺失增强策略，包括：

- 帧级缺失概率。
- 连续帧缺失概率。
- 模态级缺失概率。
- 可缺失模态集合。
- 缺失随机种子。
- 训练 epoch、学习率、EMA、选择口径。

每一行只改变模型侧缺失感知模块，不能同时改变缺失增强强度。

### 5.5 鲁棒评估协议

鲁棒评估主表使用五类协议，全部报告 Top-3 Accuracy：

| 协议 | 含义 |
|---|---|
| 完整输入 | 无缺失，确认鲁棒模块不明显伤害 clean 性能 |
| 随机帧缺失 | 每个时间帧随机缺失，检验时序鲁棒性 |
| 连续帧缺失 | 连续时间段缺失，检验 burst 场景 |
| 整模态缺失 | 某个传感器整体不可用，检验模态级鲁棒性 |
| 混合缺失 | 多种缺失同时出现，检验复杂退化场景 |

主讨论重点：

- 完整输入：不能明显牺牲干净骨干。
- 随机帧缺失：动态掩码自适应融合最稳定的优势点。
- 混合缺失：更接近真实多故障情况。

整模态缺失如果不是最强项，应如实表述为“不是当前方法的主要优势点”。

### 5.6 鲁棒模块移除验证

完整动态掩码自适应融合确定后，做小规模鲁棒模块移除验证：

```text
完整鲁棒模型
完整鲁棒模型 - 掩码编码器
完整鲁棒模型 - 可靠性估计器
完整鲁棒模型 - 跨模态注意力
```

掩码时序聚合可以主要在前向消融中证明；如果代码实现上它是基础 mask 使用方式，不一定放入最终移除表。

## 6. 四张核心实验表

### 表 1：干净骨干精炼前向消融

用途：证明哪些完整输入模块进入最终干净骨干。

场景：SC32 仅报告冻结 val 诊断；正式选择优先看 SC33/SC34 paired val。

建议列：

| 列名 | 含义 |
|---|---|
| 阶段 | 当前测试的消融阶段 |
| 新增功能组 | 相对当前最佳干净骨干尝试加入什么 |
| Val Top-3 | checkpoint selection 依据 |
| Paired ΔVal Top-3 | 相对上一阶段、同 seed 的变化 |
| 三种子均值/方向 | 判断收益是否稳定 |
| 是否入选 | 是否进入后续当前最佳干净骨干 |

Top-1、Top-2、DBA、APL 不放主表，可放附录或备注。

### 表 2：最终干净骨干三场景验证

用途：证明最终干净骨干在 SC32/SC33/SC34 稳定。

建议方法行：

| 方法 | 作用 |
|---|---|
| 原论文/复现 BeMamba 基线 | 外部或复现参考 |
| 最小 BeMamba 基线 | 受控消融起点 |
| 最终干净骨干 | 干净骨干精炼后的最终模型 |

建议列：

```text
SC32 Top-3 / SC33 Top-3 / SC34 Top-3 / 平均 Top-3
```

历史探索模型可以作为补充参考，但必须标明探索口径，不能混入严格论文主对比。

### 表 3：动态掩码自适应融合鲁棒主结果

用途：证明缺失增强和动态掩码自适应融合的鲁棒性收益。

建议方法行：

| 方法 | 含义 |
|---|---|
| 原始 BeMamba / 最小基线 | 完整输入训练，不做缺失增强 |
| 最终干净骨干 | 完整输入训练，不做缺失增强 |
| 缺失增强鲁棒基线 | 最终干净骨干 + 缺失增强，不使用显式缺失感知模块 |
| 完整动态掩码自适应融合 | 最终干净骨干 + 缺失增强 + 掩码/可靠性/跨模态模块 |

建议列：

```text
完整输入 / 随机帧缺失 / 连续帧缺失 / 整模态缺失 / 混合缺失
```

全部报告 Top-3 Accuracy。

### 表 4：鲁棒模块消融

用途：证明缺失感知模块各自的贡献。

开发消融先在 val split 上覆盖五类鲁棒协议，输出 `missing_val_result.csv`。SC32 test 已冻结，不再用于模块选择；SC33/SC34 也只有在配置预先冻结后才运行 final test。

建议包含两类结果：

1. 鲁棒前向消融：从缺失增强鲁棒基线逐步加入掩码时序聚合、掩码编码器、可靠性估计器、跨模态注意力。
2. 鲁棒模块移除验证：在完整鲁棒模型上移除入选鲁棒模块。

## 7. 当前执行优先级（2026-07-14）

建议执行顺序：

1. 先在 SC33/SC34 完成 `clean_plus_v14 + physical_kinematic` 与原 GPS 的 paired val-only 验证，确认当前 clean 基座能跨场景。
2. 若论文需要解释全部复杂模块，再跑 clean 累计链；每个 stage 使用 seeds `11/7/42`，只读 val。
3. 对最终保留的 clean 模块做 leave-one-out，避免仅靠累计链归因。
4. 在固定 clean backbone 上跑 DMAF 链：`missing_aug -> mask_pool -> mask_embed -> reliability -> cross_attn`。
5. DMAF 消融在 val 上同时报告 Clean、Frame 50%、Burst 60%、Modal 60%、Hybrid Top-3，不得只挑有利协议。
6. 只有在配置预先冻结后，才对 SC33/SC34 各运行一次 final test；SC32 test 不再用于选择。

通用 val-only 脚本格式：

```bash
bash scripts/run_controlled_ablation_val.sh clean base scenario33 11
bash scripts/run_controlled_ablation_val.sh clean three_order scenario33 11
```

DMAF 链示例：

```bash
bash scripts/run_controlled_ablation_val.sh dmaf missing_aug scenario33 11
bash scripts/run_controlled_ablation_val.sh dmaf mask_pool scenario33 11
bash scripts/run_controlled_ablation_val.sh dmaf mask_embed scenario33 11
bash scripts/run_controlled_ablation_val.sh dmaf reliability scenario33 11
bash scripts/run_controlled_ablation_val.sh dmaf cross_attn scenario33 11
```

脚本统一固定 `physical_kinematic` GPS、val selection、`--skip-final-test`、扫描顺序和优化参数。DMAF family 额外使用 `--eval-selection-missing`，因此会在 val 上生成完整缺失协议结果而不读取 test。SC33/SC34 自动使用 `camera_data_mask_yolo`，SC32 使用 `camera_data_mask`。
