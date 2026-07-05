# Project GPS Context

This context names the experiment lines and beam prediction concepts used in this repository. It exists to keep clean-data accuracy work separate from missing-data robustness work.

## Language

**Clean-Plus**:
The clean-data model-structure optimization line, trained and evaluated on complete sensor inputs with the goal of improving clean Top-K beam prediction.
_Avoid_: clean reproduction, baseline tuning, paper reproduction

**Own Clean Baseline**:
The paper's own complete-input baseline built on the reproduced BeMamba pipeline and allowed to include clearly stated clean-data structural improvements.
_Avoid_: Paper Baseline, unmodified BeMamba, robustness method

**最小 BeMamba 基线**:
The smallest BeMamba-style reproduction baseline used for architecture ablations: ResNet/GPS projection, Temporal Mamba, three-order MBMamba fusion, and a plain MLP classifier.
_Avoid_: Baseline-0, single-order baseline, clean-plus backbone

**Clean Backbone**:
The strong complete-input main model that provides clean Top-3 beam prediction ability before missing-aware robustness modules are added; it does not claim missing-input robustness by itself.
_Avoid_: ResNet-only backbone, DMAF, missing-aware module

**干净骨干精炼**:
The clean-data ablation line that identifies which complete-input scoring, fusion, beam-refinement, and regularization modules should be kept inside the Clean Backbone.
_Avoid_: third final method, DMAF, missing-aware robustness module

**两项主创新**:
论文最终叙事中的创新组织方式，将贡献归纳为干净骨干精炼和动态掩码自适应融合两部分。
_Avoid_: 三个主方法, clean 冲分作为独立最终方法, 模块堆叠叙事

**两层结构图**:
论文方法图的组织方式，上层表达干净骨干如何形成完整输入预测能力，下层表达动态掩码自适应融合如何赋予缺失输入鲁棒性。
_Avoid_: 全模块大杂烩图, 单层堆叠图, clean 冲分和鲁棒模块平铺

**四张核心实验表**:
论文主实验的表格组织方式，依次覆盖干净骨干前向消融、最终干净骨干三场景验证、动态掩码自适应融合鲁棒主结果和鲁棒模块消融。
_Avoid_: 单张大表, 模块和鲁棒结果混表, 只报最终模型

**干净消融主表指标**:
干净骨干精炼前向消融表的指标范围，只围绕验证集 Top-3、测试集 Top-3、Top-3 增量和是否入选来呈现。
_Avoid_: Top-1 主表决策, DBA 主表决策, 辅助指标挤占主表

**干净骨干主结果指标**:
最终干净骨干三场景验证表的指标范围，以 SC32、SC33、SC34 的 Top-3 Accuracy 和平均 Top-3 作为主呈现。
_Avoid_: 单场景主结果, 辅助指标主结果, 历史探索口径混入严格主表

**受控干净消融版本**:
干净骨干精炼中为论文消融专门定义的模型版本，每个版本只新增一个预先命名的功能组，避免复用历史 clean_plus 版本造成模块耦合。
_Avoid_: 历史版本代替消融, 多模块同时变化, test-select 版本筛选

**受控消融入口**:
用于选择受控干净消融版本的训练配置入口，使论文消融和历史 clean_plus 探索版本分离。
_Avoid_: 复用 model_variant 表示论文消融, 手动组合零散开关

**前向加入消融**:
一种干净骨干精炼实验方式，从最小 BeMamba 基线开始按顺序加入候选模块，用来判断每一步是否带来主指标收益。
_Avoid_: 随机堆模块, 只报最终模型

**累积式前向加入**:
前向加入消融的默认阶段语义，每个阶段继承之前阶段已经加入的功能组，并且只新增当前阶段对应的功能组。
_Avoid_: 单模块开关, 非累积模块测试, 随机组合搜索

**筛选式前向加入**:
干净骨干精炼中的前向实验规则，按预设顺序测试功能组，但只有满足 Top-3 去留规则的功能组才进入后续当前最佳干净骨干。
_Avoid_: 失败模块继续堆叠, 全模块强制累积, 只报成功模块

**固定骨干容量**:
受控干净消融中的公平性约束，主消融表固定同一档 ResNet 特征输出层级，只比较干净骨干精炼功能组。
_Avoid_: stage 混合消融, 容量和模块同时变化, backbone 搜索

**固定主损失**:
受控干净消融中的训练目标约束，所有阶段保持同一个主分类损失，只有当对应功能组加入时才启用其必要辅助损失。
_Avoid_: 每阶段更换主损失, curriculum 主消融, loss-first 替代结构消融

**最终移除验证**:
一种干净骨干精炼实验方式，在最终干净骨干上逐个移除关键模块，用来验证该模块在完整组合中仍然有贡献。
_Avoid_: 单独模块测试, 重新搜索结构

**入选模块移除验证**:
最终移除验证的范围约束，只对通过筛选式前向加入并进入最终干净骨干的功能组做移除实验。
_Avoid_: 移除失败模块, 全模块移除表, 补充负结果重复验证

**顺序融合增强**:
干净骨干精炼中的融合模块组，用自适应门控为三种模态顺序的 MBMamba 输出分配权重。
_Avoid_: 单顺序替代, 缺失鲁棒融合

**预测头增强**:
干净骨干精炼中的分类头模块组，用注意力式预测头替代普通 MLP 预测头以增强融合 token 的读出能力。
_Avoid_: 波束重排, 分支集成

**分支监督增强**:
干净骨干精炼中的多分支输出模块组，利用融合分支和各模态分支的互补 logits 改善完整输入预测。
_Avoid_: 模型集成, 两阶段重排

**波束查询增强**:
干净骨干精炼中的波束感知模块组，让每个候选波束从融合 token 中取证并细化 logits。
_Avoid_: 候选重排, 普通分类头

**波束邻域增强**:
干净骨干精炼中的局部波束结构模块组，利用相邻 beam 的物理邻近关系进行 logits 校准。
_Avoid_: 标签平滑, 序关系先验

**候选重排增强**:
干净骨干精炼中的 Top-K 排名模块组，只在模型自身高置信候选集合内部调整排序，目标是把可恢复错误推入 Top-3。
_Avoid_: 两阶段模型, 全类别重分类

**训练期模态正则**:
干净骨干精炼中的训练期正则模块组，在完整输入训练时随机抑制 latent 模态流以缓解 clean 过拟合。
_Avoid_: 缺失增强, 测试时传感器缺失

**DMAF**:
The missing-aware robustness line that uses explicit availability masks to adapt multimodal fusion under incomplete sensor input.
_Avoid_: clean-plus, ordinary augmentation

**单基座鲁棒消融**:
动态掩码自适应融合的鲁棒性消融方式，所有缺失增强和缺失感知模块都接在同一个最终干净骨干上比较。
_Avoid_: 双基座消融, 最小基线鲁棒消融, 混合骨干对比

**缺失增强鲁棒基线**:
单基座鲁棒消融的起点，在最终干净骨干上加入训练期缺失模拟，但不使用显式缺失感知融合模块。
_Avoid_: 原始 BeMamba 缺失增强, 动态掩码自适应融合, 最小基线鲁棒起点

**掩码时序聚合**:
动态掩码自适应融合中的帧级聚合机制，使用可用性 mask 对时序特征进行加权汇总。
_Avoid_: 掩码编码器, 缺失增强, 普通时间求和

**掩码编码器**:
动态掩码自适应融合中的显式状态编码模块，把每个时间点或模态的缺失/存在状态注入到特征表示中。
_Avoid_: 掩码时序聚合, 可靠性估计器

**固定缺失增强策略**:
单基座鲁棒消融中的公平性约束，所有对比行使用相同的训练期缺失模拟设置，只改变模型侧缺失感知模块。
_Avoid_: 缺失强度搜索, 混合训练协议, 每行不同缺失配置

**鲁棒评估协议**:
动态掩码自适应融合的缺失输入评估范围，包含完整输入、随机帧缺失、连续帧缺失、整模态缺失和混合缺失。
_Avoid_: 单一缺失测试, 只报完整输入, 只报最优缺失场景

**鲁棒主结果指标**:
动态掩码自适应融合主结果表的指标范围，对原始基线、最终干净骨干、缺失增强鲁棒基线和完整鲁棒模型在五类鲁棒评估协议下报告 Top-3 Accuracy。
_Avoid_: 只报保持率, 只报最优缺失场景, 缺失增强和显式鲁棒模块混为一谈

**鲁棒模块移除验证**:
动态掩码自适应融合的最终移除实验，在完整鲁棒模型上逐个移除入选缺失感知模块以验证其组合内贡献。
_Avoid_: 干净骨干移除验证, 移除未入选鲁棒模块, 重新搜索缺失策略

**SC32 鲁棒模块消融**:
动态掩码自适应融合中用于判断缺失感知模块贡献的主文消融范围，优先在 SC32 上覆盖完整五类鲁棒评估协议。
_Avoid_: 三场景全量鲁棒模块消融, 只报最终鲁棒模型, 缺失协议筛选

**Missing-Aug**:
Training-time simulation of missing sensor data without requiring the model to consume explicit mask-aware fusion modules.
_Avoid_: DMAF, dropout

**Beam Neighborhood**:
The local ordering relationship between nearby beam indices where adjacent predictions can still be physically meaningful.
_Avoid_: class smoothing, label trick

**Clean Error Analysis**:
The diagnostic pass that studies clean-test prediction failures by target rank, beam distance, Top-K recoverability, and power loss.
_Avoid_: validation, robustness evaluation

**Candidate Reranking**:
The clean-plus step that reorders a model's own high-confidence beam candidates instead of trying to discover new candidates from scratch.
_Avoid_: image-source switching, backbone scaling

**Reranker Head**:
A lightweight reranking module attached to the Clean Backbone that adjusts candidate logits inside the same end-to-end model.
_Avoid_: standalone second-stage model, offline post-processing, separate dataset

**Standalone Reranker Model**:
A separate second-stage model trained to reorder candidates produced by a frozen or pre-trained first-stage beam predictor.
_Avoid_: reranker head, single-stage classifier, ordinary classification head

**Two-Stage Clean Reranker**:
The clean-data reranking experiment where a frozen Clean Backbone first produces a Top-K beam candidate set and a separately trained second-stage model learns to reorder only those candidates.
_Avoid_: end-to-end reranker head, missing-aware robustness, seed-only validation

**Selective Two-Stage Reranking**:
A conservative two-stage reranking setup that applies direct target-ranking supervision only to Stage-1 recoverable misses and uses preservation penalties to avoid damaging already-correct Stage-1 Top-3 predictions.
_Avoid_: stronger reranker everywhere, full-list reclassification, backbone retraining

**Reranker Negative Result**:
The SC32 clean finding that both ordinary and selective two-stage rerankers fail to improve over the frozen Stage-1 Top-3 result, with best performance occurring at the identity-like first epoch.
_Avoid_: reranker strength issue, more reranker epochs, reranker-only breakthrough

**Beam-Aware Candidate Reranking**:
The Clean Backbone innovation focus that uses beam neighborhood structure and power-distribution information to move plausible candidates into the Top-3.
_Avoid_: generic classifier head, larger CNN, missing-aware robustness

**Top-3-Aware Rerank Loss**:
A Clean Backbone training objective that directly penalizes cases where the correct beam is recoverable within the candidate set but remains outside the Top-3.
_Avoid_: seed validation, generic cross-entropy, Top-1-only loss

**Hard Top-3 Candidate Margin**:
The first Top-3-Aware Rerank Loss candidate: when the target beam is inside the candidate set but outside Top-3, require its logit to exceed the current third-ranked candidate by a margin; it only acts on recoverable candidate misses.
_Avoid_: soft reranking, full listwise sorting, Top-1 margin

**Recoverable Candidate Miss**:
A clean-test or training sample where the correct beam is inside the model's wider candidate set, such as Top-7, but outside the metric-critical Top-3.
_Avoid_: representation failure, Top-1 miss, missing-input error

**Top-7 Candidate Set**:
The fixed wider candidate set used by the first Clean Backbone reranking experiments, chosen to target recoverable Top-3 misses without broadening the experiment with candidate-width changes.
_Avoid_: Top-9 search, exhaustive candidate sweep, representation expansion

**Loss-First Clean Probe**:
A Clean Backbone experiment that changes Top-3-aligned supervision before changing the model body, used to test whether the current bottleneck is ranking supervision rather than representation capacity.
_Avoid_: backbone scaling, seed sweep, multi-change experiment

**clean_plus_v11**:
The first Loss-First Clean Probe: reuse the clean_plus_v10 structure and add the Hard Top-3 Candidate Margin as the new default Top-3-aligned training objective.
_Avoid_: new backbone, new fusion module, candidate-width sweep

**clean_plus_v12**:
The bounded soft reranker Clean Backbone probe: reuse the clean_plus_v10 structure but constrain candidate rerank logit deltas with a tanh-bounded residual.
_Avoid_: Hard Top-3 Candidate Margin, unbounded reranking, backbone scaling

**clean_plus_v13**:
The regularized candidate-ranking Clean Backbone probe: reuse the clean_plus_v10 candidate-reranking structure while reducing candidate beam-id memorization and anchoring ranking supervision to the measured power distribution.
_Avoid_: bounded reranking, hard margin, bigger backbone

**clean_plus_v14**:
The fusion-regularized Clean Backbone probe that keeps the clean_plus_v10 candidate-reranking structure and targets SC32 Clean Overfitting through Modality Feature Dropout.
_Avoid_: two-stage reranker, missing-aware robustness, image augmentation

**clean_plus_v15**:
The curriculum-regularized Clean Backbone probe that reuses the clean_plus_v14 structure while ramping Top-3 and candidate reranking auxiliary losses from easy-to-hard training.
_Avoid_: new backbone, DMAF, stronger dropout

**Modality Feature Dropout**:
A Clean Backbone regularization concept that suppresses whole latent modality streams during clean-data training to reduce modality-specific overfitting while preserving complete-input evaluation.
_Avoid_: Missing-Aug, DMAF mask, sensor dropout, test-time missing input

**Bounded Soft Candidate Reranking**:
A candidate reranking strategy that keeps Top-7 reranking but limits each candidate logit correction to a small fixed range, reducing train-split ranking overfit.
_Avoid_: hard margin, unconstrained logit rewriting, representation expansion

**Power-Anchored Candidate Supervision**:
A Top-3-oriented clean training signal that uses the measured power distribution's strongest beams as a stable candidate set, so ranking supervision is not only defined by the model's current Top-K predictions.
_Avoid_: hard target only, model-selected candidate loss, seed tuning

**Regularized Candidate Reranking**:
The clean candidate-reranking direction that keeps the Top-7 reranker but discourages it from relying too heavily on beam identity memorization during training.
_Avoid_: capacity expansion, random seed search, full backbone replacement

**Clean Rerank Loss Pairing**:
The clean_plus_v11 training setup that keeps candidate power-soft reranking for physical power-distribution consistency and adds Hard Top-3 Candidate Margin for direct Top-3 metric alignment.
_Avoid_: replacing all losses, single-purpose CE, unweighted loss stacking

**Conservative v11 Loss Defaults**:
The first clean_plus_v11 setting keeps candidate-rerank-weight at 0.05, sets top3-candidate-margin-weight to 0.03, and uses top3-candidate-margin 0.15.
_Avoid_: aggressive loss stacking, default weight sweep, removing v10 loss

**v11 First Probe Run**:
The first clean_plus_v11 experiment runs only SC32 with seed 11, no train-val merge, camera_data_mask images, dropout 0.25, candidate-rerank-weight 0.05, top3-candidate-margin-weight 0.03, and top3-candidate-margin 0.15.
_Avoid_: multi-scenario sweep, seed validation, simultaneous structure changes

**Clean-Plus Primary Metric**:
Top-3 Accuracy is the single primary metric for clean-data model-structure optimization and paper-facing comparison.
_Avoid_: composite score, Top-1-first tuning, DBA-first tuning

**严格论文口径**:
使用验证集选择 checkpoint，并且只在最终报告时使用测试集的完整输入评估协议。
_Avoid_: test-select, 探索口径, held-out 直接报最好点

**SC32 模块筛选**:
干净骨干精炼中用于判断候选模块去留的单场景消融范围，优先使用 SC32 的严格论文口径结果。
_Avoid_: 三场景全量模块搜索, seed sweep

**Top-3 去留规则**:
干净骨干精炼中决定候选模块是否进入最终干净骨干的规则，只以严格论文口径下的 Top-3 Accuracy 作为主判断。
_Avoid_: Top-1 决策, DBA 决策, 辅助指标替代主指标

**三场景骨干验证**:
最终干净骨干在 SC32、SC33、SC34 上的跨场景完整输入验证，用来证明入选骨干不是单场景偶然结果。
_Avoid_: 模块筛选, 探索口径复盘

**Paper-Facing Checkpoint Selection**:
The reporting protocol where a validation split selects the checkpoint and the test split is used only for final measurement.
_Avoid_: test-set early stopping, leaderboard-style checkpoint picking, seed validation

**Paper-Reported Validation Result**:
The original-paper style result when the held-out 20% split is called validation/evaluation and reported directly, without a clearly separate final test split.
_Avoid_: paper-safe test result, three-way validation protocol, final hidden test

**Paper-Aligned Sequence Order**:
The reproduction-audit sequence convention that follows the original BeMamba order assumptions before adding Clean-Plus changes.
_Avoid_: row-reverse shortcut, post-hoc evaluation mismatch, hidden implementation drift

**SC32 Clean Overfitting**:
The repeated SC32 clean-data pattern where train Top-3 continues rising after the best epoch while test Top-3 and test loss worsen, indicating capacity and regularization are now the main bottleneck.
_Avoid_: undertraining, seed search, bigger backbone

**SC32 Clean Two-Bottleneck Pattern**:
The current SC32 clean-data state where the model often recovers the correct beam in a wider candidate set but fails to place it in Top-3, while stronger ranking supervision quickly overfits the train split.
_Avoid_: single-cause diagnosis, seed search, backbone scaling

**Clean Regularization Ladder**:
The ordered Clean Backbone improvement path that first tests low-risk regularization controls, then moves to difficulty-aware supervision, and only later considers larger structural changes.
_Avoid_: seed sweep, direct DMAF work, bigger backbone first

**Cheap Regularization Probe**:
The first Clean Regularization Ladder step: use existing training controls to reduce clean-data overfitting without changing the Clean Backbone architecture or test protocol.
_Avoid_: new model variant, missing augmentation, test-select tuning

**Difficulty-Aware Curriculum**:
A later Clean Backbone direction that gradually strengthens supervision for hard or ambiguous beam samples instead of applying strong Top-3 ranking pressure from the beginning of training.
_Avoid_: standalone reranker, factor-graph branding, one-shot hard mining

**Curriculum Rerank Ramp**:
A Difficulty-Aware Curriculum mechanism that linearly ramps Top-3 margin and candidate rerank auxiliary losses while early training downweights ambiguous power-distribution samples.
_Avoid_: fixed loss weights, hard mining, larger model

**Train-Only Image Augmentation**:
Light camera-image augmentation used only during clean-data training to reduce SC32 overfitting; it is photometric-only and keeps deterministic resizing without random crop, blur, or geometry changes.
_Avoid_: Missing-Aug, test-time augmentation, sensor missing simulation, random crop

**Seed Validation**:
The stability check for a promising Clean Backbone candidate after it has already shown a meaningful Top-3 gain; it is not the primary path for discovering a large improvement.
_Avoid_: breakthrough search, exhaustive seed sweep, paper method

**Paper Baseline**:
The published BeMamba clean-data result used as an external reference point for reproduction and clean-plus comparison.
_Avoid_: local baseline, current champion

**中文实现说明**:
协作语言约定：说明代码实现、实验结论、运行方案和下一步计划时，默认用中文解释清楚；命令、参数名、文件路径、模型名和指标名可以保留原始写法。
_Avoid_: 英文复盘, 中英混杂解释
