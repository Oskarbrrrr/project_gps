# Project GPS Context

This context names the experiment lines and beam prediction concepts used in this repository. It exists to keep clean-data accuracy work separate from missing-data robustness work.

## Language

**Clean-Plus**:
The clean-data model-structure optimization line, trained and evaluated on complete sensor inputs with the goal of improving clean Top-K beam prediction.
_Avoid_: clean reproduction, baseline tuning, paper reproduction

**Own Clean Baseline**:
The paper's own complete-input baseline built on the reproduced BeMamba pipeline and allowed to include clearly stated clean-data structural improvements.
_Avoid_: Paper Baseline, unmodified BeMamba, robustness method

**Clean Backbone**:
The strong complete-input main model that provides clean Top-3 beam prediction ability before missing-aware robustness modules are added; it does not claim missing-input robustness by itself.
_Avoid_: ResNet-only backbone, DMAF, missing-aware module

**DMAF**:
The missing-aware robustness line that uses explicit availability masks to adapt multimodal fusion under incomplete sensor input.
_Avoid_: clean-plus, ordinary augmentation

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

**Bounded Soft Candidate Reranking**:
A candidate reranking strategy that keeps Top-7 reranking but limits each candidate logit correction to a small fixed range, reducing train-split ranking overfit.
_Avoid_: hard margin, unconstrained logit rewriting, representation expansion

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

**SC32 Clean Overfitting**:
The repeated SC32 clean-data pattern where train Top-3 continues rising after the best epoch while test Top-3 and test loss worsen, indicating capacity and regularization are now the main bottleneck.
_Avoid_: undertraining, seed search, bigger backbone

**SC32 Clean Two-Bottleneck Pattern**:
The current SC32 clean-data state where the model often recovers the correct beam in a wider candidate set but fails to place it in Top-3, while stronger ranking supervision quickly overfits the train split.
_Avoid_: single-cause diagnosis, seed search, backbone scaling

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
