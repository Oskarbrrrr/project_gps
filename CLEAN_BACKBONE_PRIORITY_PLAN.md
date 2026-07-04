# Clean Backbone Priority Plan

Date: 2026-07-04

This note records the current clean-data improvement order. It is a working plan for the `Own Clean Baseline` / `Clean Backbone`, not the DMAF missing-input robustness line.

## Fixed Evaluation Rules

- Main metric: `Top-3 Accuracy` only.
- First target: `scenario32`, because it shows the clearest clean overfitting and the largest gap.
- Paper-facing protocol: `./Data/splits_paper80_val --selection-split val --no-merge-trainval`.
- Final comparison: val selects the checkpoint, test is reported only after selection.
- Default order: `--temporal-order forward --spatial-scan vertical`.
- Recommended SC32 image folder: `camera_data_mask`.
- Current base variant: `clean_plus_v14`.

## Priority

1. Cheap Regularization Probe.
   Use existing training controls before adding a new method: `weight_decay`, `modality-feature-dropout`, `backbone-lr-scale`, small `label_smoothing`, and optionally train-only `image_aug`.

2. Difficulty-Aware Curriculum.
   If Step 1 does not give a meaningful Top-3 gain, add a `clean_plus_v15`-style training schedule that ramps Top-3/rerank losses and reduces pressure on ambiguous samples early in training.

3. Lightweight Beam Prior.
   If curriculum helps but is still unstable, add a small beam-neighborhood or power-distribution prior. This should be a lightweight prior, not a full factor-graph system.

4. Defer Large External Models.
   Do not prioritize tabular foundation models or synthetic-prior pretraining for the clean backbone. They do not match the multimodal image/radar/lidar/GPS failure mode, and reranker-only work has already been a negative result.

## Step 1 Run Set

The first run set keeps the model architecture fixed and changes only regularization-related training controls:

| Run | Purpose | Key Change |
|---|---|---|
| `reg00_v14_ref` | Reproduce the current v14 paper-facing reference if needed | existing v14 defaults |
| `reg01_wd5e5` | Check mild L2/AdamW regularization | `--weight-decay 5e-5` |
| `reg02_wd1e4` | Check stronger but still conservative L2/AdamW regularization | `--weight-decay 1e-4` |
| `reg03_wd1e4_mfd015` | Test whether slightly stronger modality stream regularization helps | `--weight-decay 1e-4 --modality-feature-dropout 0.15` |
| `reg04_wd1e4_backbone05` | Test whether slowing pretrained backbones reduces overfit | `--weight-decay 1e-4 --backbone-lr-scale 0.5` |

## Step 1 Results

| Date | Run | Val Best Top-3 | Final Test Top-3 | Readout |
|---|---|---:|---:|---|
| 2026-07-05 | `reg01_wd5e5` | 84.59 | 82.66 | Mild AdamW/L2 regularization did not improve the paper-facing final test result; val selection became more optimistic than test. |
| 2026-07-05 | `reg02_wd1e4` | 84.75 | 83.31 | Stronger AdamW/L2 recovered the current v14 paper-facing Top-3 but did not improve the primary metric; auxiliary metrics improved. |
| 2026-07-05 | `reg03_wd1e4_mfd015` | 85.07 | 82.50 | Stronger modality stream dropout made validation look better but hurt final test Top-3, so `0.15` is too aggressive for SC32 paper-facing selection. |
| 2026-07-05 | `reg04_wd1e4_backbone05` | 84.27 | 82.02 | Slowing the pretrained modality backbones did not reduce the paper-facing generalization gap and hurt final test Top-3. |

Step 1 summary:

- Pure `weight_decay` did not beat the current v14 paper-facing Top-3.
- `weight_decay 1e-4` is the only tolerable cheap regularization setting, but it only matches the current Top-3 and mainly improves auxiliary metrics.
- Stronger modality stream dropout and lower backbone LR both hurt final test Top-3.
- The next improvement attempt should move to Difficulty-Aware Curriculum rather than continuing ordinary regularization sweeps.

Stop rule for Step 1:

- If no run beats v14 final test Top-3 `83.31%` under the same val-select protocol, move to Difficulty-Aware Curriculum.
- If one run improves Top-3 by at least `+0.3pp`, rerun the winner on SC33/SC34 before calling it a paper-facing Clean Backbone update.

## Step 2: Difficulty-Aware Curriculum

Step 2 starts after the cheap regularization probes failed to beat the v14 paper-facing Top-3 result.

`clean_plus_v15` keeps the `clean_plus_v14` model structure and changes only training-time auxiliary supervision:

- `rank_margin_weight` and `candidate_rerank_weight` are multiplied by a curriculum factor.
- The default curriculum starts at epoch 6 and ramps to full strength over 8 epochs.
- During the ramp, samples with flatter measured power distributions receive lower auxiliary ranking pressure.
- The base `power_soft_ce` loss remains active for every epoch and every sample.

First SC32 run:

```bash
python train.py \
  --data-root ./Data/Multi_Modal \
  --split-root ./Data/splits_paper80_val \
  --output-root ./outputs_clean_curriculum_probe_sc32/v15_ramp6_8 \
  --scenarios scenario32 \
  --image-subdir camera_data_mask \
  --lidar-representation count \
  --model-variant clean_plus_v15 \
  --selection-split val \
  --no-merge-trainval \
  --seed 11 \
  --epochs 30 \
  --patience 10 \
  --dropout 0.25 \
  --loss power_soft_ce \
  --soft-power-temperature 0.15 \
  --hard-loss-weight 0.5 \
  --ema-decay 0.99 \
  --ema-start-epoch 8 \
  --grad-clip-norm 1.0 \
  --rank-margin-weight 0.05 \
  --candidate-rerank-weight 0.05 \
  --candidate-rerank-topk 7 \
  --candidate-rerank-temperature 0.15 \
  --modality-feature-dropout 0.10 \
  --curriculum-start-epoch 6 \
  --curriculum-ramp-epochs 8 \
  --curriculum-start-factor 0.0 \
  --curriculum-min-sample-weight 0.25 \
  --temporal-order forward \
  --spatial-scan vertical
```
