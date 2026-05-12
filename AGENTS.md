# AGENTS

## Project scope

This repository is a local reproduction workspace for:

- `BeMamba: Efficient Multimodal Sensing-Aided Beamforming via State Space Model`

The repo usually contains code only. Datasets, checkpoints, and experiment logs live on AutoDL unless explicitly copied back.

Core files:

- [train.py](/D:/code/project_gps/train.py)
- [src/dataset.py](/D:/code/project_gps/src/dataset.py)
- [src/model.py](/D:/code/project_gps/src/model.py)
- [src/utils.py](/D:/code/project_gps/src/utils.py)
- [visualize_lidar.py](/D:/code/project_gps/visualize_lidar.py)
- [AUTODL_WORKFLOW.md](/D:/code/project_gps/AUTODL_WORKFLOW.md)

## Current architecture

### Data

`MultimodalDataset` returns:

- `imgs`: `[5, 3, 256, 256]`
- `radars`: `[5, 2, 256, 256]`
- `lidars`: `[5, 1, 256, 256]`
- `gps`: `[2, 2]`
- `target`: scalar beam index in `[0, 63]`
- `power_vec`: `[64]`

### LiDAR

LiDAR preprocessing is no longer the original random virtual-point heuristic.

Current implementation:

1. Fixed-range BEV projection in `[-30 m, 30 m] x [-30 m, 30 m]`
2. Coarse motion cue from frame-to-frame point-count difference
3. Region filtering on coarse connected components
4. High-resolution local density enhancement for virtual points
5. Final LiDAR input is `max(base_bev, virtual_points)`

Important note:

- This is a quality-oriented approximation of the paper behavior, not a guaranteed line-by-line reconstruction of the original unpublished preprocessing code.

### Model

`src/model.py` now follows a clearer two-stage structure:

1. Per-modality CNN encoder
2. Per-modality temporal Mamba stack
3. GPS start/end token projection
4. Multi-order modal fusion with bidirectional Mamba stacks
5. Mean pooling + classification head

Main configuration lives in `BeMambaConfig`.

Key design choices:

- image branch: `ResNet34`
- radar branch: `ResNet18` with 2-channel stem
- lidar branch: `ResNet18` with 1-channel stem
- pooled spatial token grid: configurable, default `6 x 6`
- temporal modeling: patch-wise across 5 frames
- modal fusion: three orderings, averaged after fusion

### Training

`train.py` is now a configurable experiment entrypoint.

Features:

- CLI arguments for scenarios, optimizer, scheduler, model width, and runtime options
- fixed random seed
- `AdamW`
- warmup + cosine scheduler
- AMP support
- gradient clipping
- scenario-specific output directories under `./outputs`
- CSV logs and saved config snapshot per run
- focal loss with optional adaptive alpha weights

Example:

```bash
python train.py --scenarios scenario34 --batch-size 16 --epochs 80
```

## Collaboration rules for this repo

### Local vs AutoDL

Local workspace:

- edit code
- review diffs
- maintain git history
- inspect generated plots copied back from AutoDL

AutoDL:

- stores datasets
- runs training
- produces checkpoints and logs
- is the source of runtime validation

### Preferred workflow

1. Edit locally
2. Commit locally
3. Push to remote
4. Pull on AutoDL
5. Run experiment on AutoDL
6. Bring back logs or plots for review

Avoid mixing uncommitted manual edits on AutoDL with local git-driven changes. That caused version confusion already.

## Current reconstruction status

What is reasonably aligned:

- multimodal input structure
- 5-frame temporal setup
- radar two-channel loading
- fixed-range LiDAR BEV with motion-focused enhancement
- Mamba-based temporal and cross-modal fusion idea

What is still approximate:

- exact LiDAR virtual point generation from the paper
- exact GPS normalization strategy from the paper
- exact training hyperparameters from the paper
- exact split policy if the paper used a different official split

## Practical reminders

- Use `visualize_lidar.py` first when LiDAR behavior looks suspicious.
- If AutoDL git pull fails, check for local dirty files before debugging model behavior.
- The latest meaningful training/model refactor happened on `2026-05-12` and introduced:
  - the new `BeMambaConfig`
  - the refactored `BeMambaModel`
  - the configurable `train.py`

