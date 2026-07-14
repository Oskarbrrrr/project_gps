#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-base}"
SEED="${2:-11}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./logs_clean_ablation_sc32"
mkdir -p "${LOG_DIR}"

python -u train.py \
  --data-root ./Data/Multi_Modal \
  --split-root ./Data/splits_paper80_val \
  --output-root "./outputs_clean_ablation_sc32/${STAGE}" \
  --scenarios scenario32 \
  --image-subdir camera_data_mask \
  --gps-feature-mode physical_kinematic \
  --gps-future-steps 3 \
  --clean-ablation-stage "${STAGE}" \
  --selection-split val \
  --no-merge-trainval \
  --skip-final-test \
  --epochs 30 \
  --patience 10 \
  --batch-size 16 \
  --num-workers 8 \
  --optimizer adamw \
  --lr 1e-4 \
  --weight-decay 0.0 \
  --dropout 0.25 \
  --loss power_soft_ce \
  --soft-power-temperature 0.15 \
  --hard-loss-weight 0.6 \
  --ema-decay 0.99 \
  --ema-start-epoch 8 \
  --seed "${SEED}" \
  2>&1 | tee "${LOG_DIR}/${STAGE}_seed${SEED}_${RUN_ID}.log"
