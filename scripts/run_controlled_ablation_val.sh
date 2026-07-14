#!/usr/bin/env bash
set -euo pipefail

FAMILY="${1:-clean}"
STAGE="${2:-}"
SCENARIO="${3:-scenario33}"
SEED="${4:-11}"

case "${SCENARIO}" in
  scenario32)
    IMAGE_SUBDIR="camera_data_mask"
    ;;
  scenario33|scenario34)
    IMAGE_SUBDIR="camera_data_mask_yolo"
    ;;
  *)
    echo "Unsupported scenario: ${SCENARIO}" >&2
    exit 2
    ;;
esac

case "${FAMILY}" in
  clean)
    STAGE="${STAGE:-base}"
    STAGE_ARGS=(--model-variant bemamba --clean-ablation-stage "${STAGE}")
    ;;
  dmaf)
    STAGE="${STAGE:-missing_aug}"
    STAGE_ARGS=(
      --model-variant clean_plus_v14
      --dmaf-ablation-stage "${STAGE}"
      --eval-selection-missing
      --missing-frame-prob 0.10
      --missing-burst-prob 0.05
      --missing-modality-prob 0.05
      --missing-seed 7
    )
    ;;
  *)
    echo "Usage: $0 {clean|dmaf} STAGE {scenario32|scenario33|scenario34} [SEED]" >&2
    exit 2
    ;;
esac

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="./outputs_controlled_ablation/${FAMILY}/${STAGE}/seed${SEED}"
LOG_DIR="./logs_controlled_ablation/${FAMILY}/${SCENARIO}"
mkdir -p "${LOG_DIR}"

python -u train.py \
  --data-root ./Data/Multi_Modal \
  --split-root ./Data/splits_paper80_val \
  --output-root "${OUTPUT_ROOT}" \
  --scenarios "${SCENARIO}" \
  --image-subdir "${IMAGE_SUBDIR}" \
  --gps-feature-mode physical_kinematic \
  --gps-future-steps 3 \
  "${STAGE_ARGS[@]}" \
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
  --temporal-order forward \
  --spatial-scan vertical \
  --seed "${SEED}" \
  2>&1 | tee "${LOG_DIR}/${STAGE}_seed${SEED}_${RUN_ID}.log"
