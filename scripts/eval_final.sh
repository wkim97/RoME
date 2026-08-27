#!/bin/bash
set -euo pipefail

MODE=${1:-cifar10_max}
GPU=${2:-0}
CKPT_DIR=${CKPT_DIR:-./checkpoints}
DATA_DIR=${DATA_DIR:-./datasets}
LOG_DIR=${LOG_DIR:-./logs}

DATASET=${MODE%_*}
STRATEGY=${MODE##*_}

case "$DATASET" in
    cifar10)     SUBDIR=ViT_CIFAR10;     CKPT=cifar10;     BS=64 ;;
    imagenet100) SUBDIR=ViT_ImageNet100; CKPT=imagenet100; BS=32 ;;
    imagenet1k)  SUBDIR=ViT_ImageNet1K;  CKPT=imagenet1k;  BS=32 ;;
    *) echo "unknown mode: $MODE" >&2; exit 1 ;;
esac

python evaluate.py \
    --dataset "$DATASET" \
    --ckpt "${CKPT_DIR}/${SUBDIR}/$(echo "$STRATEGY" | tr '[:lower:]' '[:upper:]')_apgd_${CKPT}.pth" \
    --data-dir "$DATA_DIR" \
    --batch-size "$BS" \
    --s 4 --b 6 \
    --gpu "$GPU" \
    --log-path "${LOG_DIR}/${MODE}.txt"
