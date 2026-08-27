#!/bin/bash
set -euo pipefail

STRATEGY=${1:-max}
GPU=${2:-0}
DATA_DIR=${DATA_DIR:-./datasets}

CUDA_VISIBLE_DEVICES="$GPU" python train.py \
    --dataset imagenet1k \
    --strategy "$STRATEGY" \
    --data-dir "$DATA_DIR" \
    --epochs 5 \
    --batch-size 128 \
    --lr 1e-4 \
    --lr-schedule warmup-linear \
    --warmup-epochs 1 --final-epochs 1 \
    --iters "15 15 15" \
    --num-lora 4 \
    --rank 16 --lora-alpha 16 \
    --s 4 --b 6 \
    --lam 0.1
