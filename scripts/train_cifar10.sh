#!/bin/bash
set -euo pipefail

STRATEGY=${1:-max}
GPU=${2:-0}
DATA_DIR=${DATA_DIR:-./datasets}

CUDA_VISIBLE_DEVICES="$GPU" python train.py \
    --dataset cifar10 \
    --strategy "$STRATEGY" \
    --data-dir "$DATA_DIR" \
    --epochs 20 \
    --batch-size 64 \
    --lr 1e-3 \
    --lr-schedule warmup-linear \
    --warmup-epochs 4 --final-epochs 4 \
    --iters "15 15 15" \
    --num-lora 4 \
    --rank 16 --lora-alpha 16 \
    --s 4 --b 6 \
    --lam 0.1
