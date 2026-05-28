#!/bin/bash

MODELS=(
    "qwen3-0.6b"
    # "llama3.2-1b"
)

VARIANT="SaLR-base"
DEVICE="cuda"
BATCH_SIZE=32
CACHE_DIR="../../train/cache/representations"
USE_CACHE=1
REFRESH_CACHE=0

echo "========================================"
echo "Evaluating SaLR Generalization on Qwen3GuardTest"
echo "========================================"
echo ""

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Model: $MODEL"
    echo "Variant: $VARIANT"
    echo "========================================"

    python evaluate_think_salr.py \
        --model $MODEL \
        --variant $VARIANT \
        --device $DEVICE \
        --batch_size $BATCH_SIZE \
        --cache_dir $CACHE_DIR \
        --use_cache $USE_CACHE \
        --refresh_cache $REFRESH_CACHE
done
