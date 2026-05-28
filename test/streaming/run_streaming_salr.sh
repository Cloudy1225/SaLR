#!/bin/bash

MODEL="qwen3-0.6b"
VARIANT="SaLR-base"
DEVICE="cuda"
CACHE_DIR="../../train/cache/representations"
USE_CACHE=0
REFRESH_CACHE=0

echo "========================================"
echo "SaLR Streaming Evaluation"
echo "========================================"
echo "Model: $MODEL"
echo "Variant: $VARIANT"
echo ""

python evaluate_streaming_salr.py \
    --model $MODEL \
    --variant $VARIANT \
    --splits thinking_loc \
    --device $DEVICE \
    --cache_dir $CACHE_DIR \
    --use_cache $USE_CACHE \
    --refresh_cache $REFRESH_CACHE

if [ $? -ne 0 ]; then
    echo "ERROR: Streaming evaluation failed"
    exit 1
fi

echo ""
echo "Done!"
