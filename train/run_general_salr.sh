#!/bin/bash

MODEL="qwen3-4b"
DEVICE="cuda"
BATCH_SIZE=32
TRAIN_BATCH_SIZE=256
EVAL_BATCH_SIZE=512
VAL_RATIO=0.2
VARIANT="SaLR-base"
POOLING_TYPES="residual_mean"
ENSEMBLE_METHOD="stacking"      # auto: SaLR-stack=>stacking, otherwise topk_average
TOPK_CANDIDATES="default"   # default: 1,L/4,L/2,L
# TOPK_CANDIDATES="4,8,12,16,20,24,28"

GAMMA_BASE=1.0
A_S=0.0
A_H=0.2
LAMBDA_ALIGN=1.0
M_S=3.0
M_H=1.5
LAMBDA_ASYM=0.5
HIDDEN_DIM=512
DROPOUT=0.

EPOCHS=64
PATIENCE=10
CACHE_DIR="cache/representations"
USE_CACHE=1
REFRESH_CACHE=0

DATASETS=(
    "toxic_chat"
    "openai_moderation"
    "aegis"
    "aegis2"
    "wildguard"
    "safe_rlhf"
    "beavertails"
)

echo "========================================"
echo "Training SaLR: layer-wise + top-k ensemble"
echo "========================================"
echo "Model: $MODEL"
echo "Variant: $VARIANT"
echo "Datasets: ${DATASETS[@]}"
echo "Device: $DEVICE"
echo ""

python train_general_salr.py \
    --model "$MODEL" \
    --datasets "${DATASETS[@]}" \
    --variant "$VARIANT" \
    --batch_size "$BATCH_SIZE" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --pooling_types "$POOLING_TYPES" \
    --val_ratio "$VAL_RATIO" \
    --gamma_base "$GAMMA_BASE" \
    --a_s "$A_S" \
    --a_h "$A_H" \
    --lambda_align "$LAMBDA_ALIGN" \
    --m_s "$M_S" \
    --m_h "$M_H" \
    --lambda_asym "$LAMBDA_ASYM" \
    --hidden_dim "$HIDDEN_DIM" \
    --dropout "$DROPOUT" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --ensemble_method "$ENSEMBLE_METHOD" \
    --topk_candidates "$TOPK_CANDIDATES" \
    --cache_dir "$CACHE_DIR" \
    --use_cache "$USE_CACHE" \
    --refresh_cache "$REFRESH_CACHE" \
    --device "$DEVICE"

if [ $? -ne 0 ]; then
    echo "ERROR: train_general_salr.py failed"
    exit 1
fi

echo ""
echo "Done! Model saved to probes/${MODEL}_salr/${VARIANT}/"
