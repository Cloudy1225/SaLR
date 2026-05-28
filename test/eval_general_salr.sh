#!/bin/bash

MODEL="qwen3-0.6b"
VARIANT="SaLR-base"
DEVICE="cuda"
BATCH_SIZE=64
CACHE_DIR="../train/cache/representations"
USE_CACHE=1
REFRESH_CACHE=0
POOLING_TYPE="residual_mean"  # also supports mlp_mean

ARTIFACT_TYPE="layerwise_ensemble" # best | layerwise | layerwise_all | layerwise_ensemble

# Used for ARTIFACT_TYPE=layerwise
LAYER=0
# Used for ARTIFACT_TYPE=layerwise_ensemble.
ENSEMBLE_STRATEGY="layerwise_topk"  # saved | all | manual | layerwise_best | layerwise_topk | layerwise_threshold | topk_sweep
ENSEMBLE_LAYERS="0,4,8-12,20"       # all | used when ENSEMBLE_STRATEGY=manual
ENSEMBLE_TOP_K=28                    # used when ENSEMBLE_STRATEGY=layerwise_topk
ENSEMBLE_THRESHOLD=0.75             # used when ENSEMBLE_STRATEGY=layerwise_threshold
ENSEMBLE_METHOD="stacking"         # topk_average | stacking
TOPK_CANDIDATES="4,8,12,16,20,24,28" # default: 1,L/4,L/2,L
ENSEMBLE_DECISION_THRESHOLD=0.5
STACKING_C=1.0
STACKING_MAX_ITER=1000
STACKING_CLASS_WEIGHT="balanced"     # none | balanced

DATASETS=(
    "toxic_chat"
    # "openai_moderation"
    "aegis"
    "aegis2"
    "wildguard"
    "safe_rlhf"
    "beavertails"
)

echo "========================================"
echo "Evaluating SaLR layer-wise ensemble"
echo "========================================"
echo "Model: $MODEL"
echo "Variant: $VARIANT"
echo "Datasets: ${DATASETS[@]}"
echo ""

python evaluate_general_salr.py \
    --model "$MODEL" \
    --variant "$VARIANT" \
    --datasets "${DATASETS[@]}" \
    --artifact_type "$ARTIFACT_TYPE" \
    --pooling_type "$POOLING_TYPE" \
    --layer "$LAYER" \
    --ensemble_strategy $ENSEMBLE_STRATEGY \
    --ensemble_layers $ENSEMBLE_LAYERS \
    --ensemble_top_k $ENSEMBLE_TOP_K \
    --ensemble_threshold $ENSEMBLE_THRESHOLD \
    --ensemble_method $ENSEMBLE_METHOD \
    --threshold $ENSEMBLE_DECISION_THRESHOLD \
    --stacking_C $STACKING_C \
    --stacking_max_iter $STACKING_MAX_ITER \
    --stacking_class_weight $STACKING_CLASS_WEIGHT \
    --device $DEVICE \
    --batch_size $BATCH_SIZE \
    --cache_dir $CACHE_DIR \
    --use_cache $USE_CACHE \
    --refresh_cache $REFRESH_CACHE

echo ""
echo "Done!"
