# SaLR: Safety Knowledge Is Already There

**SaLR** (**Sa**fe-aware **L**atent **R**eshaping) is a lightweight representation-reshaping framework for harmful content detection using **frozen LLM representations**.

Instead of fine-tuning large guard models, SaLR reshapes internal latent representations of pretrained LLMs to improve harmfulness separability, enabling efficient and strong safety detection with only lightweight trainable modules.

## Project Structure

```
.
├── README.md                         # Project documentation
├── train
│   ├── preprocess.py                 # Dataset loading and preprocessing
│   ├── train_general_salr.py         # Main SaLR training script
│   ├── salr_modules.py               # SaLR modules and classifier definitions
│   └── run_general_salr.sh           # Example training script
├── test
│   ├── evaluate_general_salr.py      # Main evaluation script
│   ├── eval_general_salr.sh          # Example evaluation script
│   ├── generalization
│   │   ├── preprocess_think.py       # Preprocess Think benchmark
│   │   ├── evaluate_think_salr.py    # Generalization evaluation on Think
│   │   └── eval_gen_salr.sh          # Example Think evaluation script
│   └── streaming
│       ├── streaming_extractor.py    # Prefix representation extraction
│       ├── evaluate_streaming_salr.py# Streaming detection evaluation
│       ├── case_study_token_scores.py# Token-level case study visualization
│       └── run_streaming_salr.sh     # Example streaming evaluation script
└── utils
    ├── config.py                     # Model/dataset/path configuration
    └── model_hooks.py                # Hooks for extracting LLM representations
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
cd train
bash run_general_salr.sh
```

Equivalent explicit command:

```bash
python train_general_salr.py \
  --model qwen3-4b \
  --variant SaLR-full \
  --datasets toxic_chat openai_moderation aegis aegis2 wildguard safe_rlhf beavertails \
  --pooling_types residual_mean \
  --ensemble_method stacking \
  --topk_candidates default \
  --use_cache 1
```

## Evaluation

Evaluate the saved best ensemble:

```bash
cd test
python evaluate_general_salr.py \
  --model qwen3-4b \
  --variant SaLR-full \
  --artifact_type best \
  --datasets toxic_chat aegis aegis2 wildguard safe_rlhf beavertails
```

Evaluate one saved layer-wise classifier:

```bash
python evaluate_general_salr.py \
  --model qwen3-4b \
  --variant SaLR-full \
  --artifact_type layerwise \
  --pooling_type residual_mean \
  --layer 12 \
  --datasets toxic_chat aegis
```

Evaluate all saved layer-wise classifiers:

```bash
python evaluate_general_salr.py \
  --model qwen3-4b \
  --variant SaLR-full \
  --artifact_type layerwise_all \
  --pooling_type residual_mean \
  --datasets toxic_chat aegis
```

Build a custom ensemble at evaluation time:

```bash
python evaluate_general_salr.py \
  --model qwen3-4b \
  --variant SaLR-full \
  --artifact_type layerwise_ensemble \
  --pooling_type residual_mean \
  --ensemble_strategy manual \
  --ensemble_layers "0,4,8-12,20" \
  --ensemble_method topk_average \
  --datasets toxic_chat aegis
```

Fit a stacking ensemble on validation splits and evaluate on test splits:

```bash
python evaluate_general_salr.py \
  --model qwen3-4b \
  --variant SaLR-full \
  --artifact_type layerwise_ensemble \
  --pooling_type residual_mean \
  --ensemble_strategy topk_sweep \
  --ensemble_method stacking \
  --datasets toxic_chat aegis aegis2 wildguard safe_rlhf beavertails
```

## Generalization Evaluation

```bash
cd test/generalization
bash eval_gen_salr.sh
```

This evaluates SaLR on the Think benchmark.

## Streaming Detection

```bash
cd test/streaming
bash run_streaming_salr.sh
```

Streaming evaluation loads the saved best model, computes prefix-level harmfulness probabilities, and aggregates selected layer-wise predictions.

## Representation Caching

Frozen LLM representations are cached using readable dataset-based paths:

```
<cache_dir>/<model_name>/<rep_types>/<dataset>/<split>/representations.pkl
```

Example:

```
cache/representations/qwen3-4b/residual_mean/toxic_chat/test/representations.pkl
```

This allows:

- training/evaluation cache sharing
- avoiding repeated hidden-state extraction
- reproducible representation ordering

Useful options:

```bash
--cache_dir cache/representations
--use_cache 1
--refresh_cache 0
```

Use `--refresh_cache 1` to recompute representations.
