import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import json
import pickle

import numpy as np
import torch
from tqdm import tqdm

from preprocess_think import preprocess_think_by_backbone
from train.salr_modules import (
    binary_metrics,
    build_representation_cache_metadata,
    ensemble_predict,
    load_cached_representations,
    predict_layer_probabilities,
    representation_cache_path,
    save_cached_representations,
)
from utils.config import MODEL_CONFIGS
from utils.model_hooks import Qwen3RepresentationExtractor


def load_general_salr(model_name, variant):
    model_path = f"../../train/probes/{model_name}_salr/{variant}/best_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


def extract_representations(
    texts,
    labels,
    backbone_name,
    model_name,
    device,
    batch_size,
    pooling_type,
    cache_dir="../../train/cache/representations",
    use_cache=True,
    refresh_cache=False,
):
    model_config = MODEL_CONFIGS[model_name]
    metadata, cache_key = build_representation_cache_metadata(
        kind="think_eval",
        model_name=model_name,
        model_path=model_config["model_path"],
        rep_types=[pooling_type],
        texts=texts,
        labels=labels,
        dataset_ids=np.zeros(len(texts), dtype=np.int64),
        split_name=f"Qwen3GuardTest_thinking_{backbone_name}",
        extra={"backbone": backbone_name},
    )
    cache_path = representation_cache_path(cache_dir, model_name, cache_key)
    if use_cache:
        cached = load_cached_representations(cache_path, metadata, refresh_cache=refresh_cache)
        if cached is not None:
            return cached

    extractor = Qwen3RepresentationExtractor(
        model_config["model_path"],
        device=device,
        batch_size=batch_size,
        rep_types=[pooling_type],
    )
    extractor.register_hooks()
    representations = []
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting", leave=False):
            batch_texts = texts[i:i + batch_size]
            with torch.no_grad():
                representations.extend(extractor.extract_batch(batch_texts))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        extractor.remove_hooks()
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if use_cache:
        save_cached_representations(cache_path, metadata, representations)
    return representations


def get_harm_predictions(representations, labels, salr_artifact, batch_size, device):
    layer_artifacts = salr_artifact.get("selected_artifacts")
    if not layer_artifacts:
        by_layer = {int(a["layer"]): a for a in salr_artifact["layer_artifacts"]}
        layer_artifacts = [by_layer[int(layer)] for layer in salr_artifact["selected_layers"]]
    _y, _ds, prob_matrix = predict_layer_probabilities(layer_artifacts, representations, labels, None, batch_size, device)
    preds, probs = ensemble_predict(
        prob_matrix,
        method=salr_artifact.get("ensemble_method", "topk_average"),
        stacker=salr_artifact.get("stacker"),
        threshold=float(salr_artifact.get("threshold", 0.5)),
    )
    return preds, probs


def evaluate_backbone(backbone_name, test_data, model_name, salr_artifact, device_str, batch_size, cache_dir, use_cache, refresh_cache):
    texts = [sample["text"] for sample in test_data]
    labels = np.asarray([sample["label"] for sample in test_data], dtype=np.int64)
    device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu")

    representations = extract_representations(
        texts, labels, backbone_name, model_name, device_str, batch_size, salr_artifact["pooling_type"],
        cache_dir=cache_dir, use_cache=use_cache, refresh_cache=refresh_cache,
    )
    predictions, probs = get_harm_predictions(representations, labels, salr_artifact, batch_size, device)
    metrics = binary_metrics(labels, predictions, probs, np.zeros(len(labels), dtype=np.int64))

    print(f"  F1 Macro: {metrics['f1_macro']:.4f}, Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    return {
        "backbone": backbone_name,
        "num_samples": len(texts),
        "num_positive": int(np.sum(labels == 1)),
        "num_negative": int(np.sum(labels == 0)),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--variant", type=str, default="SaLR-full")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cache_dir", type=str, default="../../train/cache/representations")
    parser.add_argument("--use_cache", type=int, default=1)
    parser.add_argument("--refresh_cache", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading SaLR for {args.model} / {args.variant}...")
    salr_artifact = load_general_salr(args.model, args.variant)
    print(f"Loaded: pooling={salr_artifact['pooling_type']}, method={salr_artifact.get('ensemble_method')}, layers={salr_artifact.get('selected_layers')}")

    print("\nPreprocessing think models by backbone...")
    backbone_datasets = preprocess_think_by_backbone()

    all_results = []
    for backbone in ["GLM", "Qwen3", "Deepseek"]:
        if backbone not in backbone_datasets:
            continue
        print(f"\n{'=' * 60}")
        print(f"Backbone: {backbone}")
        print(f"{'=' * 60}")
        result = evaluate_backbone(
            backbone, backbone_datasets[backbone]["test"], args.model, salr_artifact, args.device, args.batch_size,
            args.cache_dir, bool(args.use_cache), bool(args.refresh_cache),
        )
        result["model"] = args.model
        result["variant"] = args.variant
        all_results.append(result)

    avg_f1_macro = np.mean([r["f1_macro"] for r in all_results])
    avg_accuracy = np.mean([r["accuracy"] for r in all_results])
    avg_precision = np.mean([r["precision"] for r in all_results])
    avg_recall = np.mean([r["recall"] for r in all_results])

    print("=" * 80)
    for r in all_results:
        print(f"\n{r['backbone']}:")
        print(f"  F1 Macro:  {r['f1_macro']:.4f}")
        print(f"  Accuracy:  {r['accuracy']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall:    {r['recall']:.4f}")

    print("\nOverall Average:")
    print(f"  F1 Macro:  {avg_f1_macro:.4f}")
    print(f"  Accuracy:  {avg_accuracy:.4f}")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  Recall:    {avg_recall:.4f}")

    output_dir = f"../../train/probes/{args.model}_salr/{args.variant}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/think_eval_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "variant": args.variant,
            "results": all_results,
            "averages": {
                "f1_macro": float(avg_f1_macro),
                "accuracy": float(avg_accuracy),
                "precision": float(avg_precision),
                "recall": float(avg_recall),
            },
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
