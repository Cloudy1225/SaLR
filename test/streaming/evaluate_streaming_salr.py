import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import argparse
import json
import pickle

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from streaming_extractor import StreamingRepresentationExtractor
from train.salr_modules import (
    build_representation_cache_metadata,
    ensemble_predict,
    load_cached_representations,
    predict_layer_probabilities,
    representation_cache_path,
    save_cached_representations,
)
from utils.config import MODEL_CONFIGS


def load_general_salr(model_name, variant):
    model_path = f"../../train/probes/{model_name}_salr/{variant}/best_model.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_predictions_batch(representations, salr_artifact, device):
    dummy_labels = np.zeros(len(representations), dtype=np.int64)
    layer_artifacts = salr_artifact.get("selected_artifacts")
    if not layer_artifacts:
        by_layer = {int(a["layer"]): a for a in salr_artifact["layer_artifacts"]}
        layer_artifacts = [by_layer[int(layer)] for layer in salr_artifact["selected_layers"]]
    _y, _ds, prob_matrix = predict_layer_probabilities(
        layer_artifacts, representations, dummy_labels, None, batch_size=max(len(representations), 1), device=device
    )
    preds, probs = ensemble_predict(
        prob_matrix,
        method=salr_artifact.get("ensemble_method", "topk_average"),
        stacker=salr_artifact.get("stacker"),
        threshold=float(salr_artifact.get("threshold", 0.5)),
    )
    return preds, probs


def extract_streaming_prefix_representation(
    *,
    assistant_ids,
    extractor_factory,
    model_name,
    model_path,
    pooling_type,
    split_name,
    sample_id,
    position_name,
    cache_dir,
    use_cache,
    refresh_cache,
):
    token_fingerprint = " ".join(str(int(x)) for x in assistant_ids)
    metadata, cache_key = build_representation_cache_metadata(
        kind="streaming_prefix",
        model_name=model_name,
        model_path=model_path,
        rep_types=[pooling_type],
        texts=[token_fingerprint],
        split_name=f"{split_name}:{sample_id}:{position_name}",
        extra={"num_tokens": len(assistant_ids), "sample_id": str(sample_id), "position": position_name},
    )
    cache_path = representation_cache_path(cache_dir, model_name, cache_key)
    if use_cache:
        cached = load_cached_representations(cache_path, metadata, refresh_cache=refresh_cache)
        if cached is not None:
            return cached

    extractor = extractor_factory()
    reps = extractor.extract_prefixes_batch([assistant_ids], [[1] * len(assistant_ids)])
    if use_cache:
        save_cached_representations(cache_path, metadata, reps)
    return reps


def test_recall_at_multiple_positions(
    sample,
    extractor_factory,
    tokenizer,
    salr_artifact,
    device,
    model_name,
    model_path,
    split_name,
    cache_dir,
    use_cache,
    refresh_cache,
):
    input_ids_full = sample["input_ids"]
    unsafe_end = sample["unsafe_end_index"]

    messages = sample["message"]
    user_text = tokenizer.apply_chat_template([messages[0]], tokenize=False, add_generation_prompt=False)
    user_ids = tokenizer.encode(user_text, add_special_tokens=False)
    assistant_start_token = len(user_ids)

    total_tokens = len(input_ids_full)
    positions = {
        "timely": unsafe_end,
        "1-32": min(total_tokens, unsafe_end + 32),
        "33-64": min(total_tokens, unsafe_end + 64),
        "65-128": min(total_tokens, unsafe_end + 128),
        "129-256": min(total_tokens, unsafe_end + 256),
    }

    results = {}
    scores = {}
    for pos_name, pos_idx in positions.items():
        if pos_idx <= assistant_start_token:
            results[pos_name] = False
            scores[pos_name] = 0.0
            continue

        assistant_ids = input_ids_full[assistant_start_token:pos_idx]
        if len(assistant_ids) == 0:
            results[pos_name] = False
            scores[pos_name] = 0.0
            continue

        reps = extract_streaming_prefix_representation(
            assistant_ids=assistant_ids,
            extractor_factory=extractor_factory,
            model_name=model_name,
            model_path=model_path,
            pooling_type=salr_artifact["pooling_type"],
            split_name=split_name,
            sample_id=sample.get("unique_id", "unknown"),
            position_name=pos_name,
            cache_dir=cache_dir,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )
        preds, probs = get_predictions_batch(reps, salr_artifact, device)
        results[pos_name] = bool(preds[0] == 1)
        scores[pos_name] = float(probs[0])
    return results, scores


def evaluate_split(model_name, variant, split_name, device_str, cache_dir, use_cache, refresh_cache):
    print(f"\n{'=' * 80}")
    print(f"SaLR Streaming Test: {split_name}")
    print(f"{'=' * 80}")

    dataset = load_dataset("Qwen/Qwen3GuardTest", split=split_name)
    salr_artifact = load_general_salr(model_name, variant)
    device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu")

    model_config = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_path"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    extractor = None

    def ensure_extractor():
        nonlocal extractor
        if extractor is None:
            extractor = StreamingRepresentationExtractor(
                model_config["model_path"],
                device=device_str,
                batch_size=32,
                rep_types=[salr_artifact["pooling_type"]],
            )
            extractor.register_hooks()
        return extractor

    results = []
    try:
        for sample in tqdm(dataset, desc="Processing Streaming Detection"):
            position_detections, position_scores = test_recall_at_multiple_positions(
                sample,
                ensure_extractor,
                tokenizer,
                salr_artifact,
                device,
                model_name,
                model_config["model_path"],
                split_name,
                cache_dir,
                use_cache,
                refresh_cache,
            )
            results.append({
                "unique_id": sample["unique_id"],
                "unsafe_start_index": sample["unsafe_start_index"],
                "unsafe_end_index": sample["unsafe_end_index"],
                "detected_timely": position_detections["timely"],
                "detected_1_32": position_detections["1-32"],
                "detected_33_64": position_detections["33-64"],
                "detected_65_128": position_detections["65-128"],
                "detected_129_256": position_detections["129-256"],
                "score_timely": position_scores["timely"],
                "score_1_32": position_scores["1-32"],
                "score_33_64": position_scores["33-64"],
                "score_65_128": position_scores["65-128"],
                "score_129_256": position_scores["129-256"],
                "source": sample["source"],
                "unsafe_type": sample["unsafe_type"],
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if extractor is not None:
            extractor.remove_hooks()
            del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    detected_timely = sum(r["detected_timely"] for r in results)
    detected_grace_32 = sum(r["detected_timely"] or r["detected_1_32"] for r in results)
    detected_grace_64 = sum(r["detected_timely"] or r["detected_1_32"] or r["detected_33_64"] for r in results)
    detected_grace_128 = sum(r["detected_timely"] or r["detected_1_32"] or r["detected_33_64"] or r["detected_65_128"] for r in results)
    detected_grace_256 = sum(r["detected_timely"] or r["detected_1_32"] or r["detected_33_64"] or r["detected_65_128"] or r["detected_129_256"] for r in results)

    n = len(results)
    metrics = {
        "split": split_name,
        "num_samples": n,
        "timely_recall": detected_timely / n if n else 0,
        "grace_period_32_recall": detected_grace_32 / n if n else 0,
        "grace_period_64_recall": detected_grace_64 / n if n else 0,
        "grace_period_128_recall": detected_grace_128 / n if n else 0,
        "grace_period_256_recall": detected_grace_256 / n if n else 0,
        "timely_count": detected_timely,
        "grace_period_32_count": detected_grace_32,
        "grace_period_64_count": detected_grace_64,
        "grace_period_128_count": detected_grace_128,
        "grace_period_256_count": detected_grace_256,
    }

    print(f"\n{'=' * 80}")
    print("STREAMING DETECTION RESULTS")
    print(f"{'=' * 80}")
    print(f"Timely (unsafe_end):               {metrics['timely_recall']:.1%} ({metrics['timely_count']}/{n})")
    print(f"Grace Period (1-32 tokens):        {metrics['grace_period_32_recall']:.1%} ({metrics['grace_period_32_count']}/{n})")
    print(f"Grace Period (33-64 tokens):       {metrics['grace_period_64_recall']:.1%} ({metrics['grace_period_64_count']}/{n})")
    print(f"Grace Period (65-128 tokens):      {metrics['grace_period_128_recall']:.1%} ({metrics['grace_period_128_count']}/{n})")
    print(f"Grace Period (129-256 tokens):     {metrics['grace_period_256_recall']:.1%} ({metrics['grace_period_256_count']}/{n})")
    return metrics, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--variant", type=str, default="SaLR-full")
    parser.add_argument("--splits", type=str, nargs="+", default=["thinking_loc"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default="../../train/cache/representations")
    parser.add_argument("--use_cache", type=int, default=1)
    parser.add_argument("--refresh_cache", type=int, default=0)
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Variant: {args.variant}")
    print(f"Device: {args.device}")

    all_results = {}
    for split in args.splits:
        metrics, results = evaluate_split(
            args.model, args.variant, split, args.device,
            args.cache_dir, bool(args.use_cache), bool(args.refresh_cache),
        )
        all_results[split] = {"metrics": metrics, "results": results}

    output_dir = f"results/{args.model}_{args.variant}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/streaming_results.json"

    def convert_json(obj):
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_json(item) for item in obj]
        return obj

    with open(output_file, "w") as f:
        json.dump({"model": args.model, "variant": args.variant, "results": convert_json(all_results)}, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
