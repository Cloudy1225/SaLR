import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import pickle
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm

from train.preprocess import preprocess_dataset
from train.salr_modules import (
    binary_metrics,
    build_representation_cache_metadata,
    choose_layer_artifacts,
    convert_json,
    default_topk_candidates,
    ensemble_predict,
    evaluate_ensemble_artifact,
    fit_logistic_stacker,
    fit_topk_ensemble_on_validation,
    layer_slug,
    load_cached_representations,
    make_dataloader,
    parse_topk_candidates,
    predict_harm_model,
    predict_layer_probabilities,
    representation_cache_path,
    save_cached_representations,
)
from utils.config import MODEL_CONFIGS
from utils.model_hooks import Qwen3RepresentationExtractor


SPLIT_ORDER_SEED = 42


def collect_single_split(dataset_name: str, dataset_idx: int, split_name: str, val_ratio: float) -> Dict[str, object]:
    ds = preprocess_dataset(dataset_name, val_ratio)[split_name]
    texts = [str(x["text"]) for x in ds]
    labels = np.asarray([int(x["label"]) for x in ds], dtype=np.int64)
    dataset_ids = np.full(len(labels), int(dataset_idx), dtype=np.int64)
    return {"texts": texts, "labels": labels, "dataset_ids": dataset_ids}


def combine_split_parts(parts: Sequence[Dict[str, object]], *, shuffle: bool = True) -> Dict[str, object]:
    texts: List[str] = []
    labels_parts: List[np.ndarray] = []
    dataset_id_parts: List[np.ndarray] = []
    representations: List[dict] = []
    has_representations = all("representations" in part for part in parts)

    for part in parts:
        part_texts = list(part["texts"])
        texts.extend(part_texts)
        labels_parts.append(np.asarray(part["labels"], dtype=np.int64))
        dataset_id_parts.append(np.asarray(part["dataset_ids"], dtype=np.int64))
        if has_representations:
            part_reps = list(part["representations"])
            if len(part_reps) != len(part_texts):
                raise ValueError(
                    f"Representation/text length mismatch while combining split parts: "
                    f"{len(part_reps)} vs {len(part_texts)}"
                )
            representations.extend(part_reps)

    labels = np.concatenate(labels_parts) if labels_parts else np.asarray([], dtype=np.int64)
    dataset_ids = np.concatenate(dataset_id_parts) if dataset_id_parts else np.asarray([], dtype=np.int64)
    indices = np.random.RandomState(SPLIT_ORDER_SEED).permutation(len(texts)) if shuffle and texts else np.arange(len(texts))

    out: Dict[str, object] = {
        "texts": [texts[int(i)] for i in indices],
        "labels": labels[indices],
        "dataset_ids": dataset_ids[indices],
    }
    if has_representations:
        out["representations"] = [representations[int(i)] for i in indices]
    return out


def collect_split(dataset_names: Sequence[str], split_name: str, val_ratio: float) -> Dict[str, object]:
    parts = [
        collect_single_split(dataset_name, dataset_idx, split_name, val_ratio)
        for dataset_idx, dataset_name in enumerate(dataset_names)
    ]
    return combine_split_parts(parts, shuffle=True)

def extract_representations(texts, model_name, device, batch_size, rep_types):
    model_config = MODEL_CONFIGS[model_name]
    extractor = Qwen3RepresentationExtractor(model_config["model_path"], device=device, batch_size=batch_size, rep_types=list(rep_types))
    extractor.register_hooks()
    representations = []
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting representations", leave=False):
            batch_texts = texts[i:i + batch_size]
            with torch.no_grad():
                batch_reps = extractor.extract_batch(batch_texts)
                representations.extend(batch_reps)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        extractor.remove_hooks()
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return representations


def extract_or_load_representations(
    *,
    texts: Sequence[str],
    labels: Sequence[int],
    dataset_ids: Sequence[int],
    model_name: str,
    device: str,
    batch_size: int,
    rep_types: Sequence[str],
    split_name: str,
    cache_dir: Optional[str],
    use_cache: bool,
    refresh_cache: bool,
    extra: Optional[Dict[str, object]] = None,
) -> List[dict]:
    model_path = MODEL_CONFIGS[model_name]["model_path"]
    metadata, cache_key = build_representation_cache_metadata(
        kind="eval",
        model_name=model_name,
        model_path=model_path,
        rep_types=rep_types,
        texts=texts,
        labels=labels,
        dataset_ids=dataset_ids,
        split_name=split_name,
        extra=extra,
    )
    cache_path = representation_cache_path(cache_dir, model_name, cache_key)
    cached = load_cached_representations(cache_path, metadata, refresh_cache=refresh_cache) if use_cache else None
    if cached is not None:
        return cached
    reps = extract_representations(texts, model_name, device, batch_size, rep_types)
    if use_cache:
        save_cached_representations(cache_path, metadata, reps)
    return reps



def extract_or_load_split_representations(
    *,
    dataset_names: Sequence[str],
    split_name: str,
    model_name: str,
    device: str,
    batch_size: int,
    rep_types: Sequence[str],
    val_ratio: float,
    cache_dir: Optional[str],
    use_cache: bool,
    refresh_cache: bool,
) -> Dict[str, object]:
    """Load/extract per-dataset caches, then apply the train-time split order.

    The cache path for each part is:
    <cache_dir>/<model_name>/<rep_types>/<dataset>/<split>/representations.pkl
    This mirrors train/train_general_salr.py and avoids using a combined-dataset
    cache for per-dataset evaluation.
    """

    parts: List[Dict[str, object]] = []
    for dataset_idx, dataset_name in enumerate(dataset_names):
        part = collect_single_split(dataset_name, dataset_idx, split_name, val_ratio)
        reps = extract_or_load_representations(
            texts=part["texts"],
            labels=part["labels"],
            dataset_ids=part["dataset_ids"],
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            rep_types=rep_types,
            split_name=split_name,
            cache_dir=cache_dir,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            extra={"dataset": dataset_name, "val_ratio": val_ratio},
        )
        if len(reps) != len(part["labels"]):
            raise ValueError(
                f"Cached representation count mismatch for {dataset_name}/{split_name}: "
                f"{len(reps)} reps vs {len(part['labels'])} labels"
            )
        part = dict(part)
        part["representations"] = reps
        parts.append(part)
    return combine_split_parts(parts, shuffle=True)

def artifact_dir(model_name: str, variant: str) -> str:
    return f"../train/probes/{model_name}_salr/{variant}"


def load_best_salr(model_name: str, variant: str):
    path = os.path.join(artifact_dir(model_name, variant), "best_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_layerwise_artifacts(model_name: str, variant: str, pooling_type: str):
    combined_path = os.path.join(artifact_dir(model_name, variant), f"layerwise_models_{pooling_type}.pkl")
    if os.path.exists(combined_path):
        with open(combined_path, "rb") as f:
            payload = pickle.load(f)
        return payload["artifacts"]
    layer_dir = os.path.join(artifact_dir(model_name, variant), "layerwise", pooling_type)
    artifacts = []
    for name in sorted(os.listdir(layer_dir)):
        if name.startswith("layer_") and name.endswith(".pkl"):
            with open(os.path.join(layer_dir, name), "rb") as f:
                artifacts.append(pickle.load(f))
    if not artifacts:
        raise FileNotFoundError(f"No layerwise artifacts found for pooling_type={pooling_type}")
    return sorted(artifacts, key=lambda a: int(a["layer"]))


def evaluate_layerwise_dataset(model_name, dataset_name, artifact, args, device_obj):
    pooling_type = artifact["pooling_type"]
    layer = int(artifact["layer"])
    split = extract_or_load_split_representations(
        dataset_names=[dataset_name],
        split_name="test",
        model_name=model_name,
        device=args.device,
        batch_size=args.batch_size,
        rep_types=[pooling_type],
        val_ratio=args.val_ratio,
        cache_dir=args.cache_dir,
        use_cache=bool(args.use_cache),
        refresh_cache=bool(args.refresh_cache),
    )
    labels = split["labels"]
    dataset_ids = split["dataset_ids"]
    reps = split["representations"]
    loader = make_dataloader(reps, labels, dataset_ids, pooling_type, [layer], args.batch_size, shuffle=False)
    y_true, preds, ds_ids, probs = predict_harm_model(artifact["model"], loader, device_obj)
    metrics = binary_metrics(y_true, preds, probs, ds_ids)
    return {
        "dataset": dataset_name,
        "num_samples": int(len(labels)),
        "num_positive": int(np.sum(labels == 1)),
        "num_negative": int(np.sum(labels == 0)),
        **metrics,
    }


def build_custom_ensemble(model_name, variant, pooling_type, args, device_obj):
    layer_artifacts = load_layerwise_artifacts(model_name, variant, pooling_type)
    if args.ensemble_strategy == "topk_sweep":
        val = extract_or_load_split_representations(
            dataset_names=args.datasets,
            split_name="validation",
            model_name=model_name,
            device=args.device,
            batch_size=args.batch_size,
            rep_types=[pooling_type],
            val_ratio=args.val_ratio,
            cache_dir=args.cache_dir,
            use_cache=bool(args.use_cache),
            refresh_cache=bool(args.refresh_cache),
        )
        fit = fit_topk_ensemble_on_validation(
            layer_artifacts,
            val["representations"],
            val["labels"],
            val["dataset_ids"],
            args.batch_size,
            device_obj,
            method=args.ensemble_method,
            topk_candidates=parse_topk_candidates(args.topk_candidates, len(layer_artifacts)),
            stacking_C=args.stacking_C,
            stacking_max_iter=args.stacking_max_iter,
            stacking_class_weight=args.stacking_class_weight,
            seed=args.seed,
            threshold=args.threshold,
        )
        selected_layers = fit["selected_layers"]
        selected_artifacts = fit["selected_artifacts"]
        stacker = fit.get("stacker")
        val_f1 = fit["val_f1"]
        topk_sweep = fit["topk_sweep"]
        layer_ranking = fit["layer_ranking"]
    else:
        selected_artifacts = choose_layer_artifacts(
            layer_artifacts,
            strategy=args.ensemble_strategy,
            layers=args.ensemble_layers,
            top_k=args.ensemble_top_k,
            threshold=args.ensemble_threshold,
        )
        selected_layers = [int(a["layer"]) for a in selected_artifacts]
        stacker = None
        val_f1 = None
        topk_sweep = []
        layer_ranking = sorted(
            [{"layer": int(a["layer"]), "val_f1": float(a.get("val_f1", 0.0))} for a in layer_artifacts],
            key=lambda x: x["val_f1"],
            reverse=True,
        )
        if args.ensemble_method == "stacking":
            val = extract_or_load_split_representations(
                dataset_names=args.datasets,
                split_name="validation",
                model_name=model_name,
                device=args.device,
                batch_size=args.batch_size,
                rep_types=[pooling_type],
                val_ratio=args.val_ratio,
                cache_dir=args.cache_dir,
                use_cache=bool(args.use_cache),
                refresh_cache=bool(args.refresh_cache),
            )
            y_val, _ds_val, prob_matrix = predict_layer_probabilities(
                selected_artifacts,
                val["representations"],
                val["labels"],
                val["dataset_ids"],
                args.batch_size,
                device_obj,
            )
            stacker = fit_logistic_stacker(
                prob_matrix,
                y_val,
                C=args.stacking_C,
                max_iter=args.stacking_max_iter,
                class_weight=args.stacking_class_weight,
                seed=args.seed,
            )
    return {
        "artifact_type": "layerwise_ensemble",
        "method": "SaLR",
        "variant": variant,
        "pooling_type": pooling_type,
        "ensemble_method": args.ensemble_method,
        "selected_layers": selected_layers,
        "k": len(selected_layers),
        "threshold": args.threshold,
        "stacker": stacker,
        "layer_artifacts": layer_artifacts,
        "selected_artifacts": selected_artifacts,
        "val_f1": val_f1,
        "topk_sweep": topk_sweep,
        "layer_ranking": layer_ranking,
    }


def evaluate_ensemble_dataset(model_name, dataset_name, ensemble_artifact, args, device_obj):
    pooling_type = ensemble_artifact["pooling_type"]
    split = extract_or_load_split_representations(
        dataset_names=[dataset_name],
        split_name="test",
        model_name=model_name,
        device=args.device,
        batch_size=args.batch_size,
        rep_types=[pooling_type],
        val_ratio=args.val_ratio,
        cache_dir=args.cache_dir,
        use_cache=bool(args.use_cache),
        refresh_cache=bool(args.refresh_cache),
    )
    labels = split["labels"]
    dataset_ids = split["dataset_ids"]
    reps = split["representations"]
    print(f"\nEvaluating on: {dataset_name}")
    print("=" * 60)
    print(f"Test samples: {len(labels)}")
    print(f"Positive (harmful): {np.sum(labels == 1)}")
    print(f"Negative (safe): {np.sum(labels == 0)}")
    result = evaluate_ensemble_artifact(ensemble_artifact, reps, labels, dataset_ids, args.batch_size, device_obj, threshold=args.threshold)
    metrics = result["metrics"]
    print(f"F1 Macro:  {metrics['f1_macro']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    return {
        "dataset": dataset_name,
        "num_samples": int(len(labels)),
        "num_positive": int(np.sum(labels == 1)),
        "num_negative": int(np.sum(labels == 0)),
        **metrics,
    }


def summarize_and_save(all_results, output_path, payload_extra):
    avg_f1_macro = float(np.mean([r["f1_macro"] for r in all_results])) if all_results else 0.0
    avg_accuracy = float(np.mean([r["accuracy"] for r in all_results])) if all_results else 0.0
    avg_precision = float(np.mean([r["precision"] for r in all_results])) if all_results else 0.0
    avg_recall = float(np.mean([r["recall"] for r in all_results])) if all_results else 0.0
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<20} {'F1 Macro':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['dataset']:<20} {r['f1_macro']:<10.4f} {r['accuracy']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f}")
    print("-" * 80)
    print(f"{'AVERAGE':<20} {avg_f1_macro:<10.4f} {avg_accuracy:<10.4f} {avg_precision:<10.4f} {avg_recall:<10.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(convert_json({
            **payload_extra,
            "results": all_results,
            "averages": {
                "f1_macro": avg_f1_macro,
                "accuracy": avg_accuracy,
                "precision": avg_precision,
                "recall": avg_recall,
            },
        }), f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--variant", type=str, default="SaLR-full")
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--artifact_type", type=str, default="best", choices=["best", "layerwise", "layerwise_all", "layerwise_ensemble"])
    parser.add_argument("--pooling_type", type=str, default="residual_mean")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--ensemble_strategy", type=str, default="saved", choices=["saved", "all", "manual", "layerwise_best", "layerwise_topk", "layerwise_threshold", "topk_sweep"])
    parser.add_argument("--ensemble_layers", type=str, default="all")
    parser.add_argument("--ensemble_top_k", type=int, default=1)
    parser.add_argument("--ensemble_threshold", type=float, default=0.0)
    parser.add_argument("--ensemble_method", type=str, default="topk_average", choices=["topk_average", "stacking"])
    parser.add_argument("--topk_candidates", type=str, default="default")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--stacking_C", type=float, default=1.0)
    parser.add_argument("--stacking_max_iter", type=int, default=1000)
    parser.add_argument("--stacking_class_weight", type=str, default="balanced")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--cache_dir", type=str, default="cache/representations")
    parser.add_argument("--use_cache", type=int, default=1)
    parser.add_argument("--refresh_cache", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.pooling_type.endswith("_mean"):
        raise ValueError(f"Only mean pooling is supported. Got {args.pooling_type!r}.")
    device_obj = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    out_dir = artifact_dir(args.model, args.variant)

    if args.artifact_type == "layerwise_all":
        layer_artifacts = load_layerwise_artifacts(args.model, args.variant, args.pooling_type)
        all_layer_summaries = []
        for artifact in layer_artifacts:
            layer_results = []
            for dataset in args.datasets:
                res = evaluate_layerwise_dataset(args.model, dataset, artifact, args, device_obj)
                res["model"] = args.model
                res["variant"] = args.variant
                res["artifact_type"] = "layerwise"
                res["layer"] = int(artifact["layer"])
                layer_results.append(res)
            mean_f1 = float(np.mean([r["f1_macro"] for r in layer_results]))
            all_layer_summaries.append({"layer": int(artifact["layer"]), "mean_f1_macro": mean_f1, "results": layer_results})
        all_layer_summaries.sort(key=lambda x: x["mean_f1_macro"], reverse=True)
        output_path = os.path.join(out_dir, f"layerwise_all_eval_{args.pooling_type}.json")
        with open(output_path, "w") as f:
            json.dump(convert_json({"model": args.model, "variant": args.variant, "pooling_type": args.pooling_type, "layers": all_layer_summaries}), f, indent=2)
        print(f"Saved layerwise_all results to {output_path}")
        print("Top layers:")
        for row in all_layer_summaries[:10]:
            print(f"  layer {row['layer']:2d}: mean_f1_macro={row['mean_f1_macro']:.4f}")
        return

    if args.artifact_type == "layerwise":
        if args.layer is None:
            raise ValueError("--artifact_type layerwise requires --layer")
        artifacts = load_layerwise_artifacts(args.model, args.variant, args.pooling_type)
        artifact = next((a for a in artifacts if int(a["layer"]) == int(args.layer)), None)
        if artifact is None:
            raise ValueError(f"No saved layerwise artifact for layer {args.layer}")
        all_results = []
        for dataset in args.datasets:
            res = evaluate_layerwise_dataset(args.model, dataset, artifact, args, device_obj)
            res.update({"model": args.model, "variant": args.variant, "artifact_type": "layerwise", "layer": int(args.layer)})
            all_results.append(res)
        output_path = os.path.join(out_dir, f"eval_results_layerwise_{args.pooling_type}_layer_{args.layer}.json")
        summarize_and_save(all_results, output_path, {"model": args.model, "variant": args.variant, "artifact_type": "layerwise", "layer": int(args.layer)})
        return

    if args.artifact_type == "best" or (args.artifact_type == "layerwise_ensemble" and args.ensemble_strategy == "saved"):
        ensemble_artifact = load_best_salr(args.model, args.variant)
        print(
            f"Loaded best SaLR ensemble: pooling={ensemble_artifact['pooling_type']} "
            f"method={ensemble_artifact.get('ensemble_method')} layers={ensemble_artifact.get('selected_layers')}"
        )
    else:
        ensemble_artifact = build_custom_ensemble(args.model, args.variant, args.pooling_type, args, device_obj)
        print(
            f"Built custom ensemble: pooling={args.pooling_type} method={args.ensemble_method} "
            f"layers={ensemble_artifact['selected_layers']}"
        )

    all_results = []
    for dataset in args.datasets:
        res = evaluate_ensemble_dataset(args.model, dataset, ensemble_artifact, args, device_obj)
        res.update({
            "model": args.model,
            "variant": args.variant,
            "artifact_type": "layerwise_ensemble",
            "ensemble_method": ensemble_artifact.get("ensemble_method"),
            "selected_layers": ensemble_artifact.get("selected_layers"),
            "k": len(ensemble_artifact.get("selected_layers", [])),
        })
        all_results.append(res)

    if args.artifact_type == "best" or args.ensemble_strategy == "saved":
        suffix = "best"
    else:
        suffix = f"{args.ensemble_strategy}_{args.ensemble_method}_layers_{layer_slug(ensemble_artifact['selected_layers'])}"
    output_path = os.path.join(out_dir, f"eval_results_{suffix}.json")
    summarize_and_save(
        all_results,
        output_path,
        {
            "model": args.model,
            "variant": args.variant,
            "artifact_type": "layerwise_ensemble",
            "ensemble_method": ensemble_artifact.get("ensemble_method"),
            "selected_layers": ensemble_artifact.get("selected_layers"),
            "topk_sweep": ensemble_artifact.get("topk_sweep"),
            "layer_ranking": ensemble_artifact.get("layer_ranking"),
        },
    )


if __name__ == "__main__":
    main()
