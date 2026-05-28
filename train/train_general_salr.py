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

from utils.config import MODEL_CONFIGS
from utils.model_hooks import Qwen3RepresentationExtractor
from train.preprocess import preprocess_dataset
from train.salr_modules import (
    build_representation_cache_metadata,
    choose_layer_artifacts,
    convert_json,
    estimate_harm_directions,
    evaluate_ensemble_artifact,
    fit_topk_ensemble_on_validation,
    get_variant_config,
    infer_feature_dim,
    layer_slug,
    load_cached_representations,
    make_dataloader,
    make_layer_classifier,
    materialize_layer_tensors,
    parse_topk_candidates,
    predict_harm_model,
    predict_harm_model_tensors,
    representation_cache_path,
    save_cached_representations,
    slice_direction_info,
    train_harm_model,
    train_harm_model_tensors,
)


SPLIT_ORDER_SEED = 42


def collect_single_split_texts(dataset_name: str, dataset_idx: int, split_name: str, val_ratio: float) -> Dict[str, object]:
    """Collect one dataset split in its native deterministic order.

    Cache files are stored per dataset/split in this order. The combined
    training/evaluation tensors are shuffled afterwards with the same fixed
    permutation as before, so model behaviour stays deterministic while cache
    files remain reusable by per-dataset evaluation.
    """

    dataset = preprocess_dataset(dataset_name, val_ratio)
    split_data = dataset[split_name]
    texts = [str(item["text"]) for item in split_data]
    labels = np.asarray([int(item["label"]) for item in split_data], dtype=np.int64)
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


def collect_split_texts(dataset_names: Sequence[str], split_name: str, val_ratio: float) -> Dict[str, object]:
    parts = [
        collect_single_split_texts(dataset_name, dataset_idx, split_name, val_ratio)
        for dataset_idx, dataset_name in enumerate(dataset_names)
    ]
    return combine_split_parts(parts, shuffle=True)

def extract_representations_with_existing_extractor(extractor, texts: Sequence[str], batch_size: int, desc: str) -> List[dict]:
    representations: List[dict] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = list(texts[i:i + batch_size])
        with torch.no_grad():
            batch_reps = extractor.extract_batch(batch_texts)
            representations.extend(batch_reps)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return representations


def extract_all_representations(
    model_name: str,
    datasets: Sequence[str],
    device: str,
    batch_size: int,
    rep_types: Sequence[str],
    val_ratio: float,
    seed: int,
    cache_dir: Optional[str] = "cache/representations",
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> Dict[str, dict]:
    model_config = MODEL_CONFIGS[model_name]
    model_path = model_config["model_path"]
    out: Dict[str, dict] = {}
    extractor = None

    def ensure_extractor():
        nonlocal extractor
        if extractor is None:
            extractor = Qwen3RepresentationExtractor(model_path, device=device, batch_size=batch_size, rep_types=list(rep_types))
            extractor.register_hooks()
        return extractor

    try:
        for split_name in ["train", "validation", "test"]:
            print(f"\nProcessing {split_name} split...")
            split_parts: List[Dict[str, object]] = []

            for dataset_idx, dataset_name in enumerate(datasets):
                part = collect_single_split_texts(dataset_name, dataset_idx, split_name, val_ratio)
                print(f"  {dataset_name}: {len(part['labels'])} samples")

                # Store cache per dataset/split, not per current dataset list. This
                # keeps the path readable and lets evaluate_general_salr.py reuse
                # the exact train-time cache for an individual dataset test split.
                metadata, cache_key = build_representation_cache_metadata(
                    kind="original",
                    model_name=model_name,
                    model_path=model_path,
                    rep_types=rep_types,
                    texts=part["texts"],
                    labels=part["labels"],
                    dataset_ids=part["dataset_ids"],
                    split_name=split_name,
                    extra={"dataset": dataset_name, "val_ratio": val_ratio},
                )
                cache_path = representation_cache_path(cache_dir, model_name, cache_key)
                part_reps = load_cached_representations(cache_path, metadata, refresh_cache=refresh_cache) if use_cache else None
                if part_reps is None:
                    part_reps = extract_representations_with_existing_extractor(
                        ensure_extractor(), part["texts"], batch_size, desc=f"Extracting {dataset_name}/{split_name}"
                    )
                    if use_cache:
                        save_cached_representations(cache_path, metadata, part_reps)
                if len(part_reps) != len(part["labels"]):
                    raise ValueError(
                        f"Cached representation count mismatch for {dataset_name}/{split_name}: "
                        f"{len(part_reps)} reps vs {len(part['labels'])} labels"
                    )
                part = dict(part)
                part["representations"] = part_reps
                split_parts.append(part)

            split = combine_split_parts(split_parts, shuffle=True)
            original_reps = list(split["representations"])
            print(f"Original samples in {split_name}: {len(split['labels'])}")

            out[split_name] = {
                "representations_original": original_reps,
                "representations": list(original_reps),
                "texts": split["texts"],
                "labels_original": split["labels"],
                "labels": np.asarray(split["labels"], dtype=np.int64),
                "dataset_ids_original": split["dataset_ids"],
                "dataset_ids": np.asarray(split["dataset_ids"], dtype=np.int64),
                "num_layers": model_config["num_layers"],
            }
    finally:
        if extractor is not None:
            extractor.remove_hooks()
            del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def train_one_layer(all_reps, pooling_type, direction_info, layer, args, variant_config, device):
    model = make_layer_classifier(
        direction_info=direction_info,
        layer=layer,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        gamma_base=args.gamma_base,
        a_s=args.a_s,
        a_h=args.a_h,
        lambda_align=args.lambda_align,
        m_s=args.m_s,
        m_h=args.m_h,
        lambda_asym=(args.lambda_asym if variant_config.use_asym_loss else 0.0),
        use_reshaping=variant_config.use_reshaping,
        use_asym_loss=variant_config.use_asym_loss,
    )
    train_features, train_labels, _train_dataset_ids = materialize_layer_tensors(
        all_reps["train"]["representations"],
        all_reps["train"]["labels"],
        all_reps["train"]["dataset_ids"],
        pooling_type,
        layer,
    )
    val_features, val_labels, val_dataset_ids = materialize_layer_tensors(
        all_reps["validation"]["representations"],
        all_reps["validation"]["labels"],
        all_reps["validation"]["dataset_ids"],
        pooling_type,
        layer,
    )
    model, val_metrics = train_harm_model_tensors(
        model,
        train_features,
        train_labels,
        val_features,
        val_labels,
        val_dataset_ids,
        device,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        show_progress=bool(args.show_progress),
    )
    test_features, test_labels, test_dataset_ids = materialize_layer_tensors(
        all_reps["test"]["representations"],
        all_reps["test"]["labels"],
        all_reps["test"]["dataset_ids"],
        pooling_type,
        layer,
    )
    y_test, pred_test, ds_test, prob_test = predict_harm_model_tensors(model, test_features, test_labels, test_dataset_ids, args.eval_batch_size, device)
    from train.salr_modules import binary_metrics
    test_metrics = binary_metrics(y_test, pred_test, prob_test, ds_test)
    artifact = {
        "artifact_type": "layerwise",
        "method": "SaLR-layerwise",
        "variant": args.variant,
        "pooling_type": pooling_type,
        "layer": int(layer),
        "selected_layers": [int(layer)],
        "feature_dim": int(model.feature_dim),
        "hidden_dim": int(args.hidden_dim),
        "direction_info": slice_direction_info(direction_info, [layer]),
        "model": model.cpu(),
        "val_metrics": val_metrics,
        "val_f1": float(val_metrics.get("val_f1", val_metrics.get("f1_macro", 0.0))),
        "test_metrics": test_metrics,
        "test_f1": float(test_metrics.get("f1_macro_per_dataset", test_metrics.get("f1_macro", 0.0))),
        "hyperparameters": vars(args),
    }
    return artifact


def save_layerwise_artifacts(layer_artifacts: Sequence[dict], output_dir: str, pooling_type: str) -> List[dict]:
    layer_dir = os.path.join(output_dir, "layerwise", pooling_type)
    os.makedirs(layer_dir, exist_ok=True)
    manifest = []
    for artifact in layer_artifacts:
        layer = int(artifact["layer"])
        path = os.path.join(layer_dir, f"layer_{layer}.pkl")
        with open(path, "wb") as f:
            pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
        manifest.append({
            "layer": layer,
            "artifact_path": path,
            "val_f1": float(artifact.get("val_f1", 0.0)),
            "test_f1": float(artifact.get("test_f1", 0.0)),
        })
    manifest_path = os.path.join(output_dir, f"layerwise_manifest_{pooling_type}.json")
    with open(manifest_path, "w") as f:
        json.dump(convert_json(manifest), f, indent=2)
    combined_path = os.path.join(output_dir, f"layerwise_models_{pooling_type}.pkl")
    with open(combined_path, "wb") as f:
        pickle.dump({"pooling_type": pooling_type, "artifacts": list(layer_artifacts)}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved layerwise artifacts to {layer_dir}")
    return manifest


def save_ensemble_artifact(artifact: dict, output_dir: str, pooling_type: str) -> str:
    method = artifact["ensemble_method"]
    layers = artifact["selected_layers"]
    ensemble_dir = os.path.join(output_dir, "ensemble", pooling_type)
    os.makedirs(ensemble_dir, exist_ok=True)
    path = os.path.join(ensemble_dir, f"ensemble_{method}_k{len(layers)}_layers_{layer_slug(layers)}.pkl")
    with open(path, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--variant", type=str, default="SaLR-full", 
                        choices=list(["SaLR-full", "SaLR-noR", "SaLR-noA", "SaLR-onlyR",
                                      "SaLR-onlyA", "SaLR-stack", "SaLR-base"]))
    parser.add_argument("--batch_size", type=int, default=32, help="LLM representation extraction batch size")
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--pooling_types", type=str, nargs="+", default=["residual_mean"], help="Mean-pooled representations only, e.g. residual_mean or mlp_mean")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--gamma_base", type=float, default=1.0)
    parser.add_argument("--a_s", type=float, default=0.0)
    parser.add_argument("--a_h", type=float, default=0.2)
    parser.add_argument("--lambda_align", type=float, default=1.0)
    parser.add_argument("--m_s", type=float, default=4.0)
    parser.add_argument("--m_h", type=float, default=1.5)
    parser.add_argument("--lambda_asym", type=float, default=0.5)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=64, help="Epochs for each layer-wise classifier")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--ensemble_method", type=str, default="auto", choices=["auto", "topk_average", "stacking"], help="auto uses SaLR-stack=>stacking and other variants=>topk_average")
    parser.add_argument("--topk_candidates", type=str, default="default", help="Comma list of k values or 'default' for {1,L/4,L/2,L}")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--stacking_C", type=float, default=1.0)
    parser.add_argument("--stacking_max_iter", type=int, default=1000)
    parser.add_argument("--stacking_class_weight", type=str, default="balanced")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default="cache/representations", help="Directory for cached frozen-LLM representations")
    parser.add_argument("--use_cache", type=int, default=1)
    parser.add_argument("--refresh_cache", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_progress", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    for rep_type in args.pooling_types:
        if not rep_type.endswith("_mean"):
            raise ValueError(f"Only mean pooling is supported in this SaLR version. Got {rep_type!r}.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    variant_config = get_variant_config(args.variant)
    ensemble_method = variant_config.default_ensemble_method if args.ensemble_method == "auto" else args.ensemble_method

    print("========================================")
    print("Training SaLR: layer-wise classifiers + top-k ensemble")
    print("========================================")
    print(f"Model: {args.model}")
    print(f"Datasets: {args.datasets}")
    print(f"Variant: {args.variant} -> {variant_config}")
    print(f"Ensemble method: {ensemble_method}")
    print(f"Device: {device}")

    output_dir = f"probes/{args.model}_salr/{args.variant}"
    os.makedirs(output_dir, exist_ok=True)

    print("\n[1/4] Extracting original representations...")
    all_reps = extract_all_representations(
        args.model,
        args.datasets,
        args.device,
        args.batch_size,
        args.pooling_types,
        args.val_ratio,
        seed=args.seed,
        cache_dir=args.cache_dir,
        use_cache=bool(args.use_cache),
        refresh_cache=bool(args.refresh_cache),
    )

    all_results = []
    for pooling_type in args.pooling_types:
        print(f"\n================ pooling_type={pooling_type} ================")
        print("\n[2/4] Estimating closed-form harmful directions from original train data...")
        direction_info = estimate_harm_directions(
            all_reps["train"]["representations_original"],
            all_reps["train"]["labels_original"],
            pooling_type,
        )
        layers = [int(x) for x in direction_info["layers"].tolist()]
        feature_dim = infer_feature_dim(all_reps["train"]["representations_original"], pooling_type, layers)
        print(f"Layers: {len(layers)} | feature_dim: {feature_dim}")

        print("\n[3/4] Training independent layer-wise SaLR classifiers...")
        layer_artifacts = []
        for layer in layers:
            print(f"\nLayer {layer}")
            artifact = train_one_layer(all_reps, pooling_type, direction_info, layer, args, variant_config, device)
            layer_artifacts.append(artifact)
            print(
                f"Layer {layer:2d}: val_f1={artifact['val_f1']:.4f} "
                f"test_f1={artifact['test_f1']:.4f}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        manifest = save_layerwise_artifacts(layer_artifacts, output_dir, pooling_type)

        print("\n[4/4] Selecting top-k layers on validation and fitting final ensemble...")
        candidates = parse_topk_candidates(args.topk_candidates, len(layer_artifacts))
        ensemble_fit = fit_topk_ensemble_on_validation(
            layer_artifacts=layer_artifacts,
            val_representations=all_reps["validation"]["representations"],
            val_labels=all_reps["validation"]["labels"],
            val_dataset_ids=all_reps["validation"]["dataset_ids"],
            batch_size=args.eval_batch_size,
            device=device,
            method=ensemble_method,
            topk_candidates=candidates,
            stacking_C=args.stacking_C,
            stacking_max_iter=args.stacking_max_iter,
            stacking_class_weight=args.stacking_class_weight,
            seed=args.seed,
            threshold=args.threshold,
        )
        selected_layers = [int(x) for x in ensemble_fit["selected_layers"]]
        selected_artifacts = choose_layer_artifacts(layer_artifacts, "manual", layers=layer_slug(selected_layers).replace("-", ","))
        ensemble_artifact = {
            "artifact_type": "layerwise_ensemble",
            "method": "SaLR",
            "variant": args.variant,
            "pooling_type": pooling_type,
            "ensemble_method": ensemble_method,
            "selected_layers": selected_layers,
            "k": int(len(selected_layers)),
            "threshold": float(args.threshold),
            "stacker": ensemble_fit.get("stacker"),
            "layer_artifacts": layer_artifacts,
            "selected_artifacts": selected_artifacts,
            "layer_ranking": ensemble_fit["layer_ranking"],
            "topk_sweep": ensemble_fit["topk_sweep"],
            "val_f1": float(ensemble_fit["val_f1"]),
            "val_metrics": ensemble_fit["val_metrics"],
            "feature_dim": int(feature_dim),
            "hidden_dim": int(args.hidden_dim),
            "direction_info": direction_info,
            "layerwise_manifest": manifest,
            "hyperparameters": vars(args),
        }
        test_eval = evaluate_ensemble_artifact(
            ensemble_artifact,
            all_reps["test"]["representations"],
            all_reps["test"]["labels"],
            all_reps["test"]["dataset_ids"],
            args.eval_batch_size,
            device,
            threshold=args.threshold,
        )
        train_eval = evaluate_ensemble_artifact(
            ensemble_artifact,
            all_reps["train"]["representations"],
            all_reps["train"]["labels"],
            all_reps["train"]["dataset_ids"],
            args.eval_batch_size,
            device,
            threshold=args.threshold,
        )
        ensemble_artifact["test_metrics"] = test_eval["metrics"]
        ensemble_artifact["train_metrics"] = train_eval["metrics"]
        ensemble_path = save_ensemble_artifact(ensemble_artifact, output_dir, pooling_type)
        ensemble_artifact["ensemble_artifact_path"] = ensemble_path
        all_results.append(ensemble_artifact)
        print(
            f"\nEnsemble pooling={pooling_type}: method={ensemble_method} k={len(selected_layers)} "
            f"layers={selected_layers} val_f1={ensemble_artifact['val_f1']:.4f} "
            f"test_f1={ensemble_artifact['test_metrics'].get('f1_macro_per_dataset', ensemble_artifact['test_metrics']['f1_macro']):.4f}"
        )

    best_overall = max(all_results, key=lambda x: x["val_f1"])
    best_path = f"{output_dir}/best_model.pkl"
    with open(best_path, "wb") as f:
        pickle.dump(best_overall, f, protocol=pickle.HIGHEST_PROTOCOL)

    json_payload = {
        "best_overall": convert_json(best_overall),
        "all_results": convert_json(sorted(all_results, key=lambda x: x["val_f1"], reverse=True)),
    }
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(json_payload, f, indent=2)

    print(f"\nSaved SaLR ensemble artifact to: {best_path}")
    print(f"Saved metrics to: {output_dir}/results.json")


if __name__ == "__main__":
    main()
