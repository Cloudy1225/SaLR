"""Reusable SaLR components.

This implementation follows the revised SaLR formulation:

1. Estimate per-layer harmful directions in closed form from class means.
2. Train one independent reshaping-aware classifier per layer.
3. Build the final detector by top-k layer selection and ensemble.

The frozen LLM is never updated; every module here operates on extracted
representations from ``Qwen3RepresentationExtractor``.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import os
import pickle
import re
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class SaLRVariantConfig:
    use_reshaping: bool
    use_asym_loss: bool
    default_ensemble_method: str = "topk_average"


SaLR_VARIANTS: Dict[str, SaLRVariantConfig] = {
    "SaLR-full": SaLRVariantConfig(True, True, "topk_average"),
    "SaLR-noR": SaLRVariantConfig(False, True, "topk_average"),
    "SaLR-onlyR": SaLRVariantConfig(True, False, "topk_average"),
    "SaLR-noA": SaLRVariantConfig(True, False, "topk_average"),
    "SaLR-onlyA": SaLRVariantConfig(False, True, "topk_average"),
    "SaLR-stack": SaLRVariantConfig(True, True, "stacking"),
    "SaLR-base": SaLRVariantConfig(False, False, "topk_average"),
}


CACHE_VERSION = 5
CACHE_SCHEMA = "plain_path_v2"


def get_variant_config(name: str) -> SaLRVariantConfig:
    if name not in SaLR_VARIANTS:
        valid = ", ".join(sorted(SaLR_VARIANTS))
        raise ValueError(f"Unknown SaLR variant {name!r}. Valid variants: {valid}")
    return SaLR_VARIANTS[name]


def _slugify_component(value: Any) -> str:
    """Convert a cache path component into a readable, filesystem-safe slug."""

    if isinstance(value, (list, tuple, set)):
        return "__".join(_slugify_component(v) for v in value) or "none"
    if value is None:
        return "none"
    text = str(value).strip()
    text = text.replace(os.sep, "__")
    if os.altsep:
        text = text.replace(os.altsep, "__")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z._=+\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "none"


def _normalise_rep_types(rep_types: Sequence[str]) -> List[str]:
    return [str(x) for x in rep_types]


def _dataset_name_from_cache_inputs(kind: str, split_name: Optional[str], extra: Optional[Dict[str, Any]]) -> Any:
    extra = extra or {}
    if extra.get("dataset") is not None:
        return extra["dataset"]
    if extra.get("datasets") is not None:
        datasets = list(extra["datasets"])
        return datasets[0] if len(datasets) == 1 else datasets
    if kind == "think_eval" and extra.get("backbone") is not None:
        return f"Qwen3GuardTest_thinking_{extra['backbone']}"
    if kind == "streaming_prefix" and split_name:
        return str(split_name).split(":", 1)[0]
    return "unspecified_dataset"


def _split_id_from_cache_inputs(kind: str, split_name: Optional[str]) -> str:
    split = str(split_name or "unspecified_split")
    lower = split.lower()
    if lower in {"train", "validation", "test"}:
        return lower
    if lower.endswith("_test"):
        return "test"
    if lower.startswith("validation"):
        return "validation"
    if lower.startswith("train"):
        return "train"
    return split


def _cache_item_from_cache_inputs(kind: str, split_name: Optional[str], extra: Optional[Dict[str, Any]]) -> str:
    extra = extra or {}
    if kind == "streaming_prefix":
        sample_id = extra.get("sample_id", "sample")
        position = extra.get("position", "position")
        return f"prefix_sample-{_slugify_component(sample_id)}_pos-{_slugify_component(position)}"
    return "representations"


def build_representation_cache_metadata(
    *,
    kind: str,
    model_name: str,
    model_path: str,
    rep_types: Sequence[str],
    texts: Sequence[str],
    labels: Optional[Sequence[int]] = None,
    dataset_ids: Optional[Sequence[int]] = None,
    split_name: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Build readable cache metadata and a readable relative cache key.

    Cache paths are intentionally based on stable, human-readable identifiers
    such as model name, representation types, dataset name, and split. They do
    not include text or label hashes, so evaluation can reuse the
    representations saved during training for the same dataset/split.
    """

    rep_types_list = _normalise_rep_types(rep_types)
    dataset_name = _dataset_name_from_cache_inputs(kind, split_name, extra)
    dataset_slug = _slugify_component(dataset_name)
    split_id = _split_id_from_cache_inputs(kind, split_name)
    cache_item = _cache_item_from_cache_inputs(kind, split_name, extra)
    rep_types_slug = _slugify_component(rep_types_list)

    metadata: Dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "kind": kind,
        "model_name": model_name,
        "model_path": model_path,
        "rep_types": rep_types_list,
        "dataset_name": dataset_name,
        "dataset_slug": dataset_slug,
        "split_name": split_name,
        "split_id": split_id,
        "cache_item": cache_item,
        "num_texts": len(texts),
        "extra": extra or {},
    }
    # Relative path below ``<cache_dir>/<model_name>/``. Keep it readable so a
    # cache can be inspected or copied without decoding a hash.
    cache_key = os.path.join(rep_types_slug, dataset_slug, _slugify_component(split_id), _slugify_component(cache_item))
    return metadata, cache_key


def representation_cache_path(cache_dir: Optional[str], model_name: str, cache_key: str) -> Optional[str]:
    if not cache_dir:
        return None
    safe_model_name = _slugify_component(model_name)
    rel_key = str(cache_key).strip().lstrip(os.sep)
    if os.altsep:
        rel_key = rel_key.lstrip(os.altsep)
    if not rel_key.endswith(".pkl"):
        rel_key = f"{rel_key}.pkl"
    return os.path.join(cache_dir, safe_model_name, rel_key)


def _metadata_matches_cache_request(saved: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    """Validate only the path-defining cache fields, not text/label hashes."""

    if not isinstance(saved, dict):
        return False
    required_fields = ["cache_version", "cache_schema", "model_name", "rep_types", "dataset_slug", "split_id", "cache_item"]
    for field in required_fields:
        if saved.get(field) != expected.get(field):
            return False
    saved_num_texts = saved.get("num_texts")
    expected_num_texts = expected.get("num_texts")
    if saved_num_texts is not None and expected_num_texts is not None and int(saved_num_texts) != int(expected_num_texts):
        return False
    return True


def load_cached_representations(cache_path: Optional[str], metadata: Dict[str, Any], refresh_cache: bool = False) -> Optional[List[dict]]:
    if not cache_path or refresh_cache or not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        saved_metadata = payload.get("metadata", {})
        if not _metadata_matches_cache_request(saved_metadata, metadata):
            print(f"[cache miss] Metadata mismatch for {cache_path}")
            return None
        representations = payload.get("representations")
        if not isinstance(representations, list):
            print(f"[cache miss] Invalid representation payload in {cache_path}")
            return None
        print(f"[cache hit] Loaded {len(representations)} LLM representations from {cache_path}")
        return representations
    except Exception as exc:
        print(f"[cache miss] Could not read {cache_path}: {exc}")
        return None


def save_cached_representations(cache_path: Optional[str], metadata: Dict[str, Any], representations: Sequence[dict]) -> None:
    if not cache_path:
        return
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".pkl", dir=cache_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump({"metadata": metadata, "representations": list(representations)}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        print(f"[cache save] Saved {len(representations)} LLM representations to {cache_path}")
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

def infer_layers(representations: Sequence[dict], selected_layers: Optional[Sequence[int]] = None) -> List[int]:
    if selected_layers is not None:
        return [int(x) for x in selected_layers]
    if not representations:
        raise ValueError("Cannot infer layers from an empty representation list.")
    return sorted(int(k) for k in representations[0].keys())


def infer_feature_dim(representations: Sequence[dict], pooling_type: str, selected_layers: Sequence[int]) -> int:
    if not representations:
        raise ValueError("Cannot infer feature dimension from an empty representation list.")
    first_layer = int(selected_layers[0])
    return int(np.asarray(representations[0][first_layer][pooling_type]).shape[-1])


def parse_layer_spec(spec: Optional[str], all_layers: Sequence[int]) -> List[int]:
    all_layers = [int(x) for x in all_layers]
    if spec is None or str(spec).strip().lower() in {"", "all"}:
        return list(all_layers)
    allowed = set(all_layers)
    selected: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            step = 1 if end >= start else -1
            selected.extend(range(start, end + step, step))
        else:
            selected.append(int(part))
    deduped: List[int] = []
    for layer in selected:
        if layer not in allowed:
            raise ValueError(f"Layer {layer} is not available. Available layers: {all_layers}")
        if layer not in deduped:
            deduped.append(layer)
    if not deduped:
        raise ValueError(f"No layers selected from spec={spec!r}")
    return deduped


def layer_slug(layers: Sequence[int]) -> str:
    return "-".join(str(int(x)) for x in layers) if layers else "none"


def default_topk_candidates(num_layers: int) -> List[int]:
    raw = [1, max(1, num_layers // 4), max(1, num_layers // 2), num_layers]
    out: List[int] = []
    for k in raw:
        k = int(min(max(1, k), num_layers))
        if k not in out:
            out.append(k)
    return out


def parse_topk_candidates(spec: Optional[str], num_layers: int) -> List[int]:
    if spec is None or str(spec).strip().lower() in {"", "default"}:
        return default_topk_candidates(num_layers)
    values: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        k = int(min(max(1, k), num_layers))
        if k not in values:
            values.append(k)
    if not values:
        values = default_topk_candidates(num_layers)
    return values


class SaLRRepresentationDataset(Dataset):
    """Lazy dataset that returns selected layer vectors from list-of-dict reps."""

    def __init__(
        self,
        representations: Sequence[dict],
        labels: Sequence[int],
        dataset_ids: Optional[Sequence[int]],
        pooling_type: str,
        selected_layers: Sequence[int],
    ) -> None:
        if len(representations) != len(labels):
            raise ValueError("representations and labels must have the same length")
        self.representations = representations
        self.labels = np.asarray(labels, dtype=np.int64)
        self.dataset_ids = np.zeros(len(labels), dtype=np.int64) if dataset_ids is None else np.asarray(dataset_ids, dtype=np.int64)
        self.pooling_type = pooling_type
        self.selected_layers = [int(x) for x in selected_layers]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rep = self.representations[idx]
        features = np.stack(
            [np.asarray(rep[layer_idx][self.pooling_type], dtype=np.float32) for layer_idx in self.selected_layers],
            axis=0,
        )
        return (
            torch.from_numpy(features),
            torch.tensor(int(self.labels[idx]), dtype=torch.long),
            torch.tensor(int(self.dataset_ids[idx]), dtype=torch.long),
        )


def make_dataloader(
    representations: Sequence[dict],
    labels: Sequence[int],
    dataset_ids: Optional[Sequence[int]],
    pooling_type: str,
    selected_layers: Sequence[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SaLRRepresentationDataset(representations, labels, dataset_ids, pooling_type, selected_layers)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def _as_pinned_if_possible(tensor: torch.Tensor, pin_memory: bool = True) -> torch.Tensor:
    """Pin CPU tensors only when CUDA is available and pinning succeeds."""
    if pin_memory and torch.cuda.is_available() and tensor.device.type == "cpu":
        try:
            return tensor.pin_memory()
        except RuntimeError:
            return tensor
    return tensor


def materialize_layer_tensors(
    representations: Sequence[dict],
    labels: Sequence[int],
    dataset_ids: Optional[Sequence[int]],
    pooling_type: str,
    layer: int,
    *,
    pin_memory: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert cached list-of-dict representations into contiguous tensors once.

    The original DataLoader path re-runs np.asarray/np.stack/torch.tensor for
    every sample on every epoch. For a tiny layer classifier this Python-side
    conversion dominates training. This function pays that cost once per split
    and returns [N, D] features that can be manually batched.
    """

    labels_arr = np.asarray(labels, dtype=np.int64)
    dataset_ids_arr = np.zeros(len(labels_arr), dtype=np.int64) if dataset_ids is None else np.asarray(dataset_ids, dtype=np.int64)
    if len(representations) != len(labels_arr):
        raise ValueError("representations and labels must have the same length")
    layer = int(layer)
    feature_dim = infer_feature_dim(representations, pooling_type, [layer])
    features = np.empty((len(labels_arr), feature_dim), dtype=np.float32)
    for idx, rep in enumerate(representations):
        features[idx] = np.asarray(rep[layer][pooling_type], dtype=np.float32)
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels_arr)
    ds = torch.from_numpy(dataset_ids_arr)
    return _as_pinned_if_possible(x, pin_memory), _as_pinned_if_possible(y, pin_memory), _as_pinned_if_possible(ds, pin_memory)


@torch.no_grad()
def predict_harm_model_tensors(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    dataset_ids: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prediction path for already-materialized [N, D] feature tensors."""

    model.eval()
    model = model.to(device)
    all_preds: List[torch.Tensor] = []
    all_probs: List[torch.Tensor] = []
    n = int(features.shape[0])
    for start in range(0, n, int(batch_size)):
        end = min(start + int(batch_size), n)
        batch_x = features[start:end].to(device, non_blocking=True)
        with _autocast_for(device):
            logits = model(batch_x)
            probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            preds = (probs >= 0.5).long()
        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
    pred_arr = torch.cat(all_preds).numpy() if all_preds else np.asarray([], dtype=np.int64)
    prob_arr = torch.cat(all_probs).numpy() if all_probs else np.asarray([], dtype=np.float32)
    return (
        labels.cpu().numpy().astype(np.int64, copy=False),
        pred_arr.astype(np.int64, copy=False),
        dataset_ids.cpu().numpy().astype(np.int64, copy=False),
        prob_arr.astype(np.float32, copy=False),
    )


def train_harm_model_tensors(
    model: SaLRLayerClassifier,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    val_dataset_ids: torch.Tensor,
    device: torch.device,
    train_batch_size: int,
    eval_batch_size: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 128,
    patience: int = 12,
    show_progress: bool = True,
) -> Tuple[SaLRLayerClassifier, Dict[str, float]]:
    """Train a layer-wise SaLR classifier without DataLoader/Dataset overhead."""

    from tqdm import tqdm

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = None
    best_val_f1 = -1.0
    patience_counter = 0
    iterator: Iterable[int] = tqdm(range(epochs), desc="Training SaLR-layer") if show_progress else range(epochs)
    n = int(train_features.shape[0])
    batch_size = int(train_batch_size)

    for _ in iterator:
        model.train()
        running_loss = 0.0
        num_batches = 0
        # Keep the permutation on CPU so this path also works when features stay in host memory.
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_x = train_features.index_select(0, idx).to(device, non_blocking=True)
            batch_y = train_labels.index_select(0, idx).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_for(device):
                logits = model(batch_x)
                loss = model.total_loss(logits.float(), batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            num_batches += 1

        y_val, pred_val, val_dataset_ids_np, _ = predict_harm_model_tensors(
            model, val_features, val_labels, val_dataset_ids, eval_batch_size, device
        )
        val_f1 = compute_per_dataset_f1(y_val, pred_val, val_dataset_ids_np)
        mean_loss = running_loss / max(num_batches, 1)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{mean_loss:.4f}", "val_f1": f"{val_f1:.4f}", "best": f"{best_val_f1:.4f}"})
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    y_val, pred_val, val_dataset_ids_np, probs_val = predict_harm_model_tensors(
        model, val_features, val_labels, val_dataset_ids, eval_batch_size, device
    )
    metrics = binary_metrics(y_val, pred_val, probs_val, val_dataset_ids_np)
    metrics["val_f1"] = metrics.get("f1_macro_per_dataset", metrics["f1_macro"])
    return model, metrics


@torch.no_grad()
def predict_layer_probabilities_fast(
    layer_artifacts: Sequence[Dict[str, Any]],
    representations: Sequence[dict],
    labels: Sequence[int],
    dataset_ids: Optional[Sequence[int]],
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return [N, K] harmful probabilities with one tensor materialization per layer."""

    if not layer_artifacts:
        raise ValueError("No layer artifacts provided for ensemble prediction.")
    probs_per_layer: List[np.ndarray] = []
    y_ref: Optional[np.ndarray] = None
    dataset_ids_ref: Optional[np.ndarray] = None
    for artifact in layer_artifacts:
        layer = int(artifact["layer"])
        pooling_type = artifact["pooling_type"]
        model = artifact["model"]
        features, label_t, dataset_id_t = materialize_layer_tensors(representations, labels, dataset_ids, pooling_type, layer)
        y_true, _preds, ds_ids, probs = predict_harm_model_tensors(model, features, label_t, dataset_id_t, batch_size, device)
        if y_ref is None:
            y_ref = y_true
            dataset_ids_ref = ds_ids
        probs_per_layer.append(probs.astype(np.float32))
        artifact["model"] = model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return y_ref if y_ref is not None else np.asarray([], dtype=np.int64), dataset_ids_ref if dataset_ids_ref is not None else np.asarray([], dtype=np.int64), np.stack(probs_per_layer, axis=1)

def compute_per_dataset_f1(y_true: Sequence[int], y_pred: Sequence[int], dataset_ids: Sequence[int]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    dataset_ids = np.asarray(dataset_ids)
    f1s: List[float] = []
    for dataset_id in np.unique(dataset_ids):
        mask = dataset_ids == dataset_id
        if np.any(mask):
            f1s.append(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0))
    return float(np.mean(f1s)) if f1s else 0.0


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int], probs: Optional[Sequence[float]] = None, dataset_ids: Optional[Sequence[int]] = None) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    out = {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)),
    }
    if dataset_ids is not None:
        out["f1_macro_per_dataset"] = compute_per_dataset_f1(y_true, y_pred, dataset_ids)
    if probs is not None and len(probs) > 0:
        out["mean_harm_prob"] = float(np.mean(probs))
    return out


def estimate_harm_directions(
    representations: Sequence[dict],
    labels: Sequence[int],
    pooling_type: str,
    selected_layers: Optional[Sequence[int]] = None,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """Closed-form Stage-1 harmful direction probe from class means."""

    layers = infer_layers(representations, selected_layers)
    feature_dim = infer_feature_dim(representations, pooling_type, layers)
    labels = np.asarray(labels, dtype=np.int64)
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("SaLR direction estimation requires at least one safe and one harmful sample.")

    pos_sum = np.zeros((len(layers), feature_dim), dtype=np.float64)
    neg_sum = np.zeros((len(layers), feature_dim), dtype=np.float64)
    pos_count = 0
    neg_count = 0
    for rep, label in zip(representations, labels):
        target = pos_sum if int(label) == 1 else neg_sum
        for layer_pos, layer_idx in enumerate(layers):
            target[layer_pos] += np.asarray(rep[layer_idx][pooling_type], dtype=np.float64)
        if int(label) == 1:
            pos_count += 1
        else:
            neg_count += 1

    mu_plus = pos_sum / max(pos_count, 1)
    mu_minus = neg_sum / max(neg_count, 1)
    diff = mu_plus - mu_minus
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    directions = diff / np.maximum(norms, eps)
    return {
        "layers": np.asarray(layers, dtype=np.int64),
        "mu_plus": mu_plus.astype(np.float32),
        "mu_minus": mu_minus.astype(np.float32),
        "directions": directions.astype(np.float32),
        "interclass_norms": norms.squeeze(-1).astype(np.float32),
    }


def slice_direction_info(direction_info: Dict[str, np.ndarray], selected_layers: Sequence[int]) -> Dict[str, np.ndarray]:
    all_layers = [int(x) for x in np.asarray(direction_info["layers"]).tolist()]
    positions = [all_layers.index(int(layer)) for layer in selected_layers]
    return {
        "layers": np.asarray([all_layers[pos] for pos in positions], dtype=np.int64),
        "mu_plus": np.asarray(direction_info["mu_plus"])[positions].astype(np.float32),
        "mu_minus": np.asarray(direction_info["mu_minus"])[positions].astype(np.float32),
        "directions": np.asarray(direction_info["directions"])[positions].astype(np.float32),
        "interclass_norms": np.asarray(direction_info["interclass_norms"])[positions].astype(np.float32),
    }


def _softplus_inverse(x: float) -> float:
    x = float(max(x, 1e-6))
    return float(np.log(np.expm1(x)))


class SaLRLayerClassifier(nn.Module):
    """Single-layer SaLR classifier.

    Input shape may be ``[batch, D]`` or ``[batch, 1, D]``. The module reshapes
    along the layer's harmful direction, then applies a two-layer GELU MLP.
    """

    def __init__(
        self,
        mu_minus: np.ndarray,
        direction: np.ndarray,
        hidden_dim: int = 512,
        dropout: float = 0.0,
        gamma_base: float = 1.0,
        rank: int = 32,
        a_s: float = 0.0,
        a_h: float = 0.2,
        m_s: float = 4.0,
        m_h: float = 1.5,
        lambda_align: float = 1.0,
        lambda_asym: float = 0.5,
        use_reshaping: bool = True,
        use_asym_loss: bool = True,
    ) -> None:
        super().__init__()

        mu_t = torch.as_tensor(mu_minus, dtype=torch.float32).view(-1)
        d_t = torch.as_tensor(direction, dtype=torch.float32).view(-1)

        if mu_t.shape != d_t.shape:
            raise ValueError("mu_minus and direction must have the same feature dimension")

        d_t = F.normalize(d_t, dim=0)

        self.feature_dim = int(mu_t.shape[0])
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.rank = int(rank)

        self.use_reshaping = bool(use_reshaping)
        self.use_asym_loss = bool(use_asym_loss)

        self.m_s = float(m_s)
        self.m_h = float(m_h)
        self.lambda_asym = float(lambda_asym)

        self.a_s = float(a_s)
        self.a_h = float(a_h)
        self.lambda_align = float(lambda_align)

        self.register_buffer("mu_minus", mu_t)
        self.register_buffer("direction", d_t)

        self.gamma_raw = nn.Parameter(
            torch.tensor(_softplus_inverse(gamma_base), dtype=torch.float32)
        )

        # Low-rank residual transform:
        # x_hat = x + U V^T (x - mu_minus) + b
        #
        # For batch-row tensors:
        # x_hat = x + (x - mu_minus) @ V @ U.T + b
        self.U = nn.Parameter(torch.zeros(self.feature_dim, self.rank))
        self.V = nn.Parameter(torch.empty(self.feature_dim, self.rank))
        self.b = nn.Parameter(torch.zeros(self.feature_dim))

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(self.hidden_dim, 2),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.V, mean=0.0, std=0.02)
        nn.init.zeros_(self.U)
        nn.init.zeros_(self.b)

        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.gamma_raw)

    def low_rank_transform(self, x: torch.Tensor) -> torch.Tensor:
        delta = x - self.mu_minus.view(1, -1)
        low_rank_residual = delta @ self.V @ self.U.T
        return x + low_rank_residual + self.b.view(1, -1)

    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self.low_rank_transform(x)

        # Only cache during training.
        if self.training:
            self._last_x_hat = x_hat

        if not self.use_reshaping:
            return x_hat

        phi_hat = (
            (x_hat - self.mu_minus.view(1, -1))
            * self.direction.view(1, -1)
        ).sum(dim=-1)

        shift = (
            self.gamma.view(1, 1)
            * F.relu(phi_hat).view(-1, 1)
            * self.direction.view(1, -1)
        )

        return x_hat + shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError(
                    f"SaLRLayerClassifier expects one layer, got input shape {tuple(x.shape)}"
                )
            x = x[:, 0, :]

        x = self.reshape(x.float())
        return self.classifier(x)

    def asymmetric_margin_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.long()

        harmful_score = logits[:, 1] - logits[:, 0]

        safe_mask = labels == 0
        harmful_mask = labels == 1

        loss = torch.zeros((), device=logits.device, dtype=logits.dtype)

        if torch.any(safe_mask):
            loss = loss + F.relu(
                harmful_score[safe_mask] + self.m_s
            ).pow(2).mean()

        if torch.any(harmful_mask):
            loss = loss + F.relu(
                self.m_h - harmful_score[harmful_mask]
            ).pow(2).mean()

        return loss

    def distribution_aligning_loss(
        self,
        labels: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if not self.training:
            return torch.zeros((), device=labels.device)

        if not hasattr(self, "_last_x_hat"):
            raise RuntimeError(
                "distribution_aligning_loss requires a prior forward pass during training."
            )

        x_hat = self._last_x_hat
        labels = labels.long().to(x_hat.device)

        displaced = x_hat - self.mu_minus.view(1, -1)

        psi_hat = (
            displaced * self.direction.view(1, -1)
        ).sum(dim=-1) / displaced.norm(dim=-1).clamp_min(eps)

        safe_mask = labels == 0
        harmful_mask = labels == 1

        loss = torch.zeros((), device=x_hat.device, dtype=x_hat.dtype)

        if torch.any(safe_mask):
            loss = loss + F.relu(
                psi_hat[safe_mask] + self.a_s
            ).pow(2).mean()

        if torch.any(harmful_mask):
            loss = loss + F.relu(
                self.a_h - psi_hat[harmful_mask]
            ).pow(2).mean()

        return loss

    def total_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.long().to(logits.device)

        loss = F.cross_entropy(logits, labels)

        if self.use_asym_loss and self.lambda_asym > 0:
            loss = loss + self.lambda_asym * self.asymmetric_margin_loss(logits, labels)

        # Only apply distribution-aligning loss during training.
        if self.training and self.lambda_align > 0:
            loss = loss + self.lambda_align * self.distribution_aligning_loss(labels)

        return loss

SaLRClassifier = SaLRLayerClassifier


def make_layer_classifier(
    direction_info: Dict[str, np.ndarray],
    layer: int,
    hidden_dim: int,
    dropout: float,
    gamma_base: float,
    a_s: float,
    a_h: float,
    lambda_align: float,
    m_s: float,
    m_h: float,
    lambda_asym: float,
    use_reshaping: bool,
    use_asym_loss: bool,
) -> SaLRLayerClassifier:
    layers = [int(x) for x in np.asarray(direction_info["layers"]).tolist()]
    pos = layers.index(int(layer))
    return SaLRLayerClassifier(
        mu_minus=np.asarray(direction_info["mu_minus"])[pos],
        direction=np.asarray(direction_info["directions"])[pos],
        hidden_dim=hidden_dim,
        dropout=dropout,
        gamma_base=gamma_base,
        a_s=a_s,
        a_h=a_h,
        lambda_align=lambda_align,
        m_s=m_s,
        m_h=m_h,
        lambda_asym=lambda_asym,
        use_reshaping=use_reshaping,
        use_asym_loss=use_asym_loss,
    )


def _autocast_for(device: torch.device):
    return torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()


@torch.no_grad()
def predict_harm_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    all_dataset_ids: List[int] = []
    all_probs: List[float] = []
    model = model.to(device)
    for batch_x, batch_y, batch_dataset_ids in dataloader:
        batch_x = batch_x.to(device, non_blocking=True)
        with _autocast_for(device):
            logits = model(batch_x)
            probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            preds = (probs >= 0.5).long()
        all_labels.extend(batch_y.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_dataset_ids.extend(batch_dataset_ids.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
    return (
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_preds, dtype=np.int64),
        np.asarray(all_dataset_ids, dtype=np.int64),
        np.asarray(all_probs, dtype=np.float32),
    )


def train_harm_model(
    model: SaLRLayerClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 128,
    patience: int = 12,
    show_progress: bool = True,
) -> Tuple[SaLRLayerClassifier, Dict[str, float]]:
    """Train one layer-wise SaLR classifier and restore best validation-F1."""

    from tqdm import tqdm

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = None
    best_val_f1 = -1.0
    patience_counter = 0
    iterator: Iterable[int] = tqdm(range(epochs), desc="Training SaLR-layer") if show_progress else range(epochs)

    for _ in iterator:
        model.train()
        running_loss = 0.0
        num_batches = 0
        for batch_x, batch_y, _batch_dataset_ids in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_for(device):
                logits = model(batch_x)
                loss = model.total_loss(logits.float(), batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            num_batches += 1

        y_val, pred_val, val_dataset_ids, _ = predict_harm_model(model, val_loader, device)
        val_f1 = compute_per_dataset_f1(y_val, pred_val, val_dataset_ids)
        mean_loss = running_loss / max(num_batches, 1)
        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix({"loss": f"{mean_loss:.4f}", "val_f1": f"{val_f1:.4f}", "best": f"{best_val_f1:.4f}"})
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    y_val, pred_val, val_dataset_ids, probs_val = predict_harm_model(model, val_loader, device)
    metrics = binary_metrics(y_val, pred_val, probs_val, val_dataset_ids)
    metrics["val_f1"] = metrics.get("f1_macro_per_dataset", metrics["f1_macro"])
    return model, metrics


@torch.no_grad()
def predict_layer_probabilities(
    layer_artifacts: Sequence[Dict[str, Any]],
    representations: Sequence[dict],
    labels: Sequence[int],
    dataset_ids: Optional[Sequence[int]],
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``[N, K]`` harmful-probability matrix for selected layer artifacts."""

    if not layer_artifacts:
        raise ValueError("No layer artifacts provided for ensemble prediction.")
    probs_per_layer: List[np.ndarray] = []
    y_ref: Optional[np.ndarray] = None
    dataset_ids_ref: Optional[np.ndarray] = None
    for artifact in layer_artifacts:
        layer = int(artifact["layer"])
        pooling_type = artifact["pooling_type"]
        model = artifact["model"]
        loader = make_dataloader(representations, labels, dataset_ids, pooling_type, [layer], batch_size, shuffle=False)
        y_true, _preds, ds_ids, probs = predict_harm_model(model, loader, device)
        if y_ref is None:
            y_ref = y_true
            dataset_ids_ref = ds_ids
        probs_per_layer.append(probs.astype(np.float32))
        # Keep CPU copies in artifacts after prediction to reduce GPU memory pressure.
        artifact["model"] = model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return y_ref if y_ref is not None else np.asarray([], dtype=np.int64), dataset_ids_ref if dataset_ids_ref is not None else np.asarray([], dtype=np.int64), np.stack(probs_per_layer, axis=1)


def fit_logistic_stacker(
    prob_matrix: np.ndarray,
    labels: Sequence[int],
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: Optional[str] = "balanced",
    seed: int = 42,
) -> LogisticRegression:
    labels_arr = np.asarray(labels, dtype=np.int64)
    class_weight_arg = None if class_weight in {None, "none", "None"} else class_weight
    clf = LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight_arg, random_state=seed)
    clf.fit(np.asarray(prob_matrix, dtype=np.float32), labels_arr)
    return clf


def ensemble_predict(
    prob_matrix: np.ndarray,
    method: str = "topk_average",
    stacker: Optional[LogisticRegression] = None,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    method = "topk_average" if method in {"avg", "average", "prob_average"} else method
    if method == "topk_average":
        probs = np.asarray(prob_matrix, dtype=np.float32).mean(axis=1)
    elif method == "stacking":
        if stacker is None:
            raise ValueError("stacking ensemble requires a fitted LogisticRegression stacker")
        probs = stacker.predict_proba(np.asarray(prob_matrix, dtype=np.float32))[:, 1]
    else:
        raise ValueError(f"Unsupported ensemble method: {method}")
    preds = (probs >= float(threshold)).astype(np.int64)
    return preds, probs.astype(np.float32)


def choose_layer_artifacts(
    layer_artifacts: Sequence[Dict[str, Any]],
    strategy: str,
    layers: Optional[str] = None,
    top_k: int = 1,
    threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    artifacts = list(layer_artifacts)
    all_layers = [int(a["layer"]) for a in artifacts]
    by_layer = {int(a["layer"]): a for a in artifacts}
    if strategy == "all":
        selected_layers = all_layers
    elif strategy == "manual":
        selected_layers = parse_layer_spec(layers, all_layers)
    else:
        ranked = sorted(artifacts, key=lambda a: float(a.get("val_f1", a.get("val_metrics", {}).get("val_f1", 0.0))), reverse=True)
        if strategy == "layerwise_best":
            selected_layers = [int(ranked[0]["layer"])]
        elif strategy == "layerwise_topk":
            selected_layers = [int(a["layer"]) for a in ranked[: max(1, int(top_k))]]
        elif strategy == "layerwise_threshold":
            selected_layers = [int(a["layer"]) for a in ranked if float(a.get("val_f1", a.get("val_metrics", {}).get("val_f1", 0.0))) >= float(threshold)]
            if not selected_layers:
                selected_layers = [int(ranked[0]["layer"])]
        else:
            raise ValueError(f"Unsupported layer selection strategy: {strategy}")
    return [by_layer[int(layer)] for layer in selected_layers]


def fit_topk_ensemble_on_validation(
    layer_artifacts: Sequence[Dict[str, Any]],
    val_representations: Sequence[dict],
    val_labels: Sequence[int],
    val_dataset_ids: Sequence[int],
    batch_size: int,
    device: torch.device,
    method: str = "topk_average",
    topk_candidates: Optional[Sequence[int]] = None,
    stacking_C: float = 1.0,
    stacking_max_iter: int = 1000,
    stacking_class_weight: Optional[str] = "balanced",
    seed: int = 42,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Rank layers by validation F1, sweep k, and fit the chosen ensemble."""

    ranked = sorted(layer_artifacts, key=lambda a: float(a.get("val_f1", a.get("val_metrics", {}).get("val_f1", 0.0))), reverse=True)
    num_layers = len(ranked)
    candidates = list(topk_candidates) if topk_candidates else default_topk_candidates(num_layers)
    candidates = [int(min(max(1, k), num_layers)) for k in candidates]
    candidates = list(dict.fromkeys(candidates))

    # Compute validation probabilities for every ranked layer once. The previous
    # implementation recomputed the top-4 layers again for top-8, top-12, etc.
    y_val, ds_val, ranked_prob_matrix = predict_layer_probabilities_fast(
        ranked, val_representations, val_labels, val_dataset_ids, batch_size, device
    )

    best: Optional[Dict[str, Any]] = None
    sweep_rows: List[Dict[str, Any]] = []
    for k in candidates:
        selected_artifacts = ranked[:k]
        prob_matrix = ranked_prob_matrix[:, :k]
        stacker = None
        if method == "stacking":
            stacker = fit_logistic_stacker(
                prob_matrix,
                y_val,
                C=stacking_C,
                max_iter=stacking_max_iter,
                class_weight=stacking_class_weight,
                seed=seed,
            )
        preds, probs = ensemble_predict(prob_matrix, method=method, stacker=stacker, threshold=threshold)
        metrics = binary_metrics(y_val, preds, probs, ds_val)
        val_score = metrics.get("f1_macro_per_dataset", metrics["f1_macro"])
        row = {
            "k": int(k),
            "selected_layers": [int(a["layer"]) for a in selected_artifacts],
            "val_f1": float(val_score),
            "metrics": metrics,
        }
        sweep_rows.append(row)
        if best is None or val_score > best["val_f1"]:
            best = {
                "method": method,
                "k": int(k),
                "selected_layers": row["selected_layers"],
                "selected_artifacts": selected_artifacts,
                "stacker": stacker,
                "val_f1": float(val_score),
                "val_metrics": metrics,
            }
    if best is None:
        raise ValueError("Could not fit ensemble: no candidate k values")
    best["layer_ranking"] = [
        {"layer": int(a["layer"]), "val_f1": float(a.get("val_f1", a.get("val_metrics", {}).get("val_f1", 0.0)))}
        for a in ranked
    ]
    best["topk_sweep"] = sweep_rows
    return best


def evaluate_ensemble_artifact(
    ensemble_artifact: Dict[str, Any],
    representations: Sequence[dict],
    labels: Sequence[int],
    dataset_ids: Optional[Sequence[int]],
    batch_size: int,
    device: torch.device,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    selected_layers = [int(x) for x in ensemble_artifact["selected_layers"]]
    all_artifacts = ensemble_artifact.get("layer_artifacts") or ensemble_artifact.get("selected_artifacts")
    if not all_artifacts:
        raise ValueError("Ensemble artifact does not contain layer artifacts/models.")
    by_layer = {int(a["layer"]): a for a in all_artifacts}
    selected_artifacts = [by_layer[layer] for layer in selected_layers]
    method = ensemble_artifact.get("ensemble_method", ensemble_artifact.get("method", "topk_average"))
    y_true, ds_ids, prob_matrix = predict_layer_probabilities_fast(selected_artifacts, representations, labels, dataset_ids, batch_size, device)
    preds, probs = ensemble_predict(
        prob_matrix,
        method=method,
        stacker=ensemble_artifact.get("stacker"),
        threshold=float(ensemble_artifact.get("threshold", 0.5) if threshold is None else threshold),
    )
    metrics = binary_metrics(y_true, preds, probs, ds_ids)
    metrics["selected_layers"] = selected_layers
    metrics["ensemble_method"] = method
    metrics["k"] = len(selected_layers)
    return {"y_true": y_true, "predictions": preds, "probs": probs, "dataset_ids": ds_ids, "metrics": metrics}


def convert_json(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: convert_json(v) for k, v in obj.items() if k not in {"model", "stacker", "layer_artifacts", "selected_artifacts"}}
    if isinstance(obj, list):
        return [convert_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [convert_json(x) for x in obj]
    return obj
