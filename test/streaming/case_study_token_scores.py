import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import argparse
import csv
import json
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from streaming_extractor import StreamingRepresentationExtractor
from train.salr_modules import (
    ensemble_predict,
    predict_layer_probabilities,
)
from utils.config import MODEL_CONFIGS


def load_general_salr(model_name: str, variant: str):
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
        layer_artifacts,
        representations,
        dummy_labels,
        None,
        batch_size=max(len(representations), 1),
        device=device,
    )

    preds, probs = ensemble_predict(
        prob_matrix,
        method=salr_artifact.get("ensemble_method", "topk_average"),
        stacker=salr_artifact.get("stacker"),
        threshold=float(salr_artifact.get("threshold", 0.5)),
    )
    return preds, probs


def pad_prefix_batch(prefixes: List[List[int]], pad_token_id: int):
    max_len = max(len(x) for x in prefixes)

    input_ids_batch = []
    attention_mask_batch = []

    for ids in prefixes:
        pad_len = max_len - len(ids)
        input_ids_batch.append(ids + [pad_token_id] * pad_len)
        attention_mask_batch.append([1] * len(ids) + [0] * pad_len)

    return input_ids_batch, attention_mask_batch


def find_sample(dataset, sample_index: Optional[int], unique_id: Optional[str]):
    if unique_id is not None:
        for sample in dataset:
            if str(sample.get("unique_id")) == str(unique_id):
                return sample
        raise ValueError(f"Cannot find sample with unique_id={unique_id}")

    if sample_index is None:
        sample_index = 0

    return dataset[int(sample_index)]


def get_assistant_start_token(sample, tokenizer):
    """
    Keep this consistent with evaluate_streaming_salr.py:
    use the number of tokens obtained by applying the chat template to the first
    user message as the assistant start position.
    """
    messages = sample["message"]
    user_text = tokenizer.apply_chat_template(
        [messages[0]],
        tokenize=False,
        add_generation_prompt=False,
    )
    user_ids = tokenizer.encode(user_text, add_special_tokens=False)
    return len(user_ids)


def build_prefixes_for_case_study(
    *,
    input_ids_full: List[int],
    assistant_start_token: int,
    scope: str,
):
    """
    Return the prefix corresponding to each token.

    scope="assistant":
        The score for the k-th assistant token is the score after the model has
        seen the prefix from assistant_start_token up to the current token.
        This is most consistent with the original streaming detection logic.

    scope="full":
        The score for the k-th token is the score after the model has seen the
        prefix from the beginning of the sample up to the current token.
        This is useful for observing how the score changes across the full
        sequence, from the user part to the assistant part.
    """
    prefixes = []
    token_global_indices = []

    if scope == "assistant":
        for end_idx in range(assistant_start_token + 1, len(input_ids_full) + 1):
            prefix = input_ids_full[assistant_start_token:end_idx]
            prefixes.append(prefix)
            token_global_indices.append(end_idx - 1)

    elif scope == "full":
        for end_idx in range(1, len(input_ids_full) + 1):
            prefix = input_ids_full[:end_idx]
            prefixes.append(prefix)
            token_global_indices.append(end_idx - 1)

    else:
        raise ValueError(f"Unknown scope: {scope}")

    return prefixes, token_global_indices


def score_prefixes_in_batches(
    *,
    prefixes: List[List[int]],
    extractor: StreamingRepresentationExtractor,
    salr_artifact,
    tokenizer,
    device,
    batch_size: int,
):
    all_scores = []
    all_preds = []

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    for start in tqdm(range(0, len(prefixes), batch_size), desc="Scoring token prefixes"):
        batch_prefixes = prefixes[start:start + batch_size]

        input_ids_batch, attention_mask_batch = pad_prefix_batch(
            batch_prefixes,
            pad_token_id=pad_token_id,
        )

        reps = extractor.extract_prefixes_batch(
            input_ids_batch,
            attention_mask_batch,
        )

        preds, probs = get_predictions_batch(
            reps,
            salr_artifact,
            device,
        )

        all_preds.extend([int(x) for x in preds])
        all_scores.extend([float(x) for x in probs])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_preds, all_scores


def save_csv(rows: List[Dict[str, Any]], output_csv: str):
    if not rows:
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload: Dict[str, Any], output_json: str):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

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
            return [convert_json(x) for x in obj]
        return obj

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(convert_json(payload), f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--variant", type=str, default="SaLR-base")
    parser.add_argument("--split", type=str, default="thinking_loc")

    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--unique_id", type=str, default=None)

    parser.add_argument(
        "--scope",
        type=str,
        choices=["assistant", "full"],
        default="assistant",
        help="assistant: score assistant prefixes only; full: score full-sequence prefixes",
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/case_study_token_scores",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Token-level Streaming Harmfulness Case Study")
    print("=" * 80)
    print(f"Model:      {args.model}")
    print(f"Variant:    {args.variant}")
    print(f"Split:      {args.split}")
    print(f"Scope:      {args.scope}")
    print(f"Device:     {args.device}")

    dataset = load_dataset("Qwen/Qwen3GuardTest", split=args.split)

    sample = find_sample(
        dataset,
        sample_index=args.sample_index,
        unique_id=args.unique_id,
    )

    model_config = MODEL_CONFIGS[args.model]
    model_path = model_config["model_path"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    salr_artifact = load_general_salr(args.model, args.variant)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )

    input_ids_full = list(sample["input_ids"])
    unsafe_start = int(sample["unsafe_start_index"])
    unsafe_end = int(sample["unsafe_end_index"])
    assistant_start_token = get_assistant_start_token(sample, tokenizer)

    print(f"unique_id:              {sample.get('unique_id')}")
    print(f"total_tokens:           {len(input_ids_full)}")
    print(f"assistant_start_token:  {assistant_start_token}")
    print(f"unsafe_start_index:     {unsafe_start}")
    print(f"unsafe_end_index:       {unsafe_end}")

    prefixes, token_global_indices = build_prefixes_for_case_study(
        input_ids_full=input_ids_full,
        assistant_start_token=assistant_start_token,
        scope=args.scope,
    )

    extractor = None

    try:
        extractor = StreamingRepresentationExtractor(
            model_path,
            device=args.device,
            batch_size=args.batch_size,
            rep_types=[salr_artifact["pooling_type"]],
        )
        extractor.register_hooks()

        preds, scores = score_prefixes_in_batches(
            prefixes=prefixes,
            extractor=extractor,
            salr_artifact=salr_artifact,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
        )

    finally:
        if extractor is not None:
            extractor.remove_hooks()
            del extractor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = []

    for local_i, global_idx in enumerate(token_global_indices):
        token_id = int(input_ids_full[global_idx])
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)

        row = {
            "unique_id": str(sample.get("unique_id")),
            "scope": args.scope,

            # Current token position in the full input_ids sequence
            "global_token_index": int(global_idx),

            # If scope=assistant, this is the token position within the assistant part.
            # If scope=full, this value is negative for the user part.
            "assistant_token_index": int(global_idx - assistant_start_token),

            "token_id": token_id,
            "token_text": token_text,
            "token_text_repr": repr(token_text),

            # Score meaning: harmfulness probability of the prefix ending at the current token
            "harmfulness_score": float(scores[local_i]),
            "harm_pred": int(preds[local_i]),

            "is_before_assistant": bool(global_idx < assistant_start_token),
            "is_assistant_token": bool(global_idx >= assistant_start_token),

            # Mark according to the unsafe span in the original sample.
            # Usually, the unsafe span is [unsafe_start, unsafe_end).
            "is_in_unsafe_span": bool(unsafe_start <= global_idx < unsafe_end),
            "is_at_unsafe_start": bool(global_idx == unsafe_start),
            "is_at_unsafe_end_minus_1": bool(global_idx == unsafe_end - 1),

            "source": sample.get("source"),
            "unsafe_type": sample.get("unsafe_type"),
        }

        rows.append(row)

    sample_id_for_name = str(sample.get("unique_id", args.sample_index or 0)).replace("/", "_")
    base_name = f"{args.model}_{args.variant}_{args.split}_{sample_id_for_name}_{args.scope}"

    output_csv = os.path.join(args.output_dir, f"{base_name}.csv")
    output_json = os.path.join(args.output_dir, f"{base_name}.json")

    payload = {
        "model": args.model,
        "variant": args.variant,
        "split": args.split,
        "sample_index": args.sample_index,
        "unique_id": sample.get("unique_id"),
        "scope": args.scope,
        "total_tokens": len(input_ids_full),
        "assistant_start_token": assistant_start_token,
        "unsafe_start_index": unsafe_start,
        "unsafe_end_index": unsafe_end,
        "source": sample.get("source"),
        "unsafe_type": sample.get("unsafe_type"),
        "message": sample.get("message"),
        "rows": rows,
    }

    save_csv(rows, output_csv)
    save_json(payload, output_json)

    print("\nSaved:")
    print(f"CSV:  {output_csv}")
    print(f"JSON: {output_json}")

    print("\nTop 20 highest-scoring tokens:")
    top_rows = sorted(rows, key=lambda x: x["harmfulness_score"], reverse=True)[:20]
    for r in top_rows:
        print(
            f"idx={r['global_token_index']:>5} "
            f"assist_idx={r['assistant_token_index']:>5} "
            f"score={r['harmfulness_score']:.4f} "
            f"pred={r['harm_pred']} "
            f"unsafe={r['is_in_unsafe_span']} "
            f"token={r['token_text_repr']}"
        )


if __name__ == "__main__":
    main()


"""
Example usage:

```bash
python case_study_token_scores.py \
    --model qwen3-4b \
    --variant SaLR-base \
    --split thinking_loc \
    --sample_index 0 \
    --scope assistant \
    --device cuda \
    --batch_size 16
```

Or specify a particular `unique_id`:

```bash
python case_study_token_scores.py \
    --model qwen3-4b \
    --variant SaLR-base \
    --split thinking_loc \
    --unique_id YOUR_SAMPLE_ID \
    --scope assistant \
    --device cuda \
    --batch_size 16
```

The most important columns in the output CSV are:

```text
global_token_index
assistant_token_index
token_text_repr
harmfulness_score
harm_pred
is_in_unsafe_span
is_at_unsafe_start
is_at_unsafe_end_minus_1
```

The meaning of `harmfulness_score` is:

```text
The probe's harmful probability output when the model has seen the prefix up to the current token.
```

In other words, the score for token `t` is **not** an isolated score for that token alone.  
Instead, it is the cumulative harmfulness score in a streaming setting — i.e., “the harmfulness estimate after reading up to this point.”

If you want the strictest consistency with the original `evaluate_streaming_salr.py`, use:

```bash
--scope assistant
```

If you actually want to inspect how the score evolves across the entire sequence from the user prompt to the assistant response, use:

```bash
--scope full
```

However, if your probe was mainly trained on assistant response representations, `scope=assistant` usually gives a cleaner interpretation.
"""
