#!/usr/bin/env python3
"""Train an LBox router to reproduce Gemma4 legal-category tags."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file

from train_lbox_router_baseline import Router


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def stratified_split(targets: np.ndarray, num_classes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    heldout: list[int] = []
    for target in range(num_classes):
        indices = np.flatnonzero(targets == target)
        if len(indices) == 0:
            continue
        rng.shuffle(indices)
        heldout_size = max(1, int(round(0.2 * len(indices))))
        heldout.extend(indices[:heldout_size].tolist())
        train.extend(indices[heldout_size:].tolist())
    return np.asarray(sorted(train)), np.asarray(sorted(heldout))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=Path("results/embed_viz_test"))
    parser.add_argument(
        "--tags-file",
        type=Path,
        default=Path(
            "results/lbox_legal_category_tags/"
            "gemma4_a4b_family_patent_merged/lbox_train_legal_categories.jsonl"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for legal-category router training")

    feature_ids = [
        str(value)
        for value in json.loads(
            (args.feature_dir / "lbox_train_hs_ids.json").read_text(encoding="utf-8")
        )
    ]
    features = np.load(args.feature_dir / "lbox_train_hs_mean.npy").astype(np.float32)
    if len(features) != len(feature_ids):
        raise RuntimeError("Train feature array and ID list have different lengths")

    tag_rows = load_jsonl(args.tags_file)
    tags_by_id = {str(row["id"]): row for row in tag_rows}
    if any(item_id not in tags_by_id for item_id in feature_ids):
        raise RuntimeError("Train features do not fully align with legal-category tags")

    ordered_tags = [tags_by_id[item_id] for item_id in feature_ids]
    categories = sorted({str(row["primary_category"]) for row in ordered_tags})
    category_to_index = {category: index for index, category in enumerate(categories)}
    targets = np.asarray(
        [category_to_index[str(row["primary_category"])] for row in ordered_tags],
        dtype=np.int64,
    )

    # Keep the current LBox router protocol: z-score from all 46,019 train inputs.
    mean = features.mean(0, keepdims=True)
    std = features.std(0, keepdims=True) + 1e-6
    features = (features - mean) / std

    train_indices, heldout_indices = stratified_split(targets, len(categories), args.seed)
    device = torch.device("cuda")
    train_x = torch.from_numpy(features[train_indices]).to(device)
    train_y = torch.from_numpy(targets[train_indices]).to(device)
    heldout_x = torch.from_numpy(features[heldout_indices]).to(device)
    heldout_y = targets[heldout_indices]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    router = Router(features.shape[1], args.hidden, len(categories), args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        router.train()
        permutation = torch.randperm(len(train_x), device=device)
        total_loss = 0.0
        for offset in range(0, len(train_x), args.batch_size):
            indices = permutation[offset : offset + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(router(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(
                f"seed={args.seed} epoch={epoch + 1} "
                f"loss={total_loss / len(train_x):.6f}",
                flush=True,
            )

    router.eval()
    with torch.inference_mode():
        logits = router(heldout_x).float().cpu().numpy()
    predicted = logits.argmax(1)
    accuracy = float(100.0 * (predicted == heldout_y).mean())
    confusion = np.zeros((len(categories), len(categories)), dtype=np.int64)
    for gold, pred in zip(heldout_y.tolist(), predicted.tolist()):
        confusion[gold, pred] += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in router.state_dict().items()},
        args.output_dir / f"router_seed{args.seed}.safetensors",
    )
    np.savez(args.output_dir / "normalizer.npz", mean=mean, std=std)
    (args.output_dir / "split_ids.json").write_text(
        json.dumps(
            {
                "train": [feature_ids[index] for index in train_indices],
                "heldout": [feature_ids[index] for index in heldout_indices],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = {
        "bank": "legal_category_gemma4_merged",
        "feature": "hs_mean",
        "loss": "cross_entropy",
        "tags_file": str(args.tags_file),
        "categories": categories,
        "architecture": {
            "hidden": args.hidden,
            "layers": 2,
            "epochs": args.epochs,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "train_examples": len(train_indices),
        "heldout_examples": len(heldout_indices),
        "normalization_examples": len(features),
        "heldout_accuracy": accuracy,
        "confusion_matrix_gold_by_predicted": confusion.tolist(),
        "train_counts_by_category": {
            category: int((targets[train_indices] == index).sum())
            for category, index in category_to_index.items()
        },
        "heldout_counts_by_category": {
            category: int((heldout_y == index).sum())
            for category, index in category_to_index.items()
        },
        "heldout_accuracy_by_category": {
            category: float(100.0 * (predicted[heldout_y == index] == index).mean())
            for category, index in category_to_index.items()
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
