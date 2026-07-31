#!/usr/bin/env python3
"""Train a three-way LBox task router over the task-prior LoRA experts."""

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


TASKS = ("casename_civil", "casename_criminal", "statute")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_name(row: dict[str, Any]) -> str:
    if row.get("task_type") == "casename":
        return f"casename_{row.get('casetype')}"
    return str(row.get("task_type"))


def stratified_split(targets: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    heldout: list[int] = []
    for target in range(len(TASKS)):
        indices = np.flatnonzero(targets == target)
        rng.shuffle(indices)
        heldout_size = max(1, int(round(0.2 * len(indices))))
        heldout.extend(indices[:heldout_size].tolist())
        train.extend(indices[heldout_size:].tolist())
    return np.asarray(sorted(train)), np.asarray(sorted(heldout))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=Path("results/embed_viz_test"))
    parser.add_argument("--data-file", type=Path, default=Path("export/lbox/lbox_train.jsonl"))
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
        raise RuntimeError("A CUDA GPU is required for task-prior router training")

    rows = load_jsonl(args.data_file)
    feature_ids = [
        str(value)
        for value in json.loads(
            (args.feature_dir / "lbox_train_hs_ids.json").read_text(encoding="utf-8")
        )
    ]
    features = np.load(args.feature_dir / "lbox_train_hs_mean.npy").astype(np.float32)
    rows_by_id = {str(row["id"]): row for row in rows}
    if len(features) != len(feature_ids) or set(feature_ids) != set(rows_by_id):
        raise RuntimeError("Train features do not align with LBox train rows")
    ordered_rows = [rows_by_id[item_id] for item_id in feature_ids]
    task_to_index = {task: index for index, task in enumerate(TASKS)}
    try:
        targets = np.asarray([task_to_index[task_name(row)] for row in ordered_rows], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"Unsupported LBox task label: {exc}") from exc

    # Match the low5/high6 protocol: normalize from all 46,019 train inputs.
    mean = features.mean(0, keepdims=True)
    std = features.std(0, keepdims=True) + 1e-6
    features = (features - mean) / std
    train_indices, heldout_indices = stratified_split(targets, args.seed)

    device = torch.device("cuda")
    train_x = torch.from_numpy(features[train_indices]).to(device)
    train_y = torch.from_numpy(targets[train_indices]).to(device)
    heldout_x = torch.from_numpy(features[heldout_indices]).to(device)
    heldout_y = targets[heldout_indices]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    router = Router(features.shape[1], args.hidden, len(TASKS), args.dropout).to(device)
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
    confusion = np.zeros((len(TASKS), len(TASKS)), dtype=np.int64)
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
        "bank": "task_prior",
        "feature": "hs_mean",
        "loss": "cross_entropy",
        "tasks": list(TASKS),
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
        "heldout_accuracy_by_task": {
            task: float(100.0 * (predicted[heldout_y == index] == index).mean())
            for index, task in enumerate(TASKS)
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
