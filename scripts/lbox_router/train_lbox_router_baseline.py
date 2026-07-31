#!/usr/bin/env python3
"""Train a fixed pre-generation MLP router from roster solve annotations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def load_features(feature_dir: Path, split: str, feature: str) -> tuple[list[str], np.ndarray]:
    if feature == "hs_mean":
        array_path = feature_dir / f"lbox_{split}_hs_mean.npy"
        ids_path = feature_dir / f"lbox_{split}_hs_ids.json"
    else:
        array_path = feature_dir / f"lbox_{split}_encoder.npy"
        ids_path = feature_dir / f"lbox_{split}_encoder_ids.json"
    return [str(item) for item in json.loads(ids_path.read_text())], np.load(array_path).astype(np.float32)


def routing_target(label: dict[str, Any], bank: dict[str, Any]) -> list[float]:
    models = bank["models"]
    target = [0.0] * len(models)
    n_solved = int(label.get("n_solved", 0))
    if n_solved >= int(bank["shared_min_solved"]):
        target[-1] = 1.0
    elif 0 < n_solved <= int(bank["specialist_max_solved"]):
        per_expert = label.get("per_expert") or {}
        for index, model in enumerate(models[:-1]):
            target[index] = float(per_expert.get(model["source_expert_id"], 0))
    return target


def stratified_split(labels: list[dict[str, Any]], seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    heldout_indices: list[int] = []
    n_solved = np.asarray([int(label.get("n_solved", 0)) for label in labels])
    for value in sorted(set(n_solved.tolist())):
        if value == 0:
            continue
        indices = np.flatnonzero(n_solved == value)
        rng.shuffle(indices)
        heldout_size = max(1, int(round(0.2 * len(indices))))
        heldout_indices.extend(indices[:heldout_size].tolist())
        train_indices.extend(indices[heldout_size:].tolist())
    return np.asarray(sorted(train_indices)), np.asarray(sorted(heldout_indices))


class Router(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, experts: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, experts),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def task_name(row: dict[str, Any]) -> str:
    if row.get("task_type") == "casename":
        return f"casename_{row.get('casetype')}"
    return str(row.get("task_type"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-config", type=Path, required=True)
    parser.add_argument("--bank", choices=["low7_high8", "low5_high6"], required=True)
    parser.add_argument("--feature", choices=["hs_mean", "encoder"], required=True)
    parser.add_argument("--feature-dir", type=Path, default=Path("results/embed_viz_test"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for router training")
    bank = json.loads(args.bank_config.read_text(encoding="utf-8"))[args.bank]
    models = bank["models"]
    feature_ids, features = load_features(args.feature_dir, "train", args.feature)
    label_rows = load_jsonl(Path("results/lbox_binning_seed20210311/binning_labels.jsonl"))
    labels_by_id = {str(row["id"]): row for row in label_rows}
    if len(feature_ids) != 46019 or any(item_id not in labels_by_id for item_id in feature_ids):
        raise RuntimeError("Feature IDs do not fully align with the 46,019 roster labels")
    ordered_labels = [labels_by_id[item_id] for item_id in feature_ids]
    targets = np.asarray([routing_target(label, bank) for label in ordered_labels], dtype=np.float32)
    train_indices, heldout_indices = stratified_split(ordered_labels)

    # Requested protocol: compute normalization statistics over all 46,019 inputs.
    mean = features.mean(0, keepdims=True)
    std = features.std(0, keepdims=True) + 1e-6
    features = (features - mean) / std
    train_ids = [feature_ids[index] for index in train_indices]
    heldout_ids = [feature_ids[index] for index in heldout_indices]
    train_x = features[train_indices]
    heldout_x = features[heldout_indices]
    train_y = targets[train_indices]
    heldout_y = targets[heldout_indices]
    device = torch.device("cuda")
    train_tensor = torch.from_numpy(train_x).to(device)
    label_tensor = torch.from_numpy(train_y).to(device)
    heldout_tensor = torch.from_numpy(heldout_x).to(device)
    logits = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        router = Router(train_x.shape[1], args.hidden, len(models), args.dropout).to(device)
        optimizer = torch.optim.AdamW(
            router.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()
        for epoch in range(args.epochs):
            router.train()
            permutation = torch.randperm(len(train_tensor), device=device)
            total_loss = 0.0
            for offset in range(0, len(train_tensor), args.batch_size):
                indices = permutation[offset:offset + args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(router(train_tensor[indices]), label_tensor[indices])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(indices)
            if epoch == 0 or (epoch + 1) % 20 == 0:
                print(f"seed={seed} epoch={epoch + 1} loss={total_loss / len(train_tensor):.6f}", flush=True)
        router.eval()
        with torch.no_grad():
            logits.append(router(heldout_tensor).float().cpu().numpy())
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in router.state_dict().items()},
            args.output_dir / f"router_seed{seed}.safetensors",
        )

    ensemble = np.mean(logits, axis=0)
    top1 = ensemble.argmax(1)
    top2 = np.argsort(ensemble, axis=1)[:, -2:]
    top1_accuracy = float(100.0 * heldout_y[np.arange(len(heldout_y)), top1].mean())
    top2_accuracy = float(100.0 * heldout_y[np.arange(len(heldout_y))[:, None], top2].max(1).mean())
    best_single = float(100.0 * heldout_y.mean(0).max())
    oracle = float(100.0 * (heldout_y.sum(1) > 0).mean())
    source_rows = {str(row["id"]): row for row in load_jsonl(Path("export/lbox/lbox_train.jsonl"))}
    task_scores: dict[str, float] = {}
    for task in ("casename_civil", "casename_criminal", "statute"):
        indices = [i for i, item_id in enumerate(heldout_ids) if task_name(source_rows[item_id]) == task]
        task_scores[task] = float(100.0 * heldout_y[indices, top1[indices]].mean())
    selection = Counter(models[index]["name"] for index in top1.tolist())
    metrics = {
        "bank": args.bank,
        "bank_label": bank["label"],
        "feature": args.feature,
        "architecture": {
            "hidden": args.hidden,
            "layers": 2,
            "epochs": args.epochs,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "seeds": args.seeds,
        },
        "train_examples": len(train_ids),
        "heldout_examples": len(heldout_ids),
        "normalization_examples": len(feature_ids),
        "excluded_all_unsolved_examples": len(feature_ids) - len(train_ids) - len(heldout_ids),
        "best_single_accuracy": best_single,
        "oracle_any_expert_accuracy": oracle,
        "top1_accuracy": top1_accuracy,
        "top2_union_accuracy": top2_accuracy,
        "top1_by_task": task_scores,
        "top1_selection_counts": dict(selection.most_common()),
        "experts": models,
    }
    np.savez(args.output_dir / "normalizer.npz", mean=mean, std=std)
    (args.output_dir / "split_ids.json").write_text(
        json.dumps({"train": train_ids, "heldout": heldout_ids}) + "\n", encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
