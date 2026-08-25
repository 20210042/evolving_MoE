#!/usr/bin/env python3
"""Train an ACC router for exclusive top-1 expert selection.

Rows may have several solving experts.  Rather than choosing an arbitrary
single label, the set loss maximizes the total probability assigned to any
solver.  All-fail and all-pass rows contain no useful routing decision and are
excluded from optimization.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_acc_soft_router import SoftRouter, embed, load_jsonl


def solve_targets(rows: list[dict], experts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(
        [[float(r["per_expert"].get(e, 0)) for e in experts] for r in rows],
        dtype=np.float32,
    )
    return targets, targets.sum(1)


def set_mass_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Negative log probability that a categorical draw lands in the solver set."""
    solver_mass = (logits.softmax(-1) * targets).sum(-1)
    return -solver_mass.clamp_min(1e-8).log().mean()


def metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probs = logits.softmax(-1)
    mass = (probs * targets).sum(-1)
    top1 = targets.gather(1, probs.argmax(-1, keepdim=True)).squeeze(1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1)
    return {
        "set_loss": float(-mass.clamp_min(1e-8).log().mean().item()),
        "solver_probability_mass": float(mass.mean().item()),
        "top1_solver_hit": float(top1.mean().item()),
        "entropy": float(entropy.mean().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="export/acc_seed20211004/acc_train.jsonl")
    ap.add_argument("--test-jsonl", default="export/acc_seed20211004/acc_test.jsonl")
    ap.add_argument("--train-labels", default="results/acc/seed20211004/binning_train_full.binned.jsonl")
    ap.add_argument("--test-labels", default="results/acc/seed20211004/binning_test_full.binned.jsonl")
    ap.add_argument("--embedding-cache-dir", default="checkpoints/router/acc_seed20211004_soft12")
    ap.add_argument("--output-dir", default="checkpoints/router/acc_seed20211004_top1_set")
    ap.add_argument("--encoder", default="google/embeddinggemma-300m")
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--embed-max-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20211004)
    ap.add_argument("--wandb-project", default="acc-seed20211004-top1-router")
    ap.add_argument("--wandb-entity", default="jongbin-kr-skiml_moe")
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_src, train_lab = load_jsonl(Path(a.train_jsonl)), load_jsonl(Path(a.train_labels))
    test_src, test_lab = load_jsonl(Path(a.test_jsonl)), load_jsonl(Path(a.test_labels))
    train_by_id = {str(r["id"]): r for r in train_lab}
    test_by_id = {str(r["id"]): r for r in test_lab}
    if set(train_by_id) != {str(r["id"]) for r in train_src}:
        raise ValueError("train source/label ID sets do not match")
    if set(test_by_id) != {str(r["id"]) for r in test_src}:
        raise ValueError("test source/label ID sets do not match")
    train_lab = [train_by_id[str(r["id"])] for r in train_src]
    test_lab = [test_by_id[str(r["id"])] for r in test_src]
    experts = list(train_lab[0]["per_expert"])
    if set(experts) != set(test_lab[0]["per_expert"]):
        raise ValueError("train/test expert sets differ")

    y_all, n_all = solve_targets(train_lab, experts)
    y_test, n_test = solve_targets(test_lab, experts)
    n_experts = len(experts)
    contested = (n_all > 0) & (n_all < n_experts)
    contested_test = (n_test > 0) & (n_test < n_experts)

    cache = Path(a.embedding_cache_dir)
    x_all = embed(train_src, a.encoder, a.embed_batch_size, a.embed_max_length, cache / "train_embeddings.npy")
    x_test = embed(test_src, a.encoder, a.embed_batch_size, a.embed_max_length, cache / "test_embeddings.npy")

    rng = np.random.default_rng(a.seed)
    eligible = np.flatnonzero(contested)
    rng.shuffle(eligible)
    n_val = max(1, round(len(eligible) * a.val_fraction))
    va, tr = eligible[:n_val], eligible[n_val:]
    mu, sd = x_all[tr].mean(0), x_all[tr].std(0) + 1e-6

    def norm(x: np.ndarray) -> np.ndarray:
        return ((x - mu) / sd).astype(np.float32)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(norm(x_all[tr])), torch.from_numpy(y_all[tr])),
        batch_size=a.batch_size,
        shuffle=True,
    )
    xv = torch.from_numpy(norm(x_all[va])).to(dev)
    yv = torch.from_numpy(y_all[va]).to(dev)
    xt = torch.from_numpy(norm(x_test[contested_test])).to(dev)
    yt = torch.from_numpy(y_test[contested_test]).to(dev)

    net = SoftRouter(x_all.shape[1], a.hidden_dim, n_experts, a.dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-3)
    import wandb

    run = wandb.init(
        project=a.wandb_project,
        entity=a.wandb_entity,
        name="acc_seed20211004_top1_set",
    )
    best_hit, best_loss, stale = -1.0, float("inf"), 0
    for epoch in range(1, a.epochs + 1):
        net.train()
        train_total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = set_mass_loss(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_total += loss.item() * len(xb)
        net.eval()
        with torch.no_grad():
            vm = metrics(net(xv), yv)
        train_loss = train_total / len(tr)
        wandb.log({"epoch": epoch, "train/set_loss": train_loss, **{f"val/{k}": v for k, v in vm.items()}})
        print(
            f"epoch={epoch} train={train_loss:.5f} val_loss={vm['set_loss']:.5f} "
            f"mass={vm['solver_probability_mass']:.4f} top1={vm['top1_solver_hit']:.4f} "
            f"entropy={vm['entropy']:.4f}",
            flush=True,
        )
        improved = vm["top1_solver_hit"] > best_hit + 1e-6 or (
            abs(vm["top1_solver_hit"] - best_hit) <= 1e-6 and vm["set_loss"] < best_loss - 1e-5
        )
        if improved:
            best_hit, best_loss, stale = vm["top1_solver_hit"], vm["set_loss"], 0
            torch.save(net.state_dict(), out / "router_state.pt")
        else:
            stale += 1
            if stale >= a.patience:
                break

    net.load_state_dict(torch.load(out / "router_state.pt", map_location=dev, weights_only=True))
    net.eval()
    with torch.no_grad():
        test_metrics = metrics(net(xt), yt)
        all_test_logits = net(torch.from_numpy(norm(x_test)).to(dev))
        route_counts = torch.bincount(all_test_logits.argmax(-1), minlength=n_experts).cpu().tolist()
    artifact = {
        "version": 1,
        "objective": "negative_log_solver_set_mass",
        "route_policy": "argmax_single_expert",
        "train_policy": "contested_only_1_to_E_minus_1",
        "experts": experts,
        "encoder": a.encoder,
        "input_dim": int(x_all.shape[1]),
        "hidden_dim": a.hidden_dim,
        "dropout": a.dropout,
        "mean": mu.tolist(),
        "std": sd.tolist(),
        "seed": a.seed,
        "train_rows": len(train_src),
        "train_contested": int(contested.sum()),
        "train_all_fail": int((n_all == 0).sum()),
        "train_all_pass": int((n_all == n_experts).sum()),
        "test_rows": len(test_src),
        "test_contested": int(contested_test.sum()),
        "test_metrics_contested": test_metrics,
        "test_route_counts": dict(zip(experts, route_counts)),
    }
    (out / "router_config.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    wandb.log({f"test_contested/{k}": v for k, v in test_metrics.items()})
    run.summary.update(test_metrics)
    run.finish()
    print(json.dumps(artifact, indent=2), flush=True)


if __name__ == "__main__":
    main()
