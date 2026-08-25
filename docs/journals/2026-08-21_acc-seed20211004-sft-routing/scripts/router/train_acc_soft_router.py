#!/usr/bin/env python3
"""Train a 12-way soft router from normalized ACC solve labels.

For a problem solved by n experts, each solver receives target mass 1/n.
All-fail rows have no identifiable correct route and are excluded from the loss.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class SoftRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_experts: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_experts),
        )

    def forward(self, x):
        return self.net(x)


def normalized_solve_targets(rows: list[dict], experts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    solved = np.asarray([[float(r["per_expert"].get(e, 0)) for e in experts] for r in rows], dtype=np.float32)
    counts = solved.sum(1)
    targets = np.divide(solved, counts[:, None], out=np.zeros_like(solved), where=counts[:, None] > 0)
    return targets, counts


def metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probs = logits.softmax(-1)
    expected_mass = (probs * (targets > 0)).sum(-1).mean().item()
    top1 = targets.gather(1, probs.argmax(-1, keepdim=True)).gt(0).float().mean().item()
    kl = torch.nn.functional.kl_div(probs.log(), targets, reduction="batchmean").item()
    return {"loss": kl, "solver_probability_mass": expected_mass, "top1_solver_hit": top1}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def embed(rows: list[dict], model_name: str, batch_size: int, max_length: int, cache: Path) -> np.ndarray:
    ids = [str(r["id"]) for r in rows]
    ids_path = cache.with_suffix(".ids.json")
    if cache.is_file() and ids_path.is_file() and json.loads(ids_path.read_text()) == ids:
        return np.load(cache)
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=(torch.bfloat16 if device == "cuda" else torch.float32)
    ).to(device).eval()
    texts = [r["instruction"] + (f"\n\nStarter code:\n{r['starter_code']}" if r.get("starter_code") else "") for r in rows]
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)
        with torch.inference_mode(): hidden = model(**batch).last_hidden_state
        mask = batch.attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        chunks.append(pooled.cpu().numpy())
        if start % (batch_size * 20) == 0:
            print(f"embedded={min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    arr = np.concatenate(chunks).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    ids_path.write_text(json.dumps(ids), encoding="utf-8")
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="export/acc_seed20211004/acc_train.jsonl")
    ap.add_argument("--test-jsonl", default="export/acc_seed20211004/acc_test.jsonl")
    ap.add_argument("--train-labels", default="results/acc/seed20211004/binning_train_full.binned.jsonl")
    ap.add_argument("--test-labels", default="results/acc/seed20211004/binning_test_full.binned.jsonl")
    ap.add_argument("--output-dir", default="checkpoints/router/acc_seed20211004_soft12")
    ap.add_argument("--encoder", default="google/embeddinggemma-300m")
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--embed-max-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20211004)
    ap.add_argument("--wandb-project", default="acc-seed20211004-soft-router")
    ap.add_argument("--wandb-entity", default="jongbin-kr-skiml_moe")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    train_src, train_lab = load_jsonl(Path(a.train_jsonl)), load_jsonl(Path(a.train_labels))
    test_src, test_lab = load_jsonl(Path(a.test_jsonl)), load_jsonl(Path(a.test_labels))
    lab_by_id = {str(r["id"]): r for r in train_lab}
    test_by_id = {str(r["id"]): r for r in test_lab}
    if set(lab_by_id) != {str(r["id"]) for r in train_src} or set(test_by_id) != {str(r["id"]) for r in test_src}:
        raise ValueError("source/label ID sets do not match")
    train_lab = [lab_by_id[str(r["id"])] for r in train_src]
    test_lab = [test_by_id[str(r["id"])] for r in test_src]
    experts = list(train_lab[0]["per_expert"])
    if set(experts) != set(test_lab[0]["per_expert"]):
        raise ValueError("train/test expert sets differ")
    y_all, n_all = normalized_solve_targets(train_lab, experts)
    y_test, n_test = normalized_solve_targets(test_lab, experts)
    keep, keep_test = n_all > 0, n_test > 0
    X_all = embed(train_src, a.encoder, a.embed_batch_size, a.embed_max_length, out / "train_embeddings.npy")
    X_test = embed(test_src, a.encoder, a.embed_batch_size, a.embed_max_length, out / "test_embeddings.npy")

    rng = np.random.default_rng(a.seed)
    eligible = np.flatnonzero(keep); rng.shuffle(eligible)
    n_val = max(1, round(len(eligible) * a.val_fraction))
    va, tr = eligible[:n_val], eligible[n_val:]
    mu, sd = X_all[tr].mean(0), X_all[tr].std(0) + 1e-6
    norm = lambda x: ((x - mu) / sd).astype(np.float32)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(TensorDataset(torch.from_numpy(norm(X_all[tr])), torch.from_numpy(y_all[tr])),
                        batch_size=a.batch_size, shuffle=True)
    Xv, yv = torch.from_numpy(norm(X_all[va])).to(dev), torch.from_numpy(y_all[va]).to(dev)
    Xt, yt = torch.from_numpy(norm(X_test[keep_test])).to(dev), torch.from_numpy(y_test[keep_test]).to(dev)
    net = SoftRouter(X_all.shape[1], a.hidden_dim, len(experts), a.dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-3)
    import wandb
    run = wandb.init(project=a.wandb_project, entity=a.wandb_entity, name="acc_seed20211004_soft12")
    best, stale = float("inf"), 0
    for epoch in range(1, a.epochs + 1):
        net.train(); total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = -(yb * net(xb).log_softmax(-1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); total += loss.item() * len(xb)
        net.eval()
        with torch.no_grad(): vm = metrics(net(Xv), yv)
        wandb.log({"epoch": epoch, "train/loss": total / len(tr), **{f"val/{k}": v for k, v in vm.items()}})
        print(f"epoch={epoch} train={total/len(tr):.5f} val_kl={vm['loss']:.5f} mass={vm['solver_probability_mass']:.4f} top1={vm['top1_solver_hit']:.4f}", flush=True)
        if vm["loss"] < best - 1e-5:
            best, stale = vm["loss"], 0
            torch.save(net.state_dict(), out / "router_state.pt")
        else:
            stale += 1
            if stale >= a.patience: break
    net.load_state_dict(torch.load(out / "router_state.pt", map_location=dev, weights_only=True)); net.eval()
    with torch.no_grad(): test_metrics = metrics(net(Xt), yt)
    artifact = {
        "version": 1, "target": "solved/n_solved", "all_fail_policy": "excluded_from_loss",
        "experts": experts, "encoder": a.encoder, "input_dim": int(X_all.shape[1]),
        "hidden_dim": a.hidden_dim, "dropout": a.dropout, "mean": mu.tolist(), "std": sd.tolist(),
        "seed": a.seed, "train_rows": len(train_src), "train_solvable": int(keep.sum()),
        "train_all_fail": int((~keep).sum()), "test_rows": len(test_src),
        "test_solvable": int(keep_test.sum()), "test_all_fail": int((~keep_test).sum()),
        "test_metrics_solvable": test_metrics,
    }
    (out / "router_config.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    wandb.log({f"test/{k}": v for k, v in test_metrics.items()}); run.summary.update(artifact["test_metrics_solvable"]); run.finish()
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
