#!/usr/bin/env python3
"""분할의 '라우팅 가능성(routability)'을 조건별로 비교한다.

질문: 같은 입력 특징·같은 라우터·같은 프로토콜에서, **어떤 분할이 라우터에게 더 잘 보이나?**

지금까지의 적합도는 union(oracle 커버리지)을 키웠는데 union은 라우터가 실현할 수 없는 양이다.
병목이 라우팅이므로, 분할을 평가할 잣대는 커버리지가 아니라 "헤드룸을 얼마나 실현할 수 있나"여야
한다. 그 실현율을 조건별로 재는 것이 이 스크립트다.

  realization = (routed_top1 - best_single) / (oracle_union - best_single)

  0 이하 = 라우팅이 최고 단일 전문가보다 나을 게 없음(분할이 라우터에게 안 보임)
  1.0    = oracle 라우팅 달성

프로토콜: eval solve 매트릭스 위 K-fold CV(held-out). 라벨이 eval에만 있는 조건들
(Random/Human-prior)까지 동일하게 비교하기 위한 기존 관행을 따른다.
라우터는 README_router.md의 BCE 레시피(expert별 독립 이진분류) 그대로.

Usage: python scripts/router_routability.py --feat emb
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

# 같은 926문제 · 같은 특징 위에서 분할만 다른 세 조건.
CONDS = {
    "Evolved(cap10)": "results/qasc/seed20210211/inference_validation_lora13.binned.jsonl",
    "Random":         "results/qasc/seed20210211/inference_validation_rndmoe.binned.jsonl",
    "Human-prior":    "results/qasc/seed20210211/inference_validation_hpmoe.binned.jsonl",
    "Evolved(cap7)":  "results/qasc/seed20210211/inference_validation_cap7moe.binned.jsonl",
}

ap = argparse.ArgumentParser()
ap.add_argument("--feat", default="emb", choices=["emb", "hs_last", "hs_mean"])
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--hid", type=int, default=1024)
ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--wd", type=float, default=1e-3)
ap.add_argument("--out", default="results/qasc/seed20210211/routability.json")
A = ap.parse_args()

sp = rc.spec("qasc")


def feature_matrix(order):
    """order(문제 id 순서)에 맞춘 입력 특징. expert 정보가 들어가지 않는 문제단위 특징만 쓴다."""
    if A.feat == "emb":
        npy, ids_path = rc.emb_paths(sp, "eval")
    else:
        npy = rc.feat_path(sp, sp.eval_split, A.feat)
        ids_path = rc.feat_path(sp, sp.eval_split, "hs_ids")
    F = np.load(npy)
    ids = rc.load_ids(ids_path)
    idx = {p: i for i, p in enumerate(ids)}
    missing = [p for p in order if p not in idx]
    if missing:
        raise SystemExit(f"특징에 없는 id {len(missing)}개 (예: {missing[:3]})")
    return np.array([F[idx[p]] for p in order], np.float32)


class MLP(nn.Module):
    def __init__(self, d, E):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, A.hid), nn.ReLU(), nn.Dropout(A.dropout), nn.Linear(A.hid, E)
        )

    def forward(self, x):
        return self.net(x)


def fit_predict(Xtr, Ytr, Xte, seed):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((Xtr - mu) / sd)
    yt = torch.tensor(Ytr)
    xe = torch.tensor((Xte - mu) / sd)
    net = MLP(Xtr.shape[1], Ytr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=A.wd)
    lossf = nn.BCEWithLogitsLoss()
    n = len(xt)
    for _ in range(A.epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            lossf(net(xt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(xe).numpy()


def auc(scores, labels):
    """rank 기반 AUC (sklearn 의존 없이)."""
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    r = scores.argsort().argsort().astype(float) + 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def run(name, path):
    binning = rc.load_binning(ROOT / path)
    ex = sorted(next(iter(binning.values()))["per_expert"])
    order = [p for p in rc.load_ids(ROOT / sp.src[sp.eval_split]) if p in binning]
    S = np.array([[binning[p]["per_expert"].get(e, 0) for e in ex] for p in order], np.float32)
    X = feature_matrix(order)
    n, E = S.shape

    best_single, oracle = rc.baselines(S)
    rng = np.random.default_rng(0)
    fold = rng.permutation(n) % A.folds
    logits = np.zeros_like(S)
    for f in range(A.folds):
        te, tr = fold == f, fold != f
        logits[te] = np.mean([fit_predict(X[tr], S[tr], X[te], s) for s in A.seeds], axis=0)

    top1 = logits.argmax(1)
    routed = 100 * S[np.arange(n), top1].mean()
    order2 = np.argsort(-logits, axis=1)[:, :2]
    routed2 = 100 * np.maximum(S[np.arange(n), order2[:, 0]], S[np.arange(n), order2[:, 1]]).mean()
    head = oracle - best_single
    realz = (routed - best_single) / head if head > 1e-9 else float("nan")
    aucs = [auc(logits[:, j], S[:, j]) for j in range(E)]

    contested = (S.sum(1) > 0) & (S.sum(1) < E)
    c_routed = 100 * S[np.arange(n), top1][contested].mean() if contested.sum() else float("nan")

    print(f"\n=== {name}  (E={E}, feat={A.feat}) ===")
    print(f"  best-single {best_single:6.2f}   oracle-union {oracle:6.2f}   헤드룸 {head:5.2f}pp")
    print(f"  routed top-1 {routed:6.2f}  → 실현율 {realz:+.3f}")
    print(f"  routed top-2(union) {routed2:6.2f}")
    print(f"  per-expert AUC 평균 {np.nanmean(aucs):.4f}  (min {np.nanmin(aucs):.3f} max {np.nanmax(aucs):.3f})")
    print(f"  contested {int(contested.sum())}문제에서 routed 적중 {c_routed:.2f}%")
    return dict(condition=name, experts=E, feat=A.feat,
                best_single=float(best_single), oracle=float(oracle), headroom=float(head),
                routed_top1=float(routed), routed_top2_union=float(routed2),
                realization=float(realz), auc_mean=float(np.nanmean(aucs)),
                contested_n=int(contested.sum()), contested_routed=float(c_routed))


out = [run(k, v) for k, v in CONDS.items() if (ROOT / v).is_file()]
dst = ROOT / A.out
dst.parent.mkdir(parents=True, exist_ok=True)
try:                       # 이전 실행이 중간에 죽어 깨진 파일을 남겼어도 진행한다
    prev = json.load(open(dst)) if dst.is_file() else {}
except json.JSONDecodeError:
    prev = {}
prev[A.feat] = out
json.dump(prev, open(dst, "w"), indent=2, ensure_ascii=False)
print(f"\n-> {dst}")
