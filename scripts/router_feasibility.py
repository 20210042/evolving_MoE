#!/usr/bin/env python3
"""MLP 라우터 feasibility: embeddinggemma 임베딩 → 12 expert 중 top-1 라우팅.

핵심 질문: 임베딩 입력만으로 라우터가 "누가 푸나"를 맞혀서
best-single-expert를 넘어 oracle(union)에 얼마나 다가가나?
(solvability ⊥ 임베딩이면 best-single 근처에서 막힐 것 = hidden-state로 가야 한다는 신호)

train: <ds> train (임베딩+binning) 로 학습
eval : <ds> eval split (임베딩+binning) 로 routed acc
비교 : best-single-expert / oracle(union)
목표지표: routed top-1 acc = 라우팅된 expert가 그 문제를 푸는 비율.

Usage: python scripts/router_feasibility.py --dataset qasc --feat emb
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

REPO = rc.REPO
torch.manual_seed(42); np.random.seed(42)

ap = argparse.ArgumentParser()
rc.add_dataset_arg(ap)
ap.add_argument("--feat", choices=["emb", "hs_last", "hs_mean"], default="emb",
                help="라우터 입력: embeddinggemma(emb) vs base llama hidden-state(hs_last/hs_mean)")
_a = ap.parse_args()
FEAT = _a.feat
sp = rc.spec(_a.dataset)


def feat_sources(sp, feat):
    """(train 특징, train ids, eval 특징, eval ids) 절대경로."""
    if feat == "emb":
        tn, ti = rc.emb_paths(sp, "train")
        vn, vi = rc.emb_paths(sp, "eval")
        return tn, ti, vn, vi
    return (rc.feat_path(sp, sp.train_split, feat), rc.feat_path(sp, sp.train_split, "hs_ids"),
            rc.feat_path(sp, sp.eval_split, feat), rc.feat_path(sp, sp.eval_split, "hs_ids"))


TR_EMB, TR_IDS, VA_EMB, VA_IDS = feat_sources(sp, FEAT)

# --- 데이터 ---
tr_bin, val_bin = rc.labels(sp)
experts = rc.experts(sp)
_, Xtr, Str = rc.align(TR_EMB, TR_IDS, tr_bin, experts)
_, Xva, Sva = rc.align(VA_EMB, VA_IDS, val_bin, experts)
# 표준화(hidden-state는 미정규화라 필수): train 통계로 z-score
mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
Xtr = ((Xtr - mu) / sd).astype(np.float32)
Xva = ((Xva - mu) / sd).astype(np.float32)
print(f"feat={FEAT} | experts={len(experts)} | train {Xtr.shape} | val {Xva.shape}")

# --- MLP ---
D = Xtr.shape[1]
net = nn.Sequential(nn.Linear(D, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, len(experts)))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss()
Xt, St = torch.tensor(Xtr), torch.tensor(Str)
for ep in range(60):
    net.train()
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 256):
        b = perm[i:i + 256]
        opt.zero_grad(); loss = lossf(net(Xt[b]), St[b]); loss.backward(); opt.step()

# --- eval ---
net.eval()
with torch.no_grad():
    logit_va = net(torch.tensor(Xva)).numpy()
top1 = logit_va.argmax(1)
routed = np.array([Sva[i, top1[i]] for i in range(len(Sva))])
per_expert = Sva.mean(0)
best_single = per_expert.max()
oracle = (Sva.sum(1) > 0).mean()
random_route = per_expert.mean()

print(f"\n=== {sp.name.upper()} {sp.eval_split} {len(Sva)}, 라우터 feasibility (MLP) ===")
print(f"  routed top-1 (MLP)     : {100*routed.mean():.1f}%")
print(f"  best single expert     : {100*best_single:.1f}%  ({experts[per_expert.argmax()]})")
print(f"  random route           : {100*random_route:.1f}%")
print(f"  oracle (union)         : {100*oracle:.1f}%")
gain = 100 * (routed.mean() - best_single)
print(f"  → routed − best_single : {gain:+.1f}pp  "
      f"({'복잡성 회수 O' if gain > 1 else 'best-single 벽 (임베딩 한계 → hidden-state 필요)'})")
print(f"  → oracle까지 회수율     : {100*(routed.mean()-best_single)/(oracle-best_single+1e-9):.0f}% of headroom")
