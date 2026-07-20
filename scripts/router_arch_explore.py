#!/usr/bin/env python3
"""목표: 아키텍처 혁신만으로 top-1 라우팅을 oracle에 근접시킬 수 있나.

(0) 과적합 진단: 입력(hs_mean)이 solvability를 '외울' 수 있나? TRAIN vs VAL routing acc.
    - TRAIN도 best-single 근처면 → 입력에 신호 없음(정보론적 한계) = 새 정보 필요.
    - TRAIN은 oracle 근처인데 VAL만 낮으면 → 일반화 문제 = 구조/규제로 개선 여지.
(1) flat classifier 용량 스윕 (top-1).
(2) two-tower 라우팅 (problem tower · expert embedding, 내적 점수) — 구조 공유로 일반화 시도.

feature = hs_mean (best). 지표 = routed top-1 acc. 기준선은 라벨에서 계산한다.

Usage: python scripts/router_arch_explore.py --dataset qasc
"""
import argparse
import json
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

R = rc.REPO
torch.manual_seed(42); np.random.seed(42)

_ap = argparse.ArgumentParser()
rc.add_dataset_arg(_ap)
sp = rc.spec(_ap.parse_args().dataset)


trb, vb = rc.labels(sp)
ex = rc.experts(sp)
_, Xtr, Str = rc.align(rc.feat_path(sp, sp.train_split, "hs_mean"),
                       rc.feat_path(sp, sp.train_split, "hs_ids"), trb, ex)
_, Xva, Sva = rc.align(rc.feat_path(sp, sp.eval_split, "hs_mean"),
                       rc.feat_path(sp, sp.eval_split, "hs_ids"), vb, ex)
mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
Xtr = ((Xtr - mu) / sd).astype(np.float32)
Xva = ((Xva - mu) / sd).astype(np.float32)
E = len(ex)
Xt, St, Xv = torch.tensor(Xtr), torch.tensor(Str), torch.tensor(Xva)
tr_oracle = (Str.sum(1) > 0).mean() * 100
va_best, va_oracle = rc.baselines(Sva)
print(f"{sp.name.upper()} | TRAIN oracle {tr_oracle:.1f} | VAL oracle {va_oracle:.1f} | "
      f"VAL best-single {va_best:.1f}\n")


def racc(logit, S):
    t1 = logit.argmax(1)
    return 100 * np.mean([S[i, t1[i]] for i in range(len(S))])


def run_flat(hid, nl, ep, drop, wd, tag):
    torch.manual_seed(42)
    L = [nn.Linear(Xt.shape[1], hid), nn.ReLU(), nn.Dropout(drop)]
    for _ in range(nl - 2):
        L += [nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(drop)]
    L += [nn.Linear(hid, E)]
    net = nn.Sequential(*L)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd)
    lf = nn.BCEWithLogitsLoss()
    for _ in range(ep):
        p = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = p[i:i + 256]
            opt.zero_grad(); lf(net(Xt[b]), St[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        ltr, lva = net(Xt).numpy(), net(Xv).numpy()
    print(f"[flat {tag}] TRAIN {racc(ltr,Str):.1f} | VAL {racc(lva,Sva):.1f}", flush=True)


class TwoTower(nn.Module):
    def __init__(self, d, dim, E):
        super().__init__()
        self.p = nn.Sequential(nn.Linear(d, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, dim))
        self.e = nn.Embedding(E, dim)

    def forward(self, x):
        return self.p(x) @ self.e.weight.T


def run_tower(dim, ep, wd, tag):
    torch.manual_seed(42)
    net = TwoTower(Xt.shape[1], dim, E)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd)
    lf = nn.BCEWithLogitsLoss()
    for _ in range(ep):
        p = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = p[i:i + 256]
            opt.zero_grad(); lf(net(Xt[b]), St[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        ltr, lva = net(Xt).numpy(), net(Xv).numpy()
    print(f"[tower {tag}] TRAIN {racc(ltr,Str):.1f} | VAL {racc(lva,Sva):.1f}", flush=True)


print("=== (0) 과적합 진단: 입력이 solvability를 외울 수 있나 (규제 없이) ===")
run_flat(2048, 3, 300, 0.0, 0.0, "overfit hid2048 L3 ep300")
print("\n=== (1) flat classifier 용량/규제 스윕 (top-1) ===")
for hid, nl, ep, drop, wd in [(512, 2, 80, 0.2, 1e-3), (1024, 2, 120, 0.3, 1e-3),
                              (2048, 3, 150, 0.3, 1e-2)]:
    run_flat(hid, nl, ep, drop, wd, f"hid{hid} L{nl} ep{ep} d{drop} wd{wd}")
print("\n=== (2) two-tower 라우팅 ===")
for dim, ep, wd in [(128, 120, 1e-3), (256, 150, 1e-3), (256, 200, 1e-2)]:
    run_tower(dim, ep, wd, f"dim{dim} ep{ep} wd{wd}")
print("\ndone")
