#!/usr/bin/env python3
"""라우터 개선 스윕: hs_mean residual(anchor = best-single + routed complement) 기준
용량(width/depth)·epoch·시드앙상블을 바꿔가며 top-2 커버리지 최대화.

Usage: python scripts/router_sweep.py --dataset qasc
"""
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

R = rc.REPO

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
# anchor는 train solve율 최고 expert로 계산한다(예전엔 QASC 코드네임이 박혀 있었음).
a = rc.anchor_expert(Str, ex)
others = [i for i in range(len(ex)) if i != a]
fail = Str[:, a] == 0


def mlp(d, hid, nl, nout, drop):
    L = [nn.Linear(d, hid), nn.ReLU(), nn.Dropout(drop)]
    for _ in range(nl - 2):
        L += [nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(drop)]
    L += [nn.Linear(hid, nout)]
    return nn.Sequential(*L)


def train(X, Y, hid, nl, ep, seed, drop=0.2, wd=1e-3):
    torch.manual_seed(seed)
    net = mlp(X.shape[1], hid, nl, Y.shape[1], drop)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd)
    lf = nn.BCEWithLogitsLoss()
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _ in range(ep):
        p = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 256):
            b = p[i:i + 256]
            opt.zero_grad()
            lf(net(Xt[b]), Yt[b]).backward()
            opt.step()
    return net


def resid(hid, nl, ep, seeds, drop=0.2, wd=1e-3):
    los = [train(Xtr[fail], Str[fail][:, others], hid, nl, ep, s, drop, wd)(
        torch.tensor(Xva)).detach().numpy() for s in seeds]
    lo = np.mean(los, 0)
    comp = np.array(others)[lo.argmax(1)]
    return np.mean([max(Sva[i, a], Sva[i, comp[i]]) for i in range(len(Sva))])


_best, _oracle = rc.baselines(Sva)
print(f"=== {sp.name.upper()} hs_mean residual | anchor={ex[a]} | "
      f"best-single {_best:.1f}, oracle {_oracle:.1f} ===")
for hid, nl, ep in [(512, 2, 60), (1024, 2, 120), (2048, 2, 120), (1024, 3, 120), (2048, 3, 150)]:
    r1 = resid(hid, nl, ep, [42])
    r3 = resid(hid, nl, ep, [42, 1, 7])
    print(f"  hid{hid} L{nl} ep{ep}: 1seed {100*r1:.1f}% | 3seed-ens {100*r3:.1f}%", flush=True)
# 정규화 강화 + drop 조합 몇 개
for drop, wd in [(0.4, 1e-3), (0.3, 1e-2)]:
    r = resid(1024, 2, 150, [42, 1, 7], drop, wd)
    print(f"  hid1024 L2 ep150 drop{drop} wd{wd}: 3seed-ens {100*r:.1f}%", flush=True)
print("done")
