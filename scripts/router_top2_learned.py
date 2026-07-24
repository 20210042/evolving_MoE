#!/usr/bin/env python3
"""학습 top-2 라우터 최대화 (top-1 실험과 동일 조건, argmax→상위2).
같은 세팅: MLP · BCE multi-label · train binning으로 학습 · val 평가.
변수: feature(hs/ans/hs+ans/emb) × 용량(width/depth) × dropout/wd × epoch × seed앙상블.
지표: top-2 union = 라우터가 고른 상위2 expert 중 하나라도 풀면 성공.
기준선은 라벨에서 계산한다.

ans(답분포)는 MCQA 전용이라 오픈 QA에서는 자동으로 빠진다.
Usage: python scripts/router_top2_learned.py --dataset qasc
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
O = rc.FEAT_DIR
torch.manual_seed(42); np.random.seed(42)

_ap = argparse.ArgumentParser()
rc.add_dataset_arg(_ap)
sp = rc.spec(_ap.parse_args().dataset)

trb, vb = rc.labels(sp)
ex = rc.experts(sp)
alignfeat = rc.align

tk, Xt_hs, St = alignfeat(rc.feat_path(sp, sp.train_split, "hs_mean"),
                          rc.feat_path(sp, sp.train_split, "hs_ids"), trb, ex)
vk, Xv_hs, Sv = alignfeat(rc.feat_path(sp, sp.eval_split, "hs_mean"),
                          rc.feat_path(sp, sp.eval_split, "hs_ids"), vb, ex)
_, Xt_em, _ = alignfeat(*rc.emb_paths(sp, "train"), trb, ex, tk)
_, Xv_em, _ = alignfeat(*rc.emb_paths(sp, "eval"), vb, ex, vk)

N = len(vk); best, oracle = rc.baselines(Sv)
print(f"{sp.name.upper()} {sp.eval_split} {N} | best-single {best:.1f} | oracle {oracle:.1f}"
      f"  (top-2 union 최대화)\n")

FEATS = {"hs": (Xt_hs, Xv_hs), "emb": (Xt_em, Xv_em)}
if sp.answer_letters is not None:
    _, Xt_an, _ = alignfeat(rc.feat_path(sp, sp.train_split, "ansprob"),
                            rc.feat_path(sp, sp.train_split, "ansprob_ids"), trb, ex, tk)
    _, Xv_an, _ = alignfeat(rc.feat_path(sp, sp.eval_split, "ansprob"),
                            rc.feat_path(sp, sp.eval_split, "ansprob_ids"), vb, ex, vk)
    FEATS["ans"] = (Xt_an, Xv_an)
    FEATS["hs+ans"] = (np.concatenate([Xt_hs, Xt_an], 1), np.concatenate([Xv_hs, Xv_an], 1))


def mlp(d, hid, nl, drop):
    L = [nn.Linear(d, hid), nn.ReLU(), nn.Dropout(drop)]
    for _ in range(nl - 2):
        L += [nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(drop)]
    L += [nn.Linear(hid, len(ex))]
    return nn.Sequential(*L)


def top2(Xt, Xv, hid, nl, ep, drop, wd, seeds):
    mu, sd = Xt.mean(0, keepdims=True), Xt.std(0, keepdims=True) + 1e-6
    Xtn = ((Xt - mu) / sd).astype(np.float32); Xvn = ((Xv - mu) / sd).astype(np.float32)
    logs = []
    for s in seeds:
        torch.manual_seed(s)
        net = mlp(Xt.shape[1], hid, nl, drop)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd)
        lf = nn.BCEWithLogitsLoss(); Xtt, Stt = torch.tensor(Xtn), torch.tensor(St)
        for _ in range(ep):
            p = torch.randperm(len(Xtt))
            for i in range(0, len(Xtt), 256):
                b = p[i:i + 256]; opt.zero_grad(); lf(net(Xtt[b]), Stt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            logs.append(net(torch.tensor(Xvn)).numpy())
    lo = np.mean(logs, 0)
    t2 = lo.argsort(1)[:, -2:]
    return 100 * np.mean([Sv[i, t2[i]].max() for i in range(N)])


configs = [(512, 2, 100, 0.2, 1e-3), (1024, 2, 150, 0.3, 1e-3),
           (2048, 2, 150, 0.3, 1e-2), (1024, 3, 150, 0.3, 1e-2)]
bestv = (-1, None)
for fname, (Xt, Xv) in FEATS.items():
    for hid, nl, ep, drop, wd in configs:
        v1 = top2(Xt, Xv, hid, nl, ep, drop, wd, [42])
        v3 = top2(Xt, Xv, hid, nl, ep, drop, wd, [42, 1, 7])
        tag = f"{fname} hid{hid} L{nl} ep{ep} d{drop} wd{wd}"
        print(f"  [{tag:38}] 1seed {v1:.1f} | 3seed {v3:.1f}", flush=True)
        if v3 > bestv[0]:
            bestv = (v3, tag)
print(f"\n★ 학습 top-2 라우터 최고: {bestv[0]:.1f}%  ({bestv[1]})")
print(f"  (best-single {best:.1f} / oracle {oracle:.1f})")
print("done")
