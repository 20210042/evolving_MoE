#!/usr/bin/env python3
"""학습 라우터 top-1 최대화 — 세 입력만: hidden-state / encoder(emb) / confidence.
각 입력 × MLP(width/depth/dropout/wd/epoch) × seed앙상블 스윕. 최고 top-1 routing acc.
기준선은 라벨에서 계산한다.

confidence는 MCQA 전용이라 오픈 QA에서는 자동으로 목록에서 빠진다.
Usage: python scripts/router_top1_sweep.py --dataset qasc
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
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEV}", flush=True)

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
# encoder(emb): train은 ids.json, eval은 원본 jsonl 순서로 생성돼 있다
_, Xt_em, _ = alignfeat(*rc.emb_paths(sp, "train"), trb, ex, tk)
_, Xv_em, _ = alignfeat(*rc.emb_paths(sp, "eval"), vb, ex, vk)

FEATS = {"hidden-state": (Xt_hs, Xv_hs), "encoder(emb)": (Xt_em, Xv_em)}

# confidence는 MCQA 전용이고 train/val 둘 다 있어야 입력으로 쓸 수 있다.
if sp.answer_letters is not None:
    ct, cv = (rc.feat_path(sp, s, "conf") for s in (sp.train_split, sp.eval_split))
    if ct.exists() and cv.exists():
        Ct = rc.align_conf(sp, sp.train_split, ex, tk)
        Cv = rc.align_conf(sp, sp.eval_split, ex, vk)
        FEATS["confidence"] = (Ct, Cv)
        FEATS["hs+conf"] = (np.concatenate([Xt_hs, Ct], 1), np.concatenate([Xv_hs, Cv], 1))

N = len(vk); best, oracle = rc.baselines(Sv)
print(f"{sp.name.upper()} {sp.eval_split} {N} | best-single {best:.1f} | oracle {oracle:.1f} | "
      f"inputs: {list(FEATS)}\n")


def racc(logit, S):
    t1 = logit.argmax(1)
    return 100 * np.mean([S[i, t1[i]] for i in range(len(S))])


def racc2(logit, S):
    t2 = logit.argsort(1)[:, -2:]
    return 100 * np.mean([S[i, t2[i]].max() for i in range(len(S))])


def run(Xt, Xv, hid, nl, ep, drop, wd, seeds):
    mu, sd = Xt.mean(0, keepdims=True), Xt.std(0, keepdims=True) + 1e-6
    Xtn = ((Xt - mu) / sd).astype(np.float32); Xvn = ((Xv - mu) / sd).astype(np.float32)
    logs = []
    Xtt, Stt = torch.tensor(Xtn).to(DEV), torch.tensor(St).to(DEV)
    Xvv = torch.tensor(Xvn).to(DEV)
    for s in seeds:
        torch.manual_seed(s)
        L = [nn.Linear(Xt.shape[1], hid), nn.ReLU(), nn.Dropout(drop)]
        for _ in range(nl - 2):
            L += [nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(drop)]
        L += [nn.Linear(hid, len(ex))]
        net = nn.Sequential(*L).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd)
        lf = nn.BCEWithLogitsLoss()
        for _ in range(ep):
            p = torch.randperm(len(Xtt), device=DEV)
            for i in range(0, len(Xtt), 256):
                b = p[i:i + 256]; opt.zero_grad(); lf(net(Xtt[b]), Stt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            logs.append(net(Xvv).cpu().numpy())
    lo = np.mean(logs, 0)
    return racc(lo, Sv), racc2(lo, Sv)


configs = [(256, 2, 100, 0.2, 1e-3), (512, 2, 120, 0.3, 1e-3), (1024, 2, 150, 0.3, 1e-3),
           (2048, 2, 150, 0.3, 1e-2), (1024, 3, 150, 0.3, 1e-2), (2048, 3, 200, 0.4, 1e-2)]
b1 = (-1, None); b2 = (-1, None)
for fname, (Xt, Xv) in FEATS.items():
    if Xt is None or Xv is None:
        continue
    print(f"--- {fname} (dim {Xt.shape[1]}) ---")
    for hid, nl, ep, drop, wd in configs:
        v1, v2 = run(Xt, Xv, hid, nl, ep, drop, wd, [42, 1, 7])
        tag = f"{fname} hid{hid} L{nl} ep{ep} d{drop} wd{wd}"
        print(f"  [{tag:42}] top-1 {v1:.1f} | top-2 {v2:.1f}", flush=True)
        if v1 > b1[0]:
            b1 = (v1, tag)
        if v2 > b2[0]:
            b2 = (v2, tag)
print(f"\n★ top-1 최고: {b1[0]:.1f}%  ({b1[1]})")
print(f"★ top-2 최고: {b2[0]:.1f}%  ({b2[1]})")
print(f"  (best-single {best:.1f} / oracle {oracle:.1f})")
print("done")
