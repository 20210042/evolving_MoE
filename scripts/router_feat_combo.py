#!/usr/bin/env python3
"""라우터 특징 조합 비교 (top-1): hs_mean / ansprob(base 답분포) / concat.
base 답분포(난이도 신호)가 hs만으론 못 넘던 best-single 벽을 뚫나?
지표 = routed top-1 acc. 기준선은 라벨에서 계산한다.

⚠️ ansprob이 MCQA 전용이라 오픈 QA 데이터셋에서는 실행되지 않는다.
Usage: python scripts/router_feat_combo.py --dataset qasc
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
torch.manual_seed(42); np.random.seed(42)

_ap = argparse.ArgumentParser()
rc.add_dataset_arg(_ap)
sp = rc.spec(_ap.parse_args().dataset)

trb, vb = rc.labels(sp)
ex = rc.experts(sp)

FE = {"hs": ("hs_mean", "hs_ids"), "ans": ("ansprob", "ansprob_ids")}


def load_feat(name):
    fk, ik = FE[name]
    tk, Xt, St = rc.align(rc.feat_path(sp, sp.train_split, fk),
                          rc.feat_path(sp, sp.train_split, ik), trb, ex)
    vk, Xv, Sv = rc.align(rc.feat_path(sp, sp.eval_split, fk),
                          rc.feat_path(sp, sp.eval_split, ik), vb, ex)
    return tk, Xt, St, vk, Xv, Sv


# 공통 id 정렬(두 특징 concat 위해 hs 기준으로 통일)
tk, Xt_hs, St, vk, Xv_hs, Sv = load_feat("hs")
tk_a, Xt_an, _, vk_a, Xv_an, _ = load_feat("ans")
ta = {p: i for i, p in enumerate(tk_a)}
va = {p: i for i, p in enumerate(vk_a)}
Xt_an = np.array([Xt_an[ta[p]] for p in tk], np.float32)
Xv_an = np.array([Xv_an[va[p]] for p in vk], np.float32)

feats = {"hs": (Xt_hs, Xv_hs), "ans": (Xt_an, Xv_an),
         "hs+ans": (np.concatenate([Xt_hs, Xt_an], 1), np.concatenate([Xv_hs, Xv_an], 1))}
va_best, va_oracle = rc.baselines(Sv)
print(f"{sp.name.upper()} | best-single {va_best:.1f} | oracle {va_oracle:.1f}\n")


def racc(logit, S):
    t1 = logit.argmax(1)
    return 100 * np.mean([S[i, t1[i]] for i in range(len(S))])


def train_eval(Xt, Xv, hid=1024, ep=150, drop=0.3):
    mu, sd = Xt.mean(0, keepdims=True), Xt.std(0, keepdims=True) + 1e-6
    Xt = ((Xt - mu) / sd).astype(np.float32); Xv = ((Xv - mu) / sd).astype(np.float32)
    torch.manual_seed(42)
    net = nn.Sequential(nn.Linear(Xt.shape[1], hid), nn.ReLU(), nn.Dropout(drop),
                        nn.Linear(hid, len(ex)))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    lf = nn.BCEWithLogitsLoss()
    Xtt, Stt = torch.tensor(Xt), torch.tensor(St)
    for _ in range(ep):
        p = torch.randperm(len(Xtt))
        for i in range(0, len(Xtt), 256):
            b = p[i:i + 256]
            opt.zero_grad(); lf(net(Xtt[b]), Stt[b]).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return racc(net(torch.tensor(Xt)).numpy(), St), racc(net(torch.tensor(Xv)).numpy(), Sv)


for name, (Xt, Xv) in feats.items():
    tr, va_ = train_eval(Xt, Xv)
    print(f"[{name:7}] TRAIN {tr:.1f} | VAL {va_:.1f}", flush=True)
print("done")
