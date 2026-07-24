#!/usr/bin/env python3
"""top-2 최대화 스윕. 기준선은 라벨에서 계산한다.
가진 신호 전부(hs_mean, base ansprob, expert confidence)를 조합해 top-2 union을 끌어올린다.

top-2 union = 라우터가 고른 2 expert 중 하나라도 풀면 성공.
방식:
  (A) 전역 최적 고정집합 — ⚠️ 라우팅 없는 상한 참고용이며 프로젝트 방침상
      배포 방식으로 쓰지 않는다. 비교 기준으로만 읽을 것.
  (B) 학습 라우터 top-2: 특징별(hs/ans/hs+ans) MLP → 상위2
  (C) anchor+residual: best-single 고정 + 잔차 라우팅으로 2번째
  (D) confidence 기반 top-2 (정규화)

ans/confidence는 MCQA 전용이라 오픈 QA에서는 해당 구간이 자동으로 빠진다.
Usage: python scripts/router_top2_push.py --dataset qasc
"""
import argparse
import sys
from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

R = rc.REPO
torch.manual_seed(42); np.random.seed(42)
O = rc.FEAT_DIR

_ap = argparse.ArgumentParser()
rc.add_dataset_arg(_ap)
sp = rc.spec(_ap.parse_args().dataset)
MCQA = sp.answer_letters is not None


alignfeat = rc.align
trb, vb = rc.labels(sp)
ex = rc.experts(sp)

# 기준 id = hs (train/eval)
tk, Xt_hs, St = alignfeat(rc.feat_path(sp, sp.train_split, "hs_mean"),
                          rc.feat_path(sp, sp.train_split, "hs_ids"), trb, ex)
vk, Xv_hs, Sv = alignfeat(rc.feat_path(sp, sp.eval_split, "hs_mean"),
                          rc.feat_path(sp, sp.eval_split, "hs_ids"), vb, ex)
Xt_an = Xv_an = Cv = None
if MCQA:
    _, Xt_an, _ = alignfeat(rc.feat_path(sp, sp.train_split, "ansprob"),
                            rc.feat_path(sp, sp.train_split, "ansprob_ids"), trb, ex, tk)
    _, Xv_an, _ = alignfeat(rc.feat_path(sp, sp.eval_split, "ansprob"),
                            rc.feat_path(sp, sp.eval_split, "ansprob_ids"), vb, ex, vk)
    # expert confidence(eval만 있음) → 라우터 입력엔 못 쓰지만 top-2 confidence 방식엔 사용
    Cv = rc.align_conf(sp, sp.eval_split, ex, vk)

N = len(vk); best, oracle = rc.baselines(Sv)
print(f"{sp.name.upper()} {sp.eval_split} {N} | best-single {best:.1f} | oracle {oracle:.1f}\n")


def union2(pick2):
    return 100 * np.mean([Sv[i, pick2[i]].max() for i in range(N)])


# (A) 전역 최적 고정집합 — 라우팅 없는 상한. 배포 방식으로 쓰지 않는다(참고선 전용).
print("=== (A) 전역최적 고정집합 (라우팅 없음, 상한 참고선) ===")
for k in (2, 3, 4):
    bestset, bp = None, -1
    for c in combinations(range(len(ex)), k):
        p = (Sv[:, c].sum(1) > 0).mean()
        if p > bp:
            bp, bestset = p, c
    print(f"  best fixed {k}-set: {100*bp:.1f}%  ({'+'.join(ex[i] for i in bestset)})")


def mlp(d, hid=1024, drop=0.3):
    return nn.Sequential(nn.Linear(d, hid), nn.ReLU(), nn.Dropout(drop), nn.Linear(hid, len(ex)))


def train_router(Xt, Xv, seeds=(42, 1, 7)):
    mu, sd = Xt.mean(0, keepdims=True), Xt.std(0, keepdims=True) + 1e-6
    Xtn = ((Xt - mu) / sd).astype(np.float32); Xvn = ((Xv - mu) / sd).astype(np.float32)
    logs = []
    for s in seeds:
        torch.manual_seed(s)
        net = mlp(Xt.shape[1]); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
        lf = nn.BCEWithLogitsLoss(); Xtt, Stt = torch.tensor(Xtn), torch.tensor(St)
        for _ in range(150):
            p = torch.randperm(len(Xtt))
            for i in range(0, len(Xtt), 256):
                b = p[i:i + 256]; opt.zero_grad(); lf(net(Xtt[b]), Stt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            logs.append(net(torch.tensor(Xvn)).numpy())
    return np.mean(logs, 0)


print("\n=== (B) 학습 라우터 top-2 union (3seed 앙상블) ===")
feats = {"hs": (Xt_hs, Xv_hs)}
if MCQA:
    feats["ans"] = (Xt_an, Xv_an)
    feats["hs+ans"] = (np.concatenate([Xt_hs, Xt_an], 1), np.concatenate([Xv_hs, Xv_an], 1))
logits = {}
for name, (Xt, Xv) in feats.items():
    lo = train_router(Xt, Xv); logits[name] = lo
    print(f"  [{name:7}] top-2 union {union2(lo.argsort(1)[:, -2:]):.1f}%")

print("\n=== (C) anchor(best-single) + 잔차 라우팅 2번째 ===")
a = rc.anchor_expert(St, ex); others = [i for i in range(len(ex)) if i != a]
print(f"  anchor = {ex[a]}")
fail = St[:, a] == 0
for name, (Xt, Xv) in feats.items():
    lo = train_router(Xt[fail], Xv)  # anchor 실패셋으로만 학습(잔차 집중)
    comp = np.array(others)[lo[:, others].argmax(1)]
    pick = np.stack([np.full(N, a), comp], 1)
    print(f"  [{name:7}] anchor+residual union {union2(pick):.1f}%")

if MCQA:
    print("\n=== (D) confidence 기반 top-2 (eval 정규화) ===")
    z = (Cv - Cv.mean(0, keepdims=True)) / (Cv.std(0, keepdims=True) + 1e-6)
    prior = Sv.mean(0, keepdims=True)
    for tag, sc in [("z-norm", z), ("z+prior", z + 3 * prior), ("raw", Cv)]:
        print(f"  [{tag:8}] top-2 union {union2(sc.argsort(1)[:, -2:]):.1f}%")
    # (E) hs 라우터 1등 + confidence 2등 (하이브리드)
    comp = np.argsort(z, 1)[:, -1]
    pick = np.stack([logits["hs"].argmax(1), comp], 1)
    print(f"\n=== (E) hs라우터 top1 + confidence top1 하이브리드: union {union2(pick):.1f}% ===")
else:
    print("\n(D)(E) confidence 구간 생략 — MCQA 전용")
print("done")
