#!/usr/bin/env python
"""max_n_solved 컷(τ) 스윕 — E=16에서 다시 잰다. 재생성 0.

규칙:
  · 1 ≤ n_solved ≤ τ  → 푼 사람(들)에게 개별 배정  (분화를 만드는 몫)
  · n_solved > τ       → **shared expert(짬통)**    (전원 성공 포함)
  · n_solved = 0       → 별도(센트로이드 배정, 이 스크립트 밖)

τ가 작을수록 분화는 커지고 데이터량은 줄어든다. acc의 τ=8은 **E=11** 기준이라 그대로 못 쓴다.
어느 τ를 고를지는 실험 설계 결정이므로 여기서는 **곡선만** 낸다.
"""
import argparse, json, itertools
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_common as rc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/sni/tau_sweep.md")
a = ap.parse_args()
sp = rc.spec("sni"); trb, _ = rc.labels(sp); ex = rc.experts(sp)
names = {p["id"]: p["name"] for p in json.load(open("results/sni/seed20212003/roster_final.json"))}
ids, X, S = rc.align(rc.feat_path(sp, "train", "hs_mean"),
                     rc.feat_path(sp, "train", "hs_ids"), trb, ex)
E = len(ex); solved = S > 0.5; ns = solved.sum(1); N = len(ids)
INVERT = {"c_18467", "c_19704", "c_61797"}
inv = [i for i, c in enumerate(ex) if c in INVERT]

L = [f"# max_n_solved 컷(τ) 스윕 — E={E}, train {N:,}문제", "",
     f"- n_solved 분포: 0명 {int((ns==0).sum()):,} · 전원 {int((ns==E).sum()):,}",
     "- 1 ≤ n_solved ≤ τ → 개별 배정 / n_solved > τ → shared expert(짬통) / 0 → 센트로이드 배정(별도)",
     "", "| τ | 개별 배정 문제 | 짬통 | expert당 평균 | 최소 | 최대 | 반전3인 평균 | 평균 쌍 Jaccard | 문제당 평균 배정 |",
     "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for tau in range(1, E):
    m = (ns >= 1) & (ns <= tau)
    sh = int(((ns > tau)).sum())
    sub = solved[m]
    cnt = sub.sum(0)
    J = []
    for i, j in itertools.combinations(range(E), 2):
        inter = float((sub[:, i] & sub[:, j]).sum()); uni = float((sub[:, i] | sub[:, j]).sum())
        if uni > 0:
            J.append(inter / uni)
    L.append(f"| {tau} | {int(m.sum()):,} | {sh:,} | {cnt.mean():,.0f} | {cnt.min():,} | "
             f"{cnt.max():,} | {cnt[inv].mean():,.0f} | {np.mean(J):.3f} | "
             f"{sub.sum()/max(1,m.sum()):.2f} |")
L += ["", "읽는 법", "",
      "- **평균 쌍 Jaccard** = 두 expert 학습셋이 얼마나 겹치나. 1이면 완전히 같은 데이터다.",
      "- **짬통**은 shared expert로 가는 몫(전원 성공 포함). 커질수록 개별 expert가 배울 게 줄어든다.",
      "- **반전3인 평균**이 너무 작으면 그 세 명은 학습이 안 된다(현재 test pass@1 26~43%).",
      "- 데이터량과 분화는 정면으로 상충한다. 어디서 자를지는 실험 설계 결정이다."]
open(a.out, "w").write("\n".join(L) + "\n")
print("\n".join(L))
