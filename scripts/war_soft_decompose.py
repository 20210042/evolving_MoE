#!/usr/bin/env python3
"""엔빵(soft_linear) 점수가 '실력'인가 '희소성'인가 — 마진 보존 순열 검정.

soft_j = Σ_i w_i · y_ij  (w_i = (E−n_i)/(E−1), contested 문제만 w>0)
      = S_j · w̄_j        S_j = contested 문제 중 j가 푼 개수(=볼륨=실력)
                          w̄_j = 그 문제들의 평균 가중치(=희소성)

두 항 중 어디서 순위가 나오는지가 핵심이다. 볼륨에서만 나온다면 엔빵은 pass@1
재정렬일 뿐이고, w̄에서 나온다면 "남들이 못 푸는 걸 푼다"를 실제로 재는 것이다.

귀무: curveball(양쪽 마진 보존) — 문제별 n_i와 expert별 S_j를 그대로 두고 배정만 섞는다.
      w_i는 n_i의 함수라 그대로 보존되므로, 이 귀무 아래 S_j도 w 분포도 고정이고
      오직 "누가 어떤 희소도의 문제를 푸느냐"만 파괴된다 → w̄_j의 z가 곧 희소성 신호다.

Usage:
  python scripts/war_soft_decompose.py --binned results/acc/seed20211004/binning_train_full.binned.jsonl --n_perm 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def curveball(mat: np.ndarray, rng: np.random.Generator, n_swap: int) -> np.ndarray:
    """양쪽 마진을 보존하는 이진 행렬 랜덤화(curveball / trade algorithm)."""
    rows = [set(np.flatnonzero(r).tolist()) for r in mat]
    R = len(rows)
    for _ in range(n_swap):
        i, j = rng.integers(0, R, 2)
        if i == j:
            continue
        a, b = rows[i], rows[j]
        only_a = a - b
        only_b = b - a
        if not only_a or not only_b:
            continue
        pool = list(only_a | only_b)
        rng.shuffle(pool)
        k = len(only_a)
        new_a = (a & b) | set(pool[:k])
        new_b = (a & b) | set(pool[k:])
        rows[i], rows[j] = new_a, new_b
    out = np.zeros_like(mat)
    for i, s in enumerate(rows):
        if s:
            out[i, list(s)] = 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binned", default="results/acc/seed20211004/binning_train_full.binned.jsonl")
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--swap_mult", type=int, default=5, help="랜덤화 1회당 스왑 시도 = mult × 행 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/acc/seed20211004/war_soft_decompose.md")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / a.binned, encoding="utf-8") if l.strip()]
    experts = list(rows[0]["per_expert"].keys())
    E = len(experts)
    Y_all = np.array([[r["per_expert"].get(e, 0) for e in experts] for r in rows], dtype=np.int8)
    n_all = Y_all.sum(1)
    contested = (n_all > 0) & (n_all < E)
    Y = Y_all[contested]
    n = n_all[contested]
    w = (E - n) / (E - 1)
    Nc = Y.shape[0]

    S = Y.sum(0).astype(float)                    # 볼륨
    soft = w @ Y                                  # 엔빵 점수
    wbar = np.divide(soft, S, out=np.zeros_like(soft), where=S > 0)   # 희소성

    rng = np.random.default_rng(a.seed)
    null_wbar = np.zeros((a.n_perm, E))
    null_spread = np.zeros(a.n_perm)
    for t in range(a.n_perm):
        Yp = curveball(Y, rng, a.swap_mult * Nc)
        sp = w @ Yp
        wb = np.divide(sp, S, out=np.zeros_like(sp), where=S > 0)
        null_wbar[t] = wb
        null_spread[t] = sp.max() - sp.min()

    mu, sd = null_wbar.mean(0), null_wbar.std(0) + 1e-12
    z = (wbar - mu) / sd
    obs_spread = soft.max() - soft.min()
    z_spread = (obs_spread - null_spread.mean()) / (null_spread.std() + 1e-12)

    # pass@1(전체 기준) 및 순위 상관
    pass1 = Y_all.sum(0) / Y_all.shape[0]

    def rankcorr(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    order = np.argsort(-soft)
    L = [f"# 엔빵 점수의 분해와 마진 보존 순열 검정 — `{a.binned}`", "",
         f"- 전체 {Y_all.shape[0]:,}문제 중 contested(0<n<{E}) **{Nc:,}문제**만 점수에 기여(나머지는 w=0)",
         f"- 귀무: curveball {a.n_perm}회 (문제별 n_i·expert별 볼륨 S_j 동시 보존 → 희소성 배정만 파괴)", "",
         "| expert | pass@1 | 볼륨 S | 희소성 w̄ | soft=S·w̄ | w̄ 귀무평균 | **z(w̄)** |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for i in order:
        L.append(f"| `{experts[i]}` | {100*pass1[i]:.2f}% | {S[i]:.0f} | {wbar[i]:.4f} | {soft[i]:.1f} | "
                 f"{mu[i]:.4f} | **{z[i]:+.2f}** |")

    L += ["", f"- soft 점수 폭(max−min) 관측 **{obs_spread:.1f}** vs 귀무 "
          f"{null_spread.mean():.1f} ± {null_spread.std():.1f} → **z = {z_spread:+.2f}**",
          f"- 순위 상관: soft vs 볼륨 S = {rankcorr(soft, S):+.3f} · soft vs 희소성 w̄ = {rankcorr(soft, wbar):+.3f} · "
          f"희소성 w̄ vs pass@1 = {rankcorr(wbar, pass1):+.3f}",
          f"- |z(w̄)| > 2 인 expert: {int((np.abs(z) > 2).sum())}/{E}", "",
          "읽는 법: z(w̄)가 전부 0 근처면 엔빵은 볼륨(=실력) 재정렬일 뿐이고, 유의하게 갈리면",
          "그 expert는 남들이 못 푸는 영역을 실제로 갖고 있다는 뜻이다.", ""]

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
