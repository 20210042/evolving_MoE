#!/usr/bin/env python3
"""셀 고유 궁합 vs 디코딩 노이즈 — 반복표본(K>1) 이원분산분석.

K=1 라벨에서는 (문제,expert) 셀의 궁합과 이항 노이즈가 원리적으로 구별되지 않는다.
같은 셀을 K번 뽑은 데이터가 있으면 분리된다:
    MS_within  = 셀 내부 분산                → 순수 디코딩 노이즈
    MS_inter   = 셀 평균의 주효과 제거 잔차   → 궁합 + 노이즈/K
    F = MS_inter / MS_within,  궁합 분산 추정 = (MS_inter − MS_within)/K

입력은 router_self_consistency*.md의 표(expert · 문제 · greedy라벨 · k회 중 pass).
문제명이 40자로 잘려 있어 이름으로 못 붙이므로, expert 블록 안의 **행 순서**로 문제를
식별한다(모든 expert가 같은 문제 목록을 같은 순서로 돈다는 스크립트 구조에 의존) —
블록 길이와 위치별 이름 일치를 먼저 검증하고, 어긋나면 중단한다.

Usage:
  python scripts/interaction_replicated_anova.py --md results/acc/router_self_consistency_full.md --k 5
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*(\d+)/(\d+)\s*\|")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="results/acc/router_self_consistency_full.md")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results/acc/interaction_replicated_anova.md")
    a = ap.parse_args()

    blocks: dict[str, list[tuple[str, int]]] = {}
    order: list[str] = []
    for line in open(ROOT / a.md, encoding="utf-8"):
        m = ROW.match(line)
        if not m:
            continue
        e, pname, _greedy, kp, kt = m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5))
        if kt != a.k:
            continue
        if e not in blocks:
            blocks[e] = []
            order.append(e)
        blocks[e].append((pname, kp))

    experts = order
    sizes = {e: len(v) for e, v in blocks.items()}
    J = len(experts)
    I = sizes[experts[0]]
    assert all(v == I for v in sizes.values()), f"expert별 행 수 불일치: {sizes}"
    for i in range(I):
        names = {blocks[e][i][0] for e in experts}
        assert len(names) == 1, f"{i}번째 행의 문제명이 expert마다 다르다: {names}"

    K = a.k
    p = [[blocks[e][i][1] / K for e in experts] for i in range(I)]   # 셀 성공률
    grand = sum(sum(r) for r in p) / (I * J)
    row = [sum(r) / J for r in p]
    col = [sum(p[i][j] for i in range(I)) / I for j in range(J)]

    # 이원분산분석(반복 K) — 이진 결과의 셀 내부 분산은 K/(K-1)·p(1-p)
    ss_row = K * J * sum((r - grand) ** 2 for r in row)
    ss_col = K * I * sum((c - grand) ** 2 for c in col)
    ss_int = K * sum((p[i][j] - row[i] - col[j] + grand) ** 2 for i in range(I) for j in range(J))
    ss_within = sum(K * p[i][j] * (1 - p[i][j]) for i in range(I) for j in range(J)) * K / (K - 1)
    # (셀당 표본분산 p(1-p)·K/(K-1) 를 K개 관측에 대해 합산)

    df_row, df_col = I - 1, J - 1
    df_int = df_row * df_col
    df_within = I * J * (K - 1)
    ms_int, ms_within = ss_int / df_int, ss_within / df_within
    F = ms_int / ms_within if ms_within else float("nan")
    var_inter = (ms_int - ms_within) / K          # 궁합 분산 성분 추정(음수면 0)
    ss_tot = ss_row + ss_col + ss_int + ss_within

    L = [f"# 셀 고유 궁합 vs 디코딩 노이즈 — 반복표본 이원분산분석 (`{a.md}`)", "",
         f"- 문제 {I} × expert {J} × K={K} = {I*J*K:,} 생성 · 셀 {I*J}", "",
         "| 성분 | SS | df | MS | 전체 대비 |", "|---|---:|---:|---:|---:|",
         f"| 문제(난이도) | {ss_row:.1f} | {df_row} | {ss_row/df_row:.4f} | {100*ss_row/ss_tot:.1f}% |",
         f"| expert(실력) | {ss_col:.1f} | {df_col} | {ss_col/df_col:.4f} | {100*ss_col/ss_tot:.1f}% |",
         f"| **궁합(문제×expert)** | {ss_int:.1f} | {df_int} | {ms_int:.4f} | {100*ss_int/ss_tot:.1f}% |",
         f"| 셀 내부(디코딩 노이즈) | {ss_within:.1f} | {df_within} | {ms_within:.4f} | {100*ss_within/ss_tot:.1f}% |",
         "",
         f"**F(궁합 / 노이즈) = {F:.3f}** (df {df_int}, {df_within})",
         f"· 궁합 분산 성분 추정 = {max(var_inter, 0.0):.5f}"
         f"{' (원값 ' + f'{var_inter:+.5f}' + ', 음수 → 0으로 절단)' if var_inter < 0 else ''}", "",
         ("> F ≈ 1: 셀 평균의 흔들림이 반복 표본의 흔들림과 같은 크기 — 셀 고유 궁합이 없다."
          if F < 1.2 else
          "> F > 1: 반복으로 설명되지 않는 셀 고유 성분이 있다 — 궁합이 실재한다."), "",
         "주의: 궁합 항은 반복이 있어야만 노이즈와 분리된다. K=1 라벨(binning)에서는",
         "이 표의 마지막 두 줄이 한 덩어리로 묶여 '잔차'로 보이며, 그걸 상호작용으로 읽으면 과대추정이다.", ""]

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
