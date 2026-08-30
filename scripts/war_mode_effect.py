#!/usr/bin/env python3
"""엔빵 가중치(soft_linear)가 hard WAR 대비 무엇을 바꾸나 — 기존 0/1 라벨만 사용.

hard : war_j = |나만 푼 문제|                        (n_i == 1 인 문제만 1점)
soft : war_j = Σ_{i∈solved(j)} (E − n_i)/(E − 1)     (여러 명이 풀면 나눠 갖기)

Usage:
  python scripts/war_mode_effect.py --binned results/acc/seed20211004/binning_train_full.binned.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def spearman(x: list[float], y: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
    den = (sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry)) ** 0.5
    return num / den if den else float("nan")


def war(rows: list[dict], experts: list[str]) -> tuple[dict, dict]:
    E = len(experts)
    hard = {e: 0.0 for e in experts}
    soft = {e: 0.0 for e in experts}
    for r in rows:
        pe = r["per_expert"]
        solvers = [e for e in experts if pe.get(e, 0)]
        n = len(solvers)
        if n == 0 or n == E:
            continue
        w = (E - n) / (E - 1)
        for e in solvers:
            soft[e] += w
            if n == 1:
                hard[e] += 1.0
    return hard, soft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binned", default="results/acc/seed20211004/binning_train_full.binned.jsonl")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--out", default="results/acc/seed20211004/war_mode_effect_train.md")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(ROOT / a.binned, encoding="utf-8") if l.strip()]
    experts = list(rows[0]["per_expert"].keys())
    E, N = len(experts), len(rows)
    passk = {e: sum(r["per_expert"].get(e, 0) for r in rows) for e in experts}
    hard, soft = war(rows, experts)

    batches = [rows[i:i + a.batch] for i in range(0, N, a.batch)]
    h_zero = h_uniq = s_uniq = 0
    for b in batches:
        h, s = war(b, experts)
        if all(v == 0 for v in h.values()):
            h_zero += 1
        if sum(1 for e in experts if h[e] == min(h.values())) == 1:
            h_uniq += 1
        if sum(1 for e in experts if s[e] == min(s.values())) == 1:
            s_uniq += 1

    oh = sorted(experts, key=lambda e: -hard[e])
    os_ = sorted(experts, key=lambda e: -soft[e])
    op = sorted(experts, key=lambda e: -passk[e])

    L = [f"# 엔빵(soft) vs hard WAR — `{a.binned}` (문제 {N:,} × expert {E}, 0/1 라벨)", "",
         "| expert | pass@1 | hard | soft | hard순위 | soft순위 | pass@1순위 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for e in os_:
        L.append(f"| `{e}` | {100*passk[e]/N:.2f}% | {hard[e]:.0f} | {soft[e]:.1f} | "
                 f"{oh.index(e)+1} | {os_.index(e)+1} | {op.index(e)+1} |")
    nb = len(batches)
    L += ["", f"배치 {a.batch}문제 × {nb}개 기준 — 최하위가 유일하게 정해지는 배치: "
          f"hard {h_uniq}/{nb} ({100*h_uniq/nb:.0f}%) · soft {s_uniq}/{nb} ({100*s_uniq/nb:.0f}%). "
          f"hard가 전원 0점인 배치 {h_zero}/{nb} ({100*h_zero/nb:.0f}%).", "",
          f"순위 상관(스피어만): soft vs pass@1 = **{spearman([soft[e] for e in experts], [passk[e] for e in experts]):+.3f}** · "
          f"hard vs pass@1 = {spearman([hard[e] for e in experts], [passk[e] for e in experts]):+.3f} · "
          f"hard vs soft = {spearman([hard[e] for e in experts], [soft[e] for e in experts]):+.3f}", ""]

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
