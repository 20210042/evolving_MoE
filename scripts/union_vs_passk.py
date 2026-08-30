#!/usr/bin/env python3
"""persona 다양성 = 시드 다양성인가 — 로스터 union vs LUCA pass@k.

두 조건은 로스터 파일 하나만 다르다(동일 test 751, 동일 T=0.7 config, 동일 파이프라인):
  luca12  : 같은 프롬프트 12복제 → 드로우 간 차이만 남는다(시드 다양성)
  roster12: 진화 페르소나 12명(persona 다양성)

union@k 곡선과, 문제별 짝지은 McNemar(전체 union 기준)를 낸다.

Usage:
  python scripts/union_vs_passk.py
"""
from __future__ import annotations

import argparse
import json
import random
from math import erfc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path: str):
    rows = [json.loads(l) for l in open(ROOT / path, encoding="utf-8") if l.strip()]
    experts = list(rows[0]["per_expert"].keys())
    Y = {r["id"]: [int(r["per_expert"].get(e, 0)) for e in experts] for r in rows}
    return Y, experts


def union_at_k(Y: dict, E: int, k: int, n_draw: int, rng: random.Random) -> float:
    ids = list(Y)
    tot = 0.0
    for _ in range(n_draw):
        idx = rng.sample(range(E), k)
        tot += sum(1 for i in ids if any(Y[i][j] for j in idx))
    return 100.0 * tot / (n_draw * len(ids))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--luca", default="results/acc/seed20211004/binning_test_luca12_sampled.binned.jsonl")
    ap.add_argument("--roster", default="results/acc/seed20211004/binning_test_roster12_sampled.binned.jsonl")
    ap.add_argument("--n_draw", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/acc/seed20211004/union_vs_passk.md")
    a = ap.parse_args()

    YL, EL = load(a.luca)
    YR, ER = load(a.roster)
    ids = sorted(set(YL) & set(YR))
    E = min(len(EL), len(ER))
    rng = random.Random(a.seed)

    rows = []
    for k in range(1, E + 1):
        rows.append((k,
                     union_at_k(YL, E, k, a.n_draw if k not in (1, E) else 1 if k == E else a.n_draw, rng),
                     union_at_k(YR, E, k, a.n_draw if k not in (1, E) else 1 if k == E else a.n_draw, rng)))

    # 전체 union(k=E)에서 문제별 짝지은 비교
    ul = {i: int(any(YL[i])) for i in ids}
    ur = {i: int(any(YR[i])) for i in ids}
    b = sum(1 for i in ids if ul[i] and not ur[i])   # LUCA만 푼 문제
    c = sum(1 for i in ids if ur[i] and not ul[i])   # 로스터만 푼 문제
    if b + c == 0:
        p = 1.0
    else:
        z = (abs(b - c) - 1) / ((b + c) ** 0.5)
        p = float(erfc(abs(z) / (2 ** 0.5)))

    mean_l = 100.0 * sum(sum(YL[i]) for i in ids) / (len(ids) * E)
    mean_r = 100.0 * sum(sum(YR[i]) for i in ids) / (len(ids) * E)

    L = ["# persona 다양성 = 시드 다양성인가 — union vs pass@k", "",
         f"- test {len(ids)}문제 · 12 드로우 · 동일 config(T=0.7/top_p 0.8/top_k 20/rep 1.05) · "
         "차이는 로스터 파일 하나뿐", "",
         "| k | LUCA×12 union@k (시드) | 로스터 union@k (persona) | 차이 |",
         "|---:|---:|---:|---:|"]
    for k, ul_k, ur_k in rows:
        L.append(f"| {k} | {ul_k:.2f}% | {ur_k:.2f}% | {ur_k-ul_k:+.2f}pp |")
    L += ["", f"- 개별 평균 pass@1: LUCA {mean_l:.2f}% · 로스터 {mean_r:.2f}%",
          f"- 전체 union(k=12) 짝지은 비교: LUCA만 푼 문제 **{b}** / 로스터만 푼 문제 **{c}** → "
          f"McNemar p = **{p:.4f}**", "",
          ("> 두 곡선이 겹치면 union 헤드룸은 **시드 다양성과 동치**다 — persona 텍스트가 "
           "커버리지를 만든 것이 아니다."), ""]

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
