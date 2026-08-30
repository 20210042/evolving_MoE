#!/usr/bin/env python
"""페르소나들이 같은 문제에서 **몇 번째 토큰부터 갈라지나** — 재생성 0.

토큰단위 MoE로 옮겼을 때 라우터가 개입할 자리가 있는지를 미리 본다.
  · 첫 토큰부터 갈린다 → 축이 형식·정책 층에 있다(토큰 라우팅이 다룰 수 있는 것)
  · 늦게 갈리거나 거의 같다 → 분화가 얕아 옮길 게 적다

**대조군이 핵심**: 같은 페르소나를 K=3으로 두 번 뽑은 쌍(=디코딩 노이즈)과 비교한다.
페르소나 간 갈라짐이 페르소나 내 갈라짐과 같으면, 토큰 수준에서 페르소나는 구분되지 않는다.

토크나이저는 생성에 쓴 것과 같은 모델(gemma-4-26B-A4B-it)을 쓴다.
"""
import argparse
import json
import random
from collections import defaultdict

import numpy as np

MODEL = "google/gemma-4-26B-A4B-it"
RAW = "results/sni/binning_seed20212003/test_raw.jsonl"
# 축 계수(results/sni/axis_coord_route.md)로 갈린 계열
INVERT = {"c_18467", "c_19704", "c_61797"}


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="표본 문제 수")
    ap.add_argument("--out", default="results/sni/token_divergence.md")
    a = ap.parse_args()
    rng = random.Random(0)

    cell = defaultdict(lambda: defaultdict(list))
    for line in open(RAW, encoding="utf-8"):
        r = json.loads(line)
        cell[r["pid"]][r["cid"]].append(r.get("code") or "")
    pids = sorted(cell)
    rng.shuffle(pids)
    pids = pids[:a.n]
    ex = sorted(cell[pids[0]])
    names = {p["id"]: p["name"] for p in
             json.load(open("results/sni/seed20212003/roster_final.json"))}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    T = {}
    for pid in pids:
        T[pid] = {c: [tok(t, add_special_tokens=False)["input_ids"] for t in cell[pid][c]]
                  for c in ex}

    cross, within, cross_in, cross_out = [], [], [], []
    lens = defaultdict(list)
    agree = np.zeros(20); agree_w = np.zeros(20); nc = nw = 0
    for pid in pids:
        for c in ex:
            for s in T[pid][c]:
                lens[c].append(len(s))
        # 페르소나 간 (rep0끼리)
        for i in range(len(ex)):
            for j in range(i + 1, len(ex)):
                A, B = T[pid][ex[i]][0], T[pid][ex[j]][0]
                k = common_prefix(A, B)
                cross.append(k / max(1, min(len(A), len(B))))
                (cross_in if (ex[i] in INVERT) == (ex[j] in INVERT) else cross_out).append(k)
                m = min(len(A), len(B), 20)
                agree[:m] += np.array([A[t] == B[t] for t in range(m)]); nc += 1
        # 같은 페르소나 다른 rep (디코딩 노이즈 바닥)
        for c in ex:
            reps = T[pid][c]
            for i in range(len(reps)):
                for j in range(i + 1, len(reps)):
                    A, B = reps[i], reps[j]
                    k = common_prefix(A, B)
                    within.append(k / max(1, min(len(A), len(B))))
                    m = min(len(A), len(B), 20)
                    agree_w[:m] += np.array([A[t] == B[t] for t in range(m)]); nw += 1

    L = [f"# 페르소나는 몇 번째 토큰부터 갈라지나 (표본 {len(pids):,}문제, 재생성 0)", "",
         f"- 토크나이저 `{MODEL}` · 페르소나 쌍 {nc:,}개 · 같은 페르소나 rep 쌍 {nw:,}개", "",
         "## 위치별 토큰 일치율", "",
         "| 토큰 위치 | 페르소나 **간** | 같은 페르소나 **다른 rep**(노이즈 바닥) |",
         "|---:|---:|---:|"]
    for t in (0, 1, 2, 4, 9, 19):
        L.append(f"| {t+1} | {100*agree[t]/nc:.1f}% | {100*agree_w[t]/nw:.1f}% |")
    L += ["", "## 공통 접두사 길이 (짧은 쪽 길이로 정규화)", "",
          "| 쌍 | 중앙값 | 평균 | 완전 동일 비율 |", "|---|---:|---:|---:|"]
    for tag, v in (("페르소나 간", cross), ("같은 페르소나 다른 rep", within)):
        v = np.array(v)
        L.append(f"| {tag} | {np.median(v):.3f} | {v.mean():.3f} | {100*(v >= 1).mean():.1f}% |")
    L += ["", "## 계열 간 vs 계열 내 (공통 접두사 토큰 수, 중앙값)", "",
          f"- 같은 계열끼리: {np.median(cross_in):.1f} 토큰",
          f"- 반전 계열 vs 나머지: {np.median(cross_out):.1f} 토큰", "",
          "## 페르소나별 출력 길이 (토큰, 중앙값)", "", "| 중앙값 | expert |", "|---:|---|"]
    for c in sorted(ex, key=lambda c: -np.median(lens[c])):
        L.append(f"| {np.median(lens[c]):.0f} | {names.get(c, c)} |")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
