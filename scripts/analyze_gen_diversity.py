#!/usr/bin/env python3
"""전문가들이 '다른 시도'를 하는지 생성물 수준에서 측정.

핵심 질문: 아무도 못 푸는 문제(코딩 75.8%)에서 12명이 서로 다른 프로그램을 내는가,
같은 오답을 12번 내는가. 전자면 능력 한계, 후자면 진화가 클론을 만든 것 -> 개선 여지.
"""
import json
import re
from itertools import combinations

import numpy as np

B = "results/acc/seed20210111_v2/ablation/"
CONDS = {"Evolved": "evolved", "Random": "rnd", "Human-prior": "hp"}


def norm(code: str) -> str:
    """주석/공백/변수명 잡음을 줄인 토큰 시퀀스."""
    c = re.sub(r"#.*", "", code or "")
    return " ".join(re.findall(r"[A-Za-z_]\w*|\d+|[^\s\w]", c))


def jac(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def analyze(tag):
    gen = {json.loads(l)["id"]: json.loads(l) for l in open(f"{B}inference_test751_{tag}.jsonl")}
    binned = {json.loads(l)["id"]: json.loads(l) for l in open(f"{B}inference_test751_{tag}.binned.jsonl")}
    E = len(next(iter(gen.values()))["expert_outputs"])

    buckets = {"all-fail": [], "contested": [], "all-solve": []}
    for pid, g in gen.items():
        b = binned[pid]
        n = b["n_solved"]
        key = "all-fail" if n == 0 else ("all-solve" if n == E else "contested")
        outs = list(g["expert_outputs"].values())
        toks = [set(norm(o).split()) for o in outs]
        sims = [jac(toks[i], toks[j]) for i, j in combinations(range(len(toks)), 2)]
        exact = len({norm(o) for o in outs})
        buckets[key].append((float(np.mean(sims)), exact, len(outs)))

    print(f"\n=== {tag}  (E={E}) ===")
    print(f"{'bucket':11s} {'n':>5s} {'mean pairwise Jaccard':>22s} {'distinct programs':>19s}")
    for k, v in buckets.items():
        if not v:
            print(f"{k:11s} {0:5d}")
            continue
        sims = np.array([x[0] for x in v])
        dis = np.array([x[1] for x in v], dtype=float)
        print(f"{k:11s} {len(v):5d} {sims.mean():22.3f} {dis.mean():13.2f} / {v[0][2]}")
    return buckets


for name, tag in CONDS.items():
    analyze(tag)
