#!/usr/bin/env python3
"""QASC판 답변 다양성 — 전문가들이 서로 다른 보기를 고르는가.

코딩의 생성 다양성(analyze_gen_diversity.py)과 같은 질문을 MC에서 직접 본다.
특히 '아무도 못 푸는' 문제에서 전원이 같은 오답에 수렴하는지(= 공통 오개념),
아니면 흩어지는지(= 다양성은 있으나 전부 틀림)를 가른다.

레터 추출은 현행(버그 수정 후) `_extract_qasc_letter`를 그대로 쓴다.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from evaluation.scorer import _extract_qasc_letter  # noqa: E402

B = ROOT / "results/qasc/seed20210211"
CONDS = {"Evolved(cap10)": "lora13", "Random": "rndmoe", "Human-prior": "hpmoe"}


def entropy(counts):
    p = np.array(list(counts.values()), dtype=float)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def analyze(name, tag):
    gen = {json.loads(l)["id"]: json.loads(l) for l in open(B / f"inference_validation_{tag}.jsonl")}
    binned = {json.loads(l)["id"]: json.loads(l) for l in open(B / f"inference_validation_{tag}.binned.jsonl")}
    E = len(next(iter(gen.values()))["expert_outputs"])

    buckets = {"all-fail": [], "contested": [], "all-solve": []}
    unparsed = 0
    for pid, g in gen.items():
        b = binned.get(pid)
        if b is None:
            continue
        n = b["n_solved"]
        key = "all-fail" if n == 0 else ("all-solve" if n == E else "contested")
        letters = []
        for out in g["expert_outputs"].values():
            L = _extract_qasc_letter(out)
            if L is None:
                unparsed += 1
                L = "?"
            letters.append(L)
        c = Counter(letters)
        buckets[key].append((len(c), entropy(c), c.most_common(1)[0][1] / len(letters),
                             str(g.get("ground_truth", "")).strip().upper(), c))

    print(f"\n=== {name} ({tag}, E={E}) ===   추출실패 {unparsed}/{len(gen)*E}")
    print(f"{'bucket':11s} {'n':>5s} {'distinct letters':>17s} {'entropy(bit)':>13s} {'modal share':>12s}")
    for k, v in buckets.items():
        if not v:
            continue
        d = np.array([x[0] for x in v], float)
        e = np.array([x[1] for x in v], float)
        m = np.array([x[2] for x in v], float)
        print(f"{k:11s} {len(v):5d} {d.mean():17.2f} {e.mean():13.3f} {m.mean():12.3f}")

    # all-fail에서 전원이 '같은 하나의 오답'에 수렴한 비율
    af = buckets["all-fail"]
    if af:
        unan = sum(1 for x in af if x[0] == 1)
        print(f"  all-fail 중 전원 동일 오답 수렴: {unan}/{len(af)} ({100*unan/len(af):.1f}%)")
        print(f"  all-fail 중 정답을 고른 전문가가 하나도 없음(검산): "
              f"{sum(1 for x in af if x[3] not in x[4])}/{len(af)}")


for name, tag in CONDS.items():
    analyze(name, tag)
