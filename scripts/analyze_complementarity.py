#!/usr/bin/env python3
"""두 도메인(QASC/coding)의 solve 매트릭스에서 상보성 구조를 뽑는다.

교수님 피드백 대응:
  - "role들이 실제로 얼마나 다른 답을 내나" -> 실패상관 / unique-solve / Jaccard
  - "role이 많으면 무조건 좋아지는 것 아닌가, 메트릭이 틀린 것 아닌가" -> union@k 곡선
    (같은 k에서 조건 비교 = size-controlled). random-subset 기대값으로 로스터 크기 통제.
  - "아예 못 푸는 문제도 표시" -> solved-by-none
"""
import json
import sys
from itertools import combinations

import numpy as np

RUNS = {
    "QASC": {
        "Evolved(cap10)": "results/qasc/seed20210211/inference_validation_lora13.binned.jsonl",
        "Random":         "results/qasc/seed20210211/inference_validation_rndmoe.binned.jsonl",
        "Human-prior":    "results/qasc/seed20210211/inference_validation_hpmoe.binned.jsonl",
        "Evolved(cap7)":  "results/qasc/seed20210211/inference_validation_cap7moe.binned.jsonl",
    },
    "CODING": {
        "Evolved(cap9)":  "results/acc/seed20210111_v2/ablation/inference_test751_evolved.binned.jsonl",
        "Random":         "results/acc/seed20210111_v2/ablation/inference_test751_rnd.binned.jsonl",
        "Human-prior":    "results/acc/seed20210111_v2/ablation/inference_test751_hp.binned.jsonl",
    },
}


def load(path):
    rows = [json.loads(l) for l in open(path)]
    experts = sorted(rows[0]["per_expert"].keys())
    M = np.array([[r["per_expert"][e] for e in experts] for r in rows], dtype=np.int8)
    return M, experts


def phi_failcorr(M):
    """실패(=0) 벡터 간 평균 pearson 상관. expert 실패가 얼마나 같이 일어나나."""
    F = 1 - M.astype(float)
    cs = []
    for i, j in combinations(range(F.shape[1]), 2):
        a, b = F[:, i], F[:, j]
        if a.std() == 0 or b.std() == 0:
            continue
        cs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(cs)) if cs else float("nan")


def mean_jaccard(M):
    js = []
    for i, j in combinations(range(M.shape[1]), 2):
        a, b = M[:, i] == 1, M[:, j] == 1
        u = (a | b).sum()
        js.append((a & b).sum() / u if u else 0.0)
    return float(np.mean(js))


def union_at_k_random(M, k, trials=3000, rng=None):
    """무작위로 고른 k명의 union 기대값 — 로스터 크기만의 효과."""
    rng = rng or np.random.default_rng(0)
    E = M.shape[1]
    if k >= E:
        return float((M.max(1) == 1).mean() * 100)
    out = []
    for _ in range(trials):
        idx = rng.choice(E, size=k, replace=False)
        out.append((M[:, idx].max(1) == 1).mean())
    return float(np.mean(out) * 100)


def union_at_k_greedy(M, k):
    """oracle greedy로 고른 k명의 union — 상보성의 상한 구조."""
    chosen, cov = [], np.zeros(M.shape[0], dtype=bool)
    for _ in range(k):
        best, bg = None, -1
        for e in range(M.shape[1]):
            if e in chosen:
                continue
            g = ((M[:, e] == 1) & ~cov).sum()
            if g > bg:
                best, bg = e, g
        chosen.append(best)
        cov |= (M[:, best] == 1)
    return float(cov.mean() * 100), chosen


def report(domain, name, path):
    M, experts = load(path)
    n, E = M.shape
    p1 = M.mean(0) * 100
    union = (M.max(1) == 1).mean() * 100
    none = (M.max(1) == 0).mean() * 100
    allsolved = (M.min(1) == 1).mean() * 100
    uniq = [(M[:, e] == 1).sum() and int(((M[:, e] == 1) & (M.sum(1) == 1)).sum()) for e in range(E)]
    print(f"\n### [{domain}] {name}   n={n} experts={E}")
    print(f"  pass@1  best {p1.max():.2f}  mean {p1.mean():.2f}  worst {p1.min():.2f}")
    print(f"  union(all E) {union:.2f}   nobody-solves {none:.2f}   everybody-solves {allsolved:.2f}")
    print(f"  failure-corr(mean pairwise) {phi_failcorr(M):.3f}   solve-Jaccard {mean_jaccard(M):.3f}")
    print(f"  unique-solve total {sum(uniq)} ({100*sum(uniq)/n:.2f}% of problems)  per-expert max {max(uniq)}")
    ks = [1, 2, 3, 4, 6, 8, min(9, E), min(12, E), E]
    ks = sorted(set(k for k in ks if k <= E))
    print("  union@k  " + "  ".join(f"k{k}" for k in ks))
    print("   random : " + "  ".join(f"{union_at_k_random(M,k):5.1f}" for k in ks))
    print("   greedy : " + "  ".join(f"{union_at_k_greedy(M,k)[0]:5.1f}" for k in ks))
    return dict(domain=domain, name=name, n=n, E=E, best=p1.max(), mean=p1.mean(),
                union=union, none=none, allsolved=allsolved,
                failcorr=phi_failcorr(M), jacc=mean_jaccard(M),
                uniq=sum(uniq),
                u_at=lambda k: union_at_k_random(M, k), M=M)


def main():
    res = []
    for domain, runs in RUNS.items():
        print("=" * 78)
        print(f"===== {domain} =====")
        for name, path in runs.items():
            try:
                res.append(report(domain, name, path))
            except FileNotFoundError:
                print(f"  !! missing {path}")

    # size-controlled 직접 대조: Evolved vs Random을 같은 k에서
    print("\n" + "=" * 78)
    print("=== SIZE-CONTROLLED: Evolved vs Random (동일 k, random-subset 기대 union) ===")
    for domain in RUNS:
        ev = next((r for r in res if r["domain"] == domain and r["name"].startswith("Evolved(cap1")
                   or r["domain"] == domain and r["name"].startswith("Evolved(cap9")), None)
        rd = next((r for r in res if r["domain"] == domain and r["name"] == "Random"), None)
        if not ev or not rd:
            continue
        E = min(ev["E"], rd["E"])
        print(f"\n[{domain}]  k :  " + "  ".join(f"{k:5d}" for k in range(1, E + 1)))
        for r, tag in ((ev, "Evolved"), (rd, "Random ")):
            print(f"  {tag}: " + "  ".join(f"{union_at_k_random(r['M'],k):5.1f}" for k in range(1, E + 1)))
        print("  diff   : " + "  ".join(
            f"{union_at_k_random(ev['M'],k)-union_at_k_random(rd['M'],k):+5.1f}" for k in range(1, E + 1)))


if __name__ == "__main__":
    main()
