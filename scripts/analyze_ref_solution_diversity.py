#!/usr/bin/env python3
"""같은 문제에 대한 '검증된 참조 솔루션들'이 서로 얼마나 다른가.

이 도메인에 접근(approach) 차원이 실재하는지에 대한 상한/근거.
우리 로스터 12명의 all-fail 생성물 겹침(0.729)과 같은 잣대로 비교한다.
  - 참조끼리 겹침이 낮다  -> 접근 차원이 실재하고 우리 로스터가 그걸 못 담고 있다
  - 참조끼리도 겹침이 높다 -> 이 도메인에 접근 다양성 자체가 얇다 (방향 재고)

analyze_gen_diversity.py와 동일한 정규화·Jaccard를 쓴다.
"""
import argparse
import json
import re
from itertools import combinations

import numpy as np

SRC = ("/data5/jaehoonjeong/.cache/huggingface/hub/datasets--QuantCat--Algorithm-Dataset-filtered"
       "/snapshots/136cf2bb8dcb8fa6a0611c4237db5703012dc505/acc_algorithm_train.jsonl")

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=SRC)
ap.add_argument("--limit", type=int, default=4000)
A = ap.parse_args()


def norm(code):
    c = re.sub(r"#.*", "", code or "")
    return " ".join(re.findall(r"[A-Za-z_]\w*|\d+|[^\s\w]", c))


def jac(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def J(v):
    return json.loads(v) if isinstance(v, str) else v


sims, ns, exact_dup = [], [], 0
n = 0
for line in open(A.src, encoding="utf-8"):
    d = json.loads(line)
    n += 1
    refs = J(d.get("reference_solutions")) or []
    ok = [s["code"] for s in refs if isinstance(s, dict) and s.get("is_known_correct")
          and s.get("language") == "python" and s.get("code")]
    if len(ok) < 2:
        continue
    toks = [set(norm(c).split()) for c in ok]
    pair = [jac(toks[i], toks[j]) for i, j in combinations(range(len(toks)), 2)]
    sims.append(float(np.mean(pair)))
    ns.append(len(ok))
    if len({norm(c) for c in ok}) < len(ok):
        exact_dup += 1
    if n >= A.limit:
        break

s = np.array(sims)
print(f"표본 문제 {n} / 참조 2개 이상 {len(s)}  (평균 참조 수 {np.mean(ns):.2f})")
print(f"같은 문제 참조솔루션 간 pairwise Jaccard: mean {s.mean():.3f}  median {np.median(s):.3f}")
print(f"  분위: p10 {np.percentile(s,10):.3f}  p25 {np.percentile(s,25):.3f} "
      f" p75 {np.percentile(s,75):.3f}  p90 {np.percentile(s,90):.3f}")
print(f"  겹침 0.5 미만(뚜렷이 다른 접근) 비율: {100*(s<0.5).mean():.1f}%")
print(f"  겹침 0.3 미만 비율: {100*(s<0.3).mean():.1f}%")
print(f"  완전 중복 포함 문제: {exact_dup} ({100*exact_dup/max(len(s),1):.1f}%)")
print(f"\n[대조] 우리 로스터 12전문가 all-fail 생성물 겹침 = 0.729 "
      f"(analyze_gen_diversity.py)")
