#!/usr/bin/env python3
"""같은 문제에서 '검증된 참조 솔루션들'과 '우리 전문가들'의 다양성을 직접 대조한다.

analyze_ref_solution_diversity.py는 train 전체 분포를 보고, 이 스크립트는
**동일한 test751 문제** 위에서 참조간 겹침 vs 전문가간 겹침을 짝지어 비교한다
(문제 난이도·내용이 통제된 비교).

읽는 법: 차이가 음수면 참조들이 전문가들보다 서로 더 다르다
        = 이 도메인에 접근 차원이 실재하고 우리 로스터가 그것을 못 담고 있다.
"""
import json
import re
from itertools import combinations

import numpy as np

SRC = ("/data5/jaehoonjeong/.cache/huggingface/hub/datasets--QuantCat--Algorithm-Dataset-filtered"
       "/snapshots/136cf2bb8dcb8fa6a0611c4237db5703012dc505/acc_algorithm_test.jsonl")
B = "results/acc/seed20210111_v2/ablation/"
TAG = "evolved"


def norm(c):
    return " ".join(re.findall(r"[A-Za-z_]\w*|\d+|[^\s\w]", re.sub(r"#.*", "", c or "")))


def jac(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def J(v):
    return json.loads(v) if isinstance(v, str) else v


binned = {json.loads(l)["id"]: json.loads(l) for l in open(f"{B}inference_test751_{TAG}.binned.jsonl")}
gen = {json.loads(l)["id"]: json.loads(l) for l in open(f"{B}inference_test751_{TAG}.jsonl")}

refs = {}
for l in open(SRC, encoding="utf-8"):
    d = json.loads(l)
    ok = [s["code"] for s in (J(d.get("reference_solutions")) or [])
          if isinstance(s, dict) and s.get("is_known_correct")
          and s.get("language") == "python" and s.get("code")]
    if ok:
        refs[d.get("problem_id")] = ok

pid_of = {json.loads(l)["id"]: json.loads(l)["problem_id"] for l in open("export/acc_v2/acc_test.jsonl")}

rows = {"all-fail": [], "contested": [], "all-solve": []}
for tid, b in binned.items():
    E = len(gen[tid]["expert_outputs"])
    n = b["n_solved"]
    key = "all-fail" if n == 0 else ("all-solve" if n == E else "contested")
    R = refs.get(pid_of.get(tid))
    if not R or len(R) < 2:
        continue
    rt = [set(norm(c).split()) for c in R]
    et = [set(norm(o).split()) for o in gen[tid]["expert_outputs"].values()]
    rows[key].append((np.mean([jac(rt[i], rt[j]) for i, j in combinations(range(len(rt)), 2)]),
                      np.mean([jac(et[i], et[j]) for i, j in combinations(range(len(et)), 2)])))

print(f"{'bucket':11s} {'n':>5s} {'참조간 겹침':>12s} {'전문가간 겹침':>14s} {'차이':>8s}")
for k, v in rows.items():
    if not v:
        continue
    a = np.array(v)
    print(f"{k:11s} {len(v):5d} {a[:,0].mean():12.3f} {a[:,1].mean():14.3f} "
          f"{a[:,0].mean()-a[:,1].mean():+8.3f}")
a = np.array(rows["all-fail"])
print(f"\nall-fail에서 참조간 겹침 0.5 미만: {100*(a[:,0]<0.5).mean():.1f}%")
print("주의: test 원본에서 참조 2개 이상인 문제만 대상이라 표본이 작다(train 전체 분포는 "
      "analyze_ref_solution_diversity.py 참조 — 3598문제 평균 0.540).")
