#!/usr/bin/env python3
"""파일럿 A(현재 배포, LoRA)/B(persona)/C(persona+few-shot) 생성 다양성 비교.

analyze_gen_diversity.py와 동일 지표(pairwise Jaccard, distinct programs)를
pilot_persona_fewshot_gen.py가 고른 동일 30문제 위에서 세 조건에 대해 계산한다.
"""
import argparse
import json
import re
from itertools import combinations

import numpy as np

ABL = "results/acc/seed20210111_v2/ablation/inference_test751_evolved.jsonl"
PILOT_DIR = "results/pilot_persona_fewshot"


def norm(code: str) -> str:
    c = re.sub(r"#.*", "", code or "")
    return " ".join(re.findall(r"[A-Za-z_]\w*|\d+|[^\s\w]", c))


def jac(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def summarize(tag: str, rows: dict, pilot_ids: list):
    sims, exacts, ns = [], [], []
    for pid in pilot_ids:
        outs = list(rows[pid]["expert_outputs"].values())
        toks = [set(norm(o).split()) for o in outs]
        pair = [jac(toks[i], toks[j]) for i, j in combinations(range(len(toks)), 2)]
        sims.append(float(np.mean(pair)) if pair else 1.0)
        exacts.append(len({norm(o) for o in outs}))
        ns.append(len(outs))
    sims = np.array(sims)
    exacts = np.array(exacts, dtype=float)
    print(f"{tag:20s} n={len(pilot_ids):3d}  mean pairwise Jaccard={sims.mean():.3f}  "
          f"distinct programs={exacts.mean():.2f} / {ns[0]}")


def correctness(tag: str, path: str, pilot_ids: list):
    try:
        binned = {json.loads(l)["id"]: json.loads(l) for l in open(path)}
    except FileNotFoundError:
        print(f"{tag:20s} (채점 결과 없음: {path})")
        return
    n_solved_any = sum(1 for pid in pilot_ids if binned[pid]["n_solved"] > 0)
    print(f"{tag:20s} union 정답 {n_solved_any}/{len(pilot_ids)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", choices=["all-fail", "contested"], default="all-fail")
    a = ap.parse_args()
    suffix = "" if a.bucket == "all-fail" else f"_{a.bucket}"

    b_rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{PILOT_DIR}/gen_B_persona{suffix}.jsonl")}
    c_rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{PILOT_DIR}/gen_C_persona_fewshot{suffix}.jsonl")}
    pilot_ids = list(b_rows.keys())

    a_full = {json.loads(l)["id"]: json.loads(l) for l in open(ABL)}
    a_rows = {pid: a_full[pid] for pid in pilot_ids}
    # A는 12 experts(roster+luca+shared), B/C는 11(shared 제외) — expert 수 차이가
    # distinct-programs 절대값에 영향줄 수 있으니 Jaccard(쌍별 평균)를 주 지표로 본다.

    print(f"=== 파일럿 비교 (동일 {len(pilot_ids)}문제, {a.bucket} 버킷) — 생성 다양성 ===")
    summarize("A: 배포(LoRA, no-persona)", a_rows, pilot_ids)
    summarize("B: persona only", b_rows, pilot_ids)
    summarize("C: persona+few-shot", c_rows, pilot_ids)

    a_binned = {json.loads(l)["id"]: json.loads(l) for l in open(
        "results/acc/seed20210111_v2/ablation/inference_test751_evolved.binned.jsonl")}
    print(f"\n=== 정답률 (union, {a.bucket} 버킷) ===")
    n_a = sum(1 for pid in pilot_ids if a_binned[pid]["n_solved"] > 0)
    print(f"{'A: 배포(LoRA, no-persona)':20s} union 정답 {n_a}/{len(pilot_ids)}")
    correctness("B: persona only", f"{PILOT_DIR}/gen_B_persona{suffix}.binned.jsonl", pilot_ids)
    correctness("C: persona+few-shot", f"{PILOT_DIR}/gen_C_persona_fewshot{suffix}.binned.jsonl", pilot_ids)


if __name__ == "__main__":
    main()
