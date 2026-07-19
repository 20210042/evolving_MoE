#!/usr/bin/env python3
"""코딩(acc) MoL 사다리용 SFT 자산 구축 — seed20210111.

acc 원본에는 참조 솔루션이 없다(실행채점 도메인). 따라서 binning 단계에서
에이전트가 생성하고 **테스트로 검증된** 코드를 SFT 타깃으로 승격시킨다(rejection sampling).

QASC와 동일 논리 유지: **문제당 canonical 정답코드 1개**를 고정하고,
per_expert는 "어떤 문제를 보느냐"만 결정한다(타깃은 expert 무관 동일).
canonical 선택 규칙: luca가 풀었으면 luca, 아니면 정렬순 첫 solver (결정적·중립).

출력:
  export/acc/acc_train.jsonl       — solution 포함 (SFT 소스, solver>=1 문제만)
  export/acc/acc_validation.jsonl  — 학습 eval용 held-out (binning 밖 문제에서)
  export/acc_binning_seed20210111/ — 라벨패키지(binning_labels/agent_mapping/summary)
"""
import json
from pathlib import Path

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
SEED = "seed20210111"
GEN = REPO / f"results/acc/{SEED}/binning_train_full.jsonl"          # expert_outputs(생성코드)
BIN = REPO / f"results/acc/{SEED}/binning_train_full.binned.jsonl"   # per_expert(정오)
SRC = REPO / "export/acc_selfconsistent/acc_train.jsonl"             # 원본 문제(솔루션 없음)
OUTD = REPO / "export/acc"
PKG = REPO / f"export/acc_binning_{SEED}"
N_EVAL = 500

gen = {str(json.loads(l)["id"]): json.loads(l)["expert_outputs"] for l in open(GEN, encoding="utf-8")}
binned = {str(json.loads(l)["id"]): json.loads(l) for l in open(BIN, encoding="utf-8")}
src = {str(json.loads(l)["id"]): json.loads(l) for l in open(SRC, encoding="utf-8")}
EX = sorted(next(iter(binned.values()))["per_expert"])
print(f"binning {len(binned)}문제 × {len(EX)} experts | 원본 {len(src)}문제")


def canonical(pid):
    """검증된 정답코드 1개 선택: luca 우선, 없으면 정렬순 첫 solver."""
    pe = binned[pid]["per_expert"]
    solvers = [e for e in EX if pe.get(e, 0) == 1]
    if not solvers:
        return None
    pick = "luca" if "luca" in solvers else solvers[0]
    code = (gen.get(pid) or {}).get(pick)
    return code.strip() if isinstance(code, str) and code.strip() else None


OUTD.mkdir(parents=True, exist_ok=True)
train_rows, no_sol = [], 0
for pid in binned:
    if pid not in src:
        continue
    sol = canonical(pid)
    if sol is None:
        no_sol += 1
        continue
    r = dict(src[pid])
    r["solution"] = sol            # SFT completion
    r["ground_truth"] = sol        # 폴백(build_regular_hf_dataset 호환)
    train_rows.append(r)
with open(OUTD / "acc_train.jsonl", "w", encoding="utf-8") as f:
    for r in train_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"acc_train.jsonl: {len(train_rows)}문제 (정답코드 없어 제외 {no_sol})")

# eval split: binning에 안 쓰인 원본 문제에서 (누수 없음). 솔루션 필요 → 없으므로
# binning 문제 중 학습에서 뺀 뒤쪽 N_EVAL개를 held-out으로 사용.
eval_rows = train_rows[-N_EVAL:]
train_rows = train_rows[:-N_EVAL]
with open(OUTD / "acc_train.jsonl", "w", encoding="utf-8") as f:
    for r in train_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUTD / "acc_validation.jsonl", "w", encoding="utf-8") as f:
    for r in eval_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"  → train {len(train_rows)} / validation {len(eval_rows)} (held-out, 학습에서 제외)")

# ---- 라벨패키지 ----
PKG.mkdir(parents=True, exist_ok=True)
train_ids = {str(r["id"]) for r in train_rows}
kept = 0
with open(PKG / "binning_labels.jsonl", "w", encoding="utf-8") as f:
    for pid, b in binned.items():
        if pid not in train_ids:            # eval held-out·솔루션없음 제외
            continue
        f.write(json.dumps({"id": pid, "dataset": "acc", "n_solved": int(b["n_solved"]),
                            "per_expert": b["per_expert"]}, ensure_ascii=False) + "\n")
        kept += 1
mapping = {e: {"name": e, "system_prompt": "You are a helpful assistant.",
               "strengths": f"acc {SEED} evolved expert", "train_pass_at_1": 0.0} for e in EX}
json.dump(mapping, open(PKG / "agent_mapping.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump({"source_train_jsonl": "export/acc/acc_train.jsonl", "experts": EX,
           "note": "verified-correct agent code as SFT target (canonical per problem)"},
          open(PKG / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"라벨패키지: {kept}문제 × {len(EX)} experts -> {PKG}")

# 볼륨 요약
import collections
for cap in (10, 9, 8):
    rows = [json.loads(l) for l in open(PKG / "binning_labels.jsonl", encoding="utf-8")]
    v = {e: sum(1 for r in rows if r["per_expert"].get(e, 0) == 1 and r["n_solved"] <= cap) for e in EX}
    print(f"  cap{cap}: min={min(v.values())} max={max(v.values())} 평균={sum(v.values())//len(v)}")
print(f"  shared(min_n_solved=10): {sum(1 for r in rows if r['n_solved']>=10)}")
