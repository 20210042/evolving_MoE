#!/usr/bin/env python3
"""조건2(Random) · 조건3(Human-prior) MoE용 라벨패키지 생성.

통제구조: 세 MoE 모두 [specialized(cap10, n_solved<=10) + shared(evolved 재사용)].
specialized 배정만 다름:
  - evolved  : roster solve-clustering (기존)
  - human-prior: qasc_llm_tags.json 8과목 disjoint
  - random   : count-matched 랜덤 (evolved 대응 expert와 동일 데이터 '양', 멤버십만 랜덤)

evolved binning_labels.jsonl에서 id·dataset·n_solved 재사용. 출력 패키지는 런처
train_sft_by_expert.sh 파싱에 맞춰 qasc_binning_seed{hp,rnd} 로 명명(DOMAIN=qasc).
"""
import json
from pathlib import Path

import numpy as np

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")
EVOLVED = REPO / "export/qasc_binning_seed20210211/binning_labels.jsonl"
TAGS = REPO / "results/embed_viz/qasc_llm_tags.json"
SRC = "export/qasc/qasc_train.jsonl"
SEED = 20210211

rows = [json.loads(l) for l in open(EVOLVED)]
ids = [str(r["id"]) for r in rows]
nsolved = {str(r["id"]): int(r["n_solved"]) for r in rows}
evolved_ex = list(rows[0]["per_expert"].keys())
CAP = 10
cap_pool = [i for i in ids if nsolved[i] <= CAP]     # specialized 후보 (5587)


def write_pkg(name, per_expert_by_id, experts, note):
    d = REPO / "export" / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "binning_labels.jsonl", "w", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps({"id": i, "dataset": "qasc", "n_solved": nsolved[i],
                                "per_expert": per_expert_by_id[i]}, ensure_ascii=False) + "\n")
    mapping = {e: {"name": e, "system_prompt": "You are a helpful assistant.",
                   "strengths": note, "train_pass_at_1": 0.0} for e in experts}
    json.dump(mapping, open(d / "agent_mapping.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump({"source_train_jsonl": SRC, "experts": experts, "note": note,
               "cap_max_n_solved": CAP}, open(d / "summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    vol = {e: sum(v.get(e, 0) for v in per_expert_by_id.values()) for e in experts}
    print(f"[{name}] experts={len(experts)}  volumes(post-cap 배정)={vol}")
    return d


# ---------- 조건3: Human-prior (8과목 disjoint) ----------
tags = json.load(open(TAGS))
TAX = ["biology", "ecology", "earth science", "chemistry", "physics",
       "astronomy", "health", "other"]
hp_ex = ["hp_" + t.replace(" ", "_") for t in TAX]
subj2ex = {t: "hp_" + t.replace(" ", "_") for t in TAX}
hp_pe = {}
for i in ids:
    subj = tags.get(i, "other")
    ex = subj2ex.get(subj, "hp_other")
    # cap10 적용: n_solved>10인 쉬운 문제는 specialized 학습 제외(shared가 담당)
    hp_pe[i] = {e: (1 if (e == ex and nsolved[i] <= CAP) else 0) for e in hp_ex}
write_pkg("qasc_binning_seedhp", hp_pe, hp_ex, "human-prior subject partition (LLM-tagged)")

# ---------- 조건2: Random (count-matched) ----------
rng = np.random.default_rng(SEED)
# evolved 각 expert의 post-cap 학습 볼륨을 목표로 랜덤 표본(겹침 허용, evolved도 겹침)
targets = {e: sum(1 for i in cap_pool if rows[ids.index(i)]["per_expert"][e] == 1)
           for e in evolved_ex}
rnd_ex = [f"rnd_{k:02d}" for k in range(len(evolved_ex))]
rnd_pe = {i: {e: 0 for e in rnd_ex} for i in ids}
pool = np.array(cap_pool)
for k, ev in enumerate(evolved_ex):
    n = targets[ev]
    sel = rng.choice(len(pool), size=n, replace=False)
    for j in sel:
        rnd_pe[pool[j]][rnd_ex[k]] = 1
write_pkg("qasc_binning_seedrnd", rnd_pe, rnd_ex,
          "random count-matched partition (evolved 대응 expert와 동일 볼륨)")

print("done. shared는 evolved 체크포인트 재사용(재학습 X).")
