#!/usr/bin/env python3
"""조건2(Random) · 조건3(Human-prior) MoE용 라벨패키지 생성.

통제구조: 세 MoE 모두 [specialized(cap, n_solved<=cap) + shared(evolved 재사용)].
specialized 배정만 다름:
  - evolved   : roster solve-clustering (기존)
  - human-prior: 사람 분류축 disjoint (qasc=LLM 8과목, acc=critic category 5범주)
  - random    : count-matched 랜덤 (evolved 대응 expert와 동일 데이터 '양', 멤버십만 랜덤)

evolved binning_labels.jsonl에서 id·dataset·n_solved 재사용. 출력 패키지는 런처
train_sft_by_expert.sh 파싱(<domain>_binning_seed<seed>)에 맞춰 명명.

사용: python scripts/build_ablation_packages.py --dataset qasc|acc
"""
import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path("/data5/jaehoonjeong/MetaAgentEvolution_Release")


def qasc_subjects(ids):
    """qasc: LLM 태깅 8과목 (외부 json, id->subject)."""
    tags = json.load(open(REPO / "results/embed_viz/qasc_llm_tags.json"))
    tax = ["biology", "ecology", "earth science", "chemistry", "physics",
           "astronomy", "health", "other"]
    return {i: tags.get(i, "other") for i in ids}, tax


def acc_subjects(ids, src_path="export/acc/acc_train.jsonl"):
    """acc: minji LLM critic 태깅 5범주 (SFT 소스의 main_critic_category)."""
    src = {str(json.loads(l)["id"]): json.loads(l)
           for l in open(REPO / src_path, encoding="utf-8")}
    tax = ["Greedy Strategy", "Constructive Implementation", "Quantitative Reasoning",
           "State-Space Reasoning", "Structured Data"]
    return {i: (src.get(i, {}).get("main_critic_category") or "other") for i in ids}, tax


CONFIGS = {
    "qasc": dict(evolved="export/qasc_binning_seed20210211/binning_labels.jsonl",
                 src="export/qasc/qasc_train.jsonl", seed=20210211, cap=10,
                 pkg_prefix="qasc_binning", subjects=qasc_subjects),
    "acc": dict(evolved="export/acc_binning_seed20210111/binning_labels.jsonl",
                src="export/acc/acc_train.jsonl", seed=20210111, cap=9,
                pkg_prefix="acc_binning", subjects=acc_subjects),
}

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="qasc", choices=sorted(CONFIGS))
# 코퍼스를 재빌드하면 evolved 패키지·SFT 소스 경로가 바뀐다. 미지정 시 기존 경로 그대로.
ap.add_argument("--evolved", default=None, help="evolved binning_labels.jsonl 경로 override")
ap.add_argument("--src", default=None, help="SFT 소스 train jsonl 경로 override")
ap.add_argument("--suffix", default="", help="출력 패키지 이름 접미사 (예: _v2)")
A = ap.parse_args()
C = CONFIGS[A.dataset]
DS = A.dataset
EVOLVED = REPO / (A.evolved or C["evolved"])
SRC = A.src or C["src"]
SEED = C["seed"]
CAP = C["cap"]

rows = [json.loads(l) for l in open(EVOLVED)]
ids = [str(r["id"]) for r in rows]
nsolved = {str(r["id"]): int(r["n_solved"]) for r in rows}
evolved_ex = list(rows[0]["per_expert"].keys())
cap_pool = [i for i in ids if nsolved[i] <= CAP]     # specialized 후보


def slug(s):
    return "hp_" + s.lower().replace(" ", "_")


def write_pkg(name, per_expert_by_id, experts, note):
    d = REPO / "export" / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "binning_labels.jsonl", "w", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps({"id": i, "dataset": DS, "n_solved": nsolved[i],
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


# ---------- 조건3: Human-prior (disjoint) ----------
subj_of, TAX = (acc_subjects(ids, SRC) if DS == "acc" else C["subjects"](ids))
hp_ex = [slug(t) for t in TAX]
subj2ex = {t: slug(t) for t in TAX}
hp_pe = {}
for i in ids:
    subj = subj_of.get(i, "other")
    ex = subj2ex.get(subj, slug("other"))
    # cap 적용: n_solved>cap인 쉬운 문제는 specialized 학습 제외(shared가 담당)
    hp_pe[i] = {e: (1 if (e == ex and nsolved[i] <= CAP) else 0) for e in hp_ex}
write_pkg(f"{C['pkg_prefix']}_seedhp{A.suffix}", hp_pe, hp_ex,
          "human-prior partition (LLM-tagged subject/category)")

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
write_pkg(f"{C['pkg_prefix']}_seedrnd{A.suffix}", rnd_pe, rnd_ex,
          "random count-matched partition (evolved 대응 expert와 동일 볼륨)")

print("done. shared는 evolved 체크포인트 재사용(재학습 X).")
