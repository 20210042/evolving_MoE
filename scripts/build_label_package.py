#!/usr/bin/env python3
"""Wrap full-train binning outputs into a self-contained handoff package for the
collaborator (weight-level MoE / per-expert SFT). Mirrors export/numina_binning_seed16/.

Per-expert SFT 합의(2026-07-13): system prompt는 도메인 baseline GEN 프롬프트로 통일,
expert별로 "그 expert가 맞춘 문제"만 학습 데이터로 분리. 페르소나는 학습에 쓰지 않음.

Produces export/<domain>_binning_seed<seed>/:
  binning_labels.jsonl   per-problem {id, solved_by[], n_solved, per_expert{id:0|1}}
  agent_mapping.json/.csv codename -> name / system_prompt / strengths / train pass@1
  agent_solves.json      per-agent solved/failed ids (copied)
  summary.json           per-expert / union UB / coverage + provenance
  README.md              backbone·dataset·seed·git + agent table + runnable quickstart
Read-only over results/; writes only under export/.
"""
from __future__ import annotations
import argparse, csv, json, shutil, subprocess, datetime
from pathlib import Path


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help="qasc | lbox")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--results", required=True, help="results/<domain>/seed<seed> dir")
    ap.add_argument("--source", required=True, help="source train jsonl (id-joined inputs)")
    ap.add_argument("--backbone", default="google/gemma-4-26B-A4B-it")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    R = Path(a.results)
    labels_p = R / "binning_train_full.binned.jsonl"
    summary_p = R / "binning_train_full.binned.summary.json"
    solves_p = R / "binning_train_full.binned.agent_solves.json"
    roster_p = R / "roster_final.json"
    out = Path(a.out or f"export/{a.domain}_binning_seed{a.seed}")
    out.mkdir(parents=True, exist_ok=True)

    summary = json.load(open(summary_p))
    roster = json.load(open(roster_p))
    pe = summary.get("per_expert_pass_at_1", {})

    # agent_mapping: codename -> profile
    mapping = {}
    for p in roster:
        aid = p["id"]
        mapping[aid] = {
            "name": p.get("name") or p.get("persona_name") or aid,
            "system_prompt": p.get("system_prompt", ""),
            "strengths": p.get("strengths", ""),
            "train_pass_at_1": round(pe.get(aid, 0.0), 2),
        }
    json.dump(mapping, open(out / "agent_mapping.json", "w"), ensure_ascii=False, indent=2)
    with open(out / "agent_mapping.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent_id", "name", "train_pass_at_1", "strengths"])
        for aid, m in mapping.items():
            w.writerow([aid, m["name"], m["train_pass_at_1"], m["strengths"]])

    shutil.copy(labels_p, out / "binning_labels.jsonl")
    if solves_p.exists():
        shutil.copy(solves_p, out / "agent_solves.json")

    prov = {
        "domain": a.domain, "seed": a.seed, "backbone": a.backbone,
        "git_commit": git_commit(), "built": datetime.date.today().isoformat(),
        "source_train_jsonl": a.source, "n_problems": summary.get("total"),
        "n_experts": len(summary.get("experts", [])),
    }
    json.dump({**prov, **summary}, open(out / "summary.json", "w"), ensure_ascii=False, indent=2)

    # agent table (sorted by train pass@1 desc)
    rows = sorted(mapping.items(), key=lambda kv: -kv[1]["train_pass_at_1"])
    tbl = "| agent_id | name | train pass@1 | strengths |\n|---|---|---|---|\n"
    for aid, m in rows:
        tbl += f"| {aid} | {m['name']} | {m['train_pass_at_1']:.1f}% | {m['strengths'][:70]} |\n"

    readme = f"""# {a.domain.upper()} per-expert binning (seed{a.seed})

최종 진화 로스터의 **{prov['n_experts']}개 전문가가 {a.domain} train 전체({prov['n_problems']:,}문제)를 각자 독립으로**
풀고, 문제별로 누가 맞췄는지 라벨링한 결과.
- backbone: `{a.backbone}` (Thinking OFF)
- dataset: `{a.domain}` train, {prov['n_problems']:,} problems
- 채점: EM (pass@1) · **union UB {summary.get('union_ub_pct', 0):.1f}%**
- provenance: seed `{a.seed}` · git `{prov['git_commit']}` · {prov['built']}

## 파일
| 파일 | 내용 |
|---|---|
| `binning_labels.jsonl` | 문제별 `{{id, solved_by[], n_solved, per_expert{{agent_id:0|1}}}}` — **핵심 학습 신호** |
| `agent_mapping.json` / `.csv` | codename → name · **system_prompt** · strengths · train pass@1 |
| `agent_solves.json` | 에이전트별 solved/failed 문제 id |
| `summary.json` | per-expert · union UB · coverage + provenance |

## 에이전트 매핑
{tbl}
전체 system_prompt은 `agent_mapping.json` 참조.

## 사용례 (⭐ 주 목적 = per-expert SFT: system prompt는 baseline GEN으로 통일, 보는 데이터만 분리)
라벨은 문제 `id`만 있고 **실제 입력(질문/facts)+정답은 원본 데이터셋**에 있음 → `id`로 조인.
**페르소나(system_prompt)는 학습에 쓰지 않음**(MoE 병합 이슈로 합의) — `agent_mapping.json`의
system_prompt는 라벨이 어떤 전문가에서 나왔는지 해석하는 참고용.

> ⚠️ 이 패키지엔 **라벨만** 포함(원본은 용량상 git 제외). 사용 전 **원본을 먼저 재생성**:
> ```bash
> python scripts/build_{a.domain}.py   # → {a.source}
> ```

러너블 경로 (권장): `src/train_sft.py`가 이 패키지를 직접 읽음.
```bash
python src/train_sft.py --label_package {out.as_posix()} --expert_id c_xxxx \\
    --eval_dataset {a.domain} --eval_split validation --data_dir export/{a.domain} ...
```

수동 조립 시(동일 로직):
```python
import json
from prompts import baseline_prompts as bp                 # PYTHONPATH=src
GEN_SYS, GEN_USER = bp.{a.domain.upper()}_GEN_SYSTEM, bp.{a.domain.upper()}_GEN_USER

P   = "{out.as_posix()}"
labels = [json.loads(l) for l in open(f"{{P}}/binning_labels.jsonl")]
agents = json.load(open(f"{{P}}/agent_mapping.json"))      # agent_id -> name/system_prompt/strengths/pass@1
src    = {{json.loads(l)["id"]: json.loads(l)              # id로 원본(입력+gold) 조인
          for l in open("{a.source}")}}

# ⭐ (a) per-expert SFT 학습셋: "그 전문가가 맞춘 문제만 + system은 baseline GEN으로 통일"
def expert_sft_set(aid):
    for r in labels:
        if r["per_expert"].get(aid) == 1:               # 이 전문가가 맞춘 문제만
            it = src[r["id"]]
            gold = it["ground_truth"]
            if isinstance(gold, list):                  # lbox statute 등은 set → 문자열화
                gold = ", ".join(gold)
            yield {{"system": GEN_SYS,
                    "input": GEN_USER.format(instruction=it["instruction"]),
                    "output": gold}}

# 예: 최고 성능 전문가의 SFT 데이터 만들기
best = max(agents, key=lambda a: agents[a]["train_pass_at_1"])
data = list(expert_sft_set(best))
print(best, agents[best]["name"], "→", len(data), "SFT examples")

# (참고) 부차 용도
router_sup = {{r["id"]: r["solved_by"] for r in labels}}            # (b) 라우터 지도
hard       = [r["id"] for r in labels if 0 < r["n_solved"] <= 2]    # (c) 난이도 커리큘럼
```

## 스키마 (라벨 1줄 예시)
```json
{{"id": "<problem id>", "dataset": "{a.domain}", "solved_by": ["luca", "c_xxxx"], "n_solved": 2, "per_expert": {{"luca": 1, "c_xxxx": 1, "c_yyyy": 0}}}}
```
`per_expert[agent_id] == 1` 이면 그 전문가가 그 문제를 맞힘. `id`로 `{a.source}` 조인.
"""
    open(out / "README.md", "w").write(readme)
    print(f"built package → {out}  ({prov['n_problems']} problems, {prov['n_experts']} experts, UB {summary.get('union_ub_pct',0):.1f}%)")


if __name__ == "__main__":
    main()
