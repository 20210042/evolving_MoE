# LBOX per-expert binning (seed20210311)

최종 진화 로스터의 **10개 전문가가 lbox train 전체(46,019문제)를 각자 독립으로**
풀고, 문제별로 누가 맞췄는지 라벨링한 결과.
- backbone: `google/gemma-4-26B-A4B-it` (Thinking OFF)
- dataset: `lbox` train, 46,019 problems
- 채점: EM (pass@1) · **union UB 56.0%**
- provenance: seed `20210311` · git `8050731` · 2026-07-10

## 파일
| 파일 | 내용 |
|---|---|
| `binning_labels.jsonl` | 문제별 `{id, solved_by[], n_solved, per_expert{agent_id:0|1}}` — **핵심 학습 신호** |
| `agent_mapping.json` / `.csv` | codename → name · **system_prompt** · strengths · train pass@1 |
| `agent_solves.json` | 에이전트별 solved/failed 문제 id |
| `summary.json` | per-expert · union UB · coverage + provenance |

## 에이전트 매핑
| agent_id | name | train pass@1 | strengths |
|---|---|---|---|
| c_28126 | Legal Case Typology Architect | 44.5% | Mastery of judicial taxonomy; ability to distinguish between substanti |
| c_24222 | Legal Nomenclature Purist | 43.6% | Mastery of the official Korean judicial lexicon; ability to distinguis |
| c_29934 | Judicial Precedent Classifier | 43.1% | Expertise in converting dense factual descriptions into standardized,  |
| c_47388 | Legal Fact Synthesis Engine | 39.1% | Expertise in multi-crime aggregation; ability to synthesize disparate  |
| c_16504 | Civil Dispute Taxonomy Expert | 38.9% | Expertise in distilling complex civil factual patterns into precise ju |
| c_31181 | Judicial Labeling Precisionist | 35.1% | Extreme focus on single-line output constraints; ability to strip away |
| c_63621 | Statutory Element Matcher | 34.7% | Expertise in performing microscopic element-to-fact mapping; identifie |
| c_4799 | Legal Recidivism Analyst | 34.6% | Expertise in analyzing criminal history (범죄전력) to identify recidivism  |
| c_31573 | Criminal Charge Aggregator | 33.7% | Expertise in multi-offense consolidation; ability to extract and list  |
| c_27344 | Legal Provision Auditor | 32.4% | Expertise in multi-provision identification; high precision in detecti |

전체 system_prompt은 `agent_mapping.json` 참조.

## 사용례 (⭐ 주 목적 = persona-specific SFT)
라벨은 문제 `id`만 있고 **실제 입력(질문/facts)+정답은 원본 데이터셋**에 있음 → `id`로 조인.

```python
import json
P   = "export/lbox_binning_seed20210311"
labels = [json.loads(l) for l in open(f"{P}/binning_labels.jsonl")]
agents = json.load(open(f"{P}/agent_mapping.json"))      # agent_id -> name/system_prompt/strengths/pass@1
src    = {json.loads(l)["id"]: json.loads(l)              # id로 원본(입력+gold) 조인
          for l in open("export/lbox/lbox_train.jsonl")}

# ⭐ (a) persona-specific SFT 학습셋: "그 전문가가 맞춘 문제 + 그 전문가 system_prompt를 페르소나로"
def expert_sft_set(aid):
    persona = agents[aid]["system_prompt"]
    for r in labels:
        if r["per_expert"].get(aid) == 1:               # 이 전문가가 맞춘 문제만
            it = src[r["id"]]
            gold = it["ground_truth"]
            if isinstance(gold, list):                  # lbox statute 등은 set → 문자열화
                gold = ", ".join(gold)
            yield {"system": persona, "input": it["instruction"], "output": gold}

# 예: 최고 성능 전문가의 SFT 데이터 만들기
best = max(agents, key=lambda a: agents[a]["train_pass_at_1"])
data = list(expert_sft_set(best))
print(best, agents[best]["name"], "→", len(data), "SFT examples")

# (참고) 부차 용도
router_sup = {r["id"]: r["solved_by"] for r in labels}            # (b) 라우터 지도
hard       = [r["id"] for r in labels if 0 < r["n_solved"] <= 2]    # (c) 난이도 커리큘럼
```

## 스키마 (라벨 1줄 예시)
```json
{"id": "<problem id>", "dataset": "lbox", "solved_by": ["luca", "c_xxxx"], "n_solved": 2, "per_expert": {"luca": 1, "c_xxxx": 1, "c_yyyy": 0}}
```
`per_expert[agent_id] == 1` 이면 그 전문가가 그 문제를 맞힘. `id`로 `export/lbox/lbox_train.jsonl` 조인.
