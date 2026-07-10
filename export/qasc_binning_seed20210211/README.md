# QASC per-expert binning (seed20210211)

최종 진화 로스터의 **12개 전문가가 qasc train 전체(8,134문제)를 각자 독립으로**
풀고, 문제별로 누가 맞췄는지 라벨링한 결과.
- backbone: `google/gemma-4-26B-A4B-it` (Thinking OFF)
- dataset: `qasc` train, 8,134 problems
- 채점: EM (pass@1) · **union UB 81.3%**
- provenance: seed `20210211` · git `8050731` · 2026-07-10

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
| c_33055 | Evolutionary Biologist | 60.1% | Expertise in natural selection, population genetics, adaptation mechan |
| c_33241 | Applied Mathematical Modeler | 59.9% | Expertise in volumetric measurements, geometric scaling, rate calculat |
| c_64009 | Cognitive Neuroscientist | 59.1% | Expertise in sensory processing, neurobiology, behavioral responses to |
| c_9221 | Microbiology Specialist | 59.0% | Expertise in microbial morphology, cellular reproduction mechanisms, p |
| c_4502 | Zoological Biologist | 58.8% | Expertise in animal kinematics, mammalian thermoregulation, migratory  |
| c_35649 | Earth Science Generalist | 58.6% | Expertise in the water cycle, condensation processes, precipitation ty |
| c_46537 | Geological Process Specialist | 57.7% | Expertise in weathering mechanisms, sediment transport, fluvial geomor |
| c_61510 | General Ecology Specialist | 57.5% | Expertise in trophic levels, food webs, basic plant and animal life cy |
| c_1516 | Human Biology Expert | 57.0% | Expertise in human diseases (HIV, HPV, cancer), human physiological re |
| c_13270 | Environmental Science Generalist | 51.3% | Expertise in ecological processes, pollution sources, habitat alterati |
| luca | LUCA | 46.4% | General programming assistance and baseline critique. |
| c_54731 | Biological Systems Specialist | 29.3% | Deep knowledge of microbiology, botany, zoology, anatomy, and ecosyste |

전체 system_prompt은 `agent_mapping.json` 참조.

## 사용례 (⭐ 주 목적 = persona-specific SFT)
라벨은 문제 `id`만 있고 **실제 입력(질문/facts)+정답은 원본 데이터셋**에 있음 → `id`로 조인.

```python
import json
P   = "export/qasc_binning_seed20210211"
labels = [json.loads(l) for l in open(f"{P}/binning_labels.jsonl")]
agents = json.load(open(f"{P}/agent_mapping.json"))      # agent_id -> name/system_prompt/strengths/pass@1
src    = {json.loads(l)["id"]: json.loads(l)              # id로 원본(입력+gold) 조인
          for l in open("export/qasc/qasc_train.jsonl")}

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
{"id": "<problem id>", "dataset": "qasc", "solved_by": ["luca", "c_xxxx"], "n_solved": 2, "per_expert": {"luca": 1, "c_xxxx": 1, "c_yyyy": 0}}
```
`per_expert[agent_id] == 1` 이면 그 전문가가 그 문제를 맞힘. `id`로 `export/qasc/qasc_train.jsonl` 조인.
