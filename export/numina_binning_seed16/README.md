# NuminaMath per-expert binning (seed16)

최종 진화 로스터의 **5개 전문가 에이전트가 numina_cot train 전체(62,185문제)를 각자 독립으로** 풀고,
문제별로 누가 맞췄는지 라벨링한 결과.
- backbone: `google/gemma-4-26B-A4B-it` (A4B MoE), Thinking OFF
- dataset: `numina_cot` train split, 62,185 problems
- 채점: math 정답 일치(MC-aware), pass@1

## 에이전트 매핑 (코드네임 → 설명)

| agent_id | 이름 | train pass@1 | 전문 영역 |
|---|---|---|---|
| c_37753 | Combinatorial Probability Expert | 73.24% | 이산수학·경우의 수: 순열/조합/확률분포 |
| c_49174 | Integral Calculus Expert | 73.14% | 정적분·부정적분: 넓이/일/누적 문제 |
| c_46599 | Polynomial Expansion Expert | 73.10% | 이항·다항 전개: 특정 계수 추출, 부분합 |
| c_48902 | Arithmetic Number Theory Expert | 72.75% | 정수론: 약수/배수, 모듈러, LCM, 소인수분해 |
| c_48046 | Analytic Proof Specialist | 72.35% | 형식 증명: 귀류법/귀납법/직접증명 |

전체 system_prompt·strengths는 `agent_mapping.json` / `agent_mapping.csv` 참조.

## 파일

| 파일 | 설명 |
|---|---|
| `binning_labels.jsonl` | 문제별 per-expert 풀이 라벨 (62,185줄). 컬럼은 아래 |
| `agent_mapping.json` | agent_id → name / system_prompt / strengths / train pass@1 |
| `agent_mapping.csv` | 위와 동일, 스프레드시트용 |
| `summary.json` | 집계 통계 |

### `binning_labels.jsonl` 컬럼

각 줄 = 문제 1개:

| 키 | 타입 | 설명 |
|---|---|---|
| `id` | str | 문제 id (예: `numina_cot_algebra_155`) |
| `dataset` | str | `numina_cot` |
| `solved_by` | list[str] | **맞춘 agent_id 목록** (빈 리스트 = 아무도 못 풂) |
| `n_solved` | int | 맞춘 에이전트 수 (0~5) |
| `per_expert` | dict[str,int] | 5명 전원에 대해 `agent_id: 1(맞춤)/0(틀림)` |

예:
```json
{"id":"numina_cot_algebra_155","dataset":"numina_cot",
 "solved_by":["c_37753","c_49174","c_48046","c_48902","c_46599"],
 "n_solved":5,
 "per_expert":{"c_37753":1,"c_49174":1,"c_48046":1,"c_48902":1,"c_46599":1}}
```

### `summary.json` 주요 필드

- `per_expert_pass_at_1` / `per_expert_solved`: 에이전트별 pass@1(%) 및 맞춘 문제 수
- `union_ub`, `union_ub_pct`: 48,242 (77.58%)
- `coverage_histogram`: `n_solved → 문제 수`
  - `0`: 13,943 (22.4%, 전원 실패)
  - `1`: 1,601 (2.6%, 5명중 1명만 푼 문제)
  - `2`: 1,159 · `3`: 1,320 · `4`: 1,982
  - `5`: 42,180 (67.8%, 전원 풀이)

> ⚠️ 해석 caveat: bin이 대거 중복(67.8% 전원풀이, 단독 2.6%뿐)이라 per-expert 학습신호가 약함