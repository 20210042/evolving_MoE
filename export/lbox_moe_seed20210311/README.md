# LBOX 10+1 expert 학습 패키지 (seed20210311, arm=ours)

- train 문제 46,019 → 학습 row 99,504 (개별 배정 구간은 한 문제가 여러 expert에 중복 등장한다)
- 평가셋(배정 없음, 조건 공통): `valid.jsonl` 7,651 · `test.jsonl` 8,203
- 원본에서 못 찾은 id: 0

## 파일

| 파일 | 내용 |
|---|---|
| `manifest.json` | 배정 규칙·expert 목록·행 수. 여기부터 읽으면 된다 |
| `train/expert_00..09_<id>.jsonl` | routed expert 10명의 학습 데이터 |
| `train/shared.jsonl` | shared expert(짬통) 학습 데이터 |
| `<split>.jsonl` | 공통 평가셋 (같은 프롬프트 형식) |

## row 스키마

```json
{"id":..., "expert":..., "kind":"indiv|shared|all_fail", "n_solved":0-10,
 "system":"...", "user":"...", "target":"gold 문자열", "targets":["gold 전체"],
 "task_type":..., "task_config":..., "casetype":...}
```

`system`/`user`는 이미 조립돼 있다. 그대로 chat template에 넣으면 된다.
**페르소나는 들어 있지 않다** — teacher가 분할을 만들 때만 썼고 student 학습에는 넣지 않는다.
`manifest.json`의 `persona_system_prompt_TEACHER_ONLY`는 참고용이다.

## expert별 학습량

| # | expert | 학습 row | 개별(n≤10) | 전원실패 |
|---:|---|---:|---:|---:|
| 0 | Judicial Precedent Classifier | **9,155** | 8,758 | 397 |
| 1 | Legal Case Typology Architect | **14,326** | 9,415 | 4,911 |
| 2 | Statutory Element Matcher | **10,362** | 5,417 | 4,945 |
| 3 | Legal Nomenclature Purist | **9,326** | 8,999 | 327 |
| 4 | Legal Fact Synthesis Engine | **8,882** | 7,084 | 1,798 |
| 5 | Legal Provision Auditor | **5,627** | 4,212 | 1,415 |
| 6 | Civil Dispute Taxonomy Expert | **7,462** | 7,009 | 453 |
| 7 | Judicial Labeling Precisionist | **10,587** | 7,239 | 3,348 |
| 8 | Legal Recidivism Analyst | **6,450** | 5,123 | 1,327 |
| 9 | Criminal Charge Aggregator | **6,253** | 4,936 | 1,317 |
| 10 | shared expert | **11,074** | 11,074 | 0 |

## 채점

정답 일치 (src/evaluation/scorer.py: score_lbox_item) — casename은 정규화 후 완전일치, statute는 gold 집합 대조

