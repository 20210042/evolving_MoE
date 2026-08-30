# SNI 16+1 expert 학습 패키지 (seed20212003, arm=random)

- train 문제 69,588 → 학습 row 104,612 (개별 배정 구간은 한 문제가 여러 expert에 중복 등장한다)
- test 8,699 (배정 없음, 공통 평가셋)
- 원본에서 못 찾은 id: 0

## 파일

| 파일 | 내용 |
|---|---|
| `manifest.json` | 배정 규칙·expert 목록·행 수. 여기부터 읽으면 된다 |
| `train/expert_00..15_<id>.jsonl` | routed expert 16명의 학습 데이터 |
| `train/shared.jsonl` | shared expert(짬통) 학습 데이터 |
| `test.jsonl` | 공통 평가셋 (같은 프롬프트 형식) |

## row 스키마

```json
{"id":..., "expert":..., "kind":"indiv|shared|all_fail", "n_solved":0-16,
 "system":"...", "user":"...", "target":"gold 문자열", "targets":["gold 전체"],
 "task_name":..., "category":..., "sni_domain":..., "task_closed":...}
```

`system`/`user`는 이미 조립돼 있다. 그대로 chat template에 넣으면 된다.
**페르소나는 들어 있지 않다** — teacher가 분할을 만들 때만 썼고 student 학습에는 넣지 않는다.
expert 슬롯 id는 파일 형식을 다른 조건과 맞추려고 재사용한 것일 뿐 **페르소나와 아무 관계가 없다**.

## expert별 학습량

| # | expert | 학습 row | 개별(n≤10) | 전원실패 |
|---:|---|---:|---:|---:|
| 0 | expert 0 | **6,845** | 2,284 | 4,561 |
| 1 | expert 1 | **3,282** | 2,884 | 398 |
| 2 | expert 2 | **4,967** | 3,318 | 1,649 |
| 3 | expert 3 | **2,573** | 2,383 | 190 |
| 4 | expert 4 | **3,299** | 2,831 | 468 |
| 5 | expert 5 | **3,280** | 2,879 | 401 |
| 6 | expert 6 | **3,846** | 2,530 | 1,316 |
| 7 | expert 7 | **3,197** | 2,872 | 325 |
| 8 | expert 8 | **4,714** | 2,785 | 1,929 |
| 9 | expert 9 | **2,987** | 2,954 | 33 |
| 10 | expert 10 | **3,169** | 2,746 | 423 |
| 11 | expert 11 | **6,805** | 3,282 | 3,523 |
| 12 | expert 12 | **3,603** | 3,333 | 270 |
| 13 | expert 13 | **3,754** | 2,273 | 1,481 |
| 14 | expert 14 | **3,660** | 2,894 | 766 |
| 15 | expert 15 | **3,753** | 3,262 | 491 |
| 16 | shared expert | **40,878** | 40,878 | 0 |

## 채점

`src/evaluation/scorer.py: score_sni_item` — **EM==100 또는 ROUGE-L>70**이면 통과.
공식(Tk-Instruct)은 임계 없이 EM·ROUGE를 병기한다. 이진 판정은 우리 쪽 요구사항이다.

