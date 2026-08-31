# SNI 16+1 expert 학습 패키지 (seed20212003, arm=human)

- train 문제 69,588 → 학습 row 69,588 (개별 배정 구간은 한 문제가 여러 expert에 중복 등장한다)
- test 8,699 (배정 없음, 공통 평가셋)
- 원본에서 못 찾은 id: 0

## 파일

| 파일 | 내용 |
|---|---|
| `manifest.json` | 배정 규칙·expert 목록·행 수. 여기부터 읽으면 된다 |
| `train/expert_00..15_<id>.jsonl` | routed expert 16명의 학습 데이터 |
| (shared 없음) | 순수 BTX 조건이라 shared expert가 없다 |
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

| # | expert 슬롯 | 학습 row | 맡은 category |
|---:|---|---:|---|
| 0 | luca | **9,338** | Question Answering, Question Generation, Answerability Classification, Question Understanding, Coreference Resolution, … 외 28개 |
| 1 | c_10299 | **7,924** | Program Execution, Question Answering, Misc., Mathematics, Question Understanding, … 외 5개 |
| 2 | c_58325 | **7,480** | Question Answering, Misc., Word Analogy, Question Generation, Fill in The Blank, … 외 29개 |
| 3 | c_24456 | **5,816** | Dialogue Generation, Speaker Identification, Negotiation Strategy Detection, Dialogue Act Recognition, Toxic Language Detection, … 외 20개 |
| 4 | c_17188 | **4,043** | Toxic Language Detection, Sentiment Analysis, Text Categorization, Text Completion, Irony Detection, … 외 8개 |
| 5 | c_18467 | **4,428** | Question Answering, Text Matching, Text Categorization, Summarization, Question Generation, … 외 20개 |
| 6 | c_56632 | **4,381** | Sentiment Analysis, Question Answering, Summarization, Title Generation, Question Generation, … 외 13개 |
| 7 | c_13393 | **3,540** | Program Execution, Text to Code, Misc., Code to Text, Answerability Classification, … 외 8개 |
| 8 | c_29228 | **4,933** | Textual Entailment, Misc., Gender Classification, Word Semantics, Stereotype Detection, … 외 17개 |
| 9 | c_53171 | **3,027** | Question Answering, Question Generation, Information Extraction, Textual Entailment, Explanation, … 외 16개 |
| 10 | c_49611 | **2,398** | Question Answering, Text Completion, Wrong Candidate Generation, Story Composition, Textual Entailment, … 외 11개 |
| 11 | c_2461 | **2,403** | Commonsense Classification, Text to Code, Code to Text, Misc., Fill in The Blank, … 외 1개 |
| 12 | c_19704 | **2,298** | Program Execution, Textual Entailment, Text Completion, Data to Text, Pos Tagging, … 외 9개 |
| 13 | c_9508 | **1,468** | Linguistic Probing, Question Answering, Sentiment Analysis, Title Generation, Cause Effect Classification, … 외 4개 |
| 14 | c_61797 | **4,297** | Question Answering, Named Entity Recognition, Information Extraction, Text Categorization, Title Generation, … 외 5개 |
| 15 | c_46890 | **1,814** | Misc., Question Answering, Text Completion, Number Conversion, Question Generation, … 외 11개 |

## 채점

`src/evaluation/scorer.py: score_sni_item` — **EM==100 또는 ROUGE-L>70**이면 통과.
공식(Tk-Instruct)은 임계 없이 EM·ROUGE를 병기한다. 이진 판정은 우리 쪽 요구사항이다.

