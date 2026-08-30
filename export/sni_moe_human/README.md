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
| 0 | luca | **12,499** | Question Answering |
| 1 | c_10299 | **7,157** | Program Execution |
| 2 | c_58325 | **4,044** | Question Generation |
| 3 | c_24456 | **3,572** | Text Matching, Dialogue Generation, Stereotype Detection, Sentence Composition, Number Conversion, … 외 1개 |
| 4 | c_17188 | **3,570** | Wrong Candidate Generation, Summarization, Data to Text, Explanation, Stance Detection, … 외 2개 |
| 5 | c_18467 | **3,567** | Textual Entailment, Linguistic Probing, Negotiation Strategy Detection, Sentence Perturbation, Sentence Expansion |
| 6 | c_56632 | **3,565** | Sentiment Analysis, Style Transfer, Preposition Prediction |
| 7 | c_13393 | **3,560** | Title Generation, Word Semantics, Cause Effect Classification, Coherence Classification, Question Decomposition, … 외 1개 |
| 8 | c_29228 | **3,524** | Toxic Language Detection, Gender Classification, Mathematics, Speaker Relation Classification |
| 9 | c_53171 | **3,519** | Named Entity Recognition, Question Rewriting, Word Analogy, Dialogue State Tracking, Irony Detection, … 외 1개 |
| 10 | c_49611 | **3,503** | Text Categorization, Fill in The Blank, Code to Text, Answer Verification, Entity Relation Classification |
| 11 | c_2461 | **3,502** | Misc., Keyword Tagging, Grammar Error Detection, Grammar Error Correction |
| 12 | c_19704 | **3,502** | Information Extraction, Text to Code, Text Quality Evaluation, Word Relation Classification, Overlap Extraction, … 외 1개 |
| 13 | c_9508 | **3,502** | Coreference Resolution, Question Understanding, Story Composition, Text Simplification, Sentence Ordering, … 외 1개 |
| 14 | c_61797 | **3,502** | Text Completion, Answerability Classification, Speaker Identification, Intent Identification, Fact Verification, … 외 1개 |
| 15 | c_46890 | **3,500** | Commonsense Classification, Pos Tagging, Dialogue Act Recognition, Paraphrasing, Sentence Compression |

## 채점

`src/evaluation/scorer.py: score_sni_item` — **EM==100 또는 ROUGE-L>70**이면 통과.
공식(Tk-Instruct)은 임계 없이 EM·ROUGE를 병기한다. 이진 판정은 우리 쪽 요구사항이다.

