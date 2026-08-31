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
| 0 | luca | **12,717** | Question Answering, Fill in The Blank, Explanation, Answer Verification, Answerability Classification |
| 1 | c_10299 | **8,000** | Program Execution, Text to Code, Mathematics, Question Understanding, Dialogue Act Recognition, … 외 5개 |
| 2 | c_58325 | **6,011** | Question Generation, Answerability Classification, Pos Tagging, Summarization, Question Understanding, … 외 15개 |
| 3 | c_24456 | **4,415** | Sentiment Analysis, Negotiation Strategy Detection, Summarization, Dialogue State Tracking, Text Quality Evaluation, … 외 9개 |
| 4 | c_17188 | **8,028** | Misc., Word Analogy, Data to Text, Word Semantics, Fill in The Blank, … 외 31개 |
| 5 | c_18467 | **3,190** | Toxic Language Detection, Irony Detection, Linguistic Probing, Stereotype Detection, Stance Detection, … 외 7개 |
| 6 | c_56632 | **2,923** | Text Categorization, Question Understanding, Stereotype Detection, Text Quality Evaluation, Keyword Tagging, … 외 5개 |
| 7 | c_13393 | **2,653** | Textual Entailment, Stereotype Detection, Cause Effect Classification, Word Semantics, Coherence Classification, … 외 10개 |
| 8 | c_29228 | **3,245** | Commonsense Classification, Text to Code, Code to Text, Question Rewriting, Word Relation Classification, … 외 7개 |
| 9 | c_53171 | **2,751** | Title Generation, Summarization, Keyword Tagging, Answerability Classification, Story Composition, … 외 12개 |
| 10 | c_49611 | **1,860** | Text Matching, Answerability Classification, Stance Detection, Linguistic Probing, Question Understanding, … 외 9개 |
| 11 | c_2461 | **2,258** | Named Entity Recognition, Style Transfer, Paraphrasing, Sentence Composition, Sentence Perturbation, … 외 14개 |
| 12 | c_19704 | **3,021** | Information Extraction, Gender Classification, Word Semantics, Keyword Tagging, Dialogue Generation, … 외 21개 |
| 13 | c_9508 | **3,236** | Wrong Candidate Generation, Story Composition, Question Rewriting, Coherence Classification, Sentence Composition, … 외 18개 |
| 14 | c_61797 | **2,131** | Coreference Resolution, Pos Tagging, Answerability Classification, Translation, Overlap Extraction, … 외 18개 |
| 15 | c_46890 | **3,149** | Text Completion, Speaker Identification, Dialogue Generation, Dialogue Act Recognition, Coherence Classification, … 외 20개 |

## 채점

`src/evaluation/scorer.py: score_sni_item` — **EM==100 또는 ROUGE-L>70**이면 통과.
공식(Tk-Instruct)은 임계 없이 EM·ROUGE를 병기한다. 이진 판정은 우리 쪽 요구사항이다.

