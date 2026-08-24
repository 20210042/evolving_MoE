# SNI 프로브 v2 — 사전등록 설계

2026-08-22 · 브랜치 `jh/sni-probe` · 선행: [REFLECTION_sni_probe.md](REFLECTION_sni_probe.md)

**사전등록 문서다.** 판독 규칙과 해석 순서를 잡 제출 **전에** 확정해 둔다.
런이 끝난 뒤 여기 없는 판독을 새로 만들어 결론을 내지 않는다.

---

## 0. 이 프로브가 묻는 것

지금까지 acc·QASC에서 확인된 것: **페르소나는 문체만 바꾸고 답은 그대로 낸다.**
그러면 물어야 할 것은 "축이 있나"가 아니라 —

> **문체가 채점에 들어가는 층이 있는가. 있다면 어떤 성질의 층인가.**
> 그리고 **어떤 축으로 자른 로스터가 출력에 어떤 변화를 주는가.**

SNI를 최종 레짐으로 쓰든 안 쓰든, 객관식 층과 서술형 층의 대비 자체가
다음 진화를 어떤 데이터에서 돌릴지, 로스터를 어떤 축으로 자를지의 근거가 된다.

## 1. 대상 모집단 — 자르지 않는다

**SNI 전수 87,089건**(875 task, task당 100 상한). 표집 규칙을 두지 않으므로 대상 풀을
설계자가 구성할 여지가 없다.

⚠️ **기술적 제외 61건(0.070%) → 대상 87,028건.** 프롬프트가 모델 컨텍스트(16,384토큰)를
넘으면 vllm이 배치 전체를 거부하고 잡이 죽는다(job 229520이 그렇게 실패). 전부 CUAD 계약서
전문 3개 task이고 최대 64,813토큰이다 — task598 24건 / task597 22건 / task599 15건으로,
세 task 자체는 남고 일부 인스턴스만 빠지므로 축·로스터 구성에는 영향이 없다.
범위 판단이 아니라 모델 한계이며, 제외 id 전량을 `results/sni/excluded_over_context.json`에
남긴다(`scripts/sni_context_filter.py`).

gold 길이·답 개방도·category는 **판독 층화 변수로만** 쓴다(표집 아님).

참고 — 코퍼스 구조(전수 실측):

| gold 중앙길이 | task | items |
|---|---:|---:|
| 1–3단어 | 611 | 60,947 |
| 3–8 | 83 | 8,231 |
| 8–15 | 107 | 10,585 |
| 15–40 | 63 | 6,300 |
| 40+ | 11 | 1,026 |

서술형(≥15단어) 74 task의 내역: Summarization 8 · Question Generation 8 · Data to Text 7 ·
Story Composition 4 · Explanation 3 · Text Simplification 3 · Translation 2 · Dialogue 2 ·
Paraphrasing · Poem Generation … **그리고 Program Execution 10 · Text to Code 4 ·
Number Conversion 2 · Pos Tagging 1**. 후자는 길지만 답이 유일하다 —
**빼지 않고 서술형 내부의 음성대조로 쓴다**(문체가 갈려도 점수가 안 움직여야 하는 층).

## 2. 로스터 — 데이터에 실재하는 축 2개 + luca

| 조건 | 인원 | 선정 |
|---|---:|---|
| `luca` | 1 | 중립 baseline |
| **category 로스터** | N | SNI `Categories` 라벨 중 item 수 상위 N (기계적) |
| **domain 로스터** | N | SNI `Domains` 최상위 라벨 중 item 수 상위 N (기계적) |

- **인원수 N을 두 축에서 맞춘다.** union·WAR은 인원수에 민감해서 크기가 다르면 축 비교가 불공정하다.
- **N = 12 제안.** 커버리지는 category 61.2% / domain 67.8%(같은 N에서 커버리지는 축마다 다르다 —
  이건 데이터의 성질이고, 인원수를 맞추는 쪽을 택했다는 사실을 판독에 명시한다).
- 페르소나 문구는 [configs/roster_sni_probe.json](../configs/roster_sni_probe.json)을 **그대로 재사용**한다.
  기존 로스터가 정확히 category 상위 12 · domain 상위 10이므로,
  새로 쓰는 것은 domain 11·12위(**Story**, **Sociology**) 두 명뿐이고 기존과 같은 문형으로 채운다.
- **문체 축 로스터는 만들지 않는다.** 문체를 프롬프트로 직접 지정하는 건 지어낸 축이고,
  축이 무엇이든 바뀌는 게 문체라는 것 자체가 이번 관측 대상이다.
- **대조는 luca만.** 노이즈 바닥은 luca를 K회 재생성한 within-luca 분산이 준다.

생성량: 25명 × 87,089 × K=3 = **6,531,675** ≈ **14.5시간**(실측 처리량 125 gen/s), 48h 헤더 안.

## 3. 프롬프트

**계약: system = 정체성(로스터) + 태스크 정의, user = 답변공간 한 줄 + 입력.**

v1의 결함은 "정의를 줬다"가 아니라 **정의를 user 턴에 둬서 페르소나를 눌렀다**는 것이다
(gemma는 user 지시를 더 강하게 따른다 — `build_expert_prompt` 주석). 정의를 아예 빼면
이번엔 서술형 태스크가 "무엇을 하라는 것인지" 알 방법이 없어져
**관측 대상이 문체가 아니라 "누가 태스크를 잘 알아맞히나"가 된다.**
그래서 정보는 다 주되 자리만 바꾼다 — 정의를 system으로 올려 페르소나와 같은 층에 두고,
페르소나를 앞에 놓아 정체성이 먼저 읽히게 한다.

```
[system] You specialize in answering questions. ...          ← 로스터 페르소나(먼저)
         In this task, we ask you to elaborate the sentence   ← 태스크 정의
         without changing its general meaning. ...
[user]   Give the output only, with no explanation.           ← answer_line
         <Input 원문>
```

실측(`results/sni/prompts_preview.md`): v1은 system 90자 / user 537자+입력으로 정의가 지배했다.
v2는 system 267~328자(페르소나가 그 중 96~105자) / user 130~151자다.
페르소나와 정의가 같은 턴에서 경쟁하고, 미스매치가 눈에 보인다
("You specialize in answering questions" + "elaborate the sentence").

**answer_line 생성 규칙(기계적, LLM 미사용).** task 내 정규화 정답 종수 V, 인스턴스 수 M:

- `V <= 20 and V/M <= 0.2` → 닫힘: `Answer with exactly one of: {원표기 라벨, 정렬}.`
- 그 외(열림) → `Give the output only, with no explanation.`
  **길이·형식을 규정하지 않는다.** 길이를 지정하면 이 실험이 재려는 문체를 프롬프트가 먼저 죽인다.
- `output_language != English` → 위 줄 뒤에 `Answer in {language}.`

875 task 전수를 `results/sni/answer_lines.md`로 덤프해 **승인 후 확정**한다
(검수 포인트: 조작이 새어든 줄이 있는가).

조립된 프롬프트는 `--dry_run`으로 **서술형/객관식/비영어 각 1문제 × 로스터 전원**을
`results/sni/prompts_preview.md`에 덤프해 **승인 전 잡 제출 금지**.

## 4. 채점

**완화가 아니라 형식 제거.** `_sni_extract(pred, item)`을 `_sni_normalize` 앞에 두고
아래만 적용한다(사전등록):

1. 선두 열거표지 제거 — `^\(?\s*\d+\s*[\).:]\s*`, `^[A-Za-z]\s*[\).]\s*`
2. 감싼 따옴표·백틱 제거
3. 선두 `Answer:` / `Output:` 제거
4. **닫힌 태스크 한정** — 출력이 `answer_space` 라벨을 **정확히 하나** 토큰경계로 포함하면 그 라벨로 확정.
   둘 이상이면 0점. 여기서 더 넓히지 않는다.

검증 두 단계(둘 다 GPU 불필요):
- 기존 `results/sni/probe_raw.jsonl`의 생성물로 **규칙 전/후 EM을 재계산해 완화 폭을 숫자로 먼저 보고**.
- 점수가 바뀐 건 중 무작위 100건을 `results/sni/scorer_extract_audit.md`로 만들어 수동검수.
  오탐이 나오면 해당 규칙을 되돌린다.

점수 지표: **닫힘 = EM(+추출규칙)**, **열림 = ROUGE-L**(`score_sni_item_partial`, 구현 완료 ·
CJK/태국어 문자단위 분기 포함).

## 5. 판독 (실행 전 확정)

**두 계열을 분리해서 잰다.**

| 계열 | 지표 |
|---|---|
| 문체(레퍼런스 무관) | 출력 토큰길이 · 문장수 · 평균문장길이 · type-token ratio · 불릿/번호 유무 · 문제당 유니크 출력률 |
| 점수 | 닫힘 EM / 열림 ROUGE-L |

**주 분석 = 조건(luca / category 로스터 / domain 로스터) × 층 의 2차원.**
각 셀에서 문체·점수 각각에 대해 분산분해(문제 / expert / 상호작용, K=3 노이즈 보정 —
`evo_multisample_pilot.py`의 `analyze()` 재사용) + 홈 어드밴티지(`sni_probe_axis.py` 재사용).

층화 변수는 **SNI를 안 쓰더라도 다른 데이터셋에 옮겨지는 성질**로 잡는다 —
답 개방도(객관식↔서술형) · gold 길이대 · 레퍼런스 수 · 채점지표 종류(EM↔ROUGE).
SNI 고유 라벨(category/domain)은 보조 축으로만.

### 해석 순서 — 이 순서로만 읽는다

1. **로스터가 luca 대비 문체지표를 안 바꾼다** → 프롬프트가 출력에 닿지 않은 것.
   manipulation 실패이므로 **점수 null은 인용 금지**, 프롬프트 재설계.
2. **로스터 내 between-expert 분산 ≈ within-luca(K회) 분산** → 관측된 효과는 노이즈. 축 주장 금지.
3. 1~2를 통과했을 때만 점수를 읽는다. **서술형 층 / 객관식 층 / 서술형 내 결정론 층**
   (Program Execution·Text to Code)을 나란히 놓고 대비를 본다.
4. category 축과 domain 축을 **같은 N에서** 비교해 어느 쪽이 더 많은 분산을 설명하는지 읽는다.

## 6. 산출물

**`docs/REPORT_regime_selection.md`** — 이 프로브의 실제 값어치는 여기 남는다.

**(a) 레짐 선택표** · 행 = 층(객관식 / 중간 / 서술형 × gold 길이대), 열 =

| 지표 | 왜 보나 |
|---|---|
| 문체 갈림(문체지표 분산 중 expert 비중) | 프롬프트가 출력을 바꾸긴 하나 |
| 점수 갈림(expert 주효과 + expert×문제 상호작용, 노이즈 보정) | 그게 채점에 들어가나 |
| 둘의 상관 | 문체→점수 경로가 실재하나 |
| 신호밀도(만장일치 아닌 문제 비율) | 진화 WAR이 쓸 문제가 남나 |
| union 헤드룸 | 상보성이 있나 |

객관식 층은 사실상 acc·QASC의 재현이다. **서술형 층과의 대비 자체가 세 번 만난 null의 원인 규명**이고,
다음 진화를 어떤 성질의 데이터에서 돌릴지의 근거가 된다.

**(b) 진화 축 선정 근거** — category 축과 domain 축 중 어느 쪽이 분산을 더 설명했는가가
그대로 진화 로스터 축의 근거다. 인수인계 미해결 3번("스카우트에게 무엇을 찾으라 할 것인가")의 답을
**스카우트에게 묻지 않고 실측으로** 얻는다. 둘 다 설명 못 하면 그 사실이 스카우트 지시문을 다시 쓸 근거다.

## 7. 실행 순서와 게이트

| 단계 | 내용 | 게이트 |
|---|---|---|
| 0 | export 재빌드(정의 분리 + answer_line), 채점 추출규칙 | `answer_lines.md` 승인 · 완화 폭 보고 |
| 1 | 프롬프트 배선 + `--dry_run` 덤프 | `prompts_preview.md` 승인 |
| 2 | 로스터 v2(N 확정) | N 승인 |
| 3 | ~~스모크~~ | 사용자 지시로 생략, 본잡 직행 |
| 4 | 본 런(전수 × 25명 × K3) | job 229520 **FAILED**(컨텍스트 초과) → **job 229893 재제출** |
| 5 | 판독 → `REPORT_regime_selection.md` | §5 해석 순서 준수 |

자원 헤더는 [run_sni_probe.sh](../scripts/sbatch/run_sni_probe.sh) 현행 유지(임의로 낮추지 않음).
생성 예산은 `configs/sni_probe_v2.yaml`에서 `max_tokens` 8192 → **4096**(전수 gold 최대 2,147토큰 ·
p99 97 · 중앙값 2라 잘리는 건 0건). 나머지 llm/vllm 블록은 v1 프로브와 같은 레짐을 유지하려고
`acc_train_seed20210101.yaml` 구조를 그대로 복제했다(shallow merge 주의).
