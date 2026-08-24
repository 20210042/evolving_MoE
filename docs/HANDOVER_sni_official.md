# HANDOVER — SNI 공식 채점·프롬프트 전환 (2026-08-24, 브랜치 `jh/sni-probe`)

선행: [REFLECTION_sni_probe.md](REFLECTION_sni_probe.md) · [PLAN_sni_probe_v2.md](PLAN_sni_probe_v2.md)
커밋: `615d420` (38파일). **push 안 함.**

---

## ⚠️ 이 세션의 결론: 지금까지의 SNI 수치는 전부 재계산 대상이다

프로브 v2(job 229893, 87,028문제 × 25명 × K3)는 **정상적으로 완주**했고 생성물은 유효하다
(`results/sni/probe_v2_raw.jsonl`, 652만 행). 무효인 것은 **채점**이다.

세션 후반에 사용자 지적으로 공식 레포를 읽어보니, 내가 만든 채점기가 공식과 전혀 달랐다.

| | 내가 만든 것 (폐기) | 공식 (`eval/automatic/evaluation.py`) |
|---|---|---|
| 태스크 분기 | `task_closed`면 EM, 아니면 ROUGE | **분기 없음.** 전부 EM·ROUGE 둘 다 계산 |
| 정규화 | 비단어문자 → 공백 (`choice/control`→`choice control`) | 구두점 **삭제** (`choicecontrol`) |
| ROUGE | 직접 구현 LCS, 스테밍 없음 | `rouge_score` + **use_stemmer=True** |
| 복수 정답 | ROUGE만 max | EM·ROUGE **각각** max |
| 형식 완화 | `_sni_extract`(열거표지 제거·라벨 확정) | 없음 |

`task_closed` 분기의 실해: 산술 태스크(`task096_conala_list_index_subtraction`)가 "정답이
인스턴스마다 다름"이라 열림으로 분류돼 ROUGE로 채점됐고, **계산 오답 `[17,-17,...]`이 0.500점**을 받았다.

### 실제 영향 (job 231623 결과, `REPORT_official_rescore.md`)

**이식 검증: 공식과 완전 일치(True).** 8개 케이스 EM·ROUGE-L이 소수점까지 동일.

**전량 재채점 EM 차이는 −0.7%p뿐이다.**

| 층 | 이전(내 기준) | 공식 EM | 차이 | 공식 ROUGE-L |
|---|---:|---:|---:|---:|
| 닫힌 라벨집합 | 72.4% | 71.5% | −0.8%p | 73.3 |
| gold 1-3단어 | 48.5% | 47.5% | −0.9%p | 57.0 |
| gold 3-8단어 | 30.5% | 31.0% | +0.4%p | 54.9 |
| gold 8-15단어 | 12.9% | 12.8% | −0.1%p | 44.4 |
| gold 15-40단어 | 17.7% | 16.4% | −1.3%p | 52.3 |
| gold 40단어+ | 0.3% | 0.3% | ±0.0%p | 19.9 |
| **전체** | 51.9% | **51.2%** | **−0.7%p** | **62.8** |

→ **EM 기반 결론은 살아 있다.** ANOVA 상호작용 29.6%, hard error 비율, 축 검정의 닫힌 층은
1%p 수준의 이동이라 부호·순위가 뒤집힐 규모가 아니다. 그래도 **숫자는 공식으로 다시 찍어야 한다.**

⚠️ **ROUGE 기반 결론은 다르다.** 내 ROUGE는 스테밍이 없었고 공식은 `use_stemmer=True`다.
공식 ROUGE-L은 전 층에서 EM보다 훨씬 높다(서술형 15-40단어 EM 16.4% vs ROUGE 52.3).
**서술형을 ROUGE로 읽은 축 검정(태스크 축 Δ+4.03%p 등)은 값이 달라질 수 있다 — 재계산 전 인용 금지.**

재계산 대상 문서: `REPORT_axis_test.md` · `REPORT_axis_anova.md` · `REPORT_interaction_share.md` ·
`REPORT_track_ub.md` · `REPORT_hard_error_rate.md` · `REPORT_regime_selection.md` ·
동료 공유용 `REPORT_for_collab_sni_axis.md`(Artifact 발행됨 — **재계산 전 배포 금지**)

---

## 바꾼 것

### 1. 채점 → 공식 이식 (`src/evaluation/scorer.py`)

- `sni_metrics(item, pred)` → `{"exact_match", "rougeL"}` 둘 다, 각각 레퍼런스 max
- `_sni_normalize` = 공식 `normalize_answer` (소문자 + `string.punctuation` 삭제 + 공백정리, 관사 유지)
- `score_sni_item` = 공식 EM / `score_sni_item_partial` = 공식 ROUGE-L
- **삭제**: `task_closed` 채점 분기 · `_sni_extract` · `_sni_token_contains` · 직접구현 `_rouge_l`/`_lcs_len`/`_sni_tokens` · `_SNI_NOSPACE_RE` CJK 분기
- `scripts/sni_rescore_audit.py`는 폐기 처리(삭제된 규칙의 감사 도구였음)

⚠️ **진화용 이진 판정("풀었다/못 풀었다")은 공식에 없는 우리 요구사항**이다. 지금은 EM을 쓴다.
ROUGE 임계로 판정하려면 `war_mode='soft_partial'` 경로. 임계값은 **미결**.

### 2. 프롬프트 → 공식 Tk-Instruct 형식 (`src/prompts/coding.py`)

공식 표준(`yizhongw/Tk-Instruct` `scripts/{train,eval,gpt3}_tk_instruct.sh`, 셋 다 동일):
```
--add_task_name False   --add_task_definition True
--num_pos_examples 2    --num_neg_examples 0    --add_explanation False
--max_source_length 1024   --max_target_length 128
```
- `sni_user_block` → 예시 블록 + `Now complete the following example -` + `Input:` / `Output: `,
  구두점 보정까지 `ni_collator.py` 그대로
- `sni_system_block` → 페르소나 + 정의 (정의를 user에 두면 페르소나가 묻힌다 — v1 무효 원인)
- 빌더 4곳 시그니처에 `positive_examples`/`negative_examples`/`num_pos_examples=2` 등 추가,
  호출부 7곳(pilot·orchestrator×3·baselines×3) 배선
- `configs/sni_probe_v2.yaml`: **`max_tokens: 128`** (공식 `max_target_length`와 동일).
  전수 gold 토큰 중앙값 2 · p99 97 · 최대 2,147 → 99%+ 커버, 나머지는 공식과 마찬가지로 잘림

### 3. export v3 (`scripts/build_sni_export.py`)

`Positive Examples` / `Negative Examples`(+explanation) 보존. v2까지 이걸 버렸던 근거
("공식 기본이 zero-shot")는 **거짓**이었다. → `export/sni_v3` (job 231627로 생성 완료)

### 4. 스카우트 → verbal RL (`src/prompts/meta.py`, `src/scout.py`, `src/roster.py`)

- 축 힌트 제거. "무슨 조작이 빠졌나"를 묻지 않고 실패 케이스만 주고
  `Whatever these failures have in common, write one new system prompt...`
- 출력 스키마: `prompt_name` / `system_prompt` / `fixes`.
  **`fixes`가 "모델이 어떤 축을 봤는가"의 기록**이고 파일럿의 관측 대상이다
- `roster.normalize_persona_fields`가 `prompt_name`→`name` 승격(기존 `persona_name`도 유지)
- `orchestrator._persona_label()` 폴백 3곳
- SNI `hard_errors`에 **정의·기대·실제 출력** 추가 — 없으면 스카우트가 Input만 보고 주제로 분화
- 스카우트 캡 4,000 → **40,000** (failure_mode와 같은 예산; 케이스당 ~670자 → 약 55~60건)
- 로스터 표: `strengths`가 없으면 `name | system prompt`로 표시

---

## 환경 변경

**공유 env `evolving_moe`에 설치함** (사용자 승인, `--no-deps`로 기존 버전 무변경):
`rouge-score 0.1.2` · `nltk 3.10.3` · `absl-py 2.5.0` · `defusedxml 0.7.1`

---

## 다음 세션이 할 일

1. ~~job 231623 결과 확인~~ **완료** — 공식과 완전 일치, EM 차이 −0.7%p (위 표).
2. **공식 기준으로 전 분석 재실행.** 스크립트는 그대로 쓸 수 있다(채점 함수만 바뀜):
   `sni_axis_test.py` · `sni_anova.py` · `sni_interaction_share.py` · `sni_track_ub.py` · `sni_hard_error_rate.py`
   ⚠️ 재생성 불필요 — `probe_v2_raw.jsonl`의 생성물로 전부 재계산된다.
3. **프롬프트가 바뀌었으므로 프로브 자체를 다시 돌릴지 결정.** 공식 형식(pos example 2건 +
   `Output: ` 구분자 + max_tokens 128)은 v2 런과 다른 프롬프트다. 재계산만으로는 이 차이가 안 잡힌다.
4. **미결 결정 넷**:
   - `answer_line`(내가 만든 줄)을 남길지 뺄지 — 공식엔 없다
   - 진화 이진 판정 임계 (EM 그대로 vs ROUGE 임계)
   - train/valid/test 분할 — **현재 87,028은 공식 train+test를 합친 전체이고 홀드아웃이 0이다.**
     공식 split은 category-disjoint. `run_evolution.py`는 `train_size` 미만이면 나머지를 자동 홀드아웃
   - batch_size (hard error 실측상 100이 적정했으나 그것도 내 채점 기준 수치)
5. **진화 파일럿.** 처리량 실측 68.4 gen/s → 20,000문제 × 로스터 8~12 = **1~3시간**,
   전수 87,028 × 12 = 4.24시간(생성만). 계산은 병목이 아니다.

## 커밋 상태

`615d420`에 38파일. **`src/orchestrator.py`·`src/evaluation/scorer.py`·`src/prompts/coding.py`에는
이전 세션의 acc seed1004 작업이 섞여 있어 분리 커밋이 불가했고, 그대로 함께 들어갔다.**
커밋 메시지에 명시했다. push 없음. 원격 브랜치 없음.

미커밋으로 남은 것: acc 계열 파일 다수(이전 세션 산물), `export/sni_v3`·`results/sni/*`(gitignore).
