# 인수인계 — SNI (2026-08-29)

브랜치 `jh/sni-probe`. **커밋 안 함.** 워킹트리에 변경분과 대량 삭제분이 그대로 있다.

---

## 1. 지금 돌고 있는 것

| job | 내용 | 상태 |
|---|---|---|
| **236193** | seed20212003 최종 로스터 **test 평가** (8,699문제 × 16명 × K=3 ≈ 417k 생성) | RUNNING, ETA 약 2시간 |

산출: `results/sni/binning_seed20212003/test.md` + `test_raw.jsonl`.
per-expert pass@1 · union UB · 커버리지가 나온다. **routed top-1은 안 나온다**(SNI는 라우팅 평가 배선 없음).

⚠️ 이 잡은 `run_sni_binning_v4.sh`를 쓴다. 이번에 `SPLITS` 환경변수를 인자화했다
(미지정이면 기존 동작 train/valid/test 그대로). train 라벨링은 69,588 × 16 × 3 = 3.3M 생성이라 안 걸었다.

---

## 2. 완주한 진화 3판

| seed | job | 설정 | UB(마지막 200스텝) | 최종 로스터 |
|---|---|---|---:|---:|
| 20212001 | 235687 | hard WAR | 73.48 | 9명 |
| 20212002 | 235861 | hard WAR + `scout_exclude_binary`(A만) | 72.10 | 9명 |
| **20212003** | **236040** | **soft_linear + 이진 A/B/C 전면 차단** | **73.54** | **16명** |

### seed20212003이 무엇을 바꿨나

**(1) `war_mode: soft_linear`** — ⚠️ **20212001·2002는 이 키가 없어서 기본값 `hard`로 떨어져 있었다**
(`run_evolution.py`: `cfg.get("war_mode", "hard")`). 두 판 모두 적합도가 "정확히 나만 푼 문제 수"였다.
soft_linear는 내가 푼 문제마다 `(E−n)/(E−1)`점 — E=로스터 총원, n=그 문제를 푼 사람 수.
E=9에서 4명이 풀면 각자 0.625, 혼자면 1.0, 전원이면 0.0.

**(2) `exclude_binary: true`** — 이진(`n_gold_types == 2`)을 **세 경로 전부**에서 제외.
v5(20212002)는 A만 막아 실패했다. "반대로 답하라"는 아무 하드에러에서나 나오는 일반 전략이라,
정보만 막고 보상 경로를 열어두면 재발명된다.

| | 지점 | 코드 |
|---|---|---|
| A | 스카우트 입력 | `orchestrator.py` scout_errors 필터 |
| B | 승인 게이트 | `probe_hard` / `worst_unique_ids` / `select_action`의 solve 행렬 |
| C | 적합도 | `compute_war_scores(score_ids=...)` · lives · 삭제 · `exclusive_solves` 기록 |

**UB(`upper_bound_rate`)는 배치 전체 기준을 유지**했다 — 이전 시드와 비교할 보고 수치라 정의를 안 바꿨다.

### ⚠️ 로스터 16명은 수렴한 크기가 아니다

`lives_mode`가 `legacy`라 윈도 경로에서 `rate > deletion_floor(0)`를 쓰는데, soft_linear는 문제를
하나라도 풀면 점수가 붙어 rate가 0이 되지 않는다. **전원이 매 스텝 목숨 +1**을 받아 아무도 0에 도달 못 했다.

- 결정 분포 **add 15 · noop 1,377 · delete 0 · swap 0**, 1,392스텝 전부 "축출 후보 없음", 최종 lives 전원 5/5
- 사실상 **add-only 런**. 이전 두 판(9명)과 로스터 크기 체제가 다르다 — UB 비교(73.54 vs 73.48 vs 72.10)에 이 차이가 섞여 있다
- 원인: config 작성 시 seed18(hard WAR) 게이트 설정을 그대로 가져오고 `lives_mode`를 안 바꿨다.
  acc의 soft 런들(`acc_train_seed20211002/1004`)은 `lives_mode: rank` / `rank_windowed`를 쓰고
  "deletion_floor는 rank 경로에서 미사용"이라고 주석까지 달려 있다. **다음 soft 런은 여기부터 확인할 것.**

---

## 3. seed20212003 최종 로스터 (16명)

`results/sni/seed20212003/roster_final.json`

### 최소화·축자 계열 (상위 10)

| avg_war | 스텝 | 이름 | system prompt |
|---:|---:|---|---|
| 3.611 | 482 | Precision-Targeted Span Minimization | Identify the absolute minimal sequence of characters required to satisfy the truth conditions of the query. Avoid any linguistic padding, grammatical completeness, or contextual elaboration. |
| 3.424 | 1094 | Extrapolative Hallucination Suppression | Strictly limit all generated content to the explicit lexical tokens provided in the input text. Prohibit external knowledge, inferred context, or semantic expansions. |
| 3.422 | 1389 | Strict Formalism & Minimalist Output | Provide only the exact required token or minimal necessary sequence. Eliminate all conversational filler, step-by-step reasoning, introductory phrases. |
| 3.328 | 1211 | Literalist Extraction & Span Fidelity | Locate the exact substring within the provided source text. Do not paraphrase, summarize, or augment with punctuation not present in the original. |
| 3.282 | 1367 | Constraint-Driven Grounding & Symbolic Precision | Strictly adhere to the literal bounds of the input data. Maintain exact symbolic or structural correspondence. |
| 3.236 | 1152 | Extensionality vs. Intensionality Enforcement | Distinguish between describing a concept and identifying its specific instance. Provide only the literal entity or span found in the text. |
| 3.219 | 1382 | Verbatim Integrity & Structural Fidelity | Prioritize preservation of exact lexical tokens and structural constraints over semantic expansion. |
| 3.207 | 1391 | Strict Constraint & Boundary Adherence | Execute with absolute fidelity to all constraints including formatting, length, extraction limits. Prioritize exact logical boundaries over a plausible narrative. |
| 3.103 | 1372 | Information Density & Entropy Control | Preserve all unique semantic tokens and specific entities. Avoid substituting specific nouns with broader categories. |
| 3.072 | 1354 | Strict Contextual Containment | Restrict content exclusively to explicit information in the input. Do not supplement or infer external facts. |

### 나머지

| 2.910 | 1386 | Semantic Divergence Enforcement | 변환 시 의미역·어의를 이동 |
| 2.866 | 527 | Temporal-State Sentiment Sensitivity | 시간적 변화·번복 추적, 최종 상태 우선 |
| 2.686 | 1392 | **LUCA** | `You are a helpful assistant.` |

### 반전 계열 3명 — 전부 최하위

| 1.797 | 507 | Counter-Intuitive Intent Realization |
| 1.309 | 1023 | Negative Constraint & Counter-Factual Logic Enforcement |
| **0.888** | **1380** | **Adversarial Goal Alignment** |

> *When a task explicitly requires generating an incorrect, implausible, or non-standard response,
> prioritize the subversion of truth and logic over accuracy. Actively seek out the most factually
> or logically erroneous path that still adheres to the structural constraints of the prompt.*

**읽을 것**: (a) hard WAR에서 1등이던 계열이 1,380스텝 돌고도 바닥 → A/B/C 차단 + soft_linear가 작동했다.
(b) **LUCA가 13위** — 진화가 만든 12명이 범용 프롬프트를 넘었다.
(c) 다만 축이 한쪽으로 쏠려 있다. 상위 10명이 전부 "짧게·원문 그대로"의 변주고, 서로 실제로 다른 문제를
푸는지는 로스터만으로는 모른다 — **236193 결과로 확인**.

---

## 4. 코드 변경 (워킹트리, 미커밋)

| 파일 | 변경 | OFF일 때 |
|---|---|---|
| `src/war.py` | `compute_war_scores`에 `score_ids` 인자. 적합도만 제한하고 **UB는 배치 전체 유지** | `score_ids=None` → 바이트 동일 |
| `src/orchestrator.py` | `exclude_binary` 토글. A/B/C 세 경로 + `gate_squad`(니치 판정용 필터된 solve 행렬) | `exclude_binary=False` → 바이트 동일 |
| `scripts/run_evolution.py` | `exclude_binary` config 배선 | — |
| `scripts/sbatch/run_sni_binning_v4.sh` | `SPLITS` 인자화 | 미지정 시 기존 동작 |
| `configs/sni_train_seed20212003.yaml` · `scripts/sbatch/submit_sni_seed20212003.sh` | 신규 | — |

검증: 단위 대조로 soft_linear 배점(4명→0.625 / 혼자→1.0 / 전원→0.0), 이진 제외 시 해당 단독해결이
사라지는 것, UB 불변, `score_ids=None`에서 hard·soft_linear 모두 기존과 일치 확인. 기존 회귀
스크립트 `scripts/verify_soft_war.py` 통과.

---

## 5. ⚠️ 대량 삭제 — 사용자 지시

이전 세션들의 **SNI 축 분석 일체를 사용자 지시로 삭제**했다. 되살리지 말 것.

**이유**: 분석의 기반이던 "출력길이층 6라벨"이 **에이전트가 발명한 축**이었다.
`stratum()`이 `gold_len_median`을 **3 / 8 / 15 / 40 단어**에서 잘랐는데, 그 절단점은 데이터에도
공식 레포에도 없고 사용자가 정한 것도 아니다. 그 위에서 나온 **"6라벨이 task 853라벨의 78%를 설명"**은
축의 성질이 아니라 절단점의 함수였다(연속량을 몇 개로 자를지가 자유 파라미터).
그 결론이 "진화가 잡은 구조는 출력 형식·길이 레짐"이 되어 후속 분석의 전제가 됐다.

삭제된 것: `sni_axis_test` / `sni_evolved_axis` / `sni_roster_compare` / `sni_track_ub` /
`sni_hard_error_rate` / `sni_anova` / `sni_length_vs_score` / `sni_exclusive_by_axis` /
`sni_probe_v2_readout` / `sni_interaction_share` / `sni_forensic_v5` / `sni_why_wrong_wins` /
`sni_emit_labels` / `sni_official_rescore` / `sni_labels` / `sni_probe_axis` / `sni_probe_sample` /
`sni_rescore_audit` + sbatch 런처 18개 + `docs/sni_official/` 전체 + `REPORT_axis_*` ·
`REPORT_track_ub` · `REPORT_regime_selection` · `REPORT_for_collab_sni_axis` ·
`SUMMARY_sni_official_and_evolution` · `RECORD_sni_seed20212001` · `HANDOVER_sni_official/probe` ·
`PLAN_sni_probe_v2` · `REFLECTION_sni_probe` 등.

**살아있는 자산**: `results/sni/official_labels.npz`(+`_index.json`) — 프로브 v2 25명 × 87,028문제 × K=3의
per-item 공식 EM·ROUGE-L·출력 단어수. 공식 `evaluation.py`와 203,972건 대조해 불일치 0을 확인한 라벨이다.
**단 그걸 만든 스크립트는 삭제됐다.** 진화 로그·로스터 스냅샷·그림 2개도 그대로 있다.

**다시 분석한다면**: 연속량은 자르지 말고 연속으로 쓰거나, 데이터에 실재하는 라벨
(`category` / `sni_domain` / `task_closed` / `task_name`)만 쓸 것.

---

## 6. 미결

1. **`lives_mode`** — soft WAR에서 삭제가 안 걸린다(§2). 다음 진화 전에 `rank_windowed`로 갈지 결정 필요.
2. **clone-adjusted gate** — 외부 피드백으로 나온 안. 게이트가 후보를 "전원 실패 문제"에 풀려 승인하는데,
   같은 (expert,문제) 쌍의 54.5%가 시도마다 갈리므로 "새 능력"과 "주사위 한 번 더"를 구분 못 한다.
   기존 멤버 clone을 같은 문제에 한 번 더 돌려 `gain(후보) − gain(clone)`을 승인 기준으로 삼는 안.
   비용은 게이트 프로브(배치당 13~23문제)뿐이라 전체의 5% 미만. **미구현.**
   미결: 재현성 조건을 어디에 걸지(신규 진입자 유예 기간 제거 여부).
3. **2×2 example ablation** (코딩 도메인) — ①공통프롬프트+예시없음(union 24.23) ·
   ④전문가프롬프트+자기예시(30.09)만 있고 **②③이 없어** 원인이 안 갈린다. **미착수.**
4. **train 라벨링** — 로스터 확정 후 판단(3.3M 생성).

---

## 7. 이번 세션에서 사고 난 것

- **첫 턴에 두 판이 어떤 적합도로 돌았는지 확인 안 하고 상태 보고**를 했다. `war_mode` 미설정을
  사용자가 직접 묻고서야 찾았다. 진화 결과를 보고할 땐 **config 실효값부터 대조**할 것.
- **발명한 축 위에 분석을 쌓았다**(§5). 새 축을 만들기 전에 그게 데이터에 실재하는지 확인할 것.
- **배정(routing) 프레임으로 분석을 만들었는데 진화에는 배정이 없다.** 진화는 전원이 모든 문제를
  풀고 WAR로 로스터만 바꾼다. 분석 대상을 정하기 전에 그 대상이 진화와 연결되는지 확인할 것.
- **best-single을 test에서 골라 배정과 비교**했다. 배정을 train에서 정했으면 대조군도 train에서.
- **반복 노이즈를 안 재고 "차이가 작다"고 단정**했다. K=3으로 같은 절차를 3번 돌려 변동 폭을 함께 낼 것.
- 진화가 끝났는데 **최종 로스터를 프롬프트까지 보고하지 않았다.** 로스터 보고는 id가 아니라 이름 +
  system prompt 전문이다.
