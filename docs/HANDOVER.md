# MetaAgentEvolution — 인수인계 문서

작성일: 2026-06-08 (갱신: 2026-06-16 — MC 채점 아티팩트 수정, eviction/hole-aware 진단 정정, scout V1/V2/V3 정리, 2×2 설계)

---

## 0. 2026-06-16 작업 요약 (채점 아티팩트 + scout 버전 + 2×2 설계)

> 이번 세션의 핵심: ① numina 채점에 큰 아티팩트 발견·수정 ② hole-aware/eviction에 대한 이전 서술 **정정** ③ scout 프롬프트 V1/V2/V3 정체 규명 ④ 깨끗한 2×2 실험 설계 착수.

### 🔴 발견 1 — MC 채점 아티팩트 (+~11pp), 수정 완료
- numina test의 **객관식 30%**(1862/6215)에서 **gold 포맷이 보기문자("B")와 값("a+b=3")으로 섞여** 있음(보기문자 806 / 값 1015). 모델이 한 포맷으로 박싱하면 **절반은 실력과 무관하게 FAIL**.
- 규모: seed12 eval **67.4→78.0% (+10.6pp)**, 회복 53건 **전수 감사 정당(FP 0)**. 전 에포크 +10.6~11.6pp 일관. (이전에 말한 "+6pp"는 과소집계였음.)
- **수정(추가 전용, 비객관식 byte-identical)**:
  - 우리: `scorer.py` score_one에 MC-aware 분기 — commit **`6f4ac55`** (jh/evolution).
  - 협업자: `metrics.py` `mc_aware_math_score` + `evaluate.py` `math_verify_mc_score` 메트릭 — commit **`939f670`** (fix/math-verify-scorer, **origin push**).
- ⚠️ **함의**: 기존 numina 수치(seed10~13, ~67%)는 ~11pp 눌려 있던 것. 백본 비교(MoE) 시 박싱 성향 차로 오염 위험 → 비교 전 MC-aware 필수.

### 🟧 정정 — eviction / hole-aware (이전 문서 오류 바로잡음)
- **적용 gate값 = scale 0.25, λ 0.05** (base.yaml; numina config가 override 안 함). 실측 로그(seed12 add 스텝 역산)로 scale=0.25 확정.
- 이 regime: **worst는 고유 solve=0일 때만 evict** → **hole-aware demote는 구조적으로 발동 불가(INERT).** 이전 문서의 "niche ≤1", "λ 낮추면 load-bearing", "seed13→hole-aware load-bearing"은 **전부 틀림**. (방향 반대: 발동시키려면 **scale↑**.)
- **eviction이 안 돌던 진짜 원인 = shared 기여 면제(`356813b`, seed05).** 데이터로 분리:

  | seed | max_lives | all-zero 면제 | shared 면제 | 후보 | del+swap |
  |---|---|---|---|---|---|
  | 02/03 | 5 | ✗ | ✗ | 7/2 | 7/2 |
  | 04 | 5 | ✓ | ✗ | 3 | 3 |
  | **05~12** | 5 | ✓ | ✓ | **0** | **0** |
  | **13** | 5 | ✓ | **✗(OFF)** | **6** | **6** |
  → **max_lives 무죄, shared 면제가 단독 킬러.** seed13(shared OFF)에서 도태 부활(로스터 5명).

### seed12 vs seed13 (MC-aware, 둘 다 V2 scout)
| | seed12 (shared ON, 7명) | seed13 (shared OFF, 5명) |
|---|---|---|
| MoE 평균 | ~78.4% | ~77.3% |
| LUCA 단독 | 78.2%(노이즈 기준) | 77.6% |
| UB union | **83.4%** (+26) | 81.4% (+19) |
→ 도태(13)는 lean하지만 **UB ~1.4pp 손해**(노이즈 보정 후). MoE 동률. **큰 로스터가 커버리지 유리.**

### 🟦 발견 2 — scout 프롬프트 V1/V2/V3 정체 (왜 "&" 범벅인가)
- numina(seed10/12/13)는 **V2** scout 사용. **V2엔 atomicity 규칙도 'and' 금지도 없음** → `&` 묶음 전문가가 무제약으로 나옴.
- 버전 정체: **V1**(atomicity 有, 단일도메인) / **V2**(규칙 제거, exclusive_solves, "&"묶음) / **V3**(approach-persona, 정체성1문장+절차; 이름만 깔끔, 속은 여전히 묶음).
- 어떤 버전도 **도메인명 prior는 안 줌**(도메인은 모델이 hard_errors 보고 선택). 차이는 **구조 prior(atomicity)뿐**.
- ablation 종합: **BigMath V1(seed08) > V2(seed09)** (UB +20 vs +14). **numina V2(seed10) > V3(seed11)** (MoE 67.6 vs 63.9, V3 반증). **단 V1은 numina에서 미검증** → 이번에 보충.

### 🆕 2×2 설계 착수 — scout(V1/V2) × shared(ON/OFF)
| | shared ON | shared OFF |
|---|---|---|
| **V2** | seed12 ✅ | seed13 ✅ |
| **V1** | **seed14** (183264, R) | **seed15** (183267, R) |
- seed14/15 = numina **V1**(atomicity), 백본 31B(MoE 아님), Thinking OFF, max_lives 5. config `numina_train_seed14/15.yaml`, submit 동명. (use_exclusive_solves 없음 → V1 라우팅.)
- 검증포인트: 첫 스텝 로그에서 V1(atomicity) 적용 + persona '&' 안 붙는지.

### 🟩 별개 트랙 — MoE 백본(gemma-4-26B-A4B-it)
- 다운로드(49GB) + **load-smoke 통과**: vLLM 0.21.0이 `Gemma4ForConditionalGeneration` 로드·생성 OK(48.5GB/tp=1), 출력 31B와 동일 깨끗(`\boxed`). 백본 기술 적합성 확인. (정확도는 별도 eval 필요. seed14/15와 무관.)

### 커밋 (이번 세션, jh/evolution 별도 명시 없으면 그 브랜치)
`754ea40` hole-aware swap · `e4cc9e2` shared 면제 토글+seed13 · `6f4ac55` MC 채점기 · (fix/math-verify-scorer) `939f670` MC eval(push).

---

## 0-A. 2026-06-15 작업 요약 (hole-aware swap — 교수님 피드백 #3)

### 교수님 피드백 3불렛
1. **(후순위)** init_roster의 generalist 여러 시작점을 비교 → 서로 다른 출발점이 결국 같은(LLM이 선호하는) 방향으로 수렴함을 보이면 좋겠다.
2. **(현행 동의)** add/delete를 2-phase로 한 큐에 정하는 설계는 말이 됨(= delete를 독립 동작으로 떼지 않는 게 맞음) → **변경 없음**.
3. **(최우선)** swap 시 worst의 marginal loss로 생긴 "빵꾸"(worst만 풀던 niche)를 newface가 채울 수 있는지(worst∩newface 교집합)를 고려해야 함.

### 🔴 결함 (코드로 확인) — gate가 swap의 niche 회수를 측정조차 안 함
- `hard_errors` = 로스터 전원(worst 포함)이 못 푼 문제. newface는 오직 `hard_errors`로만 probe됨 → `new_pass ⊆ hard_errors`.
- `worst_unique`(worst만 푼 문제)는 worst가 **푼** 것이라 `hard_errors`와 **정의상 서로소** → `new_pass ∩ worst_unique = ∅` **항상**.
- ∴ 기존 `select_action`은 swap이 일어나도 newface가 worst의 빵꾸를 메우는지 **테스트하지 않음**. swap은 독립 두 phase가 우연히 동시에 켜진 것일 뿐.

### ✅ 수정 (commit `754ea40`) — hole-aware swap + add 강등
- **orchestrator**: probe 집합을 `hard_errors ∪ worst_unique`로 확장(newface를 worst niche에도 테스트).
- **action_selector**: `recovered = new_pass ∩ worst_unique`. swap 후보(phase1∧phase2)일 때 **niche 전량 회수면 swap, 아니면 add로 강등**(worst 유지 → niche 절대 안 잃음). `ActionDecision`에 `worst_unique_count/recovered_count/demoted_swap`, gate 로그·evolution_log에 기록.
- **2-phase 독립 보존**: `gh_add`는 여전히 `hard_errors`만 intersect → phase1(add) 판정 불변. niche-recovery는 swap에만 거는 보수적 veto(add 차단 안 함, delete 유발 안 함).

### ⚠️ magnitude — **(아래 06-16에서 정정됨)**
~~λ를 낮추면 load-bearing~~ → **틀림.** 실측 적용값은 `action_gate.scale=0.25, λ=0.05`(base.yaml). 이 regime에선 lambda_del≈0.013~0.018이라 **worst는 고유 solve = 0일 때만 evict** → 보호 niche = **0**(≤1 아님) → **hole-aware demote는 구조적으로 발동 불가(INERT).** 발동시키려면 **scale을 올려야**(0.5→unique≤1, 1.0→≤2~3) — 낮추는 게 아님. 자세히는 06-16 참조.

### seed12 = seed10 + "로직만" (깨끗한 A/B) — 결과
- hole-aware swap은 config 플래그 없이 코드 전역 적용 → seed10과 하이퍼파라미터 100% 동일한 `seed20210012`. config `numina_train_seed12.yaml`, submit `submit_numina_seed20210012.sh`. (jobs 182646/7/8)

| 지표 | **seed12** | seed10(참고) |
|---|---|---|
| MoE Ep1–5 | 67.6/67.0/66.2/68.8/67.4 | 68.6/67.0/67.0/67.4/68.2 |
| MoE 평균/최고/최종 | 67.4 / 68.8 / 67.4 | 67.6 / 68.6 / 68.2 |
| LUCA 단독 | 67.6 | 66.0 |
| UB union | 75.8% (7명) | 74.8% (8명) |
| specialist가 LUCA 너머 | +41 | +44 |

- **⚠️ seed10↔12 절대비교 위험 — 평가셋이 다름**: eval/UB가 `--seed`로 held-out 500을 뽑아 seed10/12가 서로 다른 500문제. 증거: **동일 LUCA인데 단독 66.0 vs 67.6**(+1.6pp 순수 테스트셋 노이즈). → MoE(−0.2)·UB(+1.0) 차이는 전부 노이즈 밴드 안 = **사실상 동률.**
- **그리고 그게 당연 — hole-aware가 0번 발동**: seed12 evolution_log **후보 0/30, swap 0, demote 0**(add 6/noop 24, 로스터 2→7). 로직은 휴면, seed12 ≈ seed10 재추첨. → "현 shared 면제 체제에선 gate 변경이 무력"이 수치로 재확인.

### 🔑 후속 진단: eviction이 안 도는 진짜 원인 = shared 기여 면제 (데이터로 분리)
- seed10/12 모두 **worst 후보 0건** → `pick_worst`가 항상 None → delete/swap 계산 자체가 시작 안 됨(로스터 단조증가).
- git+로그 교차분석으로 범인 분리: **shared 면제(`356813b`, seed05)가 단독 킬러.** seed02/03(면제 없음, max_lives 5)은 도태 7·2건, **seed04(all-zero 면제만)도 max_lives 5에서 3건 정상 도태**, but **shared 면제가 추가된 seed05부터 seed12까지 후보 0건**. → **max_lives는 무죄, shared 면제가 WAR 신호를 단락**시킴(수학 고겹침에서 누구나 공유문제는 풀어 lives 영구만렙).

### seed13 = seed10 + shared 면제 OFF (도태 부활) — commit `e4cc9e2`, 제출됨(182813/4/5)
- **config 토글** `shared_contribution_exemption`(기본 True=기존동작 보존). seed13만 False → WAR=0 에이전트가 공유풀이해도 lives 감소(= 검증된 seed04 체제). all-zero 면제·max_lives 5 유지.
- **결과(06-16 확인)**: 도태 **부활**(후보 6/30, delete 5+swap 1, 로스터 2→5). 단 evict된 worst는 **전부 unique=0** → `swap_demoted_to_add` **0**(hole-aware 여전히 INERT — scale=0.25이라 niche 보유자는 애초에 evict 안 됨). 즉 ~~hole-aware load-bearing~~은 **틀린 기대**였음(scale↑가 별도로 필요).

### 검증 (hole-aware/토글)
- hole-aware **단위 6/6 PASS**, numina **smoke COMPLETED**(크래시·오분류 없음, niche 로그 정상). 토글 **lives 분기 시뮬 5/5 PASS**(기본 True byte-identical).

---

## 0-B. 2026-06-10 작업 요약 (NuminaMath 파일럿 + domain 분기 버그)

### NuminaMath 통합 (협업자 main merge)
- 협업자가 `Jongbin-kr/NuminaMath-CoT_filtered` 데이터셋 + loader/eval 코드를 main에 merge. dataset 키 = **`numina_cot`**, gold 필드 = `ground_truth`, gold가 일관 `\boxed{}`/숫자/객관식.
- main 머지 완료(`0b4a0cf`): 충돌 2파일 해소 — `MATH_GEN_USER`는 협업자 \boxed 버전 채택, loader는 합집합. 내 scorer 픽스(metrics.py)는 보존.

### approach-persona(V3) 진화로직 추가 (commit `397878d`)
- seed11용: scout가 `{persona_name, system_prompt(정체성 1줄), approach(방법)}` 생성(strengths 없음). 정체성→system, **approach→user 턴 주입**(Gemma가 user 지시를 더 잘 따름), 라우터는 identity+approach로 라우팅. 토글 `use_approach_persona`.

### 🔴 math/coding 분기 버그 발견·수정 (commit `a012e49`)
- **현상**: 1차 numina 파일럿(seed10/11)에서 모델이 수학을 안 풀고 **Python 코드**를 생성(\boxed 0/500). MoE ~7%로 무효.
- **원인**: 생성·scout이 **dataset 이름**(`ds in ("bigmath","math")`)으로 math/coding 분기 → `numina_cot`가 매칭 안 돼 **코딩 프롬프트**로 빠짐. scout도 코딩 meta-prompt 사용 → seed11 V3(approach) 전혀 안 돎(로스터 approach 0/9).
- **수정**: **domain 기반 분기**로 전환(7곳: coding.py 2함수, scout.py, orchestrator 3호출, routing 2호출, baselines 3호출, loader). `domain="math"` 1차 + dataset 폴백 집합(`_MATH_DATASETS`에 numina_cot 추가). 검증: numina+math→`\boxed` 프롬프트, seed11→V3(approach 파싱) 확인 완료.

### 파일럿 재실행 상태
- **1차 파일럿(코드 버그) = 무효, 폐기.**
- seed10(control, 현행 persona) / seed11(approach-cue) 재제출 — **job 178287 / 178290**, evolution+eval+UB 풀세트, tp=1, PRO6000:1.
- ⚠️ **현재 PD(GPU 대기)** — n03·n04 PRO6000 만석이라 미시작. 시작 시 **첫 스텝 출력에 \boxed·approach 나오는지 먼저 확인** 후 진행.
- config: `configs/numina_train_seed10/11.yaml` (numina_cot, enable_thinking false, exclusive_solves; 11은 use_approach_persona). submit: `scripts/sbatch/submit_numina_seed20210010/11.sh`. 범용 러너 `run_math_evolution.sh`(EVOL_CONFIG), eval/UB는 `DATASET` 파라미터화(기본 bigmath).

### 비교 설계 (10 vs 11)
둘 다 NuminaMath + 교정 scorer + exclusive_solves + Thinking OFF, **persona 설계만 다름**(현행 vs approach-cue) → "approach-cue가 효과 있나" 격리.

### 🟥 eval stale-reuse 함정 (수정됨)
1차 재실행 시 eval/UB가 옛 무효 런(코드출력)의 `inference_*.jsonl`을 재사용 — `run_inference`의 resume 로직이 "이미 500/500 처리됨"으로 **생성을 통째 skip**. 진화 산출물은 정상이었고 eval만 무효. **수정**: 옛 출력 삭제 + eval/UB 스크립트에 생성 전 `rm -f` 추가(commit `b374dea`). 재실행하니 \boxed 100%·정상 수치.

### 결과 (2026-06-10 완료, 깨끗한 데이터)
| 지표 | **seed10 (control, V2)** | **seed11 (approach-cue, V3)** |
|------|----|----|
| MoE Pass@1 평균 | **67.6%** (68.6/67.0/67.0/67.4/68.2) | 63.9% (63.6/64.2/63.6/63.4/64.8) |
| LUCA 단독 | 66.0% | 66.6% |
| UB union | **74.8%** (8명) | 72.8% (7명) |
| specialist가 LUCA 너머 | **+44** | +31 |
| 아무도 못 풂 | 25.2% | 27.2% |

**판정: approach-cue(11)는 효과 없음 — 오히려 해로움.** 모든 지표에서 control보다 낮고, 결정적으로 **seed11 MoE(63.9%)가 자기 LUCA(66.6%)보다도 낮음**(approach-persona 라우팅이 generic LUCA보다 못함). control은 LUCA +1.6. → 정체성+절차적 접근법을 user에 주입하면 전문가가 더 경직돼 성능 하락. **가설 반증.**

메타: control도 MoE 67.6 vs LUCA 66.0 = **+1.6pp뿐** — BigMath와 같은 패턴(prompt-level 전문화가 generic 대비 미미). caveat: 각 arm n=1, 절대수치는 협업자 vanilla baseline과 대조 필요. 협업자 전달용 로스터: `results/numina_cot/seed20210010/roster_final.json` (LUCA+7).

---

## 0-A. 2026-06-09 작업 요약 (채점기 버그 + 진화 재실행 + NuminaMath 준비)

> ⚠️ **결론 해석 보류**: 진화 WAR도 같은 버그 채점기로 계산됐으므로, 교정 채점기로 진화를 재실행(106/108)해 검증하기 전까지 "MoE가 baseline 못 넘는다 / math = 음성 대조군" 등 **해석은 확정하지 않는다.** 아래는 사실만.

### (A) 채점기 버그 — math_verify_score raw LaTeX false-negative
- **현상**: 모델 답이 gold와 글자까지 동일해도 오답 처리되는 케이스 대량. 재채점 시 모든 로스터·baseline이 **일괄 +16pp**.
- **원인**: `math_verify.parse()`는 `$...$`/`\boxed{}`로 감싼 LaTeX만 인식. raw LaTeX(`\frac{5\pi}{12}`, `\dfrac{1}{6}`)는 `[]`로 파싱돼 무조건 0.0.
- **조치**: [src/evaluation/metrics.py](src/evaluation/metrics.py) `math_verify_score` 수정 — `$`-래핑 + Latex/Expr extractor + 양방향 verify. 진짜 오답(13 vs 7, degree vs radian)은 그대로 0.
- **브랜치**: `fix/math-verify-scorer` (main 기반, 이 파일 하나만, origin 푸시됨) — 협업자/타인이 이것만 merge 가능. jh/evolution엔 cherry-pick(a310b01).

**재채점 결과 (OLD → FIXED, BigMath test 500):**
| | seed06 | seed08 | seed09 | raw baseline |
|---|---|---|---|---|
| MoE 평균 | 66.3→82.2 | 67.0→83.0 | 67.5→83.4 | — |
| UB union | 70.8→87.0 | 71.2→86.8 | 70.2→86.4 | — |
| raw 1-pass | | | | 68.0→**84.0** |
| 아무도 못 풂 | →13.0% | →13.2% | →13.6% | — |

### (B) 진화 재실행 (교정 WAR) — running
새 seed id로 기존 버그-WAR 런 보존하며 재실행. **106(=06,Thinking ON)·108(=08,Thinking OFF)** 풀세트(evolution+per-epoch eval+UB). 109(exclusive_solves) 보류.
- submit: `scripts/sbatch/submit_bigmath_seed2021010{6,8}.sh`, config `bigmath_train_think.yaml`(신규, Thinking ON 복원) / `bigmath_train_nothink.yaml`.
- 목적: 버그-WAR(06/08) vs 교정-WAR(106/108) 로스터·UB 비교 → 채점 버그가 진화 신호를 망쳤는지 검증.

### (C) math 생성 프롬프트 → `\boxed{}` (commit ebf97e6)
[src/prompts/baseline_prompts.py](src/prompts/baseline_prompts.py) `MATH_GEN_USER`를 `Final Answer:` → `\boxed{}`로. NuminaMath gold·math_verify 네이티브와 정합, 추출 안정, stray backtick 제거.

### (D) thinking ON/OFF — eval 출력으로 재확인
seed06(ON) eval 출력 평균 6,565자 vs seed08(OFF) 1,863자 (~3.5×). thinking은 작동하나 **정확도 이득 0** → thinking OFF 기본값 유지. 짭추론 arm은 우선순위 낮춤.

### (E) NuminaMath 파일럿 (협업자 협업)
협업자가 `Jongbin-kr/NuminaMath-CoT_filtered`(gold 전부 `\boxed{}`) + eval/training 코드 수정을 **오늘 저녁 main merge 예정**. 우리 쪽(프롬프트·진화로직) 독립이라 \boxed 준비 완료. merge 후 **300×5 한 arm** 파일럿 예정.

---

## 0-1. 2026-06-08 오후 작업 요약 (eval 방법론 수정)

기존 seed06↔seed08 ablation 비교가 **두 가지 방법론 오류**로 무효였음을 발견하고 수정·재실행 중.

1. **GMRoutingPipeline 생성 thinking 하드코딩 버그**: config `enable_thinking`은 진화(orchestrator)에만 적용되고 **eval inference에는 전달 안 됨** — `routing_inference.py`가 생성 단계에 `enable_thinking=True`를 하드코딩. 즉 seed08의 "Thinking OFF" eval이 실제로는 **생성 thinking ON**으로 측정됨.
   - **수정**: `GMRoutingPipeline`에 `gen_enable_thinking` 파라미터 추가, `run_inference.py`에서 `cfg.get("enable_thinking")`로 주입. 라우팅 단계는 원래 thinking OFF라 변경 없음. 기본값 True → 기존 Thinking ON 실험(seed06 등) 영향 없음.
2. **per-epoch vs single-final 불일치**: seed06은 **에포크별 5회**(roster_step_6/12/18/24/30) eval 후 best epoch(epoch3=67.4%) 보고. seed08은 **최종 roster 1회**(65.8%)만. 사과 대 오렌지. 실제로 seed06 **최종 에포크(epoch5)도 65.8%** = seed08과 동일.
3. **UB test-set 정렬**: ub_eval은 MoE eval과 **같은 `--seed`**로 동일 held-out 500을 써야 함. seed06 기존 ub_eval은 `--seed 0`(full-split 첫 500)이라 어긋나 있었음. seed08은 `--seed 20210008`로 바로잡음.

**조치**: seed07(177045) 취소 → GPU 확보 → seed08 재eval 2건 제출 (§2 참고). 완료되면 thinking 변수만 다른 controlled 비교(seed06 ON vs seed08 OFF) 가능.

**신규 스크립트**:
- `scripts/sbatch/run_bigmath_eval_epochs_nothink.sh` — per-epoch eval, `--config nothink.yaml`(thinking OFF + tp=2)
- `results/bigmath/seed20210008/ub_eval/run_all.sh` — per-agent UB, thinking OFF, `--seed 20210008`

---

## 1. 프로젝트 개요

BigMath 데이터셋에서 수학 전문 에이전트 로스터를 자동으로 진화시키는 시스템.  
매 스텝마다 배치 문제를 풀고, WAR 점수로 에이전트 기여도를 평가하고, Scout이 새 에이전트를 제안한다.

---

## 2. 현재 실험 상태

### 실행 중 (2026-06-08 오후)
| Job ID | 이름 | 설명 | 상태 |
|--------|------|------|------|
| 177558 | mae_eval_epochs_nothink | seed08 per-epoch eval (roster_step 6/12/18/24/30), **thinking OFF**, tp=2 | running |
| 177559 | ub_eval_seed8 | seed08 per-agent UB (9 roster), **thinking OFF**, `--seed 20210008` | running |

- **177045 (seed07, 50K)**: **취소됨** — GPU 확보 위해 scancel. 체크포인트(roster_final + evolution_log step45) 있어 `--resume`로 재개 가능.

### 완료된 실험 (수치 재검증 필요)
| Seed | Pass@1 (MoE) | Pass@1 (Luca) | UB mean | UB final | All-zero WAR |
|------|-------------|--------------|---------|----------|-------------|
| 20210006 | epoch별: 66.8/65.8/**67.4**/65.4/65.8 | ~64.8% | 70.1% | 74% | 30% |
| 20210008 | ~~65.80%~~ (thinking ON으로 측정됨, 재측정 중) | 64.80% | 재측정 중 | — | 43.3% |

- **Seed 20210006**: Thinking ON, 300×5, 기준선. eval은 epoch별(roster_step_6~30). 보고된 "67.4%"는 best epoch(epoch3)이며 final epoch5=65.8%.
- **Seed 20210008**: Thinking OFF (진화 단계), 300×5 — ablation. ⚠️ **기존 65.8%는 생성 thinking ON으로 측정된 무효 수치** (§0 버그). 177558/177559로 thinking OFF·per-epoch 재측정 중.
- ⚠️ **이전 "가설 기각" 결론은 보류** — flawed 비교에 기반. 재측정 완료 후 재판정.
- **All-zero WAR(30% vs 43.3%)는 eval thinking과 무관**하므로 유효: Thinking OFF에서 전문화 신호가 약함.

### 준비된 다음 실험
- **Seed 20210009**: exclusive_solves scout + Thinking OFF, 300×5  
  → `bash scripts/sbatch/submit_bigmath_seed20210009.sh` 로 제출  
  ※ config(`bigmath_train_seed09.yaml`) `enable_thinking: false`로 수정됨(미커밋)

---

## 3. 핵심 파일 구조

```
configs/
  base.yaml                        # 기본 설정 (tp_size=4)
  bigmath_train.yaml               # seed07: 50K, tp=2
  bigmath_train_nothink.yaml       # seed08: 300, Thinking OFF, tp=2
  bigmath_train_seed09.yaml        # seed09: 300, exclusive_solves, Thinking OFF, tp=2

scripts/sbatch/
  submit_bigmath_seed2021000N.sh   # 각 seed 제출 스크립트 (evolution + eval 자동 의존)
  run_bigmath_evolution.sh         # seed07용 (bigmath_train.yaml)
  run_bigmath_evolution_nothink.sh # seed08용 (bigmath_train_nothink.yaml)
  run_bigmath_evolution_seed09.sh  # seed09용 (bigmath_train_seed09.yaml)
  run_bigmath_eval.sh              # seed08 eval (nothink.yaml)
  run_bigmath_eval_seed09.sh       # seed09 eval (seed09.yaml)

src/
  orchestrator.py    # 진화 루프 (WAR 계산, exclusive_solves 누적, scout 호출)
  scout.py           # 새 에이전트 제안 (META_AGENT_MATH_PROMPT_V2 지원)
  prompts/meta.py    # META_AGENT_MATH_PROMPT (NON-REDUNDANCY+ATOMICITY 있음)
                     # META_AGENT_MATH_PROMPT_V2 (둘 다 없음, exclusive_solves 있음)
  war.py             # WAR 점수 계산

results/bigmath/
  seed2021000N/
    roster_final.json              # 최종 로스터
    bigmath/seed2021000N/
      evolution_log.jsonl          # 스텝별 WAR, UB, decision 로그
      roster_step_N.json           # 각 스텝 로스터 스냅샷
    eval/
      inference_moe.jsonl/.score.json   # MoE 평가 결과
      inference_luca.jsonl/.score.json  # Luca baseline 평가 결과
    ub_eval/
      roster_{agent_id}.json       # 에이전트별 단일 roster (test UB용)
      run_all.sh                   # 에이전트별 단독 추론 → UB 계산용 sbatch
      inference_{agent_id}.jsonl   # 에이전트별 추론 결과
```

---

## 4. Seed 20210009 설계 (exclusive solves scout)

### 변경 사항 (commit `26c6f8a`)
1. **`orchestrator.py`**: 배치마다 단독 풀이(다른 에이전트 없이 혼자 맞춘) 문제 텍스트를 `roster_entry["exclusive_solves"]`에 누적 (최근 10개 유지)
2. **`scout.py`**: `exclusive_solves_map` 파라미터 추가. `use_exclusive_solves=True`이면 `META_AGENT_MATH_PROMPT_V2` 사용
3. **`prompts/meta.py`**: `META_AGENT_MATH_PROMPT_V2` 추가 — NON-REDUNDANCY, ATOMICITY 규칙 없음. 대신 per-agent exclusive solve history 섹션 제공
4. **`configs/bigmath_train_seed09.yaml`**: `use_exclusive_solves: true`, `enable_thinking: false`

### 실행 방법
```bash
bash scripts/sbatch/submit_bigmath_seed20210009.sh
```

---

## 5. test UB per agent 구하는 방법 (ub_eval)

seed06에서 확립된 방식. seed08 ub_eval 디렉토리에 준비 완료.

```
results/bigmath/seed20210008/ub_eval/
  roster_{agent_id}.json   # 에이전트별 단일 roster (이미 생성됨)
  run_all.sh               # sbatch 제출용
```

```bash
# 제출
sbatch results/bigmath/seed20210008/ub_eval/run_all.sh
```

결과 해석: 각 에이전트의 Pass@1 + 합집합(union)이 oracle UB.

---

## 6. 주요 발견사항

| 실험 | 결론 |
|------|------|
| Thinking OFF (seed08) | Thinking ON 대비 UB -2.8pp, 전문화 실패 (all-zero WAR 43% vs 30%). 가설 기각. |
| Coarse-to-fine specialization | scout이 초반엔 수학 제너럴리스트 → NON-REDUNDANCY 규칙으로 점점 세부 전문화 유도. 자연 발생 패턴. |
| WAR exclusive solves (seed09 동기) | scout에게 각 에이전트가 실제로 단독으로 푼 문제를 보여주면 NON-REDUNDANCY 텍스트 규칙 불필요. |

---

## 7. SLURM 운영 팁

- **노드**: n04 (GPU 0-3, PRO6000 4장)
- **병렬 실험**: tp=2로 2장씩 나누면 두 실험 동시 실행 가능
- **resume**: `bash submit_bigmath_seed2021000N.sh --resume`
- **eval 자동화**: submit 스크립트가 `--dependency=afterok` 으로 eval 자동 제출

---

## 8. 미완료 사항

- [ ] seed20210007 resume 계속 필요 (50K 데이터, 1 epoch, 여러 번 48h 슬롯 필요)
- [ ] seed20210008 ub_eval 실행 (`sbatch results/bigmath/seed20210008/ub_eval/run_all.sh`)
- [ ] seed20210009 제출 및 분석
- [ ] MBPP coding domain ub_eval (GPU 여유 생기면)
- [ ] NuminaMath-CoT 활용 방안 논의 (CoT path 사용법)
