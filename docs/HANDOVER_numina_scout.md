# Handover — NuminaMath scout-evolution (seed16 / 17, + gatefix seed18 / 19 §9)

## 0. 용어 (혼동 방지 — 정확히)

- **hard error**: 그 batch에서 **로스터 전원이 못 푼 문제**. 어느 시드든 동일 개념. scout에 항상 전달됨.
- **틀린 풀이과정 (failed solution attempt)**: 하나의 전문가가 그 hard-error 문제에 대해 *생성한* 풀이 텍스트 — 결과가 틀린 것. ⚠️ "오답"(wrong answer)이 아니라 **풀이 과정 전체**를 말함. seed17이 hard error마다 이걸 1개(랜덤 전문가 것) 덧붙임.
- **exclusive_solves**: 각 전문가가 *단독으로만* 맞힌 문제들 (그 전문가만의 niche). 코드에 plumbing 돼 있음 (`exclusive_solves_map`, 에이전트별 최근 10개 × 200자).
- **UB union**: 전문가들을 각각 solo로 돌려 union(최소 1명이 풀면 정답) = oracle 상한. 라우팅과 무관.
- **routed pass@1**: 매니저가 문제당 전문가 1명 선택(route-to-one) 후 그 전문가 풀이 채점.

## 1. 프로젝트 한 줄

고정 백본(gemma-4-26B-A4B) 위에서 **프롬프트-레벨 전문가 로스터를 진화**(scout가 새 페르소나 제안 → 게이트가 WAR/도태로 선택). NuminaMath(math) 도메인. 우리 영역 = 진화 + 로스터 + binning 라벨 생성까지. **다운스트림 weight-MoE 학습은 협업자 영역(우리 과제 아님).**

## 2. 완료된 것

### seed16 — 주제(topic) scout (기준선)
- scout 프롬프트 = `META_AGENT_MATH_PROMPT`("What expert is missing?") → 분야로 분화. hard error = **문제 본문만** 전달.
- 최종 로스터 5명: Combinatorial Probability / Integral Calculus / Polynomial Expansion / Arithmetic Number Theory / Analytic Proof.
- **full-train binning 완료** (잡 185272): 5명 전원이 train 전체(62,185)를 각자 풀고 문제별 라벨. `--pipeline binning`([src/pipelines/binning_inference.py](../src/pipelines/binning_inference.py)) + [scripts/score_binning.py](../scripts/score_binning.py). 결과: per-expert 72-73%, **UB union 77.58%**, 단독유일 2.6%, 전원해결 67.8%.
- **전달 패키지** `export/numina_binning_seed16/` (코드네임→설명 매핑 + 라벨; 라벨 jsonl은 gitignore).

### seed17 — 실패유형(failure-mode) scout
- **seed16 대비 바뀐 것 딱 2개**: ①hard error마다 **틀린 풀이과정 1개(랜덤 전문가)를 덧붙임** ②scout 프롬프트 = `META_AGENT_MATH_PROMPT_FAILURE`("What recurring mistake do these attempts share?") → 실패유형으로 분화. config 토글 `failure_mode_scout`, batch 100→25.
- ⚠️ **실패유형 유도 = 프롬프트 + 틀린 풀이과정 노출, 둘 다가 원인** (풀이과정을 보면 모델이 "뭐가 잘못됐나"로 자연히 기움).
- 최종 로스터 5명 = 전부 실패유형: ProceduralExecutionVerifier / TypographicalErrorDetective / BoundaryConditionSpecialist / ConstraintIntegrityValidator / ContextualIntentRecoverer.
- 진화: 잡 185974 TIMEOUT→185975 resume, ~52h, step 2488 완주. 로스터 수 1~8 진동(평균 4.1).

### 수학 도메인 결과 매트릭스 (A4B, test 500, 동일 LUCA 시작) — [EXPERIMENT_LOG §12-4](EXPERIMENT_LOG.md)

| 지표 | LUCA (1인) | seed16 topic (5인) | seed17 failure (5인) | seed20 saturated (**15인**) |
|---|---|---|---|---|
| 로스터 크기 | 1 | 5 | 5 | **15** |
| routed pass@1 | 78.0 | 78.2 | 76.8 | 76.0 |
| **UB union** | — | 81.0 | 82.2 | **83.0** |
| 상보성 (UB−best) | — | +3.6 | +4.8 | +4.6 |
| 단독으로 푼 문제 (n=1) | — | 1.4% | 2.2% | 1.8% |
| 아무도 못 풂 (backbone-hard) | — | — | — | 17% (85/500) |

- **saturated 결론**: 로스터를 5→**15**로 3배 불려도 **UB는 81→83 (+1~2pp)뿐** = 강한 diminishing returns → **천장 ≈ 백본(~83%)**. 프롬프트-레벨 분화만으론 backbone-hard 17%를 못 뚫음. routed(76.0)는 UB(83.0)를 −7pp 못 거둠 = 라우터 병목(코딩과 동형). seed20 UB는 ub_eval per-agent solo를 post-hoc union(`ub_eval/binning_merged.jsonl`).

- **판정**: 실패유형 분해가 분해 *품질*(UB·상보성·단독유일·천장 4지표)은 일관되게 우수 = 덜 중복·더 상보적. **단 정확도(routed)는 오히려 ↓** — route-to-one이 UB 헤드룸을 못 거둠(병목=추출, 중복).
- **라우팅 검증**: 라우터에 system_prompt 설명 노출시켜도 routed 76.8 불변 → 라우팅 *정보 부족*이 아니라 *중복*이 원인. ([routing_inference.py](../src/pipelines/routing_inference.py) `_roster_json` fallback 추가됨, 커밋 dd32e0e.)
- **큰 그림**: 프롬프트-레벨 진화는 주제든 실패유형이든 *정확도*로는 LUCA±노이즈로 평탄. 차이를 의미있게 검증하려면 **이 분해로 weight 학습이 들어가야**(협업자 영역).

## 3. (폐기됨) 가치중립 scout 탐색 — ⚠️ 현재 seed18/19와 무관

> **주의**: 아래는 "프롬프트로 창발 유도" 방향의 *초기 탐색 기록*일 뿐이다. **현재 seed18/19 = 게이트 수정(gatefix) — §9 참조.** seed18/19 번호는 그 게이트 실험에 재사용됐고, 아래 가치중립 scout는 **구현 안 됐고 진행 중도 아님.**

**원 동기**: seed16/17 둘 다 인간이 고른 축을 프롬프트에 박음(missing expert→주제, mistake→실패유형). 축을 안 주고 "실제 빈자리를 채워라"만 시켜 인간이 예측 못한 분류가 창발하게 하려던 아이디어(잠정안: 틀린 풀이과정 빼고 exclusive_solves로 커버리지 구조 제공, 중립 프롬프트).

**폐기 이유**: 논의 끝에 (a) 완전 중립 불가(LLM이 진공을 주제 prior로 채움), (b) binning상 축 바꿔도 기능 천장(~77/82) 끈끈→정확도 평탄 예상, 결정적으로 (c) **진짜 병목은 프롬프트(proposal 축)가 아니라 게이트**임이 드러남 — 로스터 진동(제거 로직, §9) + proposal 쪽 deadlock/distractor(§10). → **게이트 수정으로 피벗.**

## 4. 측정 사실 (현재 코드 기준, A4B)

- 문제 본문 mean 249자(~71tok). 풀이과정 mean 2,436자(~696tok), median 556tok, p90 ~1,100, p99 ~3,350.
- hard error율 22.4%(5명 로스터)~27%(LUCA 단독). batch별 hard error: 100→~22, 50→~11, 25→~5.6.
- `enable_thinking`은 생성·후보probe·scout **공유 단일 전역** (seed16/17=OFF). scout 전용 thinking 노브 없음.
- scout cap: 일반 4,000자 / failure-mode 40,000자 ([src/scout.py](../src/scout.py)). max_model_len: seed16=16384, **seed17=32768**(16384이면 failure scout가 컨텍스트 초과로 죽음 — 검증됨).

## 5. 코드 지도 (변경점)

- 진화 오케스트레이터 [src/orchestrator.py](../src/orchestrator.py): `run_batch`(문제×전문가 생성+채점, hard_errors 구성, failure_mode면 틀린 풀이과정 append), `run_epoch`(WAR/게이트/scout 호출). `_fm_rng`, `failure_mode_scout` 플래그.
- scout [src/scout.py](../src/scout.py): 프롬프트 선택 분기(V1/V2/V3/failure), cap.
- 프롬프트 [src/prompts/meta.py](../src/prompts/meta.py): `META_AGENT_MATH_PROMPT`(주제), `_V2/_V3`(exclusive/approach), `_FAILURE`(실패유형). 라우터 `MANAGER_MATH_PROMPT`(일반, 축 없음).
- 라우팅/binning 추론 [src/pipelines/routing_inference.py](../src/pipelines/routing_inference.py)(GMRouting, `_roster_json`), [src/pipelines/binning_inference.py](../src/pipelines/binning_inference.py)(BinningPipeline).
- 러너 [scripts/run_evolution.py](../scripts/run_evolution.py)(config→orchestrator 배선; resume는 `evolution_log.jsonl` 읽음), [scripts/run_inference.py](../scripts/run_inference.py)(`--pipeline evolved|raw|self-refine|binning`).
- config: `numina_train_seed16.yaml`(주제,b100,16384), `numina_train_seed17.yaml`(실패,b25,32768), `numina_eval_a4b.yaml`(eval/binning용 a4b). 제출 래퍼 `scripts/sbatch/submit_numina_seed2021001N.sh`, 진화 `run_math_evolution.sh`, eval `run_numina_eval_steps.sh`/`run_bigmath_ub_eval.sh`/`run_numina_eval_luca.sh`.

## 6. 잡

| 잡 | 내용 | 상태 / 측정 |
|---|---|---|
| 183294→184646 | **seed16 진화**(batch100, cancel→resume) | COMPLETE, 742 step, **31.7h+11.9h=43.6h** (~211 s/step) |
| 185272 | seed16 full-train binning | COMPLETE |
| 185974→185975 | **seed17 진화**(batch25, 48h TIMEOUT→resume) | COMPLETE, 2478 step, **48.0h+4.0h=52.0h** (~76 s/step) |
| 188988 | seed17 UB eval | COMPLETE (UB 82.2) |
| 188989 / 189415 | seed17 routed eval (+라우터 fix 재평가) | COMPLETE (76.8, 불변) |
| 194231 | **seed18 진화**(=16+gatefix, batch100) | **COMPLETE, 43.5h, 622 step(풀 에폭)** — 결과 §9-4 |
| 194232→cancel | seed19 main(=17+gatefix, batch25) | 중단(step768 저장) → 자원 양보 위해 죽이고 아래로 직렬화 |
| 199383 | **seed19 진화 resume**(step768~, `afterany:194231`) | **COMPLETE, 47.8h, 2481 step(풀 에폭)** — 결과 §9-4/9-5 |
| 201281(+201282/283 resume) | **seed20 SATURATED**(add_only, topic 기본형 b100) | RUNNING(n04) — §9-6. ETA ~90–110h(포화 로스터 커서 느림), resume×2 |

## 7. 제약/주의 (git)

- **커밋 메시지: 무조건 한 줄. Co-Authored-By / Claude 이름 절대 금지.** (하네스 기본 트레일러 override.)
- jh/evolution(작업 브랜치)는 트레일러 정리됨. **main/origin의 과거 4커밋(d09a850 등)엔 아직 트레일러 있음** — push된 공유 히스토리라 force-push 필요(협업자 영향) → 사용자가 "놔둬"로 보류.
- 진화 변경/제안 전 **현재 코드를 끝까지 읽고** 판단할 것. 컨텍스트 예산은 char 아닌 worst-case 토큰으로 검증.
- **⚠️ HF 캐시는 반드시 /data5**(홈 쿼터 39G/하드48.8G로 작음, 모델 52G). `~/.bashrc`가 `HF_HOME=/data5/jaehoonjeong/.cache/huggingface`를 설정하나 **sbatch는 제출 셸 env에 의존** → HF_HOME 없는 셸에서 제출하면 잡이 홈으로 52G 재다운로드하다 쿼터 초과로 **모델 로드 단계에서 죽음**(seed20 201281~283 사례). **수정 완료**: `setup_job_env()`([common_bigmath.sh](../scripts/sbatch/common_bigmath.sh))가 이제 HF_HOME/TRANSFORMERS_CACHE=/data5를 강제 export → 제출자 무관하게 안전.

## 8. 다음 액션 (이어받을 때)

1. ~~seed18 = 가치중립 scout 설계~~ → **폐기/대체됨.** 논의가 "프롬프트로 창발 유도"에서 **"게이트 자체를 고친다"** 로 피벗 → §9. (seed18 번호는 게이트수정 실험에 재사용.)
2. (선택) seed17 로스터로 full-train binning → 단독유일 비율 대규모 재확인(test 500 n=1 보강).

## 9. 게이트 제거(deletion) 로직 수정 — seed18/19 (gatefix)

### 9-0. 동기 (진단)
seed16/17 진화 로그상 **로스터 크기가 감쇠수렴이 아니라 limit-cycle로 진동**(seed16 ~3↔8 주기~15step / seed17 ~2↔6 주기~25step; mean-crossings 96·199). → "최종 로스터"가 사이클 멈춘 우연한 위치라 재현성 약함. 진동의 하강엔진 = **제거 로직이 누더기**: 한 전문가의 redundancy를 3단계가 서로 다른 신호로 판단 — ①lives 자격=순간WAR+면제2종 ②pick_worst 순서=누적 average_war ③Phase2 delete trigger=**순간 단일배치 mcl**. 자격·방아쇠가 단일배치 노이즈라 *niche 문제가 그 배치에 안 뽑히면 mcl=0→오삭제→회전문/붕괴.* (+ `shared_contribution_exemption` 기본 True가 삭제를 끄는 밴드에이드, lives 전량리셋 버스트.)

### 9-1. 수정 (확정·구현 완료, 토글식 OFF=byte-identical)
**제거 경로 전체를 하나의 누적 unique-rate 윈도로 통일.** `war_scores[id]`(== 그 배치 단독해결 수)를 윈도 W배치 누적 → `unique_rate = sum/(W·batch)`:
- **방아쇠**: 단일배치 mcl → `unique_rate`로 교체(`u_delete = lambda_del − unique_rate`).
- **자격(lives)**: 순간WAR 전량리셋/−1+면제2종 폐기 → `unique_rate ≤ floor`면 −1, 위면 부분리셋(min(max,+1)). 신생엔 풀윈도 grace. 면제 2종 제거(윈도가 흡수).
- **순서**: pick_worst 정렬키 average_war → unique_rate로 통일.
- **하강완충**: `delete_cooldown` = 삭제 ≤1회/K step(연쇄 급락 차단). 오버슈트 자체는 둠(exp add 페널티가 캡).

토글(전부 0=OFF=현행): `deletion_window`(W), `deletion_floor`(floor), `delete_cooldown`(K). 윈도 상태 `recent_unique`는 roster_final.json에 저장돼 **resume 생존.**

**변경 파일**: [src/action_selector.py](../src/action_selector.py)(`worst_unique_rate` 파라미터), [src/war.py](../src/war.py)(`unique_rate_map`), [src/orchestrator.py](../src/orchestrator.py)(recent_unique 누적·lives 분기·cooldown·init 토글), [configs/base.yaml](../configs/base.yaml)(기본 OFF), [scripts/run_evolution.py](../scripts/run_evolution.py)(cfg 배선). 검증: py_compile + 단위테스트(OFF 동일, 윈도서 고-rate niche 오삭제 면제 확인). **실 진화로 OFF byte-identical은 미검증(사용자 "검증런 생략" 지시).**

### 9-2. 실험 (프롬프트 무변, 게이트만 A/B)
| 신규 | clone | batch | W | floor | K | config / submit |
|---|---|---|---|---|---|---|
| **seed18**(주제+gatefix) | seed16 | 100 | 8 | 0 | 5 | [numina_train_seed18.yaml](../configs/numina_train_seed18.yaml) / [submit_…18.sh](../scripts/sbatch/submit_numina_seed20210018.sh) |
| **seed19**(실패+gatefix) | seed17 | 25 | 30 | 0 | 15 | [numina_train_seed19.yaml](../configs/numina_train_seed19.yaml) / [submit_…19.sh](../scripts/sbatch/submit_numina_seed20210019.sh) |

- **값 근거**: W ≈ 800/batch(윈도가 덮는 문제 수 ~800 맞춤). floor=0 = "윈도 동안 단독해결 완전 0일 때만 lives 감소"(가장 보수적; 미세판별은 lambda_del trigger). K ≈ 주기의 1/3. lambda_del 스케일 참고: batch100≈0.008, batch25≈0.031 → floor는 0~그 근처.
- **ETA**(실측 per-step 환산): seed18 ~37–44h(48h 단일잡 OK), seed19 ~52h(48h 초과→resume 필요). 게이트는 per-step 비용 무변(scout는 매 step 그대로 호출).
- **잡 운용**(자원 양보): seed18·seed19 동시 RUNNING이 CPU 쿼터를 막아 사용자 타 작업 PENDING → seed19(194232/233) 중단(step768 저장)하고 **199383으로 `afterany:194231`(seed18) 뒤에 직렬화 resume.**
- **비교 측정**: seed16(OFF) vs seed18(ON), seed17(OFF) vs seed19(ON) — 로스터 크기 진동 진폭·주기(mean-crossings), 멤버십 churn(add≈delete 감소), 최종 UB·routed, 삭제가 *지속적 저 unique_rate*에만 발동하는지. 평가 잡은 진화 완료 후 별도.

기안서 원본: `~/.claude/plans/deletion-whimsical-fairy.md`.

### 9-3. 보고용 figure (seed16/17 진화 동역학)
[scripts/make_roster_periodicity_fig.py](../scripts/make_roster_periodicity_fig.py) 한 방에 4개 생성(`docs/`). 한글 라벨 = Noto Sans CJK, UB는 edge-보정 이동평균.
- **fig_roster_periodicity** — 로스터 크기 + 평활곡선(창<주기라 진동 유지): **수렴 안 함, limit-cycle**(16 주기≈15 / 17 ≈25). 패널 라벨 = "Hard error description only" / "Description + 풀이과정".
- **fig_persona_families** — 가장 자주 add된 페르소나 top-12: **같은 전문성 반복 제안**(§10). 같은 라벨.
- (미사용) fig_churn(누적 add≈delete), fig_ub_trajectory(UB ~77–78% 평탄) — 비직관적이라 보고선 제외, 근거용으로만 보관.
- **fig_gatefix_compare_seed1618** — seed16 OFF vs seed18 gatefix 로스터크기 겹쳐그림(§9-4 근거).

### 9-4. 1차 결과 — seed18 (gatefix topic, COMPLETE)
잡 194231 COMPLETE, 43.5h, 622 step(풀 에폭, seed16과 동일 per-step). seed16(OFF) 대비:

| | seed16 OFF | seed18 gatefix |
|---|---|---|
| add+swap / delete+swap | 188 / 180 | **65 / 57** |
| noop | 414 | **509** |
| 로스터 mean / range / final | 5.85 / 2–9 / 5 | **8.39 / 2–10 / 9** |
| 진동 cycles / period | 48 / 15 | **33 / 18** |

- ✅ **회전문 닫힘(1차 목표 달성)** — churn(turnover) ~1/3로 급감, noop↑ = 스텝별 안정.
- ⚠️ **floor=0 과교정** — "8배치 윈도 단독해결 *완전 0*일 때만 evict"라 삭제가 거의 안 걸림 → **로스터가 mean 8.4로 부풀어 cap(10) 근처 눌러앉음**(최종 9). 진동도 33 cycles로 줄었으나 *limit-cycle이 cap 근처로 상승*한 모양 — **수렴은 아직 아님.**
- **seed19(실패+gatefix)도 동일 패턴 확인**(COMPLETE, 47.8h, 2481 step): churn 631→199, 진동 99→25 cycles(주제보다 *더* 완화), 로스터 mean 4.0→7.7로 비대. **gatefix 효과가 두 축에서 robust 재현.**

### 9-5. ⚠️ 핵심 발견 — unique_rate 스케일은 로스터 크기에 의존 (floor 설계 재정립)
| | 로스터 mean | unique_rate median (W윈도) |
|---|---|---|
| seed17 OFF | 4.0 | **0.0120** |
| seed19 gatefix | 7.7 | **0.0027** |

같은 failure scout·batch25인데 rate 4.5배 차 — **로스터 크면 overlap↑ → 1인당 단독해결↓ → unique_rate↓.** 즉 rate 스케일 = 로스터 크기의 함수(고정 아님).
- **정정**: "W×batch 맞춰 floor 포팅" 논리는 부정확했음. seed18/19가 비슷했던 건 *둘 다 비대해져 크기가 비슷*해서지 W 스케일링 덕 아님. seed17(작은 로스터)은 0.012로 다름.
- → **절대 floor는 로스터 크기 따라 의미가 흔들림. percentile 기반 floor(최근 rate 분포 하위 X% evict)가 robust.** ← 다음 설계 방향.
- floor=0.002는 비대 regime(median 0.0027)서 ~30% evict → 하향 시작→줄면 rate↑→self-stabilize. 절대값도 equilibrium은 찾으나 멈추는 지점 예측 어려움 → **percentile 권장.**

### 다음 트랙 (2개)
1. **minimal 수렴**: percentile floor(또는 floor≈0.002 절대) 튜닝 → 로스터가 작고 안정된 크기로 수렴하나.
2. **maximal 포화 레퍼런스** (학생 아이디어) — **구현·제출 완료 → §9-6.**

### 9-6. Saturated run (seed20, add_only) — 진행 중
**목적**: Phase2(delete) 봉쇄하고 add-or-noop만 → 로스터 단조증가 → exp add-페널티 고정점에서 포화. 진동 없는 **"maximal final roster" 상한** + *maximal로도 UB ~77–78 못 넘으면 천장=backbone(distractor §10) 확정.*
- **구현**: 토글 `add_only`([action_selector.py](../src/action_selector.py) phase2 봉쇄 + orchestrator/run_evolution/base.yaml 배선, OFF=byte-identical, 단위테스트 통과). 조기종료 없음(풀에폭 쭉 — 사용자 지시).
- **config/submit**: [numina_train_seed20.yaml](../configs/numina_train_seed20.yaml)(topic 기본형·풀이과정X·b100·add_only), [submit_…20.sh](../scripts/sbatch/submit_numina_seed20210020.sh)(resume×2).
- **잡**: 201281 RUNNING(+201282/283). **ETA ~90–110h**(포화 로스터 커서 per-step ~2.5–3× seed16; 포화 후 tail도 매 스텝 전원 생성이라 비쌈).
- **관전**: 포화 크기(N*)·유지 여부(진동 없이 flat?), seed16(OFF, 진동)과 대조, 최종 UB가 천장(~77–78) 넘나.
- **다음 도메인(사용자 계획)**: 수학은 프롬프트만으론 안 풀리는 게 많음 → **역할부여가 실제 정오답에 영향 주는 도메인**으로 확장 예정(legalQA, 코딩 등).

## 10. Proposal-side deadlock & distractor (보고 메시지 3·4)

**§9(gatefix)는 delete 쪽**(중복 삭제 노이즈→진동) 문제다. 이건 **add/proposal 쪽**의 *별개* 문제 — gatefix가 안 건드림. 그래서 "중복(삭제) 고쳐도 scout는 여전히 같은 전문가를 반복 추천"이 정확한 표현.

### 진단 (코드 확인됨 — 맞음)
1. **hard error = 로스터 전원이 못 푼 문제 전부를 무차별로 scout에 전달**([orchestrator.py:258-272](../src/orchestrator.py#L258-L272)). "더 좋은 프롬프트면 풀릴 문제"와 "백본이 원천적으로 못 푸는 문제"를 **구분 안 함**(같은 더미).
2. **add는 그 hard error를 실제로 깨야만 발동**(`gh_add=|hard∩new_pass|/batch>0`, [action_selector.py:64](../src/action_selector.py#L64), probe [orchestrator.py:440-461](../src/orchestrator.py#L440)).
→ 백본이 못 푸는 hard error: scout 제안→probe 실패→**add 안 됨**→그 문제 여전히 미해결→다음 배치 또 hard error→**같은 전문성 재제안**→또 실패. **루프(deadlock).**

### 쉬운 설명 (채용 비유)
scout=채용담당, 매 라운드 "이 미해결 문제 풀 전문가 구함" 공고→후보를 바로 그 문제로 면접(probe). *커버 빈* 문제는 좋은 후보가 풀어 채용✓. 근데 *백본이 원천적으로 못 하는* 문제는 **어떤 후보도 면접 실패→영영 채용 안 됨→같은 공고 반복→같은 프로필 반복 탈락** = 채울 수 없는 영구 공석.

### distractor 연결 (교수님 논점) + UB 증거
- **타임라인이 UB 곡선에 보임**: 초반 *풀 수 있는 빈자리* 채워져 UB 빠르게 ~77↑ → 이후 hard error 더미가 **풀 수 없는 잔여분**으로 채워져 **UB 평탄(~77–78%)**. **UB 위 ~22% = real hard error**(어떤 페르소나로도 안 풀림).
- 이것들은 **긍정 시그널 0**(누구도 못 푸니 어떤 제안도 보상 못 함). 풀리는 건 더미서 빠져나가니 **시간이 갈수록 scout가 보는 더미는 풀 수 없는 잔여분이 지배** → scout가 유령 쫓으며 같은 doomed 전문성 반복. **중립 아니라 능동적 distractor.**
- **방향**: 지속적으로 안 풀리는 hard error(여러 배치 생존 / N번 probe 실패 / 안정 UB 위)를 **scout 타겟에서 제외** → scout가 채울 수 있는 빈자리만 봄 → 제안이 실제 add·수렴·반복 멎음. (미구현, 다음 후보 작업.)

### calibration (보고 시 과장 금지)
1. 완전 deadlock 아닌 **잔여분 deadlock** — 풀리는 빈자리는 초반에 채워짐(add 188/330 실제 발생). 정확한 표현 = "*풀 수 있는 걸 다 채운 뒤 남은 신호가 불가능 demand라 scout가 헛돈다*".
2. fig_persona_families는 **seed16/17(gatefix 전)**이라 반복에 **중복-churn + deadlock이 섞임.** 둘을 깨끗이 분리하는 건 **seed18/19(gatefix 후)** — 거기서도 같은 전문가 반복되면 deadlock 단독 증거. 보고선 "seed18/19로 확증 예정".
