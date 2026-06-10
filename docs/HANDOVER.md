# MetaAgentEvolution — 인수인계 문서

작성일: 2026-06-08 (갱신: 2026-06-10 — NuminaMath 통합 + math/coding 분기 버그 수정, 파일럿 재실행)

---

## 0. 2026-06-10 작업 요약 (NuminaMath 파일럿 + domain 분기 버그)

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
