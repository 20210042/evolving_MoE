# MetaAgentEvolution — 인수인계 문서

작성일: 2026-06-08 (갱신: 2026-06-08 오후 — eval 방법론 수정 작업 반영)

---

## 0. 2026-06-08 오후 작업 요약 (eval 방법론 수정)

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
