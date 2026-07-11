# BigMath Evolution Ablation Study

> ⚠️ **(2026-06-09) 이 문서의 Pass@1/UB 수치는 채점기 버그(raw LaTeX false-negative) 영향을 받은 값이다.**
> 재채점 시 모든 값이 +16pp 상승(예: raw baseline 68→84%, MoE ~67→~83%, UB ~71→~87%). 채점기는 수정됨(`fix/math-verify-scorer`).
> 또한 진화 WAR도 같은 버그 채점기로 계산됐으므로 교정 채점기로 진화 재실행 중(seed 20210106/108). **재실행 완료 전까지 결론 해석은 보류.** 상세: [HANDOVER.md](HANDOVER.md) §0.

BigMath (`Jongbin-kr/BIG-MATH_filtered`) 데이터셋에 대한 Meta-Agent Evolution 실험 기록.  
각 시드는 이전 시드에서 발견된 문제를 하나씩 수정하는 누적 ablation 구조다.

---

## 공통 설정

| 항목 | 값 |
|---|---|
| 모델 | `google/gemma-4-31B-it` |
| Pipeline | One-step MoE (routing → expert 1-pass generation) |
| Scoring | `math_verify_score` (LaTeX/수식 정규화 후 비교) |
| Train split | 300문제, 배치 50, Epoch 5 |
| Eval split | test 500문제 (`--max_items 500`) |

---

## Seed별 설정 비교

| | **20210001** | **20210002** | **20210003** | **20210004** | **20210005** | **20210006** | **20210007** |
|---|---|---|---|---|---|---|---|
| **max_lives** | 3 | 5 | 5 | 5 | 5 | 5 | 5 |
| **Scout 프롬프트** | 구버전 | 구버전 | **신버전** | 신버전 | 신버전 | **최소개입 (Verbal RL)** | 최소개입 |
| **Router** | MANAGER_PROMPT (코딩용) | MANAGER_PROMPT (코딩용) | **MANAGER_MATH_PROMPT** | MANAGER_MATH_PROMPT | MANAGER_MATH_PROMPT | **최소개입** | 최소개입 |
| **NON-REDUNDANCY** | 일반 | 일반 | **(CRITICAL)** | (CRITICAL) | (CRITICAL) | (CRITICAL) | (CRITICAL) |
| **ATOMICITY** | 없음 | 없음 | **추가** | 추가 | 추가 | 추가 | 추가 |
| **`and` 이름 금지** | 없음 | 없음 | **추가** | 추가 | 추가 | 추가 | 추가 |
| **All-zero WAR 패널티** | 감산함 | 감산함 | 감산함 | **면제** | 면제 | 면제 | 면제 |
| **Shared 기여 lives 면제** | 없음 | 없음 | 없음 | 없음 | **추가** | 추가 | 추가 |
| **train_size** | 300 | 300 | 300 | 300 | 300 | 300 | **50,000** |
| **tp_size** | 1 | 1 | 1 | 1 | 1 | 1 | **4** |
| **squad_solves 로깅** | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | **추가** |
| **Action gate 정규화** | 없음 | 없음 | 없음 | 없음 | 없음 | 없음 | **batch_norm** |

---

## 변경 이유 (누적)

### 20210006 → 20210007: 대규모 데이터 + 인프라 확장
**목표**: system prompt 수준 전문화의 ceiling 확인 → LoRA-MoE 학습 데이터 수집

**변경:**
- `train_size`: 300 → 50,000 (BigMath train 전체)
- `epochs`: 5 → 1 (50K × 1회 순회, 1,000 스텝)
- `tp_size`: 1 → 4 (n04 PRO6000 Max-Q × 4, cross-NUMA PCIe)
- `max_num_seqs`: 64 → 128
- `squad_solves` 로깅 추가: 매 스텝마다 {agent_id: [solved_problem_ids]} 기록 → WAR 기반 expert 라벨링 데이터 확보
- Action gate `batch_size_ref=50` 정규화: batch_size 변화에 무관하게 lambda 스케일 유지

**기대:**
- 에이전트가 더 다양한 문제 유형에 노출 → roster 전문화 심화
- 1에폭 완료 후 최종 roster로 50K 재추론 → 문제별 expert 라벨 생성 → LoRA-MoE 학습

**인프라 메모:**
- n04 GPU 현황: 실측 topology → NVLink 없음, cross-NUMA NODE 연결 (GPU 0: NUMA 0, GPU 1-3: NUMA 1)
- GPU 모델: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96GB GDDR7)
- 48h SLURM 잡 × 약 6회 resume 예상 (`bash submit_bigmath_seed20210007.sh --resume`)
- SLURM job ID: 176757

### 20210001 → 20210002: `max_lives` 3→5
- **문제**: Epoch 3에서 3연속 all-zero WAR 배치로 lives=3이 동시 소진 → 3명 연속 방출, roster 붕괴
- **변경**: lives를 늘려 cascade eviction 완충

### 20210002 → 20210003: Scout/Router 프롬프트 전면 수정
**발견된 문제들:**
1. **Router 부재**: `MANAGER_PROMPT`가 "software engineering team" / 코딩 도메인 예시로 수학 라우팅을 처리하고 있었음
2. **NON-REDUNDANCY 미흡**: `(CRITICAL)` 없이 소프트한 권고 수준 → 동일 페르소나 중복 스카우트 발생 (e.g. `Foundational_Mathematics_Specialist` 2회)
3. **Atomicity 없음**: `Geometry_and_Algebra_Specialist`처럼 짬뽕 이름으로 실질적 도메인 중복이 발생해도 모델이 인지 못함 → WAR 전원 0의 구조적 원인
4. **`gap_not_covered_by`**: 코드에서 실제로 사용되지 않는 dead field

**변경:**
- `MANAGER_MATH_PROMPT` 신규 추가, `routing_inference.py`에서 domain별 분기
- `META_AGENT_MATH_PROMPT`: `NON-REDUNDANCY (CRITICAL)` + "Read each member's strengths" 추가
- ATOMICITY 규칙 추가: "exactly one tightly-scoped domain, no combining"
- persona_name에서 `and` 금지 (기계적 강제 수단)
- `gap_not_covered_by` 제거

### 20210005 → 20210006: Scout/Router 최소개입 (Verbal RL framing)
**발견된 문제:**
- seed20210005까지 scout이 제안하는 페르소나가 `Calculus Specialist`, `Number Theory Specialist` 등 인간이 설계해도 나올 수학 교과서적 분류 체계를 재현함
- 프롬프트에 "mathematical domain", "problem families", "techniques required" 등 인간 prior가 내재된 언어가 포함 → 모델이 학습 데이터의 수학 분류 체계를 그대로 따라가는 구조
- 논문 주장("인간 prior 없이 모델이 스스로 전문가를 발견")과 실제 구현 간 불일치

**변경:**
- 스카우트: 도메인/전략 레이블 지시 완전 제거. 실패 문제들만 제공하고 "What expert is missing? Define that expert yourself."만 요청
- 라우터: 도메인 매칭 가이드라인 제거. "Pick the specialist most likely to solve this problem."만 남김
- 모델이 system_prompt 내용·형식·페르소나 이름을 자유롭게 결정 (최소 제약: NON-REDUNDANCY, ATOMICITY, `and` 금지, 3문장 이내)

### 20210004 → 20210005: Shared 기여 에이전트 lives 패널티 면제
**발견된 문제:**
- MoE eval Pass@1(65~66%)이 LUCA 단독 baseline(68%)보다 낮음
- 원인 분석: WAR=0이어도 shared 문제를 푼 에이전트(자기 전공 배치에서 다른 전문가와 겹쳐 독점 기여 없음)가 동일하게 lives 패널티를 받는 것이 부당
- "Luck 기반 생존" 문제: 자기 전공 문제가 배치에 안 나오면 억울하게 사망
- Calculus Specialist가 lives=1로 생존 위기에 처한 케이스가 대표적

**변경:**
```python
agent_solved_any = bool(squad_results.get(p_id, set()))
if current_score > 0:
    p["lives"] = self.max_lives
elif all_zero_war:
    pass  # batch-level collective failure
elif agent_solved_any:
    pass  # shared contribution — not penalized
else:
    p["lives"] = max(0, p.get("lives", self.max_lives) - 1)
```

### 20210003 → 20210004: All-zero WAR 배치 lives 패널티 면제
**발견된 문제:**
- 전원 WAR=0 빈도: 27%(seed1) → 29%(seed2) → 15%(seed3)
- Upper bound 58~78%인데도 전원 WAR=0이 나오는 경우 존재 → 에이전트들이 같은 문제를 동시에 풀어서 독점 기여가 없는 배치 (batch-level failure)
- 이를 개인 실패로 처리해 lives를 감산하는 것은 부당

**변경:**
```python
all_zero_war = war_scores and all(v == 0 for v in war_scores.values())
# all_zero_war이면 lives 감산 skip → "collective failure" 처리
```

---

## 진행 결과 (현재까지)

| | **20210001** | **20210002** | **20210003** | **20210004** | **20210005** | **20210006** |
|---|---|---|---|---|---|---|
| Train UB 최고 | 76% | **82%** | 78% | **84%** | 74% | — |
| Train UB 최저 | 36% | 58% | 58% | 60% | 52% | — |
| Train UB 에포크 평균 | — | — | — | — | Ep1 61.3% → Ep5 63.7% | — |
| WAR 전원 0 비율 | 27% | 29% | 15% | **23%** | **47%** | **30%** |
| `and` 이름 비율 | 90% | 61% | **0%** | **0%** | **0%** | **0%** |
| Cascade 방출 | Epoch 3에서 3연속 | Epoch 5에서 3연속 | 없음 | 없음 | 없음 | 없음 |
| Eval Pass@1 최고 | — | — | **66.60%** (Ep4) | 66.40% (Ep1/4) | **67.00%** (Ep2) | **67.40%** (Ep3) |
| Eval Pass@1 (Ep별) | — | — | — | — | Ep1 66.2% / Ep2 67.0% / Ep3 65.6% / Ep4 66.0% / Ep5 66.6% | Ep1 66.8% / Ep2 65.8% / Ep3 67.4% / Ep4 65.4% / Ep5 65.8% |
| Eval Pass@1 평균 | — | — | 65.8% | 65.9% | 66.3% | 66.3% |
| LUCA 단독 baseline | — | — | — | **68.00%** (측정 완료) | 68.00% | 68.00% |
| Test UB (최종 로스터) | — | — | — | **70.40%** (5명 union) | **71.80%** (9명 union) | **70.80%** (6명 union) |
| 최종 로스터 | — | — | Analytic Geo / Stereometry / Combinatorics / Number Theory | Analytic Geo / Calculus / Trigonometry / Probability / Algebra | LUCA + Calculus / Number Theory / Analytic Geo / Combinatorics / Trigonometry / Euclidean Geo / Probability / Algebra (8명) | LUCA + 5 strategy-based specialists |

†LUCA 단독 baseline 68.00%보다 MoE eval이 낮음 → 라우팅 손실 + system prompt 수준 전문화 불충분이 원인  
†seed20210006: 에이전트 수 가장 적음(6명)에도 eval/UB 모두 최고 — verbal RL(최소개입) 방향의 효율성 확인

---

## 주요 관찰

### Scout 이름 패턴 변화
| | `and` 포함 | 중복 발생 |
|---|---|---|
| seed20210001 | 90% (27/30) | 2종 중복 |
| seed20210002 | 61% (17/28) | 2종 중복 |
| seed20210003 | **0% (0/27)** | 없음 |

### seed20210003 특이점
- `Stereometry Specialist` 등장 — 입체기하 단일 전문가, avg_war 0.542로 1위
- `Combinatorics Specialist` 3연속 제안 / `Calculus Specialist` 4연속 → action gate noop 후 새 이름 제안하는 루프 (새 문제)
- Luca → c_20976 순으로 정상 단독 방출, cascade 없음

### UB 분석 (seed20210006 기준)

| 에이전트 | Pass@1 | LUCA 너머 추가 | LUCA와 겹침 |
|---|---|---|---|
| LUCA | 67.0% | — | — |
| c_21320 | 65.4% | +4문제 | 323 |
| c_22261 | 65.8% | +7문제 | 322 |
| c_6780 | 66.4% | +8문제 | 324 |
| c_45632 | 65.8% | +10문제 | 319 |
| c_2185 | 66.2% | +10문제 | 321 |
| **Union** | **70.8%** | **+19문제** | — |

- specialist 5명이 LUCA 너머 추가 커버: **19문제(3.8%)** — LUCA와 96~99% 겹침
- 아무도 못 푼 문제: **146/500 (29.2%)** — 모델 자체 능력 한계
- **system prompt 수준 전문화의 ceiling 확인**: weights가 동일하므로 실질적 분기가 미약
- 다음 단계 가설: 진화로 발견한 분류 방향으로 LoRA 학습 → LoRA-MoE 구성 시 UB 개선폭이 훨씬 클 것으로 예상

### 남은 과제
- **seed20210005 Test UB 완료**: 71.80% (9명 union, LUCA 너머 +23문제. c_53240/c_12166/c_38007은 추가 기여 0~1개로 포화)
- **WAR 동질성**: 수학은 코딩보다 에이전트 간 문제 중복 해결이 많아 WAR 독점 기여 자체가 희박 → all-zero 면제(seed4)로 완화 시도
- **scout 루프**: 같은 도메인을 반복 제안하는 패턴 (e.g. `Calculus Specialist` 4회) — 이미 로스터에 있는 도메인 명시적 열거 강화 필요 가능성

---

## 20210008 — Thinking OFF ablation (seed06 분기)

> seed06(Verbal RL, 최소개입, 300×5)에서 **진화 단계의 thinking만 OFF**로 바꾼 분기. tp=2 병렬화.
> 가설: "thinking이 전문화에 기여하는가?"

**설정 (vs seed06)**: `enable_thinking: false` (solver/scout/router 진화 단계), tp_size 1→2, 나머지 동일.

**진화 결과**: All-zero WAR **43.3%** (seed06 30% 대비 ↑) — Thinking OFF에서 독점 기여 신호가 더 약함. 최종 로스터 9명(LUCA + 8 specialists).

### ⚠️ Eval 방법론 수정 (2026-06-08 오후)

기존 seed08 eval 수치(MoE 65.8%)와 seed06과의 비교가 **두 가지 오류**로 무효임을 발견, 재측정 중.

**오류 1 — 생성 thinking 하드코딩**: config `enable_thinking`은 **진화(orchestrator)에만** 적용되고 **eval inference에는 전달되지 않았음**. `routing_inference.py`의 `GMRoutingPipeline`이 생성 단계에 `enable_thinking=True`를 하드코딩 → seed08의 "Thinking OFF" eval이 실제로는 **생성 thinking ON**으로 측정됨.
- **수정**: `GMRoutingPipeline(gen_enable_thinking=...)` 파라미터 추가, `run_inference.py`에서 `cfg.get("enable_thinking")`로 주입. 라우팅 단계(`enable_thinking=False`)는 seed06·08 공통이라 변경 없음. 기본값 True → 기존 Thinking ON 실험(seed01~06) 수치 영향 없음.

**오류 2 — per-epoch vs single-final 비교**: seed01~06은 **에포크별**(roster_step_6/12/18/24/30) eval 후 best epoch 보고. seed08은 **최종 roster 1회**만(65.8%). 보고서가 seed06 **best epoch(Ep3=67.4%)** vs seed08 **single-final(65.8%)**을 비교 → 사과 대 오렌지. 실제로 seed06 **final epoch(Ep5)=65.8%** = seed08과 동일.

**오류 3 — UB test-set 정렬**: ub_eval은 MoE eval과 동일 `--seed`로 같은 held-out 500을 써야 함. 기존 seed06 ub_eval은 `--seed 0`(full-split 첫 500)이라 자기 MoE eval과도 어긋남. seed08 ub_eval은 `--seed 20210008`로 정렬.

### 재측정 (진행 중)
- seed07(50K, job 177045) **취소** → GPU 확보
- **177558** `run_bigmath_eval_epochs_nothink.sh` — seed08 per-epoch eval (roster_step 6~30), 생성 thinking OFF, tp=2
- **177559** `ub_eval/run_all.sh` — seed08 per-agent UB(9 roster), thinking OFF, `--seed 20210008`

### 재측정 결과 (2026-06-08 완료, jobs 177558/177559)

**seed08 per-epoch MoE Pass@1 (생성 thinking OFF):**

| Ep1 | Ep2 | Ep3 | Ep4 | Ep5 | 평균 | 최고 | 최종(Ep5) |
|----|----|----|----|----|------|------|-----------|
| 67.6 | 67.4 | 67.0 | 65.6 | 67.2 | **67.0** | 67.6 | 67.2 |

**seed08 Test UB (per-agent, held-out 500, thinking OFF, `--seed 20210008`):**

| 에이전트 | Pass@1 | | 집계 | 값 |
|---|---|---|---|---|
| LUCA | 67.2% (336) | | LUCA 단독 | 67.20% |
| c_23871 | 68.0% (340) | | **UB union (9명)** | **71.20%** (356) |
| c_33267 | **68.4%** (342) | | specialist가 LUCA 너머 | **+20문제 (4.0%)** |
| c_56511 | 68.2% (341) | | 아무도 못 푼 문제 | 144/500 (28.8%) |
| 나머지 5명 | 67.2~67.4% | | | |

### seed06 (ON) vs seed08 (OFF) 정면 비교 — 가설 판정

| 지표 | seed06 (Thinking ON) | seed08 (Thinking OFF) | 차이 |
|------|---------------------|----------------------|------|
| MoE Pass@1 평균 | 66.3% | **67.0%** | +0.7 |
| MoE 최고 epoch | 67.4 (Ep3) | 67.6 (Ep1) | +0.2 |
| MoE 최종 epoch | 65.8 | **67.2** | +1.4 |
| LUCA 단독 | ~67.0 | 67.2 | ≈ |
| Test UB (union) | 70.8% | **71.2%** | +0.4 |
| specialist 추가 기여 | +19 (3.8%) | +20 (4.0%) | ≈ |
| 아무도 못 풂 | 29.2% | 28.8% | ≈ |
| **All-zero WAR** | **30%** | **43.3%** | **+13.3** |

**판정:**
1. **성능(MoE)·UB 저하 없음 — 사실상 동률** (OFF가 오히려 미세하게 높음). 기존 보고서의 "Thinking OFF -1.6pp 저하"는 **측정 버그의 산물**: seed06 best-epoch(67.4) vs seed08 single-final(65.8) 비교였고, 게다가 seed08은 생성 thinking이 ON으로 잘못 측정됨. 정합 측정(per-epoch + 생성 thinking OFF)하니 **65.8 → 67.2로 회복**. → **"Thinking OFF가 성능 저하"는 기각.**
2. **유일하게 살아남은 실제 차이 = All-zero WAR 30%→43.3%**. Thinking OFF는 진화 중 **독점 기여(WAR) 신호를 약화**시키지만, 그것이 end-to-end 성능으로 이어지지 않음. 이유: 두 조건 모두 specialist가 LUCA 너머 +4%만 보태고 ~29%는 누구도 못 풂 → **system-prompt 수준 전문화의 ceiling**이 thinking 변수보다 지배적. (= 위 §"UB 분석"의 "LoRA-MoE 필요" 논지 강화)

**caveat**: seed06/08은 seed가 달라 held-out set이 다르고 seed06 UB는 `--seed 0`(full-split)로 측정됨 → 절대 수치 cross-seed 비교는 seed 교란 있음. 다만 **within-seed 구조(MoE≈LUCA, UB가 LUCA+4%, ~29% 미해결)가 양쪽에서 거의 동일**하다는 점이 오히려 견고한 발견.

---

## 20210009 — exclusive_solves scout (seed08에서 NON-REDUNDANCY 텍스트 규칙 제거)

> seed08(Thinking OFF, 텍스트 NON-REDUNDANCY+ATOMICITY 규칙)에서 **scout 입력만 교체**한 분기.
> 텍스트 규칙 대신 각 에이전트가 **실제로 단독 해결한 문제(exclusive_solves)**를 보여줌(`META_AGENT_MATH_PROMPT_V2`, ATOMICITY·`and`금지 규칙 없음). Thinking OFF 유지.
> 가설: "단독풀이 증거를 데이터로 보여주면 텍스트 규칙 없이도 자연 분화한다."

**결과 (per-epoch eval + UB, 모두 Thinking OFF, jobs 177946~177948):**
- per-epoch MoE: 67.0 / 67.6 / 68.0 / 68.0 / 67.0 → 평균 **67.5%**, 최고 68.0, 최종 67.0
- Test UB union(7명) **70.2%**, LUCA 단독 67.4%, specialist가 LUCA 너머 **+14(2.8%)**, 아무도 못 풂 149/500(29.8%)
- All-zero WAR **46.7%**, 최종 로스터 7명(LUCA + 6 specialists)

### seed06 / seed08 / seed09 3-way 종합

| 지표 | seed06 (ON, 텍스트규칙) | seed08 (OFF, 텍스트규칙) | seed09 (OFF, exclusive_solves) |
|------|------|------|------|
| MoE Pass@1 평균 | 66.3 | 67.0 | **67.5** |
| MoE 최고 epoch | 67.4 | 67.6 | **68.0** |
| MoE 최종 epoch | 65.8 | 67.2 | 67.0 |
| LUCA 단독 | ~67.0 | 67.2 | 67.4 |
| **Test UB (union)** | 70.8 | **71.2** | 70.2 |
| specialist 추가 기여 | +19 (3.8%) | **+20 (4.0%)** | +14 (2.8%) |
| 라우팅 갭 (UB−MoE최종) | 5.0 | 4.0 | **3.2** |
| MoE/UB 실현율 | 93.0% | 94.4% | **95.4%** |
| 아무도 못 풂 | 29.2% | 28.8% | 29.8% |
| 로스터 크기 | 6 | 9 | 7 |
| **All-zero WAR** | 30% | 43.3% | **46.7%** |

### 전문가 분화 양상 (이름·유형)

- **seed08 (텍스트 ATOMICITY 규칙)** → **원자적 단일 도메인** 전문가 8명. 이름도 단일: Real Analyst / Complex Analyst / Calculus Specialist / Discrete Logician / Set-Theoretic Probabilist / Analytical Geometer / Competition Mathematician / Applied Numerist. 도메인 중복 최소.
- **seed09 (exclusive_solves, ATOMICITY 규칙 없음)** → **다중 도메인 묶음형** 전문가 6명. 이름에 `&`/`and` 다수: "Discrete **&** Foundations", "Algebraic **&** Precalculus", "Geometry **&** Trigonometry", 그리고 c_48067은 프롬프트에 "**synthesizing multiple domains**"라고 명시. 기하가 6명 중 4명, 삼각이 4명에 등장 → **부분 중복 큼**.

### 해석 — 부분 교집합(완전겹침 아님)이 만드는 두 가지 애매함

핵심: seed09 전문가들은 서로 **완전히 겹치지도, 완전히 분리되지도 않은 "부분 교집합"** 상태다. 각자 공유 문제 + 약간의 단독 문제를 가진다. 이 구조가 진화와 라우팅 양쪽에 애매함을 만든다.

1. **방출(eviction)이 애매해짐** — 데이터로 확증됨.
   WAR/lives는 "독점 기여"로 생존을 판정한다. 부분 교집합이면 **누구도 명백히 잉여가 아니다**(각자 단독 문제가 조금씩 있음) → 명확히 뺄 대상이 없어 반(半)중복 전문가가 로스터에 잔류. 동시에 한 배치의 문제가 여러 전문가에게 나뉘어 **아무도 깔끔한 독점 크레딧을 못 받음** → All-zero WAR가 가장 높음(46.7%). 즉 *"누굴 뺄지 애매하다"*는 직관이 그대로 수치로 나타남.

2. **라우팅 — 직관과 미묘하게 갈리는 지점.**
   라우터는 문제당 전문가 1명을 골라야 한다. 부분 교집합이면 경계가 흐려 "정답 전문가"가 모호해지는 건 맞다. **그러나** 겹치기 때문에 **오라우팅이 관대해진다** — 교집합 영역 문제는 엉뚱한(그래도 겹치는) 전문가를 골라도 풀린다. 그 결과 seed09의 라우팅 갭(UB−MoE최종)이 **3.2로 가장 작다**(MoE/UB 실현율 95.4%, 천장에 가장 근접). 반대로 seed08의 원자적 전문가는 "기하 문제는 반드시 기하 전문가로" 정밀 라우팅이 필요해 갭이 크지만(4.0, 실현율 94.4%), **천장(UB) 자체가 더 높다(+20)**. ※ seed06 갭(5.0)은 UB가 `--seed 0`(다른 500)로 측정돼 정합성 약함 — 참고만.

   → 정리하면 **트레이드오프**다: 부분 교집합(seed09)은 *라우팅은 쉬워지지만(관대) 전문화 천장이 낮고*, 원자적 분화(seed08)는 *라우팅 정밀도를 요구하지만 천장이 높다*. seed09의 "+14(2.8%)"라는 낮은 UB는 이 낮은 천장의 직접 증거다.

3. **가설 판정**: exclusive_solves 데이터만으로는 분화가 *덜* 일어났다. 텍스트 ATOMICITY 규칙(seed08)이 오히려 더 깔끔한 단일 도메인 분화 + 더 높은 UB(+20)를 만들었다. → **seed09 가설(데이터가 텍스트 규칙을 대체)은 지지받지 못함.** 단, 세 조건 모두 UB ~70–71% / ~29% 미해결로 수렴 → **system-prompt 수준 전문화의 천장**이 scout 전략·thinking보다 지배적이라는 결론(= LoRA-MoE 필요)을 재확인.

---

## 20210010 / 20210011 — NuminaMath 전환 + approach-persona ablation (2026-06-10)

> **데이터셋 전환**: BigMath → **NuminaMath-CoT_filtered** (`numina_cot`, 협업자 main merge). gold가 일관 `\boxed{}`/숫자/객관식이라 BigMath의 형식 불일치(채점기 raw-LaTeX 버그) 문제가 구조적으로 줄어듦. gold 필드=`ground_truth`.
> seed09(verbal-RL, exclusive_solves)를 베이스로, **persona 설계만** 비교하는 ablation.

### 설정 (10 vs 11 — persona 설계만 다름)
| | seed20210010 (control) | seed20210011 (treatment) |
|---|---|---|
| 공통 | \multicolumn{2}{c}{NuminaMath · 교정 scorer · exclusive_solves scout · Thinking OFF · 300×5 · tp=1} |
| scout 출력 | V2: `{persona_name, system_prompt, strengths}` | **V3**: `{persona_name, system_prompt(정체성 1줄), approach}` (strengths 없음) |
| 생성 프롬프트 | system=persona | system=identity, **approach→user 턴 주입** |
| 라우터가 보는 것 | name + strengths | name + identity + **approach** |

### 🐛 이 분기에서 잡은 버그 2개 (둘 다 결과를 한 번씩 무효화시킴)
1. **math/coding 분기 버그** (`a012e49`): 생성·scout이 **dataset 이름**(`("bigmath","math")`)으로 분기 → `numina_cot`가 코딩으로 빠져 모델이 **Python 코드** 생성(\boxed 0%, ~7%). **domain 기반 분기**로 수정(7곳). seed11 V3도 이때문에 안 돌다가 수정 후 작동(approach 6/6).
2. **eval stale-reuse 함정** (`b374dea`): `run_inference` resume 로직이 옛 무효 런의 출력(500/500)을 "이미 처리됨"으로 보고 **생성 skip → 옛 결과 재사용**. 진화는 정상, eval만 무효. 옛 출력 삭제 + 스크립트에 생성 전 `rm -f` 추가로 해결. (교훈: 절대 수치 믿기 전 raw 출력·\boxed 확인.)

### 결과 (깨끗한 데이터, \boxed 100%)
| 지표 | **seed10 (control, V2)** | **seed11 (approach-cue, V3)** |
|------|----|----|
| MoE Pass@1 평균 | **67.6%** (68.6/67.0/67.0/67.4/68.2) | 63.9% (63.6/64.2/63.6/63.4/64.8) |
| LUCA 단독 | 66.0% | 66.6% |
| **Test UB union** | **74.8%** (8명) | 72.8% (7명) |
| specialist가 LUCA 너머 | **+44** | +31 |
| 아무도 못 풂 | 25.2% | 27.2% |

### 판정: approach-cue는 효과 없음 — 오히려 해로움
- seed11이 **모든 지표에서 control보다 낮음** (MoE −3.7, UB −2.0, 고유기여 +44→+31).
- 결정적: **seed11 MoE(63.9%) < 자기 LUCA(66.6%)** — approach-persona로 라우팅하는 게 generic LUCA보다 못함. control은 LUCA +1.6.
- → 정체성+절차적 접근법을 user에 주입하면 전문가가 **더 경직**돼 성능 하락. **가설 반증.**
- **메타**: control도 MoE 67.6 vs LUCA 66.0 = **+1.6pp뿐** — BigMath와 동일 패턴(prompt-level 전문화가 generic 대비 미미). NuminaMath에서도 전문화 천장 재확인.
- **caveat**: 각 arm n=1(seed 1개) → noise 가능성 배제 못 함. 절대 수치는 협업자 **vanilla baseline**과 대조 필요. 협업자 전달용 로스터: `results/numina_cot/seed20210010/roster_final.json`.

---

## 20210012 — hole-aware swap 로직 (seed10 분기, 교수님 피드백 #3) (2026-06-15)

> seed10(control)에서 **gate 로직만** 바꾼 깨끗한 A/B. persona·scout·데이터·하이퍼파라미터(λ=0.05) 모두 seed10과 동일.

### 동기 (교수님 피드백 #3)
swap 시 worst의 marginal loss로 생긴 "빵꾸"(worst만 풀던 niche)를 newface가 채울 수 있는지(worst∩newface)를 고려해야 함. (#1 generalist 시작점 비교=후순위, #2 2-phase=현행 동의로 변경 없음.)

### 🔴 결함
`hard_errors`(전원 못 푼 문제)와 `worst_unique`(worst만 푼 문제)는 **정의상 서로소** → newface는 `hard_errors`로만 probe되어 `new_pass ∩ worst_unique = ∅` **항상**. 즉 기존 gate는 swap이 worst의 niche를 날리는지 **측정조차 안 함**(독립 두 phase가 우연히 동시에 켜진 게 swap).

### ✅ 수정 (`754ea40`)
- probe 집합 → `hard_errors ∪ worst_unique` (newface를 worst niche에도 테스트).
- swap 후보(phase1∧phase2)일 때 **niche 전량 회수면 swap, 아니면 add로 강등**(worst 유지). `gh_add`는 hard_errors만 intersect → **phase1(add) 독립 보존**(보수적 veto만 추가).
- `evolution_log`에 `worst_unique_n/niche_recovered_n/swap_demoted_to_add` 기록.

### ⚠️ magnitude — **(2026-06-16 정정)**
실측 적용값 `scale=0.25, λ=0.05`(base.yaml; numina override 안 함, seed12 로그 역산으로 확정) → lambda_del≈0.013~0.018 → **worst는 고유 solve = 0일 때만 evict** → 보호 niche = **0** → hole-aware demote **구조적으로 발동 불가(INERT).** 이전 "niche≤1 / λ 낮추면 load-bearing"은 scale=0.5 가정한 **오류**. 발동시키려면 방향 반대로 **scale↑**(0.5→unique≤1, 1.0→≤2~3).

### 검증 / 결과 (jobs 182646/7/8, 완료)
- 단위테스트 6/6 PASS, numina smoke COMPLETED(크래시·오분류 없음, niche 로그 정상).

| 지표 | **seed12** | seed10(참고) |
|---|---|---|
| MoE 평균/최고/최종 | 67.4 / 68.8 / 67.4 | 67.6 / 68.6 / 68.2 |
| LUCA 단독 | 67.6 | 66.0 |
| UB union | 75.8% (7명) | 74.8% (8명) |
| specialist가 LUCA 너머 | +41 | +44 |

- **⚠️ seed10↔12 절대비교 위험**: eval/UB가 `--seed`로 held-out 500을 뽑아 두 seed가 **서로 다른 500문제**. 증거: 동일 LUCA인데 단독 66.0 vs 67.6(+1.6pp 테스트셋 노이즈). MoE·UB 차이는 노이즈 밴드 안 = **사실상 동률.**
- **hole-aware 0번 발동**: evolution_log 후보 **0/30**, swap 0, demote 0(add 6/noop 24, 로스터 2→7). 로직 휴면 → seed12 ≈ seed10 재추첨. **현 shared 면제 체제에선 gate 변경 무력**을 수치로 재확인.

### 🔑 후속: eviction 안 도는 원인 분리 = shared 기여 면제 (단독 킬러)
git+로그 교차분석으로 eviction 후보 발생을 seed별 집계:

| seed | max_lives | all-zero 면제 | shared 면제 | 후보존재 | del+swap |
|---|---|---|---|---|---|
| 01 | 3 | ✗ | ✗ | 6/30 | 6 (cascade) |
| 02 | 5 | ✗ | ✗ | 7/28 | 7 |
| 03 | 5 | ✗ | ✗ | 2/30 | 2 |
| **04** | 5 | ✓ | ✗ | **3/30** | **3** ← all-zero만: 정상 |
| **05** | 5 | ✓ | ✓ | **0/30** | **0** ← shared 추가: 즉사 |
| 06/08/09/10/12 | 5 | ✓ | ✓ | 0 | 0 |

→ **shared 면제(`356813b`, seed05)가 단독 킬러. max_lives 무죄**(seed02/03/04가 5에서 정상 도태). 수학 고겹침에서 누구나 공유문제를 풀어 lives가 영구 만렙 → WAR 신호 단락.

### → 20210013 = seed10 + shared 면제 OFF (도태 부활), commit `e4cc9e2`
- **config 토글** `shared_contribution_exemption`(기본 True=기존동작 보존), seed13만 False(= 검증된 seed04 체제). all-zero 면제·max_lives 5 유지. lives 분기 시뮬 5/5 PASS.
- **결과(2026-06-16)**: 도태 **부활**(후보 6/30, delete 5+swap 1, 로스터 2→5). 단 evict된 worst는 **전부 unique=0** → `swap_demoted_to_add` **0** (hole-aware 여전히 INERT — scale=0.25). MC-aware MoE ~77.3% / UB 81.4%(5명). vs seed12(7명, UB 83.4%) → **도태로 lean하지만 UB ~1.4pp 손해**.

---

## 20210014 / 20210015 — scout V1(atomicity) × shared on/off (2×2 완성) (2026-06-16)

> **동기**: seed10/12/13은 모두 **V2 scout**(atomicity 규칙 없음) → persona가 `&` 묶음형(예: "Geometry **&** Trig"). BigMath ablation에선 **V1(atomicity, seed08) > V2(seed09)** (UB +20 vs +14)였으나 **numina에선 V1 미검증**. V3(seed11, approach)는 이름만 깔끔·속은 묶음·성능 패(반증). → numina에 **V1 복원**해 2×2 완성.

### scout 버전 정체 (코드 확인)
| | atomicity 규칙 | exclusive_solves | approach 분리 | persona |
|---|---|---|---|---|
| V1 | ✓ | ✗ | ✗ | 단일 도메인(깔끔, 'and' 0%) |
| V2(seed09~13) | ✗ | ✓ | ✗ | "&" 묶음 |
| V3(seed11) | ✗ | ✓ | ✓ | 이름만 깔끔, 속은 묶음 |
- 어떤 버전도 **도메인명 prior 안 줌**(도메인은 모델이 hard_errors서 선택). V1의 유일 prior = **구조적 atomicity**(도메인 무관).

### 2×2 설계
| | shared ON | shared OFF |
|---|---|---|
| **V2** | seed12 (UB 83.4, 7명) | seed13 (UB 81.4, 5명) |
| **V1** | **seed14** | **seed15** |
- seed14/15: numina + V1 + 백본 31B(MoE 아님) + Thinking OFF + max_lives 5. shared만 ON/OFF. config/submit `numina_train_seed14/15.yaml`. 제출 183264 / 183267(R).
- 세로축 = atomicity 효과, 가로축 = 도태 효과. **모든 수치는 MC-aware 채점 기준으로 비교.**

### 🔴 MC 채점 아티팩트 (+~11pp) — 모든 numina 수치에 적용
- numina test 객관식 30%의 gold가 **보기문자↔값 혼재** → 박싱 포맷만 안 맞으면 FAIL. seed12 **67.4→78.0%(+10.6pp)**, 회복 53건 전수 정당(FP 0).
- 수정: scorer.py(우리, `6f4ac55`) + metrics.py/evaluate.py(협업자, `939f670` push). **추가 전용**(기존 PASS 불변).

---

## Git 이력 (jh/evolution 브랜치)

| 커밋 | 내용 |
|---|---|
| `b220fbb` | BigMath evolution 초기 구현 (scorer, router, prompt, sbatch) |
| `309a564` | 수학 프롬프트 및 라우터 코딩과 대응 정렬 |
| `9f3b436` | ATOMICITY 규칙 및 `and` 금지 추가 |
| `0085996` | All-zero WAR 배치 lives 패널티 면제 |
| `356813b` | Shared 기여 에이전트 lives 패널티 면제 (solved_any) |
| `c48f723` | Scout/Router 최소개입 (Verbal RL framing) — 인간 prior 제거 |
| `4960cbf` | 코딩 Scout/Router verbal RL 간소화 + roster 테이블 정리 |
| `3f3edb7` | Action gate lambda batch_size 정규화 (batch_size_ref=50) |
| `ebccef5` | squad_solves 로깅 추가 (WAR 기반 expert 라벨링용) |
| `8730c9a` | seed20210007 설정 — 50K BigMath, tp=4, 1 epoch |
| `8fe16f1` | seed20210008 — Thinking OFF ablation, tp=2 병렬화, eval 자동화 |
| `26c6f8a` | seed20210009 — data-driven scout with exclusive solve history |
| `a310b01` | **fix(eval)**: math_verify_score raw LaTeX false-negative 수정 (+16pp); `fix/math-verify-scorer` 브랜치 |
| `397878d` | approach-persona(V3) 진화로직 + NuminaMath seed10/11 풀세트 래퍼 |
| `0b4a0cf` | main(NuminaMath loader/eval) 머지 |
| `a012e49` | **fix**: math/coding 분기를 dataset 이름 → domain 기반으로 (numina_cot 오분류 수정) |
| `b374dea` | **fix(eval)**: 생성 전 `rm -f`로 stale 출력 재사용 방지 |
| `ebf97e6` | math 생성 프롬프트 `\boxed{}` 전환 |
| `754ea40` | **feat(gate)**: hole-aware swap — worst niche를 newface가 회수 못 하면 swap→add 강등 (교수님 피드백 #3); seed20210012 A/B |
| `e4cc9e2` | **feat(gate)**: shared 기여 면제 토글(`shared_contribution_exemption`, 기본 True) + seed20210013 (면제 OFF=도태 부활) |
| `6f4ac55` | **fix(scorer)**: MC-aware math scoring — 객관식 보기↔값 동치 인정 (numina gold 포맷 +~11pp). 추가 전용 |
| `939f670` | **fix(eval)**: 협업자 경로(metrics.py+evaluate.py) MC-aware `math_verify_mc_score` (fix/math-verify-scorer, **origin push**) |
| *(config)* | seed20210014/15 — numina scout V1(atomicity) × shared on/off (2×2 완성), 백본 31B |
