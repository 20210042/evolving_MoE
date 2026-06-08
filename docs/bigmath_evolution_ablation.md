# BigMath Evolution Ablation Study

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
