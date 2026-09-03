# 로스터 16명 공식 지표 재채점 — SNI test 8,699 (재생성 0)

저장된 출력 417,552건(8,699문제 × 16명 × K=3)을 `sni_metrics`로 다시 쟀다. 결측 0칸은 0으로 뒀다.
협업자 수치는 **단일 모델 · 단일 생성**이다. 아래에서 같은 층은 `rep 0` 열뿐이고, UB는 16명을 다 돌려본 뒤 최선을 고른 오라클이므로 같은 층이 아니다.

## 0. 협업자 기준선과 나란히

| 모델 | EM | ROUGE-L | 층 |
|---|---:|---:|---|
| Dense Llama (ckpt-4350) | 55.63 | 68.93 | 단일 모델 · 단일 생성 (llama student) |
| 4x1 MoE (ckpt-4000) | 56.24 | 69.47 | 단일 모델 · 단일 생성 (llama student) |
| gemma teacher — best-single (EM 기준: Strict Formalism & Minimalist Output) | **56.91** | 69.31 | 단일 모델 · 단일 생성 |
| gemma teacher — best-single (ROUGE 기준: Strict Formalism & Minimalist Output) | 56.91 | **69.31** | 단일 모델 · 단일 생성 |
| gemma teacher — 16명 평균 (무작위 배정) | 50.40 | 62.80 | 단일 생성 |
| **UB (16명 오라클, K=1)** | **67.08** | **80.10** | ⚠️ 오라클 — 같은 층 아님 |
| **UB (16명 × K=3 오라클, 48회)** | **68.24** | **81.34** | ⚠️ 오라클 — 같은 층 아님 |

## 1. expert별

| expert | EM (rep 0) | ROUGE-L (rep 0) | EM (K=3 평균) | ROUGE-L (K=3 평균) |
|---|---:|---:|---:|---:|
| Strict Formalism & Minimalist Output | 56.91 | 69.31 | 56.86 | 69.29 |
| Strict Constraint & Boundary Adherence | 56.83 | 69.00 | 56.71 | 68.96 |
| Verbatim Integrity & Structural Fidelity | 56.41 | 68.97 | 56.35 | 68.94 |
| Constraint-Driven Grounding & Symbolic Precision | 56.36 | 68.87 | 56.34 | 68.81 |
| Precision-Targeted Span Minimization | 56.08 | 68.49 | 56.13 | 68.53 |
| Extensionality vs. Intensionality Enforcement | 56.01 | 68.44 | 56.01 | 68.41 |
| Literalist Extraction & Span Fidelity | 55.97 | 68.65 | 55.96 | 68.62 |
| Information Density & Entropy Control | 55.73 | 68.12 | 55.82 | 68.20 |
| Strict Contextual Containment | 55.04 | 67.21 | 54.84 | 67.09 |
| Semantic Divergence Enforcement | 54.83 | 65.92 | 54.89 | 65.98 |
| Extrapolative Hallucination Suppression | 54.45 | 66.59 | 54.39 | 66.49 |
| LUCA | 53.89 | 65.74 | 53.84 | 65.73 |
| Temporal-State Sentiment Sensitivity | 53.66 | 65.73 | 53.70 | 65.73 |
| Counter-Intuitive Intent Realization | 37.15 | 50.25 | 37.22 | 50.33 |
| Negative Constraint & Counter-Factual Logic Enforcement | 25.45 | 38.80 | 25.50 | 38.84 |
| Adversarial Goal Alignment | 21.54 | 34.70 | 21.64 | 34.71 |

## 2. 헤드룸

| | EM | ROUGE-L |
|---|---:|---:|
| best-single (각 지표 기준 최고) | 56.91 | 69.31 |
| UB K=1 (16명 오라클) | 67.08 | 80.10 |
| **헤드룸** | **+10.16pp** | **+10.79pp** |

## 3. 우리 이진 판정과의 관계

우리 진화·비닝이 쓴 통과 기준은 **EM==100 or ROUGE-L>70**이다. 위 표의 EM·ROUGE는 그 임계를 적용하지 않은 공식 지표 원값이라 서로 다른 숫자다.
참고: rep 0에서 이진 통과율 best-single은 62.71, 16명 오라클은 73.84.

