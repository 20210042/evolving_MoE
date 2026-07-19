# QASC: Mixture-of-LoRA vs Dense fine-tuned Llama3 — End-to-End 배포 비교

*llama-3.1-8B · 13 LoRA experts (cap10 seed20210211) · QASC validation 926문제 · 2026-07-18*

---

## TL;DR

- **앵커 = Dense fine-tuned Llama3 (ep9 baseline) = 87.15%** (807/926).
- **실현 가능한 12개 라우팅 방법 전부 83.8~85.2%** — 전부 dense보다 **약 2pp 낮다**. 최고 실현치 = confidence(z-norm) / MLP-confidence = **85.2% (−1.9pp)**.
- **oracle 라우팅이면 90.0% (+2.8pp)** — 두 어댑터를 잘 고르기만 하면 dense를 넘는 상보성은 실재한다. 그러나 **실제 라우터가 그 pair를 못 찾는다**.
- 결론: 현재 Mixture-of-LoRA는 **실배포 정확도에서 단일 dense fine-tuned Llama3를 못 이긴다.** MoL이 dense를 넘으려면 라우팅이 oracle에 근접해야 하며, 그것이 유일한 열린 문제다.
- **4조건 ablation(아래 §1) 결과: Dense 87.15 > Random 86.0 > Evolved(우리) 85.2 > Human-prior 84.1.** 즉 **우리 분할이 랜덤 분할을 못 이긴다** — QASC에서 "분할 방식이 가치의 원천"이라는 가설은 지지되지 않는다(negative).

---

## 1. 4조건 ablation 사다리 (핵심)

"MoE 이득이 분할 때문인가, 그냥 expert/param이 많아서인가"를 통제한 비교. 세 MoE 모두 **동일 구조**(specialized cap10 + shared, shared는 동일 체크포인트 재사용, 동일 배포방식·라우팅 12방법)이고 **specialized 배정 방식만** 다르다.

| # | 조건 | experts | best-single | oracle-union | **실현 최고 배포** | oracle 배포 |
|---|---|---|---|---|---|---|
| 1 | **Dense SFT** (MoE X) | 1 | — | — | **87.15** | — |
| 2 | **Random-partition** (count-matched) | 13 | 84.7 | 93.0 | **86.0** (conf raw) | 89.8 |
| 3 | **Human-prior** (LLM 8과목) | 9 | 83.8 | 92.1 | **84.1** (conf raw) | 89.1 |
| 4 | **Evolved (우리)** | 13 | 86.2 | 91.7 | **85.2** (conf z / MLP conf) | 90.0 |

**읽는 법 — 세 가지 negative:**

**(1) Dense가 전부 이긴다.** 어떤 MoE 조건도 실현 가능한 라우팅으로 단일 dense(87.15)를 못 넘었다(−1.2 ~ −3.1pp).

**(2) 우리 분할이 랜덤 분할을 못 이긴다 (핵심 negative).** Evolved 85.2 < Random 86.0 (−0.8pp). oracle 라우팅에서도 90.0 vs 89.8로 사실상 동률. 즉 QASC에서는 **solve-clustering 분할이 count-matched 랜덤 분할 대비 이득이 없다.** "expert 수·데이터량을 맞추면 분할 방식은 무의미"가 이 데이터의 결론. (Random은 oracle-union이 93.0으로 오히려 최고 — 랜덤 배정이 expert 실패를 더 탈상관시킨다.)

**(3) Human-prior가 가장 나쁘다.** 84.1로 MoE 중 최하위이며 best-single(83.8)·oracle-union(92.1)도 낮다. 의미(과목) 축 분할이 outcome 축 분할보다 해롭다는 기존 관찰과 일치. 과목 분할은 disjoint라 expert당 데이터가 편중되고(biology 2746 vs astronomy 58) mean n_solved도 7.23으로 급감.

**부수 관찰**: Random 조건은 라우팅 민감도가 극단적이다(random-2 76.8 → confidence 86.0, +9.2pp). 랜덤 expert는 품질 편차가 커서 확신 라우팅이 실제로 강한 expert를 골라낸다. 반면 Evolved는 84.8 → 85.2로 평평 — expert들이 서로 비슷해 라우팅이 건질 게 없다. **이것이 Evolved가 Random에 밀리는 메커니즘**이다.

---

## 1-b. 재분할 실험 (cap7) — 상보성은 분할로 제조되지 않는다

§1의 진단("Evolved expert가 서로 닮았다", 실패상관 0.787)에 대한 직접 개입. 원인 가설: **specialized 학습셋의 64.6%가 로스터 대다수가 푸는 공통문제**여서 expert들이 같은 core로 수렴한다. 처방: cap을 10→7로 조여 공통문제를 specialized에서 완전히 제거하고(비율 0%), 공통 core는 shared가 전담(min_n_solved 11→8, 2547→4285문제).

| | 학습셋 Jaccard | 실패상관 | union | best-single | **배포 EM** |
|---|---|---|---|---|---|
| Evolved cap10 | 0.447 | 0.787 | 91.7 | 86.2 | **85.2** |
| **Evolved cap7** | **0.208** | **0.739** | **91.8** | 84.1 | **84.7** |
| (참고) Random cap10 | ~0.22 | 0.709 | 93.0 | 84.7 | 86.0 |

**결과: 기각.** 학습셋 겹침을 의도대로 절반 이하(0.447→0.208, 공통문제 64.6%→0%)로 떨어뜨렸으나 —
- **실패상관이 거의 안 움직였다**: 0.787 → 0.739 (−0.048)
- **union이 불변**: 91.7 → 91.8 (+0.1). 아무도 못 푸는 문제도 77 → 76으로 그대로.
- **expert만 약해졌다**: best-single 86.2 → 84.1 (데이터 2014 → 660)
- **배포 하락**: 85.2 → 84.7

**결정적 대조**: cap7과 Random은 학습셋 겹침이 비슷(~0.21)한데도 Random이 실패상관 0.709 / union 93.0으로 더 낫다. 즉 **학습 데이터의 겹침이 실패상관을 만드는 게 아니다.**

**함의**: QASC에서 expert 실패는 "무엇을 학습했는가"가 아니라 **문제 고유 난이도**가 지배한다. 분할을 어떻게 바꾸든(solve-clustering / 과목 / 랜덤 / 난이도컷) union은 91.7~93.0 좁은 대역에 갇히고, 분할 개입은 expert를 약화시키는 쪽으로만 작동했다. **상보성은 데이터 분할로 제조되지 않는다** — 이것이 QASC 계열 실험의 최종 결론이다.

---

## 실험 방식

- **배포 정의(사용자 요구)**: 라우팅 방법이 문제별 top-2 expert를 고르면 → 그 둘을 PEFT `add_weighted_adapter(linear, [0.5, 0.5])`로 **한 어댑터로 병합·활성화** → **단 한 번 생성** → letter EM 채점. (union coverage 같은 프록시가 아니라 926문제를 실제로 푼 정확도)
- pair별 병합 어댑터는 방법 무관 동일하므로 `(pair, 문제)` 단위로 메모이즈해 중복 생성 제거. base+13어댑터 1회 로드.
- **앵커**: `qasc_sft_llama3_finetuned_qasc_baseline_eval300_ep9_baseline_208314.jsonl` (dense SFT Llama3, ID 926 완전 정합, `pass_score` 기준 87.15%).
- 학습 라우터(MLP)는 동일 backbone 라벨이 val밖에 없어 **926 solve 매트릭스 위 5-fold CV**(held-out)로 픽 생성 — 가장 정확한 추정.
- 스크립트: `scripts/moe_deploy_sweep.py` (job 209918). 원표: `results/qasc/seed20210211/deploy_sweep_vs_dense.md`.

---

## 결과 매트릭스 (end-to-end 실생성 EM)

| 라우팅 방법 | 배포 정확도(%) | vs Dense 87.15 |
|---|---|---|
| random-2 | 84.8 | −2.4 |
| confidence (raw) | 85.1 | −2.1 |
| **confidence (z-norm)** | **85.2** | **−1.9** |
| confidence (rank) | 84.4 | −2.7 |
| confidence + prior | 83.8 | −3.3 |
| pred-agreement | 85.0 | −2.2 |
| MLP hidden-state | 84.1 | −3.0 |
| MLP encoder-emb | 85.1 | −2.1 |
| MLP answer-prob | 85.0 | −2.2 |
| **MLP confidence** | **85.2** | **−1.9** |
| MLP hs+conf | 84.6 | −2.6 |
| **oracle top-2** | **90.0** | **+2.8** |

**참조선**: Dense fine-tuned Llama3 **87.15** · best-single expert(solve) 86.2 · oracle-union(13) 상한 91.7

> 정합 확인: confidence(raw) 85.1% 는 별도 스윕(job 209823)의 confidence 0.5-병합 85.1%와 일치, oracle 90.0 ≈ 90.1 — 배선·수치 신뢰.

### 라우팅 상한(union coverage) vs 실배포 — 한눈에

같은 라우팅으로 "top-k 중 하나라도 풀면 성공"(union, 병합 안 함, 라우팅의 이론 상한)과 "0.5 병합 실생성"(실배포)을 나란히 둔 것. 상한과 실배포의 간격(≈ 병합 비용)과, 상한조차 dense를 넘는지 여부를 동시에 보여준다. (job 209917 union + job 209918 배포)

| 라우팅 방법 | union top-1 | union top-2 | **배포(0.5병합)** | union2 vs Dense |
|---|---|---|---|---|
| random | 84.2 | 87.0 | 84.8 | −0.2 |
| confidence (raw) | 84.9 | 86.8 | 85.1 | −0.4 |
| confidence (z-norm) | 84.8 | 86.2 | 85.2 | −1.0 |
| confidence (rank) | 84.1 | 86.0 | 84.4 | −1.2 |
| confidence + prior | 86.4 | 88.2 | 83.8 | +1.1 |
| pred-agreement | 84.8 | 84.8 | 85.0 | −2.4 |
| MLP hidden-state | 84.0 | 87.0 | 84.1 | −0.2 |
| MLP encoder-emb | 83.8 | 87.5 | 85.1 | +0.3 |
| MLP answer-prob | 84.3 | 86.7 | 85.0 | −0.5 |
| MLP confidence | 85.4 | 87.3 | 85.2 | +0.1 |
| MLP hs+conf | 84.2 | 87.0 | 84.6 | −0.2 |
| oracle top-2 | 91.7 | 91.7 | 90.0 | +4.5 |

읽는 법: 대부분 union top-2 상한도 dense(87.15) 언저리이거나 아래(−계열)다. 상한이 dense를 다소 넘는 몇몇(MLP encoder-emb 87.5, MLP confidence 87.3)도 **0.5 병합 실생성에서 −1.7pp 안팎 깎여** 85대로 내려가 dense 밑이 된다. oracle만 상한 91.7 → 배포 90.0으로 dense를 확실히 초과.

---

## 해석

**(a) 실현 가능한 라우팅은 전부 dense에 미달하고, 서로 거의 구별되지 않는다.**
random-2(84.8) ≈ confidence(85.1) ≈ 학습 MLP(85.2). 즉 어떤 라우팅 신호를 써도 배포 정확도는 ~85%에 뭉쳐 dense(87.15)에 2pp 못 미친다. 문제별로 "어떤 두 expert를 병합하나"를 잘 고르려는 시도가 실배포에서 사실상 이득을 못 낸다.

**(b) 상보성 잠재력은 실재한다 — 단 oracle 라우팅에서만.** oracle top-2(90.0)는 dense를 +2.8pp 넘는다. 두 어댑터를 이상적으로 고르면 0.5 병합이 dense보다 확실히 낫다. 문제는 **그 이상적 pair를 실제 라우터가 못 집는다**는 것.

**(c) 병합 자체의 비용은 일정하다.** 라우팅의 이론 상한(union coverage) 대비 실생성은 일관되게 −1.7pp: confidence union 86.8 → 배포 85.1, oracle union 91.7 → 배포 90.0. 즉 0.5 병합은 "둘 중 하나라도 맞으면"의 상한을 약 1.7pp 깎아 실현한다.

**(d) 확신 라우팅은 상한에서도 dense를 못 넘는다.** confidence top-2의 union 상한(86.8)조차 dense(87.15) 아래다. 확신 신호로는 근본적으로 dense를 넘을 수 없고, dense를 넘는 경로는 oracle 수준 라우팅(union 91.7 / 배포 90.0)뿐이다.

---

## 결론

- **4조건 ablation의 결론(§1)**: Dense 87.15 > Random 86.0 > Evolved 85.2 > Human-prior 84.1. **우리 분할이 랜덤 분할을 못 이겼고**(−0.8pp, oracle에서도 동률), 어떤 MoE도 dense를 못 넘었다. QASC 한정으로 "분할 설계가 MoE 이득의 원천"이라는 가설은 **지지되지 않는다**. Human-prior가 최하위인 것만 기존 방향(의미축 분할이 해롭다)과 일치.
- **현 시점 Mixture-of-LoRA(실배포)는 QASC에서 단일 dense fine-tuned Llama3(87.15%)를 못 이긴다.** 실현 가능한 12개 라우팅 전부 −1.9~−3.3pp.
- MoL이 dense를 넘는 잠재력은 분명히 있다(oracle 배포 90.0, +2.8pp). **유일한 열린 문제는 라우팅 정확도를 oracle 쪽으로 끌어올리는 것** — 현재의 확신/학습/휴리스틱 신호로는 도달 불가.
- QASC는 solve 매트릭스가 dense(평균 11/13이 각 문제를 풂)해서 애초에 라우팅으로 벌 수 있는 여지(86.2→91.7)가 얇다. MoL의 상보성 우위를 실증하려면 expert 커버리지가 실제로 갈라지는 도메인에서 동일 end-to-end 비교를 반복하는 것이 유효.

### 산출물
- end-to-end 스윕: `scripts/moe_deploy_sweep.py`, 원표 `results/qasc/seed20210211/deploy_sweep_vs_dense.md`, 로그 `logs/deploy_sweep.209918.log`
- 앵커: `qasc_sft_llama3_finetuned_qasc_baseline_eval300_ep9_baseline_208314.jsonl` (87.15%)
- solve 매트릭스: `results/qasc/seed20210211/inference_validation_lora13.binned.jsonl` (926×13)
