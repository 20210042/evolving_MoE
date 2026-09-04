# 분할 빌드 — lbox 진화 10명 (학습 τ=8, 센트로이드 tc=4 가중, feature=embed_viz)

- train 46,019문제 · 개별 14,707(32.0%) · shared 11,074(24.1%) · 전원실패 20,238(44.0%)
- 최종 학습셋 평균 쌍 Jaccard **0.276** (최대 0.634)
- 전원실패 센트로이드 1·2위 차 < 1e-3: 202건 (1.0%)
- ⚠️ 전원실패 몫은 임베딩으로 배정했으므로 정의상 입력에서 예측 가능하다. 라우터 평가 시 갈림 몫과 분리해 보고할 것.

| expert | 합계 | 개별(n_solved≤τ) | 전원실패(센트로이드) |
|---|---:|---:|---:|
| Legal Case Typology Architect | **14,326** | 9,415 | 4,911 |
| Judicial Labeling Precisionist | **10,587** | 7,239 | 3,348 |
| Statutory Element Matcher | **10,362** | 5,417 | 4,945 |
| Legal Nomenclature Purist | **9,326** | 8,999 | 327 |
| Judicial Precedent Classifier | **9,155** | 8,758 | 397 |
| Legal Fact Synthesis Engine | **8,882** | 7,084 | 1,798 |
| Civil Dispute Taxonomy Expert | **7,462** | 7,009 | 453 |
| Legal Recidivism Analyst | **6,450** | 5,123 | 1,327 |
| Criminal Charge Aggregator | **6,253** | 4,936 | 1,317 |
| Legal Provision Auditor | **5,627** | 4,212 | 1,415 |
| **shared expert** | **11,074** | 11,074 | — |

전원실패 배정 쏠림: 최다 4,945 / 최소 327 (균등이면 2,023)

## 센트로이드 진단

`응집도` = L2 정규화 **전** 센트로이드 norm — 그 expert가 맡은 문제들이 임베딩 공간에서 얼마나 뭉쳐 있나(흩어지면 상쇄돼 0에 가까워진다).
L2가 이 값을 1로 만들어 버리므로, 흩어진 expert의 잔차 방향이 뾰족한 expert와 동등하게 경쟁하게 된다.

| expert | 응집도(정규화 전 norm) | 유효표본(ESS) | 다른 센트로이드와의 평균 코사인 | 전원실패 배정 |
|---|---:|---:|---:|---:|
| Statutory Element Matcher | 0.3611 | 831 | +0.0707 | 4,945 |
| Criminal Charge Aggregator | 0.2980 | 1,431 | -0.0680 | 1,317 |
| Judicial Labeling Precisionist | 0.2682 | 2,081 | -0.1315 | 3,348 |
| Legal Provision Auditor | 0.2080 | 1,119 | -0.0440 | 1,415 |
| Legal Recidivism Analyst | 0.1823 | 689 | +0.0556 | 1,327 |
| Legal Fact Synthesis Engine | 0.1771 | 1,408 | -0.0875 | 1,798 |
| Legal Case Typology Architect | 0.1660 | 2,241 | -0.0366 | 4,911 |
| Judicial Precedent Classifier | 0.1371 | 1,958 | +0.0073 | 397 |
| Legal Nomenclature Purist | 0.1267 | 2,132 | +0.0990 | 327 |
| Civil Dispute Taxonomy Expert | 0.1264 | 1,442 | +0.0273 | 453 |

응집도 ↔ 전원실패 배정 수 상관 **r = +0.593** (음수면 '흩어진 expert가 어려운 문제를 빨아들인다'는 가설이 맞다)
센트로이드 쌍 코사인: 평균 -0.0108 · 최소 -0.7775 · 최대 +0.9168

산출: `export/lbox_split_seed20210311/split.jsonl`
