# SNI — 다음 에이전트 인수인계 (2026-08-30, Human 조건 추가 후 갱신)

브랜치 `jh/sni-probe`. 대상은 **진화 로스터 16명 seed20212003** 하나뿐이다.
프로브 25명 시절 산출물은 전부 삭제했다(아래 §완료 참조).

---

## 지금 상태 한 줄

분석은 끝났고 협업자 인계 패키지 **3조건(Ours·Random·Human) 전부 준비됐다**.
**teacher 쪽에 남은 작업은 없다. 나머지는 student 학습 이후에만 나오는 수치들이다.**

---

## 남은 작업

### 1. Human prior split 패키지 — **완료 (2026-08-30)**

사용자 결정: **순수 BTX식**. (구조 맞춤안은 만들었다가 철회했다)

- 빌더 `scripts/sni_build_split_human.py` → `export/sni_split_human/split.jsonl`
- 패키지 `scripts/sni_export_moe_package.py --arm human` → `export/sni_moe_human/`
- **shared expert 없음(16 routed only) · 문제당 1명 · 중복 없음 · 학습 row 69,588**
  (Ours·Random의 104,612와 다르다. 중복 배정이 없으니 당연하고, 이 조건의 정의다)
- 진화 쪽 n_solved 구간(indiv/shared/all_fail)은 **쓰지 않았다** — teacher가 만든 사실이라
  사람 사전지식 조건에 넣으면 그 축이 새어 들어온다. `kind`는 전부 `btx`.
- ⚠️ 상위 16 category만 쓰는 안은 기각했다 — top-16이 68.0%라 나머지 56개 22,236건(32%)이
  버려져 다른 조건과 데이터량이 달라진다. **72개 category를 겹치지 않는 16그룹으로 묶었다**
  (크기 내림차순 그리디 LPT). **category는 한 번도 쪼개지지 않았다**(실측 0건).
- expert별 학습량 12,499(QA 단독) ~ 3,500. 균등이면 4,349. 최대 category가 통째로 한 명에게
  가서 생기는 편차이고 **사람 택소노미의 성질이라 보정하지 않는다**.
- 검증: 프롬프트·타깃·메타 불일치 0건(69,588 전수 대조), id 집합 Ours와 동일,
  문제 중복 0, `test.jsonl` 세 조건 md5 동일.

### 2. student 학습 이후에만 나오는 수치 (협업자 영역)

teacher 쪽에서는 더 잴 것이 없다. 학습해야 나온다.

| 조건 | 데이터 | 상태 |
|---|---|---|
| Dense | 분할 없음 | 협업자 완료 — EM 55.63 / ROUGE-L 68.93 (ckpt-4350) |
| Sparse upcycling | 분할 없음 | 협업자 완료 — EM 56.24 / ROUGE-L 69.47 (4x1 MoE, ckpt-4000) |
| Ours | `export/sni_moe_seed20212003/` | **패키지 완료**, 학습 대기 |
| Random split | `export/sni_moe_random/` | **패키지 완료**, 학습 대기 |
| Human prior split (순수 BTX) | `export/sni_moe_human/` | **패키지 완료**, 학습 대기 |

구조는 **16 routed + 1 shared**. sparse upcycling으로 llama-3.1-8B에 올린다(BTX식 병합 아님).
⚠️ Human prior 조건만 **16 routed, shared 없음** — 순수 BTX라 shared expert가 정의상 없다.

### 3. 손대지 않기로 하고 남긴 한계

- **전원실패 배정 쏠림 2.7배** — 최다 LUCA 6,845 / 최소 Semantic Divergence 2,573.
  원인은 응집도가 아니라(r=+0.267) 센트로이드끼리 붙어 있는 쌍이다
  (Adversarial ↔ Negative +0.969, Strict Constraint ↔ Strict Formalism +0.883).
  argmax 승자독식. 해소하려면 용량 제약 배정(expert당 상한 1,139, 그리디/Sinkhorn)이 필요한데
  **적용하지 않기로 하고 넘어갔다.** 다시 열려면 `scripts/sni_build_split.py`에 상한을 넣으면 된다.
- **전원실패 26.2%는 임베딩으로 배정했다 = 정의상 입력에서 예측 가능하다.**
  라우터가 이 분할을 얼마나 되찾는지 잴 때 갈림 몫과 **반드시 나눠서** 보고할 것.
- **진화가 add-only로 돌았다** — delete 0, 16명은 수렴이 아니라 누적이다(`lives_mode` 미설정 → legacy).
  WAR 최하위 3명이 pass@1 최하위 3명과 정확히 일치한다(반전 계열). 재진화 여부는 미결.

---

## 하지 말 것 (이미 해봤거나, 통화가 틀린 것)

- **teacher 쪽에서 random split 성능을 다시 재지 말 것.** 그 수는 이미 있다 —
  16명 평균(무작위 배정) **EM 50.40 / ROUGE-L 62.80**. split.jsonl은 학습 데이터 배정 파일이라
  student를 학습해야 성능이 나온다. (이 착각으로 job 236990·236995를 낭비했다)
- 사후·입력단위 라우터를 더 튜닝하지 말 것 — 상한 검정 z=−7.36으로 구조적 한계가 확인됐다.
  다음 개선은 토큰단위·동시학습(아키텍처 MoE)에서만 온다.
- ROUGE 최근접·k-NN 투표로 전원실패를 배정하지 말 것 — 둘 다 검토 후 폐기했다(이유는 스크립트 docstring).

---

## 산출물 위치

| 무엇 | 어디 |
|---|---|
| 본 보고서 | `docs/REPORT_sni_split_to_moe.md` (368줄). §연구메시지 → §0 요약 → §8 파이프라인 순으로 읽으면 된다 |
| 반성문 | `docs/REFLECTION_sni_2026-08-29.md` |
| 진화 요약·곡선·로스터 표 | `results/sni/evolution_summary.md` · `results/sni/fig_sni_evolution_seed20212003.png` |
| 공식 지표 재채점 | `results/sni/official_metrics_test.md` |
| 분할 사다리 | `results/sni/partition_compare.md` |
| 센트로이드 비교 | `results/sni/centroid_compare.md` |
| 분할 빌드 | `results/sni/split_build.md` · `split_build_random.md` · `split_build_human.md` |
| 인계 패키지 | `export/sni_moe_seed20212003/` · `export/sni_moe_random/` · `export/sni_moe_human/` |

## 주요 수치 (전부 test 8,699, rep 0 단일 생성)

- best-single **EM 56.91 / ROUGE-L 69.31** (Strict Formalism & Minimalist Output)
- 문제별 오라클(UB) **EM 67.08 / ROUGE-L 80.10** — 헤드룸 EM +10.16pp / ROUGE +10.79pp
- 16명 평균(무작위 배정) EM 50.40 / ROUGE-L 62.80
- 이진 통과율(EM==100 or ROUGE-L>70): best-single 62.71 · 오라클 73.84
- 분할 사다리(이진): Ours 오라클 73.63 vs category 63.20 · domain 63.11 · random 63.12
  → **사람 택소노미로 쪼개는 것은 이 데이터에서 무작위와 같다**

---

## 완료한 것 (다시 하지 말 것)

상보성 분산분해 · 사후 라우터 3형태 + 상한 검정 · 축 정체(좌표 회귀) · 택소노미 대조(R²·NMI) ·
분할 사다리 · 토큰 발산 · BTX 원문 대조 · gold vs 페르소나 타깃 · τ 스윕 · 센트로이드 컷 비교 ·
분할 빌드(Ours·Random·Human) · 인계 패키지(Ours·Random·Human) · 공식 지표 재채점 · 진화 곡선·로스터 표.

**삭제 완료(약 1.75GB, 복구 불가)**: `export/sni`·`sni_v2`·`sni_v3`·`sni_xl`,
`results/sni/binning_seed20212001`, `seed20212001`(+`_failed_232104`, `_scoutcap128`),
`seed20212002`, `smoke`, 프로브 시절 docs 5건·results md 7건·스크립트 4건.
