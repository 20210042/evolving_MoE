# HANDOVER — SNI 프로브 (2026-08-22, 브랜치 `jh/sni-probe`)

## 왜 시작했나

진단: **프롬프트로 만들 수 있는 출력분포 변화가 우리가 관측하는 E2E 성능과 거의 무관하다.**
acc·QASC에서 페르소나 축은 스타일만 갈렸고(p=0.001) E2E는 0이었다.
그래서 "덜 orthogonal한 레짐"을 찾으려고 Super-NaturalInstructions로 왔다.

의도한 것: **진화를 돌리지 않고**, 축을 보고 로스터를 미리 고정한 뒤
"프롬프트만으로 결과에 영향을 주는 축이 있나"를 프로빙으로 먼저 확인한다.

---

## ⚠️ 이번 프로브 런(job 229352)의 결론은 전부 무효다

돌리긴 했다(23명 × 600문제 × K=3 = 41,400 생성, 5.5분). 결과는 "두 축 다 없음"이었으나
**설계가 null을 낳도록 되어 있었으므로 인용 금지.** 세 가지 결함:

### 1. user 턴에 태스크 정의를 넣었다 (치명적)

기존 설계(QASC/acc)는 `system = 정체성(변함), user = 출력형식 지시 + 문제`다.
그런데 SNI export를 만들 때 `instruction = Definition + Input`으로 빌드해서,
user 턴에 태스크 정의 537자가 들어갔다. 페르소나는 system 90자.

```
[system] Your expertise is natural science: biology, chemistry, ...      ← 90자
[user]   Perform the task exactly as described below.
         Output only the answer itself — ...
         You are given a question on college mathematics. You are also   ← 537자 정의
         given 4 answer options ... You should only answer with the      ← 조작·형식을
         choice letter, not the whole answer.                            ← 이미 다 지정
         Input: <문제>
```

정의가 조작과 출력형식을 완전히 못박으므로 페르소나가 개입할 자리가 없다.
**"축이 없다"가 아니라 "프롬프트를 잘못 짰다"일 수 있다.**

**고칠 방향(사용자 지시)**: user에는 **"답을 무엇으로 내라"는 지시 한 줄 + 문제**만.
태스크 정의를 통째로 주지 말 것. 정체성은 system에서만 바뀐다.
단, Definition 첫 문장이 지시가 아닌 경우가 많아(예: `"In this task, you will be given a
list of integers."`) 한 줄 지시를 어떻게 뽑을지는 미해결.

### 2. 축 단위로 표집을 잘랐다

`scripts/sni_probe_sample.py`로 category 12구역 + domain 10구역에 문제를 균등 배분했다.
**real run은 이렇게 뽑지 않는다.** 프로빙은 real run을 미리 보는 것인데 다른 것을 봤다.
사용자 지적: 공통 데이터셋을 그대로 보고, 축 단위로 어색하게 쪼개지 말 것.

### 3. 양쪽 절반이 모두 구조적으로 null

데이터에서 직접 판정(태스크 내 정답 종수 / 인스턴스 수):

| | task | items |
|---|---:|---:|
| 닫힌 답(분류형) | 442종 | 44,106 |
| 중간 | 84종 | 8,374 |
| 열린 답(생성형) | 349종 | 34,609 |

- **닫힌 태스크**: 답이 한 단어라 문체가 낄 자리가 없음 → 갈릴 수 없음(23명 중 57.7%가 바이트 동일)
- **열린 태스크**: 실제로 갈렸으나(평균 9.2종) **EM이 전원 0점**이라 분석 기여 0

즉 한쪽은 갈릴 수 없고 다른 쪽은 갈려도 못 잡는다. 뽑힌 600문제 구성은 닫힘 342 / 열림 208 / 중간 50.

### 4. 프롬프트를 사전에 보여주지 않고 잡을 올렸다 (프로세스 위반)

---

## 살아있는 것

### 배선 (재사용 가능, 회귀 테스트 통과)

진화 로직(`war.py` / `action_selector.py` / `roster.py`)은 **한 글자도 안 건드림**.
전부 `sni` 게이트 뒤 추가. 기존 도메인(acc/qasc/lbox/math/mbpp/humaneval/lcb)의
family·scoring_kind 회귀 확인, `_score_pair_partial` 디스패치도 acc·legacy 경로 유지 확인.

배선 지점: `data/loader.py` · `utils/domains.py` · `evaluation/scorer.py` ·
`prompts/coding.py`(4곳) · `prompts/baseline_prompts.py`(SNI_*) ·
`prompts/meta.py`(META_AGENT_SNI_PROMPT, MANAGER_SNI_PROMPT) · `scout.py` ·
`pipelines/baselines.py` · `pipelines/routing_inference.py`.

채점: `score_sni_item`(0/100 EM, 이진 계약) / `score_sni_item_partial`(레퍼런스 최대 ROUGE-L).
ROUGE-L은 직접 구현(공유 env 설치 금지 + 토크나이즈 통제). **CJK/태국어는 문자단위로 분기** —
공백 토크나이즈면 ROUGE가 EM으로 붕괴한다(일본어 부분중첩 90.0 vs 0.0 실측).

### 데이터셋 사실

- 공식 split은 **train/test의 category가 완전 disjoint**(cross-task generalization용) → 우리 용도에 못 씀. 합쳐서 한 덩어리로 본다: **875 task / 87,089 items**(task당 100 상한).
- 라벨 축: **category 72종**, **domain 105종**(계층 라벨 → 최상위만 쓰면 74종), Reasoning 37종(331 task는 결측), Source 243종.
- 80% 커버 필요 인원: category 26명 / domain(합침) 19명. 60%면 12명 / 10명. 꼬리가 길다.
- 두 축은 대부분 얽혀 있다: Code(category 다양성 2)·Sociology(3)·Mathematics(6)는 사실상 category와 동의어. 독립적인 건 Wikipedia(21)·Dialogue(20)·News(14) 정도.

### 채점 결함 (수정 필요, 미수정)

EM이 **맞은 답을 형식 때문에 버린다**. 채점 0인데 정답 문자열이 출력에 포함된 사례 610건:

```
gt='b'                     out='b ) 42'
gt='User Choice/Control'   out='(3) User Choice/Control'
gt='Advertising'           out='targeted advertising'
```

QASC 채점 버그와 같은 계열. **다음 런 전에 반드시 고쳐야 한다.**

---

## 미해결 — 다음 세션이 정할 것

1. **user 턴의 "지시 한 줄"을 어떻게 만들 것인가.** Definition 전문을 주면 안 되고,
   첫 문장은 지시가 아닌 경우가 많다. 라벨 집합이 정의에만 있는 분류 태스크도 있다.
2. **축을 category/domain 2개로 고정한 것 자체가 의문.** 사용자 지적: 스카우트에게
   무엇을 찾으라고 시키느냐에 따라 축은 계속 달라질 수 있다. 2개만 보는 건 근거가 없다.
3. **스카우트 지시문.** `"What kind of task specialist is missing?"`처럼 "전문가를 찾아라"로
   물으면 모델은 정체성 문구를 만들고, 정체성은 문체만 바꾼다. 그리고 문체가 세지면
   태스크 지시를 침범해 오히려 해롭다(이번 런에서 `cat_title` 48.1%, `cat_qgen` 48.7%로
   23명 중 최하위 둘 — 둘 다 페르소나가 지시를 덮어쓴 케이스). 다른 방식으로 물어야 한다.
4. **표집**: real run과 같은 방식으로. 축 단위 균등 배분 금지.

---

## 파일

**커밋됨**
- `scripts/build_sni_export.py` — SNI → export 빌더. ⚠️ 현재 `instruction = Definition + Input`이라 **위 결함 1을 그대로 갖고 있다. 고쳐야 함.**
- `scripts/sni_probe_sample.py` — 축 단위 균등 표집. ⚠️ **결함 2의 원인. 그대로 쓰면 안 됨.**
- `scripts/sni_probe_axis.py` — 판독(home advantage). 도구 자체는 유효.
- `scripts/sbatch/run_sni_probe.sh` — PRO6000×2 / 48h. 5.5분에 끝났으므로 시간은 과다.
- `configs/roster_sni_probe.json` — 23명(category 12 + domain 10 + LUCA).
- 배선 7개 파일(loader/domains/meta/baseline_prompts/scout/baselines/routing_inference).

**커밋 안 됨 (작업트리에 남음)** — 아래 3개는 이번 세션 이전의 acc(seed1004) 작업과
내 SNI 훅이 한 파일에 섞여 있어 분리 커밋이 불가했다. SNI 훅이 빠지면 배선이 동작하지 않는다:
- `src/evaluation/scorer.py` — SNI 채점 블록(~75줄)
- `src/orchestrator.py` — `_score_pair_partial`의 SNI 분기(~10줄), 나머지 154줄은 acc seed1004 작업
- `src/prompts/coding.py` — SNI 프롬프트 분기 4곳(~30줄)

**재생성 가능(gitignore)**: `export/sni/`, `export/sni_xl/`, `results/sni/`.
원본 레포는 `/data5/jaehoonjeong/datasets/natural-instructions` (shallow clone, 3.1G).
