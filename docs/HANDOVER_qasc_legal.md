# Handover — 신규 도메인 온보딩: QASC(과학 MC) + Legal(Lbox_open)

> 후임자용 **step-by-step 실행 계획**. acc(코딩) 온보딩([HANDOVER_acc_coding.md](HANDOVER_acc_coding.md))과 동일 프로세스를 QASC·Legal에 적용. 각 단계에 **정확한 파일/함수/명령/검증/함정**을 적었다. 순서대로, 각 단계 검증 통과 후 다음으로.

## 2026-07-04 진행 현황

### 완료 커밋
- `accde7b feat(domains): add qasc lbox paths`
  - 기존 `math` vs coding 2분기를 `math/coding/qasc/lbox` task family로 일반화.
  - QASC/Lbox 자연어 EM 태스크는 코드블록 추출을 타지 않고 raw answer를 채점하도록 `finalize_generation_output` 배선.
  - QASC/Lbox 전용 generation/scout/router prompt와 scorer 배선 완료.
  - 대상 테스트: `pytest tests/test_scorer.py tests/test_prompts.py` 통과.
- `bee78c4 feat(qasc): add onboarding scripts`
  - QASC builder/config/sbatch 추가.
  - `scripts/build_qasc.py`로 local JSONL 생성 가능.

### QASC 현재 상태
- 데이터 생성 완료:
  - `export/qasc/qasc_train.jsonl`: **8,134**
  - `export/qasc/qasc_validation.jsonl`: **926**
- 검증 완료:
  - `get_dataset("qasc", split="validation", local_dir="export/qasc")` → 926 rows.
  - 첫 샘플 gold letter 채점 `100.0`, wrong letter `0.0`.
- 아직 미실행:
  - QASC smoke evolution 제출.
  - QASC validation LUCA baseline / final-roster UB / routed top-1 eval.

### Legal 현재 상태
- Phase 1 데이터 생성 완료(casename + statute only, `ljp_*` 제외):
  - `export/lbox/lbox_train.jsonl`: **46,019**
  - `export/lbox/lbox_valid.jsonl`: **7,651**
  - valid task 분포: `casename=4,999`, `statute=2,652`.
- 검증 완료:
  - `get_dataset("lbox", split="valid", local_dir="export/lbox")` → 7,651 rows.
  - casename/statute 샘플 gold 채점 `100.0`, wrong answer `0.0`.
- 추가 파일:
  - [scripts/build_lbox.py](../scripts/build_lbox.py)
  - [configs/lbox_train_seed20210301.yaml](../configs/lbox_train_seed20210301.yaml)
  - [configs/lbox_eval_a4b.yaml](../configs/lbox_eval_a4b.yaml)
  - [scripts/sbatch/submit_lbox_seed20210301_smoke.sh](../scripts/sbatch/submit_lbox_seed20210301_smoke.sh)
  - [scripts/sbatch/run_lbox_eval.sh](../scripts/sbatch/run_lbox_eval.sh)
  - [scripts/sbatch/run_lbox_eval_routed.sh](../scripts/sbatch/run_lbox_eval_routed.sh)
- 아직 미실행:
  - Legal smoke evolution 제출.
  - Legal valid LUCA baseline / final-roster UB / routed top-1 eval.

### 전체 테스트 참고
- 전체 `pytest`는 기존 테스트 4개 실패:
  - `tests/test_action_selector.py` 2개: 현재 `ActionGateConfig.scale=0.5` + batch norm 수식과 테스트 기대값이 불일치.
  - `tests/test_lives.py`, `tests/test_roster.py` 각 1개: 테스트 함수 안에서 `orchestrator` 인스턴스명을 `import orchestrator` 모듈로 shadowing.
- 이번 멀티도메인/QASC 경로 대상 테스트는 통과.

## 0. 대전제 (읽고 시작)
- 백본 = `google/gemma-4-26B-A4B-it` 고정. 프롬프트-레벨 로스터 진화(scout 제안→게이트 WAR/도태). **표준 레시피 = seed18**(windowed-deletion gatefix + topic scout, 문제설명만, LUCA 단독 시작, `enable_thinking=false`).
- **적합성 판정 지표 = UB**(binning union). 진화 후 UB가 baseline보다 높고 상보성 있으면 "역할분화가 먹히는 도메인". 낮고 평탄하면 천장=백본.
- **이번 방향 = reference 없이 맨몸 UB부터 측정.** QASC는 fact1/2 안 줌, Legal은 facts만 줌(법조문 후보·판례 안 줌). UB 낮으면 그때 reference(보기 fact / 판례 retrieval) 투입 재검토.
- **핵심 통찰(acc와 동형)**: 한 도메인 안에 채점기 여러 개를 **디스패치**해도 됨. acc=`eval_mode`(stdin/function_call/gfg), Legal=`task_type`(casename/statute/ljp). 최종 판정은 다 **EM**.

### 인프라 함정 (반드시 지킬 것)
- **HF 캐시 = /data5** (홈 쿼터 작음). sbatch는 [common_bigmath.sh](../scripts/sbatch/common_bigmath.sh) `setup_job_env()`가 `HF_HOME=/data5...` 강제 — 새 sbatch는 반드시 이걸 source.
- **n05 노드 전력문제로 사용 금지** → 모든 잡 제출에 `--exclude=n05`.
- 데이터 로드는 실행 env로: `PY=/data5/jaehoonjeong/miniconda3/envs/evolving_moe/bin/python`. figure는 matplotlib 있는 `/data5/jaehoonjeong/miniconda3/bin/python`.
- **Lbox split 이름은 `valid`** (not `validation`). QASC는 `validation`.
- 커밋 메시지 한 줄, Co-Authored/Claude 이름 금지. 커밋 전 `git fetch`로 divergence 확인. push/force는 명시 지시 있을 때만.

---

# A. QASC (가장 싼 probe — 먼저)

과학 8지선다(A–H) EM. 데이터 작고 채점 trivial 하니 "MC도 헤드룸 나나"를 싸게 확인.

### A-1. 데이터 준비 → `export/qasc/qasc_{train,valid}.jsonl`
- **완료(2026-07-04, `bee78c4`)**: 실제 파일명은 loader 규칙에 맞춰 `qasc_train.jsonl`, `qasc_validation.jsonl`.
- 소스: `allenai/qasc`. split: **train 8134(진화용) / validation 926(홀드아웃 eval용, answerKey 있음)**. test는 라벨 없음 → 안 씀.
- 스크립트 [scripts/build_qasc.py](../scripts/build_qasc.py): 각 레코드를 아래 형태로 emit.
  ```json
  {"id": "...", "instruction": "<formatted_question 그대로>",
   "ground_truth": "F", "domain": "qasc", "dataset": "qasc",
   "scoring_kind": "qasc", "num_choices": 8}
  ```
  - `instruction` = 데이터셋의 `formatted_question` 필드(이미 "질문 (A) .. (H) .." 포맷 완성됨). reference(fact1/2)는 **넣지 않는다**(맨몸).
  - `ground_truth` = `answerKey`(A~H 한 글자).
- **검증**: emit 후 `head -1`로 한 줄 열어 instruction에 보기 8개 다 있고 ground_truth가 한 글자인지 눈으로 확인.

### A-2. 채점기 → `score_qasc_item`
- **완료(2026-07-04, `accde7b`)**.
- [src/evaluation/scorer.py](../src/evaluation/scorer.py)에 함수 추가:
  ```python
  def score_qasc_item(item, prediction):
      # 모델 출력에서 선택지 letter 1개 추출(정규식 첫 A-H 대문자, 또는 "(A)"/"Answer: A")
      # gold = item["ground_truth"]; EM이면 100 아니면 0
  ```
  - 추출 규칙: 마지막에 나오는 단독 대문자 A–H 우선, 없으면 보기 텍스트 일치 fallback. **후임 주의**: LLM이 "The answer is (F) local weather conditions"처럼 답함 → letter 우선 파싱.
- `score_one`에 배선: `if kind == "qasc": return score_qasc_item(...)`.
- [src/data/loader.py](../src/data/loader.py) `scoring_kind_for_dataset`: `if n == "qasc": return "qasc"`. `load_qasc()` 또는 get_dataset local_dir(`qasc_{split}.jsonl`)로 로드.
- **검증**: 정답 letter/오답/보기텍스트답 3케이스 유닛 테스트로 100/0/100 확인 (acc 채점 검증했던 방식).

### A-3. 프롬프트
- **완료(2026-07-04, `accde7b`)**.
- expert-gen: [src/prompts/coding.py](../src/prompts/coding.py) `build_expert_prompt`에 `task_family=qasc` 분기 추가. "정답 letter 하나만" 출력.
- scout/router: [src/prompts/meta.py](../src/prompts/meta.py)에 QASC 전용 `META_AGENT_QASC_PROMPT`, `MANAGER_QASC_PROMPT` 추가.
- 출력 후처리: [src/utils/helpers.py](../src/utils/helpers.py) `finalize_generation_output`에서 QASC는 raw natural-language answer로 유지(코드블록 추출 안 함).

### A-4. config → `configs/qasc_train_seed20210201.yaml` (QASC seed=20210201)
- **완료(2026-07-04, `bee78c4`)**: [configs/qasc_train_seed20210201.yaml](../configs/qasc_train_seed20210201.yaml).
- eval config: [configs/qasc_eval_a4b.yaml](../configs/qasc_eval_a4b.yaml).

### A-5. 스모크 진화 + 관측
- submit 래퍼 = [scripts/sbatch/submit_qasc_seed20210201_smoke.sh](../scripts/sbatch/submit_qasc_seed20210201_smoke.sh), **`--exclude=n05` 포함**.
- 실행 명령:
  ```bash
  bash scripts/sbatch/submit_qasc_seed20210201_smoke.sh
  ```
- 로그서 관측: 스텝별 **UB %**(`STATIC UPPER BOUND`), 로스터 성장, add/noop. **UB가 baseline보다 유의미하게 높나?** MC라 baseline이 이미 높으면(예: >85%) 헤드룸 작음 → 그 자체가 결론.

### A-6. eval (홀드아웃 = validation 926)
- **acc와 달리 test split이 있으니 test_ids.json 불필요.** `--split validation`으로 바로.
- eval 래퍼:
  - [scripts/sbatch/run_qasc_eval.sh](../scripts/sbatch/run_qasc_eval.sh): LUCA baseline + final roster UB.
  - [scripts/sbatch/run_qasc_eval_routed.sh](../scripts/sbatch/run_qasc_eval_routed.sh): final roster routed top-1.
- 측정 3종: **LUCA baseline(routed) / 최종로스터 UB(binning union) / routed top-1.** score는 `score_outputs.py`(qasc kind 자동).

---

# B. Legal (Lbox_open — facts→판단, EM 통합 도메인)

**6개 subset을 `task_type`으로 합쳐 한 도메인.** summarization(생성)·precedent_corpus(코퍼스)는 제외.

### B-0. subset 매핑 (확정)
| task_type | config(train/valid) | 입력 | 출력(gold) | 채점 |
|---|---|---|---|---|
| casename | casename_classification(+_plus) | facts | 죄명 string | **string EM**(공백정규화) |
| statute | statute_classification(+_plus) | facts | 법조문 list | **set EM**(순서무관 집합 일치) |
| ljp_criminal | ljp_criminal | facts | label.{fine_lv, imprisonment_*_lv} | **bucket EM**(파서 필요) |
| ljp_civil | ljp_civil | facts(+gist_of_claim) | claim_acceptance_lv | **bucket EM** |

- **단계 권장**: **Phase 1 = casename + statute만**(순수 EM, 파서 없음)으로 데이터·채점·진화·eval 전 파이프라인 완주 → Phase 2에서 ljp(양형 bucket 파서) 붙여 통합. 후임자는 Phase 1부터.

### B-1. 데이터 준비 → `export/lbox/lbox_{train,valid}.jsonl`
- **Phase 1 완료(2026-07-04)**: 실제 파일명은 loader 규칙에 맞춰 `lbox_train.jsonl`, `lbox_valid.jsonl`.
- 스크립트 [scripts/build_lbox.py](../scripts/build_lbox.py). 각 config를 로드(**split=`valid`/`train`**, `load_dataset("lbox/lbox_open", cfg, split=...)`)해 통합 레코드로:
  ```json
  {"id": "casename_80", "task_type": "casename", "casetype": "criminal",
   "facts": "<사실관계>", "ground_truth": "감염병의예방및관리에관한법률위반",
   "instruction": "<task별 지시문> + 사실관계:\n<facts>", "domain": "lbox",
   "dataset": "lbox", "scoring_kind": "lbox"}
  ```
  - statute면 `ground_truth`=statutes 리스트(그대로 list 저장).
  - **task별 지시문**(B-3) 을 instruction 앞에 붙여 emit(생성 프롬프트가 task를 알게).
  - id는 `{task_type}_{원본id}`로 충돌 방지.
- 진화용 train = 각 task train을 합쳐 셔플. eval용 valid = 각 task valid 합침(또는 task별 따로 측정).
- **self-consistency**: acc처럼 ref 실행 검증은 **불필요**(gold가 정답 라벨). 대신 emit 후 **gold 비어있지 않은지 / statutes가 list인지**만 체크.
- **검증**: task_type별 1건씩 열어 facts·ground_truth·instruction 정합 확인.

### B-2. 채점기 → `score_lbox_item` (task_type 디스패치)
- **Phase 1 완료(2026-07-04, `accde7b`)**: casename string EM, statute set EM 배선 및 유닛테스트 통과.
- [src/evaluation/scorer.py](../src/evaluation/scorer.py):
  ```python
  def score_lbox_item(item, prediction):
      t = item["task_type"]
      if t == "casename":  # string EM
          return 100 if _norm(pred_casename(prediction)) == _norm(item["ground_truth"]) else 0
      if t == "statute":   # set EM
          return 100 if set(parse_statutes(prediction)) == set(item["ground_truth"]) else 0
      if t.startswith("ljp"):  # bucket EM (Phase 2)
          return 100 if bucketize(prediction) == gold_bucket(item) else 0
  ```
  - `pred_casename`: 모델 출력에서 죄명 추출(마지막 줄/"죄명:" 뒤). `parse_statutes`: "형법 제298조" 패턴 정규식으로 집합화.
  - **후임 주의**: 한국어 정규화(공백·중점·괄호). EM이 너무 빡세면 casename은 부분일치 대신 정규화 후 완전일치 유지(진짜 EM), 대신 프롬프트로 출력형식을 고정("죄명만 정확히 한 줄").
- `score_one` 배선: `if kind == "lbox": return score_lbox_item(...)`. loader `scoring_kind_for_dataset`: `if n == "lbox": return "lbox"`.
- **검증**: casename 정답/오답, statute 집합일치/부분일치(=오답) 유닛테스트.

### B-3. task별 지시문 (B-1에서 instruction에 삽입)
- casename: `"다음 사실관계에 해당하는 사건명 또는 죄명을 정확히 한 줄로 답하라."`
- statute: `"다음 사실관계에 적용되는 법조문을 모두 나열하라(예: 형법 제298조)."`
- ljp(Phase2): `"다음 사실관계에 대한 양형(형종과 형량)을 예측하라."`
- scout/router는 공용(`META_AGENT_PROMPT`/`MANAGER_PROMPT`). scout hard error = task 섞인 전원-오답 문제(설명만) → 로스터가 법영역/태스크로 분화될 것.

### B-4. config / 진화 / eval
- config [configs/lbox_train_seed20210301.yaml](../configs/lbox_train_seed20210301.yaml) (Legal seed=20210301) = dataset=lbox, data_dir=export/lbox, LUCA, batch50, gatefix, thinking off, tp2. 스모크 train_size 2500.
- 진화 submit = [scripts/sbatch/submit_lbox_seed20210301_smoke.sh](../scripts/sbatch/submit_lbox_seed20210301_smoke.sh), `--exclude=n05` 포함.
- eval: **valid split 존재** → `--split valid`, test_ids 불필요. LUCA baseline / UB(binning) / routed 3종. score_outputs가 lbox kind 자동 디스패치.
  - [scripts/sbatch/run_lbox_eval.sh](../scripts/sbatch/run_lbox_eval.sh): LUCA baseline + final roster UB.
  - [scripts/sbatch/run_lbox_eval_routed.sh](../scripts/sbatch/run_lbox_eval_routed.sh): final roster routed top-1.
- **관측 포인트**: task_type별 UB도 쪼개 보면(casename vs statute) 어느 task가 역할분화 이득 큰지 보임.

---

## C. 공통 체크리스트 (각 도메인 완료 기준)
1. [x] QASC/Legal Phase 1 데이터 emit + 샘플 1건 눈으로 검증(instruction/gold 정합).
2. [x] QASC/Lbox 채점기 유닛테스트(정답100/오답0) 통과.
3. [ ] vanilla LUCA로 **UB 스모크** — 백본 맨몸 실력 + 헤드룸 유무.
4. [ ] 스모크 진화(batch50, ~50스텝) — 로스터 성장·UB 궤적.
5. [ ] eval 3종(baseline/UB/routed) 홀드아웃.
6. [ ] `docs/HANDOVER_<domain>.md` (acc 형식: 결과표 빈칸 골격 + 로스터 figure + 병목 분석).
7. [ ] 메모리 갱신([[project_domain_onboarding_process]]).

## D. 예상 판정 프레임 (보고 시)
- UB ≫ routed → 라우터 병목(코딩 재현). top-k union은 **QASC/Legal 다 EM이라 정당**(둘 중 맞으면 정답).
- UB ≈ baseline·평탄 → 백본 천장(수학 재현) → reference 투입(QASC fact, Legal 판례 retrieval) 재검토.
- Legal은 task_type별로 갈릴 수 있음(분류는 분화 먹고 ljp는 백본천장 등) — task별 분해해 보고.

## E. 참고 (현재 코드 자산 재사용처)
- 채점 배선 패턴: acc의 `score_acc_item`([scorer.py](../src/evaluation/scorer.py)) + `scoring_kind_for_dataset`([loader.py](../src/data/loader.py)) 그대로 복제.
- 진화/eval 스크립트: [run_math_evolution.sh](../scripts/sbatch/run_math_evolution.sh), [run_acc_eval.sh](../scripts/sbatch/run_acc_eval.sh), [run_acc_eval_routed.sh](../scripts/sbatch/run_acc_eval_routed.sh), [score_outputs.py](../scripts/score_outputs.py), 상한 top-k는 [score_outputs_topk.py](../scripts/score_outputs_topk.py).
- 로스터 figure: [make_acc_roster_fig.py](../scripts/make_acc_roster_fig.py) 복제(로그 경로만 교체).
- **seed 번호(확정)**: QASC=**20210201**, Legal=**20210301**. config/자원/제출은 사용자 OK 후.
