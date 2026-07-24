# 라우터 실험 스크립트

MoE/MoL에서 "문제별로 어떤 expert를 쓸 것인가"를 실험하는 코드 묶음.
전부 [`router_common.py`](router_common.py)의 데이터셋 레지스트리를 쓰므로
`--dataset qasc|lbox`로 전환한다. 새 도메인은 `SPECS`에 한 항목 추가하면 된다.

## 학습 레시피 (전 스크립트 공통)

```python
mu, sd = Xt.mean(0), Xt.std(0) + 1e-6      # train 통계로만 z-정규화
net  = Linear(d, hid) → ReLU → Dropout(drop)
       [→ Linear(hid,hid) → ReLU → Dropout  × (nl-2)]
       → Linear(hid, E)                     # E = expert 수
opt  = AdamW(lr=1e-3, weight_decay=wd)
loss = BCEWithLogitsLoss()
batch = 256, seed 앙상블 = 로짓 평균
```

**손실이 BCE인 게 핵심.** softmax로 "누가 1등이냐"를 고르는 게 아니라 expert마다
독립 이진분류 — "expert e가 이 문제를 풀까?"를 E개 병렬로 예측한다. 라벨은
solve 벡터 `[1,0,1,1,...]`. 라우팅은 그 로짓을 정렬해 top-1 / top-2를 뽑는다.

스윕 격자: `(hid, layers, epoch, dropout, wd)` =
`(256,2,100,.2,1e-3) (512,2,120,.3,1e-3) (1024,2,150,.3,1e-3) (2048,2,150,.3,1e-2)
(1024,3,150,.3,1e-2) (2048,3,200,.4,1e-2)`, seeds `[42,1,7]`.

## 입력 특징

| 특징 | 생성 | 차원 | 성격 | 오픈 QA |
|---|---|---|---|---|
| hidden-state | `extract_hidden_states.py` | 4096 | base LLM 마지막 레이어 pooling | ✅ |
| encoder-emb | `embed_expert_viz.py --stage embed` | 768 | embeddinggemma-300m | ✅ |
| answer-prob | `extract_answer_logits.py` | \|레터\| | base의 정답 레터 분포 = 난이도 신호 | ❌ |
| confidence | `extract_expert_confidence.py` | E | 어댑터별 자기확신 | ❌ |

앞의 둘은 **문제 단위**라 입력에 expert 구분 정보가 없다(사실상 solve 패턴 암기).
confidence만 expert-side 계산이고 QASC에서 가장 좋았다.

> ⚠️ **answer-prob / confidence는 MCQA 전용.** 다음 토큰 위치에서 정답 레터
> (A~H) 분포를 읽는 방식이라 고정된 답 어휘를 전제한다. LBOX처럼 답을 생성해
> 문자열 EM으로 채점하는 오픈 QA에는 이 정의가 성립하지 않는다. 그대로 쓰면
> 틀린 값이 나오므로 `require_mcqa()`가 막아둔다 —

## 스크립트

| 파일 | 질문 |
|---|---|
| `router_arch_explore.py` | **여기서 시작.** 규제 0으로 TRAIN을 재는 과적합 진단 — TRAIN도 낮으면 입력에 신호가 없는 것(정보론적 한계), TRAIN만 높으면 일반화 문제. + two-tower |
| `router_feasibility.py` | 임베딩만으로 top-1이 best-single을 넘나 |
| `router_feat_combo.py` | 난이도 신호(ansprob)가 hs의 벽을 뚫나 |
| `router_top1_sweep.py` | top-1 최대화 — 입력 × 용량 × 규제 전면 스윕 |
| `router_top2_learned.py` | 동일 조건 argmax → 상위2 |
| `router_top2_push.py` | top-2 총력전 (학습라우터 / anchor+residual / confidence) |
| `router_sweep.py` | anchor + 잔차 라우팅 용량 스윕 |
| `conf_route_eval.py` | 확신도 argmax 라우팅 + 캘리브레이션 |
| `moe_deploy_sweep.py` | **유일한 end-to-end** — top-2를 0.5 병합해 실제 생성·채점 |

`moe_deploy_sweep.py` 하나에 라우팅 방법 12개가 전부 들어 있다. 도메인 무관은
4종(random-2, oracle top-2, MLP hidden-state, MLP encoder-emb)이고 나머지 8종은
confidence/답분포에 의존해 MCQA에서만 붙는다.

> `router_top2_push.py`의 (A) 전역최적 고정집합은 **상한 참고선일 뿐 배포 방식이
> 아니다.** 프로젝트 방침상 고정 집합은 쓰지 않는다.

## 실행

연산은 전부 SLURM으로 (`scripts/sbatch/` 참고). 로그인 노드에서 돌리지 말 것.

```bash
# 1) 특징 추출 (GPU)
python scripts/extract_hidden_states.py --dataset lbox --split train
python scripts/extract_hidden_states.py --dataset lbox --split valid
# 2) 진단부터
python scripts/router_arch_explore.py --dataset lbox
# 3) end-to-end (어댑터 학습 완료 후)
python scripts/moe_deploy_sweep.py --dataset lbox \
    --binned <per-expert binned jsonl> --dense <dense SFT baseline jsonl> \
    --out results/lbox/deploy_sweep.md
```

## ⚠️ QASC 기존 결과에 대한 경고

`router_*.py`가 학습 타깃으로 쓰는 QASC gemma 라벨
(`export/qasc_binning_seed20210211/`, `inference_validation_binning_final.binned.jsonl`)은
채점 버그(커밋 `93550a7` 이전)의 영향을 받는다. 스크립트에 남아 있던 기준선
"best-single 83.9 / oracle 94.8"은 수정 후 **84.8 / 94.4**이고, 라벨 행렬 자체가
크게 달라진다(validation 전원-solve 192 → 555). **QASC로 낸 기존 라우터 수치는
재측정 전까지 인용하지 말 것.** 기준선은 이제 전부 라벨에서 계산하므로 하드코딩된
숫자는 남아 있지 않다.

LBOX는 채점기가 다르므로(`score_lbox_item`) 이 버그와 무관하다.
`moe_deploy_sweep.py`가 쓰는 llama LoRA 라벨도 무관하다(출력이 1글자라 파싱 오류 불가).
