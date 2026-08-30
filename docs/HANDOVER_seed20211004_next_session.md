# seed20211004 인수인계 (2026-08-20)

이 세션은 여기서 중단. 다음 세션은 이 문서에서 이어서 시작.

## 지금 상태 (사실만)

- 브랜치: `collab/acc-roster-binning-20211004`. 커밋 `00b1d7b`에 로스터+binning 산출물 번들.
- 진화 잡 3개(223386 evolution → 223387 train binning → 223388 test binning, `--dependency=afterok` 체인) 전부 COMPLETED.
- 설정: `configs/acc_train_seed20211004.yaml` — `war_mode: soft_partial`, `lives_mode: rank_windowed`, batch=100, train_size=11097, 1 epoch(111스텝).
  - 이 seed는 이전 20211001/20211002/20211003의 버그(스케일 안 맞는 `unique_rate_map`, `all_zero_war` 면제 누락)를 고친 **정상화 버전**. 코드 fix는 `src/orchestrator.py`의 `rank_windowed` 경로.
  - 삭제 이벤트는 여전히 0/111이지만, 이번엔 `u_delete`가 -0.03~-0.035로 경계 근처(버그였던 -4~-5와 다름) — 근소 미달로 보는 게 맞다는 게 이전 세션 판단. **재검증 안 됨, 확정 아님.**
- 최종 로스터 12명. `luca` lives=2, `c_63819` lives=3, 나머지 lives=5(max). `c_30658`/`c_56276`/`c_56422`는 active_steps가 71/44/37로 늦게 합류.
- **binning 결과 (train, 11,097문제)**: per-expert pass@1 전원 72.6~73.8%(스프레드 1.2pp, 거의 평평), union UB=83.78%. coverage: 전원해결 58.2%, 전원실패 16.2%, contested(1~11명) 25.6%.
- **binning 결과 (test, 751문제)**: per-expert pass@1 71.0~73.8%, 구조는 train과 유사(상세는 `binning_test_full.binned.summary.json` 참고, 아직 union/coverage 요약 다 안 읽음 — 다음 세션에서 확인).

## 이번 세션에서 시작했다가 미완성으로 끝난 것 — 반드시 다시 봐야 함

**질문**: union UB +11pp(LUCA 대비)가 진짜 전문가별 전문화 때문인지, 아니면 예전 Random 조건 때처럼 그냥 확률적 다양성(pass@12류) 인지.

**시도한 것**: `scripts/interaction_lowrank_test.py` (원핫, cell holdout, contested band 1~11)를 seed20211004 train binning에 돌림 (job 227945). 결과: 궁합(상호작용) 몫 +0.00%, z=-0.98 — 신호 없음.

**⚠️ 이 결과를 신뢰하면 안 되는 이유**: `results/embed_viz_test/acc_train_emb*` 캐시가 예전 로스터 시절 산출물이라 **전체 11,097문제 중 7,079개만** 있고, contested band로 더 좁혀서 실제로는 **1,933문제**로만 검정했다. 결론을 내리기 전에 이 제약을 먼저 사용자에게 확인받았어야 하는데 그러지 못해서 사용자가 "왜 계속 전체로 안 하냐"며 크게 불만을 표함. **이 결과는 폐기하고, 전체 11,097문제로 다시 해야 한다.**

**다음 세션 최우선 작업**:
1. seed20211004 train 11,097문제 전체에 대해 임베딩(`acc_train_emb`) 및 필요하면 hidden-state(`hs_last`/`hs_mean`)를 **새로 전체 계산**한다. 기존 스크립트는 `scripts/extract_embed_acc.py`, `scripts/extract_embed_hs*.sh` 계열로 추정(확인 필요) — 이건 모델 로드/torch 연산이므로 **master에서 절대 돌리지 말고 SLURM으로**, 스펙은 표준(`gpu:PRO6000:1 / 48h / cpu2 / mem32G`).
2. 전체 임베딩 확보 후 `scripts/interaction_lowrank_test.py`를 `BINNED=results/acc/seed20211004/binning_train_full.binned.jsonl`로 다시 돌려서 (onehot 먼저, 그다음 emb/hs_last) 진짜 결론을 낸다. `scripts/sbatch/run_acc_interaction_lowrank.sh`에 이미 `BINNED` env var 추가해뒀음.
3. **결론 내리기 전에 표본이 전체(11,097)인지 반드시 확인하고 보고할 것.** (신규 메모리: `feedback_full_dataset_scope.md`)

## 사용자가 명시적으로 요청한, 아직 시작 안 한 작업

1. **LLM router + Our roster 다운스트림 배포평가** (seed20211004 로스터, top-1/top-2 라우팅 실배포 정확도). 착수 전에 seed20211004 12명 전문가의 LoRA SFT 체크포인트가 이미 학습됐는지부터 확인 필요(README상 "다운스트림 LoRA 학습용" 문구 — 학습 안 됐을 가능성 있음, 확인 안 함).
2. **동일 평가를 human-prior 분류에도** — Dense/Human-prior/Random/Evolved 4조건 비교 패턴(`docs/SUMMARY_partition_ablation_qasc_coding.md` 참고)과 맞춰서.
3. 사용자가 명시: "hard war 기준으로 지금을 예측하는 건 말이 안 된다" — 즉 예전 hard-WAR 시절 분석 문서(`docs/SUMMARY_soft_war_signal_diagnosis.md`, `docs/REPORT_hard_vs_soft_war_comparison.md`)는 **seed20211001(구버전, 5명, 버그 있던 시절, test-only)** 기준이라 이번 20211004에 그대로 끌어다 쓰면 안 됨. 참고만 하고 새로 검증.

## 지켜야 할 표준 제약 (반복 강조)

- master 노드에서 torch/모델로드/대량연산 금지 — 전부 SLURM.
- 새 sbatch는 `gpu:PRO6000:1 / --time=48:00:00 / cpus 2 / mem 32G` 고정.
- 결론 내기 전 전체 데이터 범위 확보 여부 확인(이번에 어긴 것).
- 진화 재실행 요청 자제 — 이미 충분히 재실행했고 사용자가 지쳐 있음. 기존 데이터로 분석 우선.
