#!/bin/bash
# seed20211001 풀런: soft-WAR + 순위 기반 목숨 게이트.
#   config = configs/acc_train_seed20211001.yaml (war_mode=soft_linear, lives_mode=rank,
#   deletion_window=0). 데이터는 export/acc_v2 풀 11,097문제, batch 50 → 222스텝.
#   자원은 기존 acc 진화(submit_acc_seed20210101_smoke.sh)와 동일하게 유지 — vllm tp_size=2라
#   GPU 2장이 필요하다. 시간만 사용자 표준 48h.
#   ⚠️ TRAIN_SIZE/BATCH_SIZE 환경변수는 로그 출력용이고 실제 값은 config YAML에서 읽는다.
# Usage: bash scripts/sbatch/submit_acc_seed20211001_soft.sh
set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

SEED=20211001
DATASET="acc"
EVOL_CFG="configs/acc_train_seed20211001.yaml"
TRAIN_SIZE=11097
MAX_EPOCHS=1
BATCH_SIZE=50

GPUS=2
CPUS=4
MEM=64G
TIME=48:00:00

EVOL_JOB=$(SEED=${SEED} DATASET=${DATASET} TRAIN_SIZE=${TRAIN_SIZE} MAX_EPOCHS=${MAX_EPOCHS} BATCH_SIZE=${BATCH_SIZE} EVOL_CONFIG=${EVOL_CFG} \
  sbatch --parsable --job-name=mae_evolve_acc_soft \
    --gres=gpu:PRO6000:${GPUS} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_math_evolution.sh")
echo "Submitted acc soft-WAR seed${SEED}: ${EVOL_JOB}  (GPU=${GPUS} CPU=${CPUS} batch=${BATCH_SIZE} train=${TRAIN_SIZE})"
echo "  results -> results/acc/seed${SEED}/"
