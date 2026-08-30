#!/bin/bash
# seed20211003 풀런: soft_partial + rank_windowed(16배치 누적 최하위) + batch 100.
#   config = configs/acc_train_seed20211003.yaml. 데이터는 export/acc_v2 풀 11,097문제,
#   batch 100 → 111스텝. 자원은 기존 acc 진화와 동일(vllm tp_size=2 → GPU 2장 필요), 시간 48h.
# Usage: bash scripts/sbatch/submit_acc_seed20211003_rank_windowed.sh
set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

SEED=20211003
DATASET="acc"
EVOL_CFG="configs/acc_train_seed20211003.yaml"
TRAIN_SIZE=11097
MAX_EPOCHS=1
BATCH_SIZE=100

GPUS=2
CPUS=4
MEM=64G
TIME=48:00:00

EVOL_JOB=$(SEED=${SEED} DATASET=${DATASET} TRAIN_SIZE=${TRAIN_SIZE} MAX_EPOCHS=${MAX_EPOCHS} BATCH_SIZE=${BATCH_SIZE} EVOL_CONFIG=${EVOL_CFG} \
  sbatch --parsable --job-name=mae_evolve_acc_rank_windowed \
    --gres=gpu:PRO6000:${GPUS} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_math_evolution.sh")
echo "Submitted acc rank_windowed seed${SEED}: ${EVOL_JOB}  (GPU=${GPUS} CPU=${CPUS} batch=${BATCH_SIZE} train=${TRAIN_SIZE})"
echo "  results -> results/acc/seed${SEED}/"
