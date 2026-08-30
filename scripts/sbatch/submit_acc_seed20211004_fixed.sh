#!/bin/bash
# seed20211004 풀런: soft_partial + rank_windowed, mcl 스케일 버그 수정판.
#   config = configs/acc_train_seed20211004.yaml. 코드 수정: src/orchestrator.py의
#   rank_windowed 경로가 unique_rate_map을 batch_n으로 정규화하도록 + all_zero_war 면제 추가.
#   데이터는 export/acc_v2 풀 11,097문제, batch 100 → 111스텝. GPU 2장(tp_size=2), 48h.
# Usage: bash scripts/sbatch/submit_acc_seed20211004_fixed.sh
set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

SEED=20211004
DATASET="acc"
EVOL_CFG="configs/acc_train_seed20211004.yaml"
TRAIN_SIZE=11097
MAX_EPOCHS=1
BATCH_SIZE=100

GPUS=2
CPUS=4
MEM=64G
TIME=48:00:00

EVOL_JOB=$(SEED=${SEED} DATASET=${DATASET} TRAIN_SIZE=${TRAIN_SIZE} MAX_EPOCHS=${MAX_EPOCHS} BATCH_SIZE=${BATCH_SIZE} EVOL_CONFIG=${EVOL_CFG} \
  sbatch --parsable --job-name=mae_evolve_acc_rank_windowed_fixed \
    --gres=gpu:PRO6000:${GPUS} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_math_evolution.sh")
echo "Submitted acc rank_windowed(fixed) seed${SEED}: ${EVOL_JOB}  (GPU=${GPUS} CPU=${CPUS} batch=${BATCH_SIZE} train=${TRAIN_SIZE})"
echo "  results -> results/acc/seed${SEED}/"
