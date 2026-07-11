#!/bin/bash
# seed20210101 SMOKE: 코딩 도메인(acc = QuantCat/TACO self-consistent 수정본) UB 스모크.
#   LUCA 단독 시작, batch 50, train_size 500(=10스텝). 목적: 하드에러 ~20-25/스텝 나오나 + 모델 UB.
#   config = configs/acc_train_seed20210101.yaml (train_size=500 스모크). HF_HOME은 common_bigmath에서 /data5 고정.
# Usage: bash submit_acc_seed20210101_smoke.sh
set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

SEED=20210101
DATASET="acc"
EVOL_CFG="configs/acc_train_seed20210101.yaml"
TRAIN_SIZE=2500
MAX_EPOCHS=1
BATCH_SIZE=50

GPUS=2
CPUS=4
MEM=64G
TIME=12:00:00

EVOL_JOB=$(SEED=${SEED} DATASET=${DATASET} TRAIN_SIZE=${TRAIN_SIZE} MAX_EPOCHS=${MAX_EPOCHS} BATCH_SIZE=${BATCH_SIZE} EVOL_CONFIG=${EVOL_CFG} \
  sbatch --parsable --job-name=mae_evolve_acc_smoke \
    --gres=gpu:PRO6000:${GPUS} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_math_evolution.sh")
echo "Submitted acc smoke seed${SEED}: ${EVOL_JOB}  (GPU=${GPUS} CPU=${CPUS} batch=${BATCH_SIZE} train=${TRAIN_SIZE})"
