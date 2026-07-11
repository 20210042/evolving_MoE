#!/bin/bash
# seed20210301 SMOKE: LBox Open Phase 1 legal EM evolution.
# Usage: bash scripts/sbatch/submit_lbox_seed20210301_smoke.sh
set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

SEED=20210301
DATASET="lbox"
EVOL_CFG="configs/lbox_train_seed20210301.yaml"
TRAIN_SIZE="${TRAIN_SIZE:-2500}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-50}"

GPUS="${GPUS:-1}"
TP_SIZE="${TP_SIZE:-${GPUS}}"
CPUS="${CPUS:-2}"
MEM="${MEM:-32G}"
TIME="${TIME:-12:00:00}"

EVOL_JOB=$(SEED=${SEED} DATASET=${DATASET} TRAIN_SIZE=${TRAIN_SIZE} MAX_EPOCHS=${MAX_EPOCHS} BATCH_SIZE=${BATCH_SIZE} EVOL_CONFIG=${EVOL_CFG} VLLM_TP_SIZE=${TP_SIZE} \
  sbatch --parsable --job-name=mae_evolve_lbox_smoke --exclude=n05 \
    --gres=gpu:PRO6000:${GPUS} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_math_evolution.sh")
echo "Submitted lbox smoke seed${SEED}: ${EVOL_JOB}  (GPU=${GPUS} TP=${TP_SIZE} CPU=${CPUS} batch=${BATCH_SIZE} train=${TRAIN_SIZE})"
