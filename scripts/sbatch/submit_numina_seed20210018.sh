#!/bin/bash
# seed20210018: 게이트 제거 로직 수정(windowed deletion) — seed16(주제 scout)과 프롬프트 동일,
#   게이트만 누적 unique-rate 윈도로 통일(W=8, floor=0, cooldown=5). config = numina_train_seed18.yaml.
# Usage: bash submit_numina_seed20210018.sh [--resume]
set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

# ── 실험 파라미터 ───────────────────────────────────────────────
SEED=20210018
DATASET="numina_cot"
EVOL_CFG="configs/numina_train_seed18.yaml"
TRAIN_SIZE=62185
MAX_EPOCHS=1
BATCH_SIZE=100

# ── SLURM 자원 (수정은 여기서만) ────────────────────────────────
GPUS=2          # PRO6000 장수
CPUS=4          # cpus-per-task
MEM=64G         # 시스템 RAM
TIME=48:00:00

# ────────────────────────────────────────────────────────────────
RESUME_FLAG=""
if [ "${1:-}" = "--resume" ]; then RESUME_FLAG="--resume"; export RESUME=true; fi

EVOL_JOB=$(SEED=${SEED} DATASET=${DATASET} TRAIN_SIZE=${TRAIN_SIZE} MAX_EPOCHS=${MAX_EPOCHS} BATCH_SIZE=${BATCH_SIZE} EVOL_CONFIG=${EVOL_CFG} \
  sbatch --parsable --job-name=mae_evolve_seed18 \
    --gres=gpu:PRO6000:${GPUS} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_math_evolution.sh" ${RESUME_FLAG})
echo "Submitted evolution seed${SEED}: ${EVOL_JOB}  (GPU=${GPUS} CPU=${CPUS} MEM=${MEM} TIME=${TIME})"
