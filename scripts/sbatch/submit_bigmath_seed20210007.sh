#!/bin/bash
# seed20210007: 50K BigMath, tp=4, n04, 1 epoch
# Usage:
#   bash submit_bigmath_seed20210007.sh          # fresh start
#   bash submit_bigmath_seed20210007.sh --resume  # resume from last checkpoint

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

RESUME_FLAG=""
if [ "${1:-}" = "--resume" ]; then
    RESUME_FLAG="--resume"
    export RESUME=true
fi

SEED=20210007 \
TRAIN_SIZE=50000 \
MAX_EPOCHS=1 \
BATCH_SIZE=50 \
sbatch \
    --job-name=mae_evolve_bigmath \
    --gres=gpu:PRO6000:2 \
    --nodelist=n04 \
    --cpus-per-task=4 \
    --mem=64G \
    --time=48:00:00 \
    --output="${REPO}/logs/%x.%j.out" \
    --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_bigmath_evolution.sh" ${RESUME_FLAG}

echo "Submitted seed20210007 (RESUME=${RESUME:-false})"
