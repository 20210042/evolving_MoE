#!/bin/bash
# seed20210009: Thinking ON + exclusive_solves scout, no NON-REDUNDANCY/ATOMICITY
# Usage:
#   bash submit_bigmath_seed20210009.sh          # fresh start
#   bash submit_bigmath_seed20210009.sh --resume  # resume from last checkpoint

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

RESUME_FLAG=""
if [ "${1:-}" = "--resume" ]; then
    RESUME_FLAG="--resume"
    export RESUME=true
fi

EVOL_JOB=$(SEED=20210009 TRAIN_SIZE=300 MAX_EPOCHS=5 BATCH_SIZE=50 \
  sbatch --parsable \
    --job-name=mae_evolve_bigmath \
    --gres=gpu:PRO6000:2 \
    --cpus-per-task=4 \
    --mem=64G \
    --time=48:00:00 \
    --output="${REPO}/logs/%x.%j.out" \
    --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_bigmath_evolution_seed09.sh" ${RESUME_FLAG})

echo "Submitted evolution seed20210009: job ${EVOL_JOB} (RESUME=${RESUME:-false})"

EVAL_JOB=$(SEED=20210009 sbatch --parsable \
    --dependency=afterok:${EVOL_JOB} \
    --job-name=mae_eval_bigmath \
    --gres=gpu:PRO6000:2 \
    --cpus-per-task=4 \
    --mem=64G \
    --time=4:00:00 \
    --output="${REPO}/logs/%x.%j.out" \
    --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_bigmath_eval_seed09.sh")

echo "Submitted eval seed20210009: job ${EVAL_JOB} (runs after ${EVOL_JOB})"
