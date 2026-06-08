#!/bin/bash
# seed20210008: 300 train / 500 eval, tp=2, Thinking OFF (solver + scout + router)
# Usage:
#   bash submit_bigmath_seed20210008.sh          # fresh start
#   bash submit_bigmath_seed20210008.sh --resume  # resume from last checkpoint

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

RESUME_FLAG=""
if [ "${1:-}" = "--resume" ]; then
    RESUME_FLAG="--resume"
    export RESUME=true
fi

EVOL_JOB=$(
    SEED=20210008 \
    TRAIN_SIZE=300 \
    MAX_EPOCHS=5 \
    BATCH_SIZE=50 \
    sbatch --parsable \
        --job-name=mae_evolve_bigmath \
        --gres=gpu:PRO6000:2 \
        --cpus-per-task=4 \
        --mem=64G \
        --time=48:00:00 \
        --output="${REPO}/logs/%x.%j.out" \
        --error="${REPO}/logs/%x.%j.err" \
        "${REPO}/scripts/sbatch/run_bigmath_evolution_nothink.sh" ${RESUME_FLAG}
)
echo "Evolution job: ${EVOL_JOB}"

EVAL_JOB=$(
    SEED=20210008 \
    sbatch --parsable \
        --dependency=afterok:${EVOL_JOB} \
        --job-name=mae_eval_bigmath \
        --gres=gpu:PRO6000:2 \
        --cpus-per-task=4 \
        --mem=64G \
        --time=4:00:00 \
        --output="${REPO}/logs/%x.%j.out" \
        --error="${REPO}/logs/%x.%j.err" \
        "${REPO}/scripts/sbatch/run_bigmath_eval.sh"
)
echo "Eval job:      ${EVAL_JOB} (depends on ${EVOL_JOB})"
echo "Submitted seed20210008 (RESUME=${RESUME:-false})"
