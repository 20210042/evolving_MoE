#!/bin/bash
# seed20210108: seed08 재실행 (Thinking OFF, 텍스트 NON-REDUNDANCY 규칙) — 교정 채점기(math_verify) 적용
# Usage:
#   bash submit_bigmath_seed20210108.sh           # fresh start
#   bash submit_bigmath_seed20210108.sh --resume   # resume from last checkpoint

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

RESUME_FLAG=""
if [ "${1:-}" = "--resume" ]; then
    RESUME_FLAG="--resume"
    export RESUME=true
fi

EVOL_JOB=$(SEED=20210108 TRAIN_SIZE=300 MAX_EPOCHS=5 BATCH_SIZE=50 \
  sbatch --parsable \
    --job-name=mae_evolve_bigmath \
    --gres=gpu:PRO6000:2 \
    --cpus-per-task=2 \
    --mem=64G \
    --time=48:00:00 \
    --output="${REPO}/logs/%x.%j.out" \
    --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_bigmath_evolution_nothink.sh" ${RESUME_FLAG})

echo "Submitted evolution seed20210108: job ${EVOL_JOB} (RESUME=${RESUME:-false})"

# Per-epoch eval (roster_step 6/12/18/24/30), Thinking OFF via nothink config.
EVAL_JOB=$(SEED=20210108 TRAIN_SIZE=300 MAX_EPOCHS=5 BATCH_SIZE=50 \
    EVAL_CONFIG=configs/bigmath_train_nothink.yaml \
    sbatch --parsable \
    --dependency=afterok:${EVOL_JOB} \
    --job-name=mae_eval_epochs_seed108 \
    --gres=gpu:PRO6000:2 \
    --cpus-per-task=2 \
    --mem=64G \
    --time=48:00:00 \
    --output="${REPO}/logs/%x.%j.out" \
    --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_bigmath_eval_epochs_nothink.sh")

echo "Submitted per-epoch eval seed20210108: job ${EVAL_JOB} (runs after ${EVOL_JOB})"

# Per-agent test UB from final roster.
UB_JOB=$(SEED=20210108 \
    EVAL_CONFIG=configs/bigmath_train_nothink.yaml \
    sbatch --parsable \
    --dependency=afterok:${EVOL_JOB} \
    --job-name=mae_ub_eval_seed108 \
    --gres=gpu:PRO6000:2 \
    --cpus-per-task=4 \
    --mem=64G \
    --time=12:00:00 \
    --output="${REPO}/logs/%x.%j.out" \
    --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_bigmath_ub_eval.sh")

echo "Submitted UB eval seed20210108: job ${UB_JOB} (runs after ${EVOL_JOB})"