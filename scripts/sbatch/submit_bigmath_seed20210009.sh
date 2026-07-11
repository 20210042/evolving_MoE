#!/bin/bash
# seed20210009: Thinking OFF + exclusive_solves scout, no NON-REDUNDANCY/ATOMICITY
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

# Per-epoch eval (roster_step_6/12/18/24/30), Thinking OFF via seed09 config.
# Reuses run_bigmath_eval_epochs_nothink.sh with EVAL_CONFIG override (gen thinking
# follows cfg.enable_thinking=false). Matches seed06/seed08 methodology.
EVAL_JOB=$(SEED=20210009 TRAIN_SIZE=300 MAX_EPOCHS=5 BATCH_SIZE=50 \
    EVAL_CONFIG=configs/bigmath_train_seed09.yaml \
    sbatch --parsable \
    --dependency=afterok:${EVOL_JOB} \
    --job-name=mae_eval_epochs_seed09 \
    --gres=gpu:PRO6000:2 \
    --cpus-per-task=4 \
    --mem=64G \
    --time=12:00:00 \
    --output="${REPO}/logs/%x.%j.out" \
    --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_bigmath_eval_epochs_nothink.sh")

echo "Submitted per-epoch eval seed20210009: job ${EVAL_JOB} (runs after ${EVOL_JOB})"

# Per-agent test UB from final roster, Thinking OFF, same held-out 500.
UB_JOB=$(SEED=20210009 \
    EVAL_CONFIG=configs/bigmath_train_seed09.yaml \
    sbatch --parsable \
    --dependency=afterok:${EVOL_JOB} \
    --job-name=mae_ub_eval_seed09 \
    --gres=gpu:PRO6000:2 \
    --cpus-per-task=4 \
    --mem=64G \
    --time=12:00:00 \
    --output="${REPO}/logs/%x.%j.out" \
    --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_bigmath_ub_eval.sh")

echo "Submitted UB eval seed20210009: job ${UB_JOB} (runs after ${EVOL_JOB})"
