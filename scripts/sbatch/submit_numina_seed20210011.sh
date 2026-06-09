#!/bin/bash
# seed20210011 (treatment): NuminaMath + 교정 scorer + verbal-RL scout(exclusive_solves)
#   + approach-persona(identity→system 1줄 + approach→user, strengths 없음) + Thinking OFF.
#   evolution + per-epoch eval + UB 풀세트.
# Usage: bash submit_numina_seed20210011.sh [--resume]

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

# ⚠️ NuminaMath loader 키 — 협업자 merge 후 실제 값으로 (config의 dataset과 일치해야 함)
DATASET="numina_cot"
EVOL_CFG="configs/numina_train_seed11.yaml"

RESUME_FLAG=""
if [ "${1:-}" = "--resume" ]; then RESUME_FLAG="--resume"; export RESUME=true; fi

EVOL_JOB=$(SEED=20210011 DATASET=${DATASET} TRAIN_SIZE=300 MAX_EPOCHS=5 BATCH_SIZE=50 EVOL_CONFIG=${EVOL_CFG} \
  sbatch --parsable --job-name=mae_evolve_seed11 --gres=gpu:PRO6000:2 --cpus-per-task=2 --mem=64G --time=48:00:00 \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_math_evolution.sh" ${RESUME_FLAG})
echo "Submitted evolution seed20210011: ${EVOL_JOB}"

EVAL_JOB=$(SEED=20210011 DATASET=${DATASET} TRAIN_SIZE=300 MAX_EPOCHS=5 BATCH_SIZE=50 EVAL_CONFIG=${EVOL_CFG} \
  sbatch --parsable --dependency=afterok:${EVOL_JOB} --job-name=mae_eval_epochs_seed11 \
    --gres=gpu:PRO6000:2 --cpus-per-task=2 --mem=64G --time=12:00:00 \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_bigmath_eval_epochs_nothink.sh")
echo "Submitted per-epoch eval seed20210011: ${EVAL_JOB} (after ${EVOL_JOB})"

UB_JOB=$(SEED=20210011 DATASET=${DATASET} EVAL_CONFIG=${EVOL_CFG} \
  sbatch --parsable --dependency=afterok:${EVOL_JOB} --job-name=mae_ub_eval_seed11 \
    --gres=gpu:PRO6000:2 --cpus-per-task=2 --mem=64G --time=12:00:00 \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_bigmath_ub_eval.sh")
echo "Submitted UB eval seed20210011: ${UB_JOB} (after ${EVOL_JOB})"
