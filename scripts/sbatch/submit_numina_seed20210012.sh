#!/bin/bash
# seed20210012: = seed10(control)에서 hole-aware swap 로직만 바뀐 재현.
#   NuminaMath + 교정 scorer + verbal-RL scout(exclusive_solves) + 현행 persona + Thinking OFF.
#   하이퍼파라미터 seed10과 동일, 차이는 코드의 swap niche-recovery veto뿐.
#   evolution + per-epoch eval + UB 풀세트.
# Usage: bash submit_numina_seed20210012.sh [--resume]

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

DATASET="numina_cot"
EVOL_CFG="configs/numina_train_seed12.yaml"

RESUME_FLAG=""
if [ "${1:-}" = "--resume" ]; then RESUME_FLAG="--resume"; export RESUME=true; fi

EVOL_JOB=$(SEED=20210012 DATASET=${DATASET} TRAIN_SIZE=300 MAX_EPOCHS=5 BATCH_SIZE=50 EVOL_CONFIG=${EVOL_CFG} \
  sbatch --parsable --job-name=mae_evolve_seed12 --gres=gpu:PRO6000:1 --cpus-per-task=2 --mem=64G --time=48:00:00 \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_math_evolution.sh" ${RESUME_FLAG})
echo "Submitted evolution seed20210012: ${EVOL_JOB}"

EVAL_JOB=$(SEED=20210012 DATASET=${DATASET} TRAIN_SIZE=300 MAX_EPOCHS=5 BATCH_SIZE=50 EVAL_CONFIG=${EVOL_CFG} \
  sbatch --parsable --dependency=afterok:${EVOL_JOB} --job-name=mae_eval_epochs_seed12 \
    --gres=gpu:PRO6000:1 --cpus-per-task=2 --mem=64G --time=12:00:00 \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_bigmath_eval_epochs_nothink.sh")
echo "Submitted per-epoch eval seed20210012: ${EVAL_JOB} (after ${EVOL_JOB})"

UB_JOB=$(SEED=20210012 DATASET=${DATASET} EVAL_CONFIG=${EVOL_CFG} \
  sbatch --parsable --dependency=afterok:${EVOL_JOB} --job-name=mae_ub_eval_seed12 \
    --gres=gpu:PRO6000:1 --cpus-per-task=2 --mem=64G --time=12:00:00 \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_bigmath_ub_eval.sh")
echo "Submitted UB eval seed20210012: ${UB_JOB} (after ${EVOL_JOB})"
