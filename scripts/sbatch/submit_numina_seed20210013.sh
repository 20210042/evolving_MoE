#!/bin/bash
# seed20210013: = seed10(control) + shared 기여 면제 OFF (도태 부활).
#   유일 변경: shared_contribution_exemption=false → WAR=0 에이전트가 lives 잃음(= seed04 체제).
#   all-zero 면제 유지, max_lives 5 유지, hole-aware swap(코드 전역) 이제 비로소 작동.
#   evolution + per-epoch eval + UB 풀세트.
# Usage: bash submit_numina_seed20210013.sh [--resume]

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"

DATASET="numina_cot"
EVOL_CFG="configs/numina_train_seed13.yaml"

RESUME_FLAG=""
if [ "${1:-}" = "--resume" ]; then RESUME_FLAG="--resume"; export RESUME=true; fi

EVOL_JOB=$(SEED=20210013 DATASET=${DATASET} TRAIN_SIZE=300 MAX_EPOCHS=5 BATCH_SIZE=50 EVOL_CONFIG=${EVOL_CFG} \
  sbatch --parsable --job-name=mae_evolve_seed13 --gres=gpu:PRO6000:1 --cpus-per-task=2 --mem=64G --time=48:00:00 \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_math_evolution.sh" ${RESUME_FLAG})
echo "Submitted evolution seed20210013: ${EVOL_JOB}"

EVAL_JOB=$(SEED=20210013 DATASET=${DATASET} TRAIN_SIZE=300 MAX_EPOCHS=5 BATCH_SIZE=50 EVAL_CONFIG=${EVOL_CFG} \
  sbatch --parsable --dependency=afterok:${EVOL_JOB} --job-name=mae_eval_epochs_seed13 \
    --gres=gpu:PRO6000:1 --cpus-per-task=2 --mem=64G --time=12:00:00 \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_bigmath_eval_epochs_nothink.sh")
echo "Submitted per-epoch eval seed20210013: ${EVAL_JOB} (after ${EVOL_JOB})"

UB_JOB=$(SEED=20210013 DATASET=${DATASET} EVAL_CONFIG=${EVOL_CFG} \
  sbatch --parsable --dependency=afterok:${EVOL_JOB} --job-name=mae_ub_eval_seed13 \
    --gres=gpu:PRO6000:1 --cpus-per-task=2 --mem=64G --time=12:00:00 \
    --output="${REPO}/logs/%x.%j.out" --error="${REPO}/logs/%x.%j.err" \
    "${REPO}/scripts/sbatch/run_bigmath_ub_eval.sh")
echo "Submitted UB eval seed20210013: ${UB_JOB} (after ${EVOL_JOB})"
