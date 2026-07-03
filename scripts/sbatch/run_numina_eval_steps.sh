#!/bin/bash
#SBATCH --job-name=mae_eval_steps
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# Step-interval test eval (Thinking OFF). For single-epoch full-dataset runs (seed16)
# where per-epoch eval gives only one point: evaluates roster snapshots at fixed STEP
# intervals instead. Default = every 10k examples (=100 steps @ batch 100) + final.
# Env: SEED, DATASET, EVAL_CONFIG, STEPS ("100 200 ..."), EVAL_SIZE.

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_bigmath.sh
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env
print_experiment_config

EVAL_CONFIG="${EVAL_CONFIG:-configs/numina_train_seed16.yaml}"
STEPS="${STEPS:-100 200 300 400 500 600 622}"
echo "=== Eval config: ${EVAL_CONFIG} (Thinking OFF) | steps: ${STEPS} ==="

for STEP in ${STEPS}; do
    ROSTER="${ROSTER_SNAPSHOT_DIR}/roster_step_${STEP}.json"

    if [ ! -f "${ROSTER}" ]; then
        echo "ERROR: roster snapshot not found at ${ROSTER}"
        exit 1
    fi

    echo "=========================================================================="
    echo "=== Step ${STEP} eval (roster_step_${STEP}.json) ==="
    echo "=========================================================================="

    echo "=== [Step ${STEP}] Inference: ${DATASET} test ==="
    # 재발방지: 옛 출력이 있으면 run_inference의 resume가 생성을 통째로 skip해 stale 결과를 재사용함 → 항상 새로 생성
    rm -f "${RESULTS_DIR}/inference_test_step${STEP}.jsonl"
    python scripts/run_inference.py \
        --config "${EVAL_CONFIG}" \
        --dataset "${DATASET}" \
        --split test \
        --seed "${SEED}" \
        --roster_path "${ROSTER}" \
        --max_items "${EVAL_SIZE:-500}" \
        --output_file "${RESULTS_DIR}/inference_test_step${STEP}.jsonl"

    echo "=== [Step ${STEP}] Score: ${DATASET} test ==="
    python scripts/score_outputs.py \
        --input "${RESULTS_DIR}/inference_test_step${STEP}.jsonl" \
        --dataset "${DATASET}" \
        --split test
done

echo "=== Step-interval evaluations done (Thinking OFF) ==="
