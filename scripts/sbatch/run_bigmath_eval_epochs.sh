#!/bin/bash
#SBATCH --job-name=mae_eval_epochs_bigmath
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# Step 2 — Inference + scoring at end of each evolution epoch

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_bigmath.sh
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env
print_experiment_config

STEPS_PER_EPOCH="$(steps_per_epoch)"

for EPOCH in $(seq 1 "${MAX_EPOCHS}"); do
    STEP="$(epoch_end_step "${EPOCH}")"
    ROSTER="${ROSTER_SNAPSHOT_DIR}/roster_step_${STEP}.json"

    if [ ! -f "${ROSTER}" ]; then
        echo "ERROR: roster snapshot not found for Epoch ${EPOCH} (Step ${STEP}) at ${ROSTER}"
        exit 1
    fi

    echo "=========================================================================="
    echo "=== Epoch ${EPOCH} eval (roster_step_${STEP}.json) ==="
    echo "=========================================================================="

    echo "=== [Epoch ${EPOCH}] Inference: BigMath test ==="
    python scripts/run_inference.py \
        --dataset bigmath \
        --split test \
        --seed "${SEED}" \
        --roster_path "${ROSTER}" \
        --max_items "${EVAL_SIZE:-500}" \
        --output_file "results/bigmath/seed${SEED}/inference_test_epoch${EPOCH}.jsonl"

    echo "=== [Epoch ${EPOCH}] Score: BigMath test ==="
    python scripts/score_outputs.py \
        --input "results/bigmath/seed${SEED}/inference_test_epoch${EPOCH}.jsonl" \
        --dataset bigmath \
        --split test
done

echo "=== Epoch evaluations done ==="
