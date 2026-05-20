#!/bin/bash
#SBATCH --job-name=mae_eval_lcb
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# Step 2/2 — LCB inference + scoring at end of each evolution epoch
# Evaluates on the 500 held-out LCB problems (test_ids.json)

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_lcb.sh
source "${REPO}/scripts/sbatch/common_lcb.sh"
setup_job_env
print_experiment_config

STEPS_PER_EPOCH="$(steps_per_epoch)"
echo "  epoch checkpoints: $(for e in $(seq 1 "${MAX_EPOCHS}"); do echo -n "E${e}=step$(epoch_end_step "${e}") "; done)"

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

    echo "=== [Epoch ${EPOCH}] Inference: LCB held-out (500 problems) ==="
    python scripts/run_inference.py \
        --dataset livecodebench \
        --split test \
        --seed "${SEED}" \
        --roster_path "${ROSTER}" \
        --output_file "${RESULTS_DIR}/inference_test_epoch${EPOCH}.jsonl"

    echo "=== [Epoch ${EPOCH}] Score: LCB ==="
    python scripts/score_outputs.py \
        --input "${RESULTS_DIR}/inference_test_epoch${EPOCH}.jsonl" \
        --dataset livecodebench \
        --split test

done

echo "=== Epoch evaluations done ==="
