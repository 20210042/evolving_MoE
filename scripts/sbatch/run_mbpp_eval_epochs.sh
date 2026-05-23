#!/bin/bash
#SBATCH --job-name=mae_eval_epochs
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# Step 2/3 — Inference + scoring at end of each evolution epoch (starts with LUCA)
# Uses roster_step_{steps_per_epoch * epoch}.json (not batch_size for inference itself)

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_mbpp.sh
source "${REPO}/scripts/sbatch/common_mbpp.sh"
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

    echo "=== [Epoch ${EPOCH}] Inference: MBPP test ==="
    python scripts/run_inference.py \
        --dataset mbpp \
        --split test \
        --seed "${SEED}" \
        --roster_path "${ROSTER}" \
        --output_file "results/mbpp/seed${SEED}/inference_test_epoch${EPOCH}.jsonl"

    echo "=== [Epoch ${EPOCH}] Score: MBPP test ==="
    python scripts/score_outputs.py \
        --input "results/mbpp/seed${SEED}/inference_test_epoch${EPOCH}.jsonl" \
        --dataset mbpp \
        --split test

    echo "=== [Epoch ${EPOCH}] Inference: HumanEval ==="
    python scripts/run_inference.py \
        --dataset humaneval \
        --split test \
        --seed "${SEED}" \
        --roster_path "${ROSTER}" \
        --output_file "results/humaneval/seed${SEED}/inference_epoch${EPOCH}.jsonl"

    echo "=== [Epoch ${EPOCH}] Score: HumanEval ==="
    python scripts/score_outputs.py \
        --input "results/humaneval/seed${SEED}/inference_epoch${EPOCH}.jsonl" \
        --dataset humaneval \
        --split test
done

echo "=== Epoch evaluations done ==="
