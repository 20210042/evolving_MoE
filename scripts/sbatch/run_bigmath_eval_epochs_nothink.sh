#!/bin/bash
#SBATCH --job-name=mae_eval_epochs_nothink
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# Per-epoch test eval with Thinking OFF (gen + router), matching seed06 methodology.
# Uses configs/bigmath_train_nothink.yaml -> enable_thinking=false propagates to
# GMRoutingPipeline generation (gen_enable_thinking) and sets tp_size=2.

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_bigmath.sh
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env
print_experiment_config

EVAL_CONFIG="${EVAL_CONFIG:-configs/bigmath_train_nothink.yaml}"
echo "=== Eval config: ${EVAL_CONFIG} (Thinking OFF) ==="

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

    echo "=== [Epoch ${EPOCH}] Inference: ${DATASET} test ==="
    # 재발방지: 옛 출력이 있으면 run_inference의 resume가 생성을 통째로 skip해 stale 결과를 재사용함 → 항상 새로 생성
    rm -f "${RESULTS_DIR}/inference_test_epoch${EPOCH}.jsonl"
    python scripts/run_inference.py \
        --config "${EVAL_CONFIG}" \
        --dataset "${DATASET}" \
        --split test \
        --seed "${SEED}" \
        --roster_path "${ROSTER}" \
        --max_items "${EVAL_SIZE:-500}" \
        --output_file "${RESULTS_DIR}/inference_test_epoch${EPOCH}.jsonl"

    echo "=== [Epoch ${EPOCH}] Score: ${DATASET} test ==="
    python scripts/score_outputs.py \
        --input "${RESULTS_DIR}/inference_test_epoch${EPOCH}.jsonl" \
        --dataset "${DATASET}" \
        --split test
done

echo "=== Epoch evaluations done (Thinking OFF) ==="
