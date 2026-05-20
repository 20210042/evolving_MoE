#!/bin/bash
#SBATCH --job-name=mae_eval_epochs
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err

set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO"
mkdir -p logs

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "${CONDA_ENV:-MoE}"

export PYTHONPATH="$REPO/src"
DATA_ROOT="${DATA_DIR:-$REPO/data}"
SEED="${SEED:-20210042}"
MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"

# Roster snapshots directory
ROSTER_DIR="$REPO/results/mbpp/seed${SEED}/mbpp/seed${SEED}"

EPOCHS=(1 2 3)
STEPS=(19 38 57)

for i in "${!EPOCHS[@]}"; do
    EPOCH="${EPOCHS[$i]}"
    STEP="${STEPS[$i]}"
    
    ROSTER="${ROSTER_DIR}/roster_step_${STEP}.json"
    
    if [ ! -f "${ROSTER}" ]; then
        echo "ERROR: roster snapshot not found for Epoch ${EPOCH} (Step ${STEP}) at ${ROSTER}"
        exit 1
    fi
    
    echo "=========================================================================="
    echo "=== Running Evaluation for Epoch ${EPOCH} (Step ${STEP}) ==="
    echo "=========================================================================="
    
    # 1. Inference: MBPP test
    echo "=== [Epoch ${EPOCH}] Inference: MBPP test ==="
    python scripts/run_inference.py \
        --dataset mbpp \
        --split test \
        --seed "${SEED}" \
        --model "${MODEL}" \
        --roster_path "${ROSTER}" \
        --output_file "results/mbpp/seed${SEED}/inference_test_epoch${EPOCH}.jsonl" \
        --data_dir "${DATA_ROOT}"
        
    # 2. Score: MBPP test
    echo "=== [Epoch ${EPOCH}] Score: MBPP test ==="
    python scripts/score_outputs.py \
        --input "results/mbpp/seed${SEED}/inference_test_epoch${EPOCH}.jsonl" \
        --dataset mbpp \
        --split test \
        --data_dir "${DATA_ROOT}"
        
    # 3. Inference: HumanEval
    echo "=== [Epoch ${EPOCH}] Inference: HumanEval ==="
    python scripts/run_inference.py \
        --dataset humaneval \
        --split test \
        --seed "${SEED}" \
        --model "${MODEL}" \
        --roster_path "${ROSTER}" \
        --output_file "results/humaneval/seed${SEED}/inference_epoch${EPOCH}.jsonl" \
        --data_dir "${DATA_ROOT}"
        
    # 4. Score: HumanEval
    echo "=== [Epoch ${EPOCH}] Score: HumanEval ==="
    python scripts/score_outputs.py \
        --input "results/humaneval/seed${SEED}/inference_epoch${EPOCH}.jsonl" \
        --dataset humaneval \
        --split test \
        --data_dir "${DATA_ROOT}"
        
done

echo "=== Epoch Evaluations Done! ==="
