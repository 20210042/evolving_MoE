#!/bin/bash
#SBATCH --job-name=lbox_legal_tag
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=n05
#SBATCH --array=0-2%1
#SBATCH --output=logs/lbox_legal_tag/%x.%A_%a.log
#SBATCH --error=logs/lbox_legal_tag/%x.%A_%a.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-MoE}"
export PYTHONPATH="$REPO/src"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER=1
mkdir -p logs/lbox_legal_tag results/lbox_legal_category_tags/gemma4_a4b

case "${SLURM_ARRAY_TASK_ID}" in
    0) SPLIT="${SPLIT_OVERRIDE:-train}" ;;
    1) SPLIT="${SPLIT_OVERRIDE:-valid}" ;;
    2) SPLIT="${SPLIT_OVERRIDE:-test}" ;;
    *) echo "ERROR: unsupported array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

python scripts/tag_lbox_legal_categories.py \
    --split "$SPLIT" \
    --data-dir "${DATA_DIR:-export/lbox}" \
    --output-dir "${OUTPUT_DIR:-results/lbox_legal_category_tags/gemma4_a4b}" \
    --model "${MODEL:-google/gemma-4-26B-A4B-it}" \
    --batch-size "${BATCH_SIZE:-128}" \
    --max-model-len "${MAX_MODEL_LEN:-16384}" \
    --max-tokens "${MAX_TOKENS:-256}" \
    --tp-size "${TP_SIZE:-1}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --max-facts-chars "${MAX_FACTS_CHARS:-12000}" \
    --max-instruction-chars "${MAX_INSTRUCTION_CHARS:-4000}"
