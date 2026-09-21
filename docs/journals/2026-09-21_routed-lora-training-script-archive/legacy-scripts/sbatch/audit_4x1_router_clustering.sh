#!/bin/bash
#SBATCH --job-name=audit_4x1_router
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE

export PYTHONPATH="/data6/jongbinwon/pytorch_upcycling/src:$REPO:$REPO/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

: "${DATASET:?Set DATASET to lbox or sni}"
BASE_MODEL="Jongbin-kr/llama-3.1-8b-instruct-4x1-moe"
OUTPUT_DIR="$REPO/results/${DATASET}_4x1_router_clustering_audit"

case "$DATASET" in
  lbox)
    PREDICTIONS="$REPO/results/recent_sft_best_test/lbox_lbox_moe_best_checkpoint-4200_baseline_235807.jsonl"
    ADAPTER="Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-lbox-ffn-lora-sft-5ep"
    SOURCE="$REPO/export/lbox/lbox_test.jsonl"
    EXTRA_ANALYSIS=(
      --category-labels "$REPO/results/lbox_legal_category_tags/gemma4_a4b_family_patent_merged/lbox_test_legal_categories.jsonl"
      --human-prior "$REPO/results/lbox_human_prior/inference_test_human_prior.jsonl"
    )
    ;;
  sni)
    PREDICTIONS="$REPO/results/recent_sft_best_test/sni_sni_moe_best_checkpoint-4000_baseline_235806.jsonl"
    ADAPTER="$REPO/checkpoints/sft_sni_llama31_8b_4x1_moe_ffn_lora/checkpoint-4000"
    SOURCE="$REPO/data/sni_split/sni_test.jsonl"
    EXTRA_ANALYSIS=()
    ;;
  *)
    echo "Unsupported DATASET=$DATASET" >&2
    exit 2
    ;;
esac

for path in "$PREDICTIONS" "$SOURCE"; do
  test -f "$path" || { echo "Missing required file: $path" >&2; exit 2; }
done
mkdir -p "$OUTPUT_DIR"

run_audit() {
  local variant="$1"
  local adapter="$2"
  torchrun --standalone --nproc_per_node=2 "$REPO/scripts/audit_4x1_router.py" \
    --model "$BASE_MODEL" \
    --adapter "$adapter" \
    --predictions "$PREDICTIONS" \
    --output-dir "$OUTPUT_DIR" \
    --variant "$variant" \
    --dataset "$DATASET" \
    --max-length 8192 \
    --seed 42
}

run_audit raw none
run_audit sft "$ADAPTER"

python "$REPO/scripts/analyze_4x1_router_clustering.py" \
  --dataset "$DATASET" \
  --trace-dir "$OUTPUT_DIR" \
  --source "$SOURCE" \
  "${EXTRA_ANALYSIS[@]}"

echo "Audit complete: $OUTPUT_DIR/report.md"
