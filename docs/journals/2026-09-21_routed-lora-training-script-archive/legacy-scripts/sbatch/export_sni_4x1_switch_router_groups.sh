#!/bin/bash
#SBATCH --job-name=sni4x1_route_groups
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE

export PYTHONPATH="$REPO:$REPO/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BASE_MODEL="Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-switch-top1"
PREDICTIONS="$REPO/results/recent_sft_best_test/sni_sni_moe_best_checkpoint-4000_baseline_235806.jsonl"
SOURCE="$REPO/data/sni_split/sni_test.jsonl"
ROOT_OUTPUT="$REPO/results/sni_4x1_switch_router_groups"

run_variant() {
  local name="$1"
  local adapter="$2"
  local output="$ROOT_OUTPUT/$name"
  mkdir -p "$output/traces"
  torchrun --standalone --nproc_per_node=1 "$REPO/scripts/audit_4x1_router.py" \
    --model "$BASE_MODEL" \
    --adapter "$adapter" \
    --predictions "$PREDICTIONS" \
    --output-dir "$output/traces" \
    --variant sft \
    --dataset sni \
    --max-length 8192 \
    --seed 42
  python "$REPO/scripts/export_sni_4x1_router_groups.py" \
    --trace-dir "$output/traces" \
    --variant sft \
    --source "$SOURCE" \
    --predictions "$PREDICTIONS" \
    --output-dir "$output" \
    --model-label "$adapter"
}

run_variant \
  constant_aux_0.01_best \
  Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-sni-switch-top1-router-ffn-lora-sft-5ep

run_variant \
  annealed_aux_best \
  Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-sni-switch-top1-router-ffn-lora-aux-anneal-5ep

echo "Routing group export complete: $ROOT_OUTPUT"
