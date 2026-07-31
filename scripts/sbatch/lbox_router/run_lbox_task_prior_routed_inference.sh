#!/bin/bash
#SBATCH --job-name=lbox_task_routed
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/lbox_router/%x.%j.log
#SBATCH --error=logs/lbox_router/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
RESULTS_ROOT="${RESULTS_ROOT:?RESULTS_ROOT is required}"
ROUTER_DIR="${ROUTER_DIR:?ROUTER_DIR is required}"
cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-MoE}"
export PYTHONPATH="$REPO/src:$REPO/scripts"
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p logs/lbox_router "$RESULTS_ROOT"

srun --chdir="$REPO" python scripts/lbox_router/run_lbox_top1_routed_inference.py \
    --bank task_prior \
    --router-dir "$ROUTER_DIR" \
    --seed 42 \
    --split test \
    --output-dir "$RESULTS_ROOT" \
    --max-model-len 16384 \
    --max-new-tokens 512 \
    --vanilla-baseline results/qasc_lbox_sft_eval/lbox_Llama-3.1-8B-Instruct_vanilla_baseline_208397.jsonl \
    --dense-baseline results/qasc_lbox_sft_eval/lbox_sft_llama3_finetuned_lbox_baseline_eval500_full_eval_snapshot_checkpoint-12000_baseline_208278.jsonl
