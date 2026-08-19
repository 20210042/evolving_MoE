#!/bin/bash
#SBATCH --job-name=compose_lora_moe_5cat
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE_mergoo_a6000

python "$REPO/scripts/compose_lora_moe.py" \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --expert algebra=checkpoints/sft_llama3_numina_cot_algebra \
    --expert calculus=checkpoints/sft_llama3_numina_cot_calculus \
    --expert combinatorics=checkpoints/sft_llama3_numina_cot_combinatorics \
    --expert geometry=checkpoints/sft_llama3_numina_cot_geometry \
    --expert number_theory=checkpoints/sft_llama3_numina_cot_number_theory \
    --output_dir checkpoints/mergoo_lora_moe_5cat_top2 \
    --num_experts_per_tok 2 \
    --dtype bfloat16 \
    --device_map auto \
    --max_shard_size 9GB
