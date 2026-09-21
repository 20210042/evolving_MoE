#!/bin/bash
# One short smoke test for the materialized fallback SFT path.
#SBATCH --job-name=lbox_fallback_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -c 'from huggingface_hub import get_token; assert get_token(), "Hugging Face authentication is required"'

PACKAGE_DIR="${LBOX_FALLBACK_PACKAGE_DIR:-$REPO/export/lbox_moe_seed20210311}"
TRAIN_JSONL="$PACKAGE_DIR/train/expert_06_c_16504.jsonl"
EVAL_JSONL="$PACKAGE_DIR/valid.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/checkpoints/lbox_fallback_smoke}"

srun --ntasks=1 --gpus-per-task=1 --chdir="$REPO" \
    python "$REPO/src/train_sft.py" \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --trust_remote_code true --dtype bfloat16 --attn_implementation sdpa \
    --train_dataset lbox --eval_dataset lbox --train_split train --eval_split valid \
    --data_dir "$PACKAGE_DIR" --eval_data_dir "$PACKAGE_DIR" \
    --prebuilt_train_jsonl "$TRAIN_JSONL" --prebuilt_eval_jsonl "$EVAL_JSONL" \
    --train_sft_with_lora true --sft_lora_rank 16 --sft_lora_alpha 32 \
    --sft_lora_dropout 0.05 --sft_lora_target_modules gate_proj,up_proj,down_proj \
    --output_dir "$OUTPUT_DIR" --run_name lbox_fallback_smoke \
    --num_train_epochs 1 --max_steps 2 --max_length 1024 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 --gradient_checkpointing true \
    --learning_rate 2e-5 --bf16 true --tf32 true --logging_steps 1 \
    --eval_strategy steps --eval_steps 2 --save_strategy steps --save_steps 2 \
    --save_total_limit 1 --load_best_model_at_end true --metric_for_best_model eval_loss \
    --greater_is_better false --completion_only_loss true --report_to none \
    --wandb_project sft_dense_lbox_roster_ffn_only --wandb_entity cvar_ddpo \
    --push_to_hub false

echo "Fallback FFN-only smoke test passed."
