#!/bin/bash
# Resume the best 4x1 sparse-upcycled SNI FFN-LoRA checkpoint for 3 more epochs.
#SBATCH --job-name=sni_4x1_resume3ep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=150G
#SBATCH --time=48:00:00
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

MODEL_NAME="Jongbin-kr/llama-3.1-8b-instruct-4x1-moe"
RESUME_CHECKPOINT="$REPO/checkpoints/sft_sni_llama31_8b_4x1_moe_ffn_lora/checkpoint-4000"
OUTPUT_DIR="$REPO/checkpoints/sft_sni_llama31_8b_4x1_moe_ffn_lora_resume3ep"
HUB_MODEL_ID="Jongbin-kr/llama-3.1-8b-instruct-4x1-moe-sni-ffn-lora"
DATA_DIR="$REPO/data/sni_split"
DEEPSPEED_CONFIG="$REPO/configs/deepspeed_zero2.json"

python -c 'from huggingface_hub import get_token; assert get_token(), "Hugging Face authentication is required"'
python -c 'import wandb; assert wandb.api.api_key, "W&B authentication is required"'
test -f "$RESUME_CHECKPOINT/global_step4000/mp_rank_00_model_states.pt"

echo "=== Resume SNI 4x1 FFN-LoRA: checkpoint-4000 + 3 epochs ==="
echo "resume=$RESUME_CHECKPOINT output=$OUTPUT_DIR hub=$HUB_MODEL_ID"

srun --ntasks=1 --gpus-per-task=2 --chdir="$REPO" \
  torchrun --standalone --nnodes=1 --nproc_per_node=2 "$REPO/src/train_sft.py" \
    --model_name_or_path "$MODEL_NAME" \
    --trust_remote_code true --dtype bfloat16 --attn_implementation sdpa \
    --train_dataset sni --eval_dataset sni --train_split train --eval_split valid \
    --data_dir "$DATA_DIR" --seed 42 \
    --train_sft_with_lora true --sft_lora_rank 16 --sft_lora_alpha 32 \
    --sft_lora_dropout 0.05 --sft_lora_target_modules gate_proj,up_proj,down_proj \
    --output_dir "$OUTPUT_DIR" --run_name sni_4x1_ffn_lora_resume3ep \
    --resume_from_checkpoint "$RESUME_CHECKPOINT" \
    --num_train_epochs 3.9196987986434443 --max_length 16384 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 --gradient_checkpointing true \
    --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_ratio 0.03 \
    --bf16 true --tf32 true --deepspeed "$DEEPSPEED_CONFIG" \
    --logging_steps 10 --eval_strategy steps --eval_steps 500 \
    --save_strategy steps --save_steps 500 --save_total_limit 3 \
    --load_best_model_at_end true --metric_for_best_model eval_loss --greater_is_better false \
    --completion_only_loss true --report_to wandb \
    --wandb_project sft_dense_sni --wandb_entity cvar_ddpo \
    --push_to_hub true --hub_strategy every_save --hub_model_id "$HUB_MODEL_ID"

echo "=== Resume complete; best checkpoint saved and pushed ==="
