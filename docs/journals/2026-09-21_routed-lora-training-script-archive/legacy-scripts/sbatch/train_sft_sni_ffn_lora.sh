#!/bin/bash
#SBATCH --job-name=sft_sni_ffn_lora
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
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

python -c 'from huggingface_hub import get_token; assert get_token(), "Hugging Face authentication is required for gated model download and Hub pushes. Run: hf auth login"'
python -c 'import wandb; assert wandb.api.api_key, "Weights & Biases authentication is required. Run: wandb login"'

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME to the Hugging Face base model ID}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME}"
HUB_MODEL_ID="${HUB_MODEL_ID:?Set HUB_MODEL_ID}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${RUN_NAME}}"
DATA_DIR="${DATA_DIR:-$REPO/data/sni_split}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$REPO/configs/deepspeed_zero3.json}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"

echo "=== SNI FFN-only LoRA SFT: ${RUN_NAME} ==="
echo "MODEL_NAME=${MODEL_NAME}"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "HUB_MODEL_ID=${HUB_MODEL_ID}"
echo "GPUs=${NPROC_PER_NODE}, CPUs=${SLURM_CPUS_PER_TASK:-2}, SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unknown}MB"

srun --ntasks=1 --gpus-per-task="${NPROC_PER_NODE}" --chdir="$REPO" \
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" "$REPO/src/train_sft.py" \
    --model_name_or_path "${MODEL_NAME}" \
    --trust_remote_code true \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --train_dataset sni \
    --eval_dataset sni \
    --train_split train \
    --eval_split valid \
    --data_dir "${DATA_DIR}" \
    --seed 42 \
    --train_sft_with_lora true \
    --sft_lora_rank 16 \
    --sft_lora_alpha 32 \
    --sft_lora_dropout 0.05 \
    --sft_lora_target_modules gate_proj,up_proj,down_proj \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --num_train_epochs 1 \
    --max_length 16384 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --bf16 true \
    --tf32 true \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 500 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --greater_is_better false \
    --completion_only_loss true \
    --report_to wandb \
    --wandb_project sft_dense_sni \
    --push_to_hub true \
    --hub_strategy every_save \
    --hub_model_id "${HUB_MODEL_ID}"

echo "=== SNI FFN-only LoRA SFT complete: ${RUN_NAME} ==="
