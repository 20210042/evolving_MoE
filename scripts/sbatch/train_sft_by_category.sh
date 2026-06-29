#!/bin/bash
#SBATCH --job-name=sft_llama3_numina_cot
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE

export PYTHONPATH="$REPO/src"


CATEGORY="${CATEGORY:-}"
if [ -z "${CATEGORY}" ]; then
    echo "ERROR: CATEGORY is required. e.g. CATEGORY=Algebra sbatch $0" >&2
    exit 1
fi

CATEGORY_SLUG="${CATEGORY,,}"
CATEGORY_SLUG="${CATEGORY_SLUG// /_}"
RUN_NAME="${RUN_NAME:-sft_llama3_numina_cot_persona_${CATEGORY_SLUG}}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${RUN_NAME}}"
HUB_MODEL_ID="${HUB_MODEL_ID:-Jongbin-kr/llama3_NuminaCoT_persona_${CATEGORY_SLUG}}"
DEEPSPEED_CONFIG="$REPO/configs/deepspeed_zero3.json"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"

CATEGORY_FLAG=(--categories "${CATEGORY}")

echo "=== 카테고리 학습 시작: ${CATEGORY} / 출력: ${OUTPUT_DIR} ==="
echo "=== GPUs: ${NPROC_PER_NODE}, GA: ${GRADIENT_ACCUMULATION_STEPS}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="

srun --ntasks=1 --gpus-per-task="${NPROC_PER_NODE}" --chdir="$REPO" \
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" "$REPO/src/train_sft.py" \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --dtype bfloat16 \
    --attn_implementation sdpa \
    --train_dataset numina_cot \
    --eval_dataset numina_cot \
    --data_ratio 1.0 \
    --seed 42 \
    "${CATEGORY_FLAG[@]}" \
    --train_sft_with_lora true \
    --sft_lora_rank 16 \
    --sft_lora_alpha 32 \
    --sft_lora_dropout 0.05 \
    --output_dir "${OUTPUT_DIR}" \
    --run_name "${RUN_NAME}" \
    --num_train_epochs 7 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --bf16 true \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --metric_for_best_model eval_loss \
    --report_to wandb \
    --wandb_project sft_dense \
    --push_to_hub True \
    --hub_model_id "${HUB_MODEL_ID}"
    

echo "=== SFT 학습 완료. 체크포인트: ${OUTPUT_DIR} ==="
