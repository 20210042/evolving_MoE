#!/bin/bash
#SBATCH --job-name=sft_bigmath_by_category
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
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

CATEGORIES=(
    algebra
    calculus_and_precalculus
    geometry
    number_theory_and_discrete_math
    statistics_and_probability
)

for CATEGORY in "${CATEGORIES[@]}"; do
    RUN_NAME="sft_bigmath_${CATEGORY}_$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="checkpoints/${RUN_NAME}"

    echo "=== 카테고리 시작: ${CATEGORY} / 출력: ${OUTPUT_DIR} ==="

    srun --chdir="$REPO" python "$REPO/src/train_sft.py" \
        --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
        --dtype bfloat16 \
        --attn_implementation eager \
        --train_dataset bigmath \
        --eval_dataset bigmath \
        --data_ratio 1.0 \
        --seed 42 \
        --category "${CATEGORY}" \
        --train_sft_with_lora true \
        --sft_lora_rank 16 \
        --sft_lora_alpha 32 \
        --sft_lora_dropout 0.05 \
        --output_dir "${OUTPUT_DIR}" \
        --run_name "${RUN_NAME}" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --lr_scheduler_type cosine \
        --bf16 true \
        --logging_steps 10 \
        --eval_strategy steps \
        --eval_steps 200 \
        --save_strategy steps \
        --save_steps 200 \
        --save_total_limit 3 \
        --load_best_model_at_end true \
        --metric_for_best_model eval_loss \
        --report_to wandb \
        --wandb_project sft_dense \
        --push_to_hub True \
        --hub_model_id "Jongbin-kr/Llama3.1_inst_BIG-MATH_${CATEGORY}"

    echo "=== 카테고리 완료: ${CATEGORY} ==="
done

echo "=== 전체 카테고리 학습 완료 ==="
