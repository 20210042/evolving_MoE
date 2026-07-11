#!/bin/bash
#SBATCH --job-name=eval_vanilla_gemma4_NO-reasoning
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
#SBATCH --dependency=174148

set -euo pipefail

export VLLM_USE_FLASHINFER_SAMPLER=0

# ── 평가할 LoRA HuggingFace Hub ID 목록 (빈 문자열 = vanilla) ─────────
LORA_PATHS=(
    ""    # vanilla (LoRA 없음)
    # "Jongbin-kr/Llama3.1_inst_BIG-MATH"
    # "Jongbin-kr/Llama3.1_inst_BIG-MATH_statistics_and_probability"
    # "Jongbin-kr/Llama3.1_inst_BIG-MATH_number_theory_and_discrete_math"
    # "Jongbin-kr/Llama3.1_inst_BIG-MATH_geometry"
    # "Jongbin-kr/Llama3.1_inst_BIG-MATH_calculus_and_precalculus"
    # "Jongbin-kr/Llama3.1_inst_BIG-MATH_algebra"
)
# ─────────────────────────────────────────────────────────────────────

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE

export PYTHONPATH="$REPO/src"

for LORA_PATH in "${LORA_PATHS[@]}"; do
    if [ -z "${LORA_PATH}" ]; then
        RUN_NAME="vanilla_gemma4_NO-reasoning"
        LORA_FLAG=""
    else
        RUN_NAME="eval_$(basename "${LORA_PATH}")"
        LORA_FLAG="--finetuned_lora_path ${LORA_PATH}"
    fi

    echo "=== 모델: ${RUN_NAME} ==="

    srun --chdir="$REPO" python "$REPO/src/evaluate.py" \
        --model_name_or_path "google/gemma-4-31B-it" \
        ${LORA_FLAG} \
        --test_dataset bigmath \
        --data_ratio 1.0 \
        --inference_mode vllm \
        --max_model_len 32768 \
        --max_new_tokens 16384 \
        --temperature 0.0 \
        --output_dir "results/${RUN_NAME}" \
        --wandb_run_name "${RUN_NAME}" \
        --wandb_project "gemma4-31b" \
        --enable_thinking \
        --seed 42

    echo "=== 완료: ${RUN_NAME} → results/${RUN_NAME} ==="
done
