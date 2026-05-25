#!/bin/bash
# 허브 모델들을 BIG-MATH_filtered test split으로 평가하는 sbatch job 일괄 제출

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="$REPO/scripts/sbatch/eval_sft.sh"

MODELS=(
    "Jongbin-kr/Llama3.1_inst_BIG-MATH"
    "Jongbin-kr/Llama3.1_inst_BIG-MATH_statistics_and_probability"
    "Jongbin-kr/Llama3.1_inst_BIG-MATH_number_theory_and_discrete_math"
    "Jongbin-kr/Llama3.1_inst_BIG-MATH_geometry"
    "Jongbin-kr/Llama3.1_inst_BIG-MATH_calculus_and_precalculus"
    "Jongbin-kr/Llama3.1_inst_BIG-MATH_algebra"
)

for MODEL in "${MODELS[@]}"; do
    MODEL_BASENAME=$(basename "$MODEL")
    RUN_NAME="eval_bigmath_${MODEL_BASENAME}_$(date +%Y%m%d_%H%M%S)"

    echo "Submitting: $MODEL"
    sbatch \
        --job-name="eval_${MODEL_BASENAME}" \
        --export=ALL,MODEL="$MODEL",DATASET=bigmath,SPLIT=test,RUN_NAME="$RUN_NAME",WANDB_PROJECT=sft_eval \
        "$SCRIPT"
done

echo "All jobs submitted."
