#!/bin/bash
set -euo pipefail

# Launch one SLURM job per model/LoRA so evaluations can run in parallel.
# Usage:
#   bash scripts/sbatch/launch_eval_sft_models.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKER="${SCRIPT_DIR}/eval_sft_model.sh"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
TEST_DATASET="${TEST_DATASET:-numina_cot}"
DATA_RATIO="${DATA_RATIO:-1.0}"
INFERENCE_MODE="${INFERENCE_MODE:-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-0.0}"
WANDB_PROJECT="${WANDB_PROJECT:-eval_numina_cot_metric0629}"
WANDB_ENTITY="${WANDB_ENTITY:-jongbin-kr-skiml_moe}"
SEED="${SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
CONDA_ENV="${CONDA_ENV:-MoE}"
SBATCH_GRES="${SBATCH_GRES:-}"

if [ -z "${OUTPUT_DIR:-}" ]; then
    OUTPUT_DIR="results/llama3_numina_cot_persona_persona"
fi

LORA_SPECS=(
    # Format: "<LoRA path>|<fixed prompt spec>".
    # Use LUCA for the default system prompt: "You are a helpful math assistant."

    # # Vanilla base model. Empty LoRA path means no adapter.
    # "|LUCA"
    # "|Calculus"
    # "|Combinatorics"
    # "|Number Theory"
    # "|Geometry"
    # "|Algebra"

    # # All-category SFT. (3 epoch)
    # "Jongbin-kr/llama3_NuminaCoT_all|LUCA"

    # # Category-specific SFT. (3 epoch)
    # "Jongbin-kr/llama3_NuminaCoT_calculus|LUCA"
    # "Jongbin-kr/llama3_NuminaCoT_combinatorics|LUCA"
    # "Jongbin-kr/llama3_NuminaCoT_number_theory|LUCA"
    # "Jongbin-kr/llama3_NuminaCoT_geometry|LUCA"
    # "Jongbin-kr/llama3_NuminaCoT_algebra|LUCA"

    # # SFT-more. (4.5~5 epoch)
    # "Jongbin-kr/llama3_NuminaCoT_more_calculus|LUCA"
    # "Jongbin-kr/llama3_NuminaCoT_more_combinatorics|LUCA"
    # "Jongbin-kr/llama3_NuminaCoT_more_number_theory|LUCA"
    # "Jongbin-kr/llama3_NuminaCoT_more_geometry|LUCA"
    # "Jongbin-kr/llama3_NuminaCoT_more_algebra|LUCA"

    # Persona SFT with its own fixed training-category persona prompt.
    "Jongbin-kr/llama3_NuminaCoT_persona_calculus|Calculus"
    "Jongbin-kr/llama3_NuminaCoT_persona_combinatorics|Combinatorics"
    "Jongbin-kr/llama3_NuminaCoT_persona_number_theory|Number Theory"
    "Jongbin-kr/llama3_NuminaCoT_persona_geometry|Geometry"
    "Jongbin-kr/llama3_NuminaCoT_persona_algebra|Algebra"

    # # GRPO.
    # "Jongbin-kr/llama3_NuminaCoT_grpo_all|LUCA"
)

if [ ! -f "${WORKER}" ]; then
    echo "ERROR: worker script not found: ${WORKER}" >&2
    exit 1
fi

echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "SBATCH_GRES=${SBATCH_GRES:-script default}"
echo "NUM_JOBS=${#LORA_SPECS[@]}"

SBATCH_ARGS=()
if [ -n "${SBATCH_GRES}" ]; then
    SBATCH_ARGS+=(--gres="${SBATCH_GRES}")
fi

for SPEC in "${LORA_SPECS[@]}"; do
    LORA_PATH="${SPEC%%|*}"
    PROMPT_SPEC="${SPEC#*|}"

    if [ -z "${LORA_PATH}" ]; then
        RUN_NAME="vanilla_llama3"
        LORA_EXPORT="__NONE__"
    else
        RUN_NAME="$(basename "${LORA_PATH}")"
        LORA_EXPORT="${LORA_PATH}"
    fi

    if [ "${PROMPT_SPEC}" = "LUCA" ] || [ "${PROMPT_SPEC}" = "false" ] || [ "${PROMPT_SPEC}" = "0" ]; then
        USE_CATEGORY_PROMPT_EXPORT="false"
    else
        USE_CATEGORY_PROMPT_EXPORT="${PROMPT_SPEC}"
        PROMPT_SPEC_SLUG="${PROMPT_SPEC// /_}"
        RUN_NAME="${RUN_NAME}_prompt_${PROMPT_SPEC_SLUG}"
    fi

    JOB_NAME="eval_${RUN_NAME}"
    echo "Submitting ${JOB_NAME} with prompt=${PROMPT_SPEC}"
    sbatch \
        "${SBATCH_ARGS[@]}" \
        --job-name="${JOB_NAME}" \
        --export=ALL,REPO="${REPO}",CONDA_ENV="${CONDA_ENV}",MODEL_NAME="${MODEL_NAME}",LORA_PATH="${LORA_EXPORT}",RUN_NAME="${RUN_NAME}",TEST_DATASET="${TEST_DATASET}",DATA_RATIO="${DATA_RATIO}",INFERENCE_MODE="${INFERENCE_MODE}",MAX_MODEL_LEN="${MAX_MODEL_LEN}",MAX_NEW_TOKENS="${MAX_NEW_TOKENS}",TEMPERATURE="${TEMPERATURE}",OUTPUT_DIR="${OUTPUT_DIR}",WANDB_PROJECT="${WANDB_PROJECT}",WANDB_ENTITY="${WANDB_ENTITY}",SEED="${SEED}",ENABLE_THINKING="${ENABLE_THINKING}",USE_CATEGORY_PROMPT="${USE_CATEGORY_PROMPT_EXPORT}" \
        "${WORKER}"
done
