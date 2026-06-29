#!/bin/bash
# """Handoff note: Local all-in-one runner for the ACC algorithm SFT workflow.
# Supported actions are `data`, `train_all`, `train_critics`, `eval`, and `all`; the script sets the
# MoE conda environment, HuggingFace cache paths, default GPUs, shared train arguments, checkpoint
# naming, and evaluation discovery logic."""

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

source ~/.bashrc
if [ -f "$HOME/data/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/data/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate MoE

export PYTHONPATH="$REPO/src"
export HF_HOME="/home/minjikim/minji_link/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ACTION="${1:-all}"
DATA_DIR="${DATA_DIR:-$REPO/data/acc_algorithm}"
PUSH_TO_HUB="${PUSH_TO_HUB:-False}"
MODEL_NAME="${MODEL_NAME:-google/gemma-4-31B-it}"
WANDB_PROJECT_TRAIN="${WANDB_PROJECT_TRAIN:-sft_dense}"
WANDB_PROJECT_EVAL="${WANDB_PROJECT_EVAL:-evolving-moe}"

COMMON_TRAIN_ARGS=(
  --model_name_or_path "$MODEL_NAME"
  --dtype bfloat16
  --attn_implementation eager
  --device_map sequential
  --train_dataset acc_algorithm
  --eval_dataset acc_algorithm
  --data_dir "$DATA_DIR"
  --data_ratio 1.0
  --seed 42
  --train_sft_with_lora true
  --sft_lora_rank 16
  --sft_lora_alpha 32
  --sft_lora_dropout 0.05
  --sft_lora_target_modules all-linear
  --num_train_epochs 3
  --per_device_train_batch_size 2
  --gradient_accumulation_steps 8
  --learning_rate 2e-5
  --lr_scheduler_type cosine
  --bf16 true
  --logging_steps 10
  --eval_strategy steps
  --eval_steps 200
  --save_strategy steps
  --save_steps 200
  --save_total_limit 3
  --load_best_model_at_end true
  --metric_for_best_model eval_loss
  --report_to wandb
  --wandb_project "$WANDB_PROJECT_TRAIN"
  --push_to_hub "$PUSH_TO_HUB"
)

safe_name() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_//; s/_$//'
}

build_data() {
  echo "=== [data] building balanced ACC algorithm dataset ==="
  python "$REPO/scripts/build_acc_sft_dataset_algorithm.py" --output-dir "$DATA_DIR"
}

train_all() {
  echo "=== [train_all] starting all-critic LoRA training | GPUs=${CUDA_VISIBLE_DEVICES} ==="
  RUN_NAME="${RUN_NAME:-sft_acc_algorithm_all_$(date +%Y%m%d_%H%M%S)}"
  OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/${RUN_NAME}}"
  python "$REPO/src/train_sft_algorithm.py" \
    "${COMMON_TRAIN_ARGS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --hub_model_id "${HUB_MODEL_ID:-Jongbin-kr/Gemma4_31B_it_ACC_algorithm_all}"
  echo "=== [train_all] complete: ${OUTPUT_DIR} ==="
}

train_critics() {
  echo "=== [train_critics] starting five critic-specific LoRA trainings sequentially | GPUs=${CUDA_VISIBLE_DEVICES} ==="
  CATEGORIES=(
    "Constructive Implementation"
    "Quantitative Reasoning"
    "State-Space Reasoning"
    "Structured Data"
    "Greedy Strategy"
  )
  for CATEGORY in "${CATEGORIES[@]}"; do
    SAFE_CATEGORY="$(safe_name "$CATEGORY")"
    RUN_NAME="sft_acc_algorithm_${SAFE_CATEGORY}_$(date +%Y%m%d_%H%M%S)"
    OUTPUT_DIR="checkpoints/${RUN_NAME}"
    echo "=== [train_critics] ${CATEGORY} start -> ${OUTPUT_DIR} ==="
    python "$REPO/src/train_sft_algorithm.py" \
      "${COMMON_TRAIN_ARGS[@]}" \
      --categories "$CATEGORY" \
      --output_dir "$OUTPUT_DIR" \
      --run_name "$RUN_NAME" \
      --hub_model_id "Jongbin-kr/Gemma4_31B_it_ACC_algorithm_${SAFE_CATEGORY}"
    echo "=== [train_critics] ${CATEGORY} complete ==="
  done
}

eval_models() {
  echo "=== [eval] starting evaluation for vanilla and latest LoRA checkpoints | GPUs=${CUDA_VISIBLE_DEVICES} ==="
  LORA_PATHS=("")
  PATTERNS=(
    "checkpoints/sft_acc_algorithm_all_*"
    "checkpoints/sft_acc_algorithm_constructive_implementation_*"
    "checkpoints/sft_acc_algorithm_quantitative_reasoning_*"
    "checkpoints/sft_acc_algorithm_state_space_reasoning_*"
    "checkpoints/sft_acc_algorithm_structured_data_*"
    "checkpoints/sft_acc_algorithm_greedy_strategy_*"
  )
  for PATTERN in "${PATTERNS[@]}"; do
    LATEST="$(ls -td ${PATTERN} 2>/dev/null | head -n 1 || true)"
    if [ -n "$LATEST" ]; then
      LORA_PATHS+=("$LATEST")
    fi
  done
  EXTRA_LORA_PATHS="${EXTRA_LORA_PATHS:-}"
  if [ -n "$EXTRA_LORA_PATHS" ]; then
    read -r -a EXTRA_ARRAY <<< "$EXTRA_LORA_PATHS"
    LORA_PATHS+=("${EXTRA_ARRAY[@]}")
  fi
  for LORA_PATH in "${LORA_PATHS[@]}"; do
    if [ -z "$LORA_PATH" ]; then
      RUN_NAME="eval_acc_algorithm_vanilla_gemma"
      LORA_FLAG=()
    else
      RUN_NAME="eval_acc_algorithm_$(basename "$LORA_PATH")"
      LORA_FLAG=(--finetuned_lora_path "$LORA_PATH")
    fi
    echo "=== [eval] ${RUN_NAME} start ==="
    python "$REPO/src/evaluate_algorithm.py" \
      --model_name_or_path "$MODEL_NAME" \
      "${LORA_FLAG[@]}" \
      --test_dataset acc_algorithm \
      --data_dir "$DATA_DIR" \
      --data_ratio 1.0 \
      --inference_mode vllm \
      --tensor_parallel_size 1 \
      --max_model_len 16384 \
      --max_new_tokens 8192 \
      --temperature 1.0 \
      --top_p 0.95 \
      --top_k 64 \
      --repetition_penalty 1.05 \
      --gpu_memory_utilization 0.90 \
      --enable_thinking true \
      --output_dir "results/acc_algorithm/${RUN_NAME}" \
      --wandb_run_name "$RUN_NAME" \
      --wandb_project "$WANDB_PROJECT_EVAL" \
      --seed 42
    echo "=== [eval] ${RUN_NAME} complete ==="
  done
}

case "$ACTION" in
  data)
    build_data
    ;;
  train_all)
    train_all
    ;;
  train_critics)
    train_critics
    ;;
  eval)
    eval_models
    ;;
  all)
    build_data
    train_all
    train_critics
    eval_models
    ;;
  *)
    echo "Usage: bash run_algorithm.sh {data|train_all|train_critics|eval|all}"
    exit 2
    ;;
esac
