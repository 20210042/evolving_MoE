#!/bin/bash
# """Handoff note: SLURM job for evaluating Llama ACC algorithm checkpoints.
# It mirrors the Gemma evaluation wrapper but uses `meta-llama/Llama-3.1-8B-Instruct` as the base model
# and writes results under `results/acc_algorithm/`."""
#SBATCH --job-name=eval_acc_alg
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
mkdir -p "$REPO/logs"

source ~/.bashrc

if [ -f "$HOME/data/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/data/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "ERROR: conda not found"
    exit 1
fi

conda activate evolving_moe

echo "=== Slurm allocation ==="
echo "HOSTNAME=$(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

nvidia-smi \
    --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv

GPU_NAME="$(
    nvidia-smi \
        --query-gpu=name \
        --format=csv,noheader |
    head -n 1
)"

if [[ "$GPU_NAME" != *"RTX PRO 6000"* ]]; then
    echo "ERROR: RTX PRO 6000 was not allocated."
    echo "Allocated GPU: $GPU_NAME"
    exit 2
fi

echo "Confirmed GPU: $GPU_NAME"

python - <<'PY'
import torch
import vllm

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("GPU capability:", torch.cuda.get_device_capability(0))
print(
    "GPU memory GiB:",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3,
        2,
    ),
)
PY

# Disable the FlashInfer sampler to avoid current RTX PRO 6000 SM 12.x runtime issues.
# Model execution still uses vLLM; only this sampler backend is disabled.
export VLLM_USE_FLASHINFER_SAMPLER=0

# Flush Python logs promptly into the SLURM log file.
export PYTHONUNBUFFERED=1

export PYTHONPATH="$REPO/src"
export HF_HOME="/home/minjikim/minji_link/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

DATA_DIR="${DATA_DIR:-$REPO/data/acc_algorithm}"

if [ ! -f "$DATA_DIR/acc_algorithm_test.jsonl" ]; then
    echo "=== ACC algorithm data not found. Building: $DATA_DIR ==="

    python -u \
        "$REPO/scripts/build_acc_sft_dataset_algorithm.py" \
        --output-dir "$DATA_DIR"
fi

LORA_PATHS=()

if [ "${SKIP_VANILLA:-false}" != "true" ]; then
    LORA_PATHS+=("")
fi

PATTERNS=(
    "checkpoints/sft_acc_algorithm_all_*"
    "checkpoints/sft_acc_algorithm_constructive_implementation_*"
    "checkpoints/sft_acc_algorithm_quantitative_reasoning_*"
    "checkpoints/sft_acc_algorithm_state_space_reasoning_*"
    "checkpoints/sft_acc_algorithm_structured_data_*"
    "checkpoints/sft_acc_algorithm_greedy_strategy_*"
)

if [ "${SKIP_DEFAULT_LORAS:-false}" != "true" ]; then
    for PATTERN in "${PATTERNS[@]}"; do
        LATEST="$(
            ls -td ${PATTERN} 2>/dev/null |
            head -n 1 ||
            true
        )"

        if [ -n "$LATEST" ]; then
            LORA_PATHS+=("$LATEST")
        fi
    done
fi

EXTRA_LORA_PATHS="${EXTRA_LORA_PATHS:-}"

if [ -n "$EXTRA_LORA_PATHS" ]; then
    read -r -a EXTRA_ARRAY <<< "$EXTRA_LORA_PATHS"
    LORA_PATHS+=("${EXTRA_ARRAY[@]}")
fi

echo "=== Evaluation targets ==="

for LORA_PATH in "${LORA_PATHS[@]}"; do
    if [ -z "$LORA_PATH" ]; then
        echo "vanilla: meta-llama/Llama-3.1-8B-Instruct"
    else
        echo "LoRA: $LORA_PATH"
    fi
done

for LORA_PATH in "${LORA_PATHS[@]}"; do
    if [ -z "$LORA_PATH" ]; then
        RUN_NAME="eval_acc_algorithm_vanilla_llama"
        LORA_FLAG=()
    else
        RUN_NAME="eval_acc_algorithm_$(basename "$LORA_PATH")"
        LORA_FLAG=(
            --finetuned_lora_path "$LORA_PATH"
        )
    fi

    OUTPUT_DIR="results/acc_algorithm/${RUN_NAME}"

    echo "============================================================"
    echo "Eval start: ${RUN_NAME}"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "============================================================"

    srun \
        --chdir="$REPO" \
        python -u "$REPO/src/evaluate_algorithm.py" \
        --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
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
        --eval_batch_size 8 \
        --resume true \
        --output_dir "$OUTPUT_DIR" \
        --wandb_run_name "$RUN_NAME" \
        --wandb_project "evolving-moe" \
        --seed 42

    echo "=== Eval complete: ${RUN_NAME} ==="
done

echo "=== All requested evaluations complete ==="
