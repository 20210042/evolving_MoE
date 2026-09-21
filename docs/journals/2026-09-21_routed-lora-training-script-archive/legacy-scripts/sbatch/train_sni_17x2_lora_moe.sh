#!/bin/bash
#SBATCH --job-name=sni_17x2_lora_moe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate MoE

export PYTHONPATH="$REPO:$REPO/src"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-sni-lora-moe}"

if [[ -n "${WANDB_RUN_ID:-}" ]]; then
    export WANDB_RESUME="${WANDB_RESUME:-must}"
fi

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    if [ ! -f "$RESUME_CHECKPOINT/adapter_model.safetensors" ] || \
       [ ! -f "$RESUME_CHECKPOINT/trainer_state.json" ]; then
        echo "ERROR: RESUME_CHECKPOINT is missing routed-LoRA or Trainer state: $RESUME_CHECKPOINT" >&2
        exit 2
    fi
    RESUME_ARGS=(--resume-from-checkpoint "$RESUME_CHECKPOINT")
fi

python -c 'from huggingface_hub import get_token; assert get_token(), "Hugging Face login required"'
python -c 'import wandb; assert wandb.api.api_key, "W&B login required"'

torchrun --standalone --nproc_per_node=2 "$REPO/scripts/train_sni_17x2_lora_moe.py" \
    --package-dir "$REPO/data/sni_moe_rosters/sni_moe_seed20212003" \
    --output-dir "$REPO/checkpoints/sni_17x2_lora_moe" \
    --hub-model-id "Jongbin-kr/llama-3.1-8b-instruct_SNI-ours-16x2-plus-generalist-lora-moe" \
    --max-length 8192 \
    --epochs 3 \
    "${RESUME_ARGS[@]}" \
    --gradient-accumulation-steps 8 \
    --router-warmup-steps 500 \
    --eval-steps 100 \
    --save-steps 100 \
    --logging-steps 10 \
    --run-name sni_17x2_lora_moe
