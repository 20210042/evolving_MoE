#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
cd "$REPO"

RESULTS_DIR="${RESULTS_DIR:-results/lbox_persona_eval_ep5_test}"
WORKER="${WORKER:-scripts/sbatch/eval_sft_model.sh}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
WANDB_PROJECT="${WANDB_PROJECT:-evolving-moe-qasc-lbox-eval}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
CONDA_ENV="${CONDA_ENV:-MoE}"

mkdir -p "$RESULTS_DIR"

expert_prompt() {
    local expert_id="$1"
    python - "$expert_id" <<'PY'
import json
import sys
from pathlib import Path

expert_id = sys.argv[1]
mapping = json.loads(Path("results/lbox_binning_seed20210311/agent_mapping.json").read_text(encoding="utf-8"))
print(mapping[expert_id]["system_prompt"])
PY
}

submit_one() {
    local run_name="$1"
    local lora_path="$2"
    local prompt="$3"
    env \
        REPO="$REPO" \
        CONDA_ENV="$CONDA_ENV" \
        MODEL_NAME="$MODEL_NAME" \
        LORA_PATH="$lora_path" \
        RUN_NAME="$run_name" \
        TEST_DATASET="lbox" \
        SPLIT="test" \
        DATA_DIR="export/lbox" \
        INFERENCE_MODE="vllm" \
        MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
        TEMPERATURE="0.0" \
        OUTPUT_DIR="$RESULTS_DIR" \
        WANDB_PROJECT="$WANDB_PROJECT" \
        ENABLE_THINKING="false" \
        SYSTEM_PROMPT="$prompt" \
        sbatch --parsable \
            --job-name="$run_name" \
            --gres=gpu:PRO6000:1 \
            --cpus-per-task=2 \
            --mem=32G \
            --time=12:00:00 \
            --export=ALL \
            "$WORKER"
}

for expert_id in c_29934 c_28126 c_63621 c_24222 c_47388 c_27344 c_16504 c_31181 c_4799 c_31573; do
    run_name="eval_${expert_id}_low5_persona_ep5_test"
    lora_path="checkpoints/sft_lbox_roster_${expert_id}_low5_persona_eval_ep5"
    submit_one "$run_name" "$lora_path" "$(expert_prompt "$expert_id")"
done

submit_one \
    "eval_generalist_high6_persona_ep5_test" \
    "checkpoints/sft_lbox_generalist_high6_persona_eval_ep5" \
    "You are a Korean legal classification generalist. Given Korean legal facts, return the exact requested case name, charge, or statutory provision in the required one-line format without explanation."
