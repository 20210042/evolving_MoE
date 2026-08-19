#!/bin/bash
set -euo pipefail

REPO="${REPO:-$(git rev-parse --show-toplevel)}"
cd "$REPO"

WORKER="${WORKER:-docs/journals/lbox_MoE/sft/scripts/sbatch/train_sft_qasc_lbox_luca.sh}"
TAGS_PATH="${TAGS_PATH:-results/lbox_legal_category_tags/gemma4_a4b_family_patent_merged/lbox_train_legal_categories.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-checkpoints/sft_lbox_legal_category}"
HUB_PREFIX="${HUB_PREFIX:-Jongbin-kr/llama3_lbox_legal_category}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"

CATEGORIES=(
  civil_property_obligation
  civil_family_inheritance
  criminal_property
  criminal_non_property
  admin_traffic
  admin_labor
  admin_other
  family_patent_special
)

mkdir -p logs/qasc_lbox_sft

for category in "${CATEGORIES[@]}"; do
  run_name="sft_lbox_legal_${category}_step5000"
  output_dir="${OUTPUT_PREFIX}_${category}_step5000"
  hub_model_id="${HUB_PREFIX}_${category}_step5000"

  echo "Submitting ${category}: run=${run_name} output=${output_dir} hub=${hub_model_id}"
  sbatch --parsable \
    --job-name="${run_name}" \
    --gres=gpu:PRO6000:1 \
    --cpus-per-task=1 \
    --mem=64G \
    --time=48:00:00 \
    --export=ALL,REPO="${REPO}",DATASET=lbox,DATA_DIR=export/lbox,PROMPT_SYSTEM=baseline,TRAIN_SPLIT=train,EVAL_SPLIT=valid,RUN_NAME="${run_name}",OUTPUT_DIR="${output_dir}",LEGAL_CATEGORY_TAGS_PATH="${TAGS_PATH}",LEGAL_CATEGORY="${category}",MAX_STEPS=5000,EVAL_STEPS=250,SAVE_STEPS=250,GRADIENT_ACCUMULATION_STEPS=4,DATA_RATIO=1.0,PUSH_TO_HUB="${PUSH_TO_HUB}",HUB_MODEL_ID="${hub_model_id}" \
    "${WORKER}"
done
