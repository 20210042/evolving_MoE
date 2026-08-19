#!/bin/bash
#SBATCH --job-name=lbox_legal_router
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lbox_router/%x.%j.log
#SBATCH --error=logs/lbox_router/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
TAGS_FILE="${TAGS_FILE:-results/lbox_legal_category_tags/gemma4_a4b_family_patent_merged/lbox_train_legal_categories.jsonl}"

cd "$REPO"
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-MoE}"
export PYTHONPATH="$REPO/src:$REPO/scripts:$REPO/scripts/lbox_router"
mkdir -p logs/lbox_router "$OUTPUT_DIR"

python scripts/lbox_router/train_lbox_legal_category_router.py \
    --tags-file "$TAGS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --hidden 512 \
    --epochs 120 \
    --dropout 0.3 \
    --weight-decay 1e-2 \
    --learning-rate 1e-3 \
    --batch-size 256 \
    --seed 42
