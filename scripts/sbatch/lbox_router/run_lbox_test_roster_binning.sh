#!/bin/bash
#SBATCH --job-name=lbox_test_roster_binning
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/lbox_router/%x.%j.log
#SBATCH --error=logs/lbox_router/%x.%j.log

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
RESULTS_ROOT="${RESULTS_ROOT:?RESULTS_ROOT is required}"
MAPPING="${MAPPING:-results/lbox_binning_seed20210311/agent_mapping.json}"
cd "$REPO"

source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-MoE}"
export PYTHONPATH="$REPO/src:$REPO/scripts"
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p logs/lbox_router "$RESULTS_ROOT"
ROSTER="$RESULTS_ROOT/roster_pre_sft.json"
RAW="$RESULTS_ROOT/binning_test_pre_sft.jsonl"
LABELS="$RESULTS_ROOT/binning_test_pre_sft.binned.jsonl"

python scripts/lbox_router/build_roster_from_agent_mapping.py \
    --mapping "$MAPPING" \
    --output "$ROSTER" \
    --expected-experts 10

srun --chdir="$REPO" python scripts/run_inference.py \
    --config configs/lbox_router/lbox_eval_a4b_test_binning.yaml \
    --dataset lbox \
    --split test \
    --data_dir export/lbox \
    --pipeline binning \
    --roster_path "$ROSTER" \
    --seed 20210311 \
    --infer_batch_size 64 \
    --ignore_test_ids \
    --output_file "$RAW"

python scripts/score_binning.py \
    --input "$RAW" \
    --dataset lbox \
    --split test \
    --data_dir export/lbox \
    --workers 2

python scripts/export_binning_solve_index.py \
    --input "$LABELS" \
    --dataset lbox \
    --split test \
    --data_dir export/lbox
