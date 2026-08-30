#!/bin/bash
#SBATCH --job-name=qasc_ub_greedy
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# UB=pass@k 검정(2026-07-26): 기존 QASC 로스터 UB(94.60, temp=0.7/top_p=0.8로 밝혀짐)와
# 정확히 같은 roster·config로, temperature만 0(greedy)으로 바꿔 재실행.
# 원본과 비교: results/qasc/seed20210211/inference_validation_binning_final.jsonl (sampled)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh; conda activate evolving_moe
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"

RESULTS_DIR="results/qasc/seed20210211"
OUT="${RESULTS_DIR}/inference_validation_binning_final_greedy.jsonl"

echo "=== [1/2] greedy 생성: QASC 로스터 UB ==="
python scripts/run_inference.py \
    --config configs/qasc_eval_a4b_greedy.yaml \
    --dataset qasc \
    --split validation \
    --seed 20210211 \
    --data_dir export/qasc \
    --pipeline binning \
    --roster_path "${RESULTS_DIR}/roster_final.json" \
    --max_items 926 \
    --output_file "${OUT}"

echo "=== [2/2] 채점 ==="
python scripts/score_binning.py \
    --input "${OUT}" \
    --dataset qasc --split validation --data_dir export/qasc

echo "=== qasc_ub_greedy 완료 -> ${OUT%.jsonl}.binned.jsonl ==="
