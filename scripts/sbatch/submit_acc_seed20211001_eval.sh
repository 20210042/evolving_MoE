#!/bin/bash
#SBATCH --job-name=acc_soft_eval
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# seed20211001 (Soft WAR) 진화 완료된 roster_final.json (5인) 전수 평가 및 라벨링
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate evolving_moe

export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"

SEED=20211001
CONFIG="configs/acc_eval_a4b_v2_sampled.yaml"
ROSTER="results/acc/seed${SEED}/roster_final.json"
RESDIR="results/acc/seed${SEED}"
OUT="${RESDIR}/inference_test_binning_sampled.jsonl"

echo "=== [1/2] soft-WAR evolved roster (${SEED}) 전수 추론 (5인 전문가 x Test set) ==="
python scripts/run_inference.py \
    --config "${CONFIG}" \
    --dataset acc \
    --split test \
    --seed "${SEED}" \
    --data_dir export/acc_v2 \
    --pipeline binning \
    --roster_path "${ROSTER}" \
    --output_file "${OUT}"

echo "=== [2/2] 전문가별 해결 결과 라벨링 & 채점 (score_binning) ==="
python scripts/score_binning.py \
    --input "${OUT}" \
    --dataset acc --split test --data_dir export/acc_v2

echo "=== 전수 라벨링 및 채점 완료: ${OUT%.jsonl}.binned.jsonl ==="
