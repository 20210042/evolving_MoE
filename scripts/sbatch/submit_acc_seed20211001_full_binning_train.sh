#!/bin/bash
#SBATCH --job-name=acc_train_bin
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# seed20211001 (Soft WAR) 최종 5인 로스터로 train 코퍼스 전수 (11,097 문제) 라벨링
# MoE 전문가 SFT 학습 데이터셋 분할을 위한 전수 binning
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate evolving_moe

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER=1
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"

SEED=20211001
DATASET="acc"
SPLIT="train"
CONFIG="configs/acc_eval_a4b_v2_sampled.yaml"
ROSTER="results/acc/seed${SEED}/roster_final.json"
RESDIR="results/acc/seed${SEED}"
OUT="${RESDIR}/binning_train_full.jsonl"
IBS=64

echo "=== [1/3] train 코퍼스 전수 라벨링 (5인 전문가 x 11,097문제) ==="
python scripts/run_inference.py \
    --config "${CONFIG}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --data_dir export/acc_v2 \
    --pipeline binning \
    --roster_path "${ROSTER}" \
    --seed "${SEED}" \
    --infer_batch_size "${IBS}" \
    --output_file "${OUT}"

echo "=== [2/3] 정오답 및 해결 전문가 자동 채점 (score_binning) ==="
python scripts/score_binning.py \
    --input "${OUT}" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --data_dir export/acc_v2

echo "=== [3/3] MoE SFT 분할용 인덱스 추출 (export_binning_solve_index) ==="
python scripts/export_binning_solve_index.py \
    --input "${OUT%.jsonl}.binned.jsonl" \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --data_dir export/acc_v2

echo "=== Train전수 라벨링 완료 -> ${OUT%.jsonl}.binned.agent_solves.json ==="
