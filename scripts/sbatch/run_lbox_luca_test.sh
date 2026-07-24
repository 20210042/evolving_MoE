#!/bin/bash
#SBATCH --job-name=lbox_luca_test
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --exclude=n05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
set -euo pipefail
REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate evolving_moe
export HF_HOME=/data5/jaehoonjeong/.cache/huggingface
export PYTHONPATH="$REPO/src"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_DISABLE_FLASHINFER=1
# LUCA = gemma base(어댑터 없음) + LBOX baseline gen prompt(evaluate.py가 자동 적용). 학습만 안 한 대조군.
python src/evaluate.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --test_dataset lbox --split test --data_dir export/lbox \
  --inference_mode vllm --max_model_len 16384 --max_new_tokens 2048 \
  --output_dir results/lbox/seed20210311 --seed 20210311
echo "=== lbox LUCA(baseline gen) test done ==="
