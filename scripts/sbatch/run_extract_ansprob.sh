#!/bin/bash
#SBATCH --job-name=extract_ansprob
#SBATCH --gres=gpu:4090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh; conda activate pro6000
export HF_HOME=/data5/jaehoonjeong/.cache/huggingface
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
python scripts/extract_answer_logits.py --split train
python scripts/extract_answer_logits.py --split validation
echo "=== ansprob 추출 완료 ==="
