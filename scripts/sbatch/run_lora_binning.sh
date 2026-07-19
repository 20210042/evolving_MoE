#!/bin/bash
#SBATCH --job-name=lora_binning
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh; conda activate evolving_moe
export HF_HOME=/data5/jaehoonjeong/.cache/huggingface
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
python scripts/generate_lora_binning.py --split validation ${LIMIT:+--limit $LIMIT}
if [ -z "${LIMIT:-}" ]; then
  python scripts/score_binning.py --input results/qasc/seed20210211/inference_validation_lora13.jsonl --dataset qasc --split validation --data_dir export/qasc
fi
echo "=== lora_binning 완료 ==="
