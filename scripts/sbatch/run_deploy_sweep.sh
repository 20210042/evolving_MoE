#!/bin/bash
#SBATCH --job-name=deploy_sweep
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh; conda activate evolving_moe
export HF_HOME=/data5/jaehoonjeong/.cache/huggingface
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
DATASET="${DATASET:-qasc}"
: "${BINNED:?Set BINNED to the per-expert binned jsonl path.}"
: "${DENSE:?Set DENSE to the dense SFT baseline jsonl path.}"
: "${OUT:?Set OUT to the result markdown path.}"
ARGS=(--dataset "${DATASET}" --binned "${BINNED}" --dense "${DENSE}" --out "${OUT}")
if [[ -n "${CKPT:-}" ]]; then
    ARGS+=(--ckpt "${CKPT}")
fi
python scripts/moe_deploy_sweep.py "${ARGS[@]}"
echo "=== deploy_sweep 완료 ==="
