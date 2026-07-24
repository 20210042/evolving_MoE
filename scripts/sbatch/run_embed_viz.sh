#!/bin/bash
#SBATCH --job-name=embed_viz
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4엥8:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# 분석 단계(CPU, 반복 가능): 캐시된 임베딩으로 t-SNE + HDBSCAN vs human prior vs
# expert-solve 3색칠 패널 + expert facet + ARI/NMI. 출력: results/embed_viz/
# 선행: scripts/sbatch/run_embed_stage.sh (GPU, 1회성)

set -euo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"

export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"

srun --ntasks=1 --chdir="$REPO" \
    python scripts/embed_expert_viz.py --stage analyze --datasets ${DATASETS:-qasc taco lbox}

echo "=== embed_viz 분석 완료: results/embed_viz/ ==="
