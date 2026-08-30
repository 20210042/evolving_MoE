#!/bin/bash
#SBATCH --job-name=pilot_persona_fewshot
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# persona/few-shot 재도입 파일럿 (LoRA 없이 순수 프롬프팅). B/C 조건 생성만 하고 채점은 안 함.
# BUCKET=all-fail(기본)|contested
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh; conda activate evolving_moe
export HF_HOME=/data5/jaehoonjeong/.cache/huggingface
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/pilot_persona_fewshot_gen.py --bucket "${BUCKET:-all-fail}"
echo "=== pilot_persona_fewshot 완료 (bucket=${BUCKET:-all-fail}) ==="
