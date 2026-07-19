#!/bin/bash
#SBATCH --job-name=moe_sweep
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

run() { echo -e "\n########## $* ##########"; python scripts/moe_merge_infer.py "$@"; }

# 배선 앵커(full 926): 병합[1,0] == 단일 top-1 재현 확인
run --k 2 --combo linear --weights 1,0 --route confidence
# 핵심 배포: top-2 linear 0.5/0.5 (confidence 라우팅)
run --k 2 --combo linear --route confidence
# 대안: cat(rank-32 union, 간섭 최소)
run --k 2 --combo cat --route confidence
# 대안: confidence 비례 가중
run --k 2 --combo linear --weights conf --route confidence
# 대안: top-3 병합
run --k 3 --combo linear --route confidence
# 라우팅 상한: oracle pair 병합(병합 자체 최대치)
run --k 2 --combo linear --route oracle

echo "=== moe_sweep 완료 ==="
