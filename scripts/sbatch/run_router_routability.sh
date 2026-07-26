#!/bin/bash
#SBATCH --job-name=router_routability
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# 분할 조건별 라우팅 가능성(realization rate) 비교 — QASC 926 solve 매트릭스 위 5-fold CV.
# 입력 특징 세 종류를 모두 돌려 특징 의존성을 배제한다.
# 사용: sbatch scripts/sbatch/run_router_routability.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh; conda activate evolving_moe
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

for FEAT in emb hs_last hs_mean; do
    echo "############ feat=${FEAT} ############"
    python scripts/router_routability.py --feat "${FEAT}"
done
