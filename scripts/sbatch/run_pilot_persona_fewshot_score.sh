#!/bin/bash
#SBATCH --job-name=pilot_persona_score
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# persona/few-shot 파일럿(B/C) 코드실행 채점 — GPU 불필요, CPU만.
# BUCKET=all-fail(기본)|contested — gen 파일명 접미사와 맞춰야 함.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc; source ~/data/miniconda3/etc/profile.d/conda.sh; conda activate evolving_moe
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"

BUCKET="${BUCKET:-all-fail}"
SUFFIX=""
[ "${BUCKET}" != "all-fail" ] && SUFFIX="_${BUCKET}"

for TAG in B_persona C_persona_fewshot; do
    echo "=== scoring ${TAG}${SUFFIX} ==="
    python scripts/score_binning.py \
        --input "results/pilot_persona_fewshot/gen_${TAG}${SUFFIX}.jsonl" \
        --dataset acc --split test --data_dir export/acc_v2
done
echo "=== pilot_persona_score 완료 (bucket=${BUCKET}) ==="
