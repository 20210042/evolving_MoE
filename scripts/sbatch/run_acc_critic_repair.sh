#!/bin/bash
#SBATCH --job-name=acc_critic_repair
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --exclude=n05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# 진화 페르소나가 solver가 아니라 critic으로서 값이 있는가 (scripts/critic_repair_probe.py).
#   arm: redraw(재시행만, 대조군) / critic_luca(일반 비평자) / critic_persona(다른 expert가 비평)
#   비평자만 다르고 수정 단계 프롬프트는 동일. 실행 피드백은 프롬프트에 넣지 않는다(채점은 사후).
# Env: N_CELLS

set -euo pipefail
REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_bigmath.sh
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env

python scripts/critic_repair_probe.py \
    --n_cells "${N_CELLS:-500}" \
    --out results/acc/seed20211004/critic_repair_probe.md
echo "=== done ==="
