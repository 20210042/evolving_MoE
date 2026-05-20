#!/bin/bash
#SBATCH --job-name=mae_evolve_mbpp
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# Step 1/3 — MBPP train evolution (roster updates + roster_step_*.json snapshots)

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_mbpp.sh
source "${REPO}/scripts/sbatch/common_mbpp.sh"
setup_job_env
print_experiment_config

RESUME_FLAG=""
if [ "${RESUME:-false}" = "true" ] || [ "${1:-}" = "--resume" ]; then
    RESUME_FLAG="--resume"
    echo "=== Resuming evolution from checkpoint ==="
fi

python scripts/run_evolution.py \
    --config configs/mbpp_train.yaml \
    --seed "${SEED}" \
    --roster_path "${RESULTS_DIR}/roster_final.json" \
    --results_dir "${RESULTS_DIR}" \
    --run_id "${RUN_ID}" \
    ${RESUME_FLAG}

echo "=== Evolution done. Snapshots under ${ROSTER_SNAPSHOT_DIR}/roster_step_*.json ==="
