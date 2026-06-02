#!/bin/bash
#SBATCH --job-name=mae_evolve_lcb
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# Step 1/2 — LCB evolution (380 train, 500 test held-out)
# Same settings as MBPP: Llama-3.1-8B, batch=50, epochs=3

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_lcb.sh
source "${REPO}/scripts/sbatch/common_lcb.sh"
setup_job_env
print_experiment_config

RESUME_FLAG=""
if [ "${RESUME:-false}" = "true" ] || [ "${1:-}" = "--resume" ]; then
    RESUME_FLAG="--resume"
    echo "=== Resuming evolution from checkpoint ==="
fi

python scripts/run_evolution.py \
    --config configs/lcb_train.yaml \
    --seed "${SEED}" \
    --roster_path "${RESULTS_DIR}/roster_final.json" \
    --results_dir "${RESULTS_DIR}" \
    --run_id "${RUN_ID}" \
    ${RESUME_FLAG}

echo "=== Evolution done. Snapshots under ${ROSTER_SNAPSHOT_DIR}/roster_step_*.json ==="
