#!/bin/bash
#SBATCH --job-name=mae_evolve_bigmath
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# Seed 20210009: Thinking ON, exclusive_solves for scout, no NON-REDUNDANCY/ATOMICITY rules

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env
print_experiment_config

RESUME_FLAG=""
if [ "${RESUME:-false}" = "true" ] || [ "${1:-}" = "--resume" ]; then
    RESUME_FLAG="--resume"
    echo "=== Resuming evolution from checkpoint ==="
fi

python scripts/run_evolution.py \
    --config configs/bigmath_train_seed09.yaml \
    --seed "${SEED}" \
    --roster_path "${RESULTS_DIR}/roster_final.json" \
    --results_dir "${RESULTS_DIR}" \
    --run_id "${RUN_ID}" \
    ${RESUME_FLAG}

echo "=== Evolution done. Snapshots under ${ROSTER_SNAPSHOT_DIR}/roster_step_*.json ==="
