#!/bin/bash
#SBATCH --job-name=mae_evolve_math
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

# Generic math evolution runner. Driven by env: SEED, DATASET, EVOL_CONFIG.
# results/<DATASET>/seed<SEED>/ 경로는 common_bigmath.sh가 DATASET로 구성.

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
# shellcheck source=scripts/sbatch/common_bigmath.sh
source "${REPO}/scripts/sbatch/common_bigmath.sh"
setup_job_env
print_experiment_config
echo "  EVOL_CONFIG=${EVOL_CONFIG}  DATASET=${DATASET}"

RESUME_FLAG=""
if [ "${RESUME:-false}" = "true" ] || [ "${1:-}" = "--resume" ]; then
    RESUME_FLAG="--resume"
    echo "=== Resuming evolution from checkpoint ==="
fi

python scripts/run_evolution.py \
    --config "${EVOL_CONFIG}" \
    --seed "${SEED}" \
    --roster_path "${RESULTS_DIR}/roster_final.json" \
    --results_dir "${RESULTS_DIR}" \
    --run_id "${RUN_ID}" \
    ${RESUME_FLAG}

echo "=== Evolution done. Snapshots under ${ROSTER_SNAPSHOT_DIR}/roster_step_*.json ==="
