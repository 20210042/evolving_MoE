#!/bin/bash
# Submit MBPP evolution + epoch eval for seed 20210048 (post prompt decontamination).
# Run from login node: bash scripts/sbatch/submit_mbpp_seed20210048.sh

set -euo pipefail

REPO="${REPO:-/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
SEED=20210048

cd "${REPO}"
mkdir -p logs

echo "=== MBPP seed ${SEED} (decontaminated prompts) ==="
echo "  config: configs/base.yaml + configs/mbpp_train.yaml (batch_size=50, epochs=5)"
echo "  results: results/mbpp/seed${SEED}/"
echo ""

EVOLVE_JOB="$(SEED="${SEED}" sbatch --parsable scripts/sbatch/run_mbpp_evolution.sh)"
echo "Evolution job: ${EVOLVE_JOB}  (logs: logs/mae_evolve_mbpp.${EVOLVE_JOB}.out)"

EVAL_JOB="$(SEED="${SEED}" sbatch --parsable --dependency=afterok:"${EVOLVE_JOB}" \
    scripts/sbatch/run_mbpp_eval_epochs.sh)"
echo "Eval job:      ${EVAL_JOB}  (logs: logs/mae_eval_epochs.${EVAL_JOB}.out)"

echo ""
echo "Monitor:"
echo "  tail -f logs/mae_evolve_mbpp.${EVOLVE_JOB}.out"
echo "  tail -f logs/mae_eval_epochs.${EVAL_JOB}.out"
