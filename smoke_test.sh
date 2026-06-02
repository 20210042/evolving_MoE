#!/bin/bash
#SBATCH --job-name=smoke_test_release
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.out
#SBATCH --error=/home/jaehoonjeong/data/MetaAgentEvolution_Release/logs/%x.%j.err

set -euo pipefail
REPO=/home/jaehoonjeong/data/MetaAgentEvolution_Release
# shellcheck source=scripts/sbatch/common.sh
source "${REPO}/scripts/sbatch/common.sh"
setup_job_env

python scripts/run_evolution.py \
    --config configs/mbpp_train.yaml \
    --batch_size 2 \
    --train_size 4 \
    --epochs 1 \
    --seed 0 \
    --roster_path results/smoke_roster.json \
    --results_dir results/smoke

python scripts/run_inference.py \
    --dataset mbpp \
    --roster_path results/smoke_roster.json \
    --output_file results/smoke/inference.jsonl \
    --seed 0 \
    || true

echo "Smoke done."
