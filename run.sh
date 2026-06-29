#!/bin/bash
# """Handoff note: SLURM batch script for converting the final execution-ready JSONL into the
# labeling/review CSV. It assumes the `agent` conda environment, runs only under SLURM, and writes its
# report to data/reports/04_execution_ready_csv_report.json."""
#SBATCH --job-name=acc_data_pipeline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=master
#SBATCH --time=0-12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/minjikim/minji_link/code/benchmark/logs/%x_%j.out
#SBATCH --error=/home/minjikim/minji_link/code/benchmark/logs/%x_%j.err

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "Refusing to run outside SLURM. Submit with: sbatch run.sh" >&2
  exit 1
fi

if [ "$(hostname -s)" = "master" ]; then
  echo "Refusing to run on master. Submit with: sbatch run.sh" >&2
  exit 1
fi

source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate agent || true
elif [ -f /data/minjikim/miniconda3/etc/profile.d/conda.sh ]; then
  source /data/minjikim/miniconda3/etc/profile.d/conda.sh
  conda activate agent || true
fi

export CUDA_VISIBLE_DEVICES=""

cd /home/minjikim/minji_link/code/benchmark
BENCHMARK_ROOT="$(pwd -P)"
mkdir -p data/processed data/reports data/intermediate logs

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python -m acc_data_pipeline.preprocessing.export_execution_ready_csv \
  --input data/processed/04_execution_ready.jsonl \
  --output data/processed/04_execution_ready.csv \
  --report data/reports/04_execution_ready_csv_report.json
