#!/bin/bash
# """Handoff note: SLURM wrapper for validation_normalized_label.py. Use it after normalized-label
# predictions exist to create inspection CSV/JSON reports for label quality checks."""
#SBATCH --job-name=validation_normalized_label
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=master
#SBATCH --time=0-02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/minjikim/minji_link/code/benchmark/src/acc_data_pipeline/labelling/logs/%x_%j.out
#SBATCH --error=/home/minjikim/minji_link/code/benchmark/src/acc_data_pipeline/labelling/logs/%x_%j.err

source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate agent

if [ -z "${OPENAI_API_KEY}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set."
  echo "Set it first, then resubmit. Example: export OPENAI_API_KEY=\"sk-...\""
  exit 1
fi

export CUDA_VISIBLE_DEVICES=""

PROJECT_ROOT="/home/minjikim/minji_link/code/benchmark"
SCRIPT_DIR="${PROJECT_ROOT}/src/acc_data_pipeline/labelling"
INPUT_CSV="${PROJECT_ROOT}/data/processed/04_execution_ready.csv"
OUTPUT_DIR="${PROJECT_ROOT}/data/labelling"
LIMIT="${LIMIT:-1000}"
WORKERS="${WORKERS:-4}"

mkdir -p "${OUTPUT_DIR}"

cd "${SCRIPT_DIR}"

python validation_normalized_label.py \
  --input "${INPUT_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --limit "${LIMIT}" \
  --workers "${WORKERS}"
