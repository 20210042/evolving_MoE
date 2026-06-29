#!/bin/bash
# """Handoff note: SLURM wrapper for normalized_label.py. It performs taxonomy cleanup and LiteLLM-backed
# filling for missing original_domain labels, so it may require model credentials in the environment."""
#SBATCH --job-name=normalized_labeling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=master
#SBATCH --time=0-04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/minjikim/minji_link/code/benchmark/src/acc_data_pipeline/labelling/logs/%x_%j.out
#SBATCH --error=/home/minjikim/minji_link/code/benchmark/src/acc_data_pipeline/labelling/logs/%x_%j.err

source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate agent

# OpenAI API key must be registered before job submission.
# Example (set in shell/profile/secret manager): export OPENAI_API_KEY="sk-..."
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
if [ -n "${RESUME_JOB_ID}" ]; then
  OUTPUT_CSV="${OUTPUT_DIR}/04_execution_ready_normalized_labels_${RESUME_JOB_ID}.csv"
else
  OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/04_execution_ready_normalized_labels_${SLURM_JOB_ID}.csv}"
fi

mkdir -p "${OUTPUT_DIR}"

cd "${SCRIPT_DIR}"

echo "Input CSV: ${INPUT_CSV}"
echo "Output CSV: ${OUTPUT_CSV}"

python - <<PY
from normalized_label import process_pipeline

input_csv = "${INPUT_CSV}"
output_csv = "${OUTPUT_CSV}"

process_pipeline(input_csv, output_csv)
print(f"Saved: {output_csv}")
PY
