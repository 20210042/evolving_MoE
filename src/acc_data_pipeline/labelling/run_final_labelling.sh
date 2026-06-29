#!/bin/bash
# """Handoff note: SLURM wrapper for final_labelling.py. Run it after normalized labels are available to
# produce the final label file and critic-category distribution artifacts."""
#SBATCH --job-name=final_labelling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=master
#SBATCH --time=0-01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/minjikim/minji_link/code/benchmark/src/acc_data_pipeline/labelling/logs/%x_%j.out
#SBATCH --error=/home/minjikim/minji_link/code/benchmark/src/acc_data_pipeline/labelling/logs/%x_%j.err

source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate agent

export CUDA_VISIBLE_DEVICES=""

PROJECT_ROOT="/home/minjikim/minji_link/code/benchmark"
SCRIPT_DIR="${PROJECT_ROOT}/src/acc_data_pipeline/labelling"
INPUT_CSV="${PROJECT_ROOT}/data/labelling/04_execution_ready_normalized_labels_163969.csv"
TAG_DISTRIBUTION_CSV="${PROJECT_ROOT}/data/reports/tag_distribution.csv"
OUTPUT_DIR="${PROJECT_ROOT}/data/labelling"

if [ -n "${FINAL_LABEL_SUFFIX}" ]; then
  SUFFIX="${FINAL_LABEL_SUFFIX}"
elif [ -n "${SLURM_JOB_ID}" ]; then
  SUFFIX="${SLURM_JOB_ID}"
else
  SUFFIX="local"
fi

mkdir -p "${OUTPUT_DIR}"
cd "${SCRIPT_DIR}"

echo "Input CSV: ${INPUT_CSV}"
echo "Tag distribution CSV: ${TAG_DISTRIBUTION_CSV}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Suffix: ${SUFFIX}"

python final_labelling.py \
  --input-csv "${INPUT_CSV}" \
  --tag-distribution-csv "${TAG_DISTRIBUTION_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --suffix "${SUFFIX}"
