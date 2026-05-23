#!/bin/bash
# Shared SLURM helpers for LCB experiments.

: "${REPO:=/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
: "${CONDA_SH:=/data5/jaehoonjeong/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=pro6000}"
: "${SEED:=20210044}"
: "${TRAIN_SIZE:=380}"
: "${BATCH_SIZE:=25}"
: "${MAX_EPOCHS:=3}"
: "${MAX_REFINE_ITERS:=2}"

RESULTS_DIR="results/lcb/seed${SEED}"
RUN_ID="lcb/seed${SEED}"
ROSTER_SNAPSHOT_DIR="${RESULTS_DIR}/${RUN_ID}"
INIT_ROSTER="${REPO}/configs/roster_init.json"

setup_job_env() {
    cd "${REPO}"
    # shellcheck source=/dev/null
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    export PYTHONPATH="${REPO}/src"
    mkdir -p "${REPO}/logs"
}

steps_per_epoch() {
    echo $(( (TRAIN_SIZE + BATCH_SIZE - 1) / BATCH_SIZE ))
}

epoch_end_step() {
    local epoch="$1"
    echo $(( $(steps_per_epoch) * epoch ))
}

print_experiment_config() {
    echo "=== Experiment ==="
    echo "  SEED=${SEED}  results=${RESULTS_DIR}"
    echo "  hyperparams: configs/base.yaml + configs/lcb_train.yaml"
    echo "  TRAIN_SIZE=${TRAIN_SIZE}  BATCH_SIZE=${BATCH_SIZE}  MAX_EPOCHS=${MAX_EPOCHS}"
    echo "  steps_per_epoch=$(steps_per_epoch)"
}
