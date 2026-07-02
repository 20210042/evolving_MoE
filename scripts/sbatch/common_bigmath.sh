#!/bin/bash
# Shared SLURM helpers for BigMath. Model / vLLM / data_dir -> configs/base.yaml.

: "${REPO:=/home/jaehoonjeong/data/MetaAgentEvolution_Release}"
: "${CONDA_SH:=/data5/jaehoonjeong/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV:=evolving_moe}"
: "${SEED:=20210044}"
: "${TRAIN_SIZE:=300}"
: "${BATCH_SIZE:=50}"
: "${MAX_EPOCHS:=5}"
: "${DATASET:=bigmath}"   # results/<DATASET>/ 경로·평가 dataset 키. NuminaMath 등은 submit에서 override.

RESULTS_DIR="results/${DATASET}/seed${SEED}"
RUN_ID="${DATASET}/seed${SEED}"
ROSTER_SNAPSHOT_DIR="${RESULTS_DIR}/${RUN_ID}"
INIT_ROSTER="${REPO}/configs/roster_init.json"

setup_job_env() {
    cd "${REPO}"
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    export PYTHONPATH="${REPO}/src"
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export VLLM_DISABLE_FLASHINFER=1
    # HF cache MUST live on /data5 (home quota is tiny). Pin it here so the job never
    # depends on the submitting shell's env (a shell without HF_HOME re-downloads the
    # 52G model into the home quota and the engine dies at load).
    export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data5/jaehoonjeong/.cache/huggingface}"
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
    echo "=== BigMath Experiment ==="
    echo "  SEED=${SEED}  results=${RESULTS_DIR}"
    echo "  hyperparams: configs/base.yaml + configs/bigmath_train.yaml"
    echo "  TRAIN_SIZE=${TRAIN_SIZE}  BATCH_SIZE=${BATCH_SIZE}  MAX_EPOCHS=${MAX_EPOCHS}"
    echo "  steps_per_epoch=$(steps_per_epoch)"
}
