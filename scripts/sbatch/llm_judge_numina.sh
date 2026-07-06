#!/bin/bash
#SBATCH --job-name=llm_judge_numina
#SBATCH --gres=gpu:PRO6000:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

set -euo pipefail

export VLLM_USE_FLASHINFER_SAMPLER=0

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"
mkdir -p logs

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-MoE}"
conda activate "${CONDA_ENV}"

export PYTHONPATH="$REPO/src"

INPUT_PATH="${INPUT_PATH:-results/llama3_numina_cot_LUCA_LUCA/numina_cot_Llama-3.1-8B-Instruct_200410.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemma-4-26B-A4B-it}"
SPLIT="${SPLIT:-test}"
LIMIT="${LIMIT:-}"
MAX_WORKERS="${MAX_WORKERS:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${SLURM_GPUS_ON_NODE:-2}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_READY_TIMEOUT_SEC="${VLLM_READY_TIMEOUT_SEC:-2400}"
PREDOWNLOAD_MODEL="${PREDOWNLOAD_MODEL:-true}"
VLLM_PORT="${VLLM_PORT:-$((18000 + SLURM_JOB_ID % 10000))}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
RESUME="${RESUME:-true}"
OVERWRITE="${OVERWRITE:-false}"

SERVER_LOG="logs/vllm_judge_${SLURM_JOB_ID}.log"

echo "=== Numina LLM judge 시작 ==="
echo "REPO=${REPO}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "INPUT_PATH=${INPUT_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR:-input directory}"
echo "JUDGE_MODEL=${JUDGE_MODEL}"
echo "SPLIT=${SPLIT}"
echo "BASE_URL=${BASE_URL}"
echo "MAX_WORKERS=${MAX_WORKERS}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "PREDOWNLOAD_MODEL=${PREDOWNLOAD_MODEL}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"

cleanup() {
    if [ -n "${VLLM_PID:-}" ] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "Stopping vLLM server pid=${VLLM_PID}"
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [ "${PREDOWNLOAD_MODEL}" = "true" ] || [ "${PREDOWNLOAD_MODEL}" = "1" ]; then
    echo "Checking/downloading judge model cache: ${JUDGE_MODEL}"
    python - "${JUDGE_MODEL}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
path = snapshot_download(repo_id=repo_id)
print(f"Model cache ready: {path}")
PY
fi

echo "Starting vLLM server. Log: ${SERVER_LOG}"
srun --ntasks=1 --gpus-per-task="${TENSOR_PARALLEL_SIZE}" --chdir="$REPO" \
    vllm serve "${JUDGE_MODEL}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --served-model-name "${JUDGE_MODEL}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --dtype bfloat16 \
    >"${SERVER_LOG}" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM health endpoint..."
READY_POLLS=$((VLLM_READY_TIMEOUT_SEC / 10))
for _ in $(seq 1 "${READY_POLLS}"); do
    if python -c "import urllib.request; urllib.request.urlopen('http://${VLLM_HOST}:${VLLM_PORT}/health', timeout=2).read()" >/dev/null 2>&1; then
        echo "vLLM is ready."
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server exited early. Last log lines:" >&2
        tail -80 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    if grep -Eqi "out of memory|cuda error|traceback|valueerror|runtimeerror|killed|cannot allocate|no space left" "${SERVER_LOG}" 2>/dev/null; then
        echo "ERROR: vLLM log contains an error while starting. Last log lines:" >&2
        tail -120 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    sleep 10
done

if ! python -c "import urllib.request; urllib.request.urlopen('http://${VLLM_HOST}:${VLLM_PORT}/health', timeout=2).read()" >/dev/null 2>&1; then
    echo "ERROR: vLLM server did not become ready. Last log lines:" >&2
    tail -80 "${SERVER_LOG}" >&2 || true
    exit 1
fi

JUDGE_ARGS=(
    --input "${INPUT_PATH}"
    --model "${JUDGE_MODEL}"
    --base_url "${BASE_URL}"
    --split "${SPLIT}"
    --max_workers "${MAX_WORKERS}"
)

if [ -n "${OUTPUT_DIR}" ]; then
    JUDGE_ARGS+=(--output_dir "${OUTPUT_DIR}")
fi

if [ -n "${LIMIT}" ]; then
    JUDGE_ARGS+=(--limit "${LIMIT}")
fi

if [ "${RESUME}" = "true" ] || [ "${RESUME}" = "1" ]; then
    JUDGE_ARGS+=(--resume)
fi

if [ "${OVERWRITE}" = "true" ] || [ "${OVERWRITE}" = "1" ]; then
    JUDGE_ARGS+=(--overwrite)
fi

echo "Running judge..."
python "$REPO/scripts/llm_judge_numina.py" "${JUDGE_ARGS[@]}"

echo "=== Numina LLM judge 완료 ==="
