#!/bin/bash
# Set up the MoE conda environment from scratch.
# Requirements: driver 550.x (CUDA 12.4 max) → uses vLLM 0.10.0+cu118.

set -euo pipefail

CONDA_BASE="/data6/jongbinwon/miniconda3"
ENV_NAME="MoE"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VERSION="0.10.0"
CUDA_TAG="cu118"
CPU_ARCH=$(uname -m)  # x86_64
VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${CUDA_TAG}-cp38-abi3-manylinux1_${CPU_ARCH}.whl"

echo "=== Initializing conda ==="
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "=== Installing Python 3.11 into ${ENV_NAME} ==="
conda install -n "${ENV_NAME}" python=3.11 -y

conda activate "${ENV_NAME}"
echo "Python: $(which python) — $(python --version)"

echo "=== Installing uv (fast pip) ==="
pip install uv --quiet

echo "=== Installing vLLM ${VLLM_VERSION}+${CUDA_TAG} ==="
uv pip install "${VLLM_WHEEL}" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "=== Installing remaining dependencies ==="
uv pip install \
    "transformers>=4.40.0" \
    "accelerate>=1.0.0" \
    "peft>=0.10.0" \
    "trl>=0.8.0" \
    "datasets>=2.14.0" \
    "numpy>=1.26.0" \
    "pandas>=2.0.0" \
    "pyyaml>=6.0" \
    "math-verify>=0.5.0" \
    "wandb>=0.16.0"

echo "=== Installing project package (editable) ==="
uv pip install -e "${REPO}"

echo "=== Smoke test ==="
python - <<'EOF'
import torch, vllm, transformers
print(f"torch     : {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
print(f"vllm      : {vllm.__version__}")
print(f"transformers: {transformers.__version__}")
EOF

echo ""
echo "=== Done. Activate with: conda activate ${ENV_NAME} ==="
