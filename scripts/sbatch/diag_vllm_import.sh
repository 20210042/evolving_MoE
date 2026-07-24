#!/bin/bash
#SBATCH --job-name=diag_vllm
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --exclude=n05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# tag_qasc_topics segfault bisect: PYTHONPATH=src 여부 × import 단계별 분리 실행.

set -uo pipefail

REPO="${REPO:-$SLURM_SUBMIT_DIR}"
cd "$REPO"

source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate evolving_moe

export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"
export PYTHONUNBUFFERED=1

step() { echo "--- $1"; shift; "$@" ; echo "    exit=$?"; }

step "bare python"                 python -c "print('ok')"
step "torch (no PYTHONPATH)"       env -u PYTHONPATH python -c "import torch; print('ok', torch.__version__)"
step "vllm (no PYTHONPATH)"        env -u PYTHONPATH python -c "import vllm; print('ok', vllm.__version__)"
export PYTHONPATH="$REPO/src"
step "torch (PYTHONPATH=src)"      python -c "import torch; print('ok')"
step "transformers (PYTHONPATH=src)" python -c "import transformers; print('ok', transformers.__version__)"
step "vllm (PYTHONPATH=src)"       python -c "import vllm; print('ok')"
step "utils.llm (PYTHONPATH=src)"  python -c "from utils.llm import llm_service_from_yaml_config; print('ok')"
step "yaml first (script order)"   python -c "import yaml, torch, vllm; print('ok')"

echo "=== diag done ==="
