#!/bin/bash
#SBATCH --job-name=acc_tokstat
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log

# acc SFT 타깃(solution)의 토큰 길이 분포 — 배포평가 max_new_tokens를 실측으로 정하기 위함.
# 사용: sbatch scripts/sbatch/diag_acc_token_len.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-evolving_moe}"
export HF_HOME="${HF_HOME:-/data5/jaehoonjeong/.cache/huggingface}"

python - <<'PY'
import json
import numpy as np
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
sol = [str(json.loads(l).get("solution") or "")
       for l in open("export/acc/acc_train.jsonl", encoding="utf-8")]
L = np.array([len(tok.encode(s, add_special_tokens=False)) for s in sol])
print(f"SFT completion(solution) token lengths, n={len(L)}")
for p in (50, 75, 90, 95, 99):
    print(f"  p{p}: {int(np.percentile(L, p))}")
print(f"  mean: {int(L.mean())}   max: {int(L.max())}")
for cap in (1024, 2048, 3072, 4096, 6144, 8192):
    print(f"  covered by max_new={cap}: {100 * (L <= cap).mean():.1f}%")
PY
