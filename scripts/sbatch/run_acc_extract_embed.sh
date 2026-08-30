#!/bin/bash
#SBATCH --job-name=extract_embed_acc
#SBATCH --gres=gpu:PRO6000:1
#SBATCH --exclude=n05
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x.%j.log
#SBATCH --error=logs/%x.%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ~/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh
conda activate evolving_moe
export HF_HOME=/data5/jaehoonjeong/.cache/huggingface
export PYTHONPATH="${SLURM_SUBMIT_DIR}/src"
# EMB_SRC를 주면 그 jsonl 하나만 뽑는다(예: 진화 train 전체 11,097).
#   EMB_SRC=export/acc_v2/acc_train.jsonl \
#   EMB_OUT_NPY=results/embed_viz_test/acc_trainfull_emb.npy \
#   EMB_OUT_IDS=results/embed_viz_test/acc_trainfull_emb_ids.json \
#     sbatch scripts/sbatch/run_acc_extract_embed.sh
EMB_SRC="${EMB_SRC:-}"
if [[ -n "${EMB_SRC}" ]]; then
    python scripts/extract_embed_acc.py --src "${EMB_SRC}" \
        --out_npy "${EMB_OUT_NPY}" --out_ids "${EMB_OUT_IDS}"
else
    python scripts/extract_embed_acc.py
fi
echo "=== acc embeddinggemma 추출 완료 ==="
