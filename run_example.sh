# """Handoff note: Example end-to-end local pipeline command sequence. It documents the intended order:
# load raw benchmarks, filter algorithmic tasks, deduplicate, prepare execution records, and validate
# the final JSONL output."""
cd /home/minjikim/data/algorithm

mkdir -p logs

cat > run.sh <<'EOF'
#!/bin/bash
#SBATCH --job-name=algorithm_run
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=master
#SBATCH --time=0-12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/minjikim/data/algorithm/logs/%x_%j.out
#SBATCH --error=/home/minjikim/data/algorithm/logs/%x_%j.err

source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agent

export CUDA_VISIBLE_DEVICES=""

cd /home/minjikim/data/algorithm

python test.py
EOF