# Meta-Agent Evolution & Inference

Train an **evolving critic roster** on coding benchmarks (MBPP, HumanEval, LiveCodeBench), then run **manager routing + refinement** for evaluation.

## Setup

```bash
git clone https://github.com/20210042/evolving_MoE.git
cd evolving_MoE
conda env create -f environment.yml --name MoE
conda activate MoE
pip install -e ".[dev]"
export PYTHONPATH="$PWD/src"
```

**LiveCodeBench** scorer uses `lcb_runner`, which is **bundled at `src/lcb_runner/`** — no extra install needed.

## Config

Everything defaults from **`configs/base.yaml`** (+ **`configs/mbpp_train.yaml`** for MBPP evolution via `--config`).

Model, vLLM, sampling, `data_dir` — edit **`configs/base.yaml`** (+ dataset YAML for evolution).

## Hugging Face (Llama 3.1 gated)

1. Hub에서 `meta-llama/Meta-Llama-3.1-8B-Instruct` 라이선스 accept  
2. **한 번** `huggingface-cli login` (compute 노드에서도 같은 `$HOME`이면 끝)

추가 `export HF_TOKEN=...` **필요 없음**. 로그인으로 저장된 토큰을 vLLM/transformers가 읽습니다.

## MBPP 실험 (SLURM)

`SEED`만 바꾸고 싶으면 제출 전에 export (기본값 `20210044` → `scripts/sbatch/common.sh`).

```bash
cd MetaAgentEvolution_Release

# 0) 선택: 스모크
sbatch smoke_test.sh

# 1) Evolution
SEED=20210044 sbatch scripts/sbatch/run_mbpp_evolution.sh

# 2) Epoch eval (진화 job id 넣기)
EVOLVE_JOB=12345
SEED=20210044 sbatch --dependency=afterok:${EVOLVE_JOB} scripts/sbatch/run_mbpp_eval_epochs.sh

# 3) Baselines
SEED=20210044 sbatch scripts/sbatch/run_mbpp_baselines.sh

# Resume
SEED=20210044 RESUME=true sbatch scripts/sbatch/run_mbpp_evolution.sh
```

| Script | Purpose |
|--------|---------|
| `run_mbpp_evolution.sh` | Roster evolution → `roster_step_*.json` |
| `run_mbpp_eval_epochs.sh` | Inference + score per epoch |
| `run_mbpp_baselines.sh` | init_persona / raw / self-refine |

Epoch checkpoint: `roster_step_{steps_per_epoch × epoch}.json` (374 train, batch 50 → 8 steps/epoch).

## LCB 실험 (SLURM)

 LiveCodeBench 데이터셋으로 동일 설정(Llama-3.1-8B) 진화 실험. **380개 train / 500개 held-out test** 분할.

```bash
# 권장: dependency로 evolution 완료 후 자동으로 eval 제출
SEED=20210044 sbatch --dependency=afterok:$(sbatch --parsable scripts/sbatch/run_lcb_evolution.sh) \
    scripts/sbatch/run_lcb_eval_epochs.sh

# 또는 별도 제출
SEED=20210044 sbatch scripts/sbatch/run_lcb_evolution.sh
EVOLVE_JOB=<job_id>
SEED=20210044 sbatch --dependency=afterok:${EVOLVE_JOB} scripts/sbatch/run_lcb_eval_epochs.sh

# 다른 seed
SEED=20210042 sbatch --dependency=afterok:$(SEED=20210042 sbatch --parsable scripts/sbatch/run_lcb_evolution.sh) \
    scripts/sbatch/run_lcb_eval_epochs.sh
```

| Script | Purpose |
|--------|---------|
| `run_lcb_evolution.sh` | LCB roster evolution (380 train) |
| `run_lcb_eval_epochs.sh` | Inference + score on 500 held-out LCB |
| `common_lcb.sh` | LCB 공통 Slurm 변수 (`SEED`, `TRAIN_SIZE=380` 등) |

Epoch checkpoint: `roster_step_{steps_per_epoch × epoch}.json` (380 train, batch 50 → 8 steps/epoch).
Results 저장: `results/lcb/seed{SEED}/`

## Local debug

```bash
python scripts/run_evolution.py --config configs/mbpp_train.yaml --seed 42
python scripts/run_inference.py --dataset mbpp --roster_path results/.../roster_step_8.json --output_file out.jsonl
python scripts/score_outputs.py --input out.jsonl --dataset mbpp
```

## Qwen 캐시 정리 (전환 후, Qwen job 없을 때)

```bash
du -sh "$HOME/.cache/huggingface/hub/models--Qwen"*
rm -rf "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct"
```

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```
