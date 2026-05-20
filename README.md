# Meta-Agent Evolution & Inference

Train an **evolving critic roster** on coding benchmarks (MBPP, HumanEval, LiveCodeBench), then run **manager routing + refinement** for evaluation.

## Setup

```bash
cd MetaAgentEvolution_Release
pip install -e ".[dev]"
export PYTHONPATH="$PWD/src"
```

**LiveCodeBench** scorer needs `lcb_runner` (install the benchmark repo or set `LIVECODEBENCH_PATH`).

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
