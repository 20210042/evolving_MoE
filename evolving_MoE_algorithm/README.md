# Meta-Agent Evolution & Inference

Train an **evolving critic roster** on coding benchmarks (LiveCodeBench, MBPP, HumanEval), then run **manager routing + refinement** for evaluation.

## Setup

```bash
git clone https://github.com/20210042/evolving_MoE.git
cd evolving_MoE
conda env create -f environment.yml --name MoE
conda activate MoE
```

**LiveCodeBench** official scorer expects the `lcb_runner` package. Install the benchmark repo next to this project or set:

```bash
export LIVECODEBENCH_PATH=/path/to/LiveCodeBench
```

Default lookup: `./LiveCodeBench`, `../MultiAgent/LiveCodeBench`.

## Evolution (training)

```bash
# MBPP train evolution (full roster training split):
python scripts/run_evolution.py --config configs/mbpp_train.yaml --seed 42
# Or MBPP test split with smaller train_size (configs/mbpp.yaml):
python scripts/run_evolution.py --config configs/mbpp.yaml --seed 42
```

- Merges [`configs/base.yaml`](configs/base.yaml) with the dataset YAML.
- Logs each step: `results/<run_id>/evolution_log.jsonl` and `roster_step_<n>.json`.
- **Action gate**: *noop* (keep roster), *add* (append specialist), or *swap* (replace WAR-selected member) using probe-based marginal coverage.

Multi-seed / multi-dataset:

```bash
python scripts/run_multi_seed.py --datasets livecodebench mbpp humaneval --seeds 17 42 1234
```

## Inference

```bash
python scripts/run_inference.py \
  --dataset mbpp \
  --roster_path results/gm_roster_v8.json \
  --output_file results/eval.jsonl
```

Uses `test_ids.json` from evolution (`--results_dir`, or next to `--output_file`, or `results/mbpp/seed<seed>/`) when those IDs match the current dataset; otherwise it runs the full split (see script log if holdout IDs do not overlap).

## Analysis

```bash
python scripts/analyze_evolution.py results/mbpp_seed42/evolution_log.jsonl
```

## Layout

- **`src/meta_agent_evo/`** — main package (`orchestrator`, `action_selector`, `evaluation`, `data`, `pipelines`).
- **Optional `legacy/`** — if present, `routing_inference` can load a Jina checkpoint helper from `legacy/jina_router.py`; the repo may ship without this directory.
- **`configs/`** — YAML defaults and per-dataset overrides.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```
