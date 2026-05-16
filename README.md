# Meta-Agent Evolution & Inference

Train an **evolving critic roster** on coding benchmarks (LiveCodeBench, MBPP, HumanEval), then run **manager routing + refinement** for evaluation.

## Setup

```bash
cd MetaAgentEvolution_Release
pip install -e ".[dev]"   # or: pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

**LiveCodeBench** official scorer expects the `lcb_runner` package. Install the benchmark repo next to this project or set:

```bash
export LIVECODEBENCH_PATH=/path/to/LiveCodeBench
```

Default lookup: `./LiveCodeBench`, `../MultiAgent/LiveCodeBench`.

## Evolution (training)

```bash
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

Uses `results/evolution_test_ids.json` when present (same directory as `--output_file`’s parent, or cwd).

## Analysis

```bash
python scripts/analyze_evolution.py results/mbpp_seed42/evolution_log.jsonl
```

## Layout

- **`src/meta_agent_evo/`** — main package (`orchestrator`, `action_selector`, `evaluation`, `data`, `pipelines`).
- **`legacy/`** — archived experiments (`ours.py`, `baselines.py`, Jina helpers, old `main.py`); not imported by default.
- **`configs/`** — YAML defaults and per-dataset overrides.

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -q
```
