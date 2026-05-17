#!/usr/bin/env python3
"""Meta-agent evolution driver (YAML config + CLI overrides)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import yaml
except ImportError:
    yaml = None

from meta_agent_evo.action_selector import ActionGateConfig
from meta_agent_evo.agents.base import Agent
from meta_agent_evo.data.loader import get_dataset
from meta_agent_evo.orchestrator import GMEvolutionOrchestrator
from meta_agent_evo.roster import save_roster
from meta_agent_evo.utils.llm import LLMService


def load_merged_config(base: Path, extra: Path | None) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required. pip install pyyaml")
    with open(base, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if extra and extra.is_file():
        with open(extra, "r", encoding="utf-8") as f:
            over = yaml.safe_load(f)
        cfg.update({k: v for k, v in over.items() if v is not None})
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GM Evolution (evolving roster)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML override merged into configs/base.yaml (e.g. configs/mbpp_train.yaml)",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--roster_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_id", type=str, default=None, help="Subdir under results/ for logs")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    base_cfg_path = ROOT / "configs" / "base.yaml"
    extra_path = Path(args.config) if args.config else None
    cfg = load_merged_config(base_cfg_path, extra_path)

    def pick(name, cli_val, default=None):
        return cli_val if cli_val is not None else cfg.get(name, default)

    model = pick("model", args.model)
    dataset = pick("dataset", args.dataset)
    split = pick("split", args.split)
    data_dir = pick("data_dir", args.data_dir)
    roster_path = pick("roster_path", args.roster_path)
    batch_size = pick("batch_size", args.batch_size)
    train_size = pick("train_size", args.train_size)
    epochs = pick("epochs", args.epochs)
    seed = pick("seed", args.seed, 42)
    run_id = args.run_id or f"{dataset}_seed{seed}"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    logging.info("Loading dataset %s", dataset)
    all_data = get_dataset(dataset, split=split, local_dir=data_dir)
    logging.info("Total problems loaded: %s", len(all_data))

    rng = random.Random(seed)
    shuffled = list(all_data)
    rng.shuffle(shuffled)
    train_data = shuffled[:train_size]
    test_data = shuffled[train_size:]

    os.makedirs(args.results_dir, exist_ok=True)
    test_ids_path = os.path.join(args.results_dir, "test_ids.json")
    with open(test_ids_path, "w", encoding="utf-8") as f:
        json.dump([item["id"] for item in test_data], f, indent=4)
    logging.info("Held out %s test ids → %s", len(test_data), test_ids_path)

    gate = cfg.get("action_gate") or {}
    swap_max_gain_raw = gate.get("swap_max_gain", None)
    action_cfg = ActionGateConfig(
        alpha_stability=float(gate.get("alpha_stability", 1.0)),
        lambda_size=float(gate.get("lambda_size", 0.01)),
        epsilon_floor=float(gate.get("epsilon_floor", 0.05)),
        swap_max_gain=float(swap_max_gain_raw) if swap_max_gain_raw is not None else None,
        use_wilson_ci=bool(gate.get("use_wilson_ci", True)),
        wilson_confidence=float(gate.get("wilson_confidence", 0.95)),
    )

    logging.info("Initializing LLM: %s", model)
    llm = LLMService(
        model_name=model,
        mode="vllm",
        tp_size=int(cfg.get("vllm_tp_size", 1)),
    )
    agent = Agent(llm, role="GM_Orchestrator")

    orchestrator = GMEvolutionOrchestrator(
        agent,
        roster_path,
        action_cfg=action_cfg,
        max_refine_iters=int(cfg.get("max_refine_iters", 2)),
        lcb_timeout=int(cfg.get("lcb_timeout", 10)),
        lcb_release_version=str(cfg.get("lcb_release_version", "release_v5")),
        code_exec_timeout=float(cfg.get("code_exec_timeout", 3.0)),
        war_tiebreak=str(cfg.get("war_tiebreak", "random")),
        probe_stability_k=int(gate.get("probe_stability_k", 8)),
        results_dir=args.results_dir,
        run_id=run_id,
        dataset_name=dataset,
        seed=seed,
    )

    step_count = 0
    for epoch in range(epochs):
        logging.info("Starting Epoch %s/%s", epoch + 1, epochs)
        epoch_train = list(train_data)
        rng.shuffle(epoch_train)
        for i in range(0, len(epoch_train), batch_size):
            batch = epoch_train[i : i + batch_size]
            step_count += 1
            orchestrator.set_log_coords(step_count, epoch + 1, i // batch_size + 1)
            logging.info("==== Step %s (Epoch %s, Batch %s) ====", step_count, epoch + 1, i // batch_size + 1)
            orchestrator.run_epoch(batch)

    save_roster(roster_path, orchestrator.roster)
    logging.info("Evolution complete. Steps=%s. Final roster → %s", step_count, roster_path)


if __name__ == "__main__":
    main()
