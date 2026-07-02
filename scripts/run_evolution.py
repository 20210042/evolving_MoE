#!/usr/bin/env python3
"""Meta-agent evolution driver (YAML config + CLI overrides)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
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

from action_selector import ActionGateConfig
from agents.base import Agent
from data.loader import get_dataset
from orchestrator import GMEvolutionOrchestrator
from roster import save_roster
from utils.llm import LLMService, llm_service_from_yaml_config


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume evolution from the last recorded step in results_dir/run_id/evolution_log.jsonl",
    )
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
    action_cfg = ActionGateConfig(
        lambda_size=float(gate.get("lambda_size", 0.05)),
        scale=float(gate.get("scale", 0.5)),
    )

    logging.info("Initializing LLM: %s", model)
    llm = llm_service_from_yaml_config(model, cfg)
    agent = Agent(llm, role="GM_Orchestrator")

    roster_init_path = pick("roster_init_path", None) or str(ROOT / "configs" / "roster_init.json")

    resume_info = None
    if args.resume:
        log_file = os.path.join(args.results_dir, run_id, "evolution_log.jsonl")
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                try:
                    last_record = json.loads(lines[-1])
                    resume_step = last_record["step"]
                    resume_epoch = last_record["epoch"]
                    resume_batch_idx = last_record["batch_idx"]
                    
                    roster_snapshot_path = os.path.join(args.results_dir, run_id, f"roster_step_{resume_step}.json")
                    if os.path.exists(roster_snapshot_path):
                        logging.info("Resuming from step %d (Epoch %d, Batch %d)", resume_step, resume_epoch, resume_batch_idx)
                        os.makedirs(os.path.dirname(os.path.abspath(roster_path)), exist_ok=True)
                        shutil.copyfile(roster_snapshot_path, roster_path)
                        resume_info = {
                            "step": resume_step,
                            "epoch": resume_epoch,
                            "batch_idx": resume_batch_idx
                        }
                    else:
                        logging.warning("Roster snapshot not found at %s. Cannot resume.", roster_snapshot_path)
                except Exception as e:
                    logging.error("Failed to parse log file for resume: %s", e)
        else:
            logging.info("Resume requested but no log file found at %s. Starting fresh.", log_file)

    if not args.resume and roster_path:
        init_src = Path(roster_init_path)
        if not init_src.is_file():
            init_src = ROOT / roster_init_path
        if init_src.is_file():
            os.makedirs(os.path.dirname(os.path.abspath(roster_path)) or ".", exist_ok=True)
            shutil.copyfile(init_src, roster_path)
            logging.info("Seeded roster from %s → %s", init_src, roster_path)
        else:
            logging.warning("roster_init not found at %s; using ensure_roster default", init_src)

    orchestrator = GMEvolutionOrchestrator(
        agent,
        roster_path,
        action_cfg=action_cfg,
        max_refine_iters=int(cfg.get("max_refine_iters", 2)),
        lcb_timeout=int(cfg.get("lcb_timeout", 10)),
        lcb_release_version=str(cfg.get("lcb_release_version", "release_v5")),
        code_exec_timeout=float(cfg.get("code_exec_timeout", 3.0)),
        war_tiebreak=str(cfg.get("war_tiebreak", "random")),
        max_lives=int(cfg.get("max_lives", 3)),
        results_dir=args.results_dir,
        run_id=run_id,
        dataset_name=dataset,
        seed=seed,
        enable_thinking=bool(cfg.get("enable_thinking", True)),
        use_exclusive_solves=bool(cfg.get("use_exclusive_solves", False)),
        use_approach_persona=bool(cfg.get("use_approach_persona", False)),
        shared_contribution_exemption=bool(cfg.get("shared_contribution_exemption", True)),
        failure_mode_scout=bool(cfg.get("failure_mode_scout", False)),
        deletion_window=int(cfg.get("deletion_window", 0)),
        deletion_floor=float(cfg.get("deletion_floor", 0.0)),
        delete_cooldown=int(cfg.get("delete_cooldown", 0)),
        add_only=bool(cfg.get("add_only", False)),
    )

    step_count = 0
    resume_step = 0
    resume_epoch = 0
    resume_batch_idx = 0
    if resume_info:
        step_count = resume_info["step"]
        resume_step = resume_info["step"]
        resume_epoch = resume_info["epoch"]
        resume_batch_idx = resume_info["batch_idx"]

    for epoch in range(epochs):
        logging.info("Starting Epoch %s/%s", epoch + 1, epochs)
        epoch_train = list(train_data)
        rng.shuffle(epoch_train)
        
        if epoch + 1 < resume_epoch:
            logging.info("Skipping Epoch %d (already completed)", epoch + 1)
            continue
            
        for i in range(0, len(epoch_train), batch_size):
            batch_idx = i // batch_size + 1
            if epoch + 1 == resume_epoch and batch_idx <= resume_batch_idx:
                logging.info("Skipping Batch %d of Epoch %d (already processed)", batch_idx, epoch + 1)
                continue
                
            batch = epoch_train[i : i + batch_size]
            step_count += 1
            orchestrator.set_log_coords(step_count, epoch + 1, batch_idx)
            logging.info("==== Step %s (Epoch %s, Batch %s) ====", step_count, epoch + 1, batch_idx)
            orchestrator.run_epoch(batch)

    save_roster(roster_path, orchestrator.roster)
    logging.info("Evolution complete. Steps=%s. Final roster → %s", step_count, roster_path)


if __name__ == "__main__":
    main()
