#!/usr/bin/env python3
"""Inference entry point — supports evolved roster, raw, and self-refine baselines."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meta_agent_evo.agents.base import Agent
from meta_agent_evo.data.loader import get_dataset
from meta_agent_evo.pipelines.routing_inference import GMRoutingPipeline
from meta_agent_evo.utils.llm import llm_service_from_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference: evolved roster / raw / self-refine")
    parser.add_argument("--model", type=str, default=None, help="Overrides configs/base.yaml model id")
    parser.add_argument("--dataset", type=str, default="mbpp")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument(
        "--pipeline",
        type=str,
        default="evolved",
        choices=["evolved", "raw", "self-refine"],
        help="evolved=GMRoutingPipeline (roster required), raw=1-pass, self-refine=2-turn",
    )
    parser.add_argument(
        "--roster_path",
        type=str,
        default=None,
        help="Path to roster JSON — required for 'evolved' pipeline",
    )
    parser.add_argument("--output_file", type=str, default="results/inference_output.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_refine_iters", type=int, default=None)
    parser.add_argument("--config", type=str, default=None, help="Optional YAML override")
    args = parser.parse_args()

    if args.pipeline == "evolved" and args.roster_path is None:
        parser.error("--roster_path is required when --pipeline=evolved")

    base_cfg_path = ROOT / "configs" / "base.yaml"
    cfg = {}
    if yaml is not None and base_cfg_path.is_file():
        with open(base_cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    if args.config and yaml is not None:
        extra_path = Path(args.config)
        if extra_path.is_file():
            with open(extra_path, "r", encoding="utf-8") as f:
                over = yaml.safe_load(f) or {}
                cfg.update({k: v for k, v in over.items() if v is not None})

    max_refine_iters = (
        args.max_refine_iters if args.max_refine_iters is not None else int(cfg.get("max_refine_iters", 2))
    )

    model_name = args.model or cfg.get("model")
    if not model_name:
        parser.error("Specify --model or define ``model`` in configs/base.yaml.")

    data_dir = args.data_dir or cfg.get("data_dir") or "/home/jaehoonjeong/data/MultiAgent/Data"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    logging.info("Loading dataset: %s", args.dataset)
    all_data = get_dataset(args.dataset, split=args.split, local_dir=data_dir)
    logging.info("Loaded %s problems.", len(all_data))

    # test_ids.json: evolution 홀드아웃 목록 필터링
    test_paths = [
        Path(args.output_file).resolve().parent / "test_ids.json",
        Path("results") / f"mbpp/seed{args.seed}" / "test_ids.json",
        Path("results") / "test_ids.json",
    ]
    test_ids_path = next((p for p in test_paths if p.is_file()), None)
    if test_ids_path is not None:
        raw = json.loads(test_ids_path.read_text(encoding="utf-8"))
        test_ids = {str(x) for x in raw}
        filtered = [d for d in all_data if str(d["id"]) in test_ids]
        if filtered:
            test_data = filtered
            logging.info("Filtered to %s problems via %s", len(test_data), test_ids_path)
        elif test_ids:
            logging.warning(
                "test_ids.json (%s) lists %s ids but none match %s split ids — "
                "wrong dataset or stale holdout file; using full split (%s problems).",
                test_ids_path,
                len(test_ids),
                args.dataset,
                len(all_data),
            )
            test_data = all_data
        else:
            test_data = all_data
            logging.info("test_ids.json is empty; using all %s problems.", len(test_data))
    else:
        test_data = all_data
        logging.info("No test_ids.json found; using all %s problems.", len(test_data))

    llm = llm_service_from_yaml_config(str(model_name), cfg)
    agent = Agent(llm, role="Inference_Agent")

    if args.pipeline == "raw":
        from meta_agent_evo.pipelines.baselines import RawPipeline

        pipeline = RawPipeline(agent, domain="coding")
        logging.info("Pipeline: Raw (1-pass, no persona)")
    elif args.pipeline == "self-refine":
        from meta_agent_evo.pipelines.baselines import SelfRefinePipeline

        pipeline = SelfRefinePipeline(agent, domain="coding", max_refine_iters=max_refine_iters)
        logging.info("Pipeline: Self-Refine (%d iters, no persona)", max_refine_iters)
    else:
        pipeline = GMRoutingPipeline(
            agent,
            scouting_report_path=args.roster_path,
            domain="coding",
            routing_memory_path=str(Path(args.output_file).resolve().parent / "routing_memory.json"),
            max_refine_iters=max_refine_iters,
        )
        logging.info("Pipeline: Evolved GMRoutingPipeline (roster=%s)", args.roster_path)

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for idx, item in enumerate(test_data, start=1):
            logging.info("Processing %s/%s: %s", idx, len(test_data), item["id"])
            result = pipeline.run(item)
            result["dataset"] = args.dataset
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    logging.info("Inference complete → %s", args.output_file)


if __name__ == "__main__":
    main()
