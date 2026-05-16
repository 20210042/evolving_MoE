#!/usr/bin/env python3
"""Inference with an evolved roster."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from meta_agent_evo.agents.base import Agent
from meta_agent_evo.data.loader import get_dataset
from meta_agent_evo.pipelines.routing_inference import GMRoutingPipeline
from meta_agent_evo.utils.llm import LLMService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with evolved roster")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--dataset", type=str, default="livecodebench")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data_dir", type=str, default="/home/jaehoonjeong/data/MultiAgent/Data")
    parser.add_argument("--roster_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="results/inference_output.jsonl")
    parser.add_argument("--jina_router_checkpoint", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    logging.info("Loading dataset: %s", args.dataset)
    all_data = get_dataset(args.dataset, split=args.split, local_dir=args.data_dir)
    logging.info("Loaded %s problems.", len(all_data))

    test_ids_path = os.path.join(os.path.dirname(args.output_file) or ".", "evolution_test_ids.json")
    if os.path.isfile(test_ids_path):
        with open(test_ids_path, "r", encoding="utf-8") as f:
            test_ids = set(json.load(f))
        test_data = [d for d in all_data if d["id"] in test_ids]
        logging.info("Filtered to %s problems via evolution_test_ids.json", len(test_data))
    else:
        test_data = all_data

    llm = LLMService(model_name=args.model, mode="vllm", tp_size=1)
    agent = Agent(llm, role="Inference_Agent")
    pipeline = GMRoutingPipeline(
        agent,
        scouting_report_path=args.roster_path,
        domain="coding",
        jina_router_checkpoint=args.jina_router_checkpoint,
    )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for idx, item in enumerate(test_data):
            logging.info("Processing %s/%s: %s", idx + 1, len(test_data), item["id"])
            result = pipeline.run(item)
            result["dataset"] = args.dataset
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    logging.info("Inference complete → %s", args.output_file)


if __name__ == "__main__":
    main()
