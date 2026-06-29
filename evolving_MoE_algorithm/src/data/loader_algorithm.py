"""ACC algorithm dataset loader and compatibility shim.

This module is the data entry point for the `_algorithm` training/evaluation path. It keeps the
original benchmark loaders available for compatibility, but its main job is to load local
`data/acc_algorithm/acc_algorithm_{split}.jsonl` files, annotate scoring metadata, apply critic
category filters, and optionally subsample examples with a deterministic seed."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset

DEFAULT_ACC_ALGORITHM_DIR = "/home/minjikim/minji_link/evolving_MoE/data/acc_algorithm"


def scoring_kind_for_dataset(name: str) -> str:
    n = name.lower()
    if n == "livecodebench":
        return "lcb"
    if n == "humaneval":
        return "humaneval_check"
    if n == "acc_algorithm":
        return "stdin_stdout"
    return "asserts"


def annotate_items(items: List[Dict[str, Any]], dataset_key: str) -> List[Dict[str, Any]]:
    sk = scoring_kind_for_dataset(dataset_key)
    for it in items:
        it["dataset"] = it.get("dataset") or dataset_key.lower()
        it["scoring_kind"] = it.get("scoring_kind") or sk
    return items


def load_humaneval(split: str = "test") -> List[Dict[str, Any]]:
    dataset = load_dataset("openai_humaneval", split=split)
    data = []
    for item in dataset:
        data.append(
            {
                "id": item["task_id"],
                "instruction": item["prompt"],
                "ground_truth": item["canonical_solution"],
                "entry_point": item["entry_point"],
                "test_code": item["test"],
                "domain": "coding",
            }
        )
    return annotate_items(data, "humaneval")


def load_mbpp(split: str = "test") -> List[Dict[str, Any]]:
    dataset = load_dataset("google-research-datasets/mbpp", "full", split=split)
    data = []
    for item in dataset:
        target_tests = "\n".join(item["test_list"])
        prompt_text = item.get("prompt", item.get("text", ""))
        full_instruction = (
            "You are an expert Python programmer. Write code for this task:\n\n"
            f"{prompt_text}\n\nYour code should pass these tests:\n{target_tests}\n"
        )
        data.append(
            {
                "id": f"mbpp_{item['task_id']}",
                "instruction": full_instruction,
                "ground_truth": item["code"],
                "test_list": item["test_list"],
                "domain": "coding",
            }
        )
    return annotate_items(data, "mbpp")


def load_math(split: str = "test") -> List[Dict[str, Any]]:
    configs = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    data: List[Dict[str, Any]] = []
    for config in configs:
        for s in ["train", "test"]:
            try:
                dataset = load_dataset("EleutherAI/hendrycks_math", config, split=s)
                for i, item in enumerate(dataset):
                    data.append(
                        {
                            "id": f"math_{config}_{s}_{i}",
                            "instruction": item["problem"],
                            "ground_truth": item["solution"],
                            "topic": config,
                            "level": item["level"],
                            "domain": "math",
                        }
                    )
            except Exception as exc:
                print(f"Warning: Failed to load MATH config {config} split {s}: {exc}")
    return data


def load_bigmath(split: str = "test", categories=None) -> List[Dict[str, Any]]:
    dataset = load_dataset("Jongbin-kr/BIG-MATH_filtered", split=split)
    categories = [categories] if isinstance(categories, str) else categories
    data: List[Dict[str, Any]] = []
    for i, item in enumerate(dataset):
        if categories and item["categories"] not in categories:
            continue
        data.append(
            {
                "id": f"bigmath_filtered_{split}_{i}",
                "instruction": item["problem"],
                "ground_truth": item["answer"],
                "domain": "math",
                "categories": item["categories"],
                "source": item["source"],
                "original_domain": item["original_domain"],
                "llama8b_solve_rate": item["llama8b_solve_rate"],
            }
        )
    return data


def load_ds1000(split: str = "test") -> List[Dict[str, Any]]:
    dataset = load_dataset("xlangai/DS-1000", split=split)
    data = []
    for i, item in enumerate(dataset):
        data.append(
            {
                "id": f"ds1000_{i}",
                "instruction": item["prompt"],
                "ground_truth": item["reference_code"],
                "code_context": item["code_context"],
                "metadata": item["metadata"],
                "domain": "ds",
            }
        )
    return data


def load_livecodebench(release_version: str = "release_v5") -> List[Dict[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download

        version_num = int(release_version.split("_v")[-1])
        data = []
        for i in range(1, version_num + 1):
            filename = "test.jsonl" if i == 1 else f"test{i}.jsonl"
            filepath = hf_hub_download(
                repo_id="livecodebench/code_generation_lite",
                filename=filename,
                repo_type="dataset",
            )
            with open(filepath, "r", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    data.append(
                        {
                            "id": item["question_id"],
                            "instruction": item["question_content"],
                            "starter_code": item["starter_code"],
                            "difficulty": item["difficulty"],
                            "platform": item["platform"],
                            "domain": "coding",
                        }
                    )
        print(f"Loaded {len(data)} items for {release_version}")
        return annotate_items(data, "livecodebench")
    except Exception as exc:
        print(f"Error loading LiveCodeBench: {exc}")
        return []


def load_from_jsonl(filepath: str, dataset_key: str) -> List[Dict[str, Any]]:
    data = []
    domain = "math" if dataset_key.lower() == "math" else "coding"
    with open(filepath, "r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            item["domain"] = item.get("domain", domain)
            data.append(item)
    return annotate_items(data, dataset_key)


def item_matches_categories(item: Dict[str, Any], categories: Optional[Union[str, List[str]]]) -> bool:
    if not categories:
        return True
    wanted = {categories} if isinstance(categories, str) else set(categories)
    item_categories = item.get("categories")
    if isinstance(item_categories, str):
        return item_categories in wanted
    if isinstance(item_categories, list):
        return bool(set(map(str, item_categories)) & wanted)
    main = item.get("main_critic_category")
    return bool(main and main in wanted)


def local_dataset_path(name: str, split: str, local_dir: Optional[str]) -> Optional[str]:
    search_dirs = []
    if local_dir:
        search_dirs.append(local_dir)
    if name.lower() == "acc_algorithm":
        search_dirs.append(DEFAULT_ACC_ALGORITHM_DIR)
    for directory in search_dirs:
        filepath = os.path.join(directory, f"{name.lower()}_{split}.jsonl")
        if os.path.exists(filepath):
            return filepath
    return None


def get_dataset(
    name: str,
    split: str = "test",
    local_dir: Optional[str] = None,
    categories: Optional[Union[str, List[str]]] = None,
    data_ratio: float = 1.0,
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    dataset_key = name.lower()
    filepath = local_dataset_path(dataset_key, split, local_dir)
    if filepath:
        print(f"Loading '{name}' from local file: {filepath}")
        data = load_from_jsonl(filepath, dataset_key)
    else:
        if dataset_key == "humaneval":
            data = load_humaneval(split)
        elif dataset_key == "mbpp":
            data = load_mbpp(split)
        elif dataset_key == "math":
            data = load_math(split)
        elif dataset_key == "bigmath":
            data = load_bigmath(split, categories=categories)
        elif dataset_key == "ds1000":
            data = load_ds1000(split)
        elif dataset_key == "livecodebench":
            data = load_livecodebench()
        else:
            raise ValueError(f"Unknown dataset or missing local JSONL: {name}")

    data = [item for item in data if item_matches_categories(item, categories)]
    if seed is not None:
        rng = random.Random(seed)
        data = list(data)
        rng.shuffle(data)
    if data_ratio < 1.0:
        data = data[: max(1, int(len(data) * data_ratio))]
    return data
