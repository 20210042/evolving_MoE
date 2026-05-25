"""Dataset loading with unified ``dataset`` / ``scoring_kind`` metadata."""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Union

from datasets import load_dataset


def scoring_kind_for_dataset(name: str) -> str:
    n = name.lower()
    if n == "livecodebench":
        return "lcb"
    if n == "humaneval":
        return "humaneval_check"
    return "asserts"


def annotate_items(items: List[Dict[str, Any]], dataset_key: str) -> List[Dict[str, Any]]:
    sk = scoring_kind_for_dataset(dataset_key)
    for it in items:
        it["dataset"] = dataset_key.lower()
        it["scoring_kind"] = sk
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

    few_shot_prompt = (
        "You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\n"
        "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n"
        "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\n"
        "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)\n"
        "```python\ndef similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res)\n```\n\n"
        "You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\n"
        "assert is_not_prime(2) == False\n"
        "assert is_not_prime(10) == True\n"
        "assert is_not_prime(35) == True\n"
        "```python\nimport math\ndef is_not_prime(n):\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            result = True\n    return result\n```\n\n"
        "You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\n"
        "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \n"
        "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \n"
        "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]\n"
        "```python\nimport heapq as hq\ndef heap_queue_largest(nums,n):\n  largest_nums = hq.nlargest(n, nums)\n  return largest_nums\n```\n\n"
    )

    data = []
    for item in dataset:
        target_tests = "\n".join(item["test_list"])
        prompt_text = item.get("prompt", item.get("text", ""))
        full_instruction = f"{few_shot_prompt}You are an expert Python programmer, and here is your task: {prompt_text} Your code should pass these tests:\n\n{target_tests}\n"

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
    splits_to_load = ["train", "test"]
    data: List[Dict[str, Any]] = []
    for config in configs:
        for s in splits_to_load:
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
            except Exception as e:
                print(f"Warning: Failed to load MATH config {config} split {s}: {e}")
    return data


def load_bigmath(split: str = "test", categories=None) -> List[Dict[str, Any]]:

    dataset = load_dataset("Jongbin-kr/BIG-MATH_filtered", split=split)

    categories = [categories] if isinstance(categories, str) else categories

    data: List[Dict[str, Any]] = []
    for i, item in enumerate(dataset):

        if categories and item["categories"] not in categories: continue
        
        data.append(
            {
               "id": f"bigmath_filtered_{split}_{i}",
               "instruction": item["problem"],
               "ground_truth": item["answer"],
               "domain": "math",
               "categories": item["categories"],
               
               ## 아래는 원본 BIG MATH 데이터셋에서 넘어온 메타데이터들인데 필요할지 모르겟삼
               "source": item["source"],
               "original_domain": item["original_domain"],
               "llama8b_solve_rate": item["llama8b_solve_rate"]
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
            with open(filepath, "r") as f:
                for line in f:
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
    except Exception as e:
        print(f"Error loading LiveCodeBench: {e}")
        import traceback

        traceback.print_exc()
        return []


def load_from_jsonl(filepath: str, dataset_key: str) -> List[Dict[str, Any]]:
    data = []
    domain = "math" if dataset_key.lower() == "math" else "coding"
    with open(filepath, "r") as f:
        for line in f:
            item = json.loads(line)
            item["domain"] = item.get("domain", domain)
            data.append(item)
    return annotate_items(data, dataset_key)



def get_dataset(
    name: str,
    split: str = "test",
    local_dir: Optional[str] = None,
    categories: Optional[Union[str, List[str]]] = None,
    data_ratio: float = 1.0,
    seed: Optional[int] = 42,
) -> List[Dict[str, Any]]:
    
    ## load from local
    if local_dir:
        filename = f"{name.lower()}_{split}.jsonl"
        filepath = os.path.join(local_dir, filename)
        if not os.path.exists(filepath):
            if name.lower() == "math":
                filepath = os.path.join(local_dir, "Math", filename)
            else:
                filepath = os.path.join(local_dir, "Coding", filename)
        if os.path.exists(filepath):
            print(f"Loading '{name}' from local file: {filepath}")
            data = load_from_jsonl(filepath, name.lower())
        else:
            print(f"Local file {filepath} not found. Falling back to HuggingFace.")
            data = None
    else:
        data = None
        
    ## huggingface fallback
    if data is None:
        n = name.lower()
        if n == "humaneval":
            data = load_humaneval(split)
        elif n == "mbpp":
            data = load_mbpp(split)
        elif n == "math":
            data = load_math(split)
        elif n == "bigmath":
            data = load_bigmath(split, categories=categories)
        elif n == "ds1000":
            data = load_ds1000(split)
        elif n == "livecodebench":
            data = load_livecodebench()
        else:
            raise ValueError(f"Unknown dataset: {name}")
    
    ## shuffle and sample
    if seed is not None:
        rng = random.Random(seed)
        data = list(data)
        rng.shuffle(data)
    if data_ratio < 1.0:
        data = data[:max(1, int(len(data) * data_ratio))]
        
    return data
