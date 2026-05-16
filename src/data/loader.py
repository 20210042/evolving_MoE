import random


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def load_humaneval(split="test"):
    """
    Load OpenAI HumanEval dataset.
    Returns list of dicts: {"id", "instruction", "ground_truth", "entry_point", "test_code", "domain": "coding"}
    """
    dataset = _load_dataset("openai_humaneval", split=split)
    data = []
    
    # HumanEval doesn't have a 'train' split usually, only 'test'.
    # prompt contains function signature and docstring.
    
    for item in dataset:
        data.append({
            "id": item["task_id"],
            "instruction": item["prompt"], # Contains signature + docstring
            "ground_truth": item["canonical_solution"],
            "entry_point": item["entry_point"],
            "test_code": item["test"],
            "domain": "coding"
            # No specific sub-topic for HumanEval
        })
    return data


def load_mbpp(split="test"):
    """
    Load Google MBPP dataset.
    Returns list of dicts: {"id", "instruction" (with 3-shot), "ground_truth", "test_list", "domain": "coding"}
    """
    dataset = _load_dataset("google-research-datasets/mbpp", "full", split=split)
    
    # Standard 3-shot prompt for MBPP
    few_shot_prompt = (
        'You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\n'
        'assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\n'
        'assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)\n'
        'assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)\n'
        '```python\ndef similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res)\n```\n\n'
        'You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\n'
        'assert is_not_prime(2) == False\n'
        'assert is_not_prime(10) == True\n'
        'assert is_not_prime(35) == True\n'
        '```python\nimport math\ndef is_not_prime(n):\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            result = True\n    return result\n```\n\n'
        'You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\n'
        'assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \n'
        'assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \n'
        'assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]\n'
        '```python\nimport heapq as hq\ndef heap_queue_largest(nums,n):\n  largest_nums = hq.nlargest(n, nums)\n  return largest_nums\n```\n\n'
    )
    
    data = []
    for item in dataset:
        target_tests = "\n".join(item["test_list"])
        prompt_text = item.get('prompt', item.get('text', ''))
        full_instruction = f"{few_shot_prompt}You are an expert Python programmer, and here is your task: {prompt_text} Your code should pass these tests:\n\n{target_tests}\n"
        
        data.append({
            "id": f"mbpp_{item['task_id']}",
            "instruction": full_instruction,
            "ground_truth": item["code"],
            "test_list": item["test_list"],
            "domain": "coding"
        })
    return data


def code_signature(sig):
    return f"```python\n{sig}\n```"

def load_math(split="test"):
    """
    Load Hendrycks MATH dataset (ALL configs, ALL splits merged).
    """
    configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    splits_to_load = ['train', 'test'] # MATH typically has these two
    
    data = []
    for config in configs:
        for s in splits_to_load:
            try:
                dataset = _load_dataset("EleutherAI/hendrycks_math", config, split=s)
                for i, item in enumerate(dataset):
                    data.append({
                        "id": f"math_{config}_{s}_{i}",
                        "instruction": item["problem"],
                        "ground_truth": item["solution"],
                        "topic": config,
                        "level": item["level"],
                        "domain": "math"
                    })
            except Exception as e:
                # Some splits might not exist for some configs, though unlikely for MATH
                print(f"Warning: Failed to load MATH config {config} split {s}: {e}")
            
    return data

def load_ds1000(split="test"):
    """
    Load DS-1000 dataset for Data Science code generation.
    """
    dataset = _load_dataset("xlangai/DS-1000", split=split)
    data = []
    
    for i, item in enumerate(dataset):
        # DS-1000 prompt includes the question and setup code.
        # We need to preserve code_context for evaluation.
        data.append({
            "id": f"ds1000_{i}",
            "instruction": item["prompt"],
            "ground_truth": item["reference_code"],
            "code_context": item["code_context"],
            "metadata": item["metadata"],
            "domain": "ds"
        })
    return data


def load_livecodebench(release_version="release_v5"):
    """
    Load LiveCodeBench code generation dataset directly from HF Hub (bypassing datasets lib).
    Returns list of dicts: {"id", "instruction", "starter_code", "difficulty", "platform", "domain": "coding"}
    """
    try:
        from huggingface_hub import hf_hub_download
        import zlib
        import pickle
        import base64
        
        # Mapping release version to files
        # release_v1: test.jsonl
        # release_v2: test.jsonl, test2.jsonl
        # ...
        # release_v5: test.jsonl ... test5.jsonl
        
        version_num = int(release_version.split("_v")[-1])
        data = []
        
        for i in range(1, version_num + 1):
            filename = 'test.jsonl' if i == 1 else f'test{i}.jsonl'
            filepath = hf_hub_download(repo_id='livecodebench/code_generation_lite', 
                                       filename=filename, 
                                       repo_type='dataset')
            
            with open(filepath, "r") as f:
                for line in f:
                    item = json.loads(line)
                    data.append({
                        "id": item["question_id"],
                        "instruction": item["question_content"],
                        "starter_code": item["starter_code"],
                        "difficulty": item["difficulty"],
                        "platform": item["platform"],
                        "domain": "coding"
                    })
                    
        print(f"Loaded {len(data)} items for {release_version}")
        return data
    except Exception as e:
        print(f"Error loading LiveCodeBench: {e}")
        import traceback
        traceback.print_exc()
        return []


import json
import os

def load_from_jsonl(filepath, domain):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            item = json.loads(line)
            # Ensure domain consistency
            item["domain"] = domain
            data.append(item)
    return data

def get_dataset(name: str, split: str = "test", local_dir: str = None):
    # If local_dir is provided, try to load from JSONL
    if local_dir:
        filename = f"{name.lower()}_{split}.jsonl"
        # Try root path first
        filepath = os.path.join(local_dir, filename)
        
        # If not found, try subdirectories (Coding/Math)
        if not os.path.exists(filepath):
            if name.lower() == "math":
                filepath = os.path.join(local_dir, "Math", filename)
            else:
                filepath = os.path.join(local_dir, "Coding", filename)
                
        if os.path.exists(filepath):
            print(f"Loading '{name}' from local file: {filepath}")
            if name.lower() == "math":
                return load_from_jsonl(filepath, "math")
            else:
                return load_from_jsonl(filepath, "coding")
        else:
            print(f"Local file {filepath} not found. Falling back to HuggingFace.")

    # Fallback to HF
    if name.lower() == "humaneval":
        return load_humaneval(split)
    elif name.lower() == "mbpp":
        return load_mbpp(split)
    elif name.lower() == "math":
        return load_math(split)
    elif name.lower() == "ds1000":
        return load_ds1000(split)
    elif name.lower() == "livecodebench":
        return load_livecodebench()
    else:
        raise ValueError(f"Unknown dataset: {name}")

# Quick test block
if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "humaneval"
    data = get_dataset(name)
    print(f"Loaded {len(data)} items from {name}")
    print("Sample item:", data[0])
