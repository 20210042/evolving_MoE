#!/usr/bin/env python3
"""Build balanced ACC algorithm SFT splits from the labeled benchmark CSV.

The input CSV can assign multiple critic categories to one problem. This builder preserves that
signal by expanding each row into one training example per critic, balancing all five critic buckets,
and ensuring that the same original problem ID does not leak across train/validation/test inside the
same critic."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SOURCE_CSV = Path(
    "/home/minjikim/minji_link/code/benchmark/data/labelling/04_execution_ready_final_labels_local.csv"
)
OUTPUT_DIR = Path("/home/minjikim/minji_link/evolving_MoE/data/acc_algorithm")
DATASET_NAME = "acc_algorithm"
SEED = 42
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1

CRITICS = [
    "Constructive Implementation",
    "Quantitative Reasoning",
    "State-Space Reasoning",
    "Structured Data",
    "Greedy Strategy",
]
CRITIC_SET = set(CRITICS)


def parse_list_cell(value: Any) -> list[str]:
    raw = str(value if value is not None else "").strip()
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw) if raw.startswith("[") else raw
    except (SyntaxError, ValueError):
        parsed = raw
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []


def parse_json_cell(value: Any, fallback: Any) -> Any:
    raw = str(value if value is not None else "").strip()
    if not raw:
        return fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return fallback


def safe_name(value: str) -> str:
    text = value.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def extract_reference_code(answer_value: Any) -> str:
    answers = parse_json_cell(answer_value, [])
    if isinstance(answers, dict):
        answers = [answers]
    if not isinstance(answers, list):
        return str(answer_value or "").strip()

    candidates = []
    for item in answers:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip()
        if not code:
            continue
        language = str(item.get("language") or "").lower()
        is_known_correct = item.get("is_known_correct")
        score = 0
        if is_known_correct is True:
            score += 4
        if "python" in language:
            score += 2
        candidates.append((score, code))
    if not candidates:
        return ""
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return candidates[0][1]


def normalize_test_cases(value: Any) -> list[dict[str, Any]]:
    parsed = parse_json_cell(value, [])
    return parsed if isinstance(parsed, list) else []


def normalize_eval_spec(value: Any) -> dict[str, Any]:
    parsed = parse_json_cell(value, {})
    return parsed if isinstance(parsed, dict) else {}


def critics_for_row(row: dict[str, str]) -> list[str]:
    critics = [critic for critic in parse_list_cell(row.get("critic_categories")) if critic in CRITIC_SET]
    if critics:
        return critics
    main = row.get("main_critic_category", "").strip()
    return [main] if main in CRITIC_SET else []


def build_item(row: dict[str, str], critic: str) -> dict[str, Any] | None:
    problem_id = str(row.get("problem_id") or "").strip()
    instruction = str(row.get("problem") or "").strip()
    ground_truth = extract_reference_code(row.get("answer"))
    test_cases = normalize_test_cases(row.get("test_cases"))
    if not problem_id or not instruction or not ground_truth or not test_cases:
        return None

    return {
        "id": f"{problem_id}__{safe_name(critic)}",
        "problem_id": problem_id,
        "instruction": instruction,
        "ground_truth": ground_truth,
        "reference_solutions": row.get("answer", ""),
        "test_cases": test_cases,
        "eval_spec": normalize_eval_spec(row.get("eval_spec")),
        "domain": "coding",
        "dataset": DATASET_NAME,
        "scoring_kind": "stdin_stdout",
        "normalized_labels": parse_list_cell(row.get("normalized_labels")),
        "critic_categories": critics_for_row(row),
        "main_critic_category": critic,
        "categories": [critic],
        "source": row.get("source", ""),
        "source_platform": row.get("source_platform", ""),
        "original_domain": row.get("original_domain", ""),
        "label_source": row.get("label_source", ""),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced ACC algorithm SFT data.")
    parser.add_argument("--input-csv", type=Path, default=SOURCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--validation-ratio", type=float, default=VALIDATION_RATIO)
    parser.add_argument("--max-per-critic", type=int, default=None)
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    rng = random.Random(args.seed)
    by_critic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped = Counter()
    seen_problem_ids = set()

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            problem_id = row.get("problem_id")
            if problem_id in seen_problem_ids:
                skipped["duplicate_problem_id"] += 1
                continue
            seen_problem_ids.add(problem_id)

            row_critics = critics_for_row(row)
            if not row_critics:
                skipped["missing_critic"] += 1
                continue

            for critic in row_critics:
                item = build_item(row, critic)
                if item is None:
                    skipped["missing_required_fields"] += 1
                    break
                by_critic[critic].append(item)

    available = {critic: len(by_critic[critic]) for critic in CRITICS}
    if any(count == 0 for count in available.values()):
        raise SystemExit(f"ERROR: at least one critic has no examples: {available}")

    per_critic = min(available.values())
    if args.max_per_critic is not None:
        per_critic = min(per_critic, args.max_per_critic)

    n_train = int(per_critic * args.train_ratio)
    n_validation = int(per_critic * args.validation_ratio)
    n_test = per_critic - n_train - n_validation
    split_counts = {"train": n_train, "validation": n_validation, "test": n_test}
    splits = {"train": [], "validation": [], "test": []}
    split_problem_ids: dict[str, dict[str, list[str]]] = {critic: {} for critic in CRITICS}

    for critic in CRITICS:
        items = list(by_critic[critic])
        rng.shuffle(items)
        items = items[:per_critic]
        train_items = items[:n_train]
        validation_items = items[n_train : n_train + n_validation]
        test_items = items[n_train + n_validation :]
        for split_name, split_items in [
            ("train", train_items),
            ("validation", validation_items),
            ("test", test_items),
        ]:
            seen = set()
            for item in split_items:
                original_id = item["problem_id"]
                if original_id in seen:
                    raise SystemExit(f"ERROR: duplicate problem inside {critic}/{split_name}: {original_id}")
                seen.add(original_id)
            splits[split_name].extend(split_items)
            split_problem_ids[critic][split_name] = sorted(seen)

    for split_items in splits.values():
        rng.shuffle(split_items)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_items in splits.items():
        write_jsonl(args.output_dir / f"{DATASET_NAME}_{split_name}.jsonl", split_items)

    report = {
        "source_csv": str(args.input_csv),
        "output_dir": str(args.output_dir),
        "dataset_name": DATASET_NAME,
        "seed": args.seed,
        "critics": CRITICS,
        "available_expanded_examples_by_critic": available,
        "balanced_examples_per_critic": per_critic,
        "split_counts_per_critic": split_counts,
        "total_rows_by_split": {split: len(rows) for split, rows in splits.items()},
        "category_distribution_by_split": {
            split: dict(Counter(row["main_critic_category"] for row in rows))
            for split, rows in splits.items()
        },
        "skipped": dict(skipped),
        "note": "Rows with multiple critic_categories are expanded into one example per critic. Within each critic, train/validation/test problem IDs are disjoint.",
    }
    (args.output_dir / "split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
