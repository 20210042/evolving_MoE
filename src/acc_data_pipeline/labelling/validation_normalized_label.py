"""Validation script for normalized-label predictions.

Samples or reads labeling rows, compares normalized labels against original_domain evidence, and writes
CSV/JSON reports that help inspect label coverage and LLM fallback quality."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from normalized_label import (
    CRITIC_TAXONOMY_MAP,
    LLM_BATCH_SIZE,
    get_labels_from_llm,
    regex_fallback,
)


DEFAULT_INPUT_CSV = "/home/minjikim/minji_link/code/benchmark/data/processed/04_execution_ready.csv"
DEFAULT_OUTPUT_DIR = "/home/minjikim/minji_link/code/benchmark/data/labelling"
DEFAULT_LIMIT = 1000
DEFAULT_WORKERS = 1


def parse_original_domain(value: Any) -> list[str]:
    if pd.isna(value):
        return []

    raw = str(value).strip()
    if raw in {"", "[]", '[""]', "['']", '""', "nan", "None", "null"}:
        return []

    try:
        parsed = ast.literal_eval(raw) if raw.startswith("[") else raw
    except (SyntaxError, ValueError):
        parsed = raw

    labels = parsed if isinstance(parsed, list) else [parsed]
    return [str(label).strip() for label in labels if str(label).strip()]


def normalize_original_domain(value: Any) -> list[str]:
    labels = parse_original_domain(value)
    return sorted(set(regex_fallback(label) for label in labels))


def labels_to_critic_categories(labels: list[str]) -> list[str]:
    return sorted(set(CRITIC_TAXONOMY_MAP.get(label, "Miscellaneous") for label in labels))


def choose_main_critic(categories: list[str]) -> str:
    return categories[0] if categories else "Miscellaneous"


def select_validation_rows(input_csv: str | Path, limit: int) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    mask = df["original_domain"].apply(lambda value: len(parse_original_domain(value)) > 0)
    return df[mask].head(limit).copy()


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)


def evaluate_validation_set(
    input_csv: str | Path = DEFAULT_INPUT_CSV,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int = DEFAULT_LIMIT,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("ERROR: OPENAI_API_KEY is not set.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_csv = output_dir / f"validation_normalized_label_top{limit}_input.csv"
    prediction_csv = output_dir / f"validation_normalized_label_top{limit}_predictions.csv"
    report_json = output_dir / f"validation_normalized_label_top{limit}_report.json"

    validation_df = select_validation_rows(input_csv, limit)
    validation_df.to_csv(validation_csv, index=False)

    output_columns = list(validation_df.columns) + [
        "true_normalized_labels",
        "true_critic_categories",
        "true_main_critic_category",
        "predicted_normalized_labels",
        "predicted_critic_categories",
        "predicted_main_critic_category",
        "critic_set_exact_match",
        "critic_set_has_overlap",
        "predicted_main_in_true_set",
    ]

    totals = Counter()
    true_main_counter = Counter()
    predicted_main_counter = Counter()

    with prediction_csv.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_columns)
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            total_rows = len(validation_df)
            batch_count = (total_rows + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
            batches = range(0, total_rows, LLM_BATCH_SIZE)

            for start in tqdm(batches, total=batch_count):
                batch = validation_df.iloc[start : start + LLM_BATCH_SIZE]
                futures = {
                    idx: executor.submit(get_labels_from_llm, row["problem"])
                    for idx, row in batch.iterrows()
                }

                for idx, row in batch.iterrows():
                    true_labels = normalize_original_domain(row["original_domain"])
                    true_categories = labels_to_critic_categories(true_labels)
                    true_main = choose_main_critic(true_categories)

                    predicted_labels = sorted(set(futures[idx].result()))
                    predicted_categories = labels_to_critic_categories(predicted_labels)
                    predicted_main = choose_main_critic(predicted_categories)

                    true_set = set(true_categories)
                    predicted_set = set(predicted_categories)
                    exact_match = true_set == predicted_set
                    has_overlap = bool(true_set & predicted_set)
                    main_in_true = predicted_main in true_set

                    totals["records"] += 1
                    totals["critic_set_exact_match"] += int(exact_match)
                    totals["critic_set_has_overlap"] += int(has_overlap)
                    totals["predicted_main_in_true_set"] += int(main_in_true)
                    true_main_counter[true_main] += 1
                    predicted_main_counter[predicted_main] += 1

                    row_dict = row.to_dict()
                    row_dict.update(
                        {
                            "true_normalized_labels": str(true_labels),
                            "true_critic_categories": str(true_categories),
                            "true_main_critic_category": true_main,
                            "predicted_normalized_labels": str(predicted_labels),
                            "predicted_critic_categories": str(predicted_categories),
                            "predicted_main_critic_category": predicted_main,
                            "critic_set_exact_match": exact_match,
                            "critic_set_has_overlap": has_overlap,
                            "predicted_main_in_true_set": main_in_true,
                        }
                    )
                    writer.writerow(row_dict)

                out_f.flush()

    records = totals["records"]
    report = {
        "input_csv": str(input_csv),
        "validation_csv": str(validation_csv),
        "prediction_csv": str(prediction_csv),
        "records": records,
        "workers": workers,
        "metrics": {
            "critic_set_exact_match_accuracy": totals["critic_set_exact_match"] / records if records else 0.0,
            "critic_set_overlap_accuracy": totals["critic_set_has_overlap"] / records if records else 0.0,
            "predicted_main_in_true_set_accuracy": totals["predicted_main_in_true_set"] / records if records else 0.0,
        },
        "true_main_critic_distribution": dict(true_main_counter),
        "predicted_main_critic_distribution": dict(predicted_main_counter),
    }
    write_json(report_json, report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate LLM critic assignment on rows with existing original_domain labels."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    evaluate_validation_set(
        input_csv=args.input,
        output_dir=args.output_dir,
        limit=args.limit,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
