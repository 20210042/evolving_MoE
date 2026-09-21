#!/usr/bin/env python
"""Export SNI examples grouped by their dominant 4x1 routing expert."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_traces(directory: Path, variant: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(directory.glob(f"{variant}_rank*.jsonl")):
        for row in read_jsonl(path):
            rows[str(row["id"])] = row
    if not rows:
        raise RuntimeError(f"No {variant}_rank*.jsonl files found under {directory}")
    return rows


def shares(counts: list[list[int]]) -> tuple[list[float], list[list[float]]]:
    values = np.asarray(counts, dtype=np.float64)
    overall = values.sum(axis=0)
    overall /= max(float(overall.sum()), 1.0)
    layer_denominator = values.sum(axis=1, keepdims=True)
    layer = np.divide(
        values,
        layer_denominator,
        out=np.zeros_like(values),
        where=layer_denominator > 0,
    )
    return overall.tolist(), layer.tolist()


def label_summary(rows: list[dict], field: str, limit: int) -> list[dict]:
    counts = Counter(str(row[field]) for row in rows if row.get(field) not in (None, ""))
    total = sum(counts.values())
    return [
        {"label": label, "count": count, "share": count / total if total else 0.0}
        for label, count in counts.most_common(limit)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--variant", default="sft")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--top-labels", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traces = load_traces(args.trace_dir, args.variant)
    source = {str(row["id"]): row for row in read_jsonl(args.source)}
    predictions = {str(row["id"]): row for row in read_jsonl(args.predictions)}
    missing = sorted(set(traces) - set(source))
    if missing:
        raise RuntimeError(f"Trace IDs missing from source: {len(missing)} (first={missing[0]})")

    groups: dict[str, list[dict]] = {f"E{index}": [] for index in range(4)}
    assignments: list[dict] = []
    for row_id in sorted(traces):
        trace = traces[row_id]
        item = source[row_id]
        prediction = predictions.get(row_id, {})
        prompt_shares, prompt_layer_shares = shares(trace["prompt_route_counts"])
        completion_shares, completion_layer_shares = shares(trace["completion_route_counts"])
        dominant = int(np.argmax(completion_shares))
        record = {
            "id": row_id,
            "dominant_completion_expert": f"E{dominant}",
            "dominant_completion_share": completion_shares[dominant],
            "completion_expert_shares": {
                f"E{index}": value for index, value in enumerate(completion_shares)
            },
            "prompt_expert_shares": {
                f"E{index}": value for index, value in enumerate(prompt_shares)
            },
            "prompt_layer_expert_shares": prompt_layer_shares,
            "completion_layer_expert_shares": completion_layer_shares,
            "prompt_tokens": trace["prompt_tokens"],
            "completion_tokens": trace["completion_tokens"],
            "task_name": item.get("task_name"),
            "category": item.get("category"),
            "domain": item.get("sni_domain") or item.get("domain"),
            "definition": item.get("definition"),
            "instruction": item.get("instruction"),
            "ground_truth": item.get("ground_truth"),
            "prediction": prediction.get("prediction"),
            "exact_match": prediction.get("em_score"),
            "rouge_l": prediction.get("rouge_l_score"),
        }
        assignments.append(record)
        groups[f"E{dominant}"].append(record)

    for records in groups.values():
        records.sort(key=lambda row: (-row["dominant_completion_share"], row["id"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = args.output_dir / "assignments.jsonl"
    with assignment_path.open("w", encoding="utf-8") as handle:
        for record in assignments:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    grouped = {
        "model": args.model_label,
        "grouping": "dominant expert over all completion-token routes and all 32 MoE layers",
        "examples": len(assignments),
        "experts": groups,
    }
    (args.output_dir / "grouped_by_completion_expert.json").write_text(
        json.dumps(grouped, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "model": args.model_label,
        "grouping": grouped["grouping"],
        "examples": len(assignments),
        "experts": {},
    }
    for expert, records in groups.items():
        summary["experts"][expert] = {
            "count": len(records),
            "example_share": len(records) / len(assignments) if assignments else 0.0,
            "mean_dominant_completion_share": float(
                np.mean([row["dominant_completion_share"] for row in records])
            )
            if records
            else 0.0,
            "top_tasks": label_summary(records, "task_name", args.top_labels),
            "top_categories": label_summary(records, "category", args.top_labels),
            "top_domains": label_summary(records, "domain", args.top_labels),
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
