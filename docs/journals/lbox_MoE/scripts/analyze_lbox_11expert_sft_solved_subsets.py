#!/usr/bin/env python3
"""Evaluate LBox methods on subsets solved by each of 11 persona-SFT experts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from analyze_lbox_presft_persona_subsets import (
    DEFAULT_METHODS,
    REPO,
    load_jsonl,
    score_lbox,
)


EXPERT_RE = re.compile(r"roster_(c_\d+)_low5")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expert-dir",
        type=Path,
        default=REPO / "results/lbox_persona_eval_ep5_test",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=REPO / "results/lbox_binning_seed20210311/agent_mapping.json",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=REPO / "export/lbox/lbox_test.jsonl",
    )
    parser.add_argument(
        "--sparse-predictions",
        type=Path,
        default=REPO / "sparse_upcycled_test_predictions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "results/lbox_11expert_sft_solved_subset_analysis",
    )
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    references = {row["id"]: row for row in load_jsonl(args.reference)}
    expected_ids = set(references)

    expert_files: dict[str, Path] = {}
    for path in sorted(args.expert_dir.glob("lbox_sft_lbox_*persona_eval_ep5_*.jsonl")):
        if "generalist_high6" in path.name:
            expert_id = "generalist_high6"
        else:
            match = EXPERT_RE.search(path.name)
            if not match:
                continue
            expert_id = match.group(1)
        if expert_id in expert_files:
            raise ValueError(f"Duplicate expert output for {expert_id}")
        expert_files[expert_id] = path

    expert_ids = [key for key in mapping if key in expert_files] + ["generalist_high6"]
    if len(expert_ids) != 11 or set(expert_ids) != set(expert_files):
        raise ValueError(f"Expected 11 expert outputs, found {sorted(expert_files)}")

    expert_correctness: dict[str, dict[str, bool]] = {}
    for expert_id in expert_ids:
        rows = load_jsonl(expert_files[expert_id])
        by_id = {row["id"]: float(row["pass_score"]) > 0 for row in rows}
        if set(by_id) != expected_ids:
            raise ValueError(f"ID coverage mismatch for {expert_id}")
        expert_correctness[expert_id] = by_id

    subsets = {
        expert_id: {item_id for item_id, solved in scores.items() if solved}
        for expert_id, scores in expert_correctness.items()
    }
    subsets["all_failed"] = {
        item_id
        for item_id in expected_ids
        if not any(expert_correctness[expert_id][item_id] for expert_id in expert_ids)
    }

    methods = dict(DEFAULT_METHODS)
    methods = {
        **{
            key: value
            for key, value in methods.items()
            if key not in {"Dense SFT", "vanilla Llama3-8B"}
        },
        "Sparse-upcycled MoE": (str(args.sparse_predictions), "prediction"),
        "Dense SFT": methods["Dense SFT"],
        "vanilla Llama3-8B": methods["vanilla Llama3-8B"],
    }

    method_correctness: dict[str, dict[str, bool]] = {}
    sources: dict[str, str] = {}
    for method, (raw_path, score_field) in methods.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO / path
        rows = load_jsonl(path)
        if score_field in {"prediction", "final_output"}:
            by_id = {
                row["id"]: score_lbox(references[row["id"]], row[score_field])
                for row in rows
            }
        else:
            by_id = {row["id"]: float(row[score_field]) > 0 for row in rows}
        if set(by_id) != expected_ids:
            raise ValueError(f"ID coverage mismatch for {method}")
        method_correctness[method] = by_id
        sources[method] = str(path.relative_to(REPO))

    def expert_name(expert_id: str) -> str:
        if expert_id == "generalist_high6":
            return "High6 Generalist"
        if expert_id == "all_failed":
            return "All 11 experts failed"
        return mapping[expert_id]["name"]

    columns = expert_ids + ["all_failed"]
    results = []
    for method, scores in method_correctness.items():
        overall_correct = sum(scores.values())
        result = {
            "method": method,
            "overall_n": len(expected_ids),
            "overall_correct": overall_correct,
            "overall_accuracy": 100.0 * overall_correct / len(expected_ids),
            "subsets": {},
        }
        for key in columns:
            ids = subsets[key]
            correct = sum(scores[item_id] for item_id in ids)
            result["subsets"][key] = {
                "n": len(ids),
                "correct": correct,
                "accuracy": 100.0 * correct / len(ids),
            }
        results.append(result)

    union_solved = len(expected_ids) - len(subsets["all_failed"])
    payload = {
        "definition": (
            "Each expert column is the overlapping set of test examples solved by that "
            "persona-SFT expert during full test traversal."
        ),
        "total": len(expected_ids),
        "union_solved": union_solved,
        "union_upper_bound": 100.0 * union_solved / len(expected_ids),
        "columns": [
            {"key": key, "name": expert_name(key), "n": len(subsets[key])}
            for key in columns
        ],
        "expert_sources": {
            key: str(path.relative_to(REPO)) for key, path in expert_files.items()
        },
        "method_sources": sources,
        "results": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fields = ["method", "overall_accuracy"] + [f"{key}_accuracy" for key in columns]
    with (args.output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "method": result["method"],
                    "overall_accuracy": f'{result["overall_accuracy"]:.2f}',
                    **{
                        f"{key}_accuracy": f'{result["subsets"][key]["accuracy"]:.2f}'
                        for key in columns
                    },
                }
            )

    headers = ["Method", f"Overall [{len(expected_ids):,}]"] + [
        f"{expert_name(key)} [{len(subsets[key]):,}]" for key in columns
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for result in results:
        values = [result["method"], f'{result["overall_accuracy"]:.2f}'] + [
            f'{result["subsets"][key]["accuracy"]:.2f}' for key in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    (args.output_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"11-expert oracle UB: {union_solved}/{len(expected_ids)} "
          f"({100.0 * union_solved / len(expected_ids):.2f}%)")
    print(f"Wrote {args.output_dir / 'table.md'}")


if __name__ == "__main__":
    main()
