#!/usr/bin/env python3
"""Aggregate independently evaluated LBox roster-SFT experts into an oracle UB."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


AGENT_RE = re.compile(r"roster_(c_\d+)_low5")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--expected-experts", type=int, default=10)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*roster_c_*_low5_persona_eval_ep5_*.jsonl"))
    if len(paths) != args.expected_experts:
        raise ValueError(f"Expected {args.expected_experts} expert outputs, found {len(paths)}")

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    expert_scores: dict[str, dict[str, bool]] = {}
    reference: dict[str, dict] | None = None
    source_files: dict[str, str] = {}

    for path in paths:
        match = AGENT_RE.search(path.name)
        if not match:
            raise ValueError(f"Cannot extract expert id from {path}")
        expert_id = match.group(1)
        rows = load_jsonl(path)
        by_id = {row["id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"Duplicate ids in {path}")
        if reference is None:
            reference = by_id
        elif set(by_id) != set(reference):
            raise ValueError(f"ID coverage differs in {path}")
        expert_scores[expert_id] = {
            item_id: float(row.get("pass_score", 0)) > 0 for item_id, row in by_id.items()
        }
        source_files[expert_id] = str(path)

    assert reference is not None
    expert_ids = sorted(expert_scores)
    binned_rows = []
    solve_count_distribution: Counter[int] = Counter()
    task_totals: Counter[str] = Counter()
    task_solved: Counter[str] = Counter()
    exclusive = Counter()

    for item_id, row in reference.items():
        solved_by = [expert_id for expert_id in expert_ids if expert_scores[expert_id][item_id]]
        task_type = "statute" if item_id.startswith("statute_") else "casename"
        task_totals[task_type] += 1
        task_solved[task_type] += bool(solved_by)
        solve_count_distribution[len(solved_by)] += 1
        if len(solved_by) == 1:
            exclusive[solved_by[0]] += 1
        binned_rows.append(
            {
                "id": item_id,
                "dataset": "lbox",
                "split": "test",
                "task_type": task_type,
                "ground_truth": row.get("ground_truth"),
                "solved_by": solved_by,
                "n_solved": len(solved_by),
                "per_expert": {
                    expert_id: int(expert_scores[expert_id][item_id]) for expert_id in expert_ids
                },
            }
        )

    total = len(binned_rows)
    union_solved = total - solve_count_distribution[0]
    per_expert = {}
    for expert_id in expert_ids:
        solved = sum(expert_scores[expert_id].values())
        per_expert[expert_id] = {
            "name": mapping[expert_id]["name"],
            "solved": solved,
            "pass_at_1": 100.0 * solved / total,
            "exclusive_solves": exclusive[expert_id],
            "source": source_files[expert_id],
        }

    summary = {
        "dataset": "lbox",
        "split": "test",
        "setup": "10 roster-derived low5 persona-SFT epoch-5 experts; router-free oracle union",
        "total": total,
        "experts": expert_ids,
        "union_solved": union_solved,
        "union_upper_bound": 100.0 * union_solved / total,
        "unsolved_by_all": solve_count_distribution[0],
        "solved_by_all": solve_count_distribution[len(expert_ids)],
        "solve_count_distribution": {
            str(i): solve_count_distribution[i] for i in range(len(expert_ids) + 1)
        },
        "per_task": {
            task_type: {
                "total": task_totals[task_type],
                "union_solved": task_solved[task_type],
                "union_upper_bound": 100.0 * task_solved[task_type] / task_totals[task_type],
            }
            for task_type in sorted(task_totals)
        },
        "per_expert": per_expert,
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    binned_path = args.output_prefix.with_suffix(".binned.jsonl")
    summary_path = args.output_prefix.with_suffix(".summary.json")
    with binned_path.open("w", encoding="utf-8") as handle:
        for row in binned_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Router-free union UB: {union_solved}/{total} ({summary['union_upper_bound']:.2f}%)")
    print(f"Wrote {binned_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
