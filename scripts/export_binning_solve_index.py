#!/usr/bin/env python3
"""Invert score_binning labels into an agent-centric solve index.

Input:
  *.binned.jsonl from scripts/score_binning.py

Output:
  *.agent_solves.json with per-agent solved/failed problem ids and summary stats.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_refs(dataset: str | None, split: str | None, data_dir: str | None) -> dict[str, dict[str, Any]]:
    if not (dataset and split and data_dir):
        return {}
    from data.loader import get_dataset

    return {str(item["id"]): item for item in get_dataset(dataset, split=split, local_dir=data_dir)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-agent solve lists from a binned jsonl.")
    parser.add_argument("--input", required=True, help="Path to *.binned.jsonl")
    parser.add_argument("--output", default=None, help="Default: <input>.agent_solves.json")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument(
        "--include_instruction",
        action="store_true",
        help="Include instruction text in per-problem entries. Off by default to keep files small.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    refs = _load_refs(args.dataset, args.split, args.data_dir)

    experts: list[str] = []
    seen = set()
    for row in rows:
        for expert_id in row.get("per_expert", {}):
            if expert_id not in seen:
                seen.add(expert_id)
                experts.append(expert_id)

    per_agent: dict[str, dict[str, Any]] = {
        expert_id: {"solved": [], "failed": [], "n_solved": 0, "n_failed": 0}
        for expert_id in experts
    }
    problems = []
    for row in rows:
        pid = str(row.get("id"))
        ref = refs.get(pid, {})
        problem_entry: dict[str, Any] = {
            "id": pid,
            "dataset": row.get("dataset") or args.dataset,
            "task_type": ref.get("task_type"),
            "solved_by": row.get("solved_by", []),
            "n_solved": row.get("n_solved", 0),
        }
        if args.include_instruction:
            problem_entry["instruction"] = ref.get("instruction")
        problems.append(problem_entry)

        for expert_id in experts:
            ok = int(row.get("per_expert", {}).get(expert_id, 0)) == 1
            target = "solved" if ok else "failed"
            entry: dict[str, Any] = {"id": pid}
            if ref.get("task_type") is not None:
                entry["task_type"] = ref.get("task_type")
            if args.include_instruction:
                entry["instruction"] = ref.get("instruction")
            per_agent[expert_id][target].append(entry)

    total = len(rows)
    for expert_id, data in per_agent.items():
        data["n_solved"] = len(data["solved"])
        data["n_failed"] = len(data["failed"])
        data["pass_at_1"] = (100.0 * data["n_solved"] / total) if total else 0.0

    output = {
        "input": str(input_path),
        "dataset": args.dataset,
        "split": args.split,
        "total": total,
        "experts": experts,
        "per_agent": per_agent,
        "problems": problems,
    }
    out_path = Path(args.output) if args.output else input_path.with_suffix(".agent_solves.json")
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({total} problems, {len(experts)} experts)")


if __name__ == "__main__":
    main()
