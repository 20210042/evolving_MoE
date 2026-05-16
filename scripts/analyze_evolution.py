#!/usr/bin/env python3
"""Summarize evolution_log.jsonl (turnover, action counts, utilities)."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_jsonl", type=str, help="Path to evolution_log.jsonl")
    args = parser.parse_args()
    path = Path(args.log_jsonl)
    if not path.is_file():
        raise SystemExit(f"Missing {path}")

    actions = collections.Counter()
    utilities = []
    roster_lens = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            actions[row.get("decision", "?")] += 1
            u = row.get("utility") or {}
            utilities.append(u.get("A2", 0.0))
            ra = row.get("roster_after") or []
            roster_lens.append(len(ra))

    print("=== Action counts ===")
    for k, v in actions.most_common():
        print(f"  {k}: {v}")
    if utilities:
        print("=== mean U(A2) per step ===", sum(utilities) / len(utilities))
    if roster_lens:
        print("=== roster size last step ===", roster_lens[-1])


if __name__ == "__main__":
    main()
