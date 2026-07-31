#!/usr/bin/env python3
"""Create one comparison table from the four LBox router baseline runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for bank in ("low7_high8", "low5_high6"):
        for feature in ("hs_mean", "encoder"):
            path = args.results_root / "routers" / f"{bank}_{feature}" / "metrics.json"
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    report = [
        "# LBox pre-generation MLP router baselines",
        "",
        "Roster multi-hot labels; n_solved-stratified 80/20 split of 46,019 train rows; "
        "three-seed logit ensemble.",
        "",
        "| expert bank | feature | best single | learned top-1 | learned top-2 union | oracle any | civil | criminal | statute |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        tasks = row["top1_by_task"]
        report.append(
            f"| {row['bank']} | {row['feature']} | {row['best_single_accuracy']:.2f} | "
            f"{row['top1_accuracy']:.2f} | {row['top2_union_accuracy']:.2f} | "
            f"{row['oracle_any_expert_accuracy']:.2f} | {tasks['casename_civil']:.2f} | "
            f"{tasks['casename_criminal']:.2f} | {tasks['statute']:.2f} |"
        )
    args.results_root.mkdir(parents=True, exist_ok=True)
    (args.results_root / "summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {args.results_root / 'summary.md'}")


if __name__ == "__main__":
    main()
